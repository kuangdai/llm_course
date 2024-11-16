import functools
import operator
import re
from typing import Annotated, Sequence

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from llm_interface import LLMInterface


class ChatAgentState(TypedDict):
    """Defines the structure of the agent state during the chat session."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: Sequence[BaseMessage]
    retrieved_poems_with_prompt: str
    history: str
    AI_reply: str
    sender: str


def parse_choice(response: str) -> str:
    """
    Parse the LLM response to extract a valid choice: A, B, or C.

    Args:
        response (str): The response from the LLM.

    Returns:
        str: The choice extracted (A, B, or C). Defaults to "A" if no valid choice is found.
    """
    # Check if the first two characters match the expected format
    if response[:2] in ["A]", "B]", "C]"]:
        return response[0]  # Extracts "A", "B", or "C"

    # Use regex as a fallback to find any standalone A, B, or C in the text
    match = re.search(r"\bA\b|\bB\b|\bC\b", response)
    if match:
        return match.group()  # Extracts "A", "B", or "C"

    # Default to "A" if no valid choice is found
    return "A"


class ChatAgent:
    """Represents a chat agent with capabilities to handle poetry queries and retrieval tasks."""

    def __init__(self, llm, history: str, user_input: str, config: dict) -> None:
        """
        Initialize the ChatAgent.

        Args:
            llm: The large language model interface.
            history (str): The conversation history.
            user_input (str): The current user input.
            config (dict): Configuration settings for the agent.
        """
        self.llm = llm
        self.history = history
        self.user_input = user_input
        self.config = config

        # Initialize the nodes for decision-making and replying
        self.choice_agent = self.create_choice(self.llm)
        self.choice_node = functools.partial(self.node_choice,
                                             agent=self.choice_agent,
                                             name="choice")
        self.reply_agent = self.create_reply(self.llm)
        self.reply_node = functools.partial(self.node_reply,
                                            agent=self.reply_agent,
                                            name="reply")

        # Create and configure the state graph
        chatgraph = StateGraph(ChatAgentState)
        chatgraph.add_node("choice", self.choice_node)
        chatgraph.add_node("reply", self.reply_node)
        chatgraph.set_entry_point("choice")
        chatgraph.add_edge("choice", "reply")
        chatgraph.add_edge("reply", END)
        self.chain = chatgraph.compile()

        # Templates
        # Load prompt template from file
        with open("templates/intent.txt", "r") as file:
            template_content = file.read()
        with open("templates/retrieval.txt", "r") as file:
            template_retrieval = file.read()
        with open("templates/conversation.txt", "r") as file:
            template_conversation = file.read()

        # Define prompt templates
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input"],
            template=template_content
        )
        self.retrieval_prompt = PromptTemplate(
            input_variables=["retrieved_poems"],
            template=template_retrieval
        )
        self.conversation_prompt = PromptTemplate(
            input_variables=["retrieved_poems_with_prompt", "history", "input"],
            template=template_conversation
        )

    def node_choice(self, state: dict, agent, name: str) -> dict:
        """
        Decision-making node to choose the appropriate retrieval method.

        Args:
            state (dict): The current state.
            agent: The choice-making agent.
            name (str): The name of the node.

        Returns:
            dict: Updated state with retrieval results if applicable.
        """
        result = agent.invoke(state)
        choice_result = parse_choice(result.content)
        retrieved_content = ""

        if choice_result == "B":
            # Use FAISS for similarity-based retrieval
            retrieved_content = LLMInterface.retrieve_faiss(
                text=self.user_input,
                server_url=self.config.get("server_url", "http://localhost:7777"),
                retrieve_poem_count=self.config.get("retrieve_poem_count", 1)
            )
        elif choice_result == "C":
            # Use NetworkX graph for keyword-based retrieval
            retrieved_content = LLMInterface.retrieve_nx_graph(
                text=self.user_input,
                server_url=self.config.get("server_url", "http://localhost:7777"),
                retrieve_poem_count=self.config.get("retrieve_poem_count", 1),
                depth=self.config.get("retrieve_depth", 2),
                depth_decay=self.config.get("retrieve_depth_decay", 0.5)
            )

        retrieved_poems_with_prompt = (
            self.retrieval_prompt.format(retrieved_poems=retrieved_content)
            if retrieved_content else ""
        )

        return {
            "messages": [result],
            "history": self.history,
            "retrieved_poems_with_prompt": retrieved_poems_with_prompt,
            "sender": name,
        }

    def create_choice(self, llm):
        """
        Create the choice-making agent.

        Args:
            llm: The large language model interface.

        Returns:
            PromptTemplate: The choice-making prompt template bound to the LLM.
        """
        return self.intent_prompt | llm

    @staticmethod
    def node_reply(state: dict, agent, name: str) -> dict:
        """
        Reply-making node to generate a response based on the state.

        Args:
            state (dict): The current state.
            agent: The reply-making agent.
            name (str): The name of the node.

        Returns:
            dict: Updated state with the AI's reply.
        """
        result = agent.invoke(state)
        return {
            "messages": [result],
            "AI_reply": result,
            "sender": name,
        }

    def create_reply(self, llm):
        """
        Create the reply-making agent.

        Args:
            llm: The large language model interface.

        Returns:
            PromptTemplate: The reply-making prompt template bound to the LLM.
        """
        return self.conversation_prompt | llm

    def run(self) -> str:
        """
        Execute the chat agent by invoking the state graph.

        Returns:
            str: The AI's final reply content.
        """
        final_state = self.chain.invoke(
            {
                "user_input": [HumanMessage(content=self.user_input)],
            },
            {"recursion_limit": 5}  # Limit recursion depth to avoid infinite loops
        )
        return final_state['AI_reply'].content
