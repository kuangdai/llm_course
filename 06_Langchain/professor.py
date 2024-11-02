import json

from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate

from llm_interface import CustomLLM


class ProfessorAgent:
    def __init__(self):
        # Load config from a JSON file
        with open("config.json", "r") as config_file:
            self.config = json.load(config_file)

        # Load templates from files
        with open("templates/retrieval.txt", "r") as file:
            template_retrieval = file.read()
        with open("templates/conversation.txt", "r") as file:
            template_conversation = file.read()

        # Define prompt templates
        self.retrieval_prompt = PromptTemplate(
            input_variables=["retrieved_poems"],
            template=template_retrieval
        )
        self.conversation_prompt = PromptTemplate(
            input_variables=["retrieved_poems_with_prompt", "history", "input"],
            template=template_conversation
        )

        # Initialize LLM interface and memory with summary LLM
        summary_llm = CustomLLM(
            server_url=self.config.get("server_url", "http://localhost:7777"),
            temperature=self.config.get("summary_temperature", 0.1),
            max_new_tokens=self.config.get("summary_max_new_tokens", 500)
        )
        self.memory = ConversationSummaryBufferMemory(
            llm=summary_llm,
            max_token_limit=self.config.get("response_memory_token_limit", 1000),
            ai_prefix="AI",
            human_prefix="User"
        )

        # Initialize response LLM for conversation chain
        response_llm = CustomLLM(
            server_url=self.config.get("server_url", "http://localhost:7777"),
            temperature=self.config.get("response_temperature", 0.5),
            max_new_tokens=self.config.get("response_max_new_tokens", 100)
        )
        self.conversation_chain = LLMChain(
            llm=response_llm,
            prompt=self.conversation_prompt,
        )

    def chat(self, user_input, retrieval):
        """Generate a response using conversation chain, with optional retrieval context."""
        # Format the retrieval content if it exists
        retrieved_poems_with_prompt = self.retrieval_prompt.format(retrieved_poems=retrieval) if retrieval else ""

        # Load conversation history from memory
        history = self.memory.load_memory_variables({}).get("history", "")

        # Generate response using the LLMChain, passing all required inputs
        response = self.conversation_chain.run({
            "retrieved_poems_with_prompt": retrieved_poems_with_prompt,
            "history": history,
            "input": user_input
        })

        # Update memory with user input and AI response
        self.memory.save_context(
            inputs={"input": user_input},
            outputs={"response": response}
        )

        return response
