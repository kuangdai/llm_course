from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate

from llm import CustomLLM, LLMInterface


class ProfessorAgent:
    def __init__(self, config):
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
            input_variables=["input_retrieved_poems", "history", "input"],
            template=template_conversation
        )

        # Initialize LLM interface and memory
        llm_interface = LLMInterface(config)
        self.llm = CustomLLM(llm_interface=llm_interface)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=config["response_memory_token_limit"],
            ai_prefix="AI", human_prefix="User"
        )

        # Initialize LLMChain with custom memory handling
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.conversation_prompt,
        )

    def chat(self, user_input, retrieval):
        """Generate a response using conversation chain, with optional retrieval context."""
        # Format the retrieval content if it exists
        input_retrieved_poems = self.retrieval_prompt.format(retrieved_poems=retrieval) if retrieval else ""

        # Load conversation history from memory
        history = self.memory.load_memory_variables({}).get("history", "")

        # Generate response using the LLMChain, passing all required inputs
        response = self.conversation_chain.run({
            "input_retrieved_poems": input_retrieved_poems,
            "history": history,
            "input": user_input
        })

        # Update memory with user input and AI response
        self.memory.save_context(
            inputs={"input": user_input},
            outputs={"response": response}
        )

        return response
