import json
from langchain.chains import ConversationChain
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
        self.memory = ConversationSummaryBufferMemory(
            llm=CustomLLM(llm_interface=llm_interface),
            max_token_limit=config["response_memory_token_limit"],
            ai_prefix="AI", human_prefix="User"
        )

        # Initialize conversation chain with custom LLM and memory
        self.conversation_chain = ConversationChain(
            llm=CustomLLM(llm_interface=llm_interface),
            memory=self.memory,
            prompt=self.conversation_prompt,
        )

    def chat(self, user_input, retrieval):
        """Generate a response using conversation chain, with optional retrieval context."""
        # Add retrieval content if it exists
        input_retrieved_poems = self.retrieval_prompt.format(retrieved_poems=retrieval) if retrieval else ""

        # Generate response using the conversation chain
        response = self.conversation_chain.run(
            input_retrieved_poems=input_retrieved_poems,
            user_input=user_input
        )

        return response
