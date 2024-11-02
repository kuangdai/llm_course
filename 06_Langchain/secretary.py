import re

from langchain.prompts import PromptTemplate

from llm import LLMInterface


def parse_choice(response):
    """Parse the LLM response to extract A, B, or C."""
    # Check if the first two characters match the expected format
    if response[:2] in ["A]", "B]", "C]"]:
        choice = response[0]  # Extracts "A", "B", or "C"

    # Use regex as a fallback to find any standalone A, B, or C in the text
    elif re.search(r"\bA\b|\bB\b|\bC\b", response):
        match = re.search(r"\bA\b|\bB\b|\bC\b", response)
        choice = match.group()  # Extracts "A", "B", or "C"

    # Default to "A" if no valid choice is found
    else:
        choice = "A"

    return choice


class SecretaryAgent:
    def __init__(self, config):
        # Load prompt template from file
        with open("templates/intent.txt", "r") as file:
            template_content = file.read()

        # Define the intent recognition prompt template
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input"],
            template=template_content
        )

        # Initialize the LLM interface with the provided configuration
        self.llm = LLMInterface(config)

    def retrieve(self, user_input):
        """Analyze user input to determine intent and perform the appropriate retrieval."""
        # Format the prompt with the user input
        formatted_prompt = self.intent_prompt.format(user_input=user_input)

        # Run the LLM to classify intent based on the prompt
        response = self.llm.generate(formatted_prompt)

        # Parse the response to identify the retrieval action
        choice = parse_choice(response)

        # Perform retrieval based on the parsed intent
        if choice == "A":
            # Skip retrieval, return an empty string
            retrieval = ""
        elif choice == "B":
            # Perform similarity-based retrieval
            retrieval = self.llm.retrieve_faiss(user_input)
        elif choice == "C":
            # Perform keyword-based retrieval
            retrieval = self.llm.retrieve_nx_graph(user_input)
        else:
            retrieval = ""  # Redundant fallback, keeping for clarity

        return retrieval
