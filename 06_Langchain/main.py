import json
import os
from professor import ProfessorAgent
from secretary import SecretaryAgent

# Load configuration from config.json
config_path = "config.json"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, "r") as f:
    config = json.load(f)

# Initialize agents
professor_agent = ProfessorAgent(config)
secretary_agent = SecretaryAgent(config)


def main():
    print("Welcome to the English Poetry Chatbot! Type 'exit' to end the conversation.\n")

    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Ending conversation. Goodbye!")
            break

        # Process retrieval based on intent (if applicable)
        retrieval = secretary_agent.retrieve(user_input)

        # Generate response from the professor agent with the retrieved information
        response = professor_agent.chat(user_input, retrieval)

        # Print the response
        print(f"AI: {response}\n")


if __name__ == "__main__":
    main()
