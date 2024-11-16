import json

from chatgraph import ChatAgent
from llm_interface import CustomLLM

# Load configuration from a JSON file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

llm_model = CustomLLM(
    server_url=config.get("server_url", "http://localhost:7777"),
    temperature=config.get("summary_temperature", 0.1),
    max_new_tokens=config.get("summary_max_new_tokens", 500)
)

# Initialize user input and conversation history
user_input = ""
history = ""

# Start the chat loop
print("Type 'exit' to end the conversation.")
while True:
    # Get user input
    user_input = input("User: ")
    if user_input.lower() == "exit":  # Allow case-insensitive 'exit' command
        print("Exiting the chat. Goodbye!")
        break

    # Prefix user input with "User:" for history tracking
    user_message = "User:" + user_input

    # Create a ChatAgent instance for the current user input
    chat_agent = ChatAgent(llm_model, history, user_message, config)

    # Update the conversation history with the user's message
    history += user_message

    # Run the ChatAgent and get the AI's response
    ai_response = chat_agent.run()

    # Update the conversation history with the AI's response
    history += "AI:" + ai_response

    # Print the AI's response
    print("AI:", ai_response)
