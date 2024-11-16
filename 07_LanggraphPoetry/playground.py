import json

from langchain_community.chat_models import ChatOllama

from chatgraph import ChatAgent

#############################
# Start OLLAMA server first #
#############################
"""
$ OLLAMA_HOST=127.0.0.1:11435 ollama serve
"""

# Load configuration from a JSON file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Initialize the Ollama model
ollama_model = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11435")

# Initialize user input and conversation history
user_input = ""
history = ""

# Create a ChatAgent instance for the current user input
chat_agent = ChatAgent(ollama_model, config)

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

    # Update chat state
    chat_agent.update_chat(history, user_message)

    # Update the conversation history with the user's message
    history += user_message

    # Run the ChatAgent and get the AI's response
    ai_response = chat_agent.run()

    # Update the conversation history with the AI's response
    history += "AI:" + ai_response

    # Print the AI's response
    print("AI:", ai_response)
