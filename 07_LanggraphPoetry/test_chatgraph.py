import json

from langchain_community.chat_models import ChatOllama

from chatgraph import ChatAgent

ollama_model = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11435")
with open("config.json", "r") as config_file:
    config = json.load(config_file)
user_input = ""
history = ""
while user_input != "exit":
    user_input = input("User:")
    if user_input == "exit":
        break
    chatagent = ChatAgent(ollama_model, history, "User:" + user_input, config)
    history += "User:" + user_input
    ai_responce = chatagent.run()
    history += "AI:" + ai_responce
    print("AI:" + ai_responce)
