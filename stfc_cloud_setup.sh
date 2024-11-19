#!/bin/bash

# Navigate to the home directory
cd "$HOME"

# Install essential Python libraries
echo "Installing essential Python libraries..."
pip install flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm

# Download spaCy English language model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Install PyTorch with audio and vision support
echo "Installing PyTorch with audio and vision support..."
pip install torch torchvision torchaudio

# Install Hugging Face libraries
echo "Installing Hugging Face libraries..."
pip install huggingface_hub transformers accelerate

# Install bitsandbytes for quantization
echo "Installing bitsandbytes..."
pip install bitsandbytes -U

# Install libraries for structured knowledge bases
echo "Installing libraries for structured knowledge bases..."
pip install faiss-cpu networkx

# Install LangChain and LangGraph
echo "Installing LangChain and LangGraph..."
pip install langchain langchain_community langgraph

# Install vLLM for optimized inference
echo "Installing vLLM..."
pip install vllm

# Install Localtunnel globally using npm
echo "Installing Localtunnel..."
npm install -g localtunnel

# Setup Ollama directory and download
OLLAMA_DIR="$HOME/ollama"
echo "Setting up Ollama in '$OLLAMA_DIR'..."
mkdir -p "$OLLAMA_DIR"
wget --show-progress -qO- https://github.com/ollama/ollama/releases/download/v0.4.2/ollama-linux-amd64.tgz | tar -xz -C "$OLLAMA_DIR"

# Start ollama serve in the background and capture its process ID
ollama serve &
SERVE_PID=$!

# Pull the model
sleep 2
ollama pull llama3.2:3b

# Kill the ollama serve process
kill $SERVE_PID

# Course repository
REPO_URL="https://github.com/kuangdai/llm_course"
REPO_DIR="$HOME/llm_course"

# Check if the repository directory exists and remove it
if [ -d "$REPO_DIR" ]; then
    echo "Repository directory '$REPO_DIR' already exists. Removing it..."
    rm -rf "$REPO_DIR"
fi

# Clone the repository
echo "Cloning LLM course repository from $REPO_URL into '$REPO_DIR'..."
git clone "$REPO_URL" "$REPO_DIR"
echo "Repository setup complete!"

# Download Llama
python llm_course/download_llama3_hf.py

echo "Setup complete! The LLM environment is ready to use."
