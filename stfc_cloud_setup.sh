#!/bin/bash

# Navigate to the home directory
cd "$HOME"

# Uninstall any conflicting libraries
echo ""
echo "Uninstalling conflicting libraries..."
echo ""
pip uninstall -y tensorflow

# Install essential Python libraries in one command with force-reinstall
echo ""
echo "Installing essential Python libraries..."
echo ""
pip install --force-reinstall flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm \
    torch torchvision torchaudio \
    huggingface_hub transformers accelerate \
    faiss-cpu networkx langchain langchain_community langgraph vllm

# Install bitsandbytes with force-reinstall and upgrade flag
echo ""
echo "Installing bitsandbytes..."
echo ""
pip install --force-reinstall bitsandbytes -U

# Download spaCy English language model
echo ""
echo "Downloading spaCy language model..."
echo ""
python -m spacy download en_core_web_sm

# Install Localtunnel globally using npm
echo ""
echo "Installing Localtunnel..."
echo ""
npm install -g localtunnel

# Setup Ollama directory and download Ollama binaries
echo ""
echo "Installing Ollama..."
echo ""
OLLAMA_DIR="$HOME/ollama"
OLLAMA_RELEASE_URL="https://github.com/ollama/ollama/releases/download/v0.4.2/ollama-linux-amd64.tgz"
echo ""
echo "Setting up Ollama in '$OLLAMA_DIR'..."
echo ""
mkdir -p "$OLLAMA_DIR"
wget --quiet --show-progress -O- "$OLLAMA_RELEASE_URL" | tar -xz -C "$OLLAMA_DIR"

# Start Ollama serve and pull the model
echo ""
echo "Starting Ollama serve and pulling the model..."
echo ""
"$OLLAMA_DIR/ollama" serve &
SERVE_PID=$!

sleep 2  # Allow serve to initialize
"$OLLAMA_DIR/ollama" pull llama3.2:3b

# Terminate the Ollama serve process
kill $SERVE_PID

# Clone or refresh the course repository
echo ""
echo "Cloning repository..."
echo ""
REPO_URL="https://github.com/kuangdai/llm_course"
REPO_DIR="$HOME/llm_course"
echo ""
echo "Setting up LLM course repository..."
echo ""
rm -rf "$REPO_DIR"  # Remove if already exists
git clone "$REPO_URL" "$REPO_DIR"

# Download Llama model via course repository script
echo ""
echo "Downloading Llama from HuggingFace..."
echo ""
python "$REPO_DIR/download_llama3_hf.py"

echo ""
echo "Setup complete! The LLM environment is ready to use."
echo ""
