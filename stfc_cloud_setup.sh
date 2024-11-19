#!/bin/bash

# Initialize Conda
conda init

# Navigate to the home directory
cd $HOME

# Load the bash configuration file
source .bashrc

# Create a Conda environment for the LLM course with Python 3.10
conda create -n llm_course python=3.10 -y

# Activate the Conda environment
conda activate llm_course

# Install essential libraries for data handling, visualization, and web development
pip install flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm

# Download the spaCy English language model for NLP tasks
python -m spacy download en_core_web_sm

# Install PyTorch with audio and vision support for deep learning tasks
pip install torch torchvision torchaudio

# Install Hugging Face libraries for language models and distributed training
pip install huggingface_hub transformers accelerate

# Install the latest version of bitsandbytes for quantization
pip install bitsandbytes -U

# Install libraries for structured knowledge bases
pip install faiss-cpu networkx

# Install LangChain and LangGraph for advanced LLM workflows
pip install langchain langchain_community langgraph

# Install vLLM, optimized for high-performance language model inference
pip install vllm

# Install Localtunnel globally using npm
npm install -g localtunnel

# Create a directory for Ollama and install it
mkdir -p $HOME/ollama
wget --show-progress -qO- https://github.com/ollama/ollama/releases/download/v0.4.2/ollama-linux-amd64.tgz | tar -xz -C $HOME/ollama

# Clone the LLM course repository
git clone https://github.com/kuangdai/llm_course

echo "Setup complete. The LLM environment is ready!"
