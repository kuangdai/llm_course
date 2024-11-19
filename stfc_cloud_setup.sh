#!/bin/bash

cd "$HOME"

echo -e "\nUninstalling conflicting libraries...\n"
pip uninstall -y tensorflow

echo -e "\nInstalling essential Python libraries...\n"
pip install --upgrade flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm

echo -e "\nInstalling PyTorch...\n"
pip install --upgrade torch torchvision torchaudio

echo -e "\nInstalling HuggingFace libraries...\n"
pip install --upgrade huggingface_hub transformers accelerate

echo -e "\nInstalling bitsandbytes...\n"
pip install bitsandbytes -U

echo -e "\nInstalling LLM libraries...\n"
pip install --upgrade faiss-cpu networkx langchain langchain_community langgraph vllm

echo -e "\nDownloading spaCy language model...\n"
python -m spacy download en_core_web_sm

echo -e "\nInstalling Localtunnel...\n"
npm install -g localtunnel

echo -e "\nInstalling Ollama...\n"
OLLAMA_DIR="$HOME/ollama"
OLLAMA_RELEASE_URL="https://github.com/ollama/ollama/releases/download/v0.4.2/ollama-linux-amd64.tgz"
mkdir -p "$OLLAMA_DIR"
wget --quiet --show-progress -O- "$OLLAMA_RELEASE_URL" | tar -xz -C "$OLLAMA_DIR"

echo -e "\nCloning repository...\n"
REPO_URL="https://github.com/kuangdai/llm_course"
REPO_DIR="$HOME/llm_course"
rm -rf "$REPO_DIR"  # Remove if already exists
git clone "$REPO_URL" "$REPO_DIR"

echo -e "\nDownloading Meta-Llama-3-8B-Instruct from HuggingFace...\n"
python "$REPO_DIR/download_llama3_hf.py"

echo -e "\nSetup complete! The LLM environment is ready to use.\n"
