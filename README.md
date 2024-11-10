# Hands-on with Large Language Models

This repository provides hands-on resources for working with Large Language Models (LLMs), focusing on inference, knowledge base preparation, 
and creating multi-agent systems.

## Contents
- **01_Inference.ipynb**: Guide to inference using Hugging Face, including memory-efficient deployment and accelerated inference techniques.
- **02_KnowledgeBase.ipynb**: Preparing a vector database and knowledge graph for use with LLMs.
- **03_MyLLMServer.py**: Script for serving LLM-based generation and knowledge retrieval over HTTPS.
- **04_LangflowBasic.md**: Introduction to LangFlow for orchestrating RAG-based multi-agent systems.
- **05_LangflowPoetry.md**: Complete example of a chatbot with expertise in English poetry using LangFlow.
- **06_LangchainPoetry**: LangChain implementation of the English poetry chatbot.
- **07_LanggraphPoetry**: LangGraph implementation of the English poetry chatbot.

## Installation

```shell
# Step 1: Create a Conda environment for the LLM course with Python 3.9
conda create -n llm_course python=3.9

# Step 2: Activate the Conda environment
conda activate llm_course

# Step 3: Install essential libraries for data handling, visualization, and web development
pip install flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm

# Step 4: Download the spaCy English language model for NLP tasks
python -m spacy download en_core_web_sm

# Step 5: Install PyTorch with audio and vision support for deep learning tasks
pip install torch==2.4.0 torchaudio torchvision

# Step 6: Install Hugging Face libraries for working with language models and distributed training
pip install huggingface_hub transformers accelerate

# Step 7: Install bitsandbytes library for quantization (use -U for the latest version)
pip install bitsandbytes -U

# Step 8: Install libraries for structured knowledge bases
pip install faiss-cpu networkx

# Step 9: Install LangChain and LangGraph libraries for advanced LLM workflows
pip install langchain langgraph

# Step 10: Install vLLM, optimized for high-performance language model inference
pip install vllm

# Step 11: Install Localtunnel for public URL access (requires npm)
# To install Node.js and npm if not already installed, use:
# sudo apt update
# sudo apt install -y nodejs npm

# Install Localtunnel globally using npm
sudo npm install -g localtunnel
```

## Acknowledgments
This repository was developed by Kuangdai Leng ([Email](kuangdai.leng@stfc.ac.uk)) with support from the NPRAISE program, funded by EPSRC.
