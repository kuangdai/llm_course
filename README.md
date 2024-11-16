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
Run one of the following commands based on your CUDA version:

For CUDA 12.1:
```shell
. install_cu121.sh
```

For CUDA 11.8:
```shell
. install_cu118.sh
```

## Acknowledgments
This repository was developed by Kuangdai Leng ([Email](kuangdai.leng@stfc.ac.uk)) with support from the NPRAISE program, funded by EPSRC.
