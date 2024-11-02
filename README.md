# llm_course

This repo is under development.

### Environment

```shell
# Create a new Conda environment with Python 3.9 for the LLM course
conda create -n llm_course python=3.9

# Activate the Conda environment
conda activate llm_course

# Install general-purpose libraries for data handling, visualization, and web development
pip install flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm

# Download the spaCy English language model
python -m spacy download en_core_web_sm

# Install PyTorch along with its audio and vision libraries for deep learning tasks
pip install torch==2.4.0 torchaudio torchvision

# Install Hugging Face libraries for language models and distributed training
pip install huggingface_hub transformers accelerate

# Install bitsandbytes for quantization, with the -U flag to ensure the latest version
pip install bitsandbytes -U

# Install additional libraries for structured knowledge bases
pip install faiss-cpu networkx

# Install langchain, langgraph
pip install langchain langgraph

# Install vllm, a large library for language model inference
# (ignore any pip errors if "Successfully installed ..." appears at the end)
pip install vllm

# Install Localtunnel (requires npm)
# Update package lists and install Node.js and npm if not already installed
# sudo apt update
# sudo apt install -y nodejs npm

# Install Localtunnel globally using npm
sudo npm install -g localtunnel  # install npm first if needed
```
