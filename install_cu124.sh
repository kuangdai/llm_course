# Step 1: Create a Conda environment for the LLM course with Python 3.10
conda create -n llm_course python=3.10 -y

# Step 2: Activate the Conda environment
conda activate llm_course

# Step 3: Install essential libraries for data handling, visualization, and web development
pip install flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm

# Step 4: Download the spaCy English language model for NLP tasks
python -m spacy download en_core_web_sm

# Step 5: Install PyTorch with audio and vision support for deep learning tasks
pip install torch torchvision torchaudio

# Step 6: Install Hugging Face libraries for working with language models and distributed training
pip install huggingface_hub transformers accelerate

# Step 7: Install bitsandbytes library for quantization (use -U for the latest version)
pip install bitsandbytes -U

# Step 8: Install libraries for structured knowledge bases
pip install faiss-cpu networkx

# Step 9: Install LangChain and LangGraph libraries for advanced LLM workflows
pip install langchain langchain_community langgraph

# Step 10: Install vLLM, optimized for high-performance language model inference
pip install vllm

# Step 11: Install Localtunnel for public URL access (requires npm)
# Install Node.js and npm if not already installed
sudo apt update
sudo apt install -y nodejs npm

# Install Localtunnel globally using npm
sudo npm install -g localtunnel

# Step 12: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Download llama3.2:3b
ollama pull llama3.2:3b