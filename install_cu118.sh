# Step 1: Create a Conda environment for the LLM course with Python 3.10
conda create -n llm_course python=3.10 -y

# Step 2: Activate the Conda environment
conda activate llm_course

# Step 3: Install essential libraries for data handling, visualization, and web development
pip install flask gdown jupyter matplotlib numpy pandas scipy spacy tqdm

# Step 4: Download the spaCy English language model for NLP tasks
python -m spacy download en_core_web_sm

# Step 5: Install PyTorch with audio and vision support for deep learning tasks
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 6: Install Hugging Face libraries for working with language models and distributed training
pip install huggingface_hub transformers accelerate

# Step 7: Install bitsandbytes library for quantization (use -U for the latest version)
pip install bitsandbytes -U

# Step 8: Install libraries for structured knowledge bases
pip install faiss-cpu networkx

# Step 9: Install LangChain and LangGraph libraries for advanced LLM workflows
pip install langchain langchain_community langgraph

# Step 10: Install vLLM, optimized for high-performance language model inference
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

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
