# PDF Q&A App with LLaMA 2, FastAPI, FAISS & Streamlit

Ask questions from your PDF using local LLMs like LLaMA 2 or Mistral with semantic search (FAISS) and a user-friendly Streamlit UI.

---

## 🚀 Features

- Upload any PDF and ask natural language questions
- Uses FAISS for semantic text search
- SentenceTransformer (`all-MiniLM-L6-v2`) for embeddings
- LLaMA 2 or Mistral via [Ollama](https://ollama.com) for responses
- FastAPI as backend
- Streamlit as frontend

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pdf-qa-llama2.git
cd pdf-qa-llama2
```
### 2. Set up Python environment using Poetry

Install Poetry:
pip install poetry

Install dependencies:
poetry install

If not using Poetry, install manually:
pip install fastapi uvicorn streamlit python-multipart sentence-transformers faiss-cpu pymupdf

### 3. Set up Ollama API key

Pull and Run LLaMA 2 or Mistral
Install Ollama:

curl -fsSL https://ollama.com/install.sh | sh

Then pull your model:

# For systems with 8+ GB RAM
ollama pull llama2:7b-chat

# For low RAM systems
ollama pull mistral

## Run the Application

1. Start FastAPI Backend

poetry run python main.py
Runs on: http://127.0.0.1:8000

2. Start Streamlit Frontend

poetry run streamlit run app_ui.py
Opens at: http://localhost:8501

---