import os
import fitz
import faiss
import tempfile
import uvicorn
import ollama
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings

from db import get_session, create_db_and_tables
from models import QARecord
from sqlmodel import Session
from db import engine

import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# App and model setup
app = FastAPI()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Store text in vectorstore
def store_text_in_vectorstore(sentences):
    docs = [Document(page_content=s) for s in sentences]
    ids = [str(uuid4()) for _ in docs]

    dim = 384
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(docs, ids=ids)
    return vector_store

# Search for similar context
def search_vectorstore(vector_store, query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results

# Ask Gemini
def ask_gemini(question, context):
    prompt = f"""Use the context to answer the question briefly. 
Do not repeat the exact text.

Context:
{context}

Question:
{question}

Answer:"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text if response.parts else "Gemini could not generate a response. Try rephrasing your input."

# Main endpoint
@app.post("/ask")
async def ask_question(file: UploadFile, question: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        extracted_text = extract_text_from_pdf(tmp_path)
        sentences = [s.strip() for s in extracted_text.split("\n") if s.strip()]
        vector_store = store_text_in_vectorstore(sentences)
        results = search_vectorstore(vector_store, question)

        best_context = results[0].page_content if results else "No relevant context found."
        answer = ask_gemini(question, best_context)

        # Save to DB
        session_gen = get_session()
        with Session(engine) as session:
            qa_record = QARecord(
                question=question,
                answer=answer,
                timestamp=datetime.utcnow()
            )
            session.add(qa_record)
            session.commit()

        return JSONResponse({
            "answer": answer,
            "context": best_context
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
