# app.py - Complete RAG System for WildEye
import os
import shutil
import uuid
import json
from typing import List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
import logging

# ============================================================
#  CONFIGURATION
# ============================================================

UPLOAD_DIR = "uploads"
CHROMA_DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WildEye RAG System", version="1.0.0")

# Enable CORS for your dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  DATA MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str
    collection_name: str = "forest_department_docs"
    k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: str
    document_count: int

# ============================================================
#  EMBEDDINGS & VECTOR STORE
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store(collection_name: str = "forest_department_docs"):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

def add_documents_to_vectorstore(texts: List[str], metadatas: List[Dict], collection_name: str = "forest_department_docs"):
    vectorstore = get_vector_store(collection_name)
    vectorstore.add_texts(texts, metadatas=metadatas)
    vectorstore.persist()
    return len(texts)

def search_documents(query: str, collection_name: str = "forest_department_docs", k: int = 5):
    vectorstore = get_vector_store(collection_name)
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    documents = []
    for doc, score in results:
        documents.append({
            'content': doc.page_content,
            'metadata': doc.metadata,
            'relevance_score': float(score)
        })
    
    return documents

# ============================================================
#  PDF PROCESSING
# ============================================================

def process_pdf(file_path: str) -> List[Dict]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        processed_chunks.append({
            'text': chunk.page_content,
            'metadata': {
                'source': os.path.basename(file_path),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'page': chunk.metadata.get('page', 0),
                'text_length': len(chunk.page_content)
            }
        })
    
    return processed_chunks

# ============================================================
#  LLM INTEGRATION
# ============================================================

def get_llm():
    # Try Hugging Face (free)
    try:
        from langchain_community.llms import HuggingFaceHub
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            return HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                model_kwargs={"temperature": 0.3, "max_length": 512},
                huggingfacehub_api_token=hf_token
            )
    except:
        pass
    
    # Fallback: return None (will use simple response)
    return None

def generate_response(query: str, context_docs: List[Dict]) -> str:
    context = "\n\n".join([doc['content'] for doc in context_docs])
    
    llm = get_llm()
    if llm:
        try:
            prompt = f"""You are a Forest Department assistant. Use the following documents to answer the question.

Documents:
{context}

Question: {query}

Answer based only on the documents provided. If you cannot find the answer, say "I don't have that information in the forest department documents."
"""
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM error: {e}")
    
    # Fallback: Return top chunks
    response_parts = []
    for doc in context_docs[:3]:
        content = doc['content']
        if len(content) > 200:
            response_parts.append(content[:200] + "...")
        else:
            response_parts.append(content)
    
    if response_parts:
        return f"📚 Based on the forest department documents:\n\n" + "\n\n---\n\n".join(response_parts)
    else:
        return "I couldn't find relevant information in the uploaded documents."

# ============================================================
#  API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "🌿 WildEye RAG System - Forest Department Documents",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload",
            "chat": "POST /chat",
            "documents": "GET /documents",
            "delete": "DELETE /document/{doc_id}",
            "stats": "GET /stats"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chunks = process_pdf(file_path)
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        num_chunks = add_documents_to_vectorstore(texts, metadatas)
        
        doc_info = {
            'id': file_id,
            'filename': file.filename,
            'upload_date': datetime.now().isoformat(),
            'page_count': len(set([chunk['metadata']['page'] for chunk in chunks])),
            'chunk_count': num_chunks
        }
        
        metadata_file = os.path.join(UPLOAD_DIR, f"{file_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(doc_info, f)
        
        return {
            "success": True,
            "message": f"Document uploaded successfully",
            "document": doc_info
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        relevant_docs = search_documents(
            request.question,
            collection_name=request.collection_name,
            k=request.k
        )
        
        if not relevant_docs:
            return ChatResponse(
                answer="I don't have any documents uploaded yet. Please upload forest department PDFs first.",
                sources=[],
                timestamp=datetime.now().isoformat(),
                document_count=0
            )
        
        answer = generate_response(request.question, relevant_docs)
        
        sources = []
        for doc in relevant_docs:
            sources.append({
                'content': doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'],
                'source': doc['metadata'].get('source', 'Unknown'),
                'page': doc['metadata'].get('page', 0),
                'relevance': doc['relevance_score']
            })
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            document_count=len(set([s['source'] for s in sources]))
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    documents = []
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith('_metadata.json'):
            with open(os.path.join(UPLOAD_DIR, filename), 'r') as f:
                doc_info = json.load(f)
                documents.append(doc_info)
    
    return {
        "total": len(documents),
        "documents": documents
    }

@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    try:
        deleted_files = []
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(doc_id):
                os.remove(os.path.join(UPLOAD_DIR, filename))
                deleted_files.append(filename)
        
        return {
            "success": True,
            "message": f"Document {doc_id} deleted",
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        vectorstore = get_vector_store()
        collection = vectorstore._collection
        count = collection.count()
        
        return {
            "total_documents": len([f for f in os.listdir(UPLOAD_DIR) if f.endswith('_metadata.json')]),
            "total_chunks": count,
            "upload_dir_size": sum(os.path.getsize(os.path.join(UPLOAD_DIR, f)) for f in os.listdir(UPLOAD_DIR)) / (1024 * 1024),
            "supported_formats": ["PDF"]
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
#  RUN THE APP
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)