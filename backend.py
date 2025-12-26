from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
import os
import uuid
from pathlib import Path
from typing import Dict, Optional
import traceback
import httpx
import json

load_dotenv()

app = FastAPI(title="RAG + Context7 Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, dict] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class IndexRequest(BaseModel):
    folder_path: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: dict = {}

class SessionResponse(BaseModel):
    session_id: str
    message: str

class RAGManager:
    def __init__(self):
        print("Initializing RAG Manager...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.qa_chain = None
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        print("RAG Manager initialized")
    
    def index_folder(self, folder_path: str):
        print(f"Starting to index: {folder_path}")
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.md', '.txt', 
                     '.java', '.cpp', '.c', '.go', '.rs', '.json', '.yaml', '.yml']
        
        all_files = []
        for ext in extensions:
            found = list(folder.rglob(f"*{ext}"))
            all_files.extend(found)
            print(f"Found {len(found)} {ext} files")
        
        ignored = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', 'faiss_index', 'dist', 'build'}
        files = [f for f in all_files if not any(ig in f.parts for ig in ignored)]
        
        print(f"Total files after filtering: {len(files)}")
        
        if len(files) == 0:
            raise ValueError(f"No code files found in {folder_path}")
        
        documents = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) > 0:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "file": str(file_path.relative_to(folder)),
                            "type": file_path.suffix,
                            "name": file_path.name
                        }
                    )
                    documents.append(doc)
                    print(f"  ✓ Indexed: {file_path.name}")
            except Exception as e:
                print(f"  ✗ Skipped {file_path.name}: {e}")
        
        print(f"Successfully read {len(documents)} files")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        self.vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        print("Indexing complete!")
        return len(files), len(chunks)
    
    def query_rag(self, question: str) -> Optional[dict]:
        if not self.qa_chain:
            return None
        try:
            print(f"RAG Query: {question}")
            result = self.qa_chain.invoke({"query": question})
            
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            print(f"RAG found {len(sources)} relevant sources")
            for src in sources[:3]:
                print(f"  - {src.metadata.get('file', 'unknown')}")
            
            return {
                "answer": answer,
                "sources": [s.metadata for s in sources]
            }
        except Exception as e:
            print(f"RAG query error: {e}")
            return None

class Context7Manager:
    def __init__(self):
        self.api_key = os.getenv("CONTEXT7_API_KEY")
        self.available = bool(self.api_key and self.api_key.startswith("ctx7sk-"))
        self.base_url = "https://context7.com/api/v2"
        
        self.fallback_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        if self.available:
            print(f"✅ Context7 API available (key: {self.api_key[:12]}...)")
        else:
            print("⚠️  Context7 API key not found, using LLM fallback")
    
    async def search_docs(self, query: str) -> tuple[str, bool]:
        """Search Context7 for documentation. Returns (response, context7_used)"""
        
        if not self.available:
            return await self._fallback_query(query)
        
        try:
            print(f"🌐 Searching Context7 for: {query}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for relevant libraries
                search_response = await client.get(
                    f"{self.base_url}/search",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    params={"query": query}
                )
                
                if search_response.status_code != 200:
                    print(f"⚠️  Context7 search failed: {search_response.status_code}")
                    return await self._fallback_query(query)
                
                results = search_response.json().get("results", [])
                
                if not results:
                    print("⚠️  No Context7 results found")
                    return await self._fallback_query(query)
                
                # Get top result
                top_result = results[0]
                lib_id = top_result.get("id")
                lib_title = top_result.get("title", "Unknown")
                
                print(f"✅ Found documentation: {lib_title} ({lib_id})")
                
                # Fetch documentation snippets
                docs_response = await client.get(
                    f"{self.base_url}/docs",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    params={
                        "id": lib_id,
                        "query": query,
                        "limit": 3
                    }
                )
                
                if docs_response.status_code == 200:
                    docs_data = docs_response.json()
                    snippets = docs_data.get("snippets", [])
                    
                    if snippets:
                        # Compile documentation
                        doc_text = f"Documentation from {lib_title}:\n\n"
                        for i, snippet in enumerate(snippets[:3], 1):
                            content = snippet.get("content", "")
                            doc_text += f"Section {i}:\n{content}\n\n"
                        
                        print(f"✅ Retrieved {len(snippets)} documentation snippets")
                        return doc_text, True
                
                # If docs fetch failed, return search result description
                description = top_result.get("description", "")
                return f"From {lib_title} documentation:\n{description}", True
        
        except Exception as e:
            print(f"⚠️  Context7 error: {e}")
            return await self._fallback_query(query)
    
    async def _fallback_query(self, query: str) -> tuple[str, bool]:
        """Fallback to LLM when Context7 is unavailable"""
        print("🤖 Using LLM fallback")
        try:
            prompt = f"""You are a helpful assistant with knowledge of programming documentation.

Question: {query}

Provide a detailed answer based on official documentation and best practices."""
            
            response = self.fallback_llm.invoke(prompt).content
            return response, False
        except Exception as e:
            print(f"❌ Fallback error: {e}")
            return "I encountered an error processing your question.", False

def get_or_create_session(session_id: str = None):
    if session_id and session_id in sessions:
        print(f"Retrieved existing session: {session_id}")
        return sessions[session_id], session_id
    
    new_session_id = session_id or str(uuid.uuid4())
    
    try:
        print(f"Creating new session: {new_session_id}")
        
        rag_manager = RAGManager()
        context7_manager = Context7Manager()
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        sessions[new_session_id] = {
            "rag": rag_manager,
            "context7": context7_manager,
            "llm": llm,
            "messages": [],
            "indexed": False
        }
        
        print(f"Session created successfully: {new_session_id}")
        print(f"  - RAG: Ready")
        print(f"  - Context7: {'Available' if context7_manager.available else 'Fallback'}")
        
        return sessions[new_session_id], new_session_id
    
    except Exception as e:
        print(f"Session creation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "RAG + Context7 Chat API is running",
        "components": {
            "rag": "Ready",
            "context7": "API-based search"
        }
    }

@app.post("/index", response_model=SessionResponse)
async def index_code(request: IndexRequest):
    try:
        print(f"Index request for session: {request.session_id}")
        session, session_id = get_or_create_session(request.session_id)
        
        files_count, chunks_count = session["rag"].index_folder(request.folder_path)
        session["indexed"] = True
        
        return SessionResponse(
            session_id=session_id,
            message=f"Indexed {files_count} files, {chunks_count} chunks"
        )
    except Exception as e:
        print(f"Indexing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print(f"\n{'='*60}")
        print(f"Chat request: {request.message}")
        
        session, session_id = get_or_create_session(request.session_id)
        
        rag_result = None
        context7_response = None
        context7_used = False
        
        # Query RAG if indexed
        if session["indexed"]:
            print("🔍 Querying RAG...")
            rag_result = session["rag"].query_rag(request.message)
            if rag_result:
                print(f"✅ RAG Response: {rag_result['answer'][:200]}...")
        else:
            print("⚠️  RAG not indexed")
        
        # Query Context7
        print("🌐 Querying Context7...")
        context7_response, context7_used = await session["context7"].search_docs(request.message)
        
        # Combine responses
        if rag_result and context7_response:
            print("🔀 Combining RAG + Context7...")
            combined_prompt = f"""You have two sources of information:

SOURCE 1 - LOCAL CODE:
{rag_result['answer']}
Files: {', '.join([s.get('name', '') for s in rag_result.get('sources', [])[:3]])}

SOURCE 2 - OFFICIAL DOCUMENTATION (Context7):
{context7_response}

Question: {request.message}

Provide a comprehensive answer combining both sources."""
            
            final_response = session["llm"].invoke(combined_prompt).content
            sources = {"rag": True, "context7": context7_used}
            
        elif rag_result:
            print("📚 Using RAG only")
            files = ', '.join([s.get('name', '') for s in rag_result.get('sources', [])[:3]])
            final_response = f"{rag_result['answer']}\n\n*Files: {files}*"
            sources = {"rag": True, "context7": False}
            
        elif context7_response:
            print(f"🌐 Using Context7 ({'API' if context7_used else 'fallback'})")
            final_response = context7_response
            sources = {"rag": False, "context7": context7_used}
            
        else:
            print("🤖 Using LLM fallback")
            final_response = session["llm"].invoke(request.message).content
            sources = {"rag": False, "context7": False}
        
        session["messages"].append({"role": "user", "content": request.message})
        session["messages"].append({"role": "assistant", "content": final_response})
        
        print(f"✅ Response sent (length: {len(final_response)})")
        print(f"   Sources: RAG={sources['rag']}, Context7={sources['context7']}")
        print(f"{'='*60}\n")
        
        return ChatResponse(
            response=final_response,
            session_id=session_id,
            sources=sources
        )
    
    except Exception as e:
        print(f"❌ Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/new", response_model=SessionResponse)
async def create_session():
    try:
        session, session_id = get_or_create_session()
        return SessionResponse(session_id=session_id, message="Session created")
    except Exception as e:
        print(f"Create session error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/clear", response_model=SessionResponse)
async def clear_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        sessions[session_id]["messages"] = []
        return SessionResponse(session_id=session_id, message="History cleared")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}", response_model=SessionResponse)
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        del sessions[session_id]
        return SessionResponse(session_id=session_id, message="Session deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)