import os
import json
import asyncio
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# LangChain Imports - GEMINI EDITION
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# Load env vars
load_dotenv()

app = FastAPI(title="GEMINI RAG CoT VISUALIZER")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE INTERCEPTOR (WebSocket Feedback) ---
class CoTAsyncHandler(AsyncCallbackHandler):
    """
    Custom Async handler for Gemini models.
    Streams retrieval and thought steps.
    """
    def __init__(self, websocket: WebSocket, parent_id: str = None):
        self.websocket = websocket
        self.parent_id = parent_id

    def _get_id(self):
        return str(uuid.uuid4())

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        new_id = self._get_id()
        await self.websocket.send_json({
            "id": new_id,
            "parent_id": self.parent_id,
            "step": "Action",
            "type": "retrieval_start",
            "content": f"Querying Gemini Knowledge: {input_str}"
        })
        self.parent_id = new_id

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        new_id = self._get_id()
        await self.websocket.send_json({
            "id": new_id,
            "parent_id": self.parent_id,
            "step": "Observation",
            "type": "retrieval_end",
            "content": f"Gemini found context: {output[:500]}..."
        })
        self.parent_id = new_id

# --- GEMINI RAG SETUP (Persistent) ---
def initialize_gemini_rag():
    data_dir = "./data/"
    persist_dir = "./chroma_db_gemini/"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return None

    # text-embedding-004 is the modern, stable embedding model for Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Loading Gemini vector database...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("Scrubbing data for Gemini-precision indexing...")
        all_docs = []
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                all_docs.extend(loader.load())
        
        if not all_docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)

        print(f"Persisting {len(splits)} chunks to Gemini DB...")
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory=persist_dir
        )
    
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "local_search",
        "Search your verified PDF or TXT knowledge base for specific facts using Gemini."
    )
    return retriever_tool

rag_tool = initialize_gemini_rag()

# --- THE GEMINI AGENT ---
def create_gemini_agent(callback_handler: CoTAsyncHandler):
    # gemini-2.0-flash-exp is high-performance and widely supported in current SDKs
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, streaming=True)
    
    tools = [rag_tool] if rag_tool else []
    
    # Modern factory-based agent
    agent = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt="You are a helpful assistant powered by Google Gemini. Use tools to verify facts from the local knowledge base."
    )
    return agent

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    root_id = str(uuid.uuid4())
    handler = CoTAsyncHandler(websocket, parent_id=root_id)
    agent_executor = create_gemini_agent(handler)

    try:
        while True:
            data = await websocket.receive_text()
            user_input = json.loads(data).get("question")
            
            if not user_input:
                continue

            print(f"Executing Gemini Reasoning for: {user_input}")
            # Call agent with modern state format
            response = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"callbacks": [handler]}
            )
            
            final_content = response["messages"][-1].content
            
            await websocket.send_json({
                "id": str(uuid.uuid4()),
                "parent_id": handler.parent_id,
                "step": "Final Answer",
                "content": final_content
            })

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
