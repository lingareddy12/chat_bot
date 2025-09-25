# Install required packages:
# pip install chromadb openai langchain

from typing import List, Dict
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, SystemMessage

from collections import defaultdict, deque

from dotenv import load_dotenv
load_dotenv()
import os

MAX_DOCS_PER_USER = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_RECENT_HISTORY=10

class ChatContextManager:
    def __init__(self, persist_directory: str = "./chroma_store"):
        """
        api_key: OpenAI API key
        persist_directory: path to save/load vector DB locally
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.deployment=os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        
        # LangChain Embedding Model
        self.embedding_model = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")
        
        # LangChain LLM Model for summarization
        self.llm = AzureChatOpenAI(azure_deployment=self.deployment,api_version="2024-05-01-preview")

        # Text splitter for chunking AI responses
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Load or create persistent Chroma vector DB
        self.persist_directory = persist_directory
        try:
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        except:
            self.vector_db = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )

      
        self.user_history = defaultdict(lambda: deque(maxlen=MAX_RECENT_HISTORY)) 
    
    def _llm_summarize(self, messages: List[str]) -> str:
        if not messages:
            return ""
        combined_text = "\n".join(messages)
        system_msg = SystemMessage(content="You are a helpful assistant that summarizes conversations concisely.")
        human_msg = HumanMessage(content=f"Summarize the following conversation:\n{combined_text}")
        summary = self.llm([system_msg, human_msg]).content.strip()
        return summary
    
    def _check_and_summarize(self, user_id: str):
        """Summarize old messages if docs exceed threshold."""
        # Fetch all messages for the user
        results = self.vector_db.similarity_search("", k=30, filter={"user_id": user_id})
        
        if len(results) > MAX_DOCS_PER_USER:
            # Extract messages from Chroma Documents
            old_messages = [doc.page_content for doc in results if doc.metadata.get("role") in ["user", "ai_response"]]
            
            # Summarize
            summary_text = self._llm_summarize(old_messages)
            
            # Delete old messages
            for doc in results:
                if doc.metadata.get("role") in ["user", "ai_response"]:
                    self.vector_db.delete(ids=[doc.metadata.get("id")])
            
            # Add summary
            chunks = self.text_splitter.split_text(summary_text)
            metadatas = [{"user_id": user_id, "role": "summary"} for _ in chunks]
            self.vector_db.add_texts(texts=chunks, metadatas=metadatas)
            # self.vector_db.persist()
            
       
    
    def add_user_message(self, user_id: str, message: str):
        # Add user message
        self.vector_db.add_texts(
            texts=[message],
            metadatas=[{"user_id": user_id, "role": "user"}]
        )

        self.user_history[user_id].append({"role": 'user', "message": message})

        # self.vector_db.persist()
        self._check_and_summarize(user_id)
    
    def add_ai_response(self, user_id: str, message: str):
        # Chunk AI response
        chunks = self.text_splitter.split_text(message)
        for chunk in chunks:
            self.vector_db.add_texts(
                texts=[chunk],
                metadatas=[{"user_id": user_id, "role": "ai_response"}]
            )

        self.user_history[user_id].append({"role": 'ai', "message": message})

        # self.vector_db.persist()
        self._check_and_summarize(user_id)
    
    def get_context(self, user_id: str, query: str, top_k: int = 5) -> str:
        results = self.vector_db.similarity_search(query, k=top_k, filter={"user_id": user_id})
        
        context_text = "\n".join([f"{doc.metadata.get('role')}: {doc.page_content}" for doc in results])
        return context_text

    
    def get_recent_conversation(self, user_id: str, n: int = 5):
        """
        Returns the last n messages for the user in order.
        """
        history = self.user_history.get(user_id, [])
        if not history:
            return "No history found"
        return "\n".join([f"{msg['role']}: {msg['message']}" for msg in history[-n:]])


    
    def delete_user_data(self, user_id: str):
        """
        Delete all documents associated with a specific user_id from the vector DB.
        """
        # Fetch all documents for the user
        results = self.vector_db.similarity_search("", k=25)  # fetch all docs
        ids_to_delete = []

        for doc in results:
            metadata = doc.metadata
            if metadata.get("user_id") == user_id:
                # Use Chroma's internal ID if available
                if hasattr(doc, "id") and doc.id is not None:
                    ids_to_delete.append(doc.id)
                else:
                    # Some Chroma versions store doc_id in metadata manually
                    doc_id = metadata.get("doc_id") or metadata.get("id")
                    if doc_id:
                        ids_to_delete.append(doc_id)

        if ids_to_delete:
            self.vector_db.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} documents for user_id '{user_id}'.")
        else:
            print(f"No documents found for user_id '{user_id}'.")



