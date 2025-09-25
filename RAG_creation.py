from langchain_community.document_loaders import DirectoryLoader
# Changed from TextLoader to PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Define the path where the ChromaDB will be persisted
CHROMA_DB_PATH = "./chroma_vector_db"

# --- RAG Insertion Class ---
class RagVectorDB:
    def __init__(self):
        
        # 1. Define Embedding Model
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2024-05-01-preview" 
        )
        
        # 2. Define LLM Model (Uses environment variable for deployment name)
        # Note: You should ensure AZURE_OPENAI_CHAT_DEPLOYMENT or similar is set in your .env
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_version="2024-05-01-preview"
        )
        
    def create_vector_db(self, doc_directory: str, db_path: str = CHROMA_DB_PATH) -> Chroma:
        """
        Loads PDF documents, splits them, and indexes them into a persistent ChromaDB.
        """
        
        print("--- Step 1: Loading PDF Documents ---")
        try:
            # 1. UPDATED: Change glob to search for .pdf files
            # 2. UPDATED: Change loader_cls to PyPDFLoader
            loader = DirectoryLoader(
                path=doc_directory, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            
            if not documents:
                print(f"Loaded 0 documents. Please check that '{doc_directory}' contains .pdf files.")
                return None
                
            print(f"Loaded {len(documents)} document pages from {doc_directory}.")
            
        except Exception as e:
            print(f"Error loading documents (Did you install 'pypdf'?): {e}")
            return None

        print("--- Step 2: Splitting Documents ---")
        # Split documents into smaller, meaningful chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")

        print("--- Step 3: Indexing (Creating Embeddings and Vector Store) ---")
        # Create and persist the Chroma vector store using Azure Embeddings
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model, 
            persist_directory=db_path       
        )
        
        vector_db.persist()
        print(f"Successfully created and persisted VectorDB at {db_path}.")
        
        return vector_db


if __name__ == "__main__":
    # Ensure your .env file has AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and 
    # AZURE_OPENAI_CHAT_DEPLOYMENT (or similar) set.
    
    rag_service = RagVectorDB()
    
    # Use the absolute path if 'docs' is not a sibling of the script
    # If the docs folder is a sibling of the script, r"docs" is correct.
    chroma_db = rag_service.create_vector_db(doc_directory=r"docs")

    if chroma_db:
        print("\nRAG DB created successfully. Ready for retrieval.")
    else:
        print("\nRAG DB creation failed.")