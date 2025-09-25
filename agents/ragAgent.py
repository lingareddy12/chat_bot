from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import os
import chromadb # Import for conceptual RAG
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()



RAG_EMBEDDING_MODEL = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2024-05-01-preview" 
    )
    
    # 2. Load the persistent vector database (only once)
PERSISTENT_VECTOR_DB = Chroma(
        embedding_function=RAG_EMBEDDING_MODEL,
        persist_directory="./chroma_vector_db"
    )



def chroma_db_retriever(query: str) -> List[str]:
    """
    Retrieves the top 'k' most relevant documents from the  knowledge base 
    based on the user's query.
    """
    if PERSISTENT_VECTOR_DB is None:
        return ["ERROR: The RAG knowledge base is unavailable. Cannot perform search."]
        

    # Perform the similarity search on the pre-loaded DB instance
    results = PERSISTENT_VECTOR_DB.similarity_search(query, k=5)

    # Format the results into a clean list of strings for the LLM
    context = []
    for r in results:
        # Extract the full file path from the 'source' metadata
        full_path = r.metadata.get('source', 'Unknown Document')
        
        # Use os.path.basename to get just the filename (e.g., "report.pdf")
        document_name = os.path.basename(full_path)
        
        page_number = r.metadata.get('page', 'N/A')
        
        # Include both the Document Name and the Page Number in the output
        context.append(
            f"Source Document (File: {document_name}, Page: {page_number}): {r.page_content}"
        )
    
    if not context:
        return ["No specific context found for this query in the knowledge base."]

    return context


# --- 3. Create the Pure RAG LlmAgent ---
ragAgent = LlmAgent(
    name="rag_knowledge_expert",
    instruction=(
   "You are a dedicated knowledge retrieval agent. Your only goal is to answer "
        "factual questions by searching the internal knowledge base (RAG).\n\n"
        "INSTRUCTIONS:\n"
        "- **ALWAYS** use the **chroma_db_retriever** tool to find context before answering "
        "any factual question, even if you think you know the answer.\n"
        "- When providing the final answer, structure it clearly and **cite the exact Source Document (File Name and Page)** "
        "from the retrieved context for every piece of information used.\n"
        "- **MANDATORY FORMATTING:** For the first piece of factual information cited in your final response, "
        "you must explicitly mention the **full source file name** from which it came. Example: 'The recommended protocol is X (Source: report.pdf, Page 5).'\n"
        "- Only generate your final response after synthesizing the context retrieved by the tool."
    ),
    tools=[
        chroma_db_retriever, 
    ],
    model="gemini-2.5-flash", 
)