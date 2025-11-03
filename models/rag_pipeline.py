# models/rag_pipeline.py (FINAL & ROBUST LCEL IMPLEMENTATION)
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # For final answer formatting
from langchain_core.runnables import RunnablePassthrough # CRITICAL LCEL component
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

# Define file path constants... (remains the same)

# --- 1. & 2. Indexing Functions (create_vector_store remains the same) ---
def create_vector_store(text_content: str, file_id: str):
    # ... (function body remains the same)
    pass 

# --- 3. & 4. RAG Chain Implementation (LCEL Direct Assembly) ---
def load_rag_chain(file_id: str):
    """Loads the RAG chain using direct LCEL components for robust assembly."""
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
    
    # 1. Load Embeddings and FAISS Retriever
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    load_path = Path(DB_FAISS_PATH) / file_id
    vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 2. Initialize the LLM (HuggingFaceHub API)
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.1, "max_length": 256}
    )

    # 3. Define the RAG Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 4. LCEL Pipeline Assembly: The most stable way to build RAG
    # format_docs function converts retrieved Document objects into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # The final RAG chain (Runnable)
    qa_chain = (
        {
            "context": retriever | format_docs, # Retrieve documents, then format them
            "input": RunnablePassthrough()       # Pass user's query as "input"
        }
        | prompt   # Apply prompt template to context and question
        | llm      # Pass result to LLM for generation
        | StrOutputParser() # Ensure final output is a string
    )
    
    # 5. Wrap the result to match the expected format of app/main.py
    def process_result(result):
        return {"result": result} # Return the string under the 'result' key
    
    return qa_chain | process_result