# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from pathlib import Path
import shutil
import uuid
# Import the OCR function from the new module
from models.ocr_parser import extract_raw_text, extract_structured_data
from app.schemas import ExtractionResponse # Import the response model
from models.rag_pipeline import create_vector_store, load_rag_chain
from typing import Dict, Any

# define constants 
UPLOAD_DIR= Path('data/uploads')
UPLOAD_DIR.mkdir(exist_ok=True) # Ensure directory

# inicialize fastapi
app = FastAPI(title = 'AI Decument Intelligence Platform')

# define the root endpoint (or server heath check)
@app.get('/')
def read_root():
    return {'status': 'Server is running. Upload and Query endpoints available.'}

# placeholder for / upload enpoint
@app.post('/upload',response_model=ExtractionResponse)
async def Upload_document(file : UploadFile =File(...)):
    # generate unique file name and path
    file_extension = Path(file.filename).suffix
    unique_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{unique_id}{file_extension}"

    # check for allowed extensions 
    if file_extension.lower() not in ['.pdf', '.jpg', '.jpeg', '.png']:
        raise HTTPException(status_code = 400, detail='Unsupported file type. Only PDF and scanned images allowed.')
    try:
        # save the file to disk using standard blocking file operations
        with open(save_path,'wb') as buffer:
            # copy the file-like object content to the disk buffer
            shutil.copyfileobj(file.file, buffer)
        # 1. Call OCR to extract raw text (Stage 2)
        raw_text = await extract_raw_text(save_path)
        
        # 2. Call structured extractor (Stage 3 - 40%)
        structured_data = extract_structured_data(raw_text)
        # 3. CRITICAL FIX: Create and save the FAISS vector store (Stage 4)
        create_vector_store(raw_text, unique_id)
        
        # 4. Clean up the uploaded file after processing
        if save_path.exists():
             save_path.unlink() # Delete the temporary file
             
        # 4. Return the validated structured data
        return ExtractionResponse(
            status="Indexing complete. Use file_id to query",
            data=structured_data,
            file_id=unique_id
        )
        
    except Exception as e:
        # Clean up file on failure
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
        


# /query endpoint (RAG Conversation)
@app.post('/query', response_model=Dict[str, Any])
def query_document(file_id : str, question:str):
    """Answers a conversational question using the RAG chain built on the file_id's index."""
    try:
        # 1. Load the RAG chain built for the specific file ID
        qa_chain = load_rag_chain(file_id)
        
        # 2. Invoke the chain with the user's question
        result = qa_chain.invoke({"input": question}) # Use 'input' key for LCEL
        
        # 3. Return the generated answer
        return {
            "file_id": file_id,
            "question": question,
            "answer": result["result"]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index for file ID '{file_id}' not found. Please upload the document first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")