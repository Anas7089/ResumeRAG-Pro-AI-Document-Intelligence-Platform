# models/ocr_parser.py
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
import re
from app.schemas import ResumeData, Experience, Education # Import schemas
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser # NEW IMPORT
# from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint

# Function to perform OCR on an image file
def _ocr_image(image_path: Path) -> str:
    """Uses Tesseract to extract text from a single image."""
    return pytesseract.image_to_string(Image.open(image_path))

# Main function to handle PDF or image files and return raw text
async def extract_raw_text(file_path: Path) -> str:
    """Handles PDF/scanned image, converts to images if PDF, and performs OCR."""
    raw_text = ""
    file_extension = file_path.suffix.lower()

    if file_extension in ['.jpg', '.jpeg', '.png', '.tiff']:
        raw_text = _ocr_image(file_path)
    elif file_extension == '.pdf':
        # 1. Convert PDF to list of images (one per page)
        images = convert_from_path(file_path) # NOTE: Requires poppler on system path
        # 2. Perform OCR on each page and concatenate text
        for image in images:
            raw_text += pytesseract.image_to_string(image) + "\n\n"
            
    # Clean up excess whitespace and return
    return raw_text.strip()

def _clean_text(text: str) -> str:
    # handling common ocr errors and excess space
    # removeing excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # removeing hyphens at the end of a line
    text = re.sub(r'-\n', '', text)
    return text.strip()

# Helper function to get the LLM (same setup as load_rag_chain but simpler)
def get_extraction_llm():
    """Initializes a simple HuggingFace LLM using the official Endpoint."""
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Use HuggingFaceEndpoint for robust API calls, especially with JSON schemas
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        # NOTE: The HuggingFaceEndpoint should automatically pick up HUGGINGFACEHUB_API_TOKEN
        temperature=0.3, 
        max_new_tokens=2048, # Use max_new_tokens for the endpoint
    )
    return llm

def extract_structured_data(raw_text: str) -> ResumeData:
    """Uses LLM to extract structured data from raw text into ResumeData schema."""
    llm = get_extraction_llm()
    parser = JsonOutputParser(pydantic_object=ResumeData)
    
    # Template to guide the LLM's JSON output
    extraction_template = (
        "You are an expert resume parsing AI. Extract all structured information from the RAW TEXT "
        "provided below. Output the result strictly as a JSON object that conforms EXACTLY to the "
        "following Pydantic schema structure. Do not include any introductory or explanatory text outside the JSON block.\n\n"
        "SCHEMA: {schema}\n\n"
        "RAW TEXT:\n{raw_text}"
    )

    prompt = ChatPromptTemplate.from_template(extraction_template)
    
    # Build the extraction chain using LCEL pipe syntax
    extraction_chain = prompt | llm | parser

    try:
        # Invoke the chain
        llm_output_dict = extraction_chain.invoke({
            "raw_text": raw_text,
            "schema": parser.get_format_instructions(),
        })
        
        # Validate and return the Pydantic model
        return ResumeData(**llm_output_dict)
        
    except Exception as e:
        print(f"LLM extraction failed: {e}. Falling back to simple regex.")
        
        # FALLBACK: If LLM fails, execute your simple regex extraction here
        # For assignment completion, this must now contain robust parsing logic
        
        # Temporary placeholder for demonstration:
        return ResumeData(
            document_title="Extraction Fallback Used", 
            candidate_name="Extraction Fallback Used", 
            email="fallback@example.com",
            phone="000-000-0000",
            professional_summary="LLM extraction failed, using placeholder data.",
            key_skills=["Placeholder"],
            work_experience=[],
            education=[]
        )