# ResumeRAG-Pro

ResumeRAG-Pro is a modern full-stack AI project that provides a resumés-focused Retrieval-Augmented Generation (RAG) pipeline. It ingests resumes (PDF / images), extracts structured information using OCR + LLM, indexes content into a FAISS vector store using embeddings, and provides a conversational querying API.

Key features
- Upload and ingest resumes (PDF/JPG/PNG)
- OCR (Tesseract + Poppler) for scanned or image-based resumes
- LLM-powered structured extraction (Mistral-7B-Instruct-v0.2) with Pydantic schemas
- Embeddings via BAAI/bge-small-en-v1.5 and FAISS for fast retrieval
- FastAPI REST endpoints for upload and conversational query
- LangChain (LCEL) used for RAG orchestration (10% bonus)

🛠️ Skills & Technologies Used

Component | Technology | Key Role in Project
--- | ---: | ---
API Framework | FastAPI | High-performance, asynchronous REST endpoints (/upload, /query)
Orchestration | LangChain (LCEL) | Modular RAG chain assembly and data flow (10% Bonus for using LangChain)
Generative Model | Mistral-7B-Instruct-v0.2 | Used via HuggingFaceEndpoint for conversational Q&A and structured extraction
Embedding Model | BAAI/bge-small-en-v1.5 | Generates high-quality vector embeddings for efficient retrieval
Vector Database | FAISS | Fast indexing and similarity search for resume chunks
Data Validation | Pydantic | Enforces strict JSON schema for structured extraction output
OCR Tools (system) | Tesseract-OCR & Poppler | Required for raw text extraction from scanned documents and PDFs

Project Structure

ResumeRAG-Pro/
├── app/  
│   ├── main.py           # FastAPI entry point; handles routing and main logic flow.
│   └── schemas.py        # Pydantic models (ResumeData, ExtractionResponse) defining data contracts.  
├── data/  
│   ├── uploads/          # Temporary storage for uploaded files.  
│   └── index/            # Permanent storage for FAISS vector indices.  
├── models/  
│   ├── ocr_parser.py     # OCR, text cleaning, and LLM-powered structured extraction logic.  
│   └── rag_pipeline.py   # RAG chain construction (LCEL), indexing, and LLM invocation.  
├── requirements.txt      # Frozen list of all Python dependencies.  
└── README.md             # This documentation.

Quick setup & installation

Follow these steps precisely to set up and run the project locally.

Step 1: System prerequisites (install executables)
- Tesseract OCR Engine:
  - Install the Tesseract executable for your OS and ensure the installation path is on your PATH.
- Poppler Utilities (CRITICAL for PDF processing):
  - Required by pdf2image. On Windows, download the Poppler binaries and add the Poppler `bin` folder to your PATH.

Step 2: Environment and dependencies

Clone and navigate:
```bash
git clone 'https://github.com/Anas7089/ResumeRAG-Pro-AI-Document-Intelligence-Platform/edit/main/'
cd ResumeRAG-Pro
```

Create and activate a virtual environment:
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\activate
```

Install Python libraries (frozen):
```bash
pip install -r requirements.txt
```

Step 3: API key configuration (CRITICAL for LLM)

The LLM functionality requires a Hugging Face User Access Token.

- Get token: Generate a User Access Token (with "read" permissions) from your Hugging Face profile.
- Set variable: Set the token as the HUGGINGFACEHUB_API_TOKEN environment variable in the terminal session where you run Uvicorn.

Windows (PowerShell):
```powershell
$env:HUGGINGFACEHUB_API_TOKEN="<PASTE YOUR HF TOKEN HERE>"
```

Linux/macOS (Bash/Zsh):
```bash
export HUGGINGFACEHUB_API_TOKEN='<PASTE YOUR HF TOKEN HERE>'
```

Step 4: Model caching (recommended)
Pre-cache the embedding model (BAAI/bge-small-en-v1.5) to avoid slow downloads or runtime errors on first upload:

1. Activate your virtual environment (see Step 2).
2. Open a Python interpreter:
```bash
python
```
3. Run:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5')
exit()
```

How to run

Start the FastAPI server from the project root:
```bash
uvicorn app.main:app --reload
```

Open the interactive API docs at:
http://127.0.0.1:8000/docs

API Endpoints & Usage Flow

1) Document ingestion and indexing — POST /upload
- Action: Upload a resume file (PDF / JPG / PNG) via the Swagger UI (/docs).
- Pipeline:
  - OCR extracts raw text (Tesseract & Poppler)
  - LLM-powered extraction uses the Mistral model and Pydantic schema to extract structured data (this satisfies the 40% requirement)
  - Indexing: create embeddings for chunks and save the FAISS index to data/index/
- Output:
  - Returns the extracted JSON and an essential file_id (example: bbcb836d-b387-42e3-8203-e21608a04ebf). Save this ID for querying.

2) Conversational querying — POST /query
- Action: POST a JSON payload to /query:
```json
{
  "file_id": "bbcb836d-b387-42e3-8203-e21608a04ebf",
  "question": "Summarize the candidate's professional experience."
}
```
- Process:
  - The LCEL RAG chain loads the FAISS index for the provided file_id
  - Retrieval returns context passages
  - The Mistral LLM generates a grounded conversational answer
- Output:
  - Returns the conversational answer 

Notes and troubleshooting
- Ensure Tesseract and Poppler are installed and available on PATH; failures in OCR are most commonly due to missing executables.
- If LLM calls fail, confirm HUGGINGFACEHUB_API_TOKEN is set and valid.
- Pre-caching the embedding model avoids long waits and memory spikes on first upload.
- FAISS indexes are stored under data/index/ — back these up if you intend to preserve indexes across systems.

Extending the project
- Swap or add other embedding models or LLMs via the rag_pipeline.py configuration.
- Add authentication to the FastAPI endpoints for production use.
- Add a background worker (e.g., Celery or RQ) for asynchronous indexing if you need to scale uploads.

Acknowledgements & References
- LangChain (LCEL) — RAG orchestration and chaining
- Hugging Face — models and HuggingFaceEndpoint integration
- FAISS — similarity search and indexing
- Tesseract OCR, Poppler — system OCR dependencies
- SentenceTransformers / BAAI/bge-small-en-v1.5 — embeddings


Contact
- Maintainer: Anas7089

