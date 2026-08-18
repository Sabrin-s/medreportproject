"""
MedReport Copilot — FastAPI Web Application Backend.
Exposes REST endpoints for medical report analysis, STT dictation,
model training triggers, health checks, and static frontend hosting.
"""

import os
import sys

# Ensure root project directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any

from services.classifier import SpecialtyClassifierService
from services.document import DocumentProcessorService
from services.rag import RAGService
from services.stt import STTService
from agents.deep_orchestrator import DeepAgentOrchestrator
from ml.train_baseline import train_and_evaluate_baseline
from ml.train_classifier import train_distilbert

app = FastAPI(
    title="MedReport Copilot API",
    description="Agentic AI Medical Report Understanding Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
classifier_service = SpecialtyClassifierService()
rag_service = RAGService()
stt_service = STTService()
orchestrator = DeepAgentOrchestrator(classifier_service, rag_service)

# Mount static frontend directory if it exists
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h2>MedReport Copilot API is running. Access /docs for API documentation.</h2>"

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "version": "1.0.0",
        "models": {
            "baseline_loaded": classifier_service.baseline_model is not None,
            "distilbert_loaded": classifier_service.distil_model is not None,
            "whisper_stt_loaded": stt_service.model is not None,
            "rag_chunks_indexed": len(rag_service.documents)
        }
    }

@app.post("/api/analyze")
async def analyze_report(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Main Report Processing Endpoint.
    Accepts raw text or PDF/Audio file upload.
    Routes execution through the Deep Agent Orchestrator.
    """
    extracted_text = ""
    input_type = "text"

    if file:
        file_bytes = await file.read()
        filename = file.filename.lower() if file.filename else ""
        
        if filename.endswith(".pdf"):
            input_type = "pdf"
            extracted_text = DocumentProcessorService.extract_text_from_pdf_bytes(file_bytes)
        elif any(filename.endswith(ext) for ext in [".wav", ".mp3", ".m4a", ".webm", ".ogg"]):
            input_type = "audio"
            stt_res = stt_service.transcribe_audio_bytes(file_bytes, filename)
            extracted_text = stt_res.get("text", "")
        else:
            # Fallback plain text read
            input_type = "text_file"
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
    elif text:
        extracted_text = text.strip()

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No medical report text or valid document provided.")

    cleaned_text = DocumentProcessorService.clean_medical_text(extracted_text)

    # Execute Multi-Agent Pipeline
    result = orchestrator.run({
        "input_type": input_type,
        "content": cleaned_text
    })
    
    result["raw_input_text"] = cleaned_text
    return result

@app.post("/api/stt")
async def process_audio_stt(file: UploadFile = File(...)):
    """Transcribes voice audio recording into text via Whisper."""
    audio_bytes = await file.read()
    stt_res = stt_service.transcribe_audio_bytes(audio_bytes, file.filename or "audio.wav")
    return stt_res

@app.get("/api/knowledge")
async def list_knowledge_base():
    """Lists indexed reference documents."""
    return {
        "total_chunks": len(rag_service.documents),
        "sources": list(set(d["source"] for d in rag_service.documents))
    }

@app.post("/api/train-baseline")
async def trigger_train_baseline(background_tasks: BackgroundTasks):
    """Triggers background training of TF-IDF baseline model."""
    background_tasks.add_task(train_and_evaluate_baseline)
    return {"status": "training_started", "model": "TF-IDF + Logistic Regression Baseline"}

@app.post("/api/train-classifier")
async def trigger_train_classifier(background_tasks: BackgroundTasks):
    """Triggers background fine-tuning of DistilBERT model."""
    background_tasks.add_task(train_distilbert)
    return {"status": "training_started", "model": "DistilBERT Fine-Tuned Specialty Classifier"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
