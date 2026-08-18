# MedReport Copilot — Agentic AI Medical Report Understanding Platform

**MedReport Copilot** is a high-performance, agentic AI platform designed to ingest complex medical reports (PDF, plain text, or voice dictation audio), perform fine-tuned ML medical specialty classification, execute confidence-based inference routing, orchestrate an 8-agent pipeline (or Deep Agent / LangGraph orchestrator), cross-verify with RAG evidence, enforce strict clinical safety guardrails, and present patient-friendly explanations with browser Text-to-Speech (TTS).

# Deploy Link https://medreportproject.onrender.com/

---

## Technical Stack & Architecture

- **Backend Framework**: Python 3.10, FastAPI, Uvicorn, Pydantic
- **Machine Learning & NLP**: PyTorch, HuggingFace `transformers` (DistilBERT), `scikit-learn` (TF-IDF + Logistic Regression Baseline)
- **Agentic Orchestration**: Deep Agent / LangGraph state graph architecture (8 subagents)
- **STT & Speech Processing**: OpenAI Whisper (Local inference), HTML5 Web Audio & Web Speech Synthesis (TTS)
- **RAG Engine**: Local TF-IDF / Cosine Similarity vector retrieval over vetted medical reference guidelines
- **Containerization & Ops**: Docker multi-stage build, Docker Compose, PyTest suite, Makefile

---

## 8 Multi-Agent Architecture Workflow

```
                        Medical Report / PDF / Voice Audio
                                        │
                                        ▼
                             Intake & STT (Whisper)
                                        │
                                        ▼
                        Agent 1: Intake & Router Agent
                                        │
                                        ▼
                     Agent 3: Medical Specialty Classifier
                           (DistilBERT / TF-IDF)
                                        │
                           ┌────────────┴────────────┐
               Confidence >= 0.75         Confidence < 0.75
                           │                         │
                           ▼                         ▼
                       FAST PATH             DEEP AGENT PATH
                    (Direct Stream)       (LangGraph Orchestrator)
                           │                         │
                           └────────────┬────────────┘
                                        ▼
                         Agent 2: Clinical NLP Agent
                     (Extract Symptoms, Vitals, Tests, Meds)
                                        │
                                        ▼
                       Agent 4: Evidence / RAG Agent
                        (Vector/TF-IDF KB Retrieval)
                                        │
                                        ▼
                      Agent 7: Patient Explanation Agent
                           (Plain English Translation)
                                        │
                                        ▼
                         Agent 5: Fact Checker Agent
                       (Detect Unsupported Claims)
                                        │
                                        ▼
                         Agent 6: Safety Guardrails
                        (Clinical Safety & Disclaimers)
                                        │
                                        ▼
                         Agent 8: Citation Verifier
                       (Verify Sources & Annotate)
                                        │
                                        ▼
                        FastAPI Web Dashboard + TTS
```

---

## ML Benchmark Story (Interview Narrative)

> "In building MedReport Copilot, I established a **TF-IDF + Logistic Regression baseline** (held-out Macro-F1 ~0.89) before fine-tuning **DistilBERT** on clinical report classification. I compared them on held-out test partitions using accuracy, Macro-F1, weighted-F1, precision, recall, and confusion matrices.
> 
> The classifier's confidence score feeds our **Intelligent Inference Router**: predictions with $\ge 75\%$ confidence take the Fast Path to minimize latency and API cost, while sub-threshold predictions trigger the full **Deep Agent Orchestrator** for multi-step RAG retrieval and double verification."

---

## Quick Start & Running Commands

### 1. Local Setup & Execution

```bash
# Navigate to project directory
cd D:\medical_report_ai

# Install dependencies
pip install -r requirements.txt

# Train baseline model & fine-tune DistilBERT
python ml/train_baseline.py
python ml/train_classifier.py

# Evaluate model metrics
python ml/evaluate_classifier.py

# Run unit tests
pytest

# Start FastAPI application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at: `http://localhost:8000`

---

### 2. Docker Deployment

#### Build Docker Image
```bash
docker build -t medreport-copilot .
```

#### Run Docker Container
```bash
docker run --rm -p 8000:8000 --env-file .env medreport-copilot
```

#### Run via Docker Compose
```bash
docker-compose up --build
```

Access application at `http://localhost:8000`

---

## Project File Map

```
D:\medical_report_ai/
├── app/
│   └── main.py                     # FastAPI application server & REST endpoints
├── agents/
│   ├── router_agent.py             # Intake & Router Agent
│   ├── clinical_nlp_agent.py       # Clinical NLP Entity Extractor
│   ├── specialty_agent.py          # Medical Specialty Classifier Agent
│   ├── rag_agent.py                # Evidence / RAG Agent
│   ├── fact_checker_agent.py       # Fact Checker (Hallucination detector)
│   ├── safety_agent.py             # Clinical Safety Guardrails
│   ├── patient_explanation_agent.py# Patient-Friendly Plain English Translation
│   ├── citation_agent.py           # Citation Verifier & Annotator
│   └── deep_orchestrator.py        #  Deep Agent / LangGraph Orchestrator
├── ml/
│   ├── dataset.py                  # Stratified clinical dataset generator
│   ├── train_baseline.py           # TF-IDF + Logistic Regression trainer
│   ├── train_classifier.py         # PyTorch DistilBERT fine-tuning pipeline
│   └── evaluate_classifier.py      # Benchmark evaluation CLI
├── services/
│   ├── classifier.py               # ML model loader & confidence scoring service
│   ├── document.py                 # PDF parsing & text cleaner
│   ├── rag.py                      # RAG knowledge base search engine
│   └── stt.py                      # OpenAI Whisper speech-to-text dictation service
├── frontend/
│   ├── index.html                  # Responsive Web Dashboard layout
│   ├── style.css                   # Custom Slate/Teal Vanilla CSS
│   └── app.js                      # MediaRecorder voice audio, REST client, TTS engine
├── data/knowledge/                 # Vetted clinical guidelines reference documents
├── models/                         # Trained model artifacts (.joblib / PyTorch checkpoints)
├── tests/                          # PyTest unit & integration tests
├── Dockerfile                      # Production multi-stage Dockerfile
├── docker-compose.yml              # Container orchestration manifest
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment settings template
├── Makefile                        # Command shortcuts
└── README.md                       # Architecture & documentation
```
