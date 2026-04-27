
```markdown
# 📚 Research Paper RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system for academic research papers. Upload PDF papers, ask questions, and get cited answers with high accuracy.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://research-paper-review-ai.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 📄 **Smart Paper Ingestion** - Process PDF research papers with section-aware chunking
- 🔍 **Hybrid Search** - Combines semantic (dense) + keyword (BM25) retrieval
- 🎓 **Academic Metrics** - Citation accuracy, factual consistency, RAGAS evaluation
- 🚀 **Free LLM** - Powered by Groq's Llama 3.3 70B (free tier)
- 📊 **Ground Truth Evaluation** - Quality monitoring with labeled Q&A pairs
- 💬 **Interactive UI** - Streamlit interface with citations and confidence scores

## 🏗️ Architecture

```
User Question → HyDE Rewriting → Dense + BM25 Search → RRF Fusion → Groq LLM → Answer
                                      ↓
                             Section-Aware Chunking
                                      ↓
                                  FAISS Index
```

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/maksudrakib44/research-paper-review-ai.git
cd research-paper-review-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your Groq API key

# Run the app
streamlit run app.py
```

### Deployed Version

Visit: [https://research-paper-review-ai.streamlit.app/](https://research-paper-review-ai.streamlit.app)

## 📁 Project Structure

```
research-paper-rag/
├── app.py                    # Streamlit UI
├── config/
│   └── settings.py           # Configuration
├── src/
│   ├── ingestion/            # PDF processing & chunking
│   │   ├── faiss_store.py    # Vector database
│   │   ├── ingester.py       # Document ingestion
│   │   └── section_chunker.py # Academic section splitting
│   ├── retrieval/
│   │   └── retriever.py      # Hybrid search (dense + BM25)
│   ├── generation/
│   │   └── generator.py      # Groq LLM integration
│   └── validation/
│       ├── ground_truth.py   # Q&A storage
│       └── metrics.py        # RAGAS evaluation
├── requirements.txt
└── .env.example
```

## 🎯 Usage Examples

### Upload a Paper
1. Go to **📄 Papers** tab
2. Upload a PDF research paper
3. Click "Ingest Papers"

### Ask Questions
Go to **💬 Chat** tab and ask:

| Question Type | Example |
|---------------|---------|
| Methodology | "What architecture was used for segmentation?" |
| Results | "What was the Dice coefficient achieved?" |
| Dataset | "Which dataset was used for validation?" |
| Authors | "Who are the authors of this paper?" |

### Evaluate Quality
1. Go to **📋 Ground Truth** tab
2. Add labeled Q&A pairs
3. Go to **🎯 Evaluation** tab
4. Run the evaluation suite

## 🔧 Configuration

Create `.env` file (copy from `.env.example`):

```env
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama-3.3-70b-versatile
CHUNK_SIZE=1024
TOP_K_DENSE=15
TOP_K_RERANK=8
HYBRID_ALPHA=0.6
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Answer grounded in retrieved context? |
| **Context Precision** | Retrieved chunks relevant to question? |
| **Citation Accuracy** | Are (Author, Year) citations correct? |
| **Factual Consistency** | No hallucinated information? |
| **Answer Correctness** | Matches ground truth answer? |

## 🧠 Core RAG Concepts

| Concept | Implementation |
|---------|----------------|
| **Chunking** | Section-aware sliding window |
| **Embedding** | all-MiniLM-L6-v2 (384-dim) |
| **Dense Search** | FAISS with cosine similarity |
| **Keyword Search** | BM25 algorithm |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Query Rewriting** | HyDE (Hypothetical Document Embeddings) |
| **LLM** | Groq Llama 3.3 70B |

## 🔑 Getting a Groq API Key

1. Go to [Groq Console](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys
4. Create a new key (starts with `gsk_`)
5. Copy the key to your `.env` file

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Ingestion speed | ~100 chunks/second |
| Query latency | 2-3 seconds |
| Context window | 8,192 tokens |
| Max paper size | Unlimited (chunked) |



## 📧 Contact

Md. Maksudul Haque - [GitHub](https://github.com/maksudrakib44)

---

**⭐ Star this repository if you find it useful!**
```

---
