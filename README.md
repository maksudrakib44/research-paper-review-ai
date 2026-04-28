
```markdown
# 📚 Research Paper RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system for academic research papers. Upload PDF/TXT papers, ask questions, and get accurate answers with citations.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://research-paper-review-ai.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 📄 **Smart Paper Ingestion** - Process PDF and TXT files with intelligent chunking
- 🔍 **Hybrid Search** - Combines dense (TF-IDF) + keyword (BM25) retrieval
- 🎓 **Ground Truth Evaluation** - RAGAS-style quality metrics (faithfulness, precision, recall)
- 🚀 **Free LLM** - Powered by Groq's Llama 3.3 70B (free tier)
- 📊 **Quality Monitoring** - Track answer quality with labeled Q&A pairs
- 💬 **Modern UI** - Clean Streamlit interface with confidence scores and references
- ⚡ **Lightweight** - Pure Python TF-IDF embeddings, no heavy dependencies

## 🏗️ Architecture

```
User Question → Retrieve Relevant Chunks → Build Context → Groq LLM → Answer + Citations
                        ↓
              FAISS Vector Search + BM25
                        ↓
                    TF-IDF Embeddings
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

Visit: [https://paper-review-ai.streamlit.app/](https://paper-review-ai.streamlit.app/)

## 📁 Project Structure

```
research-paper-rag/
├── app.py                    # Streamlit UI
├── config/
│   └── settings.py           # Configuration
├── src/
│   ├── ingestion/
│   │   ├── faiss_store.py    # FAISS vector store (TF-IDF)
│   │   └── ingester.py       # PDF/TXT ingestion
│   ├── retrieval/
│   │   └── retriever.py      # Hybrid search (dense + BM25)
│   ├── generation/
│   │   └── generator.py      # Groq LLM integration
│   └── validation/
│       └── ground_truth.py   # GT store & RAG evaluator
├── data/
│   ├── chroma_db/            # FAISS index storage
│   ├── ground_truth/         # Q&A pairs
│   └── papers/               # Uploaded papers
├── requirements.txt
└── .env.example
```

## 🎯 Usage Examples

### Upload Papers
1. Go to **📄 Upload Papers** tab
2. Upload PDF or TXT files
3. Click "Ingest Papers"

### Ask Questions
Go to **💬 Chat** tab and ask:

| Question Type | Example |
|---------------|---------|
| Paper Title | "What is the title of this paper?" |
| Authors | "Who are the authors?" |
| Results | "What is the dice coefficient?" |
| Dataset | "Which dataset was used?" |
| Methodology | "What architecture was used?" |

### Evaluate Quality
1. Go to **📋 Ground Truth** tab
2. Add labeled Q&A pairs
3. Go to **🎯 Evaluate Quality** tab
4. Run evaluation suite

## 🔧 Configuration

Create `.env` file:

```env
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=simple
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
TOP_K_DENSE=5
TOP_K_RERANK=3
HYBRID_ALPHA=0.6
GT_EVAL_THRESHOLD=0.75
```

## 📊 Evaluation Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| **Faithfulness** | 35% | Answer grounded in retrieved context? |
| **Context Precision** | 25% | Retrieved chunks relevant to question? |
| **Context Recall** | 20% | All relevant information retrieved? |
| **Answer Correctness** | 20% | Matches ground truth answer? |

**Pass Threshold:** 75% overall score

## 🧠 Core RAG Concepts

| Concept | Implementation |
|---------|----------------|
| **Chunking** | Overlapping sliding window (1500 chars, 200 overlap) |
| **Embedding** | Pure Python TF-IDF (384 features) |
| **Dense Search** | FAISS with cosine similarity |
| **Keyword Search** | BM25 algorithm |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **LLM** | Groq Llama 3.3 70B (free tier) |
| **Evaluation** | RAGAS-style ground truth metrics |

## 🔑 Getting a Groq API Key

1. Go to [Groq Console](https://console.groq.com)
2. Sign up for a free account (no credit card required)
3. Navigate to **API Keys**
4. Create a new key (starts with `gsk_`)
5. Copy the key to your `.env` file or Streamlit secrets

## 📈 Performance

| Metric | Value |
|--------|-------|
| Ingestion | ~100 chunks/second |
| Query latency | 2-3 seconds |
| Context window | 8,192 tokens |
| File support | PDF, TXT (up to 200MB) |
| Chunk size | 1,500 characters |

## 🛠️ Technologies Used

- **Groq** - Fast, free LLM inference (Llama 3.3 70B)
- **FAISS** - Efficient vector similarity search
- **BM25** - Keyword-based retrieval
- **Streamlit** - Interactive web UI
- **Pydantic** - Type-safe configuration
- **FAISS** - Vector database

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request



## 🙏 Acknowledgments

- [Groq](https://groq.com) for free LLM API
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io) for the UI framework

## 📧 Contact

Md. Maksudul Haque - [GitHub](https://github.com/maksudrakib44)

---

**⭐ Star this repository if you find it useful!**

Made with 💖 by Md. Maksudul Haque
```

