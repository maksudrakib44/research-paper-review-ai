"""
app.py - Research Paper RAG System
Professional Q&A System for Academic Papers with Ground Truth Evaluation
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Research Paper RAG | AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .feature-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #666;
        border-top: 1px solid #ddd;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #ddd;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PIPELINE INITIALIZATION - SINGLE INSTANCE
# ============================================================
def init_pipeline():
    """Initialize pipeline once and store in session state"""
    if "pipeline" not in st.session_state:
        from config.settings import settings
        from src.pipeline import ResearchPipeline
        
        # Check if API key is available
        if not settings.groq_api_key:
            st.error("❌ Groq API key not found. Please add it to .env file or Streamlit Secrets.")
            st.stop()
        
        with st.spinner("🚀 Loading AI Assistant..."):
            try:
                st.session_state.pipeline = ResearchPipeline(settings)
                st.session_state.pipeline.build_index()
                st.session_state.pipeline_ready = True
            except Exception as e:
                st.session_state.pipeline = None
                st.session_state.pipeline_ready = False
                st.error(f"❌ Pipeline error: {str(e)}")
    
    return st.session_state.pipeline


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>📚 Research RAG</h2>
        <p>AI Academic Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "📖 **Navigation**",
        ["📄 Upload Papers", "💬 Chat with Papers", "🎯 Evaluate Quality", "📋 Ground Truth"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Initialize pipeline
    pipeline = init_pipeline()
    
    if pipeline and st.session_state.get("pipeline_ready", False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Papers", pipeline.paper_count)
        with col2:
            st.metric("🎯 GT Pairs", pipeline.gt_count)
    else:
        st.error("❌ System unavailable - Check API key")
        st.stop()
    
    st.divider()
    
    with st.expander("ℹ️ **How to Use**", expanded=False):
        st.markdown("""
        **1. Upload Papers** → Add PDF/TXT files  
        **2. Ask Questions** → Type your question  
        **3. Get Answers** → AI responds with citations
        """)
    
    with st.expander("⚡ **About the AI**", expanded=False):
        st.markdown("""
        - **Model:** Llama 3.3 70B (Groq)
        - **Speed:** ~2-3 seconds per query
        - **Free Tier:** 30 requests/minute
        """)


# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📚 Research Paper RAG</h1>
    <p>Intelligent Q&A System for Academic Papers | Powered by Advanced AI</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD PAPERS PAGE
# ============================================================
if page == "📄 Upload Papers":
    st.markdown("### 📄 Upload Research Papers")
    st.markdown("Support: **PDF** and **TXT** files | Max size: 200MB per file")
    
    files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if files and st.button("📥 **Ingest Papers**", type="primary", use_container_width=True):
        success_count = 0
        for file in files:
            suffix = f".{file.name.split('.')[-1]}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                tmp_path = Path(tmp.name)
            
            with st.spinner(f"📖 Processing {file.name}..."):
                chunks = pipeline.ingest(tmp_path)
                tmp_path.unlink()
            
            if chunks > 0:
                st.success(f"✅ **{file.name}**: {chunks} chunks")
                success_count += 1
            else:
                st.error(f"❌ **{file.name}**: No text extracted")
        
        if success_count > 0:
            pipeline.build_index()
            st.success(f"🎉 {success_count} paper(s) ingested!")
            st.balloons()
            st.rerun()
    
    st.divider()
    st.markdown(f"**Total chunks indexed:** {pipeline.paper_count}")


# ============================================================
# CHAT PAGE
# ============================================================
elif page == "💬 Chat with Papers":
    st.markdown("### 💬 Ask Questions About Your Papers")
    
    if pipeline.paper_count == 0:
        st.warning("📄 **No papers uploaded yet!**")
        st.info("👉 Go to the **Upload Papers** tab to add research papers.")
        st.stop()
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "👋 Hello! Ask me anything about your uploaded papers!"}
        ]
    
    # Display chat history
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
                <div style="background: #1e3c72; color: white; padding: 0.8rem 1.2rem; border-radius: 20px; max-width: 70%;">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin: 1rem 0;">
                <div style="background: #f0f2f6; color: #333; padding: 0.8rem 1.2rem; border-radius: 20px; max-width: 80%;">
                    📚 {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if question := st.chat_input("Ask about your research papers..."):
        st.session_state.chat_messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing papers..."):
                try:
                    response = pipeline.ask(question)
                    
                    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                    icon = conf_icon.get(response.confidence.value, "⚪")
                    
                    answer_text = f"{icon} **Confidence: {response.confidence.value.upper()}**\n\n{response.answer}"
                    st.markdown(answer_text)
                    
                    if response.paper_references:
                        with st.expander("📎 References", expanded=False):
                            for ref in response.paper_references[:3]:
                                st.markdown(f"- {ref}")
                    
                    st.caption(f"⏱️ {response.latency_ms:.0f} ms | Model: {response.model_used}")
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": response.answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "👋 Hello! Ask me anything about your uploaded papers!"}
        ]
        st.rerun()


# ============================================================
# EVALUATION PAGE
# ============================================================
elif page == "🎯 Evaluate Quality":
    st.markdown("### 🎯 RAG Quality Evaluation")
    
    if pipeline.gt_count == 0:
        st.warning("⚠️ No ground truth pairs found. Add some in Ground Truth tab.")
        st.stop()
    
    if st.button("▶️ Run Evaluation Suite", type="primary"):
        with st.spinner("Running evaluation..."):
            results = pipeline.run_eval_suite()
        
        df = pd.DataFrame(results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card">Faithfulness<br><h2>{df["faithfulness"].mean():.0%}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card">Correctness<br><h2>{df["answer_correctness"].mean():.0%}</h2></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card">Overall<br><h2>{df["overall_score"].mean():.0%}</h2></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card">Pass Rate<br><h2>{df["passed"].mean():.0%}</h2></div>', unsafe_allow_html=True)
        
        st.dataframe(df)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "eval_results.csv")


# ============================================================
# GROUND TRUTH PAGE
# ============================================================
elif page == "📋 Ground Truth":
    st.markdown("### 📋 Ground Truth Management")
    
    tab1, tab2 = st.tabs(["➕ Add Pair", "📋 View Pairs"])
    
    with tab1:
        with st.form("add_gt"):
            question = st.text_area("Question")
            answer = st.text_area("Answer")
            tags = st.text_input("Tags", placeholder="comma,separated")
            
            if st.form_submit_button("Save"):
                if question and answer:
                    from src.models import ResearchGroundTruth
                    pair = ResearchGroundTruth(
                        question=question,
                        ground_truth_answer=answer,
                        domain_tags=[t.strip() for t in tags.split(",") if t.strip()]
                    )
                    pipeline.add_ground_truth(pair)
                    st.success("Saved!")
                    st.rerun()
    
    with tab2:
        pairs = pipeline.list_ground_truth()
        for pair in pairs:
            with st.expander(pair.question[:80]):
                st.write(f"**Answer:** {pair.ground_truth_answer}")
                if st.button("Delete", key=f"del_{pair.gt_id}"):
                    pipeline.delete_ground_truth(pair.gt_id)
                    st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p>🔬 <strong>Research Paper RAG</strong> | AI-Powered Academic Research Assistant</p>
    <p style="font-size: 0.8rem;">Made with 💖 by Md. Maksudul Haque</p>
</div>
""", unsafe_allow_html=True)