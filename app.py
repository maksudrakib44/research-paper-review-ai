"""
app.py - Research Paper RAG Streamlit UI
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing Research RAG Pipeline...")
def get_pipeline():
    """Initialize the RAG pipeline with error handling"""
    from config.settings import Settings
    from src.pipeline import ResearchPipeline
    
    api_key = st.session_state.get("groq_api_key", "")
    
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key")
        return None
    
    if api_key == "your_groq_api_key_here":
        st.error("❌ Please replace the placeholder with your actual Groq API key")
        return None
    
    try:
        # Validate API key format
        if not api_key.startswith("gsk_"):
            st.error("❌ Invalid API key format. Groq API keys should start with 'gsk_'")
            return None
        
        settings = Settings(groq_api_key=api_key)
        
        # Test Groq connection before initializing pipeline
        try:
            from groq import Groq
            test_client = Groq(api_key=api_key)
            test_response = test_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5
            )
            st.success("✅ Groq API connection successful!")
        except Exception as e:
            st.error(f"❌ Groq API test failed: {str(e)}")
            return None
        
        pipeline = ResearchPipeline(settings)
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Pipeline initialization failed: {str(e)}")
        st.code(str(e), language="python")
        return None


# Sidebar
with st.sidebar:
    st.title("📚 Research Paper RAG")
    st.caption("RAG System for Academic Papers")
    st.divider()
    
    page = st.radio("Navigate", ["💬 Chat", "📄 Papers", "🎯 Evaluation", "📋 Ground Truth"])
    
    st.divider()
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        key="groq_api_key",
        help="Get your API key from https://console.groq.com",
        placeholder="gsk_..."
    )
    
    if api_key:
        pipeline = get_pipeline()
    else:
        pipeline = None
        st.info("🔑 Enter your Groq API key to begin")
    
    if pipeline:
        st.success("✅ Pipeline ready")
        st.metric("Papers indexed", pipeline.paper_count)
        st.metric("Ground truth", pipeline.gt_count)
    elif api_key:
        st.error("Failed to initialize pipeline. Check the error above.")


# Chat Page
if page == "💬 Chat":
    st.header("💬 Research Paper Q&A", divider="gray")
    
    if not pipeline:
        st.info("📝 Please enter your Groq API key in the sidebar and wait for initialization.")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if question := st.chat_input("Ask about your research papers..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching papers and generating answer..."):
                try:
                    response = pipeline.ask(question)
                    st.markdown(response.answer)
                    
                    if response.paper_references:
                        with st.expander("📎 References", expanded=True):
                            for ref in response.paper_references:
                                st.markdown(f"- {ref}")
                    
                    st.caption(f"⏱ {response.latency_ms:.0f} ms | Confidence: {response.confidence.value}")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        
        st.session_state.messages.append({"role": "assistant", "content": response.answer if 'response' in locals() else "Error"})


# Papers Page
elif page == "📄 Papers":
    st.header("📄 Paper Management", divider="gray")
    
    if not pipeline:
        st.info("📝 Please enter your Groq API key in the sidebar.")
        st.stop()
    
    st.subheader("Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload academic papers in PDF or TXT format"
    )
    
    if uploaded_files and st.button("📥 Ingest Papers", type="primary"):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                tmp.write(file.read())
                tmp_path = Path(tmp.name)
            
            with st.spinner(f"Ingesting {file.name}..."):
                try:
                    chunks = pipeline.ingest(tmp_path)
                    st.success(f"✅ {file.name}: {chunks} chunks")
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")
                finally:
                    tmp_path.unlink(missing_ok=True)
        
        pipeline.build_index()
        st.success("✅ Index rebuilt!")
        st.rerun()
    
    st.divider()
    st.metric("Total chunks in index", pipeline.paper_count)


# Ground Truth Page
elif page == "📋 Ground Truth":
    st.header("📋 Ground Truth Management", divider="gray")
    
    if not pipeline:
        st.info("📝 Please enter your Groq API key in the sidebar.")
        st.stop()
    
    tab1, tab2 = st.tabs(["➕ Add Pair", "📋 View Pairs"])
    
    with tab1:
        with st.form("add_gt"):
            question = st.text_area("Question", placeholder="What is the main contribution of this paper?")
            answer = st.text_area("Ground Truth Answer", placeholder="The paper introduces a novel method for...")
            tags = st.text_input("Tags (comma-separated)", placeholder="LLM, RAG, transformers")
            
            if st.form_submit_button("Add Ground Truth Pair"):
                from src.models import ResearchGroundTruth
                if question and answer:
                    pair = ResearchGroundTruth(
                        question=question,
                        ground_truth_answer=answer,
                        domain_tags=[t.strip() for t in tags.split(",") if t.strip()]
                    )
                    pipeline.add_ground_truth(pair)
                    st.success("✅ Added!")
                    st.rerun()
    
    with tab2:
        pairs = pipeline.list_ground_truth()
        if not pairs:
            st.info("No ground truth pairs yet.")
        else:
            for pair in pairs:
                with st.expander(pair.question[:80]):
                    st.write(f"**Answer:** {pair.ground_truth_answer}")
                    st.write(f"**Tags:** {', '.join(pair.domain_tags)}")
                    if st.button("Delete", key=f"del_{pair.gt_id}"):
                        pipeline.delete_ground_truth(pair.gt_id)
                        st.rerun()


# Evaluation Page
elif page == "🎯 Evaluation":
    st.header("🎯 RAG Evaluation Dashboard", divider="gray")
    
    if not pipeline:
        st.info("📝 Please enter your Groq API key in the sidebar.")
        st.stop()
    
    if pipeline.gt_count == 0:
        st.warning("No ground truth pairs. Add some in the Ground Truth page.")
        st.stop()
    
    if st.button("▶️ Run Evaluation Suite", type="primary"):
        with st.spinner("Running evaluation..."):
            results = pipeline.run_eval_suite()
        
        import pandas as pd
        df = pd.DataFrame(results)
        
        st.subheader("📊 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.1%}" if df['faithfulness'].notna().any() else "N/A")
        col2.metric("Avg Overall Score", f"{df['overall_score'].mean():.1%}" if df['overall_score'].notna().any() else "N/A")
        col3.metric("Pass Rate", f"{df['passed'].mean():.1%}" if df['passed'].notna().any() else "N/A")
        
        st.dataframe(df)
        
        st.download_button(
            "📥 Download Results CSV",
            df.to_csv(index=False),
            "eval_results.csv"
        )