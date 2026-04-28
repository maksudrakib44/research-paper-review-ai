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
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
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
    
    /* Feature cards */
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
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    
    /* Chat message styling */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 20px;
        max-width: 70%;
    }
    
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
    }
    
    .assistant-bubble {
        background: #f0f2f6;
        color: #333;
        padding: 0.8rem 1.2rem;
        border-radius: 20px;
        max-width: 80%;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #666;
        border-top: 1px solid #ddd;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #ddd;
        margin-bottom: 1rem;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZATION
# ============================================================
@st.cache_resource
def get_pipeline():
    """Initialize the RAG pipeline"""
    from config.settings import settings
    from src.pipeline import ResearchPipeline
    try:
        pipeline = ResearchPipeline(settings)
        pipeline.build_index()
        return pipeline
    except Exception as e:
        st.error(f"❌ Pipeline initialization failed: {str(e)}")
        return None


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
    
    # Navigation
    page = st.radio(
        "📖 **Navigation**",
        ["📄 Upload Papers", "💬 Chat with Papers", "🎯 Evaluate Quality", "📋 Ground Truth"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # System Status
    pipeline = get_pipeline()
    
    if pipeline:
        st.markdown("### ✅ System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Papers", pipeline.paper_count, delta=None)
        with col2:
            st.metric("🎯 GT Pairs", pipeline.gt_count, delta=None)
    else:
        st.error("❌ System unavailable")
        st.stop()
    
    st.divider()
    
    # Info Section
    with st.expander("ℹ️ **How to Use**", expanded=False):
        st.markdown("""
        **1. Upload Papers**  
        Go to Upload tab → Add PDF/TXT files → Click Ingest
        
        **2. Ask Questions**  
        Go to Chat tab → Type your question → Get AI answer
        
        **3. Evaluate Quality**  
        Add Ground Truth pairs → Run Evaluation suite
        
        **4. View Metrics**  
        See faithfulness, correctness scores
        """)
    
    with st.expander("⚡ **About the AI**", expanded=False):
        st.markdown("""
        - **Model:** Llama 3.3 70B (Groq)
        - **Speed:** ~2-3 seconds per query
        - **Free Tier:** 30 requests/minute
        - **Privacy:** Your papers stay local
        """)
    
    with st.expander("📌 **Tips**", expanded=False):
        st.markdown("""
        - Use **TXT files** for best results
        - PDFs must have selectable text
        - Ask specific questions
        - Try different phrasings
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
# CHAT PAGE
# ============================================================
if page == "💬 Chat with Papers":
    st.markdown("### 💬 Ask Questions About Your Papers")
    
    # Check if papers are uploaded
    if pipeline.paper_count == 0:
        st.warning("📄 **No papers uploaded yet!**")
        st.info("👉 Go to the **Upload Papers** tab to add research papers first.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your research assistant. Ask me anything about your uploaded papers!"}
        ]
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="assistant-bubble">
                    📚 {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if question := st.chat_input("Ask about your research papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching papers and generating answer..."):
                response = pipeline.ask(question)
            
            # Display confidence badge
            conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}
            icon = conf_icon.get(response.confidence.value, "⚪")
            
            answer_display = f"{icon} **Confidence: {response.confidence.value.upper()}**\n\n{response.answer}"
            st.markdown(answer_display)
            
            # Show references
            if response.paper_references:
                with st.expander("📎 **References**", expanded=False):
                    for ref in response.paper_references[:3]:
                        st.markdown(f"- {ref}")
            
            # Show evaluation metrics if available
            if response.eval_metrics:
                with st.expander("📊 **Quality Metrics**", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Faithfulness", f"{response.eval_metrics.faithfulness:.0%}")
                    with col2:
                        st.metric("Correctness", f"{response.eval_metrics.answer_correctness:.0%}")
                    with col3:
                        st.metric("Overall", f"{response.eval_metrics.overall_score:.0%}")
                    with col4:
                        st.metric("Passed", "✅" if response.eval_metrics.passed else "❌")
            
            # Show latency
            st.caption(f"⏱️ {response.latency_ms:.0f} ms | Model: {response.model_used}")
            
            # Store response
            st.session_state.messages.append({"role": "assistant", "content": response.answer})
    
    # Clear chat button
    if st.button("🗑️ **Clear Chat History**", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your research assistant. Ask me anything about your uploaded papers!"}
        ]
        st.rerun()


# ============================================================
# UPLOAD PAPERS PAGE
# ============================================================
elif page == "📄 Upload Papers":
    st.markdown("### 📄 Upload Research Papers")
    st.markdown("Support: **PDF** and **TXT** files | Max size: 200MB per file")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        files = st.file_uploader(
            "Choose PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📋 Supported Formats</h4>
            <p>✅ PDF (selectable text)<br>
            ✅ TXT (plain text)<br>
            ✅ Maximum 200MB per file</p>
        </div>
        """, unsafe_allow_html=True)
    
    if files:
        st.info(f"📁 **{len(files)} file(s) selected**")
        
        if st.button("📥 **Ingest Papers**", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            success_count = 0
            
            for i, file in enumerate(files):
                suffix = f".{file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = Path(tmp.name)
                
                with st.spinner(f"📖 Processing {file.name}..."):
                    chunks = pipeline.ingest(tmp_path)
                    tmp_path.unlink()
                
                if chunks > 0:
                    st.success(f"✅ **{file.name}**: {chunks} chunks ingested")
                    success_count += 1
                else:
                    st.error(f"❌ **{file.name}**: No text extracted")
                
                progress_bar.progress((i + 1) / len(files))
            
            if success_count > 0:
                pipeline.build_index()
                st.success(f"🎉 **{success_count} paper(s) ingested successfully!**")
                st.balloons()
                st.info("👉 Go to the **Chat** tab to ask questions about your papers!")
            else:
                st.error("❌ No files were successfully ingested. Try TXT files or ensure PDFs have selectable text.")
    
    st.divider()
    
    # Show current index status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Chunks", pipeline.paper_count)
    with col2:
        st.metric("📄 Papers Indexed", len([c for c in pipeline._collection.metadatas if c.get('paper_id')]) if pipeline.paper_count > 0 else 0)
    with col3:
        st.metric("💾 Storage", f"{pipeline.paper_count * 4} KB")
    
    # Tips expander
    with st.expander("📌 **Tips for Best Results**", expanded=False):
        st.markdown("""
        - **Use TXT files** for guaranteed text extraction
        - **PDFs must have selectable text** (not scanned images)
        - **Split long papers** into sections for better results
        - **Remove special characters** that may interfere with parsing
        - **Clear the index** if you need to re-upload files
        """)


# ============================================================
# EVALUATION PAGE
# ============================================================
elif page == "🎯 Evaluate Quality":
    st.markdown("### 🎯 RAG Quality Evaluation")
    st.markdown("Evaluate how well the system answers questions against ground truth data.")
    
    if pipeline.gt_count == 0:
        st.warning("⚠️ **No Ground Truth pairs found!**")
        st.info("👉 Go to the **Ground Truth** tab to add question-answer pairs for evaluation.")
        st.stop()
    
    st.info(f"📋 **Ready to evaluate {pipeline.gt_count} ground truth pair(s)**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Evaluation Metrics:**")
        st.markdown("- Faithfulness (35%)")
        st.markdown("- Context Precision (25%)")
    with col2:
        st.markdown("- Context Recall (20%)")
        st.markdown("- Answer Correctness (20%)")
    with col3:
        st.markdown(f"- **Pass Threshold:** {pipeline._settings.gt_eval_threshold:.0%}")
    
    if st.button("▶️ **Run Evaluation Suite**", type="primary", use_container_width=True):
        with st.spinner("🔄 Running evaluation against all ground truth pairs..."):
            results = pipeline.run_eval_suite()
        
        df = pd.DataFrame(results)
        
        # Summary metrics
        st.subheader("📊 Evaluation Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Faithfulness</h3>
                <h2>{df['faithfulness'].mean():.0%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Correctness</h3>
                <h2>{df['answer_correctness'].mean():.0%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Overall</h3>
                <h2>{df['overall_score'].mean():.0%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Pass Rate</h3>
                <h2>{df['passed'].mean():.0%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 **Download Results CSV**",
            csv,
            f"rag_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


# ============================================================
# GROUND TRUTH PAGE
# ============================================================
elif page == "📋 Ground Truth":
    st.markdown("### 📋 Ground Truth Management")
    st.markdown("Add question-answer pairs to evaluate and monitor RAG quality.")
    
    tab1, tab2 = st.tabs(["➕ **Add New Pair**", "📋 **View All Pairs**"])
    
    with tab1:
        with st.form("add_gt_form"):
            st.markdown("#### 📝 Question")
            question = st.text_area(
                "Question",
                placeholder="e.g., What is the dice coefficient achieved in this paper?",
                height=80
            )
            
            st.markdown("#### ✅ Ground Truth Answer")
            answer = st.text_area(
                "Answer",
                placeholder="e.g., 0.9234",
                height=80
            )
            
            st.markdown("#### 🏷️ Tags (optional)")
            tags = st.text_input(
                "Tags (comma-separated)",
                placeholder="segmentation, dice coefficient, metric"
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                submitted = st.form_submit_button("💾 **Save Pair**", type="primary")
            
            if submitted:
                if question and answer:
                    from src.models import ResearchGroundTruth
                    pair = ResearchGroundTruth(
                        question=question.strip(),
                        ground_truth_answer=answer.strip(),
                        domain_tags=[t.strip() for t in tags.split(",") if t.strip()]
                    )
                    pipeline.add_ground_truth(pair)
                    st.success(f"✅ Ground truth pair saved successfully!")
                    st.rerun()
                else:
                    st.error("❌ Please fill in both question and answer.")
    
    with tab2:
        pairs = pipeline.list_ground_truth()
        if not pairs:
            st.info("📭 No ground truth pairs yet. Add some using the form above.")
        else:
            st.markdown(f"**Total Pairs:** {len(pairs)}")
            for pair in pairs:
                with st.expander(f"📝 **{pair.question[:80]}**", expanded=False):
                    st.markdown(f"**Answer:** {pair.ground_truth_answer}")
                    if pair.domain_tags:
                        st.markdown(f"**Tags:** `{', '.join(pair.domain_tags)}`")
                    st.markdown(f"**ID:** `{pair.gt_id}`")
                    st.markdown(f"**Created:** {pair.created_at.strftime('%Y-%m-%d %H:%M') if pair.created_at else 'N/A'}")
                    
                    if st.button("🗑️ Delete", key=f"del_{pair.gt_id}"):
                        pipeline.delete_ground_truth(pair.gt_id)
                        st.success("✅ Deleted successfully!")
                        st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p>🔬 <strong>Research Paper RAG</strong> | AI-Powered Academic Research Assistant</p>
    <p style="font-size: 0.8rem;">Powered by Groq Llama 3.3 70B | FAISS Vector Search | RAG Ground Truth Evaluation</p>
            <p style="font-size: 0.8rem;">Made with 💖 by Md. Maksudul Haque</p>
</div>
""", unsafe_allow_html=True)