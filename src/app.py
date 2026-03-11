"""
RAG Document Q&A Application
Enhanced with Demo Mode and Interactive UI
"""

import streamlit as st
import anthropic
from pathlib import Path
import importlib.util
import shutil
import sys
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import docx
import io
import os
import logging
import re
from typing import List, Tuple, Optional, Any
from name_finder import (
    collect_pdf_pages,
    discover_pdf_files,
    run_name_search_progressive,
    summarize_extraction_debug,
)
from name_search_storage import (
    fetch_test_page_storage_verification,
    get_folder_storage_summary,
    insert_page_records,
    open_storage_connection,
    parse_voter_records_from_page,
    replace_page_records,
    search_stored_records,
    update_document_status,
    upsert_document,
    upsert_page,
)
from ollama_rag import (
    create_ollama_collection,
    generate_ollama_answer,
    get_ollama_diagnostics,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .demo-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 20px 0;
    }
    .feature-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .step-number {
        background-color: #1f77b4;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0
if 'file_count' not in st.session_state:
    st.session_state.file_count = 0
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = True
if 'name_search_outcome' not in st.session_state:
    st.session_state.name_search_outcome = None
if 'name_search_show_debug' not in st.session_state:
    st.session_state.name_search_show_debug = False
if 'name_search_quick_debug' not in st.session_state:
    st.session_state.name_search_quick_debug = None
if 'name_search_start_page' not in st.session_state:
    st.session_state.name_search_start_page = 3
if 'name_search_end_page' not in st.session_state:
    st.session_state.name_search_end_page = 3
if 'name_search_enable_ocr_fallback' not in st.session_state:
    st.session_state.name_search_enable_ocr_fallback = True
if 'name_search_test_mode' not in st.session_state:
    st.session_state.name_search_test_mode = True
if 'name_search_test_max_files' not in st.session_state:
    st.session_state.name_search_test_max_files = 1
if 'name_search_test_max_pages_per_file' not in st.session_state:
    st.session_state.name_search_test_max_pages_per_file = 1
if 'name_search_test_selected_file' not in st.session_state:
    st.session_state.name_search_test_selected_file = "(auto: first discovered)"
if 'name_search_test_replace_existing' not in st.session_state:
    st.session_state.name_search_test_replace_existing = True
if 'name_search_ocr_timeout_seconds' not in st.session_state:
    st.session_state.name_search_ocr_timeout_seconds = 20
if 'name_search_overall_timeout_seconds' not in st.session_state:
    st.session_state.name_search_overall_timeout_seconds = 0
if 'name_search_scan_in_progress' not in st.session_state:
    st.session_state.name_search_scan_in_progress = False
if 'name_search_partial_results' not in st.session_state:
    st.session_state.name_search_partial_results = []
if 'name_search_last_stop_reason' not in st.session_state:
    st.session_state.name_search_last_stop_reason = None
if 'name_search_mode' not in st.session_state:
    st.session_state.name_search_mode = "Live Scan Search"
if 'name_search_stop_requested' not in st.session_state:
    st.session_state.name_search_stop_requested = False
if 'name_search_db_path' not in st.session_state:
    st.session_state.name_search_db_path = os.environ.get("DATABASE_URL", "")
if 'name_search_stored_results' not in st.session_state:
    st.session_state.name_search_stored_results = []
if 'name_search_partial_structured_results' not in st.session_state:
    st.session_state.name_search_partial_structured_results = []
if 'name_search_storage_summary' not in st.session_state:
    st.session_state.name_search_storage_summary = None
if 'name_search_last_action' not in st.session_state:
    st.session_state.name_search_last_action = None
if 'name_search_test_run_summary' not in st.session_state:
    st.session_state.name_search_test_run_summary = None
if 'name_search_test_verification_payload' not in st.session_state:
    st.session_state.name_search_test_verification_payload = None
if 'ollama_collection' not in st.session_state:
    st.session_state.ollama_collection = None
if 'ollama_documents_processed' not in st.session_state:
    st.session_state.ollama_documents_processed = False
if 'ollama_chat_history' not in st.session_state:
    st.session_state.ollama_chat_history = []
if 'ollama_chunk_count' not in st.session_state:
    st.session_state.ollama_chunk_count = 0
if 'ollama_file_count' not in st.session_state:
    st.session_state.ollama_file_count = 0
if 'ollama_ingestion_warnings' not in st.session_state:
    st.session_state.ollama_ingestion_warnings = []
if 'ollama_diagnostics' not in st.session_state:
    st.session_state.ollama_diagnostics = None
if 'ollama_base_url' not in st.session_state:
    st.session_state.ollama_base_url = "http://localhost:11434"
if 'ollama_embedding_model' not in st.session_state:
    st.session_state.ollama_embedding_model = "nomic-embed-text:latest"
if 'ollama_chat_model' not in st.session_state:
    st.session_state.ollama_chat_model = "qwen2.5:7b-instruct"

# Demo document content
DEMO_DOCUMENT = """
Machine Learning Fundamentals

What is Machine Learning?
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Types of Machine Learning:

1. Supervised Learning
Supervised learning uses labeled training data to learn the mapping between input and output. Common applications include image classification, spam detection, and price prediction. Popular algorithms include Linear Regression, Decision Trees, and Neural Networks.

2. Unsupervised Learning  
Unsupervised learning works with unlabeled data to find hidden patterns. It includes techniques like clustering (K-Means) and dimensionality reduction (PCA). Use cases include customer segmentation and anomaly detection.

3. Reinforcement Learning
Reinforcement learning involves an agent learning through trial and error by receiving rewards or penalties. It's used in robotics, game playing (like AlphaGo), and autonomous vehicles.

Deep Learning
Deep learning uses neural networks with multiple layers to automatically learn hierarchical representations of data. It has revolutionized computer vision, natural language processing, and speech recognition.

Key Components:
- Neural Networks: Input layer, hidden layers, output layer
- Activation Functions: ReLU, Sigmoid, Tanh
- Optimization: Gradient descent, Adam, RMSprop
- Regularization: Dropout, Batch Normalization

Real-World Applications:
- Healthcare: Disease diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- E-commerce: Product recommendations, price optimization
- Transportation: Autonomous vehicles, route optimization
- Manufacturing: Quality control, predictive maintenance

Best Practices:
1. Start with high-quality, representative data
2. Split data into training, validation, and test sets
3. Use cross-validation to assess model performance
4. Apply regularization to prevent overfitting
5. Monitor model performance in production
6. Consider ethical implications and bias

Challenges:
- Data quality and quantity requirements
- Computational resource demands
- Model interpretability
- Bias and fairness concerns
- Keeping models updated with new data
"""

class DocumentProcessor:
    """Handles document extraction and processing"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from Word document"""
        try:
            doc = docx.Document(io.BytesIO(file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX: {e}")
            raise
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from text file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting TXT: {e}")
            raise
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > chunk_size * 0.5:
                    end = start + boundary + 1
                    chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        return chunks

class VectorStore:
    """Handles vector database operations"""
    
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = None
    
    def initialize_collection(self, collection_name: str = "documents"):
        """Initialize or reset the vector collection"""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=collection_name)
            except:
                pass
            
            # Create new collection
            default_ef = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=default_ef
            )
            logger.info(f"Initialized collection: {collection_name}")
            return self.collection
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_documents(self, chunks: List[str], metadatas: List[dict], ids: List[str]):
        """Add documents to the vector store"""
        try:
            if chunks:
                self.collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query(self, query_text: str, n_results: int = 5) -> Tuple[List[str], List[dict]]:
        """Query the vector store for relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results['documents'][0], results['metadatas'][0]
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise

class RAGGenerator:
    """Handles answer generation using Claude"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_answer(self, query: str, context_chunks: List[str], 
                       context_metadata: List[dict]) -> str:
        """Generate answer using Claude with retrieved context"""
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {meta['source']}]\n{chunk}" 
            for chunk, meta in zip(context_chunks, context_metadata)
        ])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Cite which source document(s) you're using for your answer
- Be specific and accurate

Answer:"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

def load_demo_mode():
    """Load demo document into the system"""
    try:
        # Initialize vector store
        vector_store = VectorStore()
        collection = vector_store.initialize_collection("demo")
        
        # Process demo document
        doc_processor = DocumentProcessor()
        chunks = doc_processor.chunk_text(DEMO_DOCUMENT)
        
        # Add to vector store
        metadatas = [{"source": "Machine_Learning_Guide.txt"} for _ in chunks]
        ids = [f"demo_chunk_{i}" for i in range(len(chunks))]
        vector_store.add_documents(chunks, metadatas, ids)
        
        # Update session state
        st.session_state.collection = collection
        st.session_state.documents_processed = True
        st.session_state.demo_mode = True
        st.session_state.chunk_count = len(chunks)
        st.session_state.file_count = 1
        st.session_state.chat_history = []
        
        return True
    except Exception as e:
        logger.error(f"Error loading demo: {e}")
        return False

def process_documents(uploaded_files, api_key: str):
    """Process uploaded documents and store in vector database"""
    try:
        # Initialize vector store
        vector_store = VectorStore()
        collection = vector_store.initialize_collection()
        
        doc_processor = DocumentProcessor()
        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_id = 0
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            # Extract text based on file type
            if file.name.endswith('.pdf'):
                text = doc_processor.extract_text_from_pdf(file)
            elif file.name.endswith('.docx'):
                text = doc_processor.extract_text_from_docx(file)
            elif file.name.endswith('.txt'):
                text = doc_processor.extract_text_from_txt(file)
            else:
                continue
            
            # Chunk the text
            chunks = doc_processor.chunk_text(text)
            
            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append(chunk)
                    all_metadatas.append({"source": file.name})
                    all_ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Add all chunks to collection
        if all_chunks:
            vector_store.add_documents(all_chunks, all_metadatas, all_ids)
            
            st.session_state.collection = collection
            st.session_state.documents_processed = True
            st.session_state.demo_mode = False
            st.session_state.chunk_count = len(all_chunks)
            st.session_state.file_count = len(uploaded_files)
            st.session_state.chat_history = []
            
            status_text.text("")
            progress_bar.empty()
            
            st.sidebar.success(
                f"✅ Processed {len(uploaded_files)} files into {len(all_chunks)} chunks!"
            )
            logger.info(f"Successfully processed {len(uploaded_files)} files")
        else:
            st.sidebar.error("No text content found in uploaded files.")
            logger.warning("No text content extracted from files")
            
    except Exception as e:
        st.sidebar.error(f"Error processing documents: {str(e)}")
        logger.error(f"Document processing error: {e}")

def show_landing_page():
    """Display landing page with demo and tutorial"""
    
    st.markdown('<div class="main-header">🤖 RAG Document Q&A System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your documents using AI-powered Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
    
    # Demo Mode Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎮 Try Demo Mode", type="primary", use_container_width=True):
            with st.spinner("Loading demo document..."):
                if load_demo_mode():
                    st.success("✅ Demo loaded! Try asking questions below.")
                    st.rerun()
    
    st.markdown("---")
    
    # What is RAG?
    with st.expander("🧠 What is RAG (Retrieval-Augmented Generation)?", expanded=True):
        st.markdown("""
        <div class="feature-card">
        <h3>The Problem</h3>
        Regular AI chatbots can only answer based on their training data. They don't have access to YOUR documents and may "hallucinate" (make up) answers when they don't know something.
        </div>
        
        <div class="feature-card">
        <h3>The Solution: RAG</h3>
        RAG combines document retrieval with AI generation to give accurate, grounded answers from YOUR documents.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📄 1. Upload Documents**")
            st.write("Add your PDFs, Word docs, or text files")
        
        with col2:
            st.markdown("**🔍 2. Smart Search**")
            st.write("System finds relevant sections automatically")
        
        with col3:
            st.markdown("**💬 3. AI Answers**")
            st.write("Get accurate answers with source citations")
    
    # How It Works
    with st.expander("⚙️ How Does It Work?"):
        st.markdown("""
        <div class="demo-box">
        <h3>The RAG Pipeline</h3>
        """, unsafe_allow_html=True)
        
        steps = [
            ("Document Processing", "Your documents are split into smaller chunks"),
            ("Embeddings", "Text is converted to vectors (numbers that capture meaning)"),
            ("Vector Storage", "Stored in a database for fast semantic search"),
            ("Query Processing", "Your question is also converted to a vector"),
            ("Retrieval", "System finds the most relevant document chunks"),
            ("Generation", "AI reads the relevant chunks and generates an answer"),
            ("Citation", "Shows which documents were used")
        ]
        
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 10px 0;">
                <span class="step-number">{i}</span>
                <div>
                    <strong>{title}</strong><br>
                    <span style="color: #666;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Features
    with st.expander("✨ Key Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - ✅ **Multi-format Support**: PDF, Word, TXT
            - ✅ **Semantic Search**: Understands meaning, not just keywords
            - ✅ **Source Citations**: See which documents were used
            - ✅ **Chat Interface**: Natural conversation flow
            """)
        
        with col2:
            st.markdown("""
            - ✅ **No Hallucinations**: Answers based on your documents
            - ✅ **Privacy-First**: Documents processed locally
            - ✅ **Fast & Accurate**: Get answers in seconds
            - ✅ **Easy to Use**: No technical knowledge required
            """)
    
    # Use Cases
    with st.expander("🎯 Use Cases"):
        use_cases = [
            ("📚 Research", "Query research papers, academic articles, and study materials"),
            ("💼 Business", "Search through reports, policies, and documentation"),
            ("⚖️ Legal", "Find specific clauses in contracts and legal documents"),
            ("🏥 Healthcare", "Access medical guidelines and research"),
            ("📖 Education", "Study materials and course content"),
            ("🔧 Technical", "API docs, manuals, and technical specifications")
        ]
        
        cols = st.columns(3)
        for i, (title, desc) in enumerate(use_cases):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="feature-card">
                <h4>{title}</h4>
                <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Demo Examples
    st.markdown("---")
    st.markdown("### 🎮 Try Demo Mode to Ask Questions Like:")
    
    demo_questions = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "What is deep learning?",
        "What are some real-world applications?",
        "What are the key challenges in machine learning?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(demo_questions):
        with cols[i % 2]:
            st.info(f"❓ {question}")
    
    st.markdown("---")
    st.markdown("### 📤 Or Upload Your Own Documents in the Sidebar")


def _highlight_name_text(value: str, name_query: str) -> str:
    if not value:
        return ""
    query = (name_query or "").strip()
    if not query:
        return value
    pattern = re.compile(re.escape(query), flags=re.IGNORECASE)
    return pattern.sub(lambda match: f"**{match.group(0)}**", value)


def _record_matches_name_query(record: dict[str, Any], name_query: str) -> bool:
    normalized_query = (name_query or "").strip().lower()
    if not normalized_query:
        return False
    name_value = str(record.get("name") or "").lower()
    raw_value = str(record.get("raw_record_text") or "").lower()
    return normalized_query in name_value or normalized_query in raw_value


def _render_structured_records(records: List[dict[str, Any]], searched_name: str, title: str) -> None:
    st.markdown(f"### {title}")
    if not records:
        st.info("No structured records found.")
        return

    rows = []
    for record in records:
        rows.append(
            {
                "name": record.get("name") or "",
                "elector_id": record.get("elector_id") or "",
                "serial_number": record.get("serial_number") or "",
                "relative_name": record.get("relative_name") or "",
                "relative_type": record.get("relative_type") or "",
                "house_number": record.get("house_number") or "",
                "age": record.get("age"),
                "gender": record.get("gender") or "",
                "constituency": record.get("constituency") or "",
                "section_name": record.get("section_name") or "",
                "file_name": record.get("file_name") or "",
                "file_path": record.get("file_path") or "",
                "page_number": record.get("page_number") or 0,
                "match_type": record.get("extraction_method") or "",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    capped_records = records[:200]
    if len(records) > len(capped_records):
        st.caption(f"Showing detailed cards for first {len(capped_records)} of {len(records)} records.")

    for index, record in enumerate(capped_records, start=1):
        name_value = str(record.get("name") or "")
        highlighted_name = _highlight_name_text(name_value, searched_name)
        header_name = name_value if name_value else "Unnamed record"
        file_name = str(record.get("file_name") or "")
        page_number = int(record.get("page_number") or 0)
        match_type = str(record.get("extraction_method") or "")
        with st.expander(f"{index}. {header_name} | {file_name} | page {page_number} | {match_type}"):
            if highlighted_name:
                st.markdown(f"**Name:** {highlighted_name}")
            else:
                st.markdown("**Name:** (not parsed)")

            st.markdown(
                "\n".join(
                    [
                        f"- Elector ID: `{record.get('elector_id') or ''}`",
                        f"- Serial Number: `{record.get('serial_number') or ''}`",
                        f"- Relative: `{record.get('relative_name') or ''}` ({record.get('relative_type') or 'unknown'})",
                        f"- House Number: `{record.get('house_number') or ''}`",
                        f"- Age / Gender: `{record.get('age') or ''}` / `{record.get('gender') or ''}`",
                        f"- Constituency: `{record.get('constituency') or ''}`",
                        f"- Section: `{record.get('section_name') or ''}`",
                        f"- File: `{record.get('file_name') or ''}`",
                        f"- Path: `{record.get('file_path') or ''}`",
                        f"- Page: `{record.get('page_number') or ''}`",
                        f"- Match Type: `{record.get('extraction_method') or ''}`",
                    ]
                )
            )
            with st.expander("Raw OCR/Text block"):
                st.code(str(record.get("raw_record_text") or ""), language="text")


def show_name_search_workflow():
    """Render folder-based PDF name verification workflow."""

    st.markdown('<div class="main-header">🔎 PDF Name Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Live scan PDFs, parse structured OCR records, and search stored local data.</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Mode",
        ["Live Scan Search", "Search Stored Data"],
        horizontal=True,
        key="name_search_mode",
    )

    folder_path = st.text_input(
        "Folder Path",
        value=st.session_state.get("name_search_folder_path", ""),
        placeholder="/Users/aritra/Documents/pdfs",
        help="Local folder path to scan recursively for PDF files",
    )
    db_path = st.text_input(
        "PostgreSQL DATABASE_URL (optional override)",
        value=st.session_state.get("name_search_db_path", os.environ.get("DATABASE_URL", "")),
        help="If empty, DATABASE_URL / PG* environment variables are used",
    )
    name_input = st.text_input(
        "Name to Search",
        value=st.session_state.get("name_search_name", ""),
        placeholder="John Smith",
        help="Exact name to search for (case-insensitive)",
    )

    def _format_match_block(match) -> str:
        return (
            f"Name: {match.searched_name}\n"
            f"File: {match.file_name}\n"
            f"Path: {match.file_path}\n"
            f"Page: {match.page_number}\n"
            f"Position: {match.match_position}\n"
            f"Match Type: {match.match_type}\n"
            f'Snippet: "{match.snippet}"\n'
            "--------------------------------------------------"
        )

    def _query_folder_status_counts(connection, resolved_folder: str) -> dict[str, dict[str, int]]:
        document_rows = connection.execute(
            """
            SELECT status, COUNT(*) AS c
            FROM documents
            WHERE folder_path = %s
            GROUP BY status
            ORDER BY status
            """,
            (resolved_folder,),
        ).fetchall()
        page_rows = connection.execute(
            """
            SELECT p.status AS status, COUNT(*) AS c
            FROM pages p
            JOIN documents d ON d.id = p.document_id
            WHERE d.folder_path = %s
            GROUP BY p.status
            ORDER BY p.status
            """,
            (resolved_folder,),
        ).fetchall()
        return {
            "documents": {str(row["status"]): int(row["c"]) for row in document_rows},
            "pages": {str(row["status"]): int(row["c"]) for row in page_rows},
        }

    resolved_folder_path = str(Path(folder_path).expanduser().resolve()) if folder_path.strip() else ""

    if mode == "Search Stored Data":
        if st.button("Search Stored Data", type="primary"):
            st.session_state.name_search_folder_path = folder_path
            st.session_state.name_search_name = name_input
            st.session_state.name_search_db_path = db_path

            if not folder_path.strip():
                st.warning("Please provide a folder path.")
            elif not name_input.strip():
                st.warning("Please provide a name to search.")
            else:
                try:
                    connection = open_storage_connection(db_path.strip() or None)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to open storage DB: {exc}")
                    st.session_state.name_search_stored_results = []
                    st.session_state.name_search_storage_summary = None
                else:
                    try:
                        summary = get_folder_storage_summary(connection, resolved_folder_path)
                        status_counts = _query_folder_status_counts(connection, resolved_folder_path)
                        records = search_stored_records(
                            connection,
                            folder_path=resolved_folder_path,
                            name_query=name_input,
                            limit=1000,
                        )
                    finally:
                        connection.close()

                    st.session_state.name_search_stored_results = records
                    st.session_state.name_search_storage_summary = {
                        "counts": summary,
                        "status": status_counts,
                    }

        storage_summary_payload = st.session_state.get("name_search_storage_summary")
        if storage_summary_payload:
            counts = storage_summary_payload.get("counts", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Stored Documents", int(counts.get("documents", 0)))
            col2.metric("Stored Pages", int(counts.get("pages", 0)))
            col3.metric("Stored Parsed Records", int(counts.get("records", 0)))

            status_payload = storage_summary_payload.get("status", {})
            document_status = status_payload.get("documents", {})
            page_status = status_payload.get("pages", {})
            if document_status:
                st.caption(
                    "Document status: "
                    + ", ".join(f"{status}={count}" for status, count in document_status.items())
                )
            if page_status:
                st.caption(
                    "Page status: "
                    + ", ".join(f"{status}={count}" for status, count in page_status.items())
                )

        stored_records = st.session_state.get("name_search_stored_results", [])
        if name_input.strip():
            _render_structured_records(
                stored_records,
                searched_name=name_input.strip(),
                title="Structured Stored Records",
            )
        else:
            st.info("Provide a name and click `Search Stored Data`.")
        return

    test_mode = st.checkbox(
        "Test mode",
        value=bool(st.session_state.get("name_search_test_mode", True)),
        help="Run a small controlled sample (for example 1 file x 1 page) to validate extraction, OCR, parsing, and DB insert.",
    )
    discovered_test_files: list[Path] = []
    if folder_path.strip():
        resolved_for_test = Path(folder_path).expanduser()
        if resolved_for_test.exists() and resolved_for_test.is_dir():
            try:
                discovered_test_files = discover_pdf_files(resolved_for_test)
            except Exception:  # noqa: BLE001
                discovered_test_files = []

    sample_col1, sample_col2 = st.columns(2)
    start_page = sample_col1.number_input(
        "Start page",
        min_value=1,
        value=int(st.session_state.get("name_search_start_page", 3)),
        step=1,
        help="For each PDF, start scanning at this page number.",
    )
    end_page_default = int(st.session_state.get("name_search_end_page", 3))
    end_page_input = sample_col2.number_input(
        "End page (optional, 0 = no end page)",
        min_value=0,
        value=end_page_default,
        step=1,
        help="Use 0 for no upper page limit.",
    )
    end_page = int(end_page_input) if int(end_page_input) > 0 else None

    test_max_files = int(st.session_state.get("name_search_test_max_files", 1))
    test_max_pages_per_file = int(st.session_state.get("name_search_test_max_pages_per_file", 1))
    selected_test_file = str(st.session_state.get("name_search_test_selected_file", "(auto: first discovered)"))
    test_replace_existing = bool(st.session_state.get("name_search_test_replace_existing", True))

    if test_mode:
        test_col1, test_col2 = st.columns(2)
        test_max_files = int(
            test_col1.number_input(
                "Max files to process in test mode",
                min_value=1,
                value=int(st.session_state.get("name_search_test_max_files", 1)),
                step=1,
            )
        )
        test_max_pages_per_file = int(
            test_col2.number_input(
                "Max pages per file in test mode",
                min_value=1,
                value=int(st.session_state.get("name_search_test_max_pages_per_file", 1)),
                step=1,
            )
        )

        test_file_options = ["(auto: first discovered)"] + [str(path) for path in discovered_test_files]
        if selected_test_file not in test_file_options:
            selected_test_file = test_file_options[0]
        selected_test_file = st.selectbox(
            "Test file",
            options=test_file_options,
            index=test_file_options.index(selected_test_file),
            key="name_search_test_selected_file",
            help="Choose a specific PDF for testing. Leave auto to use the first discovered PDF.",
        )
        test_replace_existing = st.checkbox(
            "Replace existing records for selected test page",
            value=bool(st.session_state.get("name_search_test_replace_existing", True)),
            key="name_search_test_replace_existing",
            help="When enabled, existing records for the same document+page are replaced before test inserts.",
        )
        if discovered_test_files:
            st.caption(f"Discovered PDFs for test mode: {len(discovered_test_files)}")
        else:
            st.caption("No PDFs discovered yet for test mode selection.")

    enable_ocr_fallback = st.checkbox(
        "Enable OCR fallback",
        value=bool(st.session_state.get("name_search_enable_ocr_fallback", True)),
        help="Use OCR only when all standard extractors return empty/whitespace text.",
    )
    ocr_timeout_seconds = st.number_input(
        "OCR timeout per page (seconds)",
        min_value=1,
        value=int(st.session_state.get("name_search_ocr_timeout_seconds", 20)),
        step=1,
        help="Maximum OCR time per page when OCR fallback is used.",
        disabled=not enable_ocr_fallback,
    )
    overall_timeout_seconds = st.number_input(
        "Overall timeout (seconds, 0 = no timeout)",
        min_value=0,
        value=int(st.session_state.get("name_search_overall_timeout_seconds", 0)),
        step=5,
        help="Optional total runtime limit for the full scan across all files.",
    )
    st.checkbox(
        "Show extraction debug details",
        key="name_search_show_debug",
        help="Show file/page-level extractor attempts to debug text extraction issues",
    )

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    search_clicked = action_col1.button("Search PDFs", type="primary")
    process_store_clicked = action_col2.button("Process & Store")
    run_one_page_test_clicked = action_col3.button("Run 1-page DB test")
    stop_clicked = action_col4.button("Stop Scan")

    if stop_clicked:
        st.session_state.name_search_stop_requested = True
        st.session_state.name_search_last_stop_reason = "stopped by user"
        if st.session_state.get("name_search_scan_in_progress", False):
            st.warning("Stop requested. Attempting to stop the current scan and preserve partial results.")
        else:
            st.info("No scan is currently running.")
        st.rerun()

    if st.session_state.get("name_search_scan_in_progress", False):
        if st.session_state.get("name_search_stop_requested", False):
            st.warning("The previous scan was stopped by user. Showing saved partial results.")
            st.session_state.name_search_last_stop_reason = "stopped by user"
        else:
            st.warning("The previous scan appears interrupted by rerun/user action. Showing saved partial results.")
            st.session_state.name_search_last_stop_reason = "interrupted by rerun/user action"
        st.session_state.name_search_scan_in_progress = False
        st.session_state.name_search_stop_requested = False

    action_mode = None
    if run_one_page_test_clicked:
        action_mode = "test_one_page"
    elif search_clicked:
        action_mode = "search"
    elif process_store_clicked:
        action_mode = "process_store"

    if action_mode:
        st.session_state.name_search_folder_path = folder_path
        st.session_state.name_search_name = name_input
        st.session_state.name_search_db_path = db_path
        st.session_state.name_search_start_page = int(start_page)
        st.session_state.name_search_end_page = int(end_page_input)
        st.session_state.name_search_enable_ocr_fallback = bool(enable_ocr_fallback)
        st.session_state.name_search_test_mode = bool(test_mode)
        st.session_state.name_search_test_max_files = int(test_max_files)
        st.session_state.name_search_test_max_pages_per_file = int(test_max_pages_per_file)
        st.session_state.name_search_test_selected_file = str(selected_test_file)
        st.session_state.name_search_test_replace_existing = bool(test_replace_existing)
        st.session_state.name_search_ocr_timeout_seconds = int(ocr_timeout_seconds)
        st.session_state.name_search_overall_timeout_seconds = int(overall_timeout_seconds)
        st.session_state.name_search_last_action = action_mode

        use_test_sampling = bool(test_mode) or action_mode == "test_one_page"
        if action_mode == "test_one_page":
            use_test_sampling = True
            test_max_files = 1
            test_max_pages_per_file = 1
            end_page = int(start_page)

        if not folder_path.strip():
            st.session_state.name_search_outcome = None
            st.warning("Please provide a folder path.")
        elif action_mode == "search" and not name_input.strip():
            st.session_state.name_search_outcome = None
            st.warning("Please provide a name to search.")
        elif end_page is not None and int(end_page) < int(start_page):
            st.session_state.name_search_outcome = None
            st.warning("End page must be greater than or equal to start page.")
        else:
            resolved_folder = Path(folder_path).expanduser()
            if not resolved_folder.exists() or not resolved_folder.is_dir():
                st.session_state.name_search_outcome = None
                st.error(f"Folder path does not exist or is not a directory: {resolved_folder}")
            else:
                storage_connection = None
                try:
                    st.session_state.name_search_outcome = None
                    st.session_state.name_search_partial_results = []
                    st.session_state.name_search_partial_structured_results = []
                    st.session_state.name_search_stored_results = []
                    st.session_state.name_search_storage_summary = None
                    st.session_state.name_search_test_run_summary = None
                    st.session_state.name_search_test_verification_payload = None
                    st.session_state.name_search_scan_in_progress = True
                    st.session_state.name_search_last_stop_reason = "running"
                    st.session_state.name_search_stop_requested = False

                    selected_pdf_overrides: list[str] = []
                    if use_test_sampling:
                        if selected_test_file and selected_test_file != "(auto: first discovered)":
                            selected_pdf_overrides = [selected_test_file]
                        elif discovered_test_files:
                            selected_pdf_overrides = [str(discovered_test_files[0])]

                    test_run_stats: dict[str, Any] = {
                        "test_mode_run": bool(use_test_sampling),
                        "file_processed": "",
                        "page_numbers_processed": [],
                        "extraction_methods_used": {},
                        "ocr_used": False,
                        "parsed_records_count": 0,
                        "postgres_insert_success_count": 0,
                        "insert_errors": [],
                    }

                    live_status_placeholder = st.empty()
                    live_metrics_placeholder = st.empty()
                    live_progress_bar = st.progress(0.0)
                    partial_results_container = st.container()
                    partial_structured_container = st.container()

                    try:
                        storage_connection = open_storage_connection(db_path.strip() or None)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Storage disabled for this run (DB open failed): {exc}")
                        storage_connection = None

                    resolved_scan_folder = str(resolved_folder.resolve())
                    if storage_connection is not None:
                        try:
                            if use_test_sampling:
                                if selected_pdf_overrides:
                                    discovered_for_pending = [Path(path) for path in selected_pdf_overrides]
                                else:
                                    discovered_for_pending = list(discovered_test_files)
                                discovered_for_pending = discovered_for_pending[: max(1, int(test_max_files))]
                            else:
                                discovered_for_pending = discover_pdf_files(resolved_scan_folder)
                            for pending_pdf_path in discovered_for_pending:
                                upsert_document(
                                    storage_connection,
                                    folder_path=resolved_scan_folder,
                                    file_name=pending_pdf_path.name,
                                    file_path=str(pending_pdf_path),
                                    pages_total=0,
                                    status="pending",
                                    error_message=None,
                                )
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Failed to pre-mark pending documents in storage: {exc}")

                    document_state: dict[str, dict[str, int]] = {}
                    structured_seen_keys: set[tuple[Any, ...]] = set()

                    def _on_progress(update: dict):
                        total_files = int(update.get("total_files", 0))
                        current_file_index = int(update.get("current_file_index", 0))
                        current_file_name = str(update.get("current_file_name", ""))
                        current_page_number = int(update.get("current_page_number", 0))
                        current_file_total_pages = int(update.get("current_file_total_pages", 0))
                        stage = str(update.get("stage", ""))
                        pages_processed = int(update.get("pages_processed", 0))
                        matches_found = int(update.get("total_matches_found", 0))
                        skipped_pages = int(update.get("skipped_pages", 0))
                        skipped_files = int(update.get("skipped_files", 0))
                        ocr_timeout_pages = int(update.get("ocr_timeout_pages", 0))
                        elapsed_seconds = float(update.get("elapsed_seconds", 0.0))
                        new_matches = list(update.get("new_matches", []))

                        live_status_placeholder.markdown(
                            "\n".join(
                                [
                                    "### Live Scan Status",
                                    f"- File: {current_file_index} / {total_files}",
                                    f"- Current file: `{current_file_name}`",
                                    f"- Page: {current_page_number} / {current_file_total_pages}",
                                    f"- Stage: {stage}",
                                ]
                            )
                        )
                        live_metrics_placeholder.markdown(
                            "\n".join(
                                [
                                    f"- Pages processed: {pages_processed}",
                                    f"- Matches found: {matches_found}",
                                    f"- Elapsed: {elapsed_seconds:.1f}s",
                                    f"- Skipped pages/files: {skipped_pages} / {skipped_files}",
                                    f"- OCR timeout pages: {ocr_timeout_pages}",
                                ]
                            )
                        )

                        progress_fraction = 0.0
                        if total_files > 0:
                            file_fraction = 0.0
                            if current_file_total_pages > 0 and current_page_number > 0:
                                file_fraction = min(current_page_number / current_file_total_pages, 1.0)
                            progress_fraction = min(((max(current_file_index - 1, 0)) + file_fraction) / total_files, 1.0)
                        live_progress_bar.progress(progress_fraction)

                        if new_matches:
                            st.session_state.name_search_partial_results.extend(new_matches)
                            partial_results_container.markdown("### Partial Matches (Live)")
                            for match in new_matches:
                                partial_results_container.code(_format_match_block(match), language="text")

                        partial_structured_records = st.session_state.get("name_search_partial_structured_results", [])
                        if partial_structured_records:
                            partial_structured_container.markdown("### Partial Structured Records (Live)")
                            partial_structured_container.dataframe(
                                [
                                    {
                                        "name": record.get("name") or "",
                                        "elector_id": record.get("elector_id") or "",
                                        "file_name": record.get("file_name") or "",
                                        "page_number": record.get("page_number") or 0,
                                        "match_type": record.get("extraction_method") or "",
                                    }
                                    for record in partial_structured_records
                                ],
                                use_container_width=True,
                                hide_index=True,
                            )

                    def _on_lifecycle(event: dict[str, Any]) -> None:
                        if storage_connection is None:
                            return

                        event_type = str(event.get("event") or "")
                        file_path_value = str(event.get("file_path") or "")
                        file_name_value = str(event.get("file_name") or "")
                        total_pages_value = int(event.get("total_pages") or 0)

                        if event_type == "file_started":
                            document_id = upsert_document(
                                storage_connection,
                                folder_path=resolved_scan_folder,
                                file_name=file_name_value,
                                file_path=file_path_value,
                                pages_total=total_pages_value,
                                status="processing",
                                error_message=None,
                            )
                            document_state[file_path_value] = {
                                "document_id": int(document_id),
                                "pages_processed": 0,
                                "pages_total": total_pages_value,
                            }
                            return

                        if event_type == "page_finished":
                            page_number_value = int(event.get("page_number") or 0)
                            if page_number_value <= 0:
                                return

                            state = document_state.get(file_path_value)
                            if state is None:
                                document_id = upsert_document(
                                    storage_connection,
                                    folder_path=resolved_scan_folder,
                                    file_name=file_name_value,
                                    file_path=file_path_value,
                                    pages_total=total_pages_value,
                                    status="processing",
                                    error_message=None,
                                )
                                state = {"document_id": int(document_id), "pages_processed": 0, "pages_total": total_pages_value}
                                document_state[file_path_value] = state

                            document_id = int(state["document_id"])
                            page_status = str(event.get("status") or "processed")
                            extraction_method = str(event.get("extraction_method") or "")
                            raw_text_value = str(event.get("raw_text") or "")
                            normalized_text_value = str(event.get("text") or "")
                            text_for_storage = raw_text_value if raw_text_value.strip() else normalized_text_value
                            error_message = str(event.get("error_message") or "")

                            parsed_records: list[dict[str, Any]] = []
                            if text_for_storage.strip():
                                parsed_records = parse_voter_records_from_page(
                                    page_text=text_for_storage,
                                    file_name=file_name_value,
                                    file_path=file_path_value,
                                    page_number=page_number_value,
                                    extraction_method=extraction_method or "exact_text",
                                )

                            inserted_count_for_page = 0
                            try:
                                page_id = upsert_page(
                                    storage_connection,
                                    document_id=document_id,
                                    page_number=page_number_value,
                                    status=page_status,
                                    extraction_method=extraction_method or None,
                                    raw_text=text_for_storage,
                                    parsed_record_count=len(parsed_records),
                                    error_message=error_message or None,
                                )
                                if use_test_sampling and not test_replace_existing:
                                    insert_page_records(
                                        storage_connection,
                                        document_id=document_id,
                                        page_id=page_id,
                                        records=parsed_records,
                                        page_number=page_number_value,
                                        extraction_method=extraction_method,
                                    )
                                else:
                                    replace_page_records(
                                        storage_connection,
                                        document_id=document_id,
                                        page_id=page_id,
                                        records=parsed_records,
                                        page_number=page_number_value,
                                        extraction_method=extraction_method,
                                    )
                                inserted_count_for_page = len(parsed_records)
                            except Exception as exc:  # noqa: BLE001
                                test_run_stats["insert_errors"].append(
                                    f"{file_name_value} page {page_number_value}: {exc}"
                                )
                                inserted_count_for_page = 0

                            state["pages_processed"] = int(state.get("pages_processed", 0)) + 1
                            try:
                                update_document_status(
                                    storage_connection,
                                    document_id=document_id,
                                    status="processing",
                                    pages_processed=int(state["pages_processed"]),
                                    error_message=None,
                                )
                            except Exception as exc:  # noqa: BLE001
                                test_run_stats["insert_errors"].append(
                                    f"document status update failed for {file_name_value}: {exc}"
                                )

                            if use_test_sampling:
                                if not test_run_stats["file_processed"]:
                                    test_run_stats["file_processed"] = file_path_value
                                if page_number_value not in test_run_stats["page_numbers_processed"]:
                                    test_run_stats["page_numbers_processed"].append(page_number_value)
                                extraction_method_key = extraction_method or "no_text"
                                current_method_count = int(
                                    test_run_stats["extraction_methods_used"].get(extraction_method_key, 0)
                                )
                                test_run_stats["extraction_methods_used"][extraction_method_key] = current_method_count + 1
                                if bool(event.get("ocr_succeeded")) or extraction_method == "ocr_text":
                                    test_run_stats["ocr_used"] = True
                                test_run_stats["parsed_records_count"] = int(test_run_stats["parsed_records_count"]) + len(
                                    parsed_records
                                )
                                test_run_stats["postgres_insert_success_count"] = int(
                                    test_run_stats["postgres_insert_success_count"]
                                ) + inserted_count_for_page

                            searched_name_value = name_input.strip()
                            if searched_name_value:
                                for parsed_record in parsed_records:
                                    if not _record_matches_name_query(parsed_record, searched_name_value):
                                        continue
                                    dedupe_key = (
                                        parsed_record.get("file_path"),
                                        parsed_record.get("page_number"),
                                        parsed_record.get("serial_number"),
                                        parsed_record.get("elector_id"),
                                        parsed_record.get("name"),
                                        parsed_record.get("raw_record_text"),
                                    )
                                    if dedupe_key in structured_seen_keys:
                                        continue
                                    structured_seen_keys.add(dedupe_key)
                                    st.session_state.name_search_partial_structured_results.append(parsed_record)
                            return

                        if event_type == "page_started":
                            page_number_value = int(event.get("page_number") or 0)
                            if page_number_value <= 0:
                                return

                            state = document_state.get(file_path_value)
                            if state is None:
                                document_id = upsert_document(
                                    storage_connection,
                                    folder_path=resolved_scan_folder,
                                    file_name=file_name_value,
                                    file_path=file_path_value,
                                    pages_total=total_pages_value,
                                    status="processing",
                                    error_message=None,
                                )
                                state = {"document_id": int(document_id), "pages_processed": 0, "pages_total": total_pages_value}
                                document_state[file_path_value] = state

                            upsert_page(
                                storage_connection,
                                document_id=int(state["document_id"]),
                                page_number=page_number_value,
                                status="processing",
                                extraction_method=None,
                                raw_text="",
                                parsed_record_count=0,
                                error_message=None,
                            )
                            return

                        if event_type == "file_finished":
                            state = document_state.get(file_path_value)
                            if state is None:
                                document_id = upsert_document(
                                    storage_connection,
                                    folder_path=resolved_scan_folder,
                                    file_name=file_name_value,
                                    file_path=file_path_value,
                                    pages_total=total_pages_value,
                                    status="processing",
                                    error_message=None,
                                )
                                state = {"document_id": int(document_id), "pages_processed": 0, "pages_total": total_pages_value}
                                document_state[file_path_value] = state

                            update_document_status(
                                storage_connection,
                                document_id=int(state["document_id"]),
                                status=str(event.get("status") or "processed"),
                                pages_processed=int(event.get("pages_processed") or state.get("pages_processed", 0)),
                                error_message=str(event.get("error_message") or "") or None,
                            )
                            return

                    query_for_run = name_input.strip()
                    if action_mode in {"process_store", "test_one_page"} and not query_for_run:
                        query_for_run = "__process_and_store_only__"

                    spinner_text = (
                        "Running 1-page PostgreSQL test..."
                        if action_mode == "test_one_page"
                        else (
                            "Processing PDFs and storing parsed records..."
                            if action_mode == "process_store"
                            else "Scanning all PDFs progressively..."
                        )
                    )
                    end_page_for_run = int(end_page) if use_test_sampling and end_page is not None else None
                    pdf_overrides_for_run = selected_pdf_overrides if use_test_sampling and selected_pdf_overrides else None
                    max_files_for_run = int(test_max_files) if use_test_sampling else None
                    max_pages_for_run = int(test_max_pages_per_file) if use_test_sampling else None
                    with st.spinner(spinner_text):
                        outcome = run_name_search_progressive(
                            folder_path=folder_path,
                            raw_names=query_for_run,
                            start_page=int(start_page),
                            end_page=end_page_for_run,
                            enable_ocr_fallback=bool(enable_ocr_fallback),
                            ocr_timeout_per_page=float(ocr_timeout_seconds) if enable_ocr_fallback else None,
                            overall_timeout_seconds=float(overall_timeout_seconds) if overall_timeout_seconds > 0 else None,
                            pdf_files_override=pdf_overrides_for_run,
                            max_files=max_files_for_run,
                            max_pages_per_file=max_pages_for_run,
                            progress_callback=_on_progress,
                            lifecycle_callback=_on_lifecycle,
                            stop_check=lambda: bool(st.session_state.get("name_search_stop_requested", False)),
                        )
                except ValueError as exc:
                    st.session_state.name_search_outcome = None
                    st.error(str(exc))
                except BaseException as exc:  # noqa: BLE001
                    if exc.__class__.__name__ in {"RerunException", "StopException"}:
                        if st.session_state.get("name_search_stop_requested", False):
                            st.session_state.name_search_last_stop_reason = "stopped by user"
                        else:
                            st.session_state.name_search_last_stop_reason = "interrupted by rerun/user action"
                    raise
                else:
                    st.session_state.name_search_outcome = outcome
                    st.session_state.name_search_partial_results = list(outcome.results)
                    st.session_state.name_search_last_stop_reason = outcome.stop_reason

                    if storage_connection is not None:
                        try:
                            summary = get_folder_storage_summary(storage_connection, resolved_scan_folder)
                            status_counts = _query_folder_status_counts(storage_connection, resolved_scan_folder)
                            st.session_state.name_search_storage_summary = {
                                "counts": summary,
                                "status": status_counts,
                            }
                            if name_input.strip():
                                st.session_state.name_search_stored_results = search_stored_records(
                                    storage_connection,
                                    folder_path=resolved_scan_folder,
                                    name_query=name_input.strip(),
                                    limit=1000,
                                )
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Scan completed but failed to load stored records summary: {exc}")
                            test_run_stats["insert_errors"].append(f"summary/refresh query failed: {exc}")
                    elif use_test_sampling:
                        test_run_stats["insert_errors"].append("PostgreSQL connection unavailable for insert.")

                    if use_test_sampling:
                        processed_pages_sorted = sorted(int(page) for page in test_run_stats["page_numbers_processed"])
                        file_processed = str(test_run_stats["file_processed"] or (outcome.pdf_files[0] if outcome.pdf_files else ""))
                        verification_payload: dict[str, Any] | None = None
                        if storage_connection is not None and file_processed and processed_pages_sorted:
                            try:
                                verification_payload = fetch_test_page_storage_verification(
                                    storage_connection,
                                    file_path=file_processed,
                                    page_number=processed_pages_sorted[0],
                                    limit=200,
                                )
                            except Exception as exc:  # noqa: BLE001
                                test_run_stats["insert_errors"].append(f"verification query failed: {exc}")
                                verification_payload = None

                        st.session_state.name_search_test_verification_payload = verification_payload
                        st.session_state.name_search_test_run_summary = {
                            "test_mode_run": True,
                            "test_action": action_mode == "test_one_page",
                            "file_processed": file_processed,
                            "page_numbers_processed": processed_pages_sorted,
                            "selected_start_page": int(start_page),
                            "selected_end_page": int(end_page) if end_page is not None else None,
                            "extraction_methods_used": dict(test_run_stats["extraction_methods_used"]),
                            "ocr_used": bool(test_run_stats["ocr_used"]),
                            "parsed_records_count": int(test_run_stats["parsed_records_count"]),
                            "postgres_insert_success_count": int(test_run_stats["postgres_insert_success_count"]),
                            "insert_errors": list(test_run_stats["insert_errors"]),
                            "replace_existing": bool(test_replace_existing),
                            "selected_test_file": str(selected_test_file),
                        }
                    else:
                        st.session_state.name_search_test_run_summary = None
                        st.session_state.name_search_test_verification_payload = None

                    if outcome.scan_completed:
                        if action_mode == "process_store":
                            st.success(
                                f"Processing complete. PDFs discovered: {len(outcome.pdf_files)} | stop reason: {outcome.stop_reason}"
                            )
                        elif action_mode == "test_one_page":
                            st.success(
                                f"1-page DB test complete. PDFs discovered: {len(outcome.pdf_files)} | stop reason: {outcome.stop_reason}"
                            )
                        else:
                            st.success(
                                f"Scan complete. PDFs discovered: {len(outcome.pdf_files)} | stop reason: {outcome.stop_reason}"
                            )
                    else:
                        stop_reason_value = outcome.stop_reason
                        if st.session_state.get("name_search_stop_requested", False):
                            stop_reason_value = "stopped by user"
                        st.warning(
                            f"Run stopped early. PDFs discovered: {len(outcome.pdf_files)} | stop reason: {stop_reason_value}"
                        )
                finally:
                    st.session_state.name_search_scan_in_progress = False
                    st.session_state.name_search_stop_requested = False
                    if storage_connection is not None:
                        try:
                            storage_connection.close()
                        except Exception:  # noqa: BLE001
                            pass

    outcome = st.session_state.get("name_search_outcome")
    if not outcome:
        partial_results = st.session_state.get("name_search_partial_results", [])
        partial_structured = st.session_state.get("name_search_partial_structured_results", [])
        last_stop_reason = st.session_state.get("name_search_last_stop_reason")
        if partial_results and last_stop_reason in {"interrupted by rerun/user action", "stopped by user"}:
            st.warning(f"Last run stopped early ({last_stop_reason}). Showing preserved partial results.")
            for match in partial_results:
                st.code(_format_match_block(match), language="text")
            if partial_structured:
                _render_structured_records(
                    partial_structured,
                    searched_name=name_input.strip(),
                    title="Structured Partial Records",
                )
            return
        st.info("Enter a folder path and a name, then click `Search PDFs`, `Process & Store`, or `Run 1-page DB test`.")
        return

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PDFs Found", len(outcome.pdf_files))
    with col2:
        st.metric("Total Matches Found", len(outcome.results))

    scan_completed = bool(getattr(outcome, "scan_completed", True))
    stop_reason = str(getattr(outcome, "stop_reason", "completed all files"))
    pages_processed = int(getattr(outcome, "pages_processed", 0))
    skipped_pages_count = int(getattr(outcome, "skipped_pages", 0))
    skipped_files_count = int(getattr(outcome, "skipped_files_count", len(outcome.skipped_files)))
    ocr_timeout_pages = int(getattr(outcome, "ocr_timeout_pages", 0))
    elapsed_seconds = float(getattr(outcome, "elapsed_seconds", 0.0))

    status_prefix = "Run completed" if scan_completed else "Run stopped early"
    if scan_completed:
        st.success(f"{status_prefix}: {stop_reason}")
    else:
        st.warning(f"{status_prefix}: {stop_reason}")
    st.caption(
        f"Pages processed: {pages_processed} | Skipped pages: {skipped_pages_count} | "
        f"Skipped files: {skipped_files_count} | OCR timeout pages: {ocr_timeout_pages} | "
        f"Elapsed: {elapsed_seconds:.1f}s"
    )

    if not outcome.pdf_files:
        st.warning("No PDF files were discovered in the selected folder.")

    if outcome.skipped_files:
        with st.expander("Skipped Files / Warnings"):
            for skipped in outcome.skipped_files:
                st.write(f"- {skipped}")

    test_run_summary = st.session_state.get("name_search_test_run_summary")
    if test_run_summary:
        extraction_methods_used = test_run_summary.get("extraction_methods_used", {})
        extraction_methods_text = ", ".join(
            f"{method}:{count}" for method, count in extraction_methods_used.items()
        ) or "none"
        processed_pages = test_run_summary.get("page_numbers_processed", [])
        processed_pages_text = ", ".join(str(page) for page in processed_pages) if processed_pages else "none"

        st.markdown("### Test-Run Summary")
        summary_rows = [
            {"Metric": "Test-mode run", "Value": "yes" if test_run_summary.get("test_mode_run") else "no"},
            {"Metric": "Processed file", "Value": str(test_run_summary.get("file_processed") or "")},
            {"Metric": "Processed page(s)", "Value": processed_pages_text},
            {"Metric": "Selected start/end", "Value": f"{test_run_summary.get('selected_start_page')} / {test_run_summary.get('selected_end_page')}"},
            {"Metric": "Extraction method(s)", "Value": extraction_methods_text},
            {"Metric": "OCR used", "Value": "yes" if test_run_summary.get("ocr_used") else "no"},
            {"Metric": "Parsed records count", "Value": int(test_run_summary.get("parsed_records_count", 0))},
            {"Metric": "PostgreSQL insert success count", "Value": int(test_run_summary.get("postgres_insert_success_count", 0))},
            {"Metric": "Replace existing enabled", "Value": "yes" if test_run_summary.get("replace_existing") else "no"},
        ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        insert_errors = list(test_run_summary.get("insert_errors", []))
        if insert_errors:
            with st.expander("Test insert errors"):
                for error in insert_errors:
                    st.write(f"- {error}")

        verification_payload = st.session_state.get("name_search_test_verification_payload")
        st.markdown("### PostgreSQL Verification (Canonical Tables)")
        if verification_payload:
            verification_counts = verification_payload.get("counts", {})
            verification_checks = verification_payload.get("checks", {})

            checks_rows = [
                {
                    "Check": "Exactly one documents row (or existing reused row)",
                    "Result": "pass" if verification_checks.get("single_document_row") else "fail",
                    "Observed count": int(verification_counts.get("documents", 0)),
                },
                {
                    "Check": "Exactly one pages row for selected page",
                    "Result": "pass" if verification_checks.get("single_page_row") else "fail",
                    "Observed count": int(verification_counts.get("pages", 0)),
                },
                {
                    "Check": "Multiple parsed_records rows for selected page",
                    "Result": "pass" if verification_checks.get("multiple_parsed_records") else "check data",
                    "Observed count": int(verification_counts.get("parsed_records", 0)),
                },
            ]
            st.dataframe(checks_rows, use_container_width=True, hide_index=True)

            document_rows = verification_payload.get("documents", [])
            page_rows = verification_payload.get("pages", [])
            parsed_record_rows = verification_payload.get("parsed_records", [])

            st.markdown("#### documents row")
            if document_rows:
                st.dataframe(document_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No documents row found for selected file_path.")

            st.markdown("#### pages row")
            if page_rows:
                st.dataframe(page_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No pages row found for selected file_path + page_number.")

            st.markdown("#### parsed_records rows")
            if parsed_record_rows:
                st.dataframe(parsed_record_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No parsed_records rows found for selected test page.")
        else:
            st.info("No stored rows found yet for the selected test page.")

    input_name_value = name_input.strip()
    last_action = st.session_state.get("name_search_last_action")
    if last_action in {"process_store", "test_one_page"} and not input_name_value:
        searched_name = ""
    else:
        searched_name = outcome.names[0] if outcome.names else input_name_value
    deterministic_matches = [
        match for match in outcome.results if match.match_type in {"exact_text", "ocr_text", "exact"}
    ]
    if not searched_name and not deterministic_matches:
        st.info("Processing completed without a live name query. Use `Search Stored Data` for fast lookups.")
    elif not deterministic_matches:
        st.info(f'No matches found for "{searched_name}"')
    else:
        st.markdown("### Exact/Deterministic Match Results")
        for match in deterministic_matches:
            st.code(_format_match_block(match), language="text")

    storage_summary_payload = st.session_state.get("name_search_storage_summary")
    if storage_summary_payload:
        counts = storage_summary_payload.get("counts", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Stored Documents", int(counts.get("documents", 0)))
        col2.metric("Stored Pages", int(counts.get("pages", 0)))
        col3.metric("Stored Parsed Records", int(counts.get("records", 0)))
        status_payload = storage_summary_payload.get("status", {})
        document_status = status_payload.get("documents", {})
        page_status = status_payload.get("pages", {})
        if document_status:
            st.caption(
                "Document status: "
                + ", ".join(f"{status}={count}" for status, count in document_status.items())
            )
        if page_status:
            st.caption(
                "Page status: "
                + ", ".join(f"{status}={count}" for status, count in page_status.items())
            )

    stored_structured_records = st.session_state.get("name_search_stored_results", [])
    if searched_name and stored_structured_records:
        _render_structured_records(
            stored_structured_records,
            searched_name=searched_name,
            title="Structured Parsed Records",
        )

    if st.session_state.get("name_search_show_debug", False):
        debug_entries = outcome.extraction_debug
        st.markdown("### Extraction Debug")
        if not debug_entries:
            st.info("No extraction debug data available for this run.")
            return

        def yes_no(value: bool) -> str:
            return "yes" if value else "no"

        st.markdown("#### Environment Diagnostics")
        env_rows = [
            {"Metric": "sys.executable", "Value": sys.executable},
            {"Metric": "sys.version", "Value": sys.version.replace("\n", " ")},
            {"Metric": "PyPDF2 importable", "Value": yes_no(importlib.util.find_spec("PyPDF2") is not None)},
            {"Metric": "pypdf importable", "Value": yes_no(importlib.util.find_spec("pypdf") is not None)},
            {"Metric": "pdfplumber importable", "Value": yes_no(importlib.util.find_spec("pdfplumber") is not None)},
            {"Metric": "PyMuPDF (fitz) importable", "Value": yes_no(importlib.util.find_spec("fitz") is not None)},
            {"Metric": "pytesseract importable", "Value": yes_no(importlib.util.find_spec("pytesseract") is not None)},
            {"Metric": "Pillow (PIL) importable", "Value": yes_no(importlib.util.find_spec("PIL") is not None)},
            {"Metric": "tesseract command available", "Value": yes_no(shutil.which("tesseract") is not None)},
            {"Metric": "pdftotext command available", "Value": yes_no(shutil.which("pdftotext") is not None)},
        ]
        st.dataframe(env_rows, use_container_width=True, hide_index=True)

        summary = summarize_extraction_debug(debug_entries)
        col1, col2, col3 = st.columns(3)
        col1.metric("PDFs Discovered", int(summary["pdfs_discovered"]))
        col2.metric("PDFs With Text", int(summary["pdfs_with_extracted_text"]))
        col3.metric("PDFs Fully Skipped", int(summary["pdfs_fully_skipped"]))

        col4, col5, col6 = st.columns(3)
        col4.metric("Pages Processed", int(summary["total_pages_processed"]))
        col5.metric("Pages With Text", int(summary["pages_with_text"]))
        col6.metric("Pages With No Text", int(summary["pages_with_no_text"]))

        extractor_success_counts = summary["extractor_success_counts"]
        if extractor_success_counts:
            extractor_counts_text = ", ".join(
                f"{name}: {count}" for name, count in extractor_success_counts.items()
            )
            st.caption(f"Extractor success counts: {extractor_counts_text}")
        else:
            st.caption("Extractor success counts: none")

        st.markdown("#### Quick One-File Diagnostic (First 3 Pages)")
        if outcome.pdf_files:
            quick_pdf_path = st.selectbox(
                "Choose a scanned PDF for quick diagnostics",
                options=outcome.pdf_files,
                key="name_search_quick_pdf",
            )
            if st.button("Run quick diagnostic for selected file", key="run_quick_pdf_diagnostic"):
                quick_enable_ocr = bool(st.session_state.get("name_search_enable_ocr_fallback", True))
                quick_ocr_timeout = int(st.session_state.get("name_search_ocr_timeout_seconds", 20))
                quick_pages, quick_skipped, quick_debug_entries = collect_pdf_pages(
                    [Path(quick_pdf_path)],
                    include_debug=True,
                    max_pages_per_file=3,
                    enable_ocr_fallback=quick_enable_ocr,
                    ocr_timeout_per_page=float(quick_ocr_timeout) if quick_enable_ocr else None,
                )
                st.session_state.name_search_quick_debug = {
                    "pdf_path": quick_pdf_path,
                    "pages_extracted": len(quick_pages),
                    "skipped_messages": quick_skipped,
                    "debug_entries": quick_debug_entries,
                }

            quick_debug_payload = st.session_state.get("name_search_quick_debug")
            if quick_debug_payload and quick_debug_payload.get("pdf_path"):
                st.caption(
                    f"Quick diagnostic file: {quick_debug_payload['pdf_path']} "
                    f"(pages extracted: {quick_debug_payload.get('pages_extracted', 0)})"
                )
                if quick_debug_payload.get("skipped_messages"):
                    st.write("Quick diagnostic warnings:")
                    for warning in quick_debug_payload["skipped_messages"]:
                        st.write(f"- {warning}")

                quick_rows = []
                for quick_file_debug in quick_debug_payload.get("debug_entries", []):
                    for page_debug in quick_file_debug.page_debug:
                        page_ocr_attempted = bool(getattr(page_debug, "ocr_attempted", False))
                        page_ocr_succeeded = bool(getattr(page_debug, "ocr_succeeded", False))
                        page_ocr_character_count = int(getattr(page_debug, "ocr_character_count", 0) or 0)
                        page_ocr_preview = str(getattr(page_debug, "ocr_preview", "") or "")
                        page_ocr_error = str(getattr(page_debug, "ocr_error", "") or "")
                        for attempt in page_debug.attempts:
                            quick_rows.append(
                                {
                                    "file_path": page_debug.file_path,
                                    "page_number": page_debug.page_number,
                                    "extractor_name": attempt.extractor_name,
                                    "import_available": attempt.import_available,
                                    "open_attempted": attempt.open_attempted,
                                    "extraction_attempted": attempt.extraction_attempted,
                                    "success": attempt.succeeded,
                                    "extracted_char_count": attempt.character_count,
                                    "whitespace_only": attempt.whitespace_only,
                                    "preview_text_first_150_chars": attempt.preview,
                                    "error_message": attempt.error or "",
                                    "selected_as_winner": attempt.extractor_name == page_debug.successful_extractor,
                                    "ocr_attempted": page_ocr_attempted,
                                    "ocr_success": page_ocr_succeeded,
                                    "ocr_extracted_char_count": page_ocr_character_count,
                                    "ocr_preview_text_first_150_chars": page_ocr_preview,
                                    "ocr_error_message": page_ocr_error,
                                }
                            )
                if quick_rows:
                    st.dataframe(quick_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No quick page-level attempts available.")

                raw_winner_sections = []
                for quick_file_debug in quick_debug_payload.get("debug_entries", []):
                    for page_debug in quick_file_debug.page_debug:
                        winning_extractor = page_debug.successful_extractor or "none"
                        raw_winning_text = page_debug.winning_raw_text_first_500
                        if not page_debug.successful_extractor:
                            raw_winning_text = "[no winning extractor selected for this page]"
                        elif raw_winning_text == "":
                            raw_winning_text = "[winning extractor returned empty text]"

                        raw_winner_sections.append(
                            "\n".join(
                                [
                                    f"file_path: {page_debug.file_path}",
                                    f"page_number: {page_debug.page_number}",
                                    f"winning_extractor: {winning_extractor}",
                                    "winning_text_first_500_chars:",
                                    raw_winning_text,
                                ]
                            )
                        )

                if raw_winner_sections:
                    st.write("Raw winner text by page (quick diagnostic):")
                    st.code("\n\n".join(raw_winner_sections), language="text")
        else:
            st.caption("No PDFs were discovered, so quick diagnostics are unavailable.")

        max_debug_files = 10
        max_debug_pages_per_file = 50
        st.caption(
            "Detailed page diagnostics are capped to keep this page responsive. "
            f"Showing up to {max_debug_files} files and {max_debug_pages_per_file} pages per file. "
            "Summary metrics above are computed across all scanned files/pages."
        )

        displayed_files = debug_entries[:max_debug_files]
        if len(debug_entries) > max_debug_files:
            st.caption(f"Showing detailed diagnostics for first {max_debug_files} of {len(debug_entries)} files.")

        for file_debug in displayed_files:
            file_status = "skipped" if file_debug.skipped else "extracted"
            with st.expander(f"{file_debug.file_name} ({file_status})"):
                st.text(f"Path: {file_debug.file_path}")
                total_pages = len(file_debug.page_debug)
                successful_pages = sum(1 for page in file_debug.page_debug if not page.skipped)
                failed_pages = total_pages - successful_pages
                st.write(
                    f"Pages processed: {total_pages} | Successful pages: {successful_pages} | "
                    f"Failed pages: {failed_pages}"
                )
                if file_debug.skip_reason:
                    st.warning(f"File status reason: {file_debug.skip_reason}")
                if file_debug.extractor_open_debug:
                    st.write("Extractor open status:")
                    open_rows = []
                    for open_debug in file_debug.extractor_open_debug:
                        open_rows.append(
                            {
                                "Extractor": open_debug.extractor_name,
                                "Import Available": yes_no(open_debug.import_available),
                                "Open Attempted": yes_no(open_debug.open_attempted),
                                "Open Succeeded": yes_no(open_debug.open_succeeded),
                                "Error": open_debug.error or "",
                            }
                        )
                    st.dataframe(open_rows, use_container_width=True, hide_index=True)

                if not file_debug.page_debug:
                    st.caption("No page-level diagnostics available for this file.")
                    continue

                page_rows = []
                for page_debug in file_debug.page_debug[:max_debug_pages_per_file]:
                    successful_extractor = page_debug.successful_extractor or "none"
                    page_ocr_attempted = bool(getattr(page_debug, "ocr_attempted", False))
                    page_ocr_succeeded = bool(getattr(page_debug, "ocr_succeeded", False))
                    page_ocr_character_count = int(getattr(page_debug, "ocr_character_count", 0) or 0)
                    page_ocr_preview = str(getattr(page_debug, "ocr_preview", "") or "")
                    page_ocr_error = str(getattr(page_debug, "ocr_error", "") or "")
                    for attempt in page_debug.attempts:
                        page_rows.append(
                            {
                                "File Path": page_debug.file_path,
                                "Page": page_debug.page_number,
                                "Extractor": attempt.extractor_name,
                                "Import Available": yes_no(attempt.import_available),
                                "Open Attempted": yes_no(attempt.open_attempted),
                                "Extraction Attempted": yes_no(attempt.extraction_attempted),
                                "Success": yes_no(attempt.succeeded),
                                "Characters": attempt.character_count,
                                "Whitespace Only": yes_no(attempt.whitespace_only),
                                "Preview": attempt.preview[:100],
                                "Error": attempt.error or "",
                                "Winning Extractor": successful_extractor,
                                "OCR Attempted": yes_no(page_ocr_attempted),
                                "OCR Success": yes_no(page_ocr_succeeded),
                                "OCR Characters": page_ocr_character_count,
                                "OCR Preview": page_ocr_preview[:100],
                                "OCR Error": page_ocr_error,
                                "Page Status": "skipped" if page_debug.skipped else "extracted",
                            }
                        )

                st.dataframe(page_rows, use_container_width=True, hide_index=True)
                if len(file_debug.page_debug) > max_debug_pages_per_file:
                    st.caption(
                        f"Showing first {max_debug_pages_per_file} of {len(file_debug.page_debug)} pages for this file."
                    )


def _extract_chunks_for_ollama(uploaded_files) -> Tuple[List[str], List[dict], List[str], List[str]]:
    """Extract chunked documents and metadata for local Ollama RAG ingestion."""

    doc_processor = DocumentProcessor()
    chunks: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    warnings: List[str] = []
    chunk_id = 0

    for file in uploaded_files:
        file_bytes = file.getvalue()
        file_name_lower = file.name.lower()

        if file_name_lower.endswith(".pdf"):
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{file.name}: failed to open PDF ({exc})")
                continue

            if getattr(reader, "is_encrypted", False):
                try:
                    unlocked = reader.decrypt("")
                except Exception:  # noqa: BLE001
                    unlocked = 0
                if unlocked == 0:
                    warnings.append(f"{file.name}: password-protected PDF (skipped)")
                    continue

            file_has_text = False
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{file.name} page {page_number}: extraction error ({exc})")
                    page_text = ""

                page_chunks = doc_processor.chunk_text(page_text)
                for chunk in page_chunks:
                    if not chunk.strip():
                        continue
                    file_has_text = True
                    chunks.append(chunk)
                    metadatas.append({"source": file.name, "page": page_number})
                    ids.append(f"ollama_chunk_{chunk_id}")
                    chunk_id += 1

            if not file_has_text:
                warnings.append(f"{file.name}: no extractable text")
            continue

        if file_name_lower.endswith(".docx"):
            try:
                text = doc_processor.extract_text_from_docx(io.BytesIO(file_bytes))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{file.name}: DOCX extraction error ({exc})")
                continue
        elif file_name_lower.endswith(".txt"):
            try:
                text = file_bytes.decode("utf-8")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{file.name}: TXT decode error ({exc})")
                continue
        else:
            warnings.append(f"{file.name}: unsupported file type")
            continue

        text_chunks = doc_processor.chunk_text(text)
        if not text_chunks:
            warnings.append(f"{file.name}: no extractable text")
            continue

        for chunk in text_chunks:
            if not chunk.strip():
                continue
            chunks.append(chunk)
            metadatas.append({"source": file.name, "page": "N/A"})
            ids.append(f"ollama_chunk_{chunk_id}")
            chunk_id += 1

    return chunks, metadatas, ids, warnings


def _render_source_label(metadata: dict) -> str:
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    if page in (None, "", "N/A", 0):
        return source
    return f"{source} (page {page})"


def show_ollama_rag_workflow():
    """Render local Ollama-backed RAG workflow."""

    st.markdown('<div class="main-header">🧠 Local Ollama RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Local semantic retrieval and Q&A. Exact name verification remains deterministic in PDF Name Search.</div>',
        unsafe_allow_html=True,
    )

    base_url = st.text_input(
        "Ollama Base URL",
        value=st.session_state.get("ollama_base_url", "http://localhost:11434"),
        help="Local Ollama endpoint",
    )
    embedding_model = st.text_input(
        "Embedding Model",
        value=st.session_state.get("ollama_embedding_model", "nomic-embed-text:latest"),
        help="Model used for local embeddings",
    )
    chat_model = st.text_input(
        "Chat Model",
        value=st.session_state.get("ollama_chat_model", "qwen2.5:7b-instruct"),
        help="Model used for local answers",
    )

    st.session_state.ollama_base_url = base_url
    st.session_state.ollama_embedding_model = embedding_model
    st.session_state.ollama_chat_model = chat_model

    if st.button("Check Ollama Connectivity", key="check_ollama_connectivity"):
        st.session_state.ollama_diagnostics = get_ollama_diagnostics(
            base_url=base_url,
            embedding_model=embedding_model,
            chat_model=chat_model,
        )

    diagnostics = st.session_state.get("ollama_diagnostics")
    if diagnostics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Endpoint Reachable", "Yes" if diagnostics["endpoint_reachable"] else "No")
        col2.metric("Embedding Model Available", "Yes" if diagnostics["embedding_model_available"] else "No")
        col3.metric("Chat Model Available", "Yes" if diagnostics["chat_model_available"] else "No")
        if diagnostics.get("endpoint_error"):
            st.warning(f"Ollama diagnostics error: {diagnostics['endpoint_error']}")
        model_names = diagnostics.get("model_names", [])
        if model_names:
            st.caption("Available local models: " + ", ".join(model_names))

    uploaded_files = st.file_uploader(
        "Upload files for local RAG (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
        key="ollama_file_uploader",
    )

    if st.button("Process Documents for Local Ollama RAG", type="primary", disabled=not uploaded_files):
        try:
            with st.spinner("Processing documents and building local embeddings..."):
                chunks, metadatas, ids, warnings = _extract_chunks_for_ollama(uploaded_files)
                if not chunks:
                    raise ValueError("No extractable text found in uploaded documents.")

                collection = create_ollama_collection(
                    base_url=base_url,
                    embedding_model=embedding_model,
                    collection_name="ollama_documents",
                )
                collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Local Ollama RAG setup failed: {exc}")
            st.session_state.ollama_documents_processed = False
            st.session_state.ollama_collection = None
            st.session_state.ollama_chat_history = []
            st.session_state.ollama_chunk_count = 0
            st.session_state.ollama_file_count = 0
            st.session_state.ollama_ingestion_warnings = []
        else:
            st.session_state.ollama_collection = collection
            st.session_state.ollama_documents_processed = True
            st.session_state.ollama_chat_history = []
            st.session_state.ollama_chunk_count = len(chunks)
            st.session_state.ollama_file_count = len(uploaded_files)
            st.session_state.ollama_ingestion_warnings = warnings
            st.success(
                f"Processed {len(uploaded_files)} files into {len(chunks)} chunks using local Ollama embeddings."
            )

    if not st.session_state.ollama_documents_processed:
        st.info("Upload files, check Ollama connectivity, then process documents to start local RAG Q&A.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files Loaded", st.session_state.ollama_file_count)
    with col2:
        st.metric("Text Chunks", st.session_state.ollama_chunk_count)

    if st.session_state.ollama_ingestion_warnings:
        with st.expander("Ingestion Warnings"):
            for warning in st.session_state.ollama_ingestion_warnings:
                st.write(f"- {warning}")

    if st.button("Reset Local Ollama RAG", key="reset_ollama_rag"):
        st.session_state.ollama_documents_processed = False
        st.session_state.ollama_collection = None
        st.session_state.ollama_chat_history = []
        st.session_state.ollama_chunk_count = 0
        st.session_state.ollama_file_count = 0
        st.session_state.ollama_ingestion_warnings = []
        st.rerun()

    for message in st.session_state.ollama_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            sources = message.get("sources")
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")

    if query := st.chat_input("Ask a question about your documents (Local Ollama RAG)..."):
        st.session_state.ollama_chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        collection = st.session_state.ollama_collection
        try:
            with st.spinner("Retrieving relevant chunks..."):
                query_result = collection.query(query_texts=[query], n_results=5)
                context_chunks = query_result.get("documents", [[]])[0]
                context_metadata = query_result.get("metadatas", [[]])[0]

            with st.spinner("Generating local Ollama answer..."):
                answer = generate_ollama_answer(
                    base_url=base_url,
                    chat_model=chat_model,
                    query=query,
                    context_chunks=context_chunks,
                    context_metadata=context_metadata,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Local Ollama RAG query failed: {exc}")
            return

        source_labels = [_render_source_label(metadata) for metadata in context_metadata]
        st.session_state.ollama_chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_labels,
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Sources"):
                for source_label in source_labels:
                    st.write(f"- {source_label}")


def main():
    """Main application function."""

    workflow_mode = st.sidebar.radio(
        "Workflow",
        ["Document Q&A", "PDF Name Search", "Local Ollama RAG"],
        index=0,
    )

    if workflow_mode == "PDF Name Search":
        show_name_search_workflow()
        return

    if workflow_mode == "Local Ollama RAG":
        show_ollama_rag_workflow()
        return

    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Get your API key from https://console.anthropic.com/",
            value=os.environ.get("ANTHROPIC_API_KEY", "")
        )

        st.divider()

        if not st.session_state.documents_processed:
            st.header("🚀 Get Started")

            if st.button("🎮 Try Demo Mode", type="secondary", use_container_width=True):
                with st.spinner("Loading demo..."):
                    if load_demo_mode():
                        st.success("✅ Demo loaded!")
                        st.rerun()

            st.markdown("**OR**")

        st.header("📄 Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, Word, or text files"
        )

        if st.button("Process Documents", type="primary", disabled=not uploaded_files):
            if not api_key:
                st.error("Please enter your Anthropic API key first!")
            else:
                with st.spinner("Processing documents..."):
                    process_documents(uploaded_files, api_key)
                    st.rerun()

        if st.session_state.documents_processed:
            st.divider()

            if st.session_state.demo_mode:
                st.info("📌 Demo Mode Active")

            st.metric("Files Loaded", st.session_state.file_count)
            st.metric("Text Chunks", st.session_state.chunk_count)

            if st.button("🔄 Reset", type="secondary"):
                st.session_state.documents_processed = False
                st.session_state.collection = None
                st.session_state.chat_history = []
                st.session_state.chunk_count = 0
                st.session_state.file_count = 0
                st.session_state.demo_mode = False
                st.rerun()

    if not st.session_state.documents_processed:
        show_landing_page()
    else:
        if st.session_state.demo_mode:
            st.success("🎮 **Demo Mode** - Try asking questions about machine learning!")
            with st.expander("💡 Suggested Questions"):
                cols = st.columns(2)
                suggestions = [
                    "What is supervised learning?",
                    "What are the applications of deep learning?",
                    "What are the challenges in machine learning?",
                    "Explain reinforcement learning"
                ]
                for i, suggestion in enumerate(suggestions):
                    with cols[i % 2]:
                        st.code(suggestion)
        else:
            st.success(f"✅ {st.session_state.file_count} documents loaded! Ask questions below.")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, (chunk, meta) in enumerate(message["sources"], 1):
                            st.markdown(f"**Chunk {i}** (from {meta['source']})")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            if i < len(message["sources"]):
                                st.divider()

        if prompt := st.chat_input("Ask a question about your documents..."):
            if not api_key:
                st.error("Please enter your Anthropic API key in the sidebar!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.spinner("🔍 Searching documents..."):
                    vector_store = VectorStore()
                    vector_store.collection = st.session_state.collection
                    context_chunks, context_metadata = vector_store.query(prompt, n_results=5)

                with st.spinner("🤖 Generating answer..."):
                    rag_generator = RAGGenerator(api_key)
                    answer = rag_generator.generate_answer(
                        prompt,
                        context_chunks,
                        context_metadata
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": list(zip(context_chunks, context_metadata))
                })

                with st.chat_message("assistant"):
                    st.write(answer)
                    with st.expander("📚 View Sources"):
                        sources = list(zip(context_chunks, context_metadata))
                        for i, (chunk, meta) in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i}** (from {meta['source']})")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            if i < len(sources):
                                st.divider()

if __name__ == "__main__":
    main()
