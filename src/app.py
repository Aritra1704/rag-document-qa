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
from typing import List, Tuple, Optional
from name_finder import (
    collect_pdf_pages,
    run_name_search,
    summarize_extraction_debug,
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

def show_name_search_workflow():
    """Render folder-based PDF name verification workflow."""

    st.markdown('<div class="main-header">🔎 PDF Name Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Recursively scan local PDFs and verify an exact name page-by-page.</div>',
        unsafe_allow_html=True,
    )

    folder_path = st.text_input(
        "Folder Path",
        value=st.session_state.get("name_search_folder_path", ""),
        placeholder="/Users/aritra/Documents/pdfs",
        help="Local folder path to scan recursively for PDF files",
    )
    name_input = st.text_input(
        "Name to Search",
        value=st.session_state.get("name_search_name", ""),
        placeholder="John Smith",
        help="Exact name to search for (case-insensitive)",
    )
    st.checkbox(
        "Show extraction debug details",
        key="name_search_show_debug",
        help="Show file/page-level extractor attempts to debug text extraction issues",
    )

    if st.button("Search PDFs", type="primary"):
        st.session_state.name_search_folder_path = folder_path
        st.session_state.name_search_name = name_input

        if not folder_path.strip():
            st.session_state.name_search_outcome = None
            st.warning("Please provide a folder path.")
        elif not name_input.strip():
            st.session_state.name_search_outcome = None
            st.warning("Please provide a name to search.")
        else:
            resolved_folder = Path(folder_path).expanduser()
            if not resolved_folder.exists() or not resolved_folder.is_dir():
                st.session_state.name_search_outcome = None
                st.error(f"Folder path does not exist or is not a directory: {resolved_folder}")
            else:
                try:
                    with st.spinner("Scanning PDFs and searching for exact matches..."):
                        outcome = run_name_search(
                            folder_path=folder_path,
                            raw_names=name_input,
                            enable_semantic_fallback=False,
                        )
                except ValueError as exc:
                    st.session_state.name_search_outcome = None
                    st.error(str(exc))
                else:
                    st.session_state.name_search_outcome = outcome
                    st.success(f"Search complete. PDFs discovered: {len(outcome.pdf_files)}")

    outcome = st.session_state.get("name_search_outcome")
    if not outcome:
        st.info("Enter a folder path and a name, then click 'Search PDFs'.")
        return

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PDFs Found", len(outcome.pdf_files))
    with col2:
        st.metric("Total Matches Found", len(outcome.results))

    if not outcome.pdf_files:
        st.warning("No PDF files were discovered in the selected folder.")

    if outcome.skipped_files:
        with st.expander("Skipped Files / Warnings"):
            for skipped in outcome.skipped_files:
                st.write(f"- {skipped}")

    searched_name = outcome.names[0] if outcome.names else name_input.strip()
    exact_matches = [match for match in outcome.results if match.match_type == "exact"]
    if not exact_matches:
        st.info(f'No matches found for "{searched_name}"')
    else:
        for match in exact_matches:
            match_block = (
                f"Name: {match.searched_name}\n"
                f"File: {match.file_name}\n"
                f"Path: {match.file_path}\n"
                f"Page: {match.page_number}\n"
                f"Position: {match.match_position}\n"
                f"Match Type: {match.match_type}\n"
                f'Snippet: "{match.snippet}"\n'
                "--------------------------------------------------"
            )
            st.code(match_block, language="text")

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
                quick_pages, quick_skipped, quick_debug_entries = collect_pdf_pages(
                    [Path(quick_pdf_path)],
                    include_debug=True,
                    max_pages_per_file=3,
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
                        winning = page_debug.successful_extractor or "none"
                        for attempt in page_debug.attempts:
                            quick_rows.append(
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
                                    "Winning Extractor": winning,
                                }
                            )
                if quick_rows:
                    st.dataframe(quick_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No quick page-level attempts available.")
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
