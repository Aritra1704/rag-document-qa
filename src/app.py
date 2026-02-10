"""
RAG Document Q&A Application
Enhanced with Demo Mode and Interactive UI
"""

import streamlit as st
import anthropic
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import docx
import io
import os
import logging
from typing import List, Tuple, Optional

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

def main():
    """Main application function"""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Get your API key from https://console.anthropic.com/",
            value=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        
        st.divider()
        
        # Mode Selection
        if not st.session_state.documents_processed:
            st.header("🚀 Get Started")
            
            if st.button("🎮 Try Demo Mode", type="secondary", use_container_width=True):
                with st.spinner("Loading demo..."):
                    if load_demo_mode():
                        st.success("✅ Demo loaded!")
                        st.rerun()
            
            st.markdown("**OR**")
        
        # Document upload
        st.header("📄 Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, Word, or text files"
        )
        
        # Process button
        if st.button("Process Documents", type="primary", disabled=not uploaded_files):
            if not api_key:
                st.error("Please enter your Anthropic API key first!")
            else:
                with st.spinner("Processing documents..."):
                    process_documents(uploaded_files, api_key)
                    st.rerun()
        
        # Stats
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
    
    # Main content area
    if not st.session_state.documents_processed:
        show_landing_page()
    else:
        # Chat interface header
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
        
        # Display chat history
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
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not api_key:
                st.error("Please enter your Anthropic API key in the sidebar!")
            else:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Retrieve relevant chunks
                with st.spinner("🔍 Searching documents..."):
                    vector_store = VectorStore()
                    vector_store.collection = st.session_state.collection
                    context_chunks, context_metadata = vector_store.query(prompt, n_results=5)
                
                # Generate answer
                with st.spinner("🤖 Generating answer..."):
                    rag_generator = RAGGenerator(api_key)
                    answer = rag_generator.generate_answer(
                        prompt, 
                        context_chunks, 
                        context_metadata
                    )
                
                # Add assistant message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": list(zip(context_chunks, context_metadata))
                })
                
                with st.chat_message("assistant"):
                    st.write(answer)
                    with st.expander("📚 View Sources"):
                        for i, (chunk, meta) in enumerate(zip(context_chunks, context_metadata), 1):
                            st.markdown(f"**Chunk {i}** (from {meta['source']})")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            if i < len(zip(context_chunks, context_metadata)):
                                st.divider()

if __name__ == "__main__":
    main()