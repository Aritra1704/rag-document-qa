"""
RAG Document Q&A Application
Main Streamlit application file
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
            st.session_state.chunk_count = len(all_chunks)
            st.session_state.file_count = len(uploaded_files)
            
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

def main():
    """Main application function"""
    
    st.title("🤖 RAG Document Q&A System")
    
    st.markdown("""
    Ask questions about your documents using **Retrieval-Augmented Generation (RAG)**.
    Upload PDFs, Word docs, or text files to get started.
    """)
    
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
        
        # Document upload
        st.header("📄 Upload Documents")
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
        
        # Stats
        if st.session_state.documents_processed:
            st.divider()
            st.metric("Files Processed", st.session_state.file_count)
            st.metric("Text Chunks", st.session_state.chunk_count)
            
            if st.button("Clear Documents", type="secondary"):
                st.session_state.documents_processed = False
                st.session_state.collection = None
                st.session_state.chat_history = []
                st.session_state.chunk_count = 0
                st.session_state.file_count = 0
                st.rerun()
    
    # Main chat interface
    if not st.session_state.documents_processed:
        st.info("👈 Upload documents in the sidebar to get started!")
        
        # Show example
        with st.expander("📖 How RAG Works"):
            st.markdown("""
            ### The RAG Pipeline
            
            1. **Document Processing**: Upload your documents
            2. **Chunking**: Text is split into manageable pieces
            3. **Embeddings**: Chunks are converted to vectors
            4. **Storage**: Vectors stored in database (ChromaDB)
            5. **Query**: You ask a question
            6. **Retrieval**: System finds relevant chunks
            7. **Generation**: Claude answers using retrieved context
            
            ### Why RAG?
            
            - **Reduces Hallucinations**: LLM sees actual document text
            - **Source Attribution**: Know where answers come from
            - **Up-to-date Info**: Use your latest documents
            - **Private Data**: Your documents, your answers
            """)
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
                with st.spinner("Searching documents..."):
                    vector_store = VectorStore()
                    vector_store.collection = st.session_state.collection
                    context_chunks, context_metadata = vector_store.query(prompt, n_results=5)
                
                # Generate answer
                with st.spinner("Generating answer..."):
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