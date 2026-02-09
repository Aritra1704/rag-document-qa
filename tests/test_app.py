"""
Test suite for RAG Document Q&A System
Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import DocumentProcessor, VectorStore


class TestDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def doc_processor(self):
        return DocumentProcessor()
    
    def test_chunk_text_basic(self, doc_processor):
        """Test basic text chunking"""
        text = "This is a test. " * 100  # ~1500 chars
        chunks = doc_processor.chunk_text(text, chunk_size=500, overlap=100)
        
        assert len(chunks) > 1, "Should create multiple chunks"
        assert all(len(chunk) <= 600 for chunk in chunks), "Chunks should respect size limit"
    
    def test_chunk_text_overlap(self, doc_processor):
        """Test that chunks have proper overlap"""
        text = "Sentence one. Sentence two. Sentence three. " * 50
        chunks = doc_processor.chunk_text(text, chunk_size=100, overlap=20)
        
        # Check that there's actual overlap
        if len(chunks) > 1:
            # Some content from first chunk should appear in second
            assert any(
                word in chunks[1] for word in chunks[0][-50:].split()
            ), "Chunks should have overlapping content"
    
    def test_chunk_text_empty(self, doc_processor):
        """Test chunking empty text"""
        chunks = doc_processor.chunk_text("")
        assert chunks == [], "Empty text should return empty list"
    
    def test_chunk_text_small(self, doc_processor):
        """Test chunking text smaller than chunk size"""
        text = "Short text."
        chunks = doc_processor.chunk_text(text, chunk_size=1000)
        
        assert len(chunks) == 1, "Small text should create single chunk"
        assert chunks[0] == text, "Single chunk should match original text"
    
    def test_extract_text_from_txt(self, doc_processor):
        """Test text extraction from TXT file"""
        content = "Test content for TXT file"
        fake_file = io.BytesIO(content.encode('utf-8'))
        
        # Mock file object
        class FakeFile:
            def read(self):
                return content.encode('utf-8')
        
        result = doc_processor.extract_text_from_txt(FakeFile())
        assert result == content, "Should extract text correctly from TXT"
    
    def test_chunk_text_sentence_boundary(self, doc_processor):
        """Test that chunks try to end at sentence boundaries"""
        text = "First sentence. Second sentence. Third sentence. " * 20
        chunks = doc_processor.chunk_text(text, chunk_size=100, overlap=20)
        
        # Most chunks should end with a period (sentence boundary)
        ending_with_period = sum(1 for chunk in chunks if chunk.rstrip().endswith('.'))
        assert ending_with_period >= len(chunks) * 0.7, "Most chunks should end at sentence boundaries"


class TestVectorStore:
    """Test vector store functionality"""
    
    @pytest.fixture
    def vector_store(self):
        return VectorStore()
    
    def test_initialize_collection(self, vector_store):
        """Test collection initialization"""
        collection = vector_store.initialize_collection("test_collection")
        
        assert collection is not None, "Collection should be created"
        assert vector_store.collection is not None, "Collection should be stored"
    
    def test_add_and_query_documents(self, vector_store):
        """Test adding documents and querying"""
        vector_store.initialize_collection("test_docs")
        
        # Add test documents
        chunks = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a popular programming language."
        ]
        metadatas = [
            {"source": "ml.txt"},
            {"source": "dl.txt"},
            {"source": "python.txt"}
        ]
        ids = ["chunk_0", "chunk_1", "chunk_2"]
        
        vector_store.add_documents(chunks, metadatas, ids)
        
        # Query for ML-related content
        results, metadata = vector_store.query("What is machine learning?", n_results=2)
        
        assert len(results) > 0, "Should return results"
        assert len(results) <= 2, "Should respect n_results limit"
        # First result should be most relevant (ML-related)
        assert "machine learning" in results[0].lower() or "artificial intelligence" in results[0].lower()
    
    def test_query_with_no_documents(self, vector_store):
        """Test querying empty collection"""
        vector_store.initialize_collection("empty_collection")
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            vector_store.query("test query", n_results=1)
    
    def test_multiple_collections(self):
        """Test creating multiple collections"""
        vs1 = VectorStore()
        vs2 = VectorStore()
        
        vs1.initialize_collection("collection_1")
        vs2.initialize_collection("collection_2")
        
        # Add different documents to each
        vs1.add_documents(["Document 1"], [{"source": "doc1"}], ["id1"])
        vs2.add_documents(["Document 2"], [{"source": "doc2"}], ["id2"])
        
        # Query each collection
        results1, _ = vs1.query("Document", n_results=1)
        results2, _ = vs2.query("Document", n_results=1)
        
        # Each should return its own document
        assert "Document 1" in results1[0]
        assert "Document 2" in results2[0]


class TestIntegration:
    """Integration tests for the full RAG pipeline"""
    
    def test_full_pipeline_basic(self):
        """Test the complete RAG pipeline"""
        # Initialize components
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        vector_store.initialize_collection("integration_test")
        
        # Create test document
        test_doc = """
        Machine Learning Introduction
        
        Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence based on the idea that systems can learn from data,
        identify patterns and make decisions with minimal human intervention.
        
        Types of Machine Learning:
        1. Supervised Learning: Learning with labeled data
        2. Unsupervised Learning: Learning with unlabeled data
        3. Reinforcement Learning: Learning through rewards and penalties
        """
        
        # Process document
        chunks = doc_processor.chunk_text(test_doc, chunk_size=200, overlap=50)
        
        assert len(chunks) > 0, "Should create chunks from document"
        
        # Add to vector store
        metadatas = [{"source": "ml_intro.txt"} for _ in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        vector_store.add_documents(chunks, metadatas, ids)
        
        # Query
        results, metadata = vector_store.query("What is supervised learning?", n_results=2)
        
        assert len(results) > 0, "Should retrieve relevant chunks"
        # Results should mention supervised learning
        combined_results = " ".join(results).lower()
        assert "supervised" in combined_results, "Results should mention supervised learning"
    
    def test_multiple_document_types(self):
        """Test processing multiple document types"""
        doc_processor = DocumentProcessor()
        
        # Test TXT
        txt_content = "This is a text file."
        class FakeTxtFile:
            def read(self):
                return txt_content.encode('utf-8')
        
        txt_result = doc_processor.extract_text_from_txt(FakeTxtFile())
        assert txt_result == txt_content, "Should extract TXT correctly"
        
        # Chunking should work on all types
        chunks = doc_processor.chunk_text(txt_result)
        assert len(chunks) > 0, "Should chunk extracted text"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_long_document(self):
        """Test handling of very long documents"""
        doc_processor = DocumentProcessor()
        
        # Create a very long document
        long_text = "This is a sentence. " * 10000  # ~200k characters
        
        chunks = doc_processor.chunk_text(long_text, chunk_size=1000, overlap=100)
        
        assert len(chunks) > 100, "Should create many chunks for long document"
        assert all(len(chunk) <= 1100 for chunk in chunks), "All chunks should respect size"
    
    def test_special_characters(self):
        """Test handling of special characters"""
        doc_processor = DocumentProcessor()
        
        text = "Special chars: @#$%^&*() 你好 مرحبا עברית"
        chunks = doc_processor.chunk_text(text)
        
        assert len(chunks) > 0, "Should handle special characters"
        assert text in chunks[0], "Should preserve special characters"
    
    def test_empty_chunks_filtered(self):
        """Test that empty chunks are filtered out"""
        doc_processor = DocumentProcessor()
        
        text = "Text.\n\n\n\n\nMore text."
        chunks = doc_processor.chunk_text(text, chunk_size=10)
        
        assert all(chunk.strip() for chunk in chunks), "Should filter empty chunks"
    
    def test_unicode_handling(self):
        """Test Unicode character handling"""
        doc_processor = DocumentProcessor()
        
        text = "Unicode test: émojis 🚀 and symbols ∑∏∫"
        chunks = doc_processor.chunk_text(text)
        
        assert len(chunks) > 0, "Should handle Unicode"
        assert "🚀" in chunks[0], "Should preserve emojis"


class TestPerformance:
    """Performance tests (optional, can be slow)"""
    
    @pytest.mark.slow
    def test_large_batch_processing(self):
        """Test processing large batches of documents"""
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        vector_store.initialize_collection("performance_test")
        
        # Create 100 documents
        documents = [f"Document {i}: Test content about topic {i % 10}" * 20 
                    for i in range(100)]
        
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        for i, doc in enumerate(documents):
            chunks = doc_processor.chunk_text(doc)
            all_chunks.extend(chunks)
            all_metadata.extend([{"source": f"doc_{i}.txt"}] * len(chunks))
            all_ids.extend([f"chunk_{i}_{j}" for j in range(len(chunks))])
        
        # Should handle large batch
        vector_store.add_documents(all_chunks, all_metadata, all_ids)
        
        # Query should still be fast
        results, _ = vector_store.query("topic 5", n_results=5)
        assert len(results) == 5, "Should return requested number of results"


def test_imports():
    """Test that all required imports work"""
    import streamlit
    import anthropic
    import chromadb
    import PyPDF2
    import docx
    
    assert True, "All imports successful"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])