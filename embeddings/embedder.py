"""
Embeddings Module
Converts text into embeddings using sentence-transformers for vector search.
Supports multiple models including domain-specific financial embeddings.
"""

import logging
import numpy as np
import json
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.warning("sentence-transformers import failed: %s", e)
    SentenceTransformer = None

SUPPORTED_MODELS = {
    "all-MiniLM-L6-v2": {"dim": 384, "description": "General-purpose, fast"},
    "ProsusAI/finbert": {"dim": 768, "description": "Financial domain (FinBERT)"},
    "all-mpnet-base-v2": {"dim": 768, "description": "Higher quality, slower"},
}


class FinancialEmbedder:
    """
    Handles text embedding generation for financial documents.
    Supports multiple embedding models including FinBERT for domain-specific encoding.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                        Options: "all-MiniLM-L6-v2", "ProsusAI/finbert", "all-mpnet-base-v2"
        """
        self.model_name = model_name
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not available")
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
        logger.info("Embedder initialized: model=%s, dim=%d", model_name, self.model.get_sentence_embedding_dimension())
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector as numpy array
        """
        cleaned_text = self._clean_whitespace(text)
        embedding = self.model.encode(cleaned_text, normalize_embeddings=True)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
        
        Returns:
            Array of embedding vectors
        """
        cleaned_texts = [self._clean_whitespace(text) for text in texts]
        embeddings = self.model.encode(cleaned_texts, normalize_embeddings=True)
        return embeddings
    
    def _clean_whitespace(self, text: str) -> str:
        """Collapse whitespace; truncation is handled by the tokenizer."""
        return ' '.join(text.split())
    
    def embed_document_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks with 'text' field
        
        Returns:
            List of chunks with added 'embedding' field
        """
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[Dict], file_path: str):
        """
        Save embeddings to file.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            file_path: Path to save the embeddings
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved %d embeddings to %s", len(embedded_chunks), file_path)
    
    def load_embeddings(self, file_path: str) -> List[Dict]:
        """
        Load embeddings from file.
        
        Args:
            file_path: Path to load embeddings from
        
        Returns:
            List of chunks with embeddings
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                embedded_chunks = json.load(f)
            
            logger.info("Loaded %d embeddings from %s", len(embedded_chunks), file_path)
            return embedded_chunks
            
        except FileNotFoundError:
            logger.warning("Embeddings file %s not found", file_path)
            return []
        except Exception as e:
            logger.error("Error loading embeddings: %s", e)
            return []
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def find_similar_chunks(self, query_embedding: np.ndarray, 
                           embedded_chunks: List[Dict], 
                           top_k: int = 5) -> List[Dict]:
        """
        Find most similar chunks to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            embedded_chunks: List of chunks with embeddings
            top_k: Number of top similar chunks to return
        
        Returns:
            List of similar chunks with similarity scores
        """
        similarities = []
        
        for chunk in embedded_chunks:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            
            similarities.append({
                'chunk': chunk,
                'similarity': similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]


def create_financial_embeddings(data_sources: List[Dict], 
                               output_path: str = "embeddings/financial_embeddings.json"):
    """
    Create embeddings for all financial data sources.
    
    Args:
        data_sources: List of data source dictionaries
        output_path: Path to save the embeddings
    """
    embedder = FinancialEmbedder()
    
    all_chunks = []
    for source in data_sources:
        if 'chunks' in source:
            all_chunks.extend(source['chunks'])
    
    logger.info("Creating embeddings for %d chunks...", len(all_chunks))
    embedded_chunks = embedder.embed_document_chunks(all_chunks)
    embedder.save_embeddings(embedded_chunks, output_path)
    
    return embedded_chunks


def main():
    """Sample usage of the embedder."""
    print("Financial Embeddings Module")
    print("This module converts text into embeddings for vector search.")
    
    # Initialize embedder
    embedder = FinancialEmbedder()
    
    # Sample texts
    sample_texts = [
        "Apple reported strong Q4 earnings with revenue growth of 8% year-over-year.",
        "Tesla's electric vehicle sales increased significantly in the Chinese market.",
        "Microsoft's cloud computing division showed robust growth in the quarter.",
        "Amazon's e-commerce business faced challenges due to increased competition."
    ]
    
    print(f"\nGenerating embeddings for {len(sample_texts)} sample texts...")
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings_batch(sample_texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Test similarity
    query = "Apple's financial performance in the last quarter"
    query_embedding = embedder.generate_embedding(query)
    
    # Create sample chunks
    sample_chunks = [
        {'text': text, 'source': 'sample', 'type': 'news'} 
        for text in sample_texts
    ]
    
    # Embed chunks
    embedded_chunks = embedder.embed_document_chunks(sample_chunks)
    
    # Find similar chunks
    similar_chunks = embedder.find_similar_chunks(query_embedding, embedded_chunks, top_k=2)
    
    print(f"\nQuery: {query}")
    print("Most similar chunks:")
    for i, result in enumerate(similar_chunks, 1):
        print(f"{i}. Similarity: {result['similarity']:.3f}")
        print(f"   Text: {result['chunk']['text'][:100]}...")
        print()


if __name__ == "__main__":
    main()