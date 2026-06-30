from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from backend.catalog_loader import CatalogLoader

class SemanticRetriever:
    def __init__(self, catalog_loader: CatalogLoader):
        self.loader = catalog_loader
        self.catalog = self.loader.get_all()
        # Initialize a small, fast sentence-transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus_embeddings = None
        self._build_index()

    def _build_index(self):
        if not self.catalog:
            return
            
        # Create a rich text representation for each item to embed
        corpus_texts = []
        for item in self.catalog:
            text = f"Name: {item.get('name', '')}. " \
                   f"Type: {item.get('test_type', '')}. " \
                   f"Skills: {item.get('skills_tested', '')}. " \
                   f"Level: {item.get('level', '')}. " \
                   f"Description: {item.get('description', '')}"
            corpus_texts.append(text)
            
        self.corpus_embeddings = self.model.encode(corpus_texts, convert_to_tensor=False)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.catalog or self.corpus_embeddings is None:
            return []
            
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # Calculate cosine similarities manually using numpy to keep it simple and dependency-light
        norm_query = np.linalg.norm(query_embedding)
        norm_corpus = np.linalg.norm(self.corpus_embeddings, axis=1)
        
        # Prevent division by zero
        if norm_query == 0:
            return []
            
        similarities = np.dot(self.corpus_embeddings, query_embedding) / (norm_corpus * norm_query)
        
        # Get top_k indices sorted by highest similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        # threshold for relevance to avoid returning totally unrelated stuff
        threshold = 0.2 
        for idx in top_indices:
            if similarities[idx] > threshold:
                results.append(self.catalog[idx])
                
        return results
