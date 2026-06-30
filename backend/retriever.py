import os
from google import genai
from google.genai import types
import numpy as np
from typing import List, Dict, Any
from backend.catalog_loader import CatalogLoader

class SemanticRetriever:
    def __init__(self, catalog_loader: CatalogLoader):
        self.loader = catalog_loader
        self.catalog = self.loader.get_all()
        
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            
        self.corpus_embeddings = None
        self._build_index()

    def _get_embedding(self, text: str) -> np.ndarray:
        if not self.client:
            # Fallback if no API key is available
            return np.zeros(768)
        try:
            response = self.client.models.embed_content(
                model='embedding-001',
                contents=text
            )
            return np.array(response.embeddings[0].values)
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.zeros(768)

    def _build_index(self):
        if not self.catalog or not self.client:
            return
            
        corpus_texts = []
        for item in self.catalog:
            text = f"Name: {item.get('name', '')}. " \
                   f"Type: {item.get('test_type', '')}. " \
                   f"Skills: {item.get('skills_tested', '')}. " \
                   f"Level: {item.get('level', '')}. " \
                   f"Description: {item.get('description', '')}"
            corpus_texts.append(text)
            
        # Get embeddings sequentially (for small catalog, this is fast enough)
        embeddings = []
        for text in corpus_texts:
            embeddings.append(self._get_embedding(text))
            
        self.corpus_embeddings = np.array(embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.catalog or self.corpus_embeddings is None or not self.client:
            return []
            
        query_embedding = self._get_embedding(query)
        
        norm_query = np.linalg.norm(query_embedding)
        norm_corpus = np.linalg.norm(self.corpus_embeddings, axis=1)
        
        # Prevent division by zero
        if norm_query == 0:
            return []
            
        # Add epsilon to prevent division by zero in corpus norms
        norm_corpus = np.where(norm_corpus == 0, 1e-10, norm_corpus)
        
        similarities = np.dot(self.corpus_embeddings, query_embedding) / (norm_corpus * norm_query)
        
        # Get top_k indices sorted by highest similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        # lower threshold slightly for API embeddings
        threshold = 0.4
        for idx in top_indices:
            if similarities[idx] > threshold:
                results.append(self.catalog[idx])
                
        return results
