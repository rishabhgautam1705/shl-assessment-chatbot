import json
import os
from typing import List, Dict, Any

class CatalogLoader:
    def __init__(self, filepath="data/catalog.json"):
        self.filepath = filepath
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all(self) -> List[Dict[str, Any]]:
        return self.catalog

    def search_exact(self, query: str, field: str = "name") -> List[Dict[str, Any]]:
        """Fallback exact/substring matching"""
        results = []
        query_lower = query.lower()
        for item in self.catalog:
            val = item.get(field, "")
            if val and query_lower in str(val).lower():
                results.append(item)
        return results

    def search_by_metadata(self, 
                           skill: str = None, 
                           role: str = None, 
                           name: str = None, 
                           test_type: str = None, 
                           duration: str = None, 
                           keyword: str = None) -> List[Dict[str, Any]]:
        """Advanced filtering supporting multiple dimensions."""
        results = []
        for item in self.catalog:
            match = True
            
            if name and name.lower() not in item.get("name", "").lower():
                match = False
            if test_type and test_type.lower() not in item.get("test_type", "").lower():
                match = False
            if duration and duration.lower() not in item.get("duration", "").lower():
                match = False
                
            # For skills, role, or generic keywords, we search across description and skills_tested
            text_corpus = f"{item.get('description', '')} {item.get('skills_tested', '')}".lower()
            
            if skill and skill.lower() not in text_corpus:
                match = False
            if role and role.lower() not in text_corpus:
                match = False
            if keyword and keyword.lower() not in text_corpus and keyword.lower() not in item.get("name", "").lower():
                match = False
                
            if match:
                results.append(item)
                
        return results
