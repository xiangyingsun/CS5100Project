# 3. vector_db_manager.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
from chromadb.utils.embedding_functions import EmbeddingFunction

class SentenceTransformerEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.embedder = SentenceTransformer(model_name)
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Convert texts to embeddings using sentence-transformers"""
        return self.embedder.encode(input, normalize_embeddings=True).tolist()

class YogaVectorDB:
    def __init__(self, collection_name: str = "yoga_poses", persist_dir: str = "./yoga_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_function = SentenceTransformerEmbedding()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        
    def populate_db(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Batch insert yoga poses into ChromaDB"""
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logging.info(f"Successfully inserted {len(documents)} poses")
            
        except Exception as e:
            logging.error(f"DB insertion failed: {str(e)}")
            raise
    
    def query_poses(
        self, 
        query_text: str, 
        level: str = None, 
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Search poses with metadata filtering"""
        where_filter = None
        if level:
            where_filter = {"Level": {"$eq": level}}
            
        return self.collection.query(
            query_texts=[query_text],
            n_results=max_results,
            where=where_filter
        )
