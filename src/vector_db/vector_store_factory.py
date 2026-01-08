"""
Vector Database Factory and Clients
Supports Pinecone, Weaviate, and Qdrant for production deployments
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import uuid
import json


@dataclass
class VectorSearchResult:
    """Standardized vector search result"""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    chunk_type: str
    page_number: int
    section: str


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        pass
    
    @abstractmethod
    def upsert_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict]]) -> bool:
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               filter_dict: Dict = None) -> List[VectorSearchResult]:
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone vector database client
    Best for production deployments with high scale requirements
    """
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        try:
            import pinecone
            self.pinecone = pinecone
        except ImportError:
            raise ImportError("Pinecone client not installed. Run: pip install pinecone-client")
        
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        # Initialize Pinecone
        self.pinecone.init(api_key=api_key, environment=environment)
        self.index = None
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create Pinecone index"""
        try:
            # Check if index already exists
            if self.index_name not in self.pinecone.list_indexes():
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    pods=1,
                    replicas=1,
                    pod_type="p1.x1"
                )
            
            self.index = self.pinecone.Index(self.index_name)
            return True
            
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            return False
    
    def upsert_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict]]) -> bool:
        """Upload vectors to Pinecone"""
        if not self.index:
            print("Index not initialized")
            return False
        
        try:
            # Convert to Pinecone format
            pinecone_vectors = []
            for vec_id, vector, metadata in vectors:
                pinecone_vectors.append({
                    'id': vec_id,
                    'values': vector.tolist(),
                    'metadata': metadata
                })
            
            # Batch upsert (Pinecone handles batching automatically)
            self.index.upsert(vectors=pinecone_vectors)
            return True
            
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               filter_dict: Dict = None) -> List[VectorSearchResult]:
        """Search vectors in Pinecone"""
        if not self.index:
            return []
        
        try:
            # Perform search
            search_kwargs = {
                'vector': query_vector.tolist(),
                'top_k': top_k,
                'include_values': False,
                'include_metadata': True
            }
            
            if filter_dict:
                search_kwargs['filter'] = filter_dict
            
            results = self.index.query(**search_kwargs)
            
            # Convert to standardized format
            search_results = []
            for match in results['matches']:
                result = VectorSearchResult(
                    id=match['id'],
                    score=match['score'],
                    content=match['metadata'].get('content', ''),
                    metadata=match['metadata'],
                    chunk_type=match['metadata'].get('chunk_type', 'text'),
                    page_number=match['metadata'].get('page_number', 0),
                    section=match['metadata'].get('section', '')
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone"""
        if not self.index:
            return False
        
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.index:
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats['total_vector_count'],
                'dimension': stats['dimension'],
                'index_fullness': stats['index_fullness']
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}


class WeaviateVectorStore(BaseVectorStore):
    """
    Weaviate vector database client
    Best for multi-modal data and complex metadata queries
    """
    
    def __init__(self, url: str, api_key: str = None, class_name: str = "FinanceDocument"):
        try:
            import weaviate
            self.weaviate = weaviate
        except ImportError:
            raise ImportError("Weaviate client not installed. Run: pip install weaviate-client")
        
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        
        # Initialize client
        auth_config = weaviate.AuthApiKey(api_key=api_key) if api_key else None
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config
        )
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create Weaviate schema"""
        try:
            # Define schema
            schema = {
                "classes": [
                    {
                        "class": self.class_name,
                        "description": "Finance document chunks",
                        "vectorizer": "none",  # We provide our own vectors
                        "properties": [
                            {
                                "name": "content",
                                "dataType": ["text"],
                                "description": "Document content"
                            },
                            {
                                "name": "chunkType",
                                "dataType": ["string"],
                                "description": "Type of chunk (text, table, image)"
                            },
                            {
                                "name": "pageNumber",
                                "dataType": ["int"],
                                "description": "Page number in document"
                            },
                            {
                                "name": "section",
                                "dataType": ["string"],
                                "description": "Document section"
                            },
                            {
                                "name": "chapter",
                                "dataType": ["string"],
                                "description": "Document chapter"
                            },
                            {
                                "name": "metadata",
                                "dataType": ["object"],
                                "description": "Additional metadata"
                            }
                        ]
                    }
                ]
            }
            
            # Create schema if it doesn't exist
            if not self.client.schema.exists(self.class_name):
                self.client.schema.create(schema)
            
            return True
            
        except Exception as e:
            print(f"Error creating Weaviate schema: {e}")
            return False
    
    def upsert_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict]]) -> bool:
        """Upload vectors to Weaviate"""
        try:
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for vec_id, vector, metadata in vectors:
                    properties = {
                        "content": metadata.get("content", ""),
                        "chunkType": metadata.get("chunk_type", "text"),
                        "pageNumber": metadata.get("page_number", 0),
                        "section": metadata.get("section", ""),
                        "chapter": metadata.get("chapter", ""),
                        "metadata": metadata
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        uuid=vec_id,
                        vector=vector.tolist()
                    )
            
            return True
            
        except Exception as e:
            print(f"Error upserting to Weaviate: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10,
               filter_dict: Dict = None) -> List[VectorSearchResult]:
        """Search vectors in Weaviate"""
        try:
            # Build query
            query = (
                self.client.query
                .get(self.class_name, ["content", "chunkType", "pageNumber", "section", "metadata"])
                .with_near_vector({"vector": query_vector.tolist()})
                .with_limit(top_k)
                .with_additional(["id", "distance"])
            )
            
            # Add filters if provided
            if filter_dict:
                where_clause = self._build_where_clause(filter_dict)
                if where_clause:
                    query = query.with_where(where_clause)
            
            result = query.do()
            
            # Convert to standardized format
            search_results = []
            if "data" in result and "Get" in result["data"]:
                for item in result["data"]["Get"][self.class_name]:
                    result_obj = VectorSearchResult(
                        id=item["_additional"]["id"],
                        score=1.0 - item["_additional"]["distance"],  # Convert distance to similarity
                        content=item.get("content", ""),
                        metadata=item.get("metadata", {}),
                        chunk_type=item.get("chunkType", "text"),
                        page_number=item.get("pageNumber", 0),
                        section=item.get("section", "")
                    )
                    search_results.append(result_obj)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching Weaviate: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Weaviate"""
        try:
            for vec_id in ids:
                self.client.data_object.delete(
                    uuid=vec_id,
                    class_name=self.class_name
                )
            return True
            
        except Exception as e:
            print(f"Error deleting from Weaviate: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Weaviate statistics"""
        try:
            # Get object count
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            count = 0
            if "data" in result and "Aggregate" in result["data"]:
                count = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
            
            return {
                'total_vectors': count,
                'class_name': self.class_name
            }
            
        except Exception as e:
            print(f"Error getting Weaviate stats: {e}")
            return {}
    
    def _build_where_clause(self, filter_dict: Dict) -> Optional[Dict]:
        """Build Weaviate where clause from filter dictionary"""
        if not filter_dict:
            return None
        
        where_clauses = []
        
        for key, value in filter_dict.items():
            if key == "page_number":
                where_clauses.append({
                    "path": ["pageNumber"],
                    "operator": "Equal",
                    "valueInt": value
                })
            elif key == "chunk_type":
                where_clauses.append({
                    "path": ["chunkType"],
                    "operator": "Equal",
                    "valueString": value
                })
            elif key == "section":
                where_clauses.append({
                    "path": ["section"],
                    "operator": "Equal",
                    "valueString": value
                })
        
        if len(where_clauses) == 1:
            return where_clauses[0]
        elif len(where_clauses) > 1:
            return {
                "operator": "And",
                "operands": where_clauses
            }
        
        return None


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector database client
    Best for high-performance and on-premise deployments
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 api_key: str = None, collection_name: str = "finance_docs"):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.QdrantClient = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
            self.PointStruct = PointStruct
        except ImportError:
            raise ImportError("Qdrant client not installed. Run: pip install qdrant-client")
        
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        
        # Initialize client
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key
        )
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create Qdrant collection"""
        try:
            # Map metric names
            distance_map = {
                "cosine": self.Distance.COSINE,
                "euclidean": self.Distance.EUCLID,
                "dot": self.Distance.DOT
            }
            
            distance = distance_map.get(metric, self.Distance.COSINE)
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.VectorParams(
                        size=dimension,
                        distance=distance
                    )
                )
            
            return True
            
        except Exception as e:
            print(f"Error creating Qdrant collection: {e}")
            return False
    
    def upsert_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict]]) -> bool:
        """Upload vectors to Qdrant"""
        try:
            points = []
            for i, (vec_id, vector, metadata) in enumerate(vectors):
                point = self.PointStruct(
                    id=vec_id if vec_id else str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload=metadata
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return True
            
        except Exception as e:
            print(f"Error upserting to Qdrant: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10,
               filter_dict: Dict = None) -> List[VectorSearchResult]:
        """Search vectors in Qdrant"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter
            search_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    condition = FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                    conditions.append(condition)
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=search_filter,
                with_payload=True
            )
            
            # Convert to standardized format
            search_results = []
            for result in results:
                result_obj = VectorSearchResult(
                    id=str(result.id),
                    score=result.score,
                    content=result.payload.get("content", ""),
                    metadata=result.payload,
                    chunk_type=result.payload.get("chunk_type", "text"),
                    page_number=result.payload.get("page_number", 0),
                    section=result.payload.get("section", "")
                )
                search_results.append(result_obj)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Qdrant"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            return True
            
        except Exception as e:
            print(f"Error deleting from Qdrant: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'total_vectors': info.points_count,
                'vectors_config': info.config.params.vectors,
                'status': info.status
            }
            
        except Exception as e:
            print(f"Error getting Qdrant stats: {e}")
            return {}


class VectorStoreFactory:
    """
    Factory class for creating vector store instances
    """
    
    @staticmethod
    def create_vector_store(store_type: str, config: Dict[str, Any]) -> BaseVectorStore:
        """
        Create vector store instance based on type and configuration
        
        Args:
            store_type: 'pinecone', 'weaviate', or 'qdrant'
            config: Configuration dictionary for the specific store
        """
        
        if store_type.lower() == "pinecone":
            return PineconeVectorStore(**config)
        
        elif store_type.lower() == "weaviate":
            return WeaviateVectorStore(**config)
        
        elif store_type.lower() == "qdrant":
            return QdrantVectorStore(**config)
        
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")


# Configuration examples
PINECONE_CONFIG = {
    'api_key': 'your-pinecone-api-key',
    'environment': 'us-west1-gcp',
    'index_name': 'finance-docs'
}

WEAVIATE_CONFIG = {
    'url': 'https://your-cluster.weaviate.network',
    'api_key': 'your-weaviate-api-key',
    'class_name': 'FinanceDocument'
}

QDRANT_CONFIG = {
    'host': 'localhost',
    'port': 6333,
    'api_key': None,
    'collection_name': 'finance_docs'
}


# Usage example
if __name__ == "__main__":
    # Create vector store
    vector_store = VectorStoreFactory.create_vector_store('qdrant', QDRANT_CONFIG)
    
    # Initialize index
    success = vector_store.create_index(dimension=1536, metric='cosine')
    print(f"Index created: {success}")
    
    # Example vectors
    vectors = [
        ("doc1", np.random.rand(1536), {
            "content": "Sample finance document content",
            "chunk_type": "text",
            "page_number": 1,
            "section": "Executive Summary"
        })
    ]
    
    # Upload vectors
    success = vector_store.upsert_vectors(vectors)
    print(f"Vectors uploaded: {success}")
    
    # Search
    query_vector = np.random.rand(1536)
    results = vector_store.search(query_vector, top_k=5)
    print(f"Search results: {len(results)}")
