"""
Comprehensive embedding strategies for finance documents
Supports text, image, and hybrid embeddings
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from abc import ABC, abstractmethod
import openai
from sentence_transformers import SentenceTransformer
import requests
import base64
from PIL import Image
import io


class BaseEmbedding(ABC):
    """Base class for all embedding strategies"""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAIEmbeddings(BaseEmbedding):
    """
    OpenAI embeddings with support for different models
    Best for financial domain understanding
    """
    
    def __init__(self, model: str = "text-embedding-3-large", api_key: str = None):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        
        # Model specifications
        self.model_specs = {
            "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
            "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8191}
        }
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [
                    np.array(data.embedding) for data in response.data
                ]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Add zero embeddings for failed batch
                embeddings.extend([np.zeros(self.dimension)] * len(batch))
        
        return embeddings
    
    def embed_with_reduced_dimension(self, text: str, target_dimension: int) -> np.ndarray:
        """Generate embedding with reduced dimensionality"""
        if self.model != "text-embedding-3-large":
            raise ValueError("Dimension reduction only supported for text-embedding-3-large")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
                dimensions=target_dimension
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating reduced embedding: {e}")
            return np.zeros(target_dimension)
    
    @property
    def dimension(self) -> int:
        return self.model_specs[self.model]["dimension"]


class CohereEmbeddings(BaseEmbedding):
    """
    Cohere embeddings with domain-specific optimization
    Good alternative to OpenAI with built-in compression
    """
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.cohere.ai/v1/embed"
        
        # Model specifications
        self.model_specs = {
            "embed-english-v3.0": {"dimension": 1024, "max_tokens": 512},
            "embed-multilingual-v3.0": {"dimension": 1024, "max_tokens": 512}
        }
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str], input_type: str = "search_document") -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        input_type: 'search_document' for documents, 'search_query' for queries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "texts": texts,
            "model": self.model,
            "input_type": input_type,
            "embedding_types": ["float"]
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            embeddings = [np.array(emb) for emb in result["embeddings"]["float"]]
            return embeddings
            
        except Exception as e:
            print(f"Error generating Cohere embeddings: {e}")
            return [np.zeros(self.dimension)] * len(texts)
    
    @property
    def dimension(self) -> int:
        return self.model_specs[self.model]["dimension"]


class SentenceTransformerEmbeddings(BaseEmbedding):
    """
    Sentence Transformers for local/private embeddings
    Good for on-premise deployments
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.model.encode([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts)
        return [emb for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class BGEEmbeddings(BaseEmbedding):
    """
    BGE embeddings - strong open-source alternative
    Good performance for retrieval tasks
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str, instruction: str = "") -> np.ndarray:
        """
        Generate embedding with optional instruction
        For queries, use instruction like "Represent this sentence for searching relevant passages:"
        """
        if instruction:
            text = instruction + text
        
        return self.model.encode([text])[0]
    
    def embed_batch(self, texts: List[str], instruction: str = "") -> List[np.ndarray]:
        """Generate embeddings for multiple texts with instruction"""
        if instruction:
            texts = [instruction + text for text in texts]
        
        embeddings = self.model.encode(texts)
        return [emb for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class MultiModalEmbeddings:
    """
    Multi-modal embeddings for text + images
    Essential for finance documents with charts and graphs
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Load CLIP for image-text embeddings
        try:
            self.clip_model = SentenceTransformer('clip-ViT-B-32')
        except:
            print("CLIP model not available, image embeddings disabled")
            self.clip_model = None
    
    def embed_text_and_image(self, text: str, image_path: str = None, 
                            image_data: bytes = None) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for both text and image
        Returns dict with 'text' and 'image' embeddings
        """
        embeddings = {}
        
        # Text embedding
        if self.openai_client:
            text_response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                encoding_format="float"
            )
            embeddings['text'] = np.array(text_response.data[0].embedding)
        
        # Image embedding
        if self.clip_model and (image_path or image_data):
            try:
                if image_path:
                    image = Image.open(image_path)
                elif image_data:
                    image = Image.open(io.BytesIO(image_data))
                
                image_embedding = self.clip_model.encode([image])
                embeddings['image'] = image_embedding[0]
                
            except Exception as e:
                print(f"Error processing image: {e}")
                embeddings['image'] = np.zeros(512)  # CLIP dimension
        
        return embeddings
    
    def embed_financial_chart(self, chart_description: str, 
                             chart_data: Dict, image_path: str = None) -> np.ndarray:
        """
        Special embedding for financial charts combining description and data
        """
        # Combine textual information
        combined_text = f"Chart: {chart_description}\n"
        
        if 'data_summary' in chart_data:
            combined_text += f"Data: {chart_data['data_summary']}\n"
        
        if 'trend' in chart_data:
            combined_text += f"Trend: {chart_data['trend']}\n"
        
        # Get text embedding
        if self.openai_client:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=combined_text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        
        return np.zeros(3072)


class HybridEmbeddingStrategy:
    """
    Hybrid embedding strategy that combines multiple embedding methods
    Optimal for finance documents with diverse content types
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedders = {}
        
        # Initialize embedders based on config
        if 'openai' in config:
            self.embedders['openai'] = OpenAIEmbeddings(**config['openai'])
        
        if 'cohere' in config:
            self.embedders['cohere'] = CohereEmbeddings(**config['cohere'])
        
        if 'sentence_transformer' in config:
            self.embedders['sentence_transformer'] = SentenceTransformerEmbeddings(
                **config['sentence_transformer']
            )
        
        if 'bge' in config:
            self.embedders['bge'] = BGEEmbeddings(**config['bge'])
        
        if 'multimodal' in config:
            self.embedders['multimodal'] = MultiModalEmbeddings(**config['multimodal'])
    
    def embed_document_chunk(self, chunk: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a document chunk using appropriate strategy
        Returns multiple embedding types for ensemble retrieval
        """
        embeddings = {}
        content_type = chunk.get('type', 'text')
        content = chunk.get('content', '')
        
        if content_type == 'text':
            # Use primary text embedder
            primary_embedder = self.embedders.get('openai') or self.embedders.get('cohere')
            if primary_embedder:
                embeddings['primary'] = primary_embedder.embed_text(content)
            
            # Add secondary embedder for ensemble
            secondary_embedder = self.embedders.get('bge')
            if secondary_embedder:
                embeddings['secondary'] = secondary_embedder.embed_text(content)
        
        elif content_type == 'table':
            # Special handling for tables
            table_text = self._format_table_for_embedding(content)
            
            primary_embedder = self.embedders.get('openai')
            if primary_embedder:
                embeddings['table'] = primary_embedder.embed_text(table_text)
        
        elif content_type == 'image':
            # Multi-modal embedding for images
            multimodal_embedder = self.embedders.get('multimodal')
            if multimodal_embedder:
                image_path = chunk.get('image_path')
                description = chunk.get('description', content)
                
                modal_embeddings = multimodal_embedder.embed_text_and_image(
                    description, image_path
                )
                embeddings.update(modal_embeddings)
        
        return embeddings
    
    def embed_query(self, query: str, query_type: str = 'general') -> Dict[str, np.ndarray]:
        """
        Generate query embeddings optimized for different query types
        """
        embeddings = {}
        
        # Adjust query based on type
        if query_type == 'financial_calculation':
            enhanced_query = f"Financial calculation query: {query}"
        elif query_type == 'definition':
            enhanced_query = f"Define financial term: {query}"
        elif query_type == 'comparison':
            enhanced_query = f"Compare financial metrics: {query}"
        else:
            enhanced_query = query
        
        # Generate embeddings with different models
        for name, embedder in self.embedders.items():
            if name == 'multimodal':
                continue  # Skip multimodal for text queries
            
            if hasattr(embedder, 'embed_text'):
                embeddings[name] = embedder.embed_text(enhanced_query)
        
        return embeddings
    
    def _format_table_for_embedding(self, table_content: str) -> str:
        """
        Format table content for better embedding representation
        """
        # Add context to table for better semantic understanding
        formatted = f"Financial table data:\n{table_content}\n"
        
        # Extract key metrics if possible
        if '$' in table_content or '%' in table_content:
            formatted += "This table contains financial metrics and percentages."
        
        return formatted


# Configuration examples
OPENAI_CONFIG = {
    'openai': {
        'model': 'text-embedding-3-large',
        'api_key': 'your-api-key'
    }
}

HYBRID_CONFIG = {
    'openai': {
        'model': 'text-embedding-3-large',
        'api_key': 'your-openai-key'
    },
    'cohere': {
        'api_key': 'your-cohere-key',
        'model': 'embed-english-v3.0'
    },
    'bge': {
        'model_name': 'BAAI/bge-large-en-v1.5'
    },
    'multimodal': {
        'openai_api_key': 'your-openai-key'
    }
}


# Usage example
if __name__ == "__main__":
    # Initialize hybrid embedding strategy
    strategy = HybridEmbeddingStrategy(HYBRID_CONFIG)
    
    # Example document chunk
    chunk = {
        'type': 'text',
        'content': 'The company reported a 25% increase in revenue, reaching $1.5 million in Q2 2024.',
        'metadata': {'page': 1, 'section': 'Financial Summary'}
    }
    
    # Generate embeddings
    embeddings = strategy.embed_document_chunk(chunk)
    
    print(f"Generated {len(embeddings)} embedding types:")
    for emb_type, embedding in embeddings.items():
        print(f"- {emb_type}: {len(embedding)} dimensions")
    
    # Example query embedding
    query_embeddings = strategy.embed_query(
        "What was the revenue growth in Q2 2024?",
        query_type='financial_calculation'
    )
    
    print(f"\nQuery embeddings: {len(query_embeddings)} types")
    for emb_type in query_embeddings:
        print(f"- {emb_type}")
