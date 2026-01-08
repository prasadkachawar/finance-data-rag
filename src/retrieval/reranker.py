"""
Advanced re-ranking system for finance document retrieval
Implements multiple re-ranking strategies for improved relevance
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import requests
import json
import openai
from sentence_transformers import CrossEncoder


@dataclass
class RankingResult:
    """Result from re-ranking process"""
    document_id: str
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any]
    rank_position: int


class BaseReranker(ABC):
    """Abstract base class for re-ranking models"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[RankingResult]:
        pass
    
    @abstractmethod
    def score_relevance(self, query: str, document: str) -> float:
        pass


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API implementation
    Best commercial option for re-ranking
    """
    
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.cohere.ai/v1/rerank"
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[RankingResult]:
        """
        Re-rank documents using Cohere Rerank API
        """
        # Prepare documents for API
        doc_texts = [doc.get('content', '') for doc in documents]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_k": min(top_k, len(documents)),
            "return_documents": False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Convert to RankingResult objects
            ranking_results = []
            for i, rerank_result in enumerate(result["results"]):
                doc_idx = rerank_result["index"]
                original_doc = documents[doc_idx]
                
                ranking_result = RankingResult(
                    document_id=original_doc.get('id', f'doc_{doc_idx}'),
                    content=original_doc.get('content', ''),
                    original_score=original_doc.get('score', 0.0),
                    rerank_score=rerank_result["relevance_score"],
                    final_score=rerank_result["relevance_score"],
                    metadata=original_doc.get('metadata', {}),
                    rank_position=i + 1
                )
                ranking_results.append(ranking_result)
            
            return ranking_results
            
        except Exception as e:
            print(f"Error in Cohere reranking: {e}")
            return self._fallback_ranking(query, documents, top_k)
    
    def score_relevance(self, query: str, document: str) -> float:
        """Score single document relevance"""
        results = self.rerank(query, [{'content': document}], top_k=1)
        return results[0].rerank_score if results else 0.0
    
    def _fallback_ranking(self, query: str, documents: List[Dict], top_k: int) -> List[RankingResult]:
        """Fallback ranking when API fails"""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            result = RankingResult(
                document_id=doc.get('id', f'doc_{i}'),
                content=doc.get('content', ''),
                original_score=doc.get('score', 0.0),
                rerank_score=doc.get('score', 0.0),
                final_score=doc.get('score', 0.0),
                metadata=doc.get('metadata', {}),
                rank_position=i + 1
            )
            results.append(result)
        return results


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder based re-ranking using sentence transformers
    Open-source alternative for re-ranking
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"Error loading cross-encoder: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[RankingResult]:
        """
        Re-rank documents using cross-encoder model
        """
        if not self.model:
            return self._fallback_ranking(query, documents, top_k)
        
        try:
            # Prepare query-document pairs
            pairs = [(query, doc.get('content', '')) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Create results with scores
            results_with_scores = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                result = RankingResult(
                    document_id=doc.get('id', f'doc_{i}'),
                    content=doc.get('content', ''),
                    original_score=doc.get('score', 0.0),
                    rerank_score=float(score),
                    final_score=float(score),
                    metadata=doc.get('metadata', {}),
                    rank_position=0  # Will be set after sorting
                )
                results_with_scores.append(result)
            
            # Sort by rerank score and assign positions
            results_with_scores.sort(key=lambda x: x.rerank_score, reverse=True)
            for i, result in enumerate(results_with_scores):
                result.rank_position = i + 1
            
            return results_with_scores[:top_k]
            
        except Exception as e:
            print(f"Error in cross-encoder reranking: {e}")
            return self._fallback_ranking(query, documents, top_k)
    
    def score_relevance(self, query: str, document: str) -> float:
        """Score single document relevance"""
        if not self.model:
            return 0.0
        
        try:
            score = self.model.predict([(query, document)])
            return float(score[0])
        except:
            return 0.0
    
    def _fallback_ranking(self, query: str, documents: List[Dict], top_k: int) -> List[RankingResult]:
        """Fallback ranking when model fails"""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            result = RankingResult(
                document_id=doc.get('id', f'doc_{i}'),
                content=doc.get('content', ''),
                original_score=doc.get('score', 0.0),
                rerank_score=doc.get('score', 0.0),
                final_score=doc.get('score', 0.0),
                metadata=doc.get('metadata', {}),
                rank_position=i + 1
            )
            results.append(result)
        return results


class LLMReranker(BaseReranker):
    """
    LLM-based re-ranking using OpenAI or other language models
    Most flexible but slower and more expensive
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[RankingResult]:
        """
        Re-rank documents using LLM evaluation
        """
        # Limit batch size for LLM processing
        batch_size = min(20, len(documents))
        documents_to_rank = documents[:batch_size]
        
        try:
            # Create ranking prompt
            prompt = self._create_ranking_prompt(query, documents_to_rank)
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse ranking response
            ranking_text = response.choices[0].message.content
            ranked_indices = self._parse_ranking_response(ranking_text, len(documents_to_rank))
            
            # Create results based on LLM ranking
            results = []
            for position, doc_idx in enumerate(ranked_indices[:top_k]):
                if doc_idx < len(documents_to_rank):
                    doc = documents_to_rank[doc_idx]
                    # Assign decreasing scores based on position
                    rerank_score = 1.0 - (position * 0.1)
                    
                    result = RankingResult(
                        document_id=doc.get('id', f'doc_{doc_idx}'),
                        content=doc.get('content', ''),
                        original_score=doc.get('score', 0.0),
                        rerank_score=max(0.1, rerank_score),
                        final_score=max(0.1, rerank_score),
                        metadata=doc.get('metadata', {}),
                        rank_position=position + 1
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in LLM reranking: {e}")
            return self._fallback_ranking(query, documents, top_k)
    
    def score_relevance(self, query: str, document: str) -> float:
        """Score single document relevance using LLM"""
        try:
            prompt = f"""
Rate the relevance of this document to the query on a scale of 0.0 to 1.0:

Query: {query}

Document: {document[:1000]}...

Relevance Score (0.0-1.0):
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.5
                
        except Exception as e:
            print(f"Error in LLM scoring: {e}")
            return 0.5
    
    def _create_ranking_prompt(self, query: str, documents: List[Dict]) -> str:
        """Create prompt for LLM-based ranking"""
        doc_list = ""
        for i, doc in enumerate(documents):
            content_preview = doc.get('content', '')[:200] + "..."
            doc_list += f"{i+1}. {content_preview}\n\n"
        
        prompt = f"""
You are a financial document expert. Rank the following documents by relevance to the query.
Return ONLY the numbers (1, 2, 3, etc.) in order of relevance, separated by commas.

Query: {query}

Documents:
{doc_list}

Ranking (most relevant first):
"""
        return prompt
    
    def _parse_ranking_response(self, response: str, num_docs: int) -> List[int]:
        """Parse LLM ranking response"""
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', response)
            indices = [int(n) - 1 for n in numbers if 1 <= int(n) <= num_docs]
            
            # Add missing indices
            all_indices = list(range(num_docs))
            for idx in all_indices:
                if idx not in indices:
                    indices.append(idx)
            
            return indices
            
        except:
            # Fallback to original order
            return list(range(num_docs))
    
    def _fallback_ranking(self, query: str, documents: List[Dict], top_k: int) -> List[RankingResult]:
        """Fallback ranking when LLM fails"""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            result = RankingResult(
                document_id=doc.get('id', f'doc_{i}'),
                content=doc.get('content', ''),
                original_score=doc.get('score', 0.0),
                rerank_score=doc.get('score', 0.0),
                final_score=doc.get('score', 0.0),
                metadata=doc.get('metadata', {}),
                rank_position=i + 1
            )
            results.append(result)
        return results


class HybridReranker(BaseReranker):
    """
    Hybrid re-ranker that combines multiple re-ranking methods
    Uses ensemble approach for better accuracy
    """
    
    def __init__(self, rerankers: List[Tuple[BaseReranker, float]]):
        """
        Initialize with list of (reranker, weight) tuples
        """
        self.rerankers = rerankers
        self.total_weight = sum(weight for _, weight in rerankers)
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[RankingResult]:
        """
        Combine multiple re-ranking methods using weighted average
        """
        if not self.rerankers:
            return self._fallback_ranking(query, documents, top_k)
        
        # Get results from each reranker
        all_results = {}
        
        for reranker, weight in self.rerankers:
            try:
                results = reranker.rerank(query, documents, top_k=len(documents))
                
                for result in results:
                    doc_id = result.document_id
                    if doc_id not in all_results:
                        all_results[doc_id] = {
                            'result': result,
                            'weighted_scores': [],
                            'total_weight': 0
                        }
                    
                    all_results[doc_id]['weighted_scores'].append(result.rerank_score * weight)
                    all_results[doc_id]['total_weight'] += weight
                    
            except Exception as e:
                print(f"Error in reranker: {e}")
                continue
        
        # Calculate final scores and create results
        final_results = []
        for doc_id, data in all_results.items():
            result = data['result']
            if data['total_weight'] > 0:
                final_score = sum(data['weighted_scores']) / data['total_weight']
            else:
                final_score = result.original_score
            
            final_result = RankingResult(
                document_id=result.document_id,
                content=result.content,
                original_score=result.original_score,
                rerank_score=final_score,
                final_score=final_score,
                metadata=result.metadata,
                rank_position=0  # Will be set after sorting
            )
            final_results.append(final_result)
        
        # Sort by final score and assign positions
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(final_results):
            result.rank_position = i + 1
        
        return final_results[:top_k]
    
    def score_relevance(self, query: str, document: str) -> float:
        """Score using weighted combination of all rerankers"""
        total_score = 0.0
        total_weight = 0.0
        
        for reranker, weight in self.rerankers:
            try:
                score = reranker.score_relevance(query, document)
                total_score += score * weight
                total_weight += weight
            except:
                continue
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _fallback_ranking(self, query: str, documents: List[Dict], top_k: int) -> List[RankingResult]:
        """Fallback ranking when all rerankers fail"""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            result = RankingResult(
                document_id=doc.get('id', f'doc_{i}'),
                content=doc.get('content', ''),
                original_score=doc.get('score', 0.0),
                rerank_score=doc.get('score', 0.0),
                final_score=doc.get('score', 0.0),
                metadata=doc.get('metadata', {}),
                rank_position=i + 1
            )
            results.append(result)
        return results


class FinancialContextReranker(BaseReranker):
    """
    Domain-specific reranker for financial documents
    Applies financial domain knowledge in ranking
    """
    
    def __init__(self, base_reranker: BaseReranker):
        self.base_reranker = base_reranker
        
        # Financial importance weights
        self.financial_keywords = {
            'revenue': 1.2,
            'profit': 1.2, 
            'earnings': 1.2,
            'cash flow': 1.1,
            'balance sheet': 1.1,
            'financial performance': 1.3,
            'quarterly results': 1.2,
            'annual report': 1.2,
            'risk assessment': 1.1,
            'market analysis': 1.1
        }
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[RankingResult]:
        """
        Apply financial context boosting to re-ranking
        """
        # Get base ranking
        base_results = self.base_reranker.rerank(query, documents, top_k=len(documents))
        
        # Apply financial context boost
        boosted_results = []
        for result in base_results:
            boost_factor = self._calculate_financial_boost(query, result.content)
            boosted_score = result.rerank_score * boost_factor
            
            boosted_result = RankingResult(
                document_id=result.document_id,
                content=result.content,
                original_score=result.original_score,
                rerank_score=boosted_score,
                final_score=boosted_score,
                metadata=result.metadata,
                rank_position=0  # Will be reset
            )
            boosted_results.append(boosted_result)
        
        # Re-sort and assign positions
        boosted_results.sort(key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(boosted_results):
            result.rank_position = i + 1
        
        return boosted_results[:top_k]
    
    def score_relevance(self, query: str, document: str) -> float:
        """Score with financial context boosting"""
        base_score = self.base_reranker.score_relevance(query, document)
        boost_factor = self._calculate_financial_boost(query, document)
        return base_score * boost_factor
    
    def _calculate_financial_boost(self, query: str, document: str) -> float:
        """Calculate boost factor based on financial keywords"""
        boost = 1.0
        
        query_lower = query.lower()
        document_lower = document.lower()
        
        for keyword, weight in self.financial_keywords.items():
            if keyword in query_lower:
                if keyword in document_lower:
                    boost *= weight
        
        # Additional boost for documents with financial data patterns
        if self._has_financial_data(document):
            boost *= 1.1
        
        return min(boost, 1.5)  # Cap boost at 1.5x
    
    def _has_financial_data(self, document: str) -> bool:
        """Check if document contains financial data patterns"""
        import re
        
        # Look for financial patterns
        patterns = [
            r'\$[\d,]+',  # Dollar amounts
            r'\d+%',      # Percentages
            r'\d+\.\d+[KMB]',  # Numbers with K/M/B suffixes
            r'Q[1-4]',    # Quarters
            r'FY\d{2,4}'  # Fiscal years
        ]
        
        for pattern in patterns:
            if re.search(pattern, document):
                return True
        
        return False


# Factory function for creating rerankers
def create_reranker(config: Dict[str, Any]) -> BaseReranker:
    """
    Factory function to create reranker based on configuration
    """
    reranker_type = config.get('type', 'cohere')
    
    if reranker_type == 'cohere':
        return CohereReranker(
            api_key=config['api_key'],
            model=config.get('model', 'rerank-english-v3.0')
        )
    
    elif reranker_type == 'cross_encoder':
        return CrossEncoderReranker(
            model_name=config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        )
    
    elif reranker_type == 'llm':
        return LLMReranker(
            openai_api_key=config['openai_api_key'],
            model=config.get('model', 'gpt-3.5-turbo')
        )
    
    elif reranker_type == 'hybrid':
        rerankers = []
        for reranker_config in config['rerankers']:
            reranker = create_reranker(reranker_config['config'])
            weight = reranker_config['weight']
            rerankers.append((reranker, weight))
        
        return HybridReranker(rerankers)
    
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


# Usage example
if __name__ == "__main__":
    # Example configuration for hybrid reranker
    config = {
        'type': 'hybrid',
        'rerankers': [
            {
                'config': {
                    'type': 'cohere',
                    'api_key': 'your-cohere-key'
                },
                'weight': 0.7
            },
            {
                'config': {
                    'type': 'cross_encoder',
                    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                },
                'weight': 0.3
            }
        ]
    }
    
    # Create reranker
    # reranker = create_reranker(config)
    
    # Example documents
    documents = [
        {
            'id': 'doc1',
            'content': 'The company reported revenue growth of 15% in Q2 2024.',
            'score': 0.8,
            'metadata': {'page': 1}
        },
        {
            'id': 'doc2', 
            'content': 'Market analysis shows positive trends in the technology sector.',
            'score': 0.6,
            'metadata': {'page': 5}
        }
    ]
    
    query = "What was the revenue growth last quarter?"
    
    # Re-rank documents
    # results = reranker.rerank(query, documents, top_k=2)
    
    # Print results
    # for result in results:
    #     print(f"Rank {result.rank_position}: {result.content[:50]}...")
    #     print(f"Original Score: {result.original_score:.3f}, Rerank Score: {result.rerank_score:.3f}")
    print("Reranking system implemented successfully!")
