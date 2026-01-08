"""
Comprehensive evaluation metrics for RAG pipeline
Covers retrieval, generation, and end-to-end performance
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import json
import re
from collections import defaultdict


@dataclass
class EvaluationResult:
    """Standard evaluation result container"""
    metric_name: str
    score: float
    details: Dict[str, Any]
    component: str  # 'retrieval', 'generation', 'end_to_end'


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics"""
    
    @abstractmethod
    def compute(self, predictions: List[Any], ground_truth: List[Any], **kwargs) -> EvaluationResult:
        pass


class RetrievalMetrics:
    """
    Comprehensive retrieval evaluation metrics
    """
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[List[str]], 
                           relevant_docs: List[List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_docs: List of retrieved document IDs for each query
            relevant_docs: List of relevant document IDs for each query
        """
        mrr_scores = []
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            rr = 0.0
            relevant_set = set(relevant)
            
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            
            mrr_scores.append(rr)
        
        return np.mean(mrr_scores)
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[List[str]], 
                    relevant_docs: List[List[str]], k: int = 10) -> float:
        """
        Calculate Recall@K
        """
        recall_scores = []
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_k = set(retrieved[:k])
            relevant_set = set(relevant)
            
            if len(relevant_set) == 0:
                recall_scores.append(0.0)
            else:
                recall = len(retrieved_k.intersection(relevant_set)) / len(relevant_set)
                recall_scores.append(recall)
        
        return np.mean(recall_scores)
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[List[str]], 
                      relevant_docs: List[List[str]], k: int = 10) -> float:
        """
        Calculate Precision@K
        """
        precision_scores = []
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_k = set(retrieved[:k])
            relevant_set = set(relevant)
            
            if len(retrieved_k) == 0:
                precision_scores.append(0.0)
            else:
                precision = len(retrieved_k.intersection(relevant_set)) / len(retrieved_k)
                precision_scores.append(precision)
        
        return np.mean(precision_scores)
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[List[str]], 
                  relevance_scores: List[Dict[str, float]], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            retrieved_docs: List of retrieved document IDs for each query
            relevance_scores: List of dicts mapping doc_id to relevance score (0-3)
        """
        ndcg_scores = []
        
        for retrieved, relevances in zip(retrieved_docs, relevance_scores):
            # Calculate DCG
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved[:k], 1):
                relevance = relevances.get(doc_id, 0)
                dcg += (2 ** relevance - 1) / np.log2(rank + 1)
            
            # Calculate IDCG (Ideal DCG)
            sorted_relevances = sorted(relevances.values(), reverse=True)
            idcg = 0.0
            for rank, relevance in enumerate(sorted_relevances[:k], 1):
                idcg += (2 ** relevance - 1) / np.log2(rank + 1)
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)
    
    @staticmethod
    def hit_rate(retrieved_docs: List[List[str]], 
                 relevant_docs: List[List[str]]) -> float:
        """
        Calculate Hit Rate (fraction of queries with at least one relevant result)
        """
        hits = 0
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_set = set(retrieved)
            relevant_set = set(relevant)
            
            if retrieved_set.intersection(relevant_set):
                hits += 1
        
        return hits / len(retrieved_docs) if retrieved_docs else 0.0


class GenerationMetrics:
    """
    Text generation evaluation metrics
    """
    
    @staticmethod
    def bleu_score(predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU score for generated text
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]  # BLEU expects list of reference token lists
                
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            
            return np.mean(bleu_scores)
            
        except ImportError:
            print("NLTK not available for BLEU calculation")
            return 0.0
    
    @staticmethod
    def rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(rouge1_scores),
                'rouge2': np.mean(rouge2_scores),
                'rougeL': np.mean(rougeL_scores)
            }
            
        except ImportError:
            print("rouge-score not available")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    @staticmethod
    def bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore for semantic similarity
        """
        try:
            from bert_score import score
            
            P, R, F1 = score(predictions, references, lang="en", verbose=False)
            
            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item()
            }
            
        except ImportError:
            print("bert-score not available")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    @staticmethod
    def factual_accuracy(predictions: List[str], source_documents: List[List[str]]) -> float:
        """
        Measure factual accuracy by checking claims against source documents
        """
        accuracy_scores = []
        
        for pred, sources in zip(predictions, source_documents):
            # Extract numerical claims from prediction
            numerical_claims = GenerationMetrics._extract_numerical_claims(pred)
            
            if not numerical_claims:
                accuracy_scores.append(1.0)  # No claims to verify
                continue
            
            verified_claims = 0
            for claim in numerical_claims:
                if GenerationMetrics._verify_claim_in_sources(claim, sources):
                    verified_claims += 1
            
            accuracy = verified_claims / len(numerical_claims)
            accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores)
    
    @staticmethod
    def _extract_numerical_claims(text: str) -> List[str]:
        """Extract numerical claims from text"""
        # Pattern for financial numbers and percentages
        patterns = [
            r'\$[\d,]+(?:\.\d{2})?[KMB]?',  # Dollar amounts
            r'\d+(?:\.\d+)?%',              # Percentages
            r'\d{4}',                       # Years
            r'Q[1-4]',                      # Quarters
        ]
        
        claims = []
        for pattern in patterns:
            claims.extend(re.findall(pattern, text))
        
        return claims
    
    @staticmethod
    def _verify_claim_in_sources(claim: str, sources: List[str]) -> bool:
        """Verify if claim appears in source documents"""
        for source in sources:
            if claim in source:
                return True
        return False
    
    @staticmethod
    def reference_accuracy(predictions: List[str], expected_references: List[List[str]]) -> float:
        """
        Check if generated text includes correct references (page numbers, sections)
        """
        accuracy_scores = []
        
        for pred, expected_refs in zip(predictions, expected_references):
            # Extract references from prediction
            found_refs = GenerationMetrics._extract_references(pred)
            expected_set = set(expected_refs)
            found_set = set(found_refs)
            
            if not expected_set:
                accuracy_scores.append(1.0 if not found_set else 0.0)
            else:
                accuracy = len(found_set.intersection(expected_set)) / len(expected_set)
                accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores)
    
    @staticmethod
    def _extract_references(text: str) -> List[str]:
        """Extract page/section references from text"""
        patterns = [
            r'[Pp]age\s+(\d+)',
            r'\[Page\s+(\d+)\]',
            r'[Ss]ection\s+(\d+(?:\.\d+)*)',
            r'\[Section\s+([^\]]+)\]'
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        return references


class FinancialDomainMetrics:
    """
    Domain-specific metrics for financial document QA
    """
    
    @staticmethod
    def financial_concept_coverage(predictions: List[str], 
                                 financial_concepts: List[List[str]]) -> float:
        """
        Measure coverage of expected financial concepts in responses
        """
        coverage_scores = []
        
        for pred, concepts in zip(predictions, financial_concepts):
            pred_lower = pred.lower()
            covered_concepts = 0
            
            for concept in concepts:
                if concept.lower() in pred_lower:
                    covered_concepts += 1
            
            coverage = covered_concepts / len(concepts) if concepts else 1.0
            coverage_scores.append(coverage)
        
        return np.mean(coverage_scores)
    
    @staticmethod
    def numerical_accuracy(predictions: List[str], 
                          ground_truth_numbers: List[Dict[str, float]]) -> float:
        """
        Check accuracy of numerical values in financial responses
        """
        accuracy_scores = []
        
        for pred, truth_numbers in zip(predictions, ground_truth_numbers):
            extracted_numbers = FinancialDomainMetrics._extract_financial_numbers(pred)
            
            if not truth_numbers:
                accuracy_scores.append(1.0 if not extracted_numbers else 0.0)
                continue
            
            correct_numbers = 0
            total_numbers = len(truth_numbers)
            
            for key, expected_value in truth_numbers.items():
                if key in extracted_numbers:
                    if abs(extracted_numbers[key] - expected_value) < 0.01:  # Small tolerance
                        correct_numbers += 1
            
            accuracy = correct_numbers / total_numbers
            accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores)
    
    @staticmethod
    def _extract_financial_numbers(text: str) -> Dict[str, float]:
        """Extract financial numbers with context"""
        numbers = {}
        
        # Revenue pattern
        revenue_match = re.search(r'revenue[^\d]*(\$?[\d,]+(?:\.\d+)?)[KMB]?', text.lower())
        if revenue_match:
            numbers['revenue'] = FinancialDomainMetrics._parse_financial_number(revenue_match.group(1))
        
        # Profit pattern
        profit_match = re.search(r'profit[^\d]*(\$?[\d,]+(?:\.\d+)?)[KMB]?', text.lower())
        if profit_match:
            numbers['profit'] = FinancialDomainMetrics._parse_financial_number(profit_match.group(1))
        
        # Percentage patterns
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)%', text)
        if pct_matches:
            numbers['percentage'] = float(pct_matches[0])
        
        return numbers
    
    @staticmethod
    def _parse_financial_number(num_str: str) -> float:
        """Parse financial number string to float"""
        # Remove $ and commas
        clean_num = re.sub(r'[$,]', '', num_str)
        
        # Handle K, M, B suffixes
        multiplier = 1
        if clean_num.endswith('K'):
            multiplier = 1000
            clean_num = clean_num[:-1]
        elif clean_num.endswith('M'):
            multiplier = 1000000
            clean_num = clean_num[:-1]
        elif clean_num.endswith('B'):
            multiplier = 1000000000
            clean_num = clean_num[:-1]
        
        try:
            return float(clean_num) * multiplier
        except ValueError:
            return 0.0


class EndToEndEvaluator:
    """
    Comprehensive end-to-end evaluation for RAG pipeline
    """
    
    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.financial_metrics = FinancialDomainMetrics()
    
    def evaluate_retrieval(self, retrieval_results: Dict[str, Any]) -> Dict[str, EvaluationResult]:
        """Evaluate retrieval component"""
        results = {}
        
        retrieved_docs = retrieval_results['retrieved_docs']
        relevant_docs = retrieval_results['relevant_docs']
        relevance_scores = retrieval_results.get('relevance_scores', [])
        
        # MRR
        mrr = self.retrieval_metrics.mean_reciprocal_rank(retrieved_docs, relevant_docs)
        results['mrr'] = EvaluationResult(
            metric_name='MRR',
            score=mrr,
            details={'description': 'Mean Reciprocal Rank'},
            component='retrieval'
        )
        
        # Recall@K
        for k in [5, 10, 20]:
            recall = self.retrieval_metrics.recall_at_k(retrieved_docs, relevant_docs, k)
            results[f'recall@{k}'] = EvaluationResult(
                metric_name=f'Recall@{k}',
                score=recall,
                details={'k': k, 'description': f'Recall at rank {k}'},
                component='retrieval'
            )
        
        # Precision@K
        for k in [5, 10, 20]:
            precision = self.retrieval_metrics.precision_at_k(retrieved_docs, relevant_docs, k)
            results[f'precision@{k}'] = EvaluationResult(
                metric_name=f'Precision@{k}',
                score=precision,
                details={'k': k, 'description': f'Precision at rank {k}'},
                component='retrieval'
            )
        
        # NDCG@K (if relevance scores available)
        if relevance_scores:
            for k in [5, 10, 20]:
                ndcg = self.retrieval_metrics.ndcg_at_k(retrieved_docs, relevance_scores, k)
                results[f'ndcg@{k}'] = EvaluationResult(
                    metric_name=f'NDCG@{k}',
                    score=ndcg,
                    details={'k': k, 'description': f'NDCG at rank {k}'},
                    component='retrieval'
                )
        
        # Hit Rate
        hit_rate = self.retrieval_metrics.hit_rate(retrieved_docs, relevant_docs)
        results['hit_rate'] = EvaluationResult(
            metric_name='Hit Rate',
            score=hit_rate,
            details={'description': 'Fraction of queries with relevant results'},
            component='retrieval'
        )
        
        return results
    
    def evaluate_generation(self, generation_results: Dict[str, Any]) -> Dict[str, EvaluationResult]:
        """Evaluate generation component"""
        results = {}
        
        predictions = generation_results['predictions']
        references = generation_results['references']
        source_documents = generation_results.get('source_documents', [])
        
        # BLEU Score
        bleu = self.generation_metrics.bleu_score(predictions, references)
        results['bleu'] = EvaluationResult(
            metric_name='BLEU',
            score=bleu,
            details={'description': 'BLEU score for text similarity'},
            component='generation'
        )
        
        # ROUGE Scores
        rouge_scores = self.generation_metrics.rouge_scores(predictions, references)
        for rouge_type, score in rouge_scores.items():
            results[rouge_type] = EvaluationResult(
                metric_name=rouge_type.upper(),
                score=score,
                details={'description': f'{rouge_type.upper()} score'},
                component='generation'
            )
        
        # BERTScore
        bert_scores = self.generation_metrics.bert_score(predictions, references)
        for bert_type, score in bert_scores.items():
            results[bert_type] = EvaluationResult(
                metric_name=bert_type,
                score=score,
                details={'description': f'{bert_type} semantic similarity'},
                component='generation'
            )
        
        # Factual Accuracy (if source documents available)
        if source_documents:
            factual_acc = self.generation_metrics.factual_accuracy(predictions, source_documents)
            results['factual_accuracy'] = EvaluationResult(
                metric_name='Factual Accuracy',
                score=factual_acc,
                details={'description': 'Accuracy of factual claims'},
                component='generation'
            )
        
        return results
    
    def evaluate_financial_domain(self, domain_results: Dict[str, Any]) -> Dict[str, EvaluationResult]:
        """Evaluate financial domain-specific metrics"""
        results = {}
        
        predictions = domain_results['predictions']
        financial_concepts = domain_results.get('financial_concepts', [])
        ground_truth_numbers = domain_results.get('ground_truth_numbers', [])
        
        # Financial Concept Coverage
        if financial_concepts:
            concept_coverage = self.financial_metrics.financial_concept_coverage(
                predictions, financial_concepts
            )
            results['concept_coverage'] = EvaluationResult(
                metric_name='Financial Concept Coverage',
                score=concept_coverage,
                details={'description': 'Coverage of expected financial concepts'},
                component='domain'
            )
        
        # Numerical Accuracy
        if ground_truth_numbers:
            numerical_acc = self.financial_metrics.numerical_accuracy(
                predictions, ground_truth_numbers
            )
            results['numerical_accuracy'] = EvaluationResult(
                metric_name='Numerical Accuracy',
                score=numerical_acc,
                details={'description': 'Accuracy of numerical values'},
                component='domain'
            )
        
        return results
    
    def comprehensive_evaluation(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all components
        """
        all_results = {}
        
        # Retrieval evaluation
        if 'retrieval' in evaluation_data:
            retrieval_results = self.evaluate_retrieval(evaluation_data['retrieval'])
            all_results['retrieval'] = retrieval_results
        
        # Generation evaluation
        if 'generation' in evaluation_data:
            generation_results = self.evaluate_generation(evaluation_data['generation'])
            all_results['generation'] = generation_results
        
        # Financial domain evaluation
        if 'domain' in evaluation_data:
            domain_results = self.evaluate_financial_domain(evaluation_data['domain'])
            all_results['domain'] = domain_results
        
        # Calculate component averages
        component_averages = {}
        for component, metrics in all_results.items():
            scores = [result.score for result in metrics.values()]
            component_averages[component] = np.mean(scores) if scores else 0.0
        
        # Overall score
        overall_score = np.mean(list(component_averages.values())) if component_averages else 0.0
        
        return {
            'results': all_results,
            'component_averages': component_averages,
            'overall_score': overall_score,
            'summary': self._generate_summary(all_results, component_averages, overall_score)
        }
    
    def _generate_summary(self, results: Dict, component_averages: Dict, overall_score: float) -> str:
        """Generate evaluation summary"""
        summary = f"Overall Score: {overall_score:.3f}\n\n"
        
        for component, avg_score in component_averages.items():
            summary += f"{component.title()} Component: {avg_score:.3f}\n"
            
            if component in results:
                top_metrics = sorted(
                    results[component].items(), 
                    key=lambda x: x[1].score, 
                    reverse=True
                )[:3]
                
                for metric_name, result in top_metrics:
                    summary += f"  - {result.metric_name}: {result.score:.3f}\n"
            
            summary += "\n"
        
        return summary


# Usage example and test data
if __name__ == "__main__":
    # Create evaluator
    evaluator = EndToEndEvaluator()
    
    # Example evaluation data
    evaluation_data = {
        'retrieval': {
            'retrieved_docs': [
                ['doc1', 'doc2', 'doc3'],
                ['doc4', 'doc5', 'doc6']
            ],
            'relevant_docs': [
                ['doc1', 'doc3'],
                ['doc4']
            ],
            'relevance_scores': [
                {'doc1': 3, 'doc2': 1, 'doc3': 2},
                {'doc4': 3, 'doc5': 0, 'doc6': 1}
            ]
        },
        'generation': {
            'predictions': [
                "The company's revenue increased by 15% to $1.2M in Q2 2024.",
                "Market analysis shows strong growth in the technology sector."
            ],
            'references': [
                "Revenue grew 15% to $1.2 million in the second quarter of 2024.",
                "The technology sector demonstrated robust growth according to market analysis."
            ],
            'source_documents': [
                ["Revenue: $1.2M, Growth: 15%"],
                ["Technology sector growth trends"]
            ]
        },
        'domain': {
            'predictions': [
                "The company's revenue increased by 15% to $1.2M in Q2 2024.",
                "EBITDA margins improved to 25% this quarter."
            ],
            'financial_concepts': [
                ['revenue', 'growth'],
                ['ebitda', 'margins']
            ],
            'ground_truth_numbers': [
                {'revenue': 1200000, 'percentage': 15},
                {'percentage': 25}
            ]
        }
    }
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(evaluation_data)
    
    # Print summary
    print(results['summary'])
    print(f"Component Averages: {results['component_averages']}")
    print(f"Overall Score: {results['overall_score']:.3f}")
