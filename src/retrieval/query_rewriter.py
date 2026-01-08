"""
Query rewriting and optimization for finance domain
Handles intent classification, entity extraction, and context enhancement
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import openai
from abc import ABC, abstractmethod


class QueryIntent(Enum):
    DEFINITION = "definition"
    CALCULATION = "calculation" 
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE = "compliance"
    GENERAL = "general"


@dataclass
class QueryAnalysis:
    original_query: str
    rewritten_queries: List[str]
    intent: QueryIntent
    entities: List[str]
    financial_terms: List[str]
    time_references: List[str]
    confidence: float


class BaseQueryRewriter(ABC):
    """Abstract base class for query rewriters"""
    
    @abstractmethod
    def rewrite_query(self, query: str, context: Dict = None) -> QueryAnalysis:
        pass


class FinancialQueryRewriter(BaseQueryRewriter):
    """
    Advanced query rewriter specifically designed for financial documents
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Financial terminology mappings
        self.financial_synonyms = {
            'revenue': ['sales', 'income', 'turnover', 'top line'],
            'profit': ['earnings', 'net income', 'bottom line', 'margins'],
            'assets': ['holdings', 'investments', 'properties'],
            'liabilities': ['debts', 'obligations', 'payables'],
            'equity': ['shareholders equity', 'book value', 'net worth'],
            'roi': ['return on investment', 'return', 'yield'],
            'ebitda': ['earnings before interest tax depreciation amortization'],
            'capex': ['capital expenditure', 'capital spending'],
            'opex': ['operating expenses', 'operational costs']
        }
        
        # Common financial abbreviations
        self.abbreviations = {
            'p&l': 'profit and loss',
            'b/s': 'balance sheet',
            'cf': 'cash flow',
            'yoy': 'year over year',
            'qoq': 'quarter over quarter',
            'ytd': 'year to date',
            'fy': 'fiscal year',
            'q1': 'first quarter',
            'q2': 'second quarter', 
            'q3': 'third quarter',
            'q4': 'fourth quarter'
        }
        
        # Intent patterns
        self.intent_patterns = {
            QueryIntent.DEFINITION: [
                r'what is', r'define', r'meaning of', r'explain'
            ],
            QueryIntent.CALCULATION: [
                r'calculate', r'compute', r'how much', r'what was the', r'\d+%', r'\$[\d,]+'
            ],
            QueryIntent.COMPARISON: [
                r'compare', r'versus', r'vs', r'difference between', r'higher than', r'lower than'
            ],
            QueryIntent.TREND_ANALYSIS: [
                r'trend', r'over time', r'growth', r'decline', r'increase', r'decrease'
            ],
            QueryIntent.RISK_ASSESSMENT: [
                r'risk', r'volatility', r'uncertainty', r'exposure'
            ],
            QueryIntent.COMPLIANCE: [
                r'regulation', r'compliance', r'requirement', r'standard'
            ]
        }
    
    def rewrite_query(self, query: str, context: Dict = None) -> QueryAnalysis:
        """
        Comprehensive query rewriting with multiple techniques
        """
        # Step 1: Basic preprocessing
        processed_query = self._preprocess_query(query)
        
        # Step 2: Intent classification
        intent = self._classify_intent(processed_query)
        
        # Step 3: Entity extraction
        entities = self._extract_entities(processed_query)
        financial_terms = self._extract_financial_terms(processed_query)
        time_refs = self._extract_time_references(processed_query)
        
        # Step 4: Generate query variants
        query_variants = self._generate_query_variants(
            processed_query, intent, entities, financial_terms
        )
        
        # Step 5: Context enhancement
        if context:
            enhanced_variants = self._enhance_with_context(query_variants, context)
            query_variants.extend(enhanced_variants)
        
        # Step 6: LLM-based rewriting (if available)
        if self.openai_client:
            llm_variants = self._llm_query_rewriting(query, intent, context)
            query_variants.extend(llm_variants)
        
        return QueryAnalysis(
            original_query=query,
            rewritten_queries=query_variants[:10],  # Top 10 variants
            intent=intent,
            entities=entities,
            financial_terms=financial_terms,
            time_references=time_refs,
            confidence=0.8  # Placeholder confidence score
        )
    
    def _preprocess_query(self, query: str) -> str:
        """Basic query preprocessing"""
        # Expand abbreviations
        processed = query.lower()
        for abbrev, expansion in self.abbreviations.items():
            processed = re.sub(r'\b' + abbrev + r'\b', expansion, processed)
        
        # Clean up formatting
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent based on patterns"""
        query_lower = query.lower()
        
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return QueryIntent.GENERAL
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query"""
        entities = []
        
        # Company names (simple pattern)
        company_pattern = r'\b[A-Z][a-zA-Z&\s]+(?:Inc|Corp|Ltd|LLC|Group|Company)\b'
        companies = re.findall(company_pattern, query)
        entities.extend(companies)
        
        # Currency amounts
        currency_pattern = r'\$[\d,]+(?:\.\d{2})?[KMB]?'
        amounts = re.findall(currency_pattern, query)
        entities.extend(amounts)
        
        # Percentages
        percent_pattern = r'\d+(?:\.\d+)?%'
        percentages = re.findall(percent_pattern, query)
        entities.extend(percentages)
        
        # Years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        entities.extend(years)
        
        return list(set(entities))
    
    def _extract_financial_terms(self, query: str) -> List[str]:
        """Extract financial terms from query"""
        terms = []
        query_lower = query.lower()
        
        # Check for financial terms and their synonyms
        for main_term, synonyms in self.financial_synonyms.items():
            if main_term in query_lower:
                terms.append(main_term)
            
            for synonym in synonyms:
                if synonym in query_lower:
                    terms.append(main_term)
                    break
        
        return list(set(terms))
    
    def _extract_time_references(self, query: str) -> List[str]:
        """Extract time references from query"""
        time_patterns = [
            r'q[1-4]', r'quarter [1-4]', r'fiscal year', r'fy\d{2,4}',
            r'\d{4}', r'last year', r'this year', r'ytd', r'year to date',
            r'monthly', r'quarterly', r'annually', r'yoy', r'qoq'
        ]
        
        time_refs = []
        query_lower = query.lower()
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            time_refs.extend(matches)
        
        return list(set(time_refs))
    
    def _generate_query_variants(self, query: str, intent: QueryIntent, 
                                entities: List[str], financial_terms: List[str]) -> List[str]:
        """Generate multiple query variants"""
        variants = []
        
        # Original query
        variants.append(query)
        
        # Intent-specific variants
        if intent == QueryIntent.DEFINITION:
            variants.append(f"definition of {' '.join(financial_terms)}")
            variants.append(f"what does {' '.join(financial_terms)} mean")
        
        elif intent == QueryIntent.CALCULATION:
            variants.append(f"calculate {' '.join(financial_terms)}")
            variants.append(f"formula for {' '.join(financial_terms)}")
        
        elif intent == QueryIntent.COMPARISON:
            variants.append(f"compare {' '.join(financial_terms)}")
            variants.append(f"difference between {' '.join(financial_terms)}")
        
        elif intent == QueryIntent.TREND_ANALYSIS:
            variants.append(f"trend analysis {' '.join(financial_terms)}")
            variants.append(f"historical {' '.join(financial_terms)}")
        
        # Synonym expansion
        for term in financial_terms:
            if term in self.financial_synonyms:
                for synonym in self.financial_synonyms[term]:
                    variant = query.replace(term, synonym)
                    if variant != query:
                        variants.append(variant)
        
        # Entity-focused variants
        if entities:
            variants.append(f"{' '.join(entities)} financial information")
        
        return list(set(variants))
    
    def _enhance_with_context(self, queries: List[str], context: Dict) -> List[str]:
        """Enhance queries with conversation context"""
        enhanced = []
        
        # Add conversation history context
        if 'conversation_history' in context:
            last_queries = context['conversation_history'][-2:]  # Last 2 queries
            for query in queries[:3]:  # Enhance top 3 queries
                for hist_query in last_queries:
                    enhanced.append(f"{hist_query} {query}")
        
        # Add document context
        if 'current_document' in context:
            doc_info = context['current_document']
            for query in queries[:3]:
                enhanced.append(f"{query} in {doc_info}")
        
        return enhanced
    
    def _llm_query_rewriting(self, original_query: str, intent: QueryIntent, 
                           context: Dict = None) -> List[str]:
        """Use LLM for advanced query rewriting"""
        if not self.openai_client:
            return []
        
        # Context information
        context_str = ""
        if context:
            if 'conversation_history' in context:
                context_str += f"Previous questions: {context['conversation_history'][-3:]}\n"
            if 'current_document' in context:
                context_str += f"Current document: {context['current_document']}\n"
        
        prompt = f"""
You are a financial document search expert. Rewrite the following query to improve retrieval from financial documents.

Query Intent: {intent.value}
Original Query: {original_query}
{context_str}

Generate 3-5 alternative phrasings that would help find relevant information in financial documents.
Focus on:
1. Financial terminology and synonyms
2. Different ways to express the same concept
3. More specific or detailed versions
4. Related concepts that might contain the answer

Return only the alternative queries, one per line:
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            variants = response.choices[0].message.content.strip().split('\n')
            return [v.strip() for v in variants if v.strip()]
            
        except Exception as e:
            print(f"Error in LLM query rewriting: {e}")
            return []


class ConversationalQueryRewriter(BaseQueryRewriter):
    """
    Specialized query rewriter for conversational contexts
    Handles pronoun resolution and context maintenance
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.base_rewriter = FinancialQueryRewriter(openai_api_key)
    
    def rewrite_query(self, query: str, context: Dict = None) -> QueryAnalysis:
        """
        Rewrite query considering conversational context
        """
        # Step 1: Resolve pronouns and references
        resolved_query = self._resolve_references(query, context)
        
        # Step 2: Use base rewriter
        analysis = self.base_rewriter.rewrite_query(resolved_query, context)
        
        # Step 3: Add conversational variants
        conv_variants = self._generate_conversational_variants(resolved_query, context)
        analysis.rewritten_queries.extend(conv_variants)
        
        # Update original query to resolved version
        analysis.original_query = resolved_query
        
        return analysis
    
    def _resolve_references(self, query: str, context: Dict = None) -> str:
        """Resolve pronouns and contextual references"""
        if not context or not self.openai_client:
            return query
        
        conversation_history = context.get('conversation_history', [])
        if not conversation_history:
            return query
        
        # Use LLM to resolve references
        history_str = '\n'.join(conversation_history[-5:])  # Last 5 exchanges
        
        prompt = f"""
Given the conversation history, resolve any pronouns or references in the current query.

Conversation History:
{history_str}

Current Query: {query}

Rewrite the current query to be self-contained by replacing pronouns (it, this, that, they) 
and vague references with specific terms from the conversation history.
If no references need resolution, return the query unchanged.

Resolved Query:
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            resolved = response.choices[0].message.content.strip()
            return resolved if resolved != query else query
            
        except Exception as e:
            print(f"Error resolving references: {e}")
            return query
    
    def _generate_conversational_variants(self, query: str, context: Dict = None) -> List[str]:
        """Generate variants considering conversational flow"""
        variants = []
        
        if not context:
            return variants
        
        conversation_history = context.get('conversation_history', [])
        if len(conversation_history) < 2:
            return variants
        
        # Add context from previous question
        prev_query = conversation_history[-1]
        variants.append(f"{prev_query} and {query}")
        variants.append(f"following up on {prev_query}, {query}")
        
        # Add clarification variants
        variants.append(f"more specifically about {query}")
        variants.append(f"detailed information on {query}")
        
        return variants


class AdaptiveQueryRewriter:
    """
    Adaptive query rewriter that selects the best strategy based on context
    """
    
    def __init__(self, openai_api_key: str = None):
        self.financial_rewriter = FinancialQueryRewriter(openai_api_key)
        self.conversational_rewriter = ConversationalQueryRewriter(openai_api_key)
    
    def rewrite_query(self, query: str, context: Dict = None) -> QueryAnalysis:
        """
        Select and apply the best rewriting strategy
        """
        # Determine if conversational context exists
        has_conversation_context = (
            context and 
            'conversation_history' in context and 
            len(context['conversation_history']) > 0
        )
        
        # Check for pronouns or references
        has_references = bool(re.search(r'\b(it|this|that|they|them|its)\b', query.lower()))
        
        if has_conversation_context and has_references:
            # Use conversational rewriter
            return self.conversational_rewriter.rewrite_query(query, context)
        else:
            # Use standard financial rewriter
            return self.financial_rewriter.rewrite_query(query, context)


# Usage example
if __name__ == "__main__":
    rewriter = AdaptiveQueryRewriter()
    
    # Example 1: Simple financial query
    analysis1 = rewriter.rewrite_query("What was the revenue growth in Q2?")
    print("Simple Query Analysis:")
    print(f"Intent: {analysis1.intent.value}")
    print(f"Entities: {analysis1.entities}")
    print(f"Financial Terms: {analysis1.financial_terms}")
    print(f"Rewritten Queries: {analysis1.rewritten_queries[:3]}")
    print()
    
    # Example 2: Conversational query with context
    context = {
        'conversation_history': [
            "What was Apple's revenue in 2023?",
            "How did it compare to the previous year?"
        ]
    }
    
    analysis2 = rewriter.rewrite_query("What about their profit margins?", context)
    print("Conversational Query Analysis:")
    print(f"Original: {analysis2.original_query}")
    print(f"Intent: {analysis2.intent.value}")
    print(f"Rewritten Queries: {analysis2.rewritten_queries[:3]}")
