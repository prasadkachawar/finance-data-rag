"""
LLM Factory and Response Generation for Finance RAG Pipeline
Handles multiple LLM providers and specialized financial prompt templates
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import openai
from enum import Enum
import json
import re


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model_used: str
    tokens_used: int
    cost_estimate: float
    confidence_score: float
    citations: List[str]
    metadata: Dict[str, Any]


class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM implementation with financial domain optimization
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
        # Pricing per 1K tokens (as of 2024)
        self.pricing = {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with financial domain awareness"""
        
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 2048)
        system_prompt = kwargs.get('system_prompt', self._get_default_system_prompt())
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Extract citations from response
            citations = self._extract_citations(content)
            
            # Calculate confidence score (simplified)
            confidence_score = self._calculate_confidence(content, tokens_used)
            
            # Estimate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            return LLMResponse(
                content=content,
                model_used=self.model,
                tokens_used=tokens_used,
                cost_estimate=cost,
                confidence_score=confidence_score,
                citations=citations,
                metadata={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'temperature': temperature
                }
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                model_used=self.model,
                tokens_used=0,
                cost_estimate=0.0,
                confidence_score=0.0,
                citations=[],
                metadata={}
            )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD"""
        if self.model not in self.pricing:
            return 0.0
        
        prices = self.pricing[self.model]
        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]
        
        return input_cost + output_cost
    
    def _get_default_system_prompt(self) -> str:
        """Default system prompt for financial documents"""
        return """You are a financial document expert. Your role is to answer questions based ONLY on the provided document context.

CRITICAL GUIDELINES:
1. Base your answer ONLY on the provided documents
2. Always include specific page numbers and section references when available
3. If information isn't in the documents, clearly state "This information is not available in the provided documents"
4. For financial calculations, show your work step-by-step
5. Explain financial terms when first mentioned
6. Use proper financial formatting for numbers (e.g., $1.2M, 15.5%)
7. Include confidence indicators when appropriate
8. Maintain professional tone suitable for financial analysis

RESPONSE FORMAT:
- Direct answer to the question
- Supporting evidence from documents with citations
- Any calculations with step-by-step explanation
- References in format: [Page X, Section Y.Z]"""
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation references from content"""
        citation_patterns = [
            r'\[Page\s+(\d+)(?:,\s*Section\s+([^\]]+))?\]',
            r'\(Page\s+(\d+)\)',
            r'page\s+(\d+)',
            r'section\s+([\d.]+)'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend([str(match) for match in matches])
        
        return list(set(citations))
    
    def _calculate_confidence(self, content: str, tokens_used: int) -> float:
        """Calculate confidence score based on response characteristics"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for citations
        if '[Page' in content or 'page' in content.lower():
            confidence += 0.2
        
        # Boost for financial data patterns
        if re.search(r'\$[\d,]+|\d+%', content):
            confidence += 0.1
        
        # Reduce confidence for uncertainty phrases
        uncertainty_phrases = ['might be', 'possibly', 'unclear', 'not sure', 'may be']
        for phrase in uncertainty_phrases:
            if phrase in content.lower():
                confidence -= 0.15
        
        # Adjust for response length (too short might be incomplete)
        if tokens_used < 50:
            confidence -= 0.1
        elif tokens_used > 500:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude implementation for financial analysis
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        
        # Pricing per 1K tokens
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic Claude"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            temperature = kwargs.get('temperature', 0.1)
            max_tokens = kwargs.get('max_tokens', 2048)
            system_prompt = kwargs.get('system_prompt', self._get_default_system_prompt())
            
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = message.content[0].text
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            citations = self._extract_citations(content)
            confidence_score = self._calculate_confidence(content, total_tokens)
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            return LLMResponse(
                content=content,
                model_used=self.model,
                tokens_used=total_tokens,
                cost_estimate=cost,
                confidence_score=confidence_score,
                citations=citations,
                metadata={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'temperature': temperature
                }
            )
            
        except Exception as e:
            print(f"Error with Anthropic: {e}")
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                model_used=self.model,
                tokens_used=0,
                cost_estimate=0.0,
                confidence_score=0.0,
                citations=[],
                metadata={}
            )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD"""
        if self.model not in self.pricing:
            return 0.0
        
        prices = self.pricing[self.model]
        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]
        
        return input_cost + output_cost
    
    def _get_default_system_prompt(self) -> str:
        """Default system prompt for financial documents"""
        return """You are a financial document expert specializing in analyzing complex financial reports, earnings calls, and market documents.

Your expertise includes:
- Financial statement analysis
- Market trend interpretation
- Risk assessment
- Regulatory compliance
- Investment analysis

Guidelines:
1. Answer based ONLY on provided document context
2. Include specific citations with page and section numbers
3. Explain financial concepts clearly
4. Show calculations step-by-step
5. Indicate confidence levels for complex interpretations
6. Use professional financial terminology appropriately
7. Highlight any limitations or assumptions in your analysis"""
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation references from content"""
        citation_patterns = [
            r'\[Page\s+(\d+)(?:,\s*Section\s+([^\]]+))?\]',
            r'\(Page\s+(\d+)\)',
            r'page\s+(\d+)',
            r'section\s+([\d.]+)'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend([str(match) for match in matches])
        
        return list(set(citations))
    
    def _calculate_confidence(self, content: str, tokens_used: int) -> float:
        """Calculate confidence score"""
        confidence = 0.6  # Base confidence for Claude
        
        # Similar logic to OpenAI implementation
        if '[Page' in content or 'page' in content.lower():
            confidence += 0.2
        
        if re.search(r'\$[\d,]+|\d+%', content):
            confidence += 0.1
        
        uncertainty_phrases = ['might be', 'possibly', 'unclear', 'may be']
        for phrase in uncertainty_phrases:
            if phrase in content.lower():
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))


class FinancialPromptTemplates:
    """
    Specialized prompt templates for financial document analysis
    """
    
    @staticmethod
    def create_rag_prompt(query: str, context_documents: List[Dict], 
                         conversation_history: List[str] = None) -> str:
        """
        Create comprehensive RAG prompt for financial queries
        """
        # Format context documents
        context_str = ""
        for i, doc in enumerate(context_documents, 1):
            context_str += f"\n--- Document {i} ---\n"
            context_str += f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}\n"
            context_str += f"Page: {doc.get('metadata', {}).get('page_number', 'N/A')}\n"
            context_str += f"Section: {doc.get('metadata', {}).get('section', 'N/A')}\n"
            context_str += f"Content: {doc.get('content', '')}\n"
        
        # Add conversation history if available
        history_str = ""
        if conversation_history:
            history_str = "\n--- Conversation History ---\n"
            for i, msg in enumerate(conversation_history[-5:], 1):  # Last 5 messages
                history_str += f"{i}. {msg}\n"
        
        prompt = f"""Based on the provided financial documents, please answer the following question.

{history_str}

--- Current Question ---
{query}

--- Available Documents ---
{context_str}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Uses only information from the provided documents
3. Includes specific citations in format [Page X, Section Y]
4. Shows calculations step-by-step if applicable
5. Explains any financial terms or concepts
6. Indicates if information is missing or unclear

Answer:"""
        
        return prompt
    
    @staticmethod
    def create_validation_prompt(query: str, answer: str, source_documents: List[Dict]) -> str:
        """
        Create prompt for validating answer accuracy
        """
        sources_str = "\n".join([
            f"- {doc.get('content', '')[:200]}..." 
            for doc in source_documents
        ])
        
        prompt = f"""Please validate the accuracy of this answer against the source documents.

Question: {query}

Answer to Validate:
{answer}

Source Documents:
{sources_str}

Please check:
1. Are all factual claims supported by the source documents?
2. Are any numerical values correctly stated?
3. Are the citations accurate?
4. Is anything stated that isn't in the sources?

Provide a validation score from 0-100 and explain any issues found."""
        
        return prompt
    
    @staticmethod
    def create_financial_calculation_prompt(query: str, data_context: str) -> str:
        """
        Create prompt specifically for financial calculations
        """
        prompt = f"""You are performing financial calculations based on the provided data.

Query: {query}

Financial Data:
{data_context}

Please:
1. Identify what calculation is being requested
2. Extract relevant numbers from the data
3. Show the calculation step-by-step
4. Provide the final result with appropriate units
5. Include any assumptions made
6. Cite the source data used

Calculation:"""
        
        return prompt


class LLMOrchestrator:
    """
    Orchestrates multiple LLMs with fallback and routing logic
    """
    
    def __init__(self, llm_configs: List[Dict[str, Any]]):
        self.llms = []
        self.load_llms(llm_configs)
        self.prompt_templates = FinancialPromptTemplates()
    
    def load_llms(self, configs: List[Dict[str, Any]]):
        """Load LLM instances from configurations"""
        for config in configs:
            provider = config.get('provider')
            
            if provider == 'openai':
                llm = OpenAILLM(
                    api_key=config['api_key'],
                    model=config.get('model', 'gpt-4-turbo')
                )
            elif provider == 'anthropic':
                llm = AnthropicLLM(
                    api_key=config['api_key'],
                    model=config.get('model', 'claude-3-sonnet-20240229')
                )
            else:
                continue
            
            self.llms.append({
                'llm': llm,
                'priority': config.get('priority', 1),
                'cost_tier': config.get('cost_tier', 'medium'),
                'capabilities': config.get('capabilities', [])
            })
        
        # Sort by priority
        self.llms.sort(key=lambda x: x['priority'])
    
    def generate_response(self, query: str, context_documents: List[Dict], 
                         conversation_history: List[str] = None,
                         preferred_model: str = None) -> LLMResponse:
        """
        Generate response with intelligent model selection and fallback
        """
        # Create RAG prompt
        prompt = self.prompt_templates.create_rag_prompt(
            query, context_documents, conversation_history
        )
        
        # Select LLM based on query complexity and preferences
        selected_llm = self._select_llm(query, preferred_model)
        
        # Generate response with fallback
        for llm_config in [selected_llm] + [l for l in self.llms if l != selected_llm]:
            try:
                response = llm_config['llm'].generate_response(prompt)
                
                # Validate response quality
                if self._validate_response_quality(response):
                    return response
                
            except Exception as e:
                print(f"LLM failed: {e}, trying next...")
                continue
        
        # Fallback response if all LLMs fail
        return LLMResponse(
            content="I apologize, but I'm unable to generate a response at this time.",
            model_used="fallback",
            tokens_used=0,
            cost_estimate=0.0,
            confidence_score=0.0,
            citations=[],
            metadata={}
        )
    
    def _select_llm(self, query: str, preferred_model: str = None) -> Dict:
        """Select best LLM for the query"""
        if preferred_model:
            for llm_config in self.llms:
                if preferred_model in str(llm_config['llm'].model):
                    return llm_config
        
        # Simple heuristic: use higher capability models for complex queries
        query_complexity = self._assess_query_complexity(query)
        
        if query_complexity > 0.7:  # High complexity
            # Prefer more capable models
            for llm_config in self.llms:
                if 'complex_reasoning' in llm_config['capabilities']:
                    return llm_config
        
        # Default to first (highest priority) LLM
        return self.llms[0] if self.llms else None
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0-1 score)"""
        complexity_indicators = [
            'calculate', 'compare', 'analyze', 'evaluate', 'explain',
            'trend', 'correlation', 'impact', 'forecast', 'risk'
        ]
        
        query_lower = query.lower()
        complexity_score = sum(
            1 for indicator in complexity_indicators 
            if indicator in query_lower
        ) / len(complexity_indicators)
        
        # Add length factor
        length_factor = min(len(query.split()) / 50, 1.0)  # Normalize by 50 words
        
        return (complexity_score + length_factor) / 2
    
    def _validate_response_quality(self, response: LLMResponse) -> bool:
        """Basic response quality validation"""
        if not response.content or len(response.content) < 50:
            return False
        
        if response.confidence_score < 0.3:
            return False
        
        if "error" in response.content.lower() and "generating" in response.content.lower():
            return False
        
        return True


# Configuration examples
OPENAI_CONFIG = {
    'provider': 'openai',
    'api_key': 'your-openai-key',
    'model': 'gpt-4-turbo',
    'priority': 1,
    'cost_tier': 'high',
    'capabilities': ['complex_reasoning', 'financial_analysis']
}

ANTHROPIC_CONFIG = {
    'provider': 'anthropic',
    'api_key': 'your-anthropic-key',
    'model': 'claude-3-sonnet-20240229',
    'priority': 2,
    'cost_tier': 'high',
    'capabilities': ['complex_reasoning', 'long_context']
}

COST_EFFECTIVE_CONFIG = {
    'provider': 'openai',
    'api_key': 'your-openai-key',
    'model': 'gpt-3.5-turbo',
    'priority': 3,
    'cost_tier': 'low',
    'capabilities': ['basic_qa']
}


# Usage example
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = LLMOrchestrator([
        OPENAI_CONFIG,
        ANTHROPIC_CONFIG,
        COST_EFFECTIVE_CONFIG
    ])
    
    # Example context documents
    context_documents = [
        {
            'content': 'The company reported revenue of $1.5M in Q2 2024, representing a 25% increase.',
            'metadata': {
                'source': 'Q2_2024_Report.pdf',
                'page_number': 3,
                'section': '2.1'
            }
        }
    ]
    
    # Generate response
    response = orchestrator.generate_response(
        query="What was the revenue growth in Q2 2024?",
        context_documents=context_documents
    )
    
    print(f"Response: {response.content}")
    print(f"Model: {response.model_used}")
    print(f"Confidence: {response.confidence_score}")
    print(f"Cost: ${response.cost_estimate:.4f}")
    print(f"Citations: {response.citations}")
