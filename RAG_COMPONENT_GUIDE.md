# RAG Pipeline Component Guide

## 1. Chunking Strategies

### 1.1 Recommended Chunking Strategies for Finance Documents

#### **Semantic Chunking** (Primary Recommendation)
- **What**: Splits documents based on semantic boundaries rather than fixed sizes
- **Why for Finance**: Preserves financial concepts, calculations, and context
- **Implementation**: Use sentence transformers to identify semantic breaks
```python
# Example semantic chunking
from langchain.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)
```

#### **Hierarchical Chunking** (Recommended for Technical Docs)
- **What**: Creates parent-child relationships between chunks
- **Why**: Maintains document structure (chapters, sections, subsections)
- **Implementation**: Extract document hierarchy first, then chunk within structure

#### **Multi-Modal Chunking**
- **What**: Separate processing for text, tables, images, and graphs
- **Why**: Finance docs have complex visual elements
- **Strategy**: 
  - Text chunks: 512-1024 tokens
  - Table chunks: Complete tables as single chunks
  - Image chunks: Image + surrounding context

### 1.2 Embedding Methods

#### **Text Embeddings**
1. **OpenAI text-embedding-3-large** (Recommended)
   - **Dimensions**: 3072 (can be reduced to 1536, 1024)
   - **Context Length**: 8191 tokens
   - **Strengths**: Excellent for financial terminology
   - **Cost**: $0.00013 per 1K tokens

2. **OpenAI text-embedding-3-small**
   - **Dimensions**: 1536
   - **Context Length**: 8191 tokens
   - **Cost**: $0.00002 per 1K tokens
   - **Use**: Cost-effective alternative

3. **Cohere embed-english-v3.0**
   - **Dimensions**: 1024
   - **Strengths**: Good for domain-specific content
   - **Features**: Built-in compression

#### **Multi-Modal Embeddings**
1. **OpenAI CLIP** (for images and graphs)
2. **Google Universal Sentence Encoder** (text + image)
3. **BGE-M3** (multi-lingual, multi-modal)

### 1.3 OpenAI Embedding Types

| Model | Dimensions | Context Length | Price per 1K tokens | Use Case |
|-------|------------|----------------|-------------------|----------|
| text-embedding-3-large | 3072 | 8191 | $0.00013 | Best performance |
| text-embedding-3-small | 1536 | 8191 | $0.00002 | Cost-effective |
| text-embedding-ada-002 | 1536 | 8191 | $0.00010 | Legacy |

## 2. Vector Database Selection

### 2.1 Production Vector Databases

#### **Pinecone** (Recommended for Production)
- **Why**: Fully managed, excellent performance, auto-scaling
- **Best for**: High-traffic applications, minimal ops overhead
- **Pricing**: Usage-based, predictable costs
- **Features**: Real-time updates, metadata filtering, namespaces

#### **Weaviate** 
- **Why**: Open-source with managed cloud, multi-modal support
- **Best for**: Complex data types, hybrid search
- **Features**: GraphQL API, built-in vectorization

#### **Qdrant**
- **Why**: High performance, payload filtering, on-premise option
- **Best for**: Privacy-sensitive applications, custom deployments
- **Features**: Advanced filtering, quantization

### 2.2 Vector Database Comparison

| Database | Managed | Open Source | Multi-Modal | Hybrid Search | Scalability |
|----------|---------|-------------|-------------|---------------|-------------|
| Pinecone | ✅ | ❌ | ❌ | ✅ | Excellent |
| Weaviate | ✅ | ✅ | ✅ | ✅ | Very Good |
| Qdrant | ✅ | ✅ | ❌ | ✅ | Excellent |
| Chroma | Partial | ✅ | ❌ | ❌ | Good |
| Milvus | ✅ | ✅ | ❌ | ❌ | Excellent |

### 2.3 Selection Strategy

Choose based on:
1. **Scale Requirements**: Pinecone for massive scale
2. **Budget**: Qdrant self-hosted for cost control
3. **Multi-modal needs**: Weaviate for images/graphs
4. **Compliance**: On-premise Qdrant for data sovereignty
5. **Team expertise**: Managed solutions for smaller teams

## 3. Query Rewriting

### 3.1 Query Rewriting Process
```python
def rewrite_query(original_query: str, conversation_history: List[str]) -> str:
    """
    1. Analyze user intent
    2. Expand financial abbreviations
    3. Add context from conversation history
    4. Generate multiple query variations
    """
    pass
```

### 3.2 Techniques
- **Intent Classification**: Determine query type (definition, calculation, comparison)
- **Entity Recognition**: Extract financial terms, companies, metrics
- **Query Expansion**: Add synonyms and related terms
- **Contextual Enhancement**: Use conversation history

## 4. LLM for Technical Queries

### 4.1 Recommended LLMs

#### **OpenAI GPT-4 Turbo** (Primary Recommendation)
- **Why**: Best understanding of financial concepts
- **Context**: 128K tokens
- **Strengths**: Mathematical reasoning, financial analysis

#### **Claude 3 Sonnet/Opus** (Alternative)
- **Why**: Excellent analytical capabilities
- **Context**: 200K tokens
- **Strengths**: Long document analysis

#### **Mixtral 8x7B** (Cost-effective)
- **Why**: Good performance at lower cost
- **Deployment**: Can be self-hosted

## 5. Vector Database Search Methods

### 5.1 Hybrid Search Strategy
```python
def hybrid_search(query: str, top_k: int = 20) -> List[Document]:
    # 1. Dense vector search (semantic similarity)
    vector_results = vector_db.similarity_search(query_embedding, top_k)
    
    # 2. Sparse search (keyword matching)
    keyword_results = elasticsearch.search(query, top_k)
    
    # 3. Combine and rerank results
    combined_results = combine_results(vector_results, keyword_results)
    
    return combined_results
```

### 5.2 Advanced Retrieval Techniques

#### **Multi-Vector Retrieval**
- Different embeddings for different content types
- Separate vectors for summaries and detailed content

#### **Metadata Filtering**
- Filter by document type, date, section
- Pre-filter before vector search for better performance

#### **Contextual Compression**
- Compress retrieved documents to most relevant parts
- Remove redundant information

## 6. Evaluating Vector DB Hits

### 6.1 Evaluation Metrics

#### **Retrieval Metrics**
- **MRR (Mean Reciprocal Rank)**: Quality of top result
- **NDCG@K**: Relevance-weighted ranking quality
- **Recall@K**: Fraction of relevant documents retrieved
- **Precision@K**: Fraction of retrieved documents that are relevant

#### **Implementation Example**
```python
def evaluate_retrieval(queries, ground_truth, retrieved_docs):
    mrr_scores = []
    ndcg_scores = []
    
    for query, truth, retrieved in zip(queries, ground_truth, retrieved_docs):
        # Calculate MRR
        mrr = calculate_mrr(truth, retrieved)
        mrr_scores.append(mrr)
        
        # Calculate NDCG
        ndcg = calculate_ndcg(truth, retrieved, k=10)
        ndcg_scores.append(ndcg)
    
    return {
        'mrr': np.mean(mrr_scores),
        'ndcg@10': np.mean(ndcg_scores)
    }
```

### 6.2 Improvement Strategies
- **Embedding Fine-tuning**: Train on domain-specific data
- **Query Expansion**: Add financial synonyms and abbreviations
- **Negative Sampling**: Train to avoid irrelevant results
- **Feedback Loop**: Use user interactions to improve rankings

## 7. Re-ranking

### 7.1 Re-ranking Models

#### **Cohere Rerank v3** (Recommended)
- **Why**: Specifically designed for re-ranking tasks
- **Input**: Query + multiple documents
- **Output**: Relevance scores for each document

#### **BGE Reranker** (Open Source)
- **Model**: BAAI/bge-reranker-large
- **Advantage**: Can be self-hosted
- **Performance**: Competitive with commercial solutions

#### **Cross-Encoder Models**
- **Architecture**: BERT-based cross-encoders
- **Examples**: ms-marco-MiniLM-L-6-v2
- **Training**: Fine-tune on financial Q&A pairs

### 7.2 Re-ranking Process
```python
def rerank_documents(query: str, documents: List[str], top_k: int = 5):
    # 1. Get initial candidates from vector search
    candidates = vector_search(query, top_k=20)
    
    # 2. Re-rank using cross-encoder
    reranked = reranker.rank(query, candidates)
    
    # 3. Return top-k after re-ranking
    return reranked[:top_k]
```

## 8. Evaluation of Re-ranking

### 8.1 Re-ranking Metrics
- **Improvement in MRR**: Compare before/after re-ranking
- **NDCG Lift**: Improvement in ranking quality
- **Click-through Rate**: In production, measure user engagement
- **Manual Evaluation**: Human assessment of relevance

### 8.2 A/B Testing Framework
```python
def evaluate_reranker(test_queries, baseline_retriever, reranked_retriever):
    baseline_results = baseline_retriever.search_batch(test_queries)
    reranked_results = reranked_retriever.search_batch(test_queries)
    
    baseline_mrr = calculate_mrr(baseline_results, ground_truth)
    reranked_mrr = calculate_mrr(reranked_results, ground_truth)
    
    improvement = (reranked_mrr - baseline_mrr) / baseline_mrr * 100
    return f"MRR improvement: {improvement:.2f}%"
```

## 9. Context Transfer to Main LLM

### 9.1 Context Assembly Strategy
```python
def assemble_context(
    original_query: str,
    rewritten_query: str,
    conversation_history: List[str],
    reranked_documents: List[Document]
) -> str:
    
    context_parts = [
        f"Original Query: {original_query}",
        f"Processed Query: {rewritten_query}",
        f"Conversation History: {format_history(conversation_history)}",
        f"Relevant Documents:\n{format_documents(reranked_documents)}"
    ]
    
    return "\n\n".join(context_parts)
```

### 9.2 Prompt Engineering
```python
FINANCE_RAG_PROMPT = """
You are a financial document expert. Use the provided documents to answer the user's question.

IMPORTANT GUIDELINES:
1. Base your answer ONLY on the provided documents
2. Include specific page numbers and section references
3. If the information isn't in the documents, say so clearly
4. For calculations, show your work step by step
5. Explain financial terms when first mentioned

User Query: {query}

Relevant Documents:
{context}

Provide a comprehensive answer with references:
"""
```

## 10. Main LLM Selection

### 10.1 Recommended LLMs for Financial Data

#### **GPT-4 Turbo** (Primary Choice)
- **Strengths**: 
  - Excellent financial domain knowledge
  - Strong mathematical reasoning
  - Good at following complex instructions
- **Context**: 128K tokens
- **Cost**: Higher but worth it for accuracy

#### **Claude 3 Opus** (Premium Alternative)
- **Strengths**:
  - Superior analytical capabilities
  - Excellent at handling long documents
  - Good at nuanced financial interpretations
- **Context**: 200K tokens
- **Consideration**: May be more expensive

#### **GPT-4o** (Balanced Choice)
- **Strengths**:
  - Good performance with multi-modal input
  - Fast response times
  - Cost-effective
- **Best for**: Production deployments with budget constraints

### 10.2 Model Comparison for Finance

| Model | Financial Knowledge | Math Reasoning | Context Window | Cost | Speed |
|-------|-------------------|----------------|----------------|------|-------|
| GPT-4 Turbo | Excellent | Excellent | 128K | High | Medium |
| Claude 3 Opus | Excellent | Very Good | 200K | Very High | Medium |
| GPT-4o | Very Good | Good | 128K | Medium | Fast |
| Gemini Pro | Good | Good | 1M | Medium | Fast |

## 11. Answer Correctness Validation

### 11.1 Automated Validation Methods

#### **Retrieval Validation**
- Verify claims against source documents
- Check for hallucinations by comparing with retrieved content
- Validate numerical calculations

#### **Consistency Checking**
```python
def validate_answer(question: str, answer: str, source_docs: List[str]) -> Dict:
    validation_results = {
        'factual_accuracy': check_facts_against_sources(answer, source_docs),
        'calculation_accuracy': verify_calculations(answer),
        'reference_validity': validate_references(answer, source_docs),
        'hallucination_score': detect_hallucinations(answer, source_docs)
    }
    return validation_results
```

### 11.2 Human-in-the-Loop Validation
- Expert review of complex financial interpretations
- Feedback collection from end users
- Regular audit of answers for accuracy

## 12. Evaluation Metrics by Stage

### 12.1 Document Processing Stage
- **Chunk Quality**: Manual review of chunk boundaries
- **Information Preservation**: Ensure no loss of critical information
- **Processing Speed**: Documents per minute

### 12.2 Embedding Stage
- **Semantic Similarity**: Cosine similarity between related concepts
- **Dimension Reduction Impact**: Performance with different dimensions
- **Embedding Time**: Vectors per second

### 12.3 Retrieval Stage
- **Recall@K**: Fraction of relevant documents retrieved
- **Precision@K**: Fraction of retrieved documents that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain

### 12.4 Re-ranking Stage
- **Ranking Improvement**: NDCG improvement over base retrieval
- **Relevance Correlation**: Agreement with human rankings
- **Latency**: Additional time for re-ranking

### 12.5 Generation Stage
- **ROUGE Scores**: Overlap with reference answers
- **BERTScore**: Semantic similarity with gold standard
- **Factual Accuracy**: Percentage of correct claims
- **Completeness**: Coverage of query aspects

### 12.6 End-to-End Metrics
- **User Satisfaction**: Rating of answer quality
- **Task Completion**: Success rate for user goals
- **Response Time**: Total pipeline latency
- **Reference Quality**: Accuracy of page/section citations

## 13. Reference Tracking System

### 13.1 Metadata Extraction
```python
def extract_document_metadata(document_path: str) -> Dict:
    """
    Extract metadata from finance documents
    """
    metadata = {
        'page_number': extract_page_number(document_path),
        'chapter': extract_chapter_info(document_path),
        'section': extract_section_info(document_path),
        'document_type': classify_document_type(document_path),
        'table_id': extract_table_references(document_path),
        'figure_id': extract_figure_references(document_path)
    }
    return metadata
```

### 13.2 Citation Generation
```python
def generate_citations(answer: str, source_chunks: List[Chunk]) -> str:
    """
    Add proper citations to the generated answer
    """
    citations = []
    for chunk in source_chunks:
        citation = f"[Page {chunk.page}, Section {chunk.section}]"
        citations.append(citation)
    
    # Insert citations at appropriate locations in the answer
    answer_with_citations = insert_citations(answer, citations)
    return answer_with_citations
```

### 13.3 Reference Formats
- **Page References**: "According to page 45 of the Annual Report..."
- **Section References**: "As stated in Section 3.2 (Risk Management)..."
- **Table References**: "Based on Table 4.1 (Financial Ratios)..."
- **Figure References**: "The trend shown in Figure 2.3 indicates..."

## 14. Implementation Priority

### Phase 1: Foundation (Weeks 1-2)
1. Document processing pipeline
2. Basic chunking and embedding
3. Vector database setup
4. Simple retrieval system

### Phase 2: Enhancement (Weeks 3-4)
1. Query rewriting
2. Re-ranking implementation
3. LLM integration
4. Reference tracking

### Phase 3: Optimization (Weeks 5-6)
1. Evaluation framework
2. Performance tuning
3. User interface
4. Production deployment

### Phase 4: Advanced Features (Weeks 7-8)
1. Multi-modal processing
2. Advanced evaluation metrics
3. Monitoring and logging
4. Continuous improvement loop
