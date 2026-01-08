"""
FastAPI application for Finance RAG Pipeline
Complete end-to-end RAG system with chat interface
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import logging
import time
from datetime import datetime
import uuid

# Import our RAG components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_processing.chunking_strategies import AdaptiveChunker
from embeddings.hybrid_embeddings import HybridEmbeddingStrategy
from vector_db.vector_store_factory import VectorStoreFactory
from retrieval.query_rewriter import AdaptiveQueryRewriter
from retrieval.reranker import create_reranker
from llm.llm_factory import LLMOrchestrator
from evaluation.comprehensive_metrics import EndToEndEvaluator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Finance Document RAG API",
    description="Advanced RAG system for financial document analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class DocumentUpload(BaseModel):
    content: str
    metadata: Dict[str, Any]
    document_type: str = "financial_report"


class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    include_sources: bool = True
    max_sources: int = 5


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    confidence_score: float
    processing_time: float
    cost_estimate: float
    citations: List[str]


class EvaluationRequest(BaseModel):
    queries: List[str]
    expected_answers: List[str]
    evaluation_type: str = "comprehensive"


class SystemStatus(BaseModel):
    status: str
    components: Dict[str, str]
    metrics: Dict[str, Any]
    timestamp: datetime


# Global components (in production, use dependency injection)
rag_pipeline = None
conversation_history = {}  # In production, use Redis or database


async def initialize_rag_pipeline():
    """Initialize RAG pipeline components"""
    global rag_pipeline
    
    if rag_pipeline is not None:
        return rag_pipeline
    
    try:
        logger.info("Initializing RAG Pipeline...")
        
        # Configuration (should be loaded from config file)
        config = {
            'embeddings': {
                'openai': {
                    'model': 'text-embedding-3-large',
                    'api_key': os.getenv('OPENAI_API_KEY')
                },
                'cohere': {
                    'api_key': os.getenv('COHERE_API_KEY'),
                    'model': 'embed-english-v3.0'
                }
            },
            'vector_db': {
                'provider': 'qdrant',  # Default to Qdrant for local development
                'config': {
                    'host': 'localhost',
                    'port': 6333,
                    'collection_name': 'finance_docs'
                }
            },
            'reranker': {
                'type': 'cross_encoder',
                'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            },
            'llm': [
                {
                    'provider': 'openai',
                    'api_key': os.getenv('OPENAI_API_KEY'),
                    'model': 'gpt-4-turbo',
                    'priority': 1,
                    'cost_tier': 'high',
                    'capabilities': ['complex_reasoning', 'financial_analysis']
                }
            ]
        }
        
        # Initialize components
        chunker = AdaptiveChunker()
        embedding_strategy = HybridEmbeddingStrategy(config['embeddings'])
        
        # Vector store
        vector_store = VectorStoreFactory.create_vector_store(
            config['vector_db']['provider'], 
            config['vector_db']['config']
        )
        
        # Initialize vector store
        vector_store.create_index(dimension=3072, metric='cosine')
        
        # Query rewriter
        query_rewriter = AdaptiveQueryRewriter(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Reranker
        reranker = create_reranker(config['reranker'])
        
        # LLM orchestrator
        llm_orchestrator = LLMOrchestrator(config['llm'])
        
        # Evaluator
        evaluator = EndToEndEvaluator()
        
        rag_pipeline = {
            'chunker': chunker,
            'embedding_strategy': embedding_strategy,
            'vector_store': vector_store,
            'query_rewriter': query_rewriter,
            'reranker': reranker,
            'llm_orchestrator': llm_orchestrator,
            'evaluator': evaluator
        }
        
        logger.info("RAG Pipeline initialized successfully!")
        return rag_pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")


async def get_rag_pipeline():
    """Dependency to get RAG pipeline"""
    return await initialize_rag_pipeline()


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    await initialize_rag_pipeline()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "Finance Document RAG API", "status": "active"}


@app.get("/health", response_model=SystemStatus)
async def health_check(pipeline = Depends(get_rag_pipeline)):
    """Health check endpoint"""
    try:
        # Check component status
        components = {
            "vector_store": "active" if pipeline['vector_store'] else "inactive",
            "llm": "active" if pipeline['llm_orchestrator'] else "inactive",
            "embedding": "active" if pipeline['embedding_strategy'] else "inactive",
            "reranker": "active" if pipeline['reranker'] else "inactive"
        }
        
        # Get vector store stats
        vector_stats = pipeline['vector_store'].get_stats()
        
        return SystemStatus(
            status="healthy",
            components=components,
            metrics={
                "total_documents": vector_stats.get('total_vectors', 0),
                "vector_dimension": vector_stats.get('dimension', 0)
            },
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/documents/upload")
async def upload_document(
    document: DocumentUpload,
    background_tasks: BackgroundTasks,
    pipeline = Depends(get_rag_pipeline)
):
    """Upload and process a document"""
    try:
        document_id = str(uuid.uuid4())
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            document_id,
            document.content,
            document.metadata,
            pipeline
        )
        
        return {
            "document_id": document_id,
            "status": "processing",
            "message": "Document upload initiated"
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_background(
    document_id: str,
    content: str,
    metadata: Dict[str, Any],
    pipeline: Dict[str, Any]
):
    """Background task to process document"""
    try:
        logger.info(f"Processing document {document_id}")
        
        # Chunk document
        chunks = pipeline['chunker'].chunk_document(content, metadata)
        
        # Generate embeddings and store
        vectors_to_store = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{i}"
            
            # Generate embeddings
            embeddings = pipeline['embedding_strategy'].embed_document_chunk({
                'content': chunk.content,
                'type': chunk.chunk_type.value,
                'metadata': chunk.metadata
            })
            
            # Use primary embedding for storage
            primary_embedding = embeddings.get('primary') or embeddings.get('openai')
            
            if primary_embedding is not None:
                chunk_metadata = {
                    **chunk.metadata,
                    'document_id': document_id,
                    'chunk_id': chunk_id,
                    'content': chunk.content,
                    'chunk_type': chunk.chunk_type.value,
                    'page_number': chunk.page_number,
                    'section': chunk.section,
                    'chapter': chunk.chapter
                }
                
                vectors_to_store.append((chunk_id, primary_embedding, chunk_metadata))
        
        # Store in vector database
        success = pipeline['vector_store'].upsert_vectors(vectors_to_store)
        
        if success:
            logger.info(f"Document {document_id} processed successfully - {len(vectors_to_store)} chunks stored")
        else:
            logger.error(f"Failed to store document {document_id}")
            
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    pipeline = Depends(get_rag_pipeline)
):
    """Main chat endpoint for RAG queries"""
    start_time = time.time()
    
    try:
        # Get or create conversation ID
        conversation_id = message.conversation_id or str(uuid.uuid4())
        
        # Get conversation history
        history = conversation_history.get(conversation_id, [])
        
        # Step 1: Query rewriting
        query_analysis = pipeline['query_rewriter'].rewrite_query(
            message.message,
            context={'conversation_history': history}
        )
        
        logger.info(f"Query intent: {query_analysis.intent.value}")
        logger.info(f"Rewritten queries: {len(query_analysis.rewritten_queries)}")
        
        # Step 2: Vector search with multiple query variants
        all_candidates = []
        
        for query_variant in query_analysis.rewritten_queries[:3]:  # Top 3 variants
            # Generate query embedding
            query_embeddings = pipeline['embedding_strategy'].embed_query(
                query_variant, 
                query_type=query_analysis.intent.value
            )
            
            # Use primary embedding for search
            primary_embedding = query_embeddings.get('primary') or query_embeddings.get('openai')
            
            if primary_embedding is not None:
                # Search vector database
                results = pipeline['vector_store'].search(
                    query_vector=primary_embedding,
                    top_k=20,  # Get more candidates for reranking
                    filter_dict=None  # Could add filters based on query analysis
                )
                
                all_candidates.extend(results)
        
        # Remove duplicates and prepare for reranking
        unique_candidates = {}
        for candidate in all_candidates:
            if candidate.id not in unique_candidates:
                unique_candidates[candidate.id] = {
                    'id': candidate.id,
                    'content': candidate.content,
                    'score': candidate.score,
                    'metadata': candidate.metadata
                }
        
        candidates_list = list(unique_candidates.values())
        
        if not candidates_list:
            return ChatResponse(
                response="I couldn't find relevant information in the documents to answer your question.",
                sources=[],
                conversation_id=conversation_id,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                cost_estimate=0.0,
                citations=[]
            )
        
        # Step 3: Re-ranking
        reranked_results = pipeline['reranker'].rerank(
            query=message.message,
            documents=candidates_list,
            top_k=message.max_sources
        )
        
        logger.info(f"Reranked {len(reranked_results)} results")
        
        # Step 4: Generate response using LLM
        context_documents = [
            {
                'content': result.content,
                'metadata': result.metadata
            }
            for result in reranked_results
        ]
        
        llm_response = pipeline['llm_orchestrator'].generate_response(
            query=message.message,
            context_documents=context_documents,
            conversation_history=history
        )
        
        # Step 5: Prepare sources for response
        sources = []
        if message.include_sources:
            for result in reranked_results:
                source = {
                    'content': result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    'page_number': result.metadata.get('page_number', 'N/A'),
                    'section': result.metadata.get('section', 'N/A'),
                    'document_id': result.metadata.get('document_id', 'N/A'),
                    'relevance_score': result.final_score,
                    'rank': result.rank_position
                }
                sources.append(source)
        
        # Update conversation history
        history.append(f"User: {message.message}")
        history.append(f"Assistant: {llm_response.content}")
        conversation_history[conversation_id] = history[-10:]  # Keep last 10 messages
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=llm_response.content,
            sources=sources,
            conversation_id=conversation_id,
            confidence_score=llm_response.confidence_score,
            processing_time=processing_time,
            cost_estimate=llm_response.cost_estimate,
            citations=llm_response.citations
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(
    message: ChatMessage,
    pipeline = Depends(get_rag_pipeline)
):
    """Streaming chat endpoint"""
    async def generate_stream():
        try:
            # Get conversation context (simplified)
            conversation_id = message.conversation_id or str(uuid.uuid4())
            history = conversation_history.get(conversation_id, [])
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"
            
            # Query rewriting
            query_analysis = pipeline['query_rewriter'].rewrite_query(
                message.message,
                context={'conversation_history': history}
            )
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents...'})}\n\n"
            
            # Vector search (simplified for streaming)
            query_embeddings = pipeline['embedding_strategy'].embed_query(message.message)
            primary_embedding = query_embeddings.get('primary') or query_embeddings.get('openai')
            
            if primary_embedding is not None:
                results = pipeline['vector_store'].search(
                    query_vector=primary_embedding,
                    top_k=10
                )
                
                yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(results)} relevant documents'})}\n\n"
                
                # Prepare context and generate response
                context_documents = [
                    {'content': result.content, 'metadata': result.metadata}
                    for result in results
                ]
                
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
                
                # Generate response
                llm_response = pipeline['llm_orchestrator'].generate_response(
                    query=message.message,
                    context_documents=context_documents,
                    conversation_history=history
                )
                
                # Stream response
                yield f"data: {json.dumps({'type': 'response', 'content': llm_response.content})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'conversation_id': conversation_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")


@app.post("/evaluate")
async def evaluate_system(
    request: EvaluationRequest,
    pipeline = Depends(get_rag_pipeline)
):
    """Evaluate RAG system performance"""
    try:
        # Prepare evaluation data (simplified)
        evaluation_results = []
        
        for query, expected in zip(request.queries, request.expected_answers):
            # Run query through pipeline
            query_analysis = pipeline['query_rewriter'].rewrite_query(query)
            
            # Get embeddings and search
            query_embeddings = pipeline['embedding_strategy'].embed_query(query)
            primary_embedding = query_embeddings.get('primary') or query_embeddings.get('openai')
            
            if primary_embedding is not None:
                results = pipeline['vector_store'].search(
                    query_vector=primary_embedding,
                    top_k=10
                )
                
                # Generate response
                context_docs = [{'content': r.content, 'metadata': r.metadata} for r in results]
                llm_response = pipeline['llm_orchestrator'].generate_response(
                    query=query,
                    context_documents=context_docs
                )
                
                evaluation_results.append({
                    'query': query,
                    'expected': expected,
                    'actual': llm_response.content,
                    'confidence': llm_response.confidence_score,
                    'sources_found': len(results)
                })
        
        return {
            'evaluation_type': request.evaluation_type,
            'total_queries': len(request.queries),
            'results': evaluation_results,
            'summary': {
                'avg_confidence': sum(r['confidence'] for r in evaluation_results) / len(evaluation_results),
                'avg_sources': sum(r['sources_found'] for r in evaluation_results) / len(evaluation_results)
            }
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    history = conversation_history.get(conversation_id, [])
    return {
        'conversation_id': conversation_id,
        'messages': history,
        'message_count': len(history)
    }


@app.delete("/conversations/{conversation_id}")
async def clear_conversation_history(conversation_id: str):
    """Clear conversation history"""
    if conversation_id in conversation_history:
        del conversation_history[conversation_id]
        return {'status': 'cleared', 'conversation_id': conversation_id}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.get("/stats")
async def get_system_stats(pipeline = Depends(get_rag_pipeline)):
    """Get system statistics"""
    try:
        vector_stats = pipeline['vector_store'].get_stats()
        
        return {
            'vector_database': vector_stats,
            'active_conversations': len(conversation_history),
            'total_messages': sum(len(msgs) for msgs in conversation_history.values()),
            'system_status': 'operational'
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Configure for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
