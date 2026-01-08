# Finance RAG Pipeline - Architecture & Data Flow

## 🏗️ System Architecture Overview

```mermaid
graph TB
    subgraph "Data Ingestion Layer"
        A[Document Upload] --> B[Document Loader]
        B --> C[Format Detection]
        C --> D[Content Extraction]
    end
    
    subgraph "Processing Layer"
        D --> E[Chunking Strategies]
        E --> F[Multi-Modal Processing]
        F --> G[Metadata Extraction]
    end
    
    subgraph "Embedding Layer"
        G --> H[Hybrid Embeddings]
        H --> I[OpenAI Embeddings]
        H --> J[Cohere Embeddings]
        H --> K[BGE Embeddings]
    end
    
    subgraph "Storage Layer"
        I --> L[Vector Database]
        J --> L
        K --> L
        L --> M[Pinecone]
        L --> N[Weaviate]
        L --> O[Qdrant]
    end
    
    subgraph "Query Processing Layer"
        P[User Query] --> Q[Query Rewriter]
        Q --> R[Intent Classification]
        Q --> S[Entity Extraction]
        Q --> T[Context Enhancement]
    end
    
    subgraph "Retrieval Layer"
        T --> U[Vector Search]
        U --> V[Hybrid Retrieval]
        V --> W[Dense Search]
        V --> X[Sparse Search]
    end
    
    subgraph "Re-ranking Layer"
        W --> Y[Multi-Stage Reranker]
        X --> Y
        Y --> Z[Cohere Rerank]
        Y --> AA[Cross-Encoder]
        Y --> BB[LLM Rerank]
    end
    
    subgraph "Generation Layer"
        BB --> CC[LLM Orchestrator]
        CC --> DD[OpenAI GPT-4]
        CC --> EE[Anthropic Claude]
        CC --> FF[Model Router]
    end
    
    subgraph "Response Layer"
        FF --> GG[Response Generator]
        GG --> HH[Streaming Response]
        HH --> II[Final Response]
    end
    
    subgraph "Evaluation Layer"
        II --> JJ[Metrics Collection]
        JJ --> KK[Retrieval Metrics]
        JJ --> LL[Generation Metrics]
        JJ --> MM[Domain Metrics]
    end
```

## 📊 Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant QR as Query Rewriter
    participant VDB as Vector Database
    participant RR as Reranker
    participant LLM as LLM Orchestrator
    participant Eval as Evaluator
    
    User->>API: Submit Query
    API->>QR: Process Query
    QR->>QR: Intent Classification
    QR->>QR: Entity Extraction
    QR->>QR: Context Enhancement
    QR->>VDB: Enhanced Query
    
    VDB->>VDB: Dense Vector Search
    VDB->>VDB: Sparse BM25 Search
    VDB->>RR: Raw Results
    
    RR->>RR: Cohere Reranking
    RR->>RR: Cross-Encoder Scoring
    RR->>RR: LLM-based Reranking
    RR->>LLM: Top-K Results
    
    LLM->>LLM: Model Selection
    LLM->>LLM: Prompt Engineering
    LLM->>LLM: Response Generation
    LLM->>API: Generated Response
    
    API->>Eval: Log Interaction
    API->>User: Stream Response
    
    Eval->>Eval: Calculate Metrics
    Eval->>Eval: Update Performance
```

## 🗂️ Directory Structure & Components

```
finace_data_rag/
│
├── 📁 src/                          # Core application code
│   ├── 📁 api/                      # FastAPI REST API
│   │   └── main.py                  # API endpoints & routing
│   │
│   ├── 📁 data_processing/          # Document processing pipeline
│   │   ├── document_loader.py       # Multi-format document loading
│   │   └── chunking_strategies.py   # Advanced chunking algorithms
│   │
│   ├── 📁 embeddings/               # Embedding generation
│   │   └── hybrid_embeddings.py    # Multi-provider embeddings
│   │
│   ├── 📁 vector_db/               # Vector database abstraction
│   │   └── vector_store_factory.py # Unified DB interface
│   │
│   ├── 📁 retrieval/               # Query processing & retrieval
│   │   ├── query_rewriter.py       # Intelligent query enhancement
│   │   └── reranker.py             # Multi-stage result reranking
│   │
│   ├── 📁 llm/                     # Language model integration
│   │   └── llm_factory.py          # Multi-LLM orchestration
│   │
│   └── 📁 evaluation/              # Performance evaluation
│       └── comprehensive_metrics.py # End-to-end metrics
│
├── 📁 config/                      # Configuration management
│   └── settings.yaml               # Application settings
│
├── 📁 docker/                      # Containerization
│   ├── Dockerfile                  # Application container
│   └── docker-compose.yml          # Multi-service setup
│
├── 📁 docs/                        # Documentation
│   ├── README.md                   # Project overview
│   ├── ARCHITECTURE.md             # This file
│   ├── RAG_COMPONENT_GUIDE.md      # Component details
│   └── IMPLEMENTATION_SUMMARY.md   # Implementation notes
│
└── 📁 cli/                         # Command-line tools
    └── cli.py                      # Document processing CLI
```

## 🔄 Component Interaction Flow

### 1. Document Processing Pipeline
```mermaid
flowchart LR
    A[PDF/DOCX Upload] --> B[Document Loader]
    B --> C[Content Extraction]
    C --> D[Semantic Chunking]
    D --> E[Table Processing]
    E --> F[Image Extraction]
    F --> G[Metadata Tagging]
    G --> H[Embedding Generation]
    H --> I[Vector Storage]
```

### 2. Query Processing Pipeline
```mermaid
flowchart LR
    A[User Query] --> B[Query Analysis]
    B --> C[Intent Detection]
    C --> D[Entity Recognition]
    D --> E[Context Injection]
    E --> F[Query Expansion]
    F --> G[Financial Term Enhancement]
```

### 3. Retrieval & Ranking Pipeline
```mermaid
flowchart LR
    A[Enhanced Query] --> B[Vector Search]
    B --> C[Candidate Results]
    C --> D[First-Stage Rerank]
    D --> E[Cross-Encoder Score]
    E --> F[LLM Relevance Check]
    F --> G[Final Result Set]
```

### 4. Generation Pipeline
```mermaid
flowchart LR
    A[Retrieved Context] --> B[Model Selection]
    B --> C[Prompt Engineering]
    C --> D[Response Generation]
    D --> E[Quality Check]
    E --> F[Streaming Response]
```

## 🎯 Key Design Principles

### 1. **Modularity**
- Each component is independently replaceable
- Clear interfaces between layers
- Plugin-based architecture for extensibility

### 2. **Scalability**
- Horizontal scaling through Docker containers
- Async processing with FastAPI
- Distributed vector database support

### 3. **Observability**
- Comprehensive logging at each stage
- Performance metrics collection
- Real-time monitoring capabilities

### 4. **Reliability**
- Graceful degradation with fallback models
- Retry mechanisms for external APIs
- Circuit breaker patterns for resilience

## 📈 Performance Characteristics

### Throughput Expectations
- **Document Processing**: 100 pages/minute
- **Query Response**: < 2 seconds (95th percentile)
- **Concurrent Users**: 100+ with proper scaling
- **Embedding Generation**: 1000 chunks/minute

### Resource Requirements
- **Memory**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 50GB for vector indices, 100GB for document storage
- **Network**: Stable internet for API calls

## 🔧 Configuration Flow

### Environment-Based Configuration
```yaml
# Development
- Local vector DB (Qdrant)
- OpenAI embeddings only
- Single LLM model
- Detailed logging

# Production
- Cloud vector DB (Pinecone)
- Multi-provider embeddings
- LLM load balancing
- Optimized logging
```

### Feature Toggles
- **Embedding Providers**: Enable/disable providers
- **Reranking Stages**: Configure reranking pipeline
- **LLM Models**: Select available models
- **Evaluation**: Toggle metrics collection

## 🚀 Deployment Architecture

### Development Environment
```
Developer Machine
├── Local Qdrant (Docker)
├── Redis Cache
├── FastAPI (uvicorn)
└── CLI Tools
```

### Production Environment
```
Cloud Infrastructure
├── Load Balancer
├── API Containers (3x)
├── Vector Database (Pinecone)
├── Redis Cluster
├── Monitoring Stack
└── Document Storage (S3)
```

## 🔍 Monitoring & Observability

### Metrics Tracked
- **Latency**: Query processing time
- **Throughput**: Requests per second
- **Accuracy**: Retrieval and generation quality
- **Cost**: API usage and compute costs

### Alerting
- High error rates
- Increased latency
- API quota limits
- Storage capacity

This architecture provides a robust, scalable foundation for processing and querying large volumes of financial documents while maintaining high performance and accuracy standards.
