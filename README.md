# Finance Document RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system for processing and querying 10,000 pages of financial documents with graphs, technical terms, and complex financial data.

## 🏗️ Architecture Overview

```
User Query → Query Rewriting → Vector DB Search → Re-ranking → LLM Generation → Response + References
```

## 📋 Project Structure

```
finace_data_rag/
├── src/
│   ├── data_processing/
│   │   ├── document_loader.py
│   │   ├── chunking_strategies.py
│   │   ├── multimodal_processor.py
│   │   └── metadata_extractor.py
│   ├── embeddings/
│   │   ├── text_embeddings.py
│   │   ├── image_embeddings.py
│   │   └── hybrid_embeddings.py
│   ├── vector_db/
│   │   ├── pinecone_client.py
│   │   ├── weaviate_client.py
│   │   ├── qdrant_client.py
│   │   └── vector_store_factory.py
│   ├── retrieval/
│   │   ├── query_rewriter.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── context_enhancer.py
│   ├── llm/
│   │   ├── llm_factory.py
│   │   ├── prompt_templates.py
│   │   └── response_formatter.py
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   └── end_to_end_metrics.py
│   └── api/
│       ├── main.py
│       ├── chat_endpoint.py
│       └── health_check.py
├── config/
│   ├── settings.yaml
│   ├── model_configs.yaml
│   └── vector_db_configs.yaml
├── data/
│   ├── raw_documents/
│   ├── processed_chunks/
│   └── metadata/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_analysis.ipynb
│   ├── 03_embedding_comparison.ipynb
│   ├── 04_retrieval_evaluation.ipynb
│   └── 05_end_to_end_testing.ipynb
├── tests/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings**
   - Update `config/settings.yaml` with your API keys
   - Configure vector database settings

3. **Process Documents**
   ```bash
   python src/data_processing/document_loader.py
   ```

4. **Run API Server**
   ```bash
   python src/api/main.py
   ```

## 📊 Key Features

- **Multi-modal Processing**: Handles text, tables, and images
- **Advanced Chunking**: Semantic and hybrid chunking strategies
- **Production Vector DB**: Scalable vector database solutions
- **Query Optimization**: Query rewriting and context enhancement
- **Re-ranking**: Advanced re-ranking for improved relevance
- **Comprehensive Evaluation**: Metrics at every pipeline stage
- **Reference Tracking**: Page/chapter/section number references

## 🔧 Configuration

All configurations are managed through YAML files in the `config/` directory:
- Model selections and parameters
- Vector database configurations
- Embedding strategies
- Evaluation metrics

## 📈 Monitoring & Evaluation

Built-in evaluation metrics for:
- Retrieval accuracy (MRR, NDCG, Recall@K)
- Generation quality (BLEU, ROUGE, BERTScore)
- End-to-end performance
- Latency and throughput

## 🔗 Integration

- FastAPI for REST endpoints
- Docker support for deployment
- Monitoring with logging and metrics
- Extensible architecture for new components
