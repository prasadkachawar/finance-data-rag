#!/usr/bin/env python3
"""
Command-line interface for Finance RAG Pipeline
Provides tools for document processing, evaluation, and system management
"""

import click
import os
import json
import yaml
import asyncio
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='config/settings.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Finance RAG Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    
    # Load configuration
    try:
        with open(config, 'r') as f:
            ctx.obj['config'] = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        ctx.obj['config'] = {}


@cli.group()
def documents():
    """Document management commands"""
    pass


@documents.command('process')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/processed', help='Output directory')
@click.option('--format', '-f', type=click.Choice(['pdf', 'docx', 'txt']), help='Document format')
@click.option('--batch-size', '-b', default=10, help='Processing batch size')
@click.pass_context
def process_documents(ctx, input_path, output, format, batch_size):
    """Process documents for RAG pipeline"""
    try:
        from src.data_processing.document_loader import DocumentLoader
        from src.data_processing.chunking_strategies import AdaptiveChunker
        
        click.echo(f"Processing documents from: {input_path}")
        
        # Initialize components
        loader = DocumentLoader()
        chunker = AdaptiveChunker()
        
        input_path = Path(input_path)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process files
        if input_path.is_file():
            files = [input_path]
        else:
            # Find all supported files
            extensions = ['.pdf', '.docx', '.txt'] if not format else [f'.{format}']
            files = []
            for ext in extensions:
                files.extend(input_path.glob(f'**/*{ext}'))
        
        click.echo(f"Found {len(files)} files to process")
        
        processed_count = 0
        for file_path in files:
            try:
                click.echo(f"Processing: {file_path.name}")
                
                # Extract content
                content = loader.extract_text(str(file_path))
                
                if not content:
                    click.echo(f"  Warning: No content extracted from {file_path.name}")
                    continue
                
                # Generate metadata
                metadata = {
                    'source_file': str(file_path),
                    'file_type': file_path.suffix.lower(),
                    'processed_date': datetime.now().isoformat(),
                    'file_size': file_path.stat().st_size
                }
                
                # Chunk document
                chunks = chunker.chunk_document(content, metadata)
                
                # Save chunks
                output_file = output_path / f"{file_path.stem}_chunks.json"
                
                chunk_data = []
                for chunk in chunks:
                    chunk_data.append({
                        'content': chunk.content,
                        'type': chunk.chunk_type.value,
                        'metadata': chunk.metadata,
                        'page_number': chunk.page_number,
                        'section': chunk.section,
                        'chapter': chunk.chapter,
                        'token_count': chunk.token_count
                    })
                
                with open(output_file, 'w') as f:
                    json.dump(chunk_data, f, indent=2)
                
                click.echo(f"  ✓ Created {len(chunks)} chunks -> {output_file}")
                processed_count += 1
                
            except Exception as e:
                click.echo(f"  ✗ Error processing {file_path.name}: {e}")
        
        click.echo(f"\nCompleted processing {processed_count} documents")
        
    except Exception as e:
        click.echo(f"Error: {e}")
        raise click.ClickException(str(e))


@documents.command('upload')
@click.argument('chunk_files', nargs=-1, type=click.Path(exists=True))
@click.option('--endpoint', default='http://localhost:8000', help='API endpoint')
@click.pass_context
def upload_documents(ctx, chunk_files, endpoint):
    """Upload processed documents to vector database"""
    try:
        import requests
        
        if not chunk_files:
            chunk_files = list(Path('data/processed').glob('*_chunks.json'))
        
        click.echo(f"Uploading {len(chunk_files)} chunk files to {endpoint}")
        
        total_chunks = 0
        for chunk_file in chunk_files:
            click.echo(f"Uploading: {Path(chunk_file).name}")
            
            with open(chunk_file, 'r') as f:
                chunks_data = json.load(f)
            
            for chunk in chunks_data:
                try:
                    response = requests.post(
                        f"{endpoint}/documents/upload",
                        json={
                            'content': chunk['content'],
                            'metadata': chunk['metadata'],
                            'document_type': 'financial_report'
                        }
                    )
                    
                    if response.status_code == 200:
                        total_chunks += 1
                    else:
                        click.echo(f"  Warning: Upload failed with status {response.status_code}")
                
                except Exception as e:
                    click.echo(f"  Error uploading chunk: {e}")
        
        click.echo(f"Successfully uploaded {total_chunks} chunks")
        
    except Exception as e:
        click.echo(f"Error: {e}")
        raise click.ClickException(str(e))


@cli.group()
def evaluate():
    """Evaluation commands"""
    pass


@evaluate.command('retrieval')
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='evaluation_results.json', help='Output file')
@click.option('--top-k', default=10, help='Top-K for evaluation')
@click.pass_context
def evaluate_retrieval(ctx, test_file, output, top_k):
    """Evaluate retrieval performance"""
    try:
        from src.evaluation.comprehensive_metrics import EndToEndEvaluator
        
        click.echo(f"Evaluating retrieval performance with {test_file}")
        
        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        evaluator = EndToEndEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_retrieval(test_data.get('retrieval', {}))
        
        # Save results
        with open(output, 'w') as f:
            json.dump({
                'evaluation_type': 'retrieval',
                'timestamp': datetime.now().isoformat(),
                'results': {k: {'score': v.score, 'details': v.details} for k, v in results.items()}
            }, f, indent=2)
        
        click.echo(f"Evaluation complete. Results saved to {output}")
        
        # Display key metrics
        click.echo("\nKey Metrics:")
        for metric_name, result in results.items():
            click.echo(f"  {result.metric_name}: {result.score:.4f}")
        
    except Exception as e:
        click.echo(f"Error: {e}")
        raise click.ClickException(str(e))


@evaluate.command('end-to-end')
@click.argument('queries', nargs=-1, required=True)
@click.option('--endpoint', default='http://localhost:8000', help='API endpoint')
@click.option('--output', '-o', default='e2e_evaluation.json', help='Output file')
@click.pass_context
def evaluate_end_to_end(ctx, queries, endpoint, output):
    """Run end-to-end evaluation with sample queries"""
    try:
        import requests
        
        click.echo(f"Running end-to-end evaluation with {len(queries)} queries")
        
        results = []
        total_time = 0
        total_cost = 0
        
        for i, query in enumerate(queries, 1):
            click.echo(f"Query {i}: {query}")
            
            try:
                response = requests.post(
                    f"{endpoint}/chat",
                    json={
                        'message': query,
                        'include_sources': True,
                        'max_sources': 5
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    result = {
                        'query': query,
                        'response': data['response'],
                        'confidence': data['confidence_score'],
                        'processing_time': data['processing_time'],
                        'cost': data['cost_estimate'],
                        'sources_count': len(data['sources']),
                        'citations': data['citations']
                    }
                    
                    results.append(result)
                    total_time += data['processing_time']
                    total_cost += data['cost_estimate']
                    
                    click.echo(f"  ✓ Confidence: {data['confidence_score']:.2f}, Time: {data['processing_time']:.2f}s")
                
                else:
                    click.echo(f"  ✗ Request failed: {response.status_code}")
            
            except Exception as e:
                click.echo(f"  ✗ Error: {e}")
        
        # Save detailed results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'successful_queries': len(results),
            'average_time': total_time / len(results) if results else 0,
            'total_cost': total_cost,
            'average_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
            'results': results
        }
        
        with open(output, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        click.echo(f"\nEvaluation Summary:")
        click.echo(f"  Successful Queries: {len(results)}/{len(queries)}")
        click.echo(f"  Average Response Time: {total_time / len(results):.2f}s")
        click.echo(f"  Total Cost: ${total_cost:.4f}")
        click.echo(f"  Average Confidence: {evaluation_results['average_confidence']:.2f}")
        click.echo(f"  Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}")
        raise click.ClickException(str(e))


@cli.group()
def server():
    """Server management commands"""
    pass


@server.command('start')
@click.option('--host', default='0.0.0.0', help='Host address')
@click.option('--port', default=8000, help='Port number')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', default=1, help='Number of worker processes')
@click.pass_context
def start_server(ctx, host, port, reload, workers):
    """Start the RAG API server"""
    try:
        import uvicorn
        
        click.echo(f"Starting server on {host}:{port}")
        
        if reload or workers == 1:
            # Development mode
            uvicorn.run(
                "src.api.main:app",
                host=host,
                port=port,
                reload=reload
            )
        else:
            # Production mode
            import subprocess
            cmd = [
                "gunicorn",
                "src.api.main:app",
                "-w", str(workers),
                "-k", "uvicorn.workers.UvicornWorker",
                "--bind", f"{host}:{port}"
            ]
            subprocess.run(cmd)
    
    except ImportError:
        click.echo("uvicorn not installed. Install with: pip install uvicorn")
        raise click.ClickException("Missing dependency")
    except Exception as e:
        click.echo(f"Server start failed: {e}")
        raise click.ClickException(str(e))


@server.command('status')
@click.option('--endpoint', default='http://localhost:8000', help='API endpoint')
def check_status(endpoint):
    """Check server status"""
    try:
        import requests
        
        response = requests.get(f"{endpoint}/health")
        
        if response.status_code == 200:
            data = response.json()
            click.echo("✓ Server is healthy")
            click.echo(f"  Status: {data['status']}")
            click.echo(f"  Components: {data['components']}")
            click.echo(f"  Metrics: {data['metrics']}")
        else:
            click.echo(f"✗ Server unhealthy: {response.status_code}")
    
    except Exception as e:
        click.echo(f"✗ Server unreachable: {e}")


@cli.command('setup')
@click.option('--sample-data', is_flag=True, help='Download sample financial data')
@click.pass_context
def setup_environment(ctx, sample_data):
    """Setup development environment"""
    try:
        click.echo("Setting up Finance RAG Pipeline...")
        
        # Create directories
        directories = [
            'data/raw_documents',
            'data/processed_chunks',
            'data/test',
            'logs',
            'models'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            click.echo(f"  ✓ Created directory: {directory}")
        
        # Create environment file template
        env_template = """# Finance RAG Pipeline Environment Variables

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Cohere Configuration (optional)
COHERE_API_KEY=your_cohere_api_key_here

# Pinecone Configuration (optional)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
POSTGRES_URL=postgresql://raguser:ragpassword@localhost:5432/finance_rag
REDIS_URL=redis://localhost:6379

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
"""
        
        with open('.env.template', 'w') as f:
            f.write(env_template)
        
        click.echo("  ✓ Created .env.template file")
        
        if sample_data:
            click.echo("  Downloading sample financial data...")
            # Here you would implement sample data download
            click.echo("  ✓ Sample data setup (placeholder)")
        
        click.echo("\nSetup complete! Next steps:")
        click.echo("1. Copy .env.template to .env and add your API keys")
        click.echo("2. Install dependencies: pip install -r requirements.txt")
        click.echo("3. Start services: docker-compose up -d qdrant redis")
        click.echo("4. Process documents: python cli.py documents process <path>")
        click.echo("5. Start server: python cli.py server start")
        
    except Exception as e:
        click.echo(f"Setup failed: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    cli()
