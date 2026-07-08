# Changelog

## [2.0.0] - 2025-07-07

### Added
- **LangChain Integration**: Full RAG orchestration using LangChain with `RetrievalQA` chains and `RecursiveCharacterTextSplitter`
- **Hybrid Retrieval**: Combined BM25 sparse + FAISS dense retrieval with configurable alpha weighting (Reciprocal Rank Fusion)
- **Cross-Encoder Reranking**: Two-stage retrieval with `ms-marco-MiniLM-L-6-v2` for improved precision
- **SEC EDGAR Integration**: Automated download and parsing of 10-K/10-Q filings from SEC EDGAR API
- **Expanded Knowledge Base**: 12 real SEC filing excerpts from Apple, Tesla, Microsoft, and Amazon (FY2023)
- **Proper Chunking**: Overlapping text windows (512 chars, 128 overlap) using recursive character splitting
- **Evaluation Framework**: Industry-standard metrics (context recall, context precision, faithfulness, answer relevance)
- **Experiment Tracking**: Ablation study framework comparing chunking strategies and retrieval modes
- **Golden QA Set**: Expanded from 10 to 25 question-answer pairs with real financial data
- **Docker Support**: Dockerfile and docker-compose for single-command deployment
- **Structured Logging**: Replaced print statements with Python logging module
- **Index Persistence**: FAISS index save/load wired into application startup
- **Pipeline Configuration**: All RAG parameters configurable via environment variables
- **UI Enhancements**: Retrieval explainability panel, pipeline config sidebar, experiment dashboard
- **Proper Packaging**: `pyproject.toml`, `__init__.py` files, removed sys.path hacks

### Changed
- Evaluation metrics upgraded from token overlap to embedding-based semantic evaluation
- CI/CD pipeline enhanced with model caching, linting (ruff), and evaluation gating
- Requirements updated to include LangChain, rank-bm25, and removed unused packages
- README rewritten with updated architecture diagrams and benchmark results

### Removed
- Unused dependencies: `newspaper3k`, `nltk`, `scikit-learn`
- Dead code: Yahoo Finance references, unused Pinecone stubs

## [1.0.0] - 2025-01-15

### Added
- Initial RAG pipeline with FAISS vector search
- Sentence-transformers (MiniLM) embeddings
- FinBERT sentiment analysis
- Streamlit UI with dark theme
- Basic evaluation framework
- GitHub Actions CI
- PDF report parser
- News scraper with NewsAPI
