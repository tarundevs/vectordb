# PDF Semantic Search
A Python-based semantic search system for PDF documents that uses transformer models to enable intelligent, meaning-based document retrieval.

## Features
* Semantic search capabilities using transformer models
* Fast similarity search using FAISS indexing
* Score-based ranking of search results
* Page number tracking for result localization
* Configurable text chunking with overlap
* Batch processing for efficient embedding generation
* Support for multiple transformer models
* Memory-efficient document processing

## Prerequisites
* Python 3.7+

**Dependencies:**
* `sentence-transformers`
* `faiss-gpu` (or `faiss-cpu`)
* `transformers`
* `nltk`
* `PyMuPDF` (fitz)
* `numpy`
* `scipy`

## Installation

```bash
# Clone the repository
git clone https://github.com/tarundevs/vectordb.git
cd pdf-semantic-search

# Create a virtual environment in windows (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Running the Script
The script can be executed from the command line with the following syntax:
```bash
python main.py <pdf_path> <query> [k]
```

### Example usage
```bash
python main.py example.pdf "What is machine learning?"
```

```bash
python main.py research_paper.pdf "Explain the role of transformers in NLP." 5
```

Where:
- `pdf_path`: Path to the PDF document
- `query`: Search query text
- `k`: Number of results to return (optional, default=3)

### Configuration

#### Chunk Text Configuration
```python
chunker = ChunkText(
    sentences_per_chunk=5,    # Number of sentences per chunk
    overlap_sentences=2,      # Number of overlapping sentences
    max_chunk_size=512       # Maximum tokens per chunk
)
```

#### Retriever Configuration
```python
retriever = Retrieve(
    batch_size=32,           # Batch size for processing embeddings
    model_name='all-MiniLM-L6-v2',  # Name of the transformer model
    use_gpu=True,           # Whether to use GPU for computation
    index_type='L2'         # FAISS index type ('L2' or 'IP')
)
```

## Architecture

### Components

1. **ChunkText Class**
   * Handles document text chunking
   * Configurable chunk sizes and overlap
   * Token limit enforcement
   * BERT tokenizer integration
   * Sentence boundary detection
   * Overlap management

2. **Retrieve Class**
   * Main processing pipeline
   * PDF text extraction
   * Embedding generation
   * FAISS index management
   * Search functionality
   * Result ranking and scoring
   * Batch processing optimization

3. **Metadata Class**
   * Stores chunk information
   * Tracks page numbers
   * Maintains text-metadata relationships
   * Facilitates result contextualization
