from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import re
from pathlib import Path
import fitz  # PyMuPDF

# Ensure NLTK data is downloaded
nltk.download('punkt')

@dataclass
class metadata:
    """Represents a chunk of document text with metadata."""
    text: str
    sentences: List[str]
    start_idx: int
    end_idx: int
    page_number: int
    chunk_index: int
    source_path: str
    section_header: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class DocumentChunker:
    """Chunks document text into smaller pieces with section awareness."""
    def __init__(self, sentences_per_chunk: int = 3, overlap_sentences: int = 2, max_chunk_size: int = 512):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.max_chunk_size = max_chunk_size

    def detect_section_header(self, text: str) -> Optional[str]:
        """Detect section headers in document text."""
        patterns = [
            r'^\s*(?:[IVX]+|\d+)\.?\s+(.+)$',  # Roman numerals or numbers with text
            r'^\s*[A-Z][A-Z\s]+:',  # Uppercase text with colon
            r'^\s*#{1,6}\s+(.+)$',  # Markdown-style headers
            r'^\s*\[(.+)\]$'  # Bracketed sections
        ]
        for pattern in patterns:
            if match := re.match(pattern, text):
                return text.strip()
        return None

    def chunk_text(self, text: str, page_number: int) -> List[metadata]:
        """Chunk text into smaller pieces with section awareness."""
        sentences = sent_tokenize(text)
        chunks = []
        current_section = None

        for start_idx in range(0, len(sentences), self.sentences_per_chunk - self.overlap_sentences):
            end_idx = start_idx + self.sentences_per_chunk
            chunk_sentences = sentences[start_idx:end_idx]

            if not chunk_sentences:
                continue

            # Detect section header
            if header := self.detect_section_header(chunk_sentences[0]):
                current_section = header

            # Ensure chunk size is within limits
            chunk_text = ' '.join(chunk_sentences)
            tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
            if len(tokens) > self.max_chunk_size:
                while len(chunk_sentences) > 1 and len(tokens) > self.max_chunk_size:
                    chunk_sentences = chunk_sentences[:-1]
                    chunk_text = ' '.join(chunk_sentences)
                    tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)

            if len(tokens) <= self.max_chunk_size:
                chunks.append(metadata(
                    text=chunk_text,
                    sentences=chunk_sentences,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_sentences),
                    page_number=page_number,
                    chunk_index=len(chunks),
                    source_path="",  # Will be set later
                    section_header=current_section
                ))
        return chunks

class DocumentProcessor:
    """Processes PDF documents into chunks and builds a searchable index."""
    def __init__(self, batch_size: int = 32):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.chunker = DocumentChunker()
        self.batch_size = batch_size
        self.index = None
        self.chunks: List[metadata] = []

    def extract_text(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF while preserving page structure."""
        doc = fitz.open(pdf_path)
        return [(page.get_text("text").strip(), page.number + 1) for page in doc if page.get_text("text").strip()]

    def process_document(self, pdf_path: str) -> int:
        """Process a PDF document into chunks and generate embeddings."""
        text_pages = self.extract_text(pdf_path)
        all_chunks = []

        for text, page_num in text_pages:
            chunks = self.chunker.chunk_text(text, page_num)
            for chunk in chunks:
                chunk.source_path = str(Path(pdf_path).absolute())
                all_chunks.append(chunk)

        # Generate embeddings in batches
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            texts = [f"{chunk.section_header} {chunk.text}" if chunk.section_header else chunk.text for chunk in batch]
            embeddings = self.model.encode(texts, show_progress_bar=len(texts) > 100, convert_to_numpy=True)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

        self.chunks = all_chunks
        self._build_index()
        return len(all_chunks)

    def _build_index(self):
        """Build a FAISS index for fast similarity search."""
        if not self.chunks:
            return

        embeddings = np.vstack([chunk.embedding for chunk in self.chunks])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        n_lists = min(int(np.sqrt(len(self.chunks))), len(self.chunks))
        quantizer = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], n_lists, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = min(n_lists, max(10, int(n_lists * 0.1)))
        self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the document for relevant chunks."""
        if not self.index:
            return []

        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, min(k * 3, len(self.chunks)))
        results = []
        seen_sections = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            if chunk.section_header in seen_sections and score < 0.5:
                continue

            if chunk.section_header:
                seen_sections.add(chunk.section_header)

            results.append({
                'text': chunk.text,
                'score': float(score),
                'metadata': {
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.chunk_index,
                    'section': chunk.section_header,
                    'source_path': chunk.source_path,
                    'sentence_range': (chunk.start_idx, chunk.end_idx)
                }
            })

            if len(results) >= k:
                break

        return sorted(results, key=lambda x: x['score'], reverse=True)[:k]

def main():
    """Example usage of the document processor."""
    processor = DocumentProcessor()
    pdf_path = "SAiDL_Spring_Assignment_2025 (1).pdf"  # Replace with your PDF path

    num_chunks = processor.process_document(pdf_path)
    print(f"Processed document into {num_chunks} chunks")

    query = "What is Reinforcement learning?"
    print(f"\nSearch Results for: {query}")
    print("-" * 50)
    results = processor.search(query, k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"Page: {result['metadata']['page_number']}")
        if result['metadata']['section']:
            print(f"Section: {result['metadata']['section']}")
        print(f"Text: {result['text'][:200]}...")



if __name__ == "__main__":
    main()
