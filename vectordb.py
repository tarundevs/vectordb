# # # At the start of the file, add these imports and initialization
# # from dataclasses import dataclass
# # from typing import List, Optional, Dict, Any, Tuple
# # import numpy as np
# # from sentence_transformers import SentenceTransformer
# # import faiss
# # from transformers import AutoTokenizer
# # import nltk
# # from nltk.tokenize import sent_tokenize
# # import torch
# # from pathlib import Path
# # import logging
# # import fitz  # PyMuPDF

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Download NLTK data
# # def ensure_nltk_data():
# #     """Ensure required NLTK data is downloaded."""
# #     try:
# #         # Try to tokenize a sample sentence to check if punkt is available
# #         sent_tokenize("This is a test sentence.")
# #     except LookupError:
# #         logger.info("Downloading required NLTK data...")
# #         nltk.download('punkt')
# #         nltk.download('punkt_tab')
# #         logger.info("Download complete.")

# # # Call this function before defining other classes
# # ensure_nltk_data()

# # @dataclass
# # class SentenceChunk:
# #     text: str
# #     sentences: List[str]
# #     start_idx: int  # Start index of first sentence
# #     end_idx: int    # End index of last sentence
# #     page_number: int
# #     chunk_index: int
# #     source_path: str
# #     embedding: Optional[np.ndarray] = None

# # class SentenceChunker:
# #     """Chunks text into overlapping groups of sentences."""
# #     def __init__(
# #         self,
# #         sentences_per_chunk: int = 3,
# #         overlap_sentences: int = 1,
# #         max_chunk_size: int = 512,
# #         model_name: str = 'bert-base-uncased'
# #     ):
# #         self.sentences_per_chunk = sentences_per_chunk
# #         self.overlap_sentences = overlap_sentences
# #         self.max_chunk_size = max_chunk_size
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)

# #     def chunk_text(self, text: str, page_number: int) -> List[Tuple[List[str], int, int]]:
# #         """Chunk text into overlapping sentence groups."""
# #         # Split text into sentences
# #         sentences = sent_tokenize(text)
# #         chunks = []
        
# #         # Slide window over sentences
# #         for start_idx in range(0, len(sentences), self.sentences_per_chunk - self.overlap_sentences):
# #             end_idx = start_idx + self.sentences_per_chunk
# #             chunk_sentences = sentences[start_idx:end_idx]
            
# #             if not chunk_sentences:
# #                 continue
                
# #             # Combine sentences and check token length
# #             chunk_text = ' '.join(chunk_sentences)
# #             tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
            
# #             # Skip if chunk is too long
# #             if len(tokens) > self.max_chunk_size:
# #                 # Try to reduce number of sentences until it fits
# #                 while len(chunk_sentences) > 1:
# #                     chunk_sentences = chunk_sentences[:-1]
# #                     chunk_text = ' '.join(chunk_sentences)
# #                     tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
# #                     if len(tokens) <= self.max_chunk_size:
# #                         break
                
# #                 if len(tokens) > self.max_chunk_size:
# #                     continue
            
# #             chunks.append((chunk_sentences, start_idx, start_idx + len(chunk_sentences)))
            
# #         return chunks

# # class FastDocumentProcessor:
# #     def __init__(
# #         self,
# #         model_name: str = 'all-MiniLM-L6-v2',
# #         sentences_per_chunk: int = 3,
# #         overlap_sentences: int = 1,
# #     ):
# #         self.model = SentenceTransformer(model_name)
# #         self.embedding_dim = self.model.get_sentence_embedding_dimension()
# #         self.chunker = SentenceChunker(
# #             sentences_per_chunk=sentences_per_chunk,
# #             overlap_sentences=overlap_sentences
# #         )
        
# #         self.index = None
# #         self.chunks: List[SentenceChunk] = []

# #     def extract_text(self, pdf_path: str) -> List[Tuple[str, int]]:
# #         """Extract text from PDF while preserving page structure."""
# #         logger.info(f"Extracting text from {pdf_path}")
# #         doc = fitz.open(pdf_path)
# #         text_pages = []
        
# #         for page_num in range(len(doc)):
# #             page = doc[page_num]
# #             text = page.get_text("text")
# #             text_pages.append((text, page_num + 1))
            
# #         return text_pages

# #     def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
# #         """Normalize embeddings for cosine similarity."""
# #         return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# #     def process_document(self, pdf_path: str) -> int:
# #         """Process document into sentence-based chunks with embeddings."""
# #         text_pages = self.extract_text(pdf_path)
        
# #         all_chunks = []
# #         chunk_index = 0
        
# #         # Process each page
# #         for text, page_num in text_pages:
# #             # Get sentence chunks for the page
# #             sentence_chunks = self.chunker.chunk_text(text, page_num)
            
# #             # Create SentenceChunk objects
# #             for sentences, start_idx, end_idx in sentence_chunks:
# #                 chunk = SentenceChunk(
# #                     text=' '.join(sentences),
# #                     sentences=sentences,
# #                     start_idx=start_idx,
# #                     end_idx=end_idx,
# #                     page_number=page_num,
# #                     chunk_index=chunk_index,
# #                     source_path=str(Path(pdf_path).absolute())
# #                 )
# #                 all_chunks.append(chunk)
# #                 chunk_index += 1
        
# #         # Generate embeddings in batches
# #         batch_size = 32
# #         for i in range(0, len(all_chunks), batch_size):
# #             batch = all_chunks[i:i + batch_size]
# #             texts = [chunk.text for chunk in batch]
# #             embeddings = self.model.encode(texts)
            
# #             # Store embeddings in chunks
# #             for chunk, embedding in zip(batch, embeddings):
# #                 chunk.embedding = embedding
        
# #         # Store chunks and build FAISS index
# #         self.chunks = all_chunks
# #         self.build_index()
        
# #         return len(all_chunks)

# #     def build_index(self):
# #         """Build FAISS index from chunk embeddings."""
# #         if not self.chunks:
# #             return
            
# #         embeddings = np.vstack([chunk.embedding for chunk in self.chunks])
# #         embeddings = self.normalize_embeddings(embeddings)
        
# #         n_lists = min(int(np.sqrt(len(self.chunks))), len(self.chunks))
# #         n_lists = max(1, n_lists)
        
# #         quantizer = faiss.IndexFlatL2(self.embedding_dim)
# #         self.index = faiss.IndexIVFFlat(
# #             quantizer,
# #             self.embedding_dim,
# #             n_lists,
# #             faiss.METRIC_INNER_PRODUCT
# #         )
# #         self.index.nprobe = min(n_lists, 10)
        
# #         self.index.train(embeddings)
# #         self.index.add(embeddings)

# #     def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
# #         """Search for relevant chunks using sentence-based context."""
# #         if not self.index:
# #             logger.warning("No index built yet. Please process a document first.")
# #             return []
            
# #         query_embedding = self.model.encode([query])[0]
# #         query_embedding = query_embedding / np.linalg.norm(query_embedding)
# #         query_embedding = query_embedding.reshape(1, -1)
        
# #         scores, indices = self.index.search(query_embedding, min(k * 2, len(self.chunks)))
        
# #         results = []
# #         seen_texts = set()
        
# #         for score, idx in zip(scores[0], indices[0]):
# #             if idx < 0 or idx >= len(self.chunks):
# #                 continue
                
# #             chunk = self.chunks[idx]
            
# #             if chunk.text in seen_texts:
# #                 continue
            
# #             cosine_sim = float(score)
            
# #             # Get surrounding context
# #             context = self.get_context_window(chunk)
            
# #             results.append({
# #                 'text': context,
# #                 'score': cosine_sim,
# #                 'metadata': {
# #                     'page_number': chunk.page_number,
# #                     'chunk_index': chunk.chunk_index,
# #                     'source_path': chunk.source_path,
# #                     'sentence_range': (chunk.start_idx, chunk.end_idx)
# #                 }
# #             })
            
# #             seen_texts.add(chunk.text)
            
# #             if len(results) >= k:
# #                 break
        
# #         results.sort(key=lambda x: x['score'], reverse=True)
# #         return results[:k]

# #     def get_context_window(self, chunk: SentenceChunk) -> str:
# #         """Get surrounding context for a chunk using sentence-based windows."""
# #         context_chunks = []
        
# #         # Find chunks that overlap with or are adjacent to current chunk
# #         for other in self.chunks:
# #             if (other.page_number == chunk.page_number and
# #                 other.chunk_index != chunk.chunk_index and
# #                 self.chunks_adjacent(chunk, other)):
# #                 context_chunks.append(other)
        
# #         context_chunks.sort(key=lambda x: x.start_idx)
        
# #         full_text = []
# #         for c in context_chunks:
# #             if c.chunk_index == chunk.chunk_index:
# #                 full_text.append(f"[RELEVANT] {c.text} [/RELEVANT]")
# #             else:
# #                 full_text.append(c.text)
        
# #         return " ".join(full_text)

# #     def chunks_adjacent(self, chunk1: SentenceChunk, chunk2: SentenceChunk) -> bool:
# #         """Check if two chunks are adjacent or overlapping in sentence space."""
# #         # Allow for one sentence gap to connect nearby context
# #         return (abs(chunk1.start_idx - chunk2.end_idx) <= 1 or
# #                 abs(chunk2.start_idx - chunk1.end_idx) <= 1)

# # def main():
# #     # Initialize processor with sentence-based chunking
# #     processor = FastDocumentProcessor(
# #         sentences_per_chunk=3,
# #         overlap_sentences=1
# #     )
    
# #     # Process document
# #     pdf_path = "lease.pdf"
# #     num_chunks = processor.process_document(pdf_path)
# #     print(f"Processed document into {num_chunks} sentence-based chunks")
    
# #     # Example search
# #     query = "What happens if either party breaks the lease?"
# #     results = processor.search(query, k=3)
    
# #     print("\nSearch Results:")
# #     print("-" * 50)
# #     for i, result in enumerate(results, 1):
# #         print(f"\n{i}. Score: {result['score']:.3f}")
# #         print(f"Page: {result['metadata']['page_number']}")
# #         print(f"Sentences: {result['metadata']['sentence_range']}")
# #         print(f"Text: {result['text'][:200]}...")

# # if __name__ == "__main__":
# #     main()

# from dataclasses import dataclass
# from typing import List, Optional, Dict, Any, Tuple
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from transformers import AutoTokenizer
# import nltk
# from nltk.tokenize import sent_tokenize
# import torch
# from pathlib import Path
# import logging
# import fitz  # PyMuPDF

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def ensure_nltk_data():
#     """Ensure required NLTK data is downloaded."""
#     try:
#         sent_tokenize("This is a test sentence.")
#     except LookupError:
#         logger.info("Downloading required NLTK data...")
#         nltk.download('punkt')
#         nltk.download('punkt_tab')
#         logger.info("Download complete.")

# ensure_nltk_data()

# @dataclass
# class SentenceChunk:
#     text: str
#     sentences: List[str]
#     start_idx: int
#     end_idx: int
#     page_number: int
#     chunk_index: int
#     source_path: str
#     embedding: Optional[np.ndarray] = None

# class SentenceChunker:
#     """Chunks text into overlapping groups of sentences with MPNet tokenization."""
#     def __init__(
#         self,
#         sentences_per_chunk: int = 3,
#         overlap_sentences: int = 1,
#         max_chunk_size: int = 512,
#         model_name: str = 'sentence-transformers/all-mpnet-base-v2'
#     ):
#         self.sentences_per_chunk = sentences_per_chunk
#         self.overlap_sentences = overlap_sentences
#         self.max_chunk_size = max_chunk_size
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)

#     def chunk_text(self, text: str, page_number: int) -> List[Tuple[List[str], int, int]]:
#         """Chunk text into overlapping sentence groups optimized for MPNet."""
#         sentences = sent_tokenize(text)
#         chunks = []
        
#         for start_idx in range(0, len(sentences), self.sentences_per_chunk - self.overlap_sentences):
#             end_idx = start_idx + self.sentences_per_chunk
#             chunk_sentences = sentences[start_idx:end_idx]
            
#             if not chunk_sentences:
#                 continue
                
#             chunk_text = ' '.join(chunk_sentences)
#             tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
            
#             if len(tokens) > self.max_chunk_size:
#                 while len(chunk_sentences) > 1:
#                     chunk_sentences = chunk_sentences[:-1]
#                     chunk_text = ' '.join(chunk_sentences)
#                     tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
#                     if len(tokens) <= self.max_chunk_size:
#                         break
                
#                 if len(tokens) > self.max_chunk_size:
#                     continue
            
#             chunks.append((chunk_sentences, start_idx, start_idx + len(chunk_sentences)))
            
#         return chunks

# class MPNetDocumentProcessor:
#     """Document processor using MPNet for enhanced semantic understanding."""
#     def __init__(
#         self,
#         model_name: str = 'sentence-transformers/all-mpnet-base-v2',
#         sentences_per_chunk: int = 3,
#         overlap_sentences: int = 1,
#         batch_size: int = 32
#     ):
#         self.model = SentenceTransformer(model_name)
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         self.chunker = SentenceChunker(
#             sentences_per_chunk=sentences_per_chunk,
#             overlap_sentences=overlap_sentences,
#             model_name=model_name
#         )
#         self.batch_size = batch_size
#         self.index = None
#         self.chunks: List[SentenceChunk] = []

#     def extract_text(self, pdf_path: str) -> List[Tuple[str, int]]:
#         """Extract text from PDF with enhanced error handling."""
#         logger.info(f"Extracting text from {pdf_path}")
#         try:
#             doc = fitz.open(pdf_path)
#             text_pages = []
            
#             for page_num in range(len(doc)):
#                 page = doc[page_num]
#                 text = page.get_text("text")
#                 if text.strip():  # Only include non-empty pages
#                     text_pages.append((text, page_num + 1))
                
#             return text_pages
#         except Exception as e:
#             logger.error(f"Error extracting text from PDF: {e}")
#             raise

#     def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
#         """Normalize embeddings for cosine similarity with numerical stability."""
#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         norms = np.maximum(norms, 1e-12)  # Prevent division by zero
#         return embeddings / norms

#     def process_document(self, pdf_path: str) -> int:
#         """Process document with batched embedding generation."""
#         text_pages = self.extract_text(pdf_path)
        
#         all_chunks = []
#         chunk_index = 0
        
#         for text, page_num in text_pages:
#             sentence_chunks = self.chunker.chunk_text(text, page_num)
            
#             for sentences, start_idx, end_idx in sentence_chunks:
#                 chunk = SentenceChunk(
#                     text=' '.join(sentences),
#                     sentences=sentences,
#                     start_idx=start_idx,
#                     end_idx=end_idx,
#                     page_number=page_num,
#                     chunk_index=chunk_index,
#                     source_path=str(Path(pdf_path).absolute())
#                 )
#                 all_chunks.append(chunk)
#                 chunk_index += 1
        
#         # Generate embeddings in optimized batches
#         for i in range(0, len(all_chunks), self.batch_size):
#             batch = all_chunks[i:i + self.batch_size]
#             texts = [chunk.text for chunk in batch]
            
#             # Use model's encoding with showing_progress_bar for longer documents
#             embeddings = self.model.encode(
#                 texts,
#                 show_progress_bar=len(texts) > 100,
#                 convert_to_numpy=True
#             )
            
#             for chunk, embedding in zip(batch, embeddings):
#                 chunk.embedding = embedding
        
#         self.chunks = all_chunks
#         self.build_index()
        
#         return len(all_chunks)

#     def build_index(self):
#         """Build FAISS index with optimized configuration."""
#         if not self.chunks:
#             return
            
#         embeddings = np.vstack([chunk.embedding for chunk in self.chunks])
#         embeddings = self.normalize_embeddings(embeddings)
        
#         # Optimize number of clusters based on dataset size
#         n_lists = min(int(np.sqrt(len(self.chunks))), len(self.chunks))
#         n_lists = max(1, n_lists)
        
#         quantizer = faiss.IndexFlatL2(self.embedding_dim)
#         self.index = faiss.IndexIVFFlat(
#             quantizer,
#             self.embedding_dim,
#             n_lists,
#             faiss.METRIC_INNER_PRODUCT
#         )
        
#         # Adjust nprobe based on dataset size
#         self.index.nprobe = min(n_lists, max(10, int(n_lists * 0.1)))
        
#         self.index.train(embeddings)
#         self.index.add(embeddings)

#     def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
#         """Search with enhanced context retrieval."""
#         if not self.index:
#             logger.warning("No index built yet. Please process a document first.")
#             return []
            
#         query_embedding = self.model.encode([query])[0]
#         query_embedding = query_embedding / np.linalg.norm(query_embedding)
#         query_embedding = query_embedding.reshape(1, -1)
        
#         scores, indices = self.index.search(query_embedding, min(k * 2, len(self.chunks)))
        
#         results = []
#         seen_texts = set()
        
#         for score, idx in zip(scores[0], indices[0]):
#             if idx < 0 or idx >= len(self.chunks):
#                 continue
                
#             chunk = self.chunks[idx]
            
#             if chunk.text in seen_texts:
#                 continue
            
#             context = self.get_context_window(chunk)
            
#             results.append({
#                 'text': context,
#                 'score': float(score),
#                 'metadata': {
#                     'page_number': chunk.page_number,
#                     'chunk_index': chunk.chunk_index,
#                     'source_path': chunk.source_path,
#                     'sentence_range': (chunk.start_idx, chunk.end_idx)
#                 }
#             })
            
#             seen_texts.add(chunk.text)
            
#             if len(results) >= k:
#                 break
        
#         results.sort(key=lambda x: x['score'], reverse=True)
#         return results[:k]

#     def get_context_window(self, chunk: SentenceChunk) -> str:
#         """Get enhanced context window with better formatting."""
#         context_chunks = []
        
#         for other in self.chunks:
#             if (other.page_number == chunk.page_number and
#                 other.chunk_index != chunk.chunk_index and
#                 self.chunks_adjacent(chunk, other)):
#                 context_chunks.append(other)
        
#         context_chunks.sort(key=lambda x: x.start_idx)
        
#         full_text = []
#         for c in context_chunks:
#             if c.chunk_index == chunk.chunk_index:
#                 full_text.append(f"[RELEVANT] {c.text} [/RELEVANT]")
#             else:
#                 full_text.append(c.text)
        
#         return " ".join(full_text)

#     def chunks_adjacent(self, chunk1: SentenceChunk, chunk2: SentenceChunk) -> bool:
#         """Check chunk adjacency with configurable threshold."""
#         return (abs(chunk1.start_idx - chunk2.end_idx) <= 1 or
#                 abs(chunk2.start_idx - chunk1.end_idx) <= 1)

# def main():
#     # Initialize processor with MPNet model
#     processor = MPNetDocumentProcessor(
#         sentences_per_chunk=3,
#         overlap_sentences=1,
#         batch_size=32
#     )
    
#     # Process document
#     pdf_path = "lease.pdf"
#     num_chunks = processor.process_document(pdf_path)
#     print(f"Processed document into {num_chunks} chunks using MPNet")
    
#     # Example search
#     query = "What happens if either party breaks the lease?"
#     results = processor.search(query, k=3)
    
#     print("\nSearch Results:")
#     print("-" * 50)
#     for i, result in enumerate(results, 1):
#         print(f"\n{i}. Score: {result['score']:.3f}")
#         print(f"Page: {result['metadata']['page_number']}")
#         print(f"Sentences: {result['metadata']['sentence_range']}")
#         print(f"Text: {result['text'][:200]}...")

# if __name__ == "__main__":
#     main()

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
import logging
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        sent_tokenize("This is a test sentence.")
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        logger.info("Download complete.")

ensure_nltk_data()

@dataclass
class DocumentChunk:
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

class SmartDocumentChunker:
    """Intelligent document chunker with section and structure awareness."""
    def __init__(
        self,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 2,
        max_chunk_size: int = 512,
        model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    ):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.max_chunk_size = max_chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def detect_section_header(self, text: str) -> Optional[str]:
        """Detect section headers in document text."""
        header_patterns = [
            r'^\s*(?:[IVX]+|\d+)\.?\s+(.+)$',  # Roman numerals or numbers with text
            r'^\s*[A-Z][A-Z\s]+:',  # Uppercase text with colon
            r'^\s*#{1,6}\s+(.+)$',  # Markdown-style headers
            r'^\s*\[(.+)\]$'  # Bracketed sections
        ]
        
        for pattern in header_patterns:
            match = re.match(pattern, text)
            if match:
                return text.strip()
        return None

    def chunk_text(self, text: str, page_number: int) -> List[Tuple[List[str], int, int, Optional[str]]]:
        """Chunk text with structure awareness."""
        sentences = sent_tokenize(text)
        chunks = []
        current_section = None
        
        for start_idx in range(0, len(sentences), self.sentences_per_chunk - self.overlap_sentences):
            end_idx = start_idx + self.sentences_per_chunk
            chunk_sentences = sentences[start_idx:end_idx]
            
            if not chunk_sentences:
                continue
            
            potential_header = self.detect_section_header(chunk_sentences[0])
            if potential_header:
                current_section = potential_header
            
            chunk_text = ' '.join(chunk_sentences)
            tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
            
            if len(tokens) > self.max_chunk_size:
                while len(chunk_sentences) > 1:
                    chunk_sentences = chunk_sentences[:-1]
                    chunk_text = ' '.join(chunk_sentences)
                    tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
                    if len(tokens) <= self.max_chunk_size:
                        break
                
                if len(tokens) > self.max_chunk_size:
                    continue
            
            chunks.append((chunk_sentences, start_idx, start_idx + len(chunk_sentences), current_section))
            
        return chunks

class EnhancedDocumentProcessor:
    """Advanced document processor for general PDF documents."""
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 2,
        batch_size: int = 32
    ):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.chunker = SmartDocumentChunker(
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=overlap_sentences,
            model_name=model_name
        )
        self.batch_size = batch_size
        self.index = None
        self.chunks: List[DocumentChunk] = []

    def extract_text(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF while preserving page structure."""
        logger.info(f"Extracting text from {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            text_pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():  # Only include non-empty pages
                    text_pages.append((text, page_num + 1))
                    
            return text_pages
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity with numerical stability."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Prevent division by zero
        return embeddings / norms
        
    def process_document(self, pdf_path: str) -> int:
        """Process PDF document with structure awareness."""
        text_pages = self.extract_text(pdf_path)
        
        all_chunks = []
        chunk_index = 0
        
        for text, page_num in text_pages:
            sentence_chunks = self.chunker.chunk_text(text, page_num)
            
            for sentences, start_idx, end_idx, section_header in sentence_chunks:
                chunk = DocumentChunk(
                    text=' '.join(sentences),
                    sentences=sentences,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    source_path=str(Path(pdf_path).absolute()),
                    section_header=section_header
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        # Generate embeddings with section context
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            texts = [
                f"{chunk.section_header} {chunk.text}" if chunk.section_header 
                else chunk.text for chunk in batch
            ]
            
            embeddings = self.model.encode(
                texts,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
        
        self.chunks = all_chunks
        self.build_index()
        
        return len(all_chunks)

    def build_index(self):
        """Build FAISS index with optimized configuration."""
        if not self.chunks:
            return
            
        embeddings = np.vstack([chunk.embedding for chunk in self.chunks])
        embeddings = self.normalize_embeddings(embeddings)
        
        # Optimize number of clusters based on dataset size
        n_lists = min(int(np.sqrt(len(self.chunks))), len(self.chunks))
        n_lists = max(1, n_lists)
        
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            n_lists,
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Adjust nprobe based on dataset size
        self.index.nprobe = min(n_lists, max(10, int(n_lists * 0.1)))
        
        self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search with section awareness and context preservation."""
        if not self.index:
            logger.warning("No index built yet. Please process a document first.")
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
            
            # Ensure diverse section coverage while maintaining relevance
            if chunk.section_header in seen_sections and score < 0.5:
                continue
            
            if chunk.section_header:
                seen_sections.add(chunk.section_header)
            
            context = self.get_context_window(chunk)
            
            results.append({
                'text': context,
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
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def chunks_adjacent(self, chunk1: DocumentChunk, chunk2: DocumentChunk) -> bool:
        """Check if two chunks are adjacent or overlapping."""
        return (abs(chunk1.start_idx - chunk2.end_idx) <= 1 or
                abs(chunk2.start_idx - chunk1.end_idx) <= 1)

    def get_context_window(self, chunk: DocumentChunk) -> str:
        """Get context with section awareness."""
        context_chunks = []
        
        for other in self.chunks:
            if (other.page_number == chunk.page_number and
                (other.chunk_index != chunk.chunk_index and
                 self.chunks_adjacent(chunk, other) or
                 other.section_header == chunk.section_header)):
                context_chunks.append(other)
        
        context_chunks.sort(key=lambda x: x.start_idx)
        
        full_text = []
        for c in context_chunks:
            if c.chunk_index == chunk.chunk_index:
                if c.section_header:
                    full_text.append(f"[SECTION: {c.section_header}] {c.text} [/SECTION]")
                else:
                    full_text.append(f"[RELEVANT] {c.text} [/RELEVANT]")
            else:
                full_text.append(c.text)
        
        return " ".join(full_text)

def main():
    """Example usage of the document processor."""
    # Initialize processor
    processor = EnhancedDocumentProcessor(
        sentences_per_chunk=3,
        overlap_sentences=2,
        batch_size=32
    )
    
    # Process a document
    pdf_path = "SAiDL_Spring_Assignment_2025 (1).pdf"  # Replace with your PDF path
    try:
        num_chunks = processor.process_document(pdf_path)
        print(f"Processed document into {num_chunks} chunks")
        
        # Example searches
        queries = [
            "What is Reinforcement learning?"
        ]
        
        for query in queries:
            print(f"\nSearch Results for: {query}")
            print("-" * 50)
            results = processor.search(query, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.3f}")
                print(f"Page: {result['metadata']['page_number']}")
                if result['metadata']['section']:
                    print(f"Section: {result['metadata']['section']}")
                print(f"Text: {result['text'][:200]}...")
                
    except Exception as e:
        logger.error(f"Error processing document: {e}")

if __name__ == "__main__":
    main()
