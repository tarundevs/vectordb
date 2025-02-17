import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import fitz
from scipy.special import softmax

nltk.download('punkt')

@dataclass
class metadata:
    text: str
    page_number: int

class ChunkText:
    def __init__(self, sentences_per_chunk = 5, overlap_sentences = 2  , max_chunk_size = 512):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text, page_number):
        sentences = sent_tokenize(text)
        chunks = []

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > self.max_chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(metadata(
                        text=chunk_text,
                        page_number=page_number,
                    ))
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(metadata(
                text=chunk_text,
                page_number=page_number,
            ))

        return chunks

class Retrieve:
    def __init__(self, batch_size = 32):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.chunker = ChunkText()
        self.batch_size = batch_size
        self.index = None
        self.chunks: List[metadata] = []

    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return [(page.get_text("text").strip(), page.number + 1) for page in doc if page.get_text("text").strip()]

    def process_document(self, pdf_path):
        text_pages = self.extract_text(pdf_path)
        all_chunks = []

        for text, page_num in text_pages:
            chunks = self.chunker.chunk_text(text, page_num)
            for chunk in chunks:
                all_chunks.append(chunk)

        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            texts = [f"{chunk.text}" for chunk in batch]
            embeddings = self.model.encode(texts, show_progress_bar=len(texts) > 100, convert_to_numpy=True)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

        self.chunks = all_chunks
        self.make_index()

    def make_index(self):
        if not self.chunks:
            return

        embeddings = np.vstack([chunk.embedding for chunk in self.chunks])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, k = 5):
        if not self.index:
            return []

        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, min(k * 3, len(self.chunks)))
        scores = softmax(scores[0])  
        results = []

        for score, idx in zip(scores, indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            results.append({
                'text': chunk.text,
                'score': float(score),
                'metadata': {
                    'page_number': chunk.page_number,
                }
            })

            if len(results) >= k:
                break

        return sorted(results, key=lambda x: x['score'], reverse=True)[:k]

def main():
    processor = Retrieve()
    pdf_path = "SAiDL_Spring_Assignment_2025 (1).pdf"

    processor.process_document(pdf_path)

    query = "What is the reinforcement learning task?"
    print(f"\nSearch Results for: {query}")
    print("-" * 100)
    results = processor.search(query, k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"Page: {result['metadata']['page_number']}")
        print(f"Text: {result['text'][:500]}...")

if __name__ == "__main__":
    main()
