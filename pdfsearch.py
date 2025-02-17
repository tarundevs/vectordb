#Import required packages
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

#Download NLTK data
nltk.download('punkt_tab')

#Implement a metadata dataclass for page numbers
@dataclass
class metadata:
    text: str
    page_number: int

#A class to implement Text chunking
class ChunkText:
    #Initializing chunking parameters
    def __init__(self, sentences_per_chunk = 5, overlap_sentences = 2  , max_chunk_size = 512):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.max_chunk_size = max_chunk_size

    #function to split text into overlapping chunks
    def chunk_text(self, text, page_number):
        #split text into sentences
        sentences = sent_tokenize(text)
        chunks = []

        current_chunk = []
        current_length = 0

        #process each sentence
        for sentence in sentences:
            #convert sentence to tokens
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)

            #checking if it exceeds max size
            if current_length + sentence_length > self.max_chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(metadata(
                        text=chunk_text,
                        page_number=page_number,
                    ))
                    
                    #reset current_chunk
                    current_chunk = []
                    current_length = 0

            #add sentence to chunk
            current_chunk.append(sentence)
            current_length += sentence_length

        #Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(metadata(
                text=chunk_text,
                page_number=page_number,
            ))

        return chunks

#class to implement semantic search
class Retrieve:
    def __init__(self, batch_size = 32):
        #Initialize the sentence transformer model for text embedding
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.chunker = ChunkText()
        self.batch_size = batch_size
        self.index = None
        self.chunks: List[metadata] = []

    #Extract text from PDF file
    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        
        #Return only non-empty pages with their page numbers
        return [(page.get_text("text").strip(), page.number + 1) for page in doc if page.get_text("text").strip()]

    #Process PDF document: extract text, chunk it, and create embeddings
    def process_document(self, pdf_path):
        #Extract text
        text_pages = self.extract_text(pdf_path)
        all_chunks = []

        #chunk text
        for text, page_num in text_pages:
            chunks = self.chunker.chunk_text(text, page_num)
            for chunk in chunks:
                all_chunks.append(chunk)

        #create embeddings in batches
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            texts = [f"{chunk.text}" for chunk in batch]
            
            #generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=len(texts) > 100, convert_to_numpy=True)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

        self.chunks = all_chunks
        
        #create index
        self.make_index()

    #create faiss index for similarity search
    def make_index(self):
        if not self.chunks:
            return

        #Stack all embeddings and then normalize
        embeddings = np.vstack([chunk.embedding for chunk in self.chunks])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        #Create FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    #Search for similar text chunks given a query
    def search(self, query, k):
        if not self.index:
            return []

        #Create and normalize query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)

        #Perform similarity search
        scores, indices = self.index.search(query_embedding, min(k * 3, len(self.chunks)))
        scores = softmax(scores[0])  
        results = []

        #Process search results
        for score, idx in zip(scores, indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            #Formatting results with metadata
            results.append({
                'text': chunk.text,
                'score': float(score),
                'metadata': {
                    'page_number': chunk.page_number,
                }
            })

            if len(results) >= k:
                break
        
        #Return top k results sorted
        return sorted(results, key=lambda x: x['score'], reverse=True)[:k]
