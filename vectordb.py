from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import time

class VectorSearchEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path="./eval_chroma_db")
        
    def create_test_dataset(self):
        """Create a synthetic test dataset with known relevant pairs"""
        # Test documents with clear semantic relationships
        documents = [
            # Technology group
            "Artificial Intelligence is transforming modern technology",
            "Machine learning systems are revolutionizing tech industry",
            "AI and ML are driving technological innovation",
            # Nature group
            "Mountains and forests create beautiful landscapes",
            "Natural ecosystems support diverse wildlife",
            "Forests and mountains are vital for biodiversity",
            # Space group
            "Galaxies contain billions of stars and planets",
            "The universe is expanding at an increasing rate",
            "Stars and planets form complex galactic systems"
        ]
        
        # Test queries with known relevant documents
        queries = [
            ("What is the impact of AI on technology?", [0, 1, 2]),  # Should match tech group
            ("Tell me about natural landscapes", [3, 4, 5]),         # Should match nature group
            ("How do galaxies work?", [6, 7, 8])                     # Should match space group
        ]
        
        return documents, queries

    def evaluate_search(self, k=3):
        """
        Evaluate search performance using the test dataset
        """
        # Create evaluation collection
        collection_name = "eval_collection"
        
        # Remove existing collection if it exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
            
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Get test data
        documents, test_queries = self.create_test_dataset()
        
        # Add documents to collection with metadata containing index
        doc_embeddings = self.model.encode(documents).tolist()
        print(doc_embeddings)
        collection.add(
            embeddings=doc_embeddings,
            documents=documents,
            metadatas=[{"index": i} for i in range(len(documents))],
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        '''
        langchain
        '''
        # Metrics storage
        precision_scores = []
        recall_scores = []
        latency_times = []
        
        print("\nDetailed Evaluation Results:")
        print("-" * 50)
        
        # Test each query
        for query_text, relevant_indices in test_queries:
            start_time = time.time()
            
            # Get search results
            query_embedding = self.model.encode([query_text]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=["documents", "distances", "metadatas"]
            )
            
            query_time = time.time() - start_time
            latency_times.append(query_time)
            
            # Get indices from metadata
            retrieved_indices = [doc["index"] for doc in results["metadatas"][0]]
            
            # Calculate metrics
            relevant_retrieved = len(set(retrieved_indices) & set(relevant_indices))
            precision = relevant_retrieved / k
            recall = relevant_retrieved / len(relevant_indices)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            # Print detailed results for this query
            print(f"\nQuery: {query_text}")
            print(f"Retrieved Documents:")
            for i, (doc, distance, metadata) in enumerate(zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                relevance = "✓" if metadata["index"] in relevant_indices else "✗"
                print(f"{relevance} [{distance:.3f}] {doc[:100]}...")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"Query Time: {query_time*1000:.2f}ms")
            print("-" * 50)
        
        # Calculate overall metrics
        metrics = {
            'avg_precision': np.mean(precision_scores),
            'avg_recall': np.mean(recall_scores),
            'avg_latency': np.mean(latency_times),
            'latency_95th': np.percentile(latency_times, 95)
        }
        
        return metrics

def main():
    print("Initializing Vector Search Evaluator...")
    evaluator = VectorSearchEvaluator()
    
    print("Running evaluation...")
    metrics = evaluator.evaluate_search(k=3)
    
    print("\nOverall Performance Metrics:")
    print(f"Average Precision: {metrics['avg_precision']:.3f}")
    print(f"Average Recall: {metrics['avg_recall']:.3f}")
    print(f"Average Latency: {metrics['avg_latency']*1000:.2f}ms")
    print(f"95th Percentile Latency: {metrics['latency_95th']*1000:.2f}ms")

if __name__ == "__main__":
    main()