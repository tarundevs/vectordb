# Vector Search Evaluator

This project evaluates the performance of a vector-based search engine using the **Sentence Transformers** library and **ChromaDB**. It creates a synthetic test dataset and measures the precision, recall, and latency of the search engine based on a set of known relevant document-query pairs.

## Table of Contents

- [Introduction](#introduction)
- [What It Uses](#what-it-uses)
- [Dependencies](#dependencies)

## Introduction

The **Vector Search Evaluator** is designed to evaluate the search performance of a vector-based search engine. It uses the **SentenceTransformer** model to embed text documents and queries into high-dimensional vectors, which are then stored in **ChromaDB**. The evaluation involves querying the database using synthetic test queries and comparing the retrieved documents with the known relevant documents. The primary focus is on measuring search precision, recall, and latency.

## What It Uses

1. **Sentence Transformers**:
   - Used to convert text documents and queries into vector embeddings. This enables semantic search by comparing the similarity between vectors.

2. **ChromaDB**:
   - A vector database for storing document embeddings and performing similarity searches. It is used for efficient document retrieval based on query embeddings.

3. **Evaluation Metrics**:
   - **Precision**: The proportion of relevant documents retrieved among the top `k` results.
   - **Recall**: The proportion of relevant documents retrieved out of all relevant documents for a given query.
   - **Latency**: The time taken to retrieve the results from the database.

4. **Synthetic Test Dataset**:
   - The test dataset consists of three groups of documents: **Technology**, **Nature**, and **Space**. Each group is paired with a set of queries that are expected to retrieve relevant documents from the group.

5. **Performance Evaluation**:
   - The system evaluates the search engine using various test queries and calculates precision, recall, and latency for the top `k` retrieved documents.

## Dependencies

The following Python libraries are required to run this project:

- `sentence-transformers` - for converting text into embeddings.
- `chromadb` - for creating a persistent vector database and performing similarity queries.
- `numpy` - for numerical operations and calculating metrics.
- `time` - for measuring latency.
