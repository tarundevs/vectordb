from pdfsearch import Retrieve

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

if __name__=="__main__":
    main()