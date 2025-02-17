from pdfsearch import Retrieve
import sys

def main(pdf_path, query, k):
    processor = Retrieve()

    processor.process_document(pdf_path)

    print(f"\nSearch Results for: {query}")
    print("-" * 100)
    results = processor.search(query, k=k)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"Page: {result['metadata']['page_number']}")
        print(f"Text: {result['text'][:500]}...")

if __name__=="__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nUsage: python main.py <pdf_path> <query> [k]")
        print("Note: k is optional and defaults to 3.")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) == 4 else 3

    main(pdf_path, query, k)
