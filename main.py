#Import required packages
from pdfsearch import Retrieve
import sys

#main function to apply semantic search
def main(pdf_path, query, k):
    #create an object of Retrieve class
    processor = Retrieve()

    #process the document and then search
    processor.process_document(pdf_path)
    results = processor.search(query, k=k)
    
    #display results
    print(f"\nSearch Results for: {query}")
    print("-" * 100)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"Page: {result['metadata']['page_number']}")
        print(f"Text: {result['text'][:500]}...")

if __name__=="__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nUsage: python main.py <pdf_path> <query> [k]")
        print("Note: k is optional and defaults to 3.")
        sys.exit(1)
    
    #initialize variables from command line
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) == 4 else 3

    #call main function
    main(pdf_path, query, k)
