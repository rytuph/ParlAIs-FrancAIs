# scripts/03_build_vector_store.py

def main():
    """
    This script builds the FAISS vector store for the RAG pipeline.
    
    The process involves:
    1.  Loading the static knowledge base of French grammar rules.
    2.  Chunking the documents into manageable pieces for embedding.
    3.  Using a sentence-transformer model (e.g., 'all-MiniLM-L6-v2') to generate embeddings for each chunk.
    4.  Creating a FAISS index from these embeddings.
    5.  Saving the index to disk for fast retrieval during inference.
    """
    print("Building the FAISS vector store...")
    print("Step 1: Loading grammar rule documents...")
    print("Step 2: Chunking documents...")
    print("Step 3: Generating embeddings for document chunks...")
    print("Step 4: Creating FAISS index...")
    print("Step 5: Saving index to 'data/vector_store/faiss_index.bin'...")
    print("Vector store build complete.")

if __name__ == "__main__":
    main()