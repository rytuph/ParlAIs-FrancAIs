# scripts/03_build_vector_store.py
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import yaml

def main():
    """
    This script builds the FAISS vector store for the RAG pipeline.
    
    It performs the following steps:
    1. Loads the grammar knowledge base documents.
    2. Uses a pre-trained Sentence Transformer model to generate embeddings for each document.
    3. Creates a FAISS index from these embeddings.
    4. Saves the FAISS index and a corresponding mapping file to disk.
    """
    print("Starting the vector store build process...")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    knowledge_base_path = config['paths']['knowledge_base']
    output_dir = config['paths']['vector_db']
    model_name = config['model']['embedding_model_name']

    index_path = os.path.join(output_dir, 'faiss_index.bin')
    corpus_path = os.path.join(output_dir, 'corpus.json')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the knowledge base
    try:
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"Loaded {len(documents)} documents from the knowledge base.")
    except FileNotFoundError:
        print(f"Error: Knowledge base file not found at '{knowledge_base_path}'.")
        return

    # The corpus is the text content we want to embed and retrieve
    corpus = [doc['content'] for doc in documents]

    # 2. Initialize the embedding model
    print("Loading sentence transformer model ('all-MiniLM-L6-v2')...")
    # This model is small, efficient, and great for this task.
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Generate embeddings
    print("Generating embeddings for the corpus... This may take a moment.")
    embeddings = model.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
    
    # Ensure embeddings are in the right format (float32) for FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    # 4. Build the FAISS index
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings)
    
    print(f"FAISS index built successfully with {index.ntotal} vectors.")

    # 5. Save the index and the corpus mapping
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to '{index_path}'.")
    
    # The corpus file maps the index position to the original document content
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"Corpus mapping saved to '{corpus_path}'.")
    
    print("Vector store build process complete.")

if __name__ == "__main__":
    main()