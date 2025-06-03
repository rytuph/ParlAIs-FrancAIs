# src/rag_pipeline.py
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    """
    Manages the Retrieval-Augmented Generation (RAG) pipeline.
    
    This class is responsible for loading a vector store of grammar rules and a
    database of user profiles. It retrieves relevant context based on a user's
    input sentence and their learning history to augment the LLM's prompt.
    """
    def __init__(self, vector_db_path: str, user_profile_db_path: str, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the RAG pipeline by loading all necessary components.
        
        Args:
            vector_db_path (str): Path to the directory containing the FAISS index and corpus.
            user_profile_db_path (str): Path to the JSON file containing user profiles.
            model_name (str): The name of the sentence transformer model to use for embeddings.
        """
        print("Initializing RAG pipeline...")
        self.vector_db_path = vector_db_path
        self.user_profile_db_path = user_profile_db_path
        
        try:
            # Load the sentence transformer model for encoding queries
            print(f"Loading sentence transformer model: '{model_name}'...")
            self.embedding_model = SentenceTransformer(model_name)
            
            # Load the FAISS index from disk
            index_path = os.path.join(self.vector_db_path, 'faiss_index.bin')
            self.index = faiss.read_index(index_path)
            print(f"FAISS index loaded successfully from '{index_path}'.")

            # Load the corpus that maps index positions to text
            corpus_path = os.path.join(self.vector_db_path, 'corpus.json')
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
            print(f"Corpus with {len(self.corpus)} documents loaded from '{corpus_path}'.")

        except FileNotFoundError as e:
            print(f"Error initializing RAG pipeline: {e}")
            print("Please ensure you have run 'scripts/03_build_vector_store.py' to generate the necessary files.")
            raise
            
        self.is_ready = True
        print("RAG pipeline is ready.")

    def get_context(self, sentence: str, user_id: str) -> str:
        """
        Retrieves and combines context from the grammar knowledge base and user profile.
        
        Args:
            sentence (str): The user's input sentence.
            user_id (str): The unique identifier for the user.
            
        Returns:
            str: A combined string of retrieved context to be injected into the LLM prompt.
        """
        # 1. Retrieve the most relevant grammar rule from the vector store
        grammar_context = self._query_grammar_db(sentence)
        
        # 2. Retrieve personalized context from the user's learning history
        user_context = self._query_user_profile(user_id)
        
        return f"{grammar_context}\n{user_context}"

    def _query_grammar_db(self, query_sentence: str, k: int = 1) -> str:
        """
        Encodes the query sentence and performs a vector search on the FAISS index.
        
        Args:
            query_sentence (str): The sentence to search for.
            k (int): The number of top results to retrieve.
            
        Returns:
            str: The content of the most relevant grammar rule document.
        """
        query_embedding = self.embedding_model.encode([query_sentence], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search the FAISS index for the most similar vector
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve the document content using the index
        if len(indices) > 0 and len(indices[0]) > 0:
            retrieved_doc_index = indices[0][0]
            retrieved_doc = self.corpus[retrieved_doc_index]
            return f"Retrieved Grammar Rule: {retrieved_doc}"
        
        return "Retrieved Grammar Rule: No specific rule was found to be highly relevant. Relying on the model's general knowledge."

    def _query_user_profile(self, user_id: str) -> str:
        """
        Loads the user profile database and finds the user's most common error.
        
        Args:
            user_id (str): The unique identifier for the user.
            
        Returns:
            str: A string summarizing the user's learning history or a neutral message.
        """
        try:
            with open(self.user_profile_db_path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
            
            user_profile = profiles.get(user_id)
            
            if user_profile and "error_counts" in user_profile:
                # Find the error topic with the highest count
                most_common_error = max(user_profile["error_counts"], key=user_profile["error_counts"].get)
                error_count = user_profile["error_counts"][most_common_error]
                return f"Retrieved User Profile: User frequently struggles with '{most_common_error}' (logged {error_count} times). Provide extra clarity on this topic."
                
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or is empty, return a neutral message
            return "Retrieved User Profile: No user profile database found."
            
        return "Retrieved User Profile: No specific weaknesses logged for this user."