# main.py
from src.tutor import Tutor

def main():
    """
    Main entry point for demonstrating the Tutor's functionality.
    """
    print("--- ParlAIs FrancAIs Demonstration ---")
    
    # Initialize the tutor, loading the model and RAG pipeline
    french_tutor = Tutor(model_path="models/qwen3-8b-lora", vector_db_path="data/vector_store")
    
    # User session ID allows the RAG pipeline to track learning history
    user_id = "user_123"
    
    # --- Example 1 ---
    print("\n--- Correcting sentence 1 ---")
    sentence1 = "Je vais à le parc."
    response1 = french_tutor.correct(sentence1, user_id=user_id)
    print("User Sentence:", sentence1)
    print("Tutor Response:", response1)
    
    # --- Example 2 ---
    print("\n--- Correcting sentence 2 ---")
    sentence2 = "C'est le livre que j'ai besoin."
    response2 = french_tutor.correct(sentence2, user_id=user_id)
    print("User Sentence:", sentence2)
    print("Tutor Response:", response2)

if __name__ == "__main__":
    main()