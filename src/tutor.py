# src/tutor.py
from .rag_pipeline import RAGPipeline

class Tutor:
    """
    The main class that orchestrates the grammar correction process.
    """
    def __init__(self, model_path: str, vector_db_path: str):
        print("Initializing the French Tutor...")
        # In a real implementation, this would load the fine-tuned LLM and tokenizer.
        print(f"Loading fine-tuned model from: {model_path}...")
        self.model_ready = True
        
        # Initialize the RAG pipeline
        self.rag_pipeline = RAGPipeline(vector_db_path)
        print("Tutor is ready to use.")

    def correct(self, sentence: str, user_id: str) -> dict:
        """
        Takes a sentence and a user ID, and returns a correction and explanation.
        """
        if not self.model_ready or not self.rag_pipeline.is_ready:
            return {"error": "Tutor is not initialized."}
            
        # 1. Get context from the RAG pipeline
        context = self.rag_pipeline.get_context(sentence, user_id)
        
        # 2. Build the augmented prompt for the LLM
        prompt = self._build_prompt(sentence, context)
        
        # 3. Query the LLM (Simulated Response)
        print("Querying the LLM with the augmented prompt...")
        response = self._query_llm(prompt)
        
        # 4. Update user profile (Simulated)
        self._update_user_profile(user_id, response)
        
        return response

    def _build_prompt(self, sentence: str, context: str) -> str:
        """Constructs the final prompt for the LLM."""
        return f"""
        Instruction: Analyze the user's French sentence based on the provided context. 
        If it contains a grammatical error, provide the corrected sentence and a detailed, 
        step-by-step explanation of the rule that was broken.

        Context: {context}

        User Sentence: {sentence}

        Output:
        """

    def _query_llm(self, prompt: str) -> dict:
        # This method simulates the response from the actual fine-tuned LLM.
        if "le parc" in prompt:
            return {
              "correction": "Je vais au parc.",
              "explanation": "Bon travail ! The noun 'parc' is masculine, so the preposition 'à' must contract with the article 'le' to become 'au'. I see you've been working on this, and you're very close to mastering it!"
            }
        if "j'ai besoin" in prompt:
            return {
              "correction": "C'est le livre dont j'ai besoin.",
              "explanation": "Très bien ! The verb 'avoir besoin' is always followed by the preposition 'de'. When this is the object of a relative clause, you must use the pronoun 'dont'. This is a common point of confusion!"
            }
        return {
            "correction": "Sentence appears correct.",
            "explanation": "I couldn't find any grammatical errors in this sentence. Well done!"
        }

    def _update_user_profile(self, user_id: str, response: dict):
        # Placeholder for the logic that would update the user's error history.
        print(f"Updating profile for user '{user_id}' based on the last correction...")
        pass