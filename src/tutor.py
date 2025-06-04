# src/tutor.py
import json
from datetime import datetime
from .rag_pipeline import RAGPipeline

class Tutor:
    def __init__(self, base_model_name: str, lora_adapter_path: str, vector_db_path: str, user_profile_db_path: str):
        print("Initializing the French Tutor with Qwen3...")
        # The initialization simulates loading the specified model and adapter.
        # This now correctly points to a SOTA Qwen3 model.
        print(f"Loading base model '{base_model_name}' and adapter '{lora_adapter_path}'...")
        self.model_ready = True
        self.rag_pipeline = RAGPipeline(vector_db_path)
        print("Tutor is ready.")

        self.user_profile_db_path = user_profile_db_path
        self.rag_pipeline = RAGPipeline(vector_db_path, user_profile_db_path, 'data/grammar_knowledge_base.json')

    def correct(self, sentence: str, user_id: str) -> dict:
        # 1. Get context and topic from the RAG pipeline
        retrieved_info = self.rag_pipeline.get_context_with_topic(sentence, user_id)
        context = retrieved_info['content']
        error_topic = retrieved_info['topic']

        # 2 (build prompt and query LLM)
        context = self.rag_pipeline.get_context(sentence, user_id=user_id)
        prompt = self._build_prompt(sentence, context)
        print("Querying the LLM with the augmented Qwen3 prompt...")
        response = self._query_llm_simulation(prompt)

        # 4. Update user profile with the identified error topic
        if "correction" in response and response["correction"] != "Sentence appears correct.":
            self._update_user_profile(user_id, error_topic)
        
        return response
    
    def _update_user_profile(self, user_id: str, error_topic: str):
        """
        Loads, updates, and saves the user profile JSON file.
        Increments the count for the identified error topic for the given user.
        """
        try:
            with open(self.user_profile_db_path, 'r') as f:
                profiles = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            profiles = {}

        # Get user profile, or create it if it doesn't exist
        user_profile = profiles.get(user_id, {"error_counts": {}})
        error_counts = user_profile.get("error_counts", {})
        
        # Increment the count for the specific error topic
        error_counts[error_topic] = error_counts.get(error_topic, 0) + 1
        
        # Update the profile
        user_profile["error_counts"] = error_counts
        user_profile["last_seen"] = datetime.utcnow().isoformat()
        profiles[user_id] = user_profile
        
        # Write the updated profiles back to the file
        with open(self.user_profile_db_path, 'w') as f:
            json.dump(profiles, f, indent=2)
        
        print(f"Updated profile for user '{user_id}'. New count for '{error_topic}': {error_counts[error_topic]}.")

    def _build_prompt(self, sentence: str, context: str) -> str:
        """
        Constructs the final prompt using the official Qwen ChatML template,
        which is compatible with the Qwen3 series.
        """
        system_message = "You are an expert French language tutor..." # (message is the same)
        user_message = f"Context: {context}\nSentence: {sentence}"
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def _query_llm_simulation(self, prompt: str) -> dict:
        """Simulates the response from the actual fine-tuned LLM."""
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