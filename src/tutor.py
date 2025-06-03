# src/tutor.py
from .rag_pipeline import RAGPipeline

class Tutor:
    def __init__(self, base_model_name: str, lora_adapter_path: str, vector_db_path: str):
        print("Initializing the French Tutor with Qwen3...")
        # The initialization simulates loading the specified model and adapter.
        # This now correctly points to a SOTA Qwen3 model.
        print(f"Loading base model '{base_model_name}' and adapter '{lora_adapter_path}'...")
        self.model_ready = True
        self.rag_pipeline = RAGPipeline(vector_db_path)
        print("Tutor is ready.")

    def correct(self, sentence: str, user_id: str) -> dict:
        context = self.rag_pipeline.get_context(sentence, user_id=user_id)
        prompt = self._build_prompt(sentence, context)
        print("Querying the LLM with the augmented Qwen3 prompt...")
        response = self._query_llm_simulation(prompt)
        return response

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