# ParlAIs FrancAIs: A Personalized French Language Tutor LLM

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Hugging%20Face-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

ParlAIs FrancAIs is an advanced proof-of-concept for a personalized French language tutor. It leverages a fine-tuned small language model (SLM) augmented with a novel dual-component RAG pipeline to provide adaptive, context-aware grammatical corrections and explanations.

## 📝 Problem Statement

Learning the nuances of French grammar is a significant challenge for students. Standard grammar checkers often provide corrections without context, while large-scale LLMs can be generic and fail to adapt to an individual's learning journey. This project addresses the need for a specialized tutor that can not only correct errors but also explain the underlying rules and remember a user's specific weak points over time.

## ✨ Core Features

*   **Specialized Grammar LLM:** Fine-tuned a **Qwen3-8B** model using **LoRA** to specialize in explicit grammatical reasoning for the French language, achieving a **35%+ accuracy improvement** over the baseline model.
*   **Adaptive Learning with RAG:** Engineered a **dual-component RAG pipeline** with a persistent vector store (FAISS) that provides the model with both a static knowledge base of grammar rules and a dynamic memory of the user's past interactions.
*   **High-Quality Instructional Dataset:** Curated and synthesized a bilingual dataset of **20,000+** high-quality French-English grammatical examples with detailed reasoning chains, forming the backbone of the model's performance.
*   **Efficient and Modular:** Built with a modular architecture, enabling efficient inference and clear separation between the language model, the RAG pipeline, and the user-facing logic.

## 🔧 Technical Architecture

The system's core innovation lies in how it augments the LLM's prompt with rich, contextual information retrieved from a vector database. This allows a small, efficient model to perform with the apparent context of a much larger one.

User Input: "Je vais à le parc."
      |
      V
[ RAG Pipeline ]
      |
      |--> 1. Query Vector Store for User History
      |    (Retrieves: "User often confuses 'à + le'")
      |
      |--> 2. Query Vector Store for Grammar Rules
      |    (Retrieves: "Rule: Contraction of 'à' and 'le' to 'au'")
      |
      V
[ Augmented Prompt Generation ]
(Combines user input with retrieved context)
      |
      V
[ Fine-tuned Qwen3-8B LLM ]
(Processes the rich prompt to generate a detailed, personalized response)
      |
      V
Final Output: {
  "correction": "Je vais au parc.",
  "explanation": "Bon travail ! The noun 'parc' is masculine, so the preposition 'à' must contract with the article 'le' to become 'au'. I see you've been working on this, and you're very close to mastering it!"
}
```

## 📚 The Dataset

The performance of this model is primarily driven by a meticulously curated instruction-tuning dataset. The dataset was synthesized from a combination of grammar textbooks, linguistic resources, and synthetic data generation, with a focus on creating detailed, human-like explanations.

Each entry in the dataset follows a structured format:

```json
{
  "instruction": "Analyze the user's French sentence. If it contains a grammatical error, provide the corrected sentence and a detailed, step-by-step explanation of the rule that was broken. The explanation should be encouraging and educational.",
  "input": "J'ai mangé un pomme.",
  "output": {
    "correction": "J'ai mangé une pomme.",
    "explanation": "Excellent try! The error here is with the article. The noun 'pomme' (apple) is a feminine noun in French. Therefore, you need to use the feminine indefinite article 'une' instead of the masculine 'un'. Keep up the great work!"
  }
}
```

## 📊 Performance & Evaluation

The model's effectiveness was measured against the zero-shot performance of the base **Qwen3-8B** model. Using a held-out evaluation set of 500 sentences containing common grammatical errors, the fine-tuned `ParlAIs FrancAIs` model demonstrated:

*   **A 35% relative increase** in its ability to correctly identify, correct, and explain grammatical errors.
*   A significant qualitative improvement in the coherence and educational value of its explanations.

## 🚀 Usage Example

The final prototype was structured into a simple `Tutor` class that encapsulates the entire pipeline.

```python
# main.py
from src.tutor import Tutor

# Initialize the tutor, loading the model and RAG pipeline
french_tutor = Tutor(model_path="models/qwen3-8b-lora", vector_db_path="data/vector_store")

# User session ID allows the RAG pipeline to track learning history
user_id = "user_123"

# Get a correction
sentence = "C'est le livre que j'ai besoin."
response = french_tutor.correct(sentence, user_id=user_id)

print(response)
# Expected Output:
# {
#   "correction": "C'est le livre dont j'ai besoin.",
#   "explanation": "Très bien ! The verb 'avoir besoin' is always followed by the preposition 'de'. When this is the object of a relative clause, you must use the pronoun 'dont'. This is a common point of confusion!"
# }
```

## 🛠️ Setup and Installation

To set up the environment for this project:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ParlAIs-FrancAIs.git
    cd ParlAIs-FrancAIs
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Technologies Used

*   **ML/DL:** PyTorch, Hugging Face (Transformers, PEFT, Accelerate)
*   **Vector Database:** FAISS
*   **Data Manipulation:** Pandas, NumPy
*   **Core Language:** Python

## 🔮 Future Work

*   **Web Interface:** Develop a simple Streamlit or Gradio interface to make the tutor more accessible for interactive use.
*   **Multi-Turn Dialogue:** Expand the RAG's memory component to handle multi-turn conversational context, allowing for follow-up questions.
*   **Active Learning:** Implement an active learning loop where challenging or ambiguous user corrections are flagged for review to continuously improve the training dataset.
```