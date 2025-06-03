# scripts/01_prepare_dataset.py
import pandas as pd
import json

def create_instruction(row):
    """
    Formats a row of raw data into the final instruction-tuning JSON structure.
    This function enriches the concise explanation notes into a more complete,
    human-like response suitable for training an educational model.
    """
    instruction = "Analyze the user's French sentence. If it contains a grammatical error, provide the corrected sentence and a detailed, step-by-step explanation of the rule that was broken. The explanation should be encouraging and educational."
    
    explanation_map = {
        "'pomme' is feminine, use 'une' not 'un'": "Excellent try! The error here is with the article. The noun 'pomme' (apple) is a feminine noun in French. Therefore, you need to use the feminine indefinite article 'une' instead of the masculine 'un'. Keep up the great work!",
        "'avoir besoin de', relative clause requires 'dont'": "Très bien ! The verb 'avoir besoin' is always followed by the preposition 'de'. When this is the object of a relative clause, you must use the pronoun 'dont'. This is a common point of confusion!",
        "use 'Ce sont' for identification, preposition 'avec' moves before 'qui'": "Good sentence structure! In French, it's more natural to use 'Ce sont' instead of 'Ils sont' when identifying people. Also, the preposition 'avec' should come before the relative pronoun 'qui'. You're doing great!"
    }
    
    # Use the map to get the full explanation, or provide a default if a note is not found
    full_explanation = explanation_map.get(row['explanation_notes'], "A grammatical rule has been applied to correct this sentence.")

    formatted_output = {
        "correction": row['correct_sentence'],
        "explanation": full_explanation
    }
    
    return {
        "instruction": instruction,
        "input": row['incorrect_sentence'],
        "output": formatted_output
    }

def main():
    """
    Processes the raw data from CSV, transforms it into the instruction-tuning
    format required by the SFTTrainer, and saves the result as a JSON file.
    """
    print("Starting dataset preparation...")
    
    raw_data_path = 'data/raw_data.csv'
    output_path = 'data/sample_dataset.json'

    try:
        raw_df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(raw_df)} records from '{raw_data_path}'.")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{raw_data_path}'.")
        return

    print("Formatting data into instruction-tuning format...")
    formatted_data = raw_df.apply(create_instruction, axis=1).tolist()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset preparation complete. Formatted sample saved to '{output_path}'.")

if __name__ == "__main__":
    main()