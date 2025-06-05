# main.py
import yaml
from src.tutor import Tutor

def main():
    """
    Main entry point for demonstrating the Tutor's functionality.
    Loads configuration from config.yaml to initialize the system.
    """
    print("--- ParlAIs FrancAIs Demonstration ---")
    
    # Load configuration from the YAML file
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure the file exists.")
        return

    # Initialize the tutor using settings from the config file
    tutor = Tutor(
        base_model_name=config['model']['base_model_name'],
        lora_adapter_path=config['model']['lora_adapter_path'],
        vector_db_path=config['paths']['vector_db'],
        user_profile_db_path=config['paths']['user_profiles']
    )
    
    # --- The rest of the demonstration remains the same ---
    user_id = "user_123"
    sentence = "Je vais à le parc."
    print(f"\n--- Correcting sentence for {user_id}: '{sentence}' ---")
    response = tutor.correct(sentence, user_id=user_id)
    print("Tutor Response:", response)

if __name__ == "__main__":
    main()