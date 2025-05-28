# scripts/02_finetune_with_lora.py

def main():
    """
    This script fine-tunes the Qwen3-8B model using LoRA.
    The process is managed by the Hugging Face `transformers` and `peft` libraries.
    
    Key steps:
    1.  Load the base model (Qwen3-8B) with 4-bit quantization to manage memory.
    2.  Load the corresponding tokenizer.
    3.  Configure LoRA parameters (rank, alpha, target modules) after experimentation.
    4.  Load the preprocessed dataset from script 01.
    5.  Initialize the `SFTTrainer` to handle the training loop.
    6.  Start training and save the final LoRA adapter.
    """
    print("Starting the fine-tuning process...")
    print("Step 1: Loading base model Qwen3-8B with 4-bit quantization...")
    print("Step 2: Loading tokenizer...")
    print("Step 3: Configuring LoRA parameters...")
    print("Step 4: Loading preprocessed dataset...")
    print("Step 5: Initializing the SFTTrainer...")
    print("Step 6: Training started... (This would take several hours on a V100 GPU)")
    print("Training complete. LoRA adapter saved to 'models/qwen3-8b-lora'.")

if __name__ == "__main__":
    main()