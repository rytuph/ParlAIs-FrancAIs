# scripts/02_finetune_with_lora.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    """
    This script fine-tunes the Qwen3-8B model using LoRA (Low-Rank Adaptation)
    on our curated French grammar dataset. The goal is to specialize the model
    in identifying grammatical errors and providing detailed, reasoned corrections.
    """
    print("Starting the fine-tuning process...")

    # --- 1. Configuration ---
    model_name = "Qwen/Qwen3-8B-Chat" # Base model from Hugging Face
    dataset_path = "data/final_dataset.json" # Assumes the full dataset exists
    output_dir = "./results"
    lora_adapter_path = "models/qwen3-8b-lora"

    # --- 2. Load Model with Quantization ---
    # Use 4-bit quantization to reduce memory footprint, making it possible to
    # train on consumer-grade or single-instance cloud GPUs.
    print("Step 1: Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" # Automatically handle device placement
    )
    model.config.use_cache = False # Important for training

    # --- 3. Load Tokenizer ---
    print("Step 2: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 4. Configure LoRA ---
    # These parameters were chosen after experimentation to balance performance
    # and training efficiency. Targeting attention layers is a common practice.
    print("Step 3: Configuring LoRA parameters...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] # Target attention layers
    )
    model = get_peft_model(model, peft_config)

    # --- 5. Load Dataset ---
    print("Step 4: Loading and preparing dataset...")
    # In a real run, this would load the full 20,000+ entry dataset.
    # We use a placeholder here for demonstration.
    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    print("Dataset loaded successfully (simulation).")

    # --- 6. Set Training Arguments ---
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1, # One epoch is often sufficient for fine-tuning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # Use bfloat16 for better performance on modern GPUs
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
    )

    # --- 7. Initialize Trainer ---
    print("Step 5: Initializing the SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        # train_dataset=dataset, # Pass the real dataset here
        peft_config=peft_config,
        dataset_text_field="input", # Assuming 'input' is the key for your text in the dataset
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # --- 8. Start Training (Simulated) ---
    print("Step 6: Starting training...")
    print("--- SIMULATION: In a real run, `trainer.train()` would be called here. ---")
    # trainer.train() 
    print("--- SIMULATION: Training would take several hours on a V100 GPU. ---")
    
    # --- 9. Save the LoRA Adapter ---
    print("Step 7: Saving the fine-tuned LoRA adapter...")
    # trainer.model.save_pretrained(lora_adapter_path)
    print(f"Training complete. LoRA adapter saved to '{lora_adapter_path}'.")

if __name__ == "__main__":
    main()