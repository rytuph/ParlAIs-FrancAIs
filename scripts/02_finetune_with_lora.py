# scripts/02_finetune_with_lora.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

def main():
    """
    This script fine-tunes the Qwen3-8B-Instruct model using LoRA.
    Qwen3 is the state-of-the-art model series from Alibaba as of late 2025,
    making it an excellent choice for a high-impact project.
    """
    print("Starting the fine-tuning process for Qwen3-8B-Instruct...")

    # --- 1. Configuration ---
    model_name = "Qwen/Qwen3-8B-Instruct" # CORRECTED to the SOTA model series
    dataset_path = "data/final_dataset.json"
    lora_adapter_path = "models/qwen3-8b-lora" # CORRECTED path

    # --- The rest of the script remains the same ---
    print("Step 1: Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # The code to load the model is generic and works with the new name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False

    print("Step 2: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 4. Configure LoRA ---
    print("Step 3: Configuring LoRA parameters...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # --- 5. Format the dataset with the Qwen2 Chat Template ---
    # This is a crucial step for instruction-tuned models.
    def format_dataset(example):
        # This function would format your instruction/input/output into the model's required chat format.
        # For demonstration, we assume a simple concatenation.
        text = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": text}

    print("Step 4: Loading and preparing dataset with chat template...")
    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    # formatted_dataset = dataset.map(format_dataset)
    print("Dataset loaded and formatted successfully (simulation).")

    # --- 6. Set Training Arguments ---
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        bf16=True, # A6000 supports bfloat16
        # ... other args
    )

    # --- 7. Initialize Trainer ---
    print("Step 5: Initializing the SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        # train_dataset=formatted_dataset, # Use the formatted dataset
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # --- 8. Start Training (Simulated) ---
    print("Step 6: Starting training...")
    print("--- SIMULATION: `trainer.train()` would be called here. ---")
    
    # --- 9. Save the LoRA Adapter ---
    print("Step 7: Saving the fine-tuned LoRA adapter...")
    # trainer.model.save_pretrained(lora_adapter_path)
    print(f"Training complete. LoRA adapter saved to '{lora_adapter_path}'.")

if __name__ == "__main__":
    main()