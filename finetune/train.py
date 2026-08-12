"""
Nexalith Foreman — LoRA fine-tuning script.

Hardware target: AGH Cloud RTX 6000 Ada (48GB VRAM) or similar.
This version uses plain HuggingFace Transformers, PEFT, and TRL.
It bypasses Unsloth due to dependency conflicts on the cloud instance.

Run this on the cloud instance AFTER:
  1. Confirming finetune_examples.jsonl is present
  2. pip install -q transformers peft trl bitsandbytes accelerate datasets

Output: saves the LoRA adapter to ./foreman-finetuned/lora_adapter
and a merged model in safetensors format to ./foreman-finetuned/merged
"""

import json
import torch
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

# Standard Qwen repository works everywhere
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./foreman-finetuned"
DATASET_PATH = "./finetune_examples.jsonl"

MAX_SEQ_LENGTH = 4096
DTYPE = torch.bfloat16

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1

# ------------------------------------------------------------------ #
# Load base model                                                      #
# ------------------------------------------------------------------ #

print("Loading base model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for 4-bit training (gradients checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ------------------------------------------------------------------ #
# Load and format dataset                                              #
# ------------------------------------------------------------------ #

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def format_example(example: dict) -> str:
    # Qwen2.5 chat template
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

print("Loading dataset...")
raw_examples = load_jsonl(DATASET_PATH)
print(f"  {len(raw_examples)} examples loaded")

formatted = [{"text": format_example(ex)} for ex in raw_examples]
dataset = Dataset.from_list(formatted)

print("Sample formatted example (first 300 chars):")
print(dataset[0]["text"][:300])
print("...")


# ------------------------------------------------------------------ #
# Train                                                                #
# ------------------------------------------------------------------ #

print("\nStarting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,             
        bf16=True,              
        logging_steps=5,
        save_strategy="no",     
        output_dir=OUTPUT_DIR,
        report_to="none",       
        seed=42,
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining complete.")
print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")


# ------------------------------------------------------------------ #
# Save LoRA adapter                                                    #
# ------------------------------------------------------------------ #

adapter_path = f"{OUTPUT_DIR}/lora_adapter"
print(f"\nSaving LoRA adapter to {adapter_path}...")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)


# ------------------------------------------------------------------ #
# Merge and Export Model                                               #
# ------------------------------------------------------------------ #

print("\nMerging LoRA adapter with base model...")
# To merge, we need to reload the base model in fp16/bf16 (not 4-bit)
# then apply the adapter and merge.

del model
del trainer
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="cpu", # load to CPU first to avoid OOM
)

merged_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = merged_model.merge_and_unload()

merged_path = f"{OUTPUT_DIR}/merged"
print(f"Saving merged model to {merged_path}...")
merged_model.save_pretrained(merged_path, safe_serialization=True)
tokenizer.save_pretrained(merged_path)

print(f"\nDone. Merged HF model saved to {merged_path}/")
print("To convert to GGUF, run the following locally or on the instance:")
print("  python3 /path/to/llama.cpp/convert_hf_to_gguf.py ./foreman-finetuned/merged --outfile foreman-finetuned.gguf --outtype q8_0")
