"""
Nexalith Foreman — LoRA fine-tuning script.

Hardware target: AGH Cloud V100 (64GB VRAM), float16 (V100 does not
support bfloat16). Uses Unsloth for fast, memory-efficient training.

Run this on the cloud instance AFTER:
  1. pip install unsloth
  2. Uploading finetune_examples.jsonl to the instance

Output: saves a merged, GGUF-quantized model to ./foreman-finetuned/
which you download and replace your local model with.

Estimated runtime: 5-15 minutes on a V100 for 15 examples x 3 epochs.
"""

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./foreman-finetuned"
DATASET_PATH = "./finetune_examples.jsonl"

MAX_SEQ_LENGTH = 4096
DTYPE = torch.float16          # float16 required for V100
LOAD_IN_4BIT = True            # keeps VRAM usage low even on 64GB

LORA_R = 16                    # rank — higher = more capacity, more VRAM
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

BATCH_SIZE = 2
GRAD_ACCUM = 4                 # effective batch = 8
EPOCHS = 3                     # 3 passes over 15 examples = 45 effective steps
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1


# ------------------------------------------------------------------ #
# Load base model                                                      #
# ------------------------------------------------------------------ #

print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ------------------------------------------------------------------ #
# Load and format dataset                                              #
# ------------------------------------------------------------------ #

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_example(example: dict) -> str:
    """
    Convert a messages-format example into a single string using
    Qwen2.5's chat template. The tokenizer handles the exact format
    including special tokens — we just pass the messages list.
    """
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
        fp16=True,              # float16 for V100
        bf16=False,             # bf16 NOT supported on V100
        logging_steps=5,
        save_strategy="no",     # save only at the end
        output_dir=OUTPUT_DIR,
        report_to="none",       # no wandb/hub needed
        seed=42,
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining complete.")
print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")


# ------------------------------------------------------------------ #
# Save LoRA adapter + export to GGUF                                   #
# ------------------------------------------------------------------ #

print("\nSaving LoRA adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

print("Merging and exporting to GGUF (Q4_K_M)...")
# This merges the LoRA weights into the base model and quantizes
# directly to GGUF Q4_K_M — the exact format your llama-server expects.
model.save_pretrained_gguf(
    f"{OUTPUT_DIR}/gguf",
    tokenizer,
    quantization_method="q4_k_m",
)

print(f"\nDone. Files saved to {OUTPUT_DIR}/")
print("Download foreman-finetuned/gguf/*.gguf to your local machine.")
print("Replace your model file and restart llama-server to test.")
