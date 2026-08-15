# =============================================================================
# Nexalith Foreman — LoRA fine-tune on a FREE Colab T4 (no paid GPU credit needed)
#
# HOW TO USE:
#   1. Open a new notebook at https://colab.research.google.com
#   2. Runtime -> Change runtime type -> T4 GPU
#   3. Upload nexalith_os_dataset.jsonl and nexalith_os_tools.json to the
#      Colab file browser (left sidebar), or mount Google Drive
#   4. Paste this whole file into one cell (or split at the "# ---" markers
#      into multiple cells) and run
#   5. Total time on a free T4: ~5-10 minutes given only 19 examples
# =============================================================================

# --- Cell 1: install ---------------------------------------------------------
# !pip install unsloth
# !pip install --upgrade --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# --- Cell 2: load base model in 4-bit ----------------------------------------
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",   # same base your REPORT.md names
    max_seq_length=max_seq_length,
    dtype=None,          # auto-detect
    load_in_4bit=True,   # QLoRA — this is what makes a free T4 (16GB) enough
)

# --- Cell 3: attach LoRA adapters --------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- Cell 4: build training text from your dataset + tool schema ------------
import json
from datasets import Dataset

with open("nexalith_os_tools.json") as f:
    tools = json.load(f)

raw_examples = []
with open("nexalith_os_dataset.jsonl") as f:
    for line in f:
        raw_examples.append(json.loads(line))

def to_text(example):
    # Uses Qwen2.5's native tool-calling chat template so the formatting
    # exactly matches what agent/orchestrator.py sends to llama-server.
    return tokenizer.apply_chat_template(
        example["messages"],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )

texts = [to_text(ex) for ex in raw_examples]
dataset = Dataset.from_dict({"text": texts})
print(f"Training on {len(dataset)} examples")
print("--- sample formatted example ---")
print(texts[0][:800])

# --- Cell 5: train ------------------------------------------------------------
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,  # dataset is tiny — no need to pack sequences
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=6,          # small dataset -> a few more passes is fine
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

# --- Cell 6: quick sanity check against your real test prompts --------------
FastLanguageModel.for_inference(model)

test_prompts = [
    "One of our sales reps has three deals that have had no activity in over "
    "two weeks. Find them and draft a short follow-up message for each.",
    "We just hired a new employee starting Monday. Set up their onboarding "
    "checklist, and let me know what CMS content needs to be published for "
    "the new-hire announcement.",
]

for p in test_prompts:
    messages = [
        {"role": "system", "content": raw_examples[0]["messages"][0]["content"]},
        {"role": "user", "content": p},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=300, temperature=0.3)
    print("=" * 60)
    print("PROMPT:", p)
    print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))

# --- Cell 7: export straight to quantized GGUF -------------------------------
# This single call does merge + convert + quantize in one step —
# replaces the separate merge/convert/quantize steps from your first run.
model.save_pretrained_gguf(
    "nexalith_foreman_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
# Result: nexalith_foreman_gguf/*.gguf  (~1.8-2GB, download this from the
# Colab file browser, or push straight to HF — see Cell 8)

# --- Cell 8 (optional): push straight to a public HF repo --------------------
# Do this INSTEAD of Cell 7's local save if you want it hosted in one step.
# Get a token (write access) from https://huggingface.co/settings/tokens first.
#
# model.push_to_hub_gguf(
#     "jrcity/nexalith-foreman-q4km",     # change to your HF username
#     tokenizer,
#     quantization_method="q4_k_m",
#     token="hf_xxxxxxxxxxxxxxxxxxxx",
# )
#
# Then update download_model.sh's MODEL_URL to:
# https://huggingface.co/jrcity/nexalith-foreman-q4km/resolve/main/<filename>.gguf
