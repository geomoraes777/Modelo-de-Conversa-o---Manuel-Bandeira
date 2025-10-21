# cell: train_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "gpt2"
OUTPUT_DIR = "outputs/peft_bandeira"

# carregar tokenizer e tokenized dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token":"<|pad|>"})

train_ds = load_from_disk("data/processed/tokenized/train_tokenized")
val_ds   = load_from_disk("data/processed/tokenized/val_tokenized")

# carregar modelo (pequeno). Para economizar VRAM, não usar load_in_8bit se bitsandbytes faltar.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# configurar LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "q_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# training args (seguros pra sua GPU)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    fp16=torch.cuda.is_available(),  # usa fp16 se cuda disponível
    learning_rate=3e-4,
    warmup_steps=50,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)

print("Iniciando treino. CUDA disponível?:", torch.cuda.is_available())
train_result = trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Treino finalizado. Checkpoint salvo em", OUTPUT_DIR)
print("Train result keys:", train_result.keys())
