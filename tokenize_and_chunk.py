# cell: tokenize_and_chunk.py
from transformers import GPT2TokenizerFast
from datasets import load_dataset, Dataset
from pathlib import Path
import json

PROC_DIR = Path("data/processed")
TRAIN = str(PROC_DIR / "train.jsonl")
VAL   = str(PROC_DIR / "validation.jsonl")

print("Carregando datasets:", TRAIN, VAL)
ds = load_dataset("json", data_files={"train": TRAIN, "validation": VAL})

MODEL_NAME = "gpt2"   # base model
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token":"<|pad|>"})

block_size = 256   # comprimento do chunk (ajuste: 128/256/512). 256 é conservador para 4GB VRAM.
stride = 64        # overlap entre janelas

def tokenize_function(examples):
    # tokeniza preservando \n. truncation False para agrupar depois
    return tokenizer(examples["text"], add_special_tokens=False)

tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"])

# group_texts: cria blocos de size block_size com stride (overlap)
def group_texts(examples):
    # concatena todos os input_ids
    concatenated = []
    for seq in examples["input_ids"]:
        concatenated.extend(seq)
    total_length = len(concatenated)
    if total_length < block_size:
        return {"input_ids": [], "attention_mask": []}
    # criar janelas
    result_input_ids = []
    for i in range(0, total_length - block_size + 1, block_size - stride):
        chunk = concatenated[i: i + block_size]
        result_input_ids.append(chunk)
    return {"input_ids": result_input_ids}

grouped_train = tokenized["train"].map(group_texts, batched=True, remove_columns=tokenized["train"].column_names)
grouped_val   = tokenized["validation"].map(group_texts, batched=True, remove_columns=tokenized["validation"].column_names)

# transformar em datasets tokenizados com attention mask
def to_dataset(grouped):
    # cada item já é uma lista de input_ids (por batched mapping)
    import numpy as np
    inputs = []
    for row in grouped["input_ids"]:
        for seq in row:
            inputs.append({"input_ids": seq, "attention_mask": [1]*len(seq)})
    return Dataset.from_list(inputs)

train_ds = to_dataset(grouped_train)
val_ds   = to_dataset(grouped_val)

print("Tamanho train (chunks):", len(train_ds))
print("Tamanho val   (chunks):", len(val_ds))

# salvar tokenized em disco (opcional)
out_dir = PROC_DIR / "tokenized"
out_dir.mkdir(exist_ok=True)
train_ds.save_to_disk(str(out_dir / "train_tokenized"))
val_ds.save_to_disk(str(out_dir / "val_tokenized"))
print("Tokenized datasets salvos em:", out_dir)
