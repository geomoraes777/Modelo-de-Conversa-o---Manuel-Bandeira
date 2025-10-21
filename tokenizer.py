import json
from datasets import Dataset
from transformers import GPT2Tokenizer

# Caminhos
processed_jsonl = "data/processed/manuel_bandeira_corpus_processed.jsonl"

# Lê JSONL e transforma em lista de dicionários
data = []
with open(processed_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Apenas textos para treino/validação
texts = [item["text"] for item in data]

# Divide em treino/validação manualmente (como seu preprocess já fez)
train_texts = texts[:44]
val_texts = texts[44:]

# Inicializa tokenizer do GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 não tem pad token nativo

# Tokeniza
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove coluna original
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

# Ajuste de formato para PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")

print("Datasets prontos:")
print(train_dataset)
print(val_dataset)
