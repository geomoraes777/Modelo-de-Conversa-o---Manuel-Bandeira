import os
# A importação do GPT2Tokenizer foi corrigida/adicionada aqui
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,  # <-- CORREÇÃO: Adicionado aqui
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
# A importação do print_trainable_parameters foi removida daqui
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# --- INÍCIO DA FUNÇÃO AJUDANTE (print_trainable_parameters) ---
# CORREÇÃO: Definimos a função manualmente para evitar erros de importação
def print_trainable_parameters(model):
    """
    Imprime o número de parâmetros treináveis no modelo.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )
# --- FIM DA FUNÇÃO AJUDANTE ---


# --- 0. Definições Iniciais ---
model_name = "gpt2"
output_dir = "outputs/lora_gpt2"
final_model_path = "outputs/lora_gpt2_final"

# --- 1. Carrega Tokenizador ---
print(f"Carregando tokenizador para o modelo: {model_name}")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Correção: GPT-2 não tem 'pad_token'. Usamos 'eos_token' para isso.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Token de preenchimento (pad_token) definido como eos_token.")

# --- 2. Carrega Modelo ---
print(f"Carregando modelo: {model_name}")
model = GPT2LMHeadModel.from_pretrained(model_name)

# Redimensiona embeddings.
model.resize_token_embeddings(len(tokenizer))
print(f"Embeddings redimensionados para o tamanho do tokenizador: {len(tokenizer)}")

# --- 3. Configura LoRA ---
print("Configurando LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # Módulos de atenção do GPT-2
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Imprime um resumo de quais parâmetros são treináveis
print_trainable_parameters(model) # Agora usa a função que definimos acima

# --- 4. Carrega e Processa os Dados ---
# !!! ATENÇÃO: Esta é a parte mais importante que você deve adaptar !!!
# O código abaixo é um EXEMPLO que usa um dataset de teste da Hugging Face.
# Você deve substituí-lo para carregar seus próprios arquivos (ex: .txt, .csv).

print("Carregando e processando dados...")
try:
    # Exemplo: Carregando um dataset de texto (wikitext)
    # Para usar seus próprios arquivos:
    # raw_datasets = load_dataset("text", data_files={"train": "caminho/para/seu_treino.txt", 
    #                                                 "validation": "caminho/para/sua_valid.txt"})
    
    # Usando um dataset de exemplo para teste:
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split={'train': 'train[:20%]', 'validation': 'validation[:20%]'
})

    # Função para tokenizar os dados
    def tokenize_function(examples):
        # 'text' é o nome da coluna no dataset wikitext. Mude se o seu for diferente.
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    # Aplica a tokenização em lote (mais rápido)
    tokenized_datasets = raw_datasets.map(
        tokenize_function, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names
    )
    
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    print(f"Dados carregados: {len(train_dataset)} exemplos de treino, {len(val_dataset)} exemplos de validação.")

except Exception as e:
    print(f"ERRO: Não foi possível carregar os dados: {e}")
    print("O treino não pode continuar sem dados.")
    train_dataset = None
    val_dataset = None

# --- 5. Data Collator ---
# Agrupa os dados em lotes para o modelo
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # MLM=False significa que é Causal LM (como GPT), não Masked LM (como BERT)
)

# --- 6. Argumentos de Treino ---
print("Configurando argumentos de treino...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,   # Reduza se tiver erro de "Out of Memory"
    per_device_eval_batch_size=4,
    num_train_epochs=1,              # Aumente para 3 ou mais para um treino real
    learning_rate=2e-4,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    save_total_limit=2,
    fp16=False, # Mude para True SÓ SE tiver GPU NVIDIA com CUDA. Mantenha False para CPU.
    report_to="none"
)

# --- 7. Inicializa o Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 8. Treino ---
if train_dataset is not None:
    print(">>> Iniciando o treino <<<")
    trainer.train()

    # --- 9. Salva o Modelo Final ---
    print("Treino concluído!")
    print(f"Salvando modelo e tokenizador em: {final_model_path}")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("Processo finalizado.")
else:
    print("---------------------------------------------------------------")
    print("ERRO: 'train_dataset' está vazio. O treino foi cancelado.")
    print("Por favor, verifique a Seção 4 e forneça seus dados de treino.")
    print("---------------------------------------------------------------")