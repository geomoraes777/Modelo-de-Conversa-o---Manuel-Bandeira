import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# -------------------------
# CONFIGURAÇÕES DE MODELO
# -------------------------
modelo_base = "pierreguillou/gpt2-small-portuguese"
modelo_lora = r"C:\Users\Windows\Desktop\Modelo de Conversação - Manuel Bandeira\outputs\lora_gpt2_final"

# Carrega tokenizer e ajusta pad token
tokenizer = AutoTokenizer.from_pretrained(modelo_base)
tokenizer.pad_token = tokenizer.eos_token

# Carrega modelo base e aplica LoRA
base_model = AutoModelForCausalLM.from_pretrained(modelo_base)
model = PeftModel.from_pretrained(base_model, modelo_lora)

# Cria pipeline (device=-1 usa CPU; se quiser GPU, ajuste device=0)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

# -------------------------
# PROMPT DE SISTEMA + FEW-SHOT (DIÁLOGO)
# -------------------------
SYSTEM_PROMPT = """Você é um assistente conversacional e coloquial (responde como alguém do Rio quando o usuário usa esse tom).
Regras obrigatórias:
- Sempre responda como uma conversa: natural, em primeira pessoa, breve (1–4 frases).
- NUNCA imprima rótulos de meta-exemplos (ex.: "Resposta desejada: ...") — trate-os como anotações internas.
- Ao final da resposta, faça UMA pergunta curta que mantenha o diálogo (ex.: "Quer que eu explique mais?").
- Se não souber algo, diga "não sei" e ofereça buscar referência.
- Adapte o tom ao do usuário (coloquial ↔ formal).
"""

# Some few-shot examples (format: Usuário -> Assistente)
FEW_SHOT = [
    {"user": "A bandeira foi adotada pelo governo do Brasil em 17 de agosto de 1985.",
     "assistant": "Saca só: essa bandeira foi adotada em 17/08/1985 — tá aí desde então. Quer que eu te conte a história dela?"},
    {"user": "Escreva um verso curto sobre o mar.",
     "assistant": "Mar que risca a tarde, leva e traz memórias. Quer outro verso?"},
    {"user": "Escreva um verso curto sobre o carro.",
     "assistant": "Carro que corre noite adentro, deixando rastros de cidade. Quer outro com mais ritmo?"},
    {"user": "Quem foi Manuel Bandeira?",
     "assistant": "Foi um poeta brasileiro modernista, com versos íntimos e diretos. Quer saber uma obra dele?"},
    {"user": "Fale curto sobre trabalho de um homem.",
     "assistant": "Mãos que constroem o dia, cansam e orgulham. Quer que eu expanda?"},
    {"user": "Me dá uma risada curta, tipo texto.",
     "assistant": "Heh, essa foi boa. Quer eu faça uma piada curta?"},
    {"user": "Me explique vagamente o que é LoRA.",
     "assistant": "LoRA é uma técnica leve pra ajustar grandes modelos sem treinar tudo — economiza memória. Quer mais tech?"},
    {"user": "Como faço para melhorar o tom do bot?",
     "assistant": "Bota exemplos de diálogo, regula a temperatura e force instruções no system prompt. Quer que eu gere exemplos?"},
    {"user": "Escreva um verso curto sobre a cidade.",
     "assistant": "Cidade que respira luz e pressa, guarda segredos em cada esquina. Quer outro em tom mais triste?"},
    {"user": "Quero que a resposta seja conversacional e curta.",
     "assistant": "Feito: vou responder como se a gente estivesse trocando ideia. Quer testar com uma pergunta real?"}
]

def build_few_shot_dialog(few_shot_list):
    parts = []
    for ex in few_shot_list:
        parts.append(f"Usuário: {ex['user']}")
        parts.append(f"Assistente: {ex['assistant']}")
    return "\n".join(parts)

FEW_SHOT_TEXT = build_few_shot_dialog(FEW_SHOT)

# -------------------------
# DETECÇÃO E REPARO DE RESPOSTAS RUINS
# -------------------------
BAD_PATTERNS = [
    "resposta desejada",
    "resposta:",
    "não me pergunte",
    "não me pergunte se",
    "resposta desejada:"
]

def looks_bad(text: str) -> bool:
    txt = (text or "").lower()
    for p in BAD_PATTERNS:
        if p in txt:
            return True
    return False

def repair_or_regen(prompt_base: str, pipe, tokenizer, tries=2):
    for attempt in range(tries):
        temp = 0.7 if attempt == 0 else 0.5
        out = pipe(prompt_base, max_new_tokens=160, do_sample=True,
                   temperature=temp, top_p=0.9, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
        resposta = out.split("Assistente:")[-1].strip()
        if not looks_bad(resposta):
            return resposta
    return "Desculpa, não sei responder exatamente assim — prefere que eu fale sobre outro ponto ou explique de outra forma?"

# -------------------------
# Funções utilitárias para parse de histórico (robusto)
# -------------------------
def extract_text(item):
    """
    Extrai texto de várias formas de item de histórico:
    - string -> string
    - dict with 'content' or 'text' -> that value
    - list/tuple -> tenta pegar o primeiro elemento textual (0 ou 1)
    - fallback -> str(item)
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # formatos possíveis: {'role':..., 'content':...} ou {'text': '...'}
        return item.get("content") or item.get("text") or ""
    if isinstance(item, (list, tuple)):
        # procura o primeiro elemento que pareça string / dict com content
        for elem in item:
            if isinstance(elem, str):
                return elem
            if isinstance(elem, dict) and (elem.get("content") or elem.get("text")):
                return elem.get("content") or elem.get("text")
        # se não achou, tenta converter os dois primeiros itens em string
        if len(item) >= 1:
            return str(item[0])
        return ""
    return str(item)

# -------------------------
# FUNÇÃO PRINCIPAL DO CHAT (robusta)
# -------------------------
def responder(mensagens, historico=None, usar_system=True, usar_few_shot=True):
    """
    Suporta:
     - formato novo: mensagens = [ {role: 'user'|'assistant'|'system', content: '...'}, ... ]
     - formato antigo: mensagens = "<texto do usuário>", historico = [ item, item, ... ]
       onde cada item pode ser [user, bot], (user, bot), ou listas maiores com metadados.
    """
    try:
        # --- captura user_input e qualquer contexto de mensagens "novo formato" ---
        if isinstance(mensagens, list) and len(mensagens) > 0 and isinstance(mensagens[0], dict):
            user_input = mensagens[-1].get("content", "")
            contexto = ""
            for m in mensagens[:-1]:
                role = m.get("role")
                if role == "user":
                    contexto += f"Usuário: {m.get('content','')}\n"
                elif role == "assistant":
                    contexto += f"Assistente: {m.get('content','')}\n"
                elif role == "system":
                    contexto += f"Sistema: {m.get('content','')}\n"
        else:
            # --- formato antigo (ou genérico) ---
            user_input = extract_text(mensagens)
            contexto = ""
            if historico:
                for item in historico:
                    # se item for par (lista/tupla) com >=2 elementos, pega os dois primeiros
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        user_msg = extract_text(item[0])
                        bot_msg = extract_text(item[1])
                        contexto += f"Usuário: {user_msg}\nAssistente: {bot_msg}\n"
                    elif isinstance(item, dict):
                        # pode ser {'role':..., 'content':...}
                        r = item.get("role")
                        if r == "user":
                            contexto += f"Usuário: {item.get('content','')}\n"
                        elif r == "assistant":
                            contexto += f"Assistente: {item.get('content','')}\n"
                    else:
                        # fallback: tenta extrair e colocar como usuário (sem resposta)
                        text = extract_text(item)
                        if text:
                            contexto += f"Usuário: {text}\n"
    except Exception as e:
        # Em caso de parse falho, logue e siga com o que tiver
        print("Erro ao parsear mensagens/historico:", e)
        user_input = extract_text(mensagens)
        contexto = ""

    # --- monta prompt com sistema e few-shot ---
    pieces = []
    if usar_system:
        pieces.append("### INSTRUÇÕES DO SISTEMA ###")
        pieces.append(SYSTEM_PROMPT)
    if usar_few_shot:
        pieces.append("### EXEMPLOS ###")
        pieces.append(FEW_SHOT_TEXT)
    if contexto:
        pieces.append("### CONTEXTO ###")
        pieces.append(contexto.strip())

    pieces.append(f"Usuário: {user_input}\nAssistente:")
    prompt = "\n\n".join(pieces)

    # geração
    generation_kwargs = dict(
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    out = pipe(prompt, **generation_kwargs)[0]["generated_text"]
    resposta = out.split("Assistente:")[-1].strip()

    # se resposta for indesejada, tenta reparar/regenerar
    if looks_bad(resposta):
        resposta = repair_or_regen(prompt, pipe, tokenizer, tries=2)

    # garante tom conversacional (pergunta curta no final)
    if "?" not in resposta and len(resposta.splitlines()) < 3:
        if not resposta.strip().endswith("?"):
            resposta = resposta.rstrip(".") + ". Quer que eu detalhe mais?"

    return {"role": "assistant", "content": resposta}

# -------------------------
# INTERFACE GRADIO
# -------------------------
interface = gr.ChatInterface(
    fn=responder,
    title="Chatbot LoRA GPT-2 — Estilo Conversacional (corrigido)",
    description="Modelo GPT-2 + LoRA com prompt de sistema e few-shot. Tratamento robusto de histórico para evitar unpack errors.",
    theme="soft",
    type="messages",
    examples=[
        "Olá!",
        "Escreva um verso sobre o carro.",
        "Me explica o que é LoRA."
    ]
)

if __name__ == "__main__":
    interface.launch()
