# preprocess_all.py
import json, csv, hashlib, unicodedata, re, sys
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(".").resolve()
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    # NFC, unify line endings, strip spaces per line, limit empty lines
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    cleaned = []
    empty = 0
    for ln in lines:
        if ln.strip() == "":
            empty += 1
        else:
            empty = 0
        if empty <= 2:
            cleaned.append(ln)
    out = "\n".join(cleaned).strip()
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out

items = []  # dicts with title,text,source,filename

# 1) JSONL (if exists)
jsonl_path = RAW_DIR / "manuel_bandeira_poemas.jsonl"
if jsonl_path.exists():
    with open(jsonl_path, encoding="utf-8") as fr:
        for i, line in enumerate(fr, start=1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text","") or obj.get("texto","")
            title = obj.get("title","") or obj.get("titulo","")
            if text and text.strip():
                items.append({"title": title.strip(), "text": normalize_text(text), "source": "jsonl", "filename": f"jsonl_line_{i}"})

# 2) CSV (if exists)
csv_path = RAW_DIR / "manuel_bandeira_poemas_completos.csv"
if csv_path.exists():
    with open(csv_path, encoding="utf-8") as fr:
        reader = csv.DictReader(fr)
        for i, row in enumerate(reader, start=1):
            text = row.get("text") or row.get("texto") or row.get("Text") or ""
            title = row.get("title") or row.get("titulo") or ""
            if text and text.strip():
                items.append({"title": title.strip(), "text": normalize_text(text), "source": "csv", "filename": f"csv_row_{i}"})

# 3) MD files under data/raw/md_files or data/raw/md
for md_root in [RAW_DIR / "md_files", RAW_DIR / "md"]:
    if md_root.exists():
        for p in sorted(md_root.rglob("*.md")):
            try:
                txt = p.read_text(encoding="utf-8")
            except Exception:
                continue
            # simple frontmatter strip (--- YAML ---)
            body = txt
            if txt.lstrip().startswith("---"):
                parts = re.split(r"\n-{3,}\s*\n", txt, maxsplit=1)
                if len(parts) == 2:
                    body = parts[1]
            body = re.sub(r"^\s*#{1,6}\s+", "", body).strip()
            if body:
                items.append({"title": p.stem, "text": normalize_text(body), "source": "md", "filename": str(p.name)})

# 4) TXT files under data/raw/txt_files or data/raw/txt
for txt_root in [RAW_DIR / "txt_files", RAW_DIR / "txt"]:
    if txt_root.exists():
        for p in sorted(txt_root.rglob("*.txt")):
            try:
                txt = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if txt and txt.strip():
                items.append({"title": p.stem, "text": normalize_text(txt), "source": "txt", "filename": str(p.name)})

print(f"[*] Total itens brutos carregados: {len(items)}")

# deduplicate by sha256 and remove very short texts
unique = {}
short_count = 0
for it in items:
    t = it["text"].strip()
    if not t:
        continue
    if len(t) < 30:
        short_count += 1
        continue
    h = sha256_of_text(t)
    if h not in unique:
        entry = {"title": it.get("title",""), "text": t, "sha256": h, "source": it.get("source",""), "filename": it.get("filename","")}
        unique[h] = entry

final_items = list(unique.values())
print(f"[*] Itens removidos por serem muito curtos (<30 chars): {short_count}")
print(f"[*] Itens únicos após deduplicação: {len(final_items)}")

# save full processed JSONL
ALL_PATH = PROC_DIR / "manuel_bandeira_corpus_processed.jsonl"
with open(ALL_PATH, "w", encoding="utf-8") as fw:
    for i, it in enumerate(final_items, start=1):
        obj = {"id": i, "title": it["title"], "text": it["text"], "sha256": it["sha256"], "source": it["source"], "filename": it["filename"]}
        fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"[*] Salvo JSONL completo: {ALL_PATH}")

# split 80/20
texts = [it["text"] for it in final_items]
if len(texts) < 2:
    print("[!] Erro: poucos textos após preprocessamento. Saindo.")
    sys.exit(1)

train, val = train_test_split(texts, test_size=0.2, random_state=42, shuffle=True)

TRAIN = PROC_DIR / "train.jsonl"
VAL   = PROC_DIR / "validation.jsonl"
with open(TRAIN, "w", encoding="utf-8") as fw:
    for t in train:
        fw.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
with open(VAL, "w", encoding="utf-8") as fw:
    for t in val:
        fw.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

print(f"[*] Train salvo em: {TRAIN} ({len(train)} itens)")
print(f"[*] Validation salvo em: {VAL} ({len(val)} itens)")

# save also txt files for quick human inspection
TXT_OUT = PROC_DIR / "txt_files"
TXT_OUT.mkdir(parents=True, exist_ok=True)
for i, it in enumerate(final_items, start=1):
    outp = TXT_OUT / f"{i:04d}_{it['sha256'][:8]}.txt"
    outp.write_text(it["text"], encoding="utf-8")

print(f"[*] Arquivos .txt gerados em: {TXT_OUT}")

# print 3 examples
print("\n=== Exemplos (3 primeiros) ===")
for i, it in enumerate(final_items[:3], start=1):
    print(f"\n[{i}] title: {it['title']}  filename: {it['filename']} sha256: {it['sha256']}")
    print(it["text"][:800] + ("\n..." if len(it["text"])>800 else ""))

print("\n[*] Pré-processamento completo.")
