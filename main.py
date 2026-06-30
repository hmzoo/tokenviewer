"""
Token Viewer 🪙 — Backend FastAPI
Tokenisation temps réel et édition bidirectionnelle.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

# ─── Modèles disponibles ───────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "GPT-2": "gpt2",
    "BERT (uncased)": "bert-base-uncased",
    "BERT (cased)": "bert-base-cased",
    "RoBERTa": "roberta-base",
    "XLNet": "xlnet-base-cased",
    "T5 Small": "t5-small",
    "CodeBERT": "microsoft/codebert-base",
    "DistilBERT": "distilbert-base-uncased",
    "Qwen2.5 1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-VL 7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}

_tokenizer_cache: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_id: str) -> AutoTokenizer:
    if model_id not in _tokenizer_cache:
        _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _tokenizer_cache[model_id]


# ─── Helpers de tokenisation (repris du Gradio) ───────────────────────────

def _build_word_mapping(encoding, input_ids: list[int], tokens: list[str]) -> list[dict[str, Any]]:
    try:
        word_ids = encoding.word_ids()
    except (AttributeError, ValueError):
        word_ids = None

    if word_ids is not None and any(w is not None for w in word_ids):
        return _from_word_ids(tokens, input_ids, word_ids)
    return _heuristic(tokens, input_ids)


def _from_word_ids(tokens, ids, word_ids):
    groups, cur_wid, cur = [], None, []
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != cur_wid and cur:
            groups.append({"label": f"mot_{cur_wid}", "tokens": cur})
            cur = []
        cur_wid = wid
        cur.append({"text": tokens[i], "id": ids[i]})
    if cur:
        groups.append({"label": f"mot_{cur_wid}", "tokens": cur})
    return groups


def _heuristic(tokens, ids):
    groups, cur, wi = [], [], 0
    label = "mot_0"
    for i, tok in enumerate(tokens):
        if tok.startswith("Ġ") and cur:
            groups.append({"label": label, "tokens": cur})
            wi += 1
            label = f"mot_{wi}"
            cur = []
            cur.append({"text": tok[1:] or tok, "id": ids[i]})
        elif tok.startswith("##"):
            cur.append({"text": tok[2:] or tok, "id": ids[i]})
        else:
            cur.append({"text": tok, "id": ids[i]})
    if cur:
        groups.append({"label": label, "tokens": cur})
    return groups


def _rle_compress(tokens: list[str], ids: list[int]) -> list[dict[str, Any]]:
    if not tokens:
        return []
    result = []
    prev_tok, cnt, id_list = tokens[0], 1, [ids[0]]
    for i in range(1, len(tokens)):
        if tokens[i] == prev_tok:
            cnt += 1
            id_list.append(ids[i])
        else:
            result.append({"token": prev_tok, "count": cnt, "ids": id_list})
            prev_tok, cnt, id_list = tokens[i], 1, [ids[i]]
    result.append({"token": prev_tok, "count": cnt, "ids": id_list})
    return result


# ─── Schémas API ──────────────────────────────────────────────────────────

class TokenizeRequest(BaseModel):
    model: str
    text: str


class TokenizeResponse(BaseModel):
    tokens: list[str]
    ids: list[int]
    compressed: list[dict[str, Any]]
    word_mapping: list[dict[str, Any]]
    stats: dict[str, int | float]


class DecodeRequest(BaseModel):
    model: str
    ids: list[int]


class DecodeResponse(BaseModel):
    text: str


# ─── Application FastAPI ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Précharger GPT-2 au démarrage
    _get_tokenizer("gpt2")
    yield


app = FastAPI(title="Token Viewer", version="2.0.0", lifespan=lifespan)

_HERE = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=(_HERE / "templates" / "index.html").read_text())


@app.get("/api/models")
async def list_models():
    return {"models": {k: v for k, v in MODELS.items()}}


@app.post("/api/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest):
    text = req.text.strip()
    model_id = MODELS.get(req.model)
    if not model_id:
        raise HTTPException(400, f"Modèle inconnu : {req.model}")

    tokenizer = _get_tokenizer(model_id)
    encoding = tokenizer(text, return_tensors=None, add_special_tokens=True)
    ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    compressed = _rle_compress(tokens, ids)
    word_mapping = _build_word_mapping(encoding, ids, tokens)

    total = len(ids)
    unique = len(set(ids))
    rle_len = len(compressed)
    stats = {
        "total_tokens": total,
        "unique_ids": unique,
        "compressed_size": rle_len,
        "compression_ratio": round(total / max(rle_len, 1), 2),
    }

    return TokenizeResponse(
        tokens=tokens, ids=ids,
        compressed=compressed,
        word_mapping=word_mapping,
        stats=stats,
    )


@app.post("/api/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest):
    model_id = MODELS.get(req.model)
    if not model_id:
        raise HTTPException(400, f"Modèle inconnu : {req.model}")

    tokenizer = _get_tokenizer(model_id)
    try:
        text = tokenizer.decode(req.ids, skip_special_tokens=True)
    except Exception as e:
        raise HTTPException(400, f"Erreur de décodage : {e}")
    return DecodeResponse(text=text)


# ─── Compression LLMLingua ───────────────────────────────────────────────

class CompressRequest(BaseModel):
    text: str
    rate: float = 0.5  # 0.0 – 1.0, plus petit = plus compressé


class CompressResponse(BaseModel):
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float


_compressor = None


def _get_compressor():
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor
        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
        )
    return _compressor


@app.post("/api/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    if not req.text.strip():
        raise HTTPException(400, "Texte vide")

    compressor = _get_compressor()
    rate = max(0.1, min(1.0, req.rate))

    try:
        result = compressor.compress_prompt(
            req.text,
            rate=rate,
            force_tokens=["\n", ".", "!", "?", ","],
        )
        compressed = result.get("compressed_prompt", req.text)
    except Exception as e:
        raise HTTPException(500, f"Erreur de compression : {e}")

    # Compter les tokens approximativement
    orig_tokens = len(req.text.split())
    comp_tokens = len(compressed.split())

    return CompressResponse(
        compressed_text=compressed,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        compression_ratio=round(comp_tokens / max(orig_tokens, 1), 3),
    )


# ─── Lancement direct ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
