#!/usr/bin/env python3
"""
PEP8 / OOP RAG example for local inference (gguf via llama-cpp-python).

Usage:
    1. Configure MODEL_PATH below or via env var MODEL_PATH.
    2. Add documents with engine.add_documents([...], metadatas=[...])
    3. Query with engine.query("your question")

Dependencies:
    pip install sentence-transformers faiss-cpu llama-cpp-python tqdm python-dotenv
    (optional) pip install tiktoken
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Llama.cpp python bindings
try:
    from llama_cpp import Llama
except Exception as exc:  # pragma: no cover - informative error for users
    raise RuntimeError(
        "llama_cpp (llama-cpp-python) is required to run the local GGUF model. "
        "Install via `pip install llama-cpp-python` and ensure llama.cpp is available."
    ) from exc

# Optional accurate tokenizer
try:
    import tiktoken

    TOK_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOK_ENCODER = None

# ------------------------- Config -------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_CTX_TOKENS = int(os.environ.get("MAX_CTX_TOKENS", "4096"))
RESERVED_PROMPT_TOKENS = int(os.environ.get("RESERVED_PROMPT_TOKENS", "256"))
MAX_CHUNK_TOKENS = 512
EMBED_DIM = 384  # all-MiniLM-L6-v2: 384; adjust if you change model
TOP_K = int(os.environ.get("RAG_TOP_K", "6"))
LLAMA_N_CTX = int(os.environ.get("LLAMA_N_CTX", "4096"))
LLAMA_N_THREADS = int(os.environ.get("LLAMA_N_THREADS", "8"))
# ----------------------------------------------------------

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------- Utilities -------------------------
def count_tokens(text: str) -> int:
    """Return approximate token count; use tiktoken if available."""
    if TOK_ENCODER is not None:
        return len(TOK_ENCODER.encode(text))
    # fallback heuristic: approx 1.33 tokens per word
    words = re.findall(r"\w+|\S", text)
    return int(len(words) * 1.33) + 1


def chunk_text_tokenwise(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = 64,
) -> Iterable[str]:
    """Token-aware chunker: tries to split on paragraph/sentence boundaries."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    pieces: List[str] = []
    for p in paras:
        if count_tokens(p) > max_tokens:
            # split long paragraphs by sentences
            sent_parts = re.split(r"(?<=[\.\?\!])\s+", p)
            pieces.extend([s.strip() for s in sent_parts if s.strip()])
        else:
            pieces.append(p)

    cur: List[str] = []
    cur_toks = 0
    i = 0
    piece_toks = [count_tokens(p) for p in pieces]
    while i < len(pieces):
        p = pieces[i]
        pt = piece_toks[i]
        # piece alone larger than max: split by words
        if pt >= max_tokens:
            words = p.split()
            j = 0
            while j < len(words):
                sub = []
                while j < len(words):
                    sub.append(words[j])
                    if count_tokens(" ".join(sub)) > max_tokens:
                        sub.pop()
                        break
                    j += 1
                if not sub:
                    sub = [words[j]]
                    j += 1
                yield " ".join(sub)
            i += 1
            continue
        # accumulate
        if cur_toks + pt <= max_tokens:
            cur.append(p)
            cur_toks += pt
            i += 1
            continue
        # flush
        yield "\n\n".join(cur)
        # overlap
        if overlap_tokens > 0 and cur:
            tail: List[str] = []
            tail_toks = 0
            for piece in reversed(cur):
                t = count_tokens(piece)
                if tail_toks + t > overlap_tokens:
                    break
                tail.insert(0, piece)
                tail_toks += t
            
            if tail_toks + pt > max_tokens:
                cur = []
                cur_toks = 0
            else:
                cur = tail
                cur_toks = tail_toks
        else:
            cur = []
            cur_toks = 0
    if cur:
        yield "\n\n".join(cur)


# ------------------------- Data Classes -------------------------
@dataclass
class Document:
    """A simple document chunk with metadata."""
    doc_id: str
    text: str
    metadata: Dict[str, Any]


# ------------------------- Embedder -------------------------
class Embedder:
    """Wrapper around a SentenceTransformer embedder."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        # update dim if model differs
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Return normalized embeddings (ndarray, float32)."""
        embs = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        # normalize to unit vectors for cosine via inner product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        return embs.astype("float32")


# ------------------------- Vector Store -------------------------
class FaissVectorStore:
    """FAISS vector store for fast similarity search."""

    def __init__(self, dim: int, index_file: Optional[str] = None):
        self.dim = dim
        self.index_file = index_file
        self.index = faiss.IndexFlatIP(dim)
        self.id_to_meta: Dict[int, Document] = {}
        self.next_id = 0

    def add(self, docs: List[Document], embeddings: np.ndarray) -> None:
        """Add docs and embeddings (embeddings.shape[0] == len(docs))."""
        n = embeddings.shape[0]
        if n != len(docs):
            raise ValueError("Embeddings count != documents count")
        self.index.add(embeddings)
        for i, doc in enumerate(docs):
            self.id_to_meta[self.next_id] = doc
            self.next_id += 1

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Return list[(Document, score)] sorted by score desc."""
        if self.index.ntotal == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, idxs = self.index.search(query_vec.astype("float32"), top_k)
        results: List[Tuple[Document, float]] = []
        for score_row, idx_row in zip(scores, idxs):
            for score, idx in zip(score_row, idx_row):
                if idx < 0:
                    continue
                meta = self.id_to_meta.get(int(idx))
                if meta is None:
                    continue
                results.append((meta, float(score)))
        return results


# ------------------------- LLM wrapper -------------------------
class LocalLLM:
    """Small wrapper for llama-cpp-python Llama."""

    def __init__(self, model_path: str = MODEL_PATH, n_ctx: int = LLAMA_N_CTX, n_threads: int = LLAMA_N_THREADS):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        logger.info("Loading local LLM model from %s", model_path)
        self.model = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads, n_batch=2048)

    def generate(self, prompt: str, max_tokens: int = 4096, stop: Optional[List[str]] = None) -> str:
        resp = self.model(
            prompt,
            top_p=0.7,
            max_tokens=max_tokens,
            stop=(stop or []) + ["<|im_end|>"],
            echo=False,
            temperature=0.1,
            stream=True,
        )
        
        # Determine if resp is a generator (streaming mode) or dict (non-streaming)
        if hasattr(resp, '__iter__') and not isinstance(resp, dict):
            output = []
            for chunk in resp:
                text = chunk["choices"][0].get("text", "")
                output.append(text)
            return "".join(output).strip()
        return resp["choices"][0]["text"].strip()

    def generate_stream(self, prompt: str, max_tokens: int = 2048, stop: Optional[List[str]] = None):
        resp = self.model(
            prompt,
            top_p=0.7,
            max_tokens=max_tokens,
            stop=(stop or []) + ["<|im_end|>"],
            echo=False,
            temperature=0.1,
            stream=True,
        )
        if hasattr(resp, '__iter__') and not isinstance(resp, dict):
            for chunk in resp:
                yield chunk["choices"][0].get("text", "")
        else:
            yield resp["choices"][0]["text"].strip()



# ------------------------- RAG Engine -------------------------
class RAGEngine:
    """Main RAG engine: chunk -> embed -> index -> search -> answer."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: FaissVectorStore,
        llm: LocalLLM,
        top_k: int = TOP_K,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def add_documents(self, texts: Iterable[str], metadatas: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        """Chunk, embed and add documents to vector store.

        texts: iterable of long documents (pages).
        metadatas: optional iterable of dicts aligned with texts.
        """
        if metadatas is None:
            # We can't generate an infinite [None] list, so we'll just zip with an infinite generator of None
            def infinite_none():
                while True:
                    yield None
            meta_iter = infinite_none()
        else:
            meta_iter = iter(metadatas)

        docs_to_index: List[Document] = []
        for text in texts:
            meta = next(meta_iter, None)
            # chunk large pages
            for chunk in chunk_text_tokenwise(text, max_tokens=MAX_CHUNK_TOKENS):
                doc = Document(doc_id=str(uuid.uuid4()), text=chunk, metadata=meta or {})
                docs_to_index.append(doc)
        
        if not docs_to_index:
            return
        if not docs_to_index:
            return
        batches = 512
        # embed in batches to avoid memory spikes
        for i in range(0, len(docs_to_index), batches):
            batch = docs_to_index[i : i + batches]
            emb_texts = [d.text for d in batch]
            embs = self.embedder.embed_texts(emb_texts)
            self.vector_store.add(batch, embs)
        logger.info("Indexed %d chunks (total vectors: %d)", len(docs_to_index), self.vector_store.index.ntotal)

    def _build_prompt(self, query: str, contexts: List[Tuple[Document, float]]) -> str:
        ctx_parts = []
        for i, (doc, score) in enumerate(contexts, start=1):
            meta = doc.metadata or {}
            src = meta.get("source") or meta.get("url") or f"doc:{doc.doc_id}"
            ctx_parts.append(f"Source {i} (score={score:.4f}, origin={src}):\n{doc.text}\n")
        context_blob = "\n\n".join(ctx_parts)

        system = (
            "You are a helpful assistant. Answer the user's question using the provided CONTEXT when relevant.\n"
            "If you cannot find the answer in the context, you may still answer using general knowledge.\n"
            "Please format your answer using cleanly structured Markdown."
        )

        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\nCONTEXT:\n{context_blob}\n\nQUESTION:\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def query(self, query: str, top_k: Optional[int] = None, max_gen_tokens: int = 512) -> Dict[str, Any]:
        """Run a RAG query and return parsed result structure."""
        if top_k is None:
            top_k = self.top_k
        q_emb = self.embedder.embed_texts([query])[0]
        # search
        raw_results = self.vector_store.search(q_emb.reshape(1, -1), top_k=top_k)
        if not raw_results or max(score for _, score in raw_results) < 0.4:
            # fallback to pure LLM
            prompt = (
                "You are a helpful assistant. Answer the user's question.\n"
                "Please format your answer using cleanly structured Markdown.\n\n"
                "QUESTION:\n" + query + "\n\n"
                "ANSWER:\n"
            )
            gen = self.llm.generate(prompt, max_tokens=max_gen_tokens)
            return {
                "answer": gen.strip(),
                "sources": []
            }

        unique = {}
        for doc, score in raw_results:
            url = doc.metadata.get("url")
            if url not in unique:
                unique[url] = (doc, score)

        raw_results = list(unique.values())

        # assemble prompt
        prompt = self._build_prompt(query, raw_results)
        # adjust generation budget conservatively
        gen = self.llm.generate(prompt, max_tokens=max_gen_tokens)
        
        return {
            "answer": gen.strip(),
            "sources": [{"source": d.metadata.get("url", d.doc_id), "score": s} for d, s in raw_results]
        }

    def query_stream(self, query: str, top_k: Optional[int] = None, max_gen_tokens: int = 512):
        """Run a RAG query and yield streamed JSON items."""
        if top_k is None:
            top_k = self.top_k
        q_emb = self.embedder.embed_texts([query])[0]
        raw_results = self.vector_store.search(q_emb.reshape(1, -1), top_k=top_k)
        if not raw_results or max(score for _, score in raw_results) < 0.4:
            prompt = (
                "You are a helpful assistant. Answer the user's question.\n"
                "Please format your answer using cleanly structured Markdown.\n\n"
                "QUESTION:\n" + query + "\n\n"
                "ANSWER:\n"
            )
            yield {"type": "metadata", "sources": []}
        else:
            unique = {}
            for doc, score in raw_results:
                url = doc.metadata.get("url")
                if url not in unique:
                    unique[url] = (doc, score)
            raw_results = list(unique.values())
            prompt = self._build_prompt(query, raw_results)
            sources = [{"source": d.metadata.get("url", d.doc_id), "score": s} for d, s in raw_results]
            yield {"type": "metadata", "sources": sources}

        token_count = 0
        for token in self.llm.generate_stream(prompt, max_tokens=max_gen_tokens):
            if token:
                token_count += 1
                yield {"type": "token", "content": token}
        
        yield {"type": "done", "token_count": token_count}