#!/usr/bin/env python3
"""
Main orchestrator:
Crawler → RAG indexing → Interactive querying
"""

import asyncio
import json
import os
from typing import List, Dict

from crawler import Crawler, CrawlerConfig
from rag_engine import Embedder, FaissVectorStore, LocalLLM, RAGEngine


OUTPUT_FILE = "output_chunks.jsonl"
MODEL_PATH = "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"


# -----------------------------
# Step 1: Run crawler
# -----------------------------
async def run_crawler(start_url: str):
    print("\n[+] Starting crawler...\n")

    config = CrawlerConfig(
        start_url=start_url,
        max_pages=100,        # you can tune this
        concurrency=50,
        output_path=OUTPUT_FILE,
    )
    crawler = Crawler(config)

    await crawler.run()

    print(f"\n[✓] Crawling done. Output saved to {OUTPUT_FILE}\n")


# -----------------------------
# Step 2: Load chunks
# -----------------------------
def load_chunks(file_path: str) -> (List[str], List[Dict]):
    texts = []
    metas = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["chunk"])
            metas.append(row["meta"])

    return texts, metas


# -----------------------------
# Step 3: Build RAG
# -----------------------------
def build_rag(texts: List[str], metas: List[Dict]) -> RAGEngine:
    print("[+] Initializing RAG components...")

    embedder = Embedder()
    vector_store = FaissVectorStore(dim=embedder.dim)
    llm = LocalLLM(model_path=MODEL_PATH)

    engine = RAGEngine(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm
    )

    print("[+] Indexing data into vector DB...\n")
    engine.add_documents(texts, metadatas=metas)

    print("[✓] RAG ready.\n")
    return engine


# -----------------------------
# Step 4: Interactive query loop
# -----------------------------
def query_loop(engine: RAGEngine):
    print("\n===== Interactive RAG =====")
    print("Type 'exit' to quit.\n")

    while True:
        query = input(">> Enter your query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        result = engine.query(query)

        print("\n--- RESULT ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n")


# -----------------------------
# Main
# -----------------------------
def main():
    print("=== Local RAG Pipeline ===\n")

    start_url = input("Enter website URL to crawl: ").strip()

    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url
        print(f"[+] Normalized URL: {start_url}")

    if not start_url:
        print("Invalid URL.")
        return

    # Run crawler
    asyncio.run(run_crawler(start_url))

    # Load chunks
    texts, metas = load_chunks(OUTPUT_FILE)

    if not texts:
        print("No data extracted. Exiting.")
        return

    # Build RAG
    engine = build_rag(texts, metas)

    # Query loop
    query_loop(engine)


if __name__ == "__main__":
    main()