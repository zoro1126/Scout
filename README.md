# Scout: Scout Crawls Online URLs for Transformers 🕷️🔗🤖

An unblockable, dynamic web crawler tightly integrated with a fully local Retrieval-Augmented Generation (RAG) pipeline and a beautiful web-based UI. 

---

## 🚀 Features

* **Dynamic Web Crawler**: Powered by `aiohttp` and `Playwright`, automatically upgrades to a real browser to bypass anti-bot systems and render JS-heavy, dynamic single-page applications (SPAs).
* **Fully Local RAG**: Built heavily around privacy and cost savings. Uses `Faiss` for vector storage, `sentence-transformers` for embeddings, and `llama-cpp-python` running a quantized local model (Qwen 2.5 3B). 
* **ChatGPT-like Web Experience**: Features an elegant, fully localized async web server via `aiohttp.web` and `EventSource` (SSE) for streaming text generation directly into the browser in real-time. 
* **Smart Content Extraction**: Cleans HTML noise, strips headers/footers, and intelligently extracts paragraph structures using `BeautifulSoup` and `lxml`. 
* **Resilient**: Implements token-aware text chunking via `tiktoken` with sliding-window overlaps, exponential backoffs for HTTP errors, and intelligent politeness delays.

---

## 🛠 Tech Stack

* **Language**: Python 3
* **Crawler Engine**: `aiohttp` (async HTTP requests), `Playwright` (headless chromium for JS sites), `BeautifulSoup` & `lxml` (HTML parsing)
* **Embedding Model**: `BAAI/bge-small-en-v1.5` (via `sentence-transformers`)
* **Vector Store**: `faiss-cpu` (Facebook AI Similarity Search)
* **Local LLM**: `Qwen2.5-3B-Instruct` (Quantized GGUF running via `llama-cpp-python`)
* **Web Server**: `aiohttp.web`

---

## ⚙️ Installation

1. **Clone the repository.**
2. **Make the installation script executable and run it.**
   This script handles creating a virtual environment, installing dependencies, configuring Playwright browser binaries, and downloading the quantized Qwen 3B model (~2.4 GB).
   
   ```bash
   chmod +x install_deps_and_models.sh
   ./install_deps_and_models.sh
   ```

3. **Activate the Virtual Environment.**
   
   ```bash
   source .venv/bin/activate
   ```

---

## 💻 Usage

### Starting the Web UI
The easiest way to use the application is through the fully integrated web interface.

```bash
python backend/web_ui.py
```

Then, open your browser and navigate to `http://127.0.0.1:8080`. From there you can enter a URL to scrape and index, and subsequently chat with the indexed knowledge. 

### CLI Mode
You can also run the modular orchestrator from the command line:

```bash
python bckend/main.py
```
This will prompt you for a URL to crawl, run the indexing process, and open an interactive terminal REPL loop to ask questions.

---

## 🚧 Challenges Faced & Solved

1. **JavaScript Heavy / SPA Sites:** Traditional HTTP crawlers often extract blank pages from modern React/Vue sites. 
   **Solution:** Built a dynamic heuristic detector that scans payload sizes and DOM structures to dynamically upgrade the fetcher to a real, headless Playwright browser instance only when necessary, saving compute on static sites.
2. **Anti-Bot Defenses:** Many sites instantly block naive scrappers.
   **Solution:** We simulate realistic browser viewports, randomized modern user-agents, spoofed `navigator.webdriver` attributes, and randomized exponential backoffs.
3. **Loss of Context in Indexing:** Hard text slicing breaks paragraphs mid-sentence, confusing the RAG LLM.
   **Solution:** Configured `tiktoken` recursive overlaps combined with sentence-boundary detection. This prevents splitting ideas randomly and ensures seamless contextual overlaps.
4. **Memory Constraint on Local LLMs:** Running an LLM while concurrently running browser instances can crash standard dev machines. 
   **Solution:** Uses the optimized `llama-cpp-python` binding to run highly compressed Q4 quantized GGUF models. Playwright instances are aggressively rate-limited and context-managed to maintain a tiny RAM footprint.

---

## 📝 License
MIT License
