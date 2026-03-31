import asyncio
import os
from aiohttp import web

# Import functionalities from main.py without modifying it as per user rules
from main import run_crawler, load_chunks, build_rag, OUTPUT_FILE

engine = None

async def init_backend(app):
    global engine
    print("\n[+] Checking for existing data to load RAG engine...")
    if os.path.exists(OUTPUT_FILE):
        try:
            texts, metas = load_chunks(OUTPUT_FILE)
            if texts:
                print(f"[+] Loaded {len(texts)} chunks from {OUTPUT_FILE}. Building RAG...")
                loop = asyncio.get_event_loop()
                engine = await loop.run_in_executor(None, build_rag, texts, metas)
                print("[+] RAG engine ready.")
        except Exception as e:
            print(f"[-] Could not load existing chunks: {e}")

async def handle_index(request):
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")
    except Exception as e:
        return web.Response(text=f"Error loading frontend/index.html: {e}", status=500)

async def handle_chat(request):
    global engine
    try:
        data = await request.json()
        query = data.get("query", "")
        
        if not query:
            return web.json_response({"error": "Empty query"}, status=400)
            
        if engine is None:
            return web.json_response({"error": "RAG engine is not initialized. Please crawl a URL first."}, status=400)
            
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
        await response.prepare(request)
        
        loop = asyncio.get_running_loop()
        import queue
        import threading
        import json
        q = queue.Queue()
        
        def run_query():
            try:
                for chunk in engine.query_stream(query):
                    q.put(chunk)
                q.put(None)
            except Exception as e:
                q.put(e)
                
        threading.Thread(target=run_query, daemon=True).start()
        
        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                await response.write(f"data: {json.dumps({'type': 'error', 'error': str(chunk)})}\n\n".encode('utf-8'))
                break
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
            await response.drain()
            
        return response
    except Exception as e:
        print(f"Chat Error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_crawl(request):
    global engine
    try:
        data = await request.json()
        start_url = data.get("url", "")
        
        if not start_url:
            return web.json_response({"error": "Empty URL"}, status=400)
            
        if not start_url.startswith(("http://", "https://")):
            start_url = "https://" + start_url
            
        print(f"[*] API requested crawl for: {start_url}")
        
        # await run_crawler locally
        await run_crawler(start_url)
        
        print("[*] Crawler finished. Reloading chunks...")
        texts, metas = load_chunks(OUTPUT_FILE)
        
        if texts:
            loop = asyncio.get_running_loop()
            engine = await loop.run_in_executor(None, build_rag, texts, metas)
            return web.json_response({"status": "success", "message": f"Successfully scraped {start_url} and indexed RAG."})
        else:
            return web.json_response({"status": "error", "error": "No chunks found after crawling."}, status=400)
            
    except Exception as e:
        print(f"Crawl Error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_status(request):
    global engine
    return web.json_response({"ready": engine is not None})

app = web.Application()
app.on_startup.append(init_backend)

# Routes
app.router.add_get("/", handle_index)
app.router.add_get("/api/status", handle_status)
app.router.add_post("/api/chat", handle_chat)
app.router.add_post("/api/crawl", handle_crawl)

if __name__ == "__main__":
    print("====================================")
    print("Scout Web Interface is starting...")
    print("Open http://127.0.0.1:8080 or http://localhost:8080 in your browser.")
    print("====================================")
    web.run_app(app, host="127.0.0.1", port=8080)
