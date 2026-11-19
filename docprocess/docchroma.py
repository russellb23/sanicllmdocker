import asyncio
import hashlib
import os
import sqlite3
import aiohttp
from pathlib import Path

import chromadb
import numpy as np
from llama_index.core.node_parser import SentenceSplitter
from sanic import Sanic
from sanic.response import html, json
from sentence_transformers import SentenceTransformer
from tika import parser


DOCS_DIR = Path(os.getenv("DOCS_DIR", "./../datasrc"))
STATE_DIR = Path(os.getenv("STATE_DIR", "./../state"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./../chroma"))
DB_PATH = STATE_DIR / "documents.sqlite3"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
TIKA_SERVER_ENDPOINT = os.getenv("TIKA_SERVER_ENDPOINT", "http://localhost:9998")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

os.environ["TIKA_SERVER_ENDPOINT"] = TIKA_SERVER_ENDPOINT

for directory in (DOCS_DIR, STATE_DIR, CHROMA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL_NAME)
splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# Chroma saves its data under CHROMA_DIR and reloads it when the app restarts.
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

app = Sanic("ChromaDocumentService")
app.config.RESPONSE_TIMEOUT = 600
print("Sanic response timeout:", app.config.RESPONSE_TIMEOUT)
index_lock = asyncio.Lock()


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_key TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
        """)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_key(filename: str) -> str:
    return sha256(filename.encode("utf-8"))


def read_document_hash(doc_key: str):
    with connect_db() as conn:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE doc_key = ?",
            (doc_key,),
        ).fetchone()
    return row["content_hash"] if row else None


def save_document_hash(doc_key: str, filename: str, content_hash: str):
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO documents (doc_key, filename, content_hash)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_key) DO UPDATE SET
                filename = excluded.filename,
                content_hash = excluded.content_hash
        """, (doc_key, filename, content_hash))


def extract_pdf_text(pdf_path: Path) -> str:
    parsed = parser.from_file(str(pdf_path))
    content = parsed.get("metadata") or ""
    print(f"CONTENT: {type(content.get('tk:content'))}")
    if not isinstance(content.get("tk:content"), str):
        raise ValueError(f"Tika did not extract any text from the {pdf_path}.")
    return content


def make_chunk_hash(doc_key: str, chunk_text: str) -> str:
    return sha256(f"{doc_key}\0{chunk_text}".encode("utf-8"))


def update_document(filename: str, pdf_bytes: bytes) -> dict:
    safe_filename = Path(filename).name
    doc_key = document_key(safe_filename)
    content_hash = sha256(pdf_bytes)

    if read_document_hash(doc_key) == content_hash:
        return {"status": "skipped", "reason": "Document content is unchanged"}

    pdf_path = DOCS_DIR / safe_filename
    pdf_path.write_bytes(pdf_bytes)

    text = extract_pdf_text(pdf_path)
    # print(f"TEXT: {text.get("tk:content")}")
    text = text.get("tk:content")
    chunks = [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if chunk.strip()
    ]

    new_chunks = {
        make_chunk_hash(doc_key, chunk): chunk
        for chunk in chunks
    }

    # Read existing records for this document from Chroma.
    existing = collection.get(
        where={"doc_key": doc_key},
        include=["metadatas"],
    )
    old_hashes = {
        metadata["chunk_hash"]: record_id
        for record_id, metadata in zip(existing["ids"], existing["metadatas"])
    }

    old_set = set(old_hashes)
    new_set = set(new_chunks)
    unchanged = old_set & new_set
    added = new_set - old_set
    deleted = old_set - new_set

    # Remove vectors for chunks no longer present.
    if deleted:
        collection.delete(ids=[old_hashes[h] for h in deleted])

    # Embed only new or modified chunks, then store them in Chroma.
    if added:
        added_hashes = sorted(added)
        added_texts = [new_chunks[h] for h in added_hashes]
        embeddings = embedder.encode(
            added_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        collection.upsert(
            ids=[f"{doc_key}:{h}" for h in added_hashes],
            documents=added_texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "doc_key": doc_key,
                    "filename": safe_filename,
                    "chunk_hash": h,
                }
                for h in added_hashes
            ],
        )

    save_document_hash(doc_key, safe_filename, content_hash)

    return {
        "status": "updated",
        "unchanged_chunks": len(unchanged),
        "added_chunks": len(added),
        "deleted_chunks": len(deleted),
    }


init_db()


@app.get("/")
async def upload_page(_request):
    return html("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Knowledge Base</title>
  <style>
    body {
      font: 16px system-ui, sans-serif;
      max-width: 760px;
      margin: 40px auto;
      padding: 0 20px;
      color: #202124;
      background: #f5f7fb;
    }
    main {
      padding: 28px;
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
    }
    section {
      margin-top: 26px;
      padding-top: 20px;
      border-top: 1px solid #e2e5e9;
    }
    input, textarea, button { font: inherit; }
    input[type="file"], textarea { width: 100%; }
    textarea {
      box-sizing: border-box;
      min-height: 100px;
      padding: 10px;
      resize: vertical;
    }
    button {
      margin-top: 12px;
      padding: 10px 16px;
      border: 0;
      border-radius: 6px;
      color: white;
      background: #2457c5;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    .status { margin-top: 14px; white-space: pre-wrap; }
    #answer {
      margin-top: 16px;
      padding: 14px;
      background: #f5f7fb;
      border-radius: 8px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <main>
    <h1>RAG Knowledge Base</h1>
    <p>Upload PDFs, then ask questions about their contents.</p>

    <section>
      <h2>Upload a PDF</h2>
      <form id="upload-form">
        <input id="pdf-file" type="file"
               accept=".pdf,application/pdf" required>
        <br>
        <button id="upload-button" type="submit">Upload PDF</button>
      </form>
      <div id="upload-status" class="status" role="status"></div>
    </section>

    <section>
      <h2>Ask a question</h2>
      <form id="question-form">
        <textarea id="question"
                  placeholder="Enter a question about your documents"
                  required></textarea>
        <br>
        <button id="ask-button" type="submit">Ask</button>
      </form>
      <div id="answer" role="status" aria-live="polite"></div>
    </section>
  </main>

  <script>
    const uploadForm = document.getElementById("upload-form");
    const pdfInput = document.getElementById("pdf-file");
    const uploadButton = document.getElementById("upload-button");
    const uploadStatus = document.getElementById("upload-status");

    uploadForm.addEventListener("submit", async (event) => {
      event.preventDefault();

      const file = pdfInput.files[0];
      if (!file) return;

      const data = new FormData();
      data.append("file", file);

      uploadButton.disabled = true;
      uploadStatus.textContent = "Uploading and indexing…";

      try {
        const response = await fetch("/documents", {
          method: "POST",
          body: data
        });
        const result = await response.json();

        if (!response.ok) {
          throw new Error(result.error || "Upload failed");
        }

        uploadStatus.textContent = JSON.stringify(result, null, 2);
      } catch (error) {
        uploadStatus.textContent = "Error: " + error.message;
      } finally {
        uploadButton.disabled = false;
      }
    });

    const questionForm = document.getElementById("question-form");
    const questionInput = document.getElementById("question");
    const askButton = document.getElementById("ask-button");
    const answer = document.getElementById("answer");

    questionForm.addEventListener("submit", async (event) => {
      event.preventDefault();

      askButton.disabled = true;
      answer.textContent = "Searching the knowledge base…";

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: questionInput.value })
        });
        const raw = await response.text();
        let result;

        try {
            result = JSON.parse(raw);
        } catch {
        throw new Error(`Server returned HTTP ${response.status}: ${raw.slice(0, 500)}`);
        }

        if (!response.ok) {
            throw new Error(
            result.error || `Request failed with HTTP ${response.status}: ${raw}`
        );
        }

        answer.textContent = result.answer;
      } catch (error) {
        answer.textContent = "Error: " + error.message;
      } finally {
        askButton.disabled = false;
      }
    });
  </script>
</body>
</html>""")


@app.post("/documents")
async def upload_document(request):
    uploaded = request.files.get("file")
    if uploaded is None:
        return json({"error": "Upload a PDF using the 'file' field."}, status=400)

    filename = Path(uploaded.name).name
    if Path(filename).suffix.lower() != ".pdf":
        return json({"error": "Only PDF files are supported."}, status=400)

    try:
        async with index_lock:
            result = update_document(filename, uploaded.body)
        return json({"filename": filename, **result})
    except Exception as exc:
        return json({"error": str(exc)}, status=500)


@app.post("/search")
async def search_documents(request):
    body = request.json or {}
    query = body.get("query", "").strip()
    if not query:
        return json({"error": "Provide a query."}, status=400)

    if collection.count() == 0:
        return json({"results": []})

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(int(body.get("n_results", 5)), collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for i, document in enumerate(results["documents"][0]):
        matches.append({
            "text": document,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return json({"results": matches})


@app.post("/ask")
async def ask_question(request):
    body = request.json or {}
    question = body.get("question", "").strip()

    if not question:
        return json({"error": "Enter a question."}, status=400)

    if collection.count() == 0:
        return json({"error": "No documents have been indexed yet."}, status=400)

    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(5, collection.count()),
        include=["documents", "metadatas"],
    )

    documents = results["documents"][0]
    if not documents:
        return json({"error": "No relevant document chunks were found."}, status=404)

    context = "\n\n".join(documents)
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you could not find it in the documents.

Context:
{context}

Question: {question}

Answer:"""

    try:
        timeout = aiohttp.ClientTimeout(total=600)
        print("Sanic response timeout:", app.config.RESPONSE_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                    "http://localhost:8001/infer",
                json={"query": prompt},
            ) as response:
                response.raise_for_status()
                inference_result = await response.json()

        # Adapt this extraction if your inference service uses a different
        # response field name.
        answer_text = (
            inference_result.get("answer")
            or inference_result.get("response")
            or inference_result.get("result")
        )

        if answer_text is None:
            answer_text = str(inference_result)

        return json({
            "answer": answer_text,
            "sources": results["metadatas"][0],
        })

    except Exception as exc:
        return json({"error": f"Inference request failed: {exc}"}, status=502)


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", "8002")),
        single_process=True,
    )
