from sanic import Sanic
from sanic.response import json, text

import os 
import faiss
import aiohttp

from sentence_transformers import SentenceTransformer

from llama_index.core import StorageContext, Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader


MAX_CTX_TOKENS=2048
# DOCS_DIR="/home/russellb/developments/DeepLearningNLP/sanicllmdocker/datasrc"
DOCS_DIR="/app/datasrc"
EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"


# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def load_pdfs(path_name: str) -> dict:
    docs = []
    for file in os.listdir(path_name):
        path = os.path.join(path_name, file)
        if file.lower().endswith(".pdf"):
            docs.extend(PDFReader().load_data(file=path))
        else:
            docs.extend(SimpleDirectoryReader(input_files=[path]).load_data())
    return docs

# Process document text as json
docs_ = load_pdfs(DOCS_DIR)
documents_ = [dc.json() for dc in docs_]
documents = [' '.join(dcs) for dcs in documents_]

doc_embeddings = embedder.encode(documents, convert_to_numpy=True, \
                                 normalize_embeddings=True)

# Settings Sentence splitter
Settings.text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=32,
        separator=" ",
        paragraph_separator="\n\n")

# Settings embedding
# Settings.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Faiss vector store initialization
dimension = 384
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(doc_embeddings)

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(
        vector_store=vector_store)

# Index construction
# index = VectorStoreIndex.from_documents(docs_,
#                         storage_context=storage_context)


# Sanic application
app = Sanic("SanicFAISS")

app.config.RESPONSE_TIMEOUT = 12000
app.config.REQUEST_TIMEOUT = 12000

@app.post("/faiss")
async def search_faiss(request):
    query = request.json["query"]
    print(f"Query: {query}")
    query_embedding = embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = faiss_index.search(query_embedding, 3)
    retrieved_docs = [documents[i][:MAX_CTX_TOKENS] for i in indices[0]]
    print(f"\n\n{indices[0]}\n\n")
    # Build prompt
    context = "\n\n".join(retrieved_docs)

    # Constructing prompt
    prompt=f"""You are a helpful scientific assistant. Use the following
    scientific context to answer the following question.

    Context:
    {context}

    Question: {query}
    Answer"""
    async with aiohttp.ClientSession() as session:
        async with session.post("http://samy:8001/infer", json={"query": prompt}) as resp:
            result = await resp.json()
            print(f"RESULT: {context}")

    return json(result)

# Execute the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, single_process=True, debug=True) 
