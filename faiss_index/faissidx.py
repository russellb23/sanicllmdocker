from sanic import Sanic
from sanic.response import json, text

import os 
import faiss

from sentence_transformers import SentenceTransformer

from llama_index.core import StorageContext, Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader


DOCS_DIR="/home/russellb/developments/DeepLearningNLP/sanicllmdocker/datasrc"
# DOCS_DIR="/app/datasrc"
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

# Settings Sentence splitter
Settings.text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=32,
        separator=" ",
        paragraph_separator="\n\n")

# Settings embedding
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# Faiss vector store initialization
dimension = 384
faiss_index = faiss.IndexFlatIP(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(
        vector_store=vector_store)

# Index construction
index = VectorStoreIndex.from_documents(docs_,
                        storage_context=storage_context)


scores, indices = index.search()
app = Sanic("SanicFAISS")

@app.get("/faiss")
async def faiss_index(request):
    query = request.json["query"]
    query_embedding = HuggingFaceEmbedding(mode_name=EMBED_MODEL_NAME).encode(
            [query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_embedding, 3)
    retrieved_docs = [documents[i] for i in indices[0]]
    # Build prompt
    context = "\n\n".join(retrieved_docs)
    # Constructing prompt
    prompt=f"""You are a helpful scientific assistant. Use the following
    scientific context to answer the following question.

    Context:
    {context}

    Question: {query}
    Answer"""
    return text("This should go to the LLM-endpoint")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, single_process=True, debug=True) 
