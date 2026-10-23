from sanic import Sanic
from sanic.response import json, text

import os 

from sentence_transformers import SentenceTransformer

from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader


DOCS_DIR="../datasrc"
EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
# Load documents
# docs = SimpleDirectoryReader(DOCS_DIR).load_data()

# Create FAISS vector store
# faiss_store = FaissVectorStore()

# Build index
# index = VectorStoreIndex.from_documents(docs, vector_store=faiss_store)

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def load_pdfs(path_name: str) -> dict:
    docs = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if file.lower().endswith(".pdf"):
            docs.extend(PDFReader().load_data(file=path))
        else:
            docs.extend(SimpleDirectoryReader(input_files=[path]).load_data())
    return docs

# Process document text as json
docs_ = load_pdfs(DOCS_DIR)
docums = [dc.json() for dc in docs_]
docs = [' '.join(dcs) for dcs in docums]

print(f"\n\nDocuments: {Document.example()}\n\n")
print(f"\n\nDocuments: {docs_[1]}\n\n")

# Pipeline of transformation
pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=32),
            HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
            ])
            # TitleExtractor(),

nodes = pipeline.run(documents=docs_)

# Document embeddings
# doc_embeddings = embedder.encode(documents, convert_to_numpy=True, \
#                                  normalize_embeddings=True)
# dims = doc_embeddings.shape[1]
# faiss_index = faiss.IndexFlatIP(dims)
# vector_store = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# index = VectorStoreIndex(nodes)

# print("==============================")
# print(f"\n\n\nIndex: {index}\n\n\n")
# print("FAISS Vector store index created successfully!")
# print("==============================")


app = Sanic("SanicFAISS")

@app.get("/faiss")
async def faiss_index(request):
    return text(nodes[0].get_text())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, single_process=True)
