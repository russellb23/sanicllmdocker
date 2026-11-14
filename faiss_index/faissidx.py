from sanic import Sanic
from sanic.response import json, text

import os 
import faiss
import aiohttp
import glob
import nltk
import sys
import numpy as np

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from tika import parser

from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore

MAX_CTX_TOKENS=2048
# DOCS_DIR="/home/russellb/developments/DeepLearningNLP/sanicllmdocker/datasrc"
DOCS_DIR="/app/datasrc"
EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

print("cwd =", os.getcwd())
print("DOCS_DIR =", DOCS_DIR)
pathie = DOCS_DIR+"/*.pdf"
print(f"PATH: {pathie}")
# print("files =", glob.glob(pathie))

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# Ensure required NLTK is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Configure Tika to use the Docker container
os.environ["TIKA_SERVER_ENDPOINT"] = "http://server:9998"

def extract_text_and_metadata(file_path):
    """
    Extract text and metadata from a file using Apache Tika running in Docker.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    parsed = parser.from_file(file_path)

    text = parsed.get("metadata", "")#.strip()
    print(f"TEXT INSIDE xtract_metadata: {text}")
    metadata = parsed.get("metadata", {})

    return text, metadata


def chunk_sentences(text):
    """
    Splits the given text into sentences.

    Args:
        text (str):  Input text to split.

    Returns:
        list: List of sentences.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if not text.strip():
        return [] # Empty string returns empty list

    # Sentence tokenization
    sentences = sent_tokenize(text)
    return sentences



def preprocess_data(path_to_pdfs):
    """
    Extract data (text, tables, images) from pdfs and returns them as json
    """

    files = glob.glob(path_to_pdfs)
    print(f"GLOB FILES: {files}")

    xtractd_txt = []

    for idx, file in enumerate(files):
        print(f"FILE:\t{file}")

        fname_ = file.split('/')[-1]
        fname = fname_.split('.')[0]

        text, metadata = extract_text_and_metadata(file)
        print(f"\n\META-DATS: \n\n{metadata.keys()}")

        sentences_chunks = chunk_sentences(metadata.get("tk:content"))
        xtractd_txt += sentences_chunks 
        # xtractd_txt[fname] = embedder.encode(sentences_chunks,
        #                                      convert_to_numpy=True, 
        #                                      normalize_embeddings=True)

    return xtractd_txt



extracts = preprocess_data(pathie)
print(f"Extracts: {extracts[:10]}")

# Extracting dimensions
if extracts:
    documents = extracts
else:
    documents = ("")  # use your embedding width

doc_embeddings = embedder.encode(documents, convert_to_numpy=True, \
                                 normalize_embeddings=True)

print(f"Doc Embeddings: {doc_embeddings}")
# doc_embeddings = np.concatenate([v for k, v in extracts.items()])
# print(f"\n\n\ndoc_embeddings: {np.concatenate(doc_embeddings).shape}\n\n\n")
# Faiss vector store initialization
dimension = 384; #doc_embeddings.shape[1] # all-MiniLM-L6-v2
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

app.config.RESPONSE_TIMEOUT = 900
app.config.REQUEST_TIMEOUT = 900

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
    scientific context and only the context to answer the following question.
    If the answer is not in the given context say I could not find the answer
    in the context.

    Context:
    {context}

    Question: {query}
    Answer"""
    async with aiohttp.ClientSession() as session:
        async with session.post("http://server:8001/infer", json={"query": prompt}) as resp:
        # async with session.post("http://localhost:8001/infer", json={"query": prompt}) as resp:
            result = await resp.json()
            print(f"RESULT: {context}")

    return json(result)

# Execute the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, single_process=True, debug=True) 

"""
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
"""
