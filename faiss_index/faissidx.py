from sanic import Sanic
from sanic.response import json, text

import os 
import faiss
import aiohttp
import glob
import nltk
import sys
import openai
import numpy as np

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

from llama_index.core import StorageContext, Settings, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter


MAX_CTX_TOKENS=8192
DOCS_DIR="/home/russellb/developments/DeepLearningNLP/sanicllmdocker/datasrc"
# DOCS_DIR="/app/datasrc"
# EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME="BAAI/bge-small-en-v1.5"
# EMBED_MODEL_NAME="perplexity-ai/pplx-embed-v2-context-9b"

pathie = DOCS_DIR+"/*.pdf"

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Ensure required NLTK is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')



# Configure Tika to use the Docker container
os.environ["TIKA_SERVER_ENDPOINT"] = "http://localhost:9998"
from tika import parser

def extract_text_and_metadata(file_path):
    """
    Extract text and metadata from a file using Apache Tika running in Docker.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    parsed = parser.from_file(file_path)

    text = parsed.get("metadata", "")#.strip()
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

def semantic_chunking(text):
    """
    Splits text using semantics
    Args:
        text (str): Outout of Tika server
    Returns:
        list: List of semantic sentences
    """
    try:

        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        if not text.strip():
            return []
        # Create a Document object
        doc = Document(text=text)
        # Semantic chunking
        chunker = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        chunks = chunker.split_text(text)
        return chunks

    except Exception as e:
        raise RuntimeError(f"Error chunking text: {e}")



def preprocess_data(path_to_pdfs):
    """
    Extract data (text, tables, images) from pdfs and returns them as json
    """

    files = glob.glob(path_to_pdfs)

    xtractd_txt = []

    for idx, file in enumerate(files):

        fname_ = file.split('/')[-1]
        fname = fname_.split('.')[0]

        text, metadata = extract_text_and_metadata(file)

        if not isinstance(text.get("tk:content"), str):# or not text.strip():
            raise ValueError(f"No text extracted from {file}")

        # sentences_chunks = chunk_sentences(metadata.get("tk:content"))
        semantic_chunks = semantic_chunking(metadata.get("tk:content"))
        xtractd_txt += semantic_chunks 

    return xtractd_txt



extracts = preprocess_data(pathie)
print(f"Extracts: {extracts[:10]}")

"""
###################################### Evaluation routine ####################
eval_documents = extracts
data_generator = DatasetGenerator.from_documents(eval_documents)
eval_questions = data_generator.generate_questions_from_nodes()

print(f"Evaluation questions:\n\n{eval_questions}")

from sentence_transformers import SentenceTransformer, util

def relevance_score(query, retrieved_texts):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = []
    for text in retrieved_texts:
        text_emb = model.encode(text, convert_to_tensor=True)
        scores.append(util.cos_sim(query_emb, text_emb).item())
    return sum(scores) / len(scores)

def faithfulness_score(answer, retrieved_texts):
    context = "\n".join(retrieved_texts)
    prompt = f"Check if the answer is fully \
            supported by the context.\nContext:\n{context}\nAnswer: \
            \n{answer}\nReturn a score 0-1."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return float(response.choices[0].message['content'])


#############################################################################
"""

# Extracting dimensions
if extracts:
    documents = extracts
else:
    documents = ("")  # use your embedding width

doc_embeddings = embedder.encode(documents, convert_to_numpy=True, \
                                 normalize_embeddings=True)

print(f"Doc Embeddings: {doc_embeddings}")
dimension = doc_embeddings.shape[1] # all-MiniLM-L6-v2
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(doc_embeddings)

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(
        vector_store=vector_store)


# Sanic application
app = Sanic("SanicFAISS")

app.config.RESPONSE_TIMEOUT = 300
app.config.REQUEST_TIMEOUT = 300

@app.post("/faiss")
async def search_faiss(request):
    query = request.json["query"]
    print(f"Query: {query}")
    query_embedding = embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = faiss_index.search(query_embedding, 1) # K = 10
    retrieved_docs = [documents[i][:MAX_CTX_TOKENS] for i in indices[0]]
    print(f"\n\n{indices[0]}\n\n")
    # Build prompt
    context = "\n\n".join(retrieved_docs)

    # Constructing prompt
    prompt=f"""You are a helpful scientific assistant. Use the following
    scientific context to answer the following question.
    If the answer is not in the given context say I could not find the answer
    in the context.

    Context:
    {context}

    Question: {query}
    Answer"""
    async with aiohttp.ClientSession() as session:
        # async with session.post("http://server:8001/infer", json={"query": prompt}) as resp:
        async with session.post("http://localhost:8001/infer", json={"query": prompt}) as resp:
            result = await resp.json()
            print(f"RESULT: {context}")

    return json(result)

# Execute the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, single_process=True, debug=True) 
