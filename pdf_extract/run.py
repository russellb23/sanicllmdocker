#~/usr/bin/env python3

import os
import re
import sys
import glob
import nltk

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

model = SentenceTransformer("all-MiniLM-L6-v2")

module_path = os.path.abspath("./")
sys.path.insert(0, module_path)

from tikaxtract import *

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def chunk_sentences(text):
    """
    Splits the given text into sentences.

    Args:
        text (str): Input text to split.

    Returns:
        list: List of sentences.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if not text.strip():
        return []  # Empty input returns empty list

    # Sentence tokenization
    sentences = sent_tokenize(text)
    return sentences



def preprocess_data(path_to_pdfs, output_dir="extracted_pdfs"):
    """
    Extract data (text, tables, images) from pdfs and process them as json
    """

    files = glob.glob("../datasrc/*.pdf")

    xtrctd_txt = dict()

    for idx, file in enumerate(files):
        print(f"FILE:\n{file}")

        fname_ = file.split('/')[-1]
        fname = fname_.split('.')[0]

        text, metadata = extract_text_and_metadata(file)

        print(f"Extracted content: {metadata.get("tk:content")}")

        xtrctd_txt[fname] = model.encode(chunk_sentences(metadata.get("tk:content")))

    print(f"Successfully extracted all the pdf contents . . .")

    return xtrctd_txt

mm = preprocess_data("../datasrc/*.pdf")
# pap = mm.get("protein_context_shapes_the_specificity_of_SH3_proteins")
# chunked_pap = chunk_sentences(pap)
# embeddings = model.encode(chunked_pap)
print(mm.get("protein_context_shapes_the_specificity_of_SH3_proteins").shape)
