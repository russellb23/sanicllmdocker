#~/usr/bin/env python3

import os
import re
import sys
import glob

module_path = os.path.abspath("./")
sys.path.insert(0, module_path)

from tikaxtract import *

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

        xtrctd_txt[fname] = metadata.get("tk:content")

    print(f"Successfully extracted all the pdf contents . . .")

    return xtrctd_txt

mm = preprocess_data("../datasrc/*.pdf")
