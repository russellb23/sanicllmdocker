#!/usr/bin/env python3

import os
import sys
import glob

module_path = os.path.abspath("./")
print(f"Module path: {module_path}")
sys.path.insert(0, module_path)

print(f"Sys path: {sys.path}")

from xtract_pdf import *


files = glob.glob("../datasrc/*.pdf")

print("Starting pdf extraction ... ")
extractor = PDFExtractor(output_base_dir="extracted_pdfs")

for file in files:
    print(f"File path: {file}")
    xtractor = extractor.extract_content(
            pdf_path = file,
            strategy = 'hi_res',
            extract_images = True,
            extract_tables = True)

print(f"Extraction completed:\nContents: {xtractor}")
