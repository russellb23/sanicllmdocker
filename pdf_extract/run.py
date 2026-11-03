#~/usr/bin/env python3

import os
import sys
import glob

module_path = os.path.abspath("./")
sys.path.insert(0, module_path)

from xtract import *

files = glob.glob("../datasrc/*.pdf")

extractor = PDFExtractor(output_base_dir="extracted_pdfs")
for file in files:
    print(f"FILE:\n{file}")
    xtractor = extractor.extract_content(
            pdf_path = file,
            strategy="hi_res",
            extract_images=False,
            extract_tables=False)

    print(f"Extracted content: {xtractor}")
