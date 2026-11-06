#~/usr/bin/env python3

import os
import re
import sys
import glob

module_path = os.path.abspath("./")
sys.path.insert(0, module_path)

from xtract import *

def extract_trailing_number(filename):
    """
    Extracts the trailing number from a filename.
    Returns None if no trailing number is found.
    """
    match = re.search(r'(\d+)(?=\.[^.]*$|$)', filename)
    return int(match.group(1)) if match else None


def read_files_as_string(file_paths, encoding='utf-8'):
    """
    Reads multiple files and concatenates their contents into a single string.

    :param file_paths: List of file paths to read.
    :param encoding: Encoding for text files (default: utf-8).
    :return: Combined string of all file contents.
    """
    if not isinstance(file_paths, list):
        raise TypeError("file_paths must be a list of file paths.")

    combined_content = []

    for path in file_paths:
        if not isinstance(path, str):
            raise TypeError(f"File path must be a string, got {type(path)}.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            with open(path, 'r', encoding=encoding) as f:
                combined_content.append(f.read())
        except UnicodeDecodeError:
            # If file is binary or encoding fails, read in binary mode
            with open(path, 'rb') as f:
                combined_content.append(f.read().decode(encoding, errors='replace'))
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return "".join(combined_content)


def sort_files_by_trailing_number(directory):
    """
    Sorts files in the given directory based on trailing numbers.
    Files without trailing numbers are placed at the end.
    """
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return []

    # Sort using the extracted trailing number; None goes last
    sorted_files = sorted(
        files,
        key=lambda f: (extract_trailing_number(f) is None, extract_trailing_number(f))
    )
    return sorted_files



files = glob.glob("../datasrc/*.pdf")
extractor = PDFExtractor(output_base_dir="extracted_pdfs")
xtrctd_txt = dict()

for idx, file in enumerate(files):
    print(f"FILE:\n{file}")
    xtractor = extractor.extract_content(
            pdf_path = file,
            strategy="hi_res",
            extract_images=False,
            extract_tables=False)

    print(f"Extracted content: {xtractor}")
    xtractd_path = os.listdir("./extracted_pdfs/")[idx]
    data_path = "./extracted_pdfs/"+xtractd_path+"/text"
    data_ = glob.glob(data_path+"/narrativetext_*.txt")
    data_content = sorted(data_, \
            key=lambda f: (extract_trailing_number(f) is None, \
            extract_trailing_number(f)))
    text = read_files_as_string(data_content)
    xtrctd_txt[xtractd_path] = text
    print(f"Key: \
          {xtractd_path}")

print(f"Successfully extracted all the pdf contents . . .")


print(f"Keys in xtracted text: {xtrctd_txt.keys()}")
