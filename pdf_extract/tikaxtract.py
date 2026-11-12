from tika import parser
import os

# Configure Tika to use the Docker container
os.environ["TIKA_SERVER_ENDPOINT"] = "http://localhost:9998"

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



