from __future__ import annotations
from hashlib
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pypdf import PdfReader
from config import Config

@dataclass
class chunk:
    text: str
    metadata: dict =field(default_factory=dict) #The extra information about the chunk, such as its source, page number, or any other relevant metadata.

    @property
    def chunk_id(self) -> str: #Generate a unique identifier for the chunk based on its content and metadata.

        #I want to generate a unique identifier for the chunk based on its content and metadata. 
        #This method will use a hashing algorithm to create a hash value from the chunk's text and metadata, and return the first 12 characters of the hash as the chunk's unique identifier.

        return hashlib.md5(self.text.encode()).hexdigest()[:12]

    def _split_text(self): #I want to split the text into smaller chunks based on the specified chunk size and overlap. This method will return a list of chunk objects, each containing a portion of the original text and its corresponding metadata.
        pass

    def load_pdf(): # I want to load a PDF document and extract its text content. This method will read the PDF file, extract the text from each page, and return a list of chunk objects containing the extracted text and relevant metadata.
        pass

    def load_text():
        pass

    def load_document():
        pass