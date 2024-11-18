import os
from typing import List, Dict
from langchain_text_splitters.base import TextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredExcelLoader,
    PyMuPDFLoader,
    UnstructuredEmailLoader,
    TextLoader,
)


def load_documents(
    doc_paths: List[str], text_splitter: TextSplitter = None
) -> List[Document]:
    documents = []

    for doc_path in doc_paths:
        doc_type = os.path.splitext(doc_path)[-1]
        if doc_type == ".docx":
            loader = Docx2txtLoader(doc_path)
        elif doc_type == ".txt":
            loader = TextLoader(doc_path)
        elif doc_type == ".xlsx":
            loader = UnstructuredExcelLoader(doc_path)
        elif doc_type == ".pdf":
            loader = PyMuPDFLoader(doc_path)
        elif doc_type == ".eml":
            loader = UnstructuredEmailLoader(doc_path)
        else:
            raise ValueError(f"Unknown document type {doc_type}")
        docs = loader.load_and_split()
        if text_splitter is None:
            documents.extend(docs)
        else:
            documents.extend(text_splitter.split_documents(docs))
    return documents
