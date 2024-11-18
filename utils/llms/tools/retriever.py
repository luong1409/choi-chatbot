from typing import Dict, List

from loguru import logger
from functools import lru_cache
from .file_reader import load_documents
from ..models import get_embedder
from ..store.vector_store import get_dbstore

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader


def get_vector_store(doc_paths: List[str]):
    embedding_function = get_embedder(model_name="text-embedding-ada-002")
    db_store = get_dbstore(engine="chroma", embedding_function=embedding_function)
    text_splitter = SemanticChunker(embeddings=embedding_function)
    pages = load_documents(doc_paths=doc_paths, text_splitter=text_splitter)

    logger.success("Done load FAISS")
    db_store.add_page(pages)
    db_store.from_documents(pages)
    retriever = db_store.as_retriever(
        search_type="mmr", search_kwargs=dict(k=3, lambda_mult=0.7)
    )
    return retriever
