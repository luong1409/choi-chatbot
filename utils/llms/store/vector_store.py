from typing import *
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class DBStoreFactory(ABC):
    pages: List[Document] = []

    def add_page(self, pages: List[str]):
        self.pages.extend(pages)
        return self

    @abstractmethod
    def as_retriever(self):
        pass


class DBStoreFAISS(DBStoreFactory):
    def __init__(self, embedding_function) -> None:
        super().__init__()
        self.embedding_function = embedding_function

    def from_documents(self, pages):
        self.db = FAISS.from_documents(pages, self.embedding_function)
        return self

    def add_documents(self, pages):
        self.db.add_documents(pages)
        return self

    def as_retriever(self, *args, **kwargs):
        return self.db.as_retriever(*args, **kwargs)


class DBStoreChroma(DBStoreFactory):
    def __init__(self, embedding_function) -> None:
        super().__init__()
        self.embedding_function = embedding_function

    def from_documents(self, pages):
        self.db = Chroma.from_documents(pages, self.embedding_function)
        return self

    def add_documents(self, pages):
        self.db.add_documents(pages)
        return self

    def as_retriever(self, *args, **kwargs):
        return self.db.as_retriever(*args, **kwargs)


class TextSplitter:
    pass


def get_dbstore(
    engine: str = "faiss",
    *args,
    **kwargs,
):
    match engine:
        case "chroma":
            return DBStoreChroma(*args, **kwargs)
        case _:
            return DBStoreFAISS(*args, **kwargs)
