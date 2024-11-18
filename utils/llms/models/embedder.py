from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


def get_embedder(
    model_provider: str = "openai",
    model_name: str = "text-embedding-3-large",
    **kwargs,
):
    kwargs = kwargs or {}
    kwargs["model"] = model_name

    match model_provider:
        case "openai":
            return OpenAIEmbeddings(**kwargs)
        case "ollama":
            return OllamaEmbeddings(**kwargs)
        case "azure-openai":
            return AzureOpenAIEmbeddings(**kwargs)
