import os
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class LLMModel:
    def __init__(self, model_name: str = "openai"):
        pass


def get_llmmodel(
    model_provider: str = "azure-openai",
    model_name: str = "gpt-3.5-turbo",
    has_rate_limit: bool = True,
    **kwargs,
):
    kwargs = kwargs or {}
    kwargs["model"] = model_name
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )
    if has_rate_limit is True:
        if kwargs.get("rate_limiter", None) is None:
            kwargs["rate_limiter"] = rate_limiter

    match model_provider:
        case "openai":
            return ChatOpenAI(**kwargs)
        case "ollama":
            return ChatOllama(**kwargs)
        # case "azure-openai":
        #     from langchain_openai import AzureChatOpenAI

        #     os.environ["OPENAI_API_VERSION"] = "2024-02-01"
        #     os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-renew-account.openai.azure.com/"
        #     os.environ["AZURE_OPENAI_API_KEY"] = "fdc23d09116a4e039aaf9b7f9cd9cfa3"
        #     llm = AzureChatOpenAI(
        #         api_key="fdc23d09116a4e039aaf9b7f9cd9cfa3",
        #         azure_endpoint="https://test-renew-account.openai.azure.com/",
        #         azure_deployment="gpt-4o",  # or your deployment
        #         api_version="2024-02-01",  # or your api version
        #         temperature=0,
        #         max_tokens=None,
        #         timeout=None,
        #         max_retries=2,
        #     )
        #     return llm


if __name__ == "__main__":
    model = get_llmmodel(
        model_provider="ollama", model_name="llama3.1", has_rate_limit=False
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    print(model.invoke(messages))
    print("Done")
