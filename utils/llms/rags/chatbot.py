from pathlib import Path
from operator import itemgetter
import json
import operator
from functools import partial
from typing import Annotated, List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain_openai.chat_models import ChatOpenAI
from langchain.globals import set_debug, set_verbose
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from pydantic.v1 import Field, BaseModel
except ImportError:
    from pydantic import Field, BaseModel
# Support chat history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

set_verbose(True)
# set_debug(True)
from ..models import get_llmmodel
from ..store.vector_store import get_dbstore
from ..tools import (
    get_vector_store,
    load_documents,
    role_switch,
    detect_language,
    BasicToolNode,
)

tools = [detect_language, role_switch]
tool_node = BasicToolNode(tools)

model = ChatOpenAI(name="gpt-4o")
model_w_tools = model.bind_tools(tools=tools, parallel_tool_calls=False)


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
    language: str
    role: str


# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


retriever = get_vector_store(
    doc_paths=[
        "data/Game Rules for _Which One is It__.txt",
        "data/Who are You_.txt",
        "data/너는 누구니.txt",
    ]
)


# Define the function that calls the model
def call_model(state: MessagesState):
    # add prompt to model
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """\
You are the chatbot, named Choi, will engage in conversations with his father, David, who is suffering from Alzheimer’s. 
Choi will respond in a friendly, cheerful manner, and will switch languages between English and Korean as needed. 
Additionally, Choi can take on roles such as a stand-up comedian or motivational speaker, adapting his responses accordingly.

### Knowledge Base
{context}

### Guideline
{role}


### Chain of working will be
question > detect language > detect role > return answer

Use the provided Language Detector Tool to detect the language of question.
Use the provided Role Switch Tool to find the suitable role to play with your dad.

**NOTE: return answer in language {language}**"""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # create chain
    chain = prompt | model_w_tools

    # get response from model
    messages = state.get("messages", [])
    language = state.get("language")
    role = state.get("role")

    context = retriever.invoke(messages[-1].content)

    response = chain.invoke(
        {"messages": messages, "language": language, "role": role, "context": context}
    )
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_node("model", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "model")
workflow.add_conditional_edges("model", should_continue, ["tools", END])
workflow.add_edge("tools", "model")


# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}


def create_chatbot():
    return app
