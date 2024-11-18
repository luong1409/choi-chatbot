import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import *
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from fastapi.responses import StreamingResponse
from utils.llms.rags.chatbot import create_chatbot

app = FastAPI(
    title="SecondBrain App",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

model = create_chatbot()


class ChatInput(BaseModel):
    message: str
    files: List[Dict]


def chat_streamer(msg: ChatInput):
    config = {"configurable": {"thread_id": "abc123"}}
    input_messages = [HumanMessage(msg.message)]

    output = model.invoke({"messages": input_messages}, config)
    return output["messages"][-1].content


@app.post("/choi_chatbot")
def chatbot(msg: ChatInput) -> str:
    print("msg: ", msg)
    return chat_streamer(msg)


def run():
    import uvicorn

    uvicorn.run("backend.choi_chatbot:app", host="0.0.0.0", port=1234, reload=True)


if __name__ == "__main__":
    run()
