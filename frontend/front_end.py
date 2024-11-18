import gradio as gr
import requests
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from utils.llms.rags.chatbot import create_chatbot


llm_url = "http://0.0.0.0:1234/choi_chatbot"


model = create_chatbot()


def random_response(message, history):
    for i in range(10):
        yield str(message)


def chat_response(message, history):
    config = {"configurable": {"thread_id": "abc123"}}
    input_messages = [HumanMessage(message)]

    output = model.invoke({"messages": input_messages}, config)
    return output["messages"][-1].content


def run():
    global llm_url
    # with gr.Blocks(fill_height=True) as demo:
    #     chatbot = gr.ChatInterface(
    #         random_response,
    #         type="messages",
    #     )
    # demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

    gr.ChatInterface(
        chat_response,
        type="messages",
    ).launch()


if __name__ == "__main__":
    run()
