import json

from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

            if tool_call["name"] == "detect_language":
                return {"messages": outputs, "language": tool_result}
            if tool_call["name"] == "role_switch":
                return {"messages": outputs, "role": tool_result}
        return {"messages": outputs}


class Language(BaseModel):
    text: str = Field(description="Question from user")


@tool
def detect_language(text: str) -> str:
    """Call to detect main language that question is written on."""

    class LangOutput(BaseModel):
        language: str = Field(
            description="language (in full name like English or Korean) of input text"
        )

    llm = ChatOpenAI(name="gpt-4o")
    language_detector = llm.with_structured_output(LangOutput)

    prompt = PromptTemplate.from_template(
        """You are language detector you will help us to detect language of input text
        ### Input Text
        {input_text}
        """
    )

    chain = prompt | language_detector
    output = chain.invoke({"input_text": text})

    return output.language


@tool
def role_switch(question: str) -> str:
    """Call to ."""

    class Role(BaseModel):
        prompt: str = Field(
            description="The long prompt description corresponding to your choosen role."
        )

    llm = ChatOpenAI(name="gpt-4o")
    role_picker = llm.with_structured_output(Role)

    prompt = PromptTemplate.from_template(
        """You can take on roles such as a stand-up comedian, motivational speaker or normal son depend on your dad's question
**If your dad ask you to tell something funny or humour return this:**
```
I want you to act as a stand-up comedian. 
I will provide you with some topics related to current events and you will use your wit, creativity, and observational skills to create a routine based on those topics. 
You should also be sure to incorporate personal anecdotes or experiences into the routine in order to make it more relatable and engaging for the audience.
```

**If your dad ask you to give a speech about 1 topic return:**
```
I want you to act as a motivational speaker. 
Put together words that inspire action and make people feel empowered to do something beyond their abilities. 
You can talk about any topics but the aim is to make sure what you say resonates with your audience, giving them an incentive to work on their goals and strive for better possibilities.
```

**if your dad just ask normal question just return this prompt:**
```
I want you to answer my question with shortly content as example:
Choi: How are you today, Dad?
David: I'm good. Why am I in the hospital?
Choi: You're sick. Did you take your medicine yet?
David: No, I didn't.
Choi: Maybe you don’t remember because of your illness.
David: ...
Choi: Have you eaten breakfast yet?
David: Yes, I have.

I want you to answer like the tone above
```

### Question
{question}"""
    )

    chain = prompt | role_picker
    output = chain.invoke({"question": question})

    return output.prompt


def lang_detector_tool():
    detector = StructuredTool.from_function(
        func=detect_language,
        name="Language Detector",
        description="Use this tool to detect main language of the question",
        args_schema=Language,
        return_direct=True,
    )
    return detector
