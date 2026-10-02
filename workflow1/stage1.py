import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(
    model="openrouter/free",  # Free model on OpenRouter
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

def researcher_node(state: State):
    response = llm.invoke(state["messages"])

    return {"messages": [response]}

builder = StateGraph(State)

builder.add_node("researcher", researcher_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_input = {
        "messages": [HumanMessage(content="Give me 3 key facts about quantum computing.")]
    }
    
    output = graph.invoke(initial_input)
    
    print("--- ASSISTANT RESPONSE ---")
    print(output["messages"][-1].content)