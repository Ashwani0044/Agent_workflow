# MULTI AGENT PIPELINE

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

llm = ChatOpenAI(
    model="openrouter/free",  # Free model on OpenRouter
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

researcher_llm = llm.bind_tools(tools)

def researcher_node(state: State):
    system_prompt = SystemMessage(
        content="You are an expert researcher. Use search tools if you need current information. "
                "Synthesize key findings clearly."
    )

    response = researcher_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def writer_node(state: State):
    system_prompt = SystemMessage(
        content="You are a professional tech blog writer. "
                "Read the research findings in the conversation history and craft a well-structured, "
                "engaging markdown blog post with headings, bullet points, and a summary."
    )

    response = llm.invoke([system_prompt]+state["messages"])
    return {"messages": [response]}

# build graph
builder = StateGraph(State)

# add nodes to the graph
builder.add_node("researcher", researcher_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("writer", writer_node)

# add edge
builder.add_edge(START, "researcher")

# Conditional edge for researcher: loop to tools OR proceed to writer
def route_researcher(state: State):
    # If the last message contains tool calls, go to 'tools'
    # Otherwise, research is complete -> move to 'writer'
    cond_result = tools_condition(state)
    if cond_result == "tools":
        return "tools"
        
    return "writer"

builder.add_conditional_edges(
    "researcher",
    route_researcher,
    {"tools": "tools", "writer": "writer"}
)

# Loop edge from tools back to researcher
builder.add_edge("tools", "researcher")

# edge from router to end
builder.add_edge("writer", END)

graph = builder.compile()

if __name__ == "__main__":
    query = "What are the key highlights of the latest release of Python 3.13?"
    inputs = {"messages": [HumanMessage(content=query)]}
    
    print("=== STARTING MULTI-AGENT WORKFLOW ===")
    for event in graph.stream(inputs):
        for node_name, value in event.items():
            print(f"\n[ Executed Node: {node_name} ]")
            latest_msg = value["messages"][-1]
            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                print(f"-> Calling tool with query: {latest_msg.tool_calls}")
            else:
                print(f"-> Output preview:\n{latest_msg.content[:250]}...\n")