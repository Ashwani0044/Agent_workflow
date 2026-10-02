# CONDITIONAL ROUTING 

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
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

# register tools with the llm so that it knows it can call them
llm_with_tools =  llm.bind_tools(tools)

def researcher_node(state: State):
    # LLM will decide whether to respond directly or to call tool first
    response = llm_with_tools.invoke(state["messages"]) 
    return {"messages": [response]}   

# build graph
builder = StateGraph(State)

# add nodes to the graph
builder.add_node("researcher", researcher_node)
builder.add_node("tools", ToolNode(tools)) # built-in node to call the tool

# add edges
builder.add_edge(START, "researcher")

# conditional routing:
# checking if last message from researcher contains tool call
# if true, then route it to 'tools' node otherwise END
builder.add_conditional_edges(
    "researcher",
    tools_condition
)
# loop edge: after executing tools, return to researcher to synthesize results
builder.add_edge("tools", "researcher")

graph = builder.compile()

if __name__ == "__main__":
    query = "What were the latest major announcements from OpenAI?"
    inputs = {"messages": [HumanMessage(content=query)]}

    print("-------WORKFLOW EXECUTION-------")
    for event in graph.stream(inputs):
        for node_name, value in event.items():
            print(f"\n[ Executed Node: {node_name} ]")
            latest_msg = value["messages"][-1]
            
            # Print tool calls if present, otherwise print content
            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                print(f"-> Calling tool with query: {latest_msg.tool_calls}")
            else:
                print(f"-> Content snippet: {latest_msg.content[:200]}...")
