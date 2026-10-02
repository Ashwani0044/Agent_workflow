# human oversight and thread memory
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
        content="You are an expert researcher. Use search tools to collect factual data. Synthesize findings."
    )
    response = researcher_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def writer_node(state: State):
    system_prompt = SystemMessage(
        content="You are a tech blogger. Write a well-structured markdown blog post based on the research in history."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# build graph
builder = StateGraph(State)

# add nodes to the graph
builder.add_node("researcher", researcher_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("writer", writer_node)

# add edge
builder.add_edge(START, "researcher")

# function to router research to tools if tool call is there otherwise to writer node
def route_researcher(state: State):
    cond_result = tools_condition(state)
    if cond_result == "tools":
        return "tools"
    return "writer"

builder.add_conditional_edges(
    "researcher",
    route_researcher,
    {"tools": "tools", "writer": "writer"}
)

builder.add_edge("tools", "researcher")
builder.add_edge("writer", END)

memory = MemorySaver()

# telling langgraph to halt before executing the writer node for human approval
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["writer"]
)

if __name__ == "__main__":
    # Thread ID identifies this specific workflow execution/session
    config = {"configurable": {"thread_id": "session_1"}}
    
    query = "What are the latest updates on Python 3.13 features?"
    inputs = {"messages": [HumanMessage(content=query)]}
    
    print("=== PART 1: RUNNING UNTIL INTERRUPT ===")
    for event in graph.stream(inputs, config):
        for node_name, value in event.items():
            print(f"[ Executed Node: {node_name} ]")
    
    # Check graph status after pause
    snapshot = graph.get_state(config)
    print("\n----------------------------------------")
    print(f"GRAPH PAUSED BEFORE NEXT STEP: {snapshot.next}")
    print("----------------------------------------\n")
    print("=== LATEST RESEARCH SUMMARY IN STATE ===")
    print(snapshot.values["messages"][-1].content[:300] + "...\n")
    
    # Prompt the user to approve or exit
    user_approval = input("Type 'yes' to approve research and generate article: ")
    
    if user_approval.lower() == "yes":
        print("\n=== PART 2: RESUMING GRAPH EXECUTION ===")
        # Passing None as input tells LangGraph to pick up where it left off
        for event in graph.stream(None, config):
            for node_name, value in event.items():
                print(f"[ Executed Node: {node_name} ]")
                if node_name == "writer":
                    print("\n=== FINAL BLOG POST DRAFT ===")
                    print(value["messages"][-1].content)
    else:
        print("Execution cancelled by human operator.")