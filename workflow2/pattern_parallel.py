import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAi
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# messages are preserved, and individual reviewers post their analysis directly into State
class State(TypedDict):
    topic: str
    seo_feedback: str
    tech_feedback: str 
    style_feedback: str
    final_report: str 

llm = ChatOpenAi(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# defining parallel working nodes
def seo_analyst(state: State):
    prompt = f"Analyze the SEO potential and suggest key search intent keywords for this topic: '{state['topic']}'."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"seo_feedback": response.content}

def tech_reviewer(state: State):
    prompt = f"Identify core technical depth, architecture considerations, or potential pitfalls for this topic: '{state['topic']}'."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"tech_feedback": response.content}

def style_critic(state: State):
    prompt = f"Suggest tone, target audience persona, and engaging hooks for writing about: '{state['topic']}'."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"style_feedback": response.content}

# reducer/aggregator node (where all the parallel working nodes will come together)
def aggregator(state: State):
    prompt = (
        f"Synthesize the following 3 expert reviews into a cohesive content outline for topic: '{state['topic']}'\n\n"
        f"--- SEO FEEDBACK ---\n{state['seo_feedback']}\n\n"
        f"--- TECH FEEDBACK ---\n{state['tech_feedback']}\n\n"
        f"--- STYLE FEEDBACK ---\n{state['style_feedback']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}

# Build Graph with Fan-Out Edges
builder = StateGraph(State)

builder.add_node("seo", seo_analyst)
builder.add_node("tech", tech_reviewer)
builder.add_node("style", style_critic)
builder.add_node("aggregator", aggregator)

# FAN-OUT: Map START directly to multiple nodes simultaneously
builder.add_edge(START, "seo")
builder.add_edge(START, "tech")
builder.add_edge(START, "style")

# FAN-IN / REDUCE: All 3 nodes lead to the aggregator
builder.add_edge("seo", "aggregator")
builder.add_edge("tech", "aggregator")
builder.add_edge("style", "aggregator")

builder.add_edge("aggregator", END)

graph = builder.compile()

if __name__ == "__main__":
    inputs = {"topic": "Building Real-Time Event-Driven Systems with Python & Kafka"}
    
    print("=== EXECUTING PARALLEL WORKFLOW ===")
    for event in graph.stream(inputs):
        for node_name, value in event.items():
            print(f"\n Completed Node: [{node_name}]")
            if node_name == "aggregator":
                print("\n=== FINAL SYNTHESIZED REPORT ===")
                print(value["final_report"])