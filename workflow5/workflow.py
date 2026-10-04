import os
import json
from dotenv import load_dotenv
from typing import Annotated, TypedDict
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# SUB-GRAPH (writer-evaluator loop)

class WriterSubState(TypedDict):
    topic: str
    aggregated_insights: str
    draft: str
    feedback: str
    score: int
    revisions: int

def sub_writer_node(state: WriterSubState):
    revisions = state.get("revisions", 0)
    feedback = state.get("feedback", "")

    if feedback:
        prompt = (
            f"Topic: {state['topic']}\n"
            f"Previous Draft:\n{state['draft']}\n"
            f"Critic Feedback to fix: {feedback}\n\n"
            "Rewrite the article addressing all feedback points."
        )
    else:
        prompt = (
            f"Topic: {state['topic']}\n"
            f"Key Review Insights:\n{state['aggregated_insights']}\n\n"
            "Write a well-structured technical draft based on these insights."
        )

    res = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": res.content, "revisions": revisions+1}

def sub_evaluator_node(state: WriterSubState):
    sys_msg = SystemMessage(
        content="You are a strict technical editor. Grade the draft out of 10. "
                "Respond ONLY in valid JSON format with keys: 'score' (int 1-10) and 'feedback' (str)."
    )
    user_msg = f"Topic: {state['topic']}\nDraft:\n{state['draft']}"

    res = llm.invoke([sys_msg, HumanMessage(content=user_msg)])

    try:
        clean_json = res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        score = int(data.get("score", 6))
        feedback = data.get("feedback", "Improve clarity and Technical depth.")
    except Exception:
        score = 7
        feedback = "Ensure clear structure and deep technical insights."

    return {"score": score, "feedback": feedback}

def route_writer_evaluator(state: WriterSubState):
    if state["score"] >= 8 or state["revisions"] >= 3:
        return "approved"
    return "revise"

sub_builder = StateGraph(WriterSubState)

sub_builder.add_node("sub_writer", sub_writer_node)
sub_builder.add_node("sub_evaluator", sub_evaluator_node)

sub_builder.add_edge(START, "sub_writer")
sub_builder.add_edge("sub_writer", "sub_evaluator")

sub_builder.add_conditional_edges(
    "sub_evaluator",
    route_writer_evaluator,
    {"approved": END, "revise": "sub_writer"}
)

writer_subgraph = sub_builder.compile()


# PARENT-GRAPH (unified workflow)

class GlobalState(TypedDict):
    topic: str
    outline: str
    seo_feedback: str
    tech_feedback: str
    style_feedback: str
    aggregated_insights: str
    approved_by_human: bool
    draft: str
    final_article: str

# parent nodes
def outliner_node(state: GlobalState):
    res = llm.invoke([HumanMessage(content=f"Create a concise 3-point outline for: '{state['topic']}'")])
    return {"outline": res.content}

# parallel working nodes

def seo_node(state: GlobalState):
    res = llm.invoke([HumanMessage(content=f"Provide SEO keywords and search intent analysis for: '{state['topic']}'")])
    return {"seo_feedback": res.content}

def tech_node(state: GlobalState):
    res = llm.invoke([HumanMessage(content=f"Provide deep technical considerations and potential pitfalls for: '{state['topic']}'")])
    return {"tech_feedback": res.content}

def style_node(state: GlobalState):
    res = llm.invoke([HumanMessage(content=f"Suggest target persona and writing hooks for: '{state['topic']}'")])
    return {"style_feedback": res.content}

# aggregator node
def aggregator_node(state: GlobalState):
    prompt = (
        f"Synthesize these expert inputs into a unified blueprint for topic: '{state['topic']}'\n\n"
        f"Outline:\n{state['outline']}\n\n"
        f"SEO Insights:\n{state['seo_feedback']}\n\n"
        f"Tech Insights:\n{state['tech_feedback']}\n\n"
        f"Style Insights:\n{state['style_feedback']}"
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"aggregated_insights": res.content}

builder = StateGraph(GlobalState)

builder.add_node("outliner", outliner_node)
builder.add_node("seo", seo_node)
builder.add_node("tech", tech_node)
builder.add_node("style", style_node)
builder.add_node("aggregator", aggregator_node)

# mount sub-graph
builder.add_node("writer_module", writer_subgraph)

# Wiring
builder.add_edge(START, "outliner")

# Parallel Fan-Out
builder.add_edge("outliner", "seo")
builder.add_edge("outliner", "tech")
builder.add_edge("outliner", "style")

# Parallel Fan-In
builder.add_edge("seo", "aggregator")
builder.add_edge("tech", "aggregator")
builder.add_edge("style", "aggregator")

# Step into Sub-Graph
builder.add_edge("aggregator", "writer_module")
builder.add_edge("writer_module", END)

memory = MemorySaver()
app_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["writer_module"]
)