# evaluator optimizer (self-correction loop)

import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph


load_dotenv()

class State(TypedDict):
    topic: str
    draft: str
    feedback: str
    quality_score: int
    revision_count: int

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

def writer_node(state: State):
    current_revisions = state.get("revision_count", 0)
    feedback = state.get("feedback", 0)

    if feedback:
        prompt = (
            f"Topic: {state['topic']}\n"
            f"Previous Draft: {state['draft']}\n"
            f"Critic Feedback to fix: {feedback}\n\n"
            "Rewrite the draft addressing all critic feedback points."
        )
    else:
        prompt = f"Write a comprehensive explanation on: '{state['topic']}'."


    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft": response.count,
        "revision_count": current_revisions+1
    }

def evaluator_node(state: State):
    system_prompt = SystemMessage(
        content="You are a strict technical editor. Grade the draft out of 10 for clarity, technical depth, and structure. "
                "Respond ONLY in valid JSON format with keys: 'score' (int 1-10) and 'feedback' (str)."
    )
    
    user_prompt = f"Topic: {state['topic']}\n\nDraft:\n{state['draft']}"
    
    response = llm.invoke([system_prompt, HumanMessage(content=user_prompt)])

    # parse llm json output safely
    try:
        # Clean potential markdown code fences from response
        cleaned_content = response.content.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(cleaned_content)
        score = int(evaluation.get("score", 5))
        feedback = evaluation.get("feedback", "Improve clarity and technical depth.")
    except Exception:
        score = 6
        feedback = "Ensure strong structure, clear examples, and technical accuracy."

    return {
        "quality_score": score,
        "feedback": feedback
    }

# defining conditional edge router
MAX_REVISIONS = 3
TARGET_SCORE = 8

def route_evaluation(state: State):
    score = state["quality_score"]
    revisions = state["revision_count"]
    
    print(f"\n[ Evaluation Gate ] Score: {score}/10 | Revision: {revisions}/{MAX_REVISIONS}")
    print(f"Feedback: {state['feedback']}")
    
    if score >= TARGET_SCORE or revisions >= MAX_REVISIONS:
        return "approved"
    return "revise"

builder = StateGraph(State)

builder.add_node("writer", writer_node)
builder.add_node("evaluator", evaluator_node)

builder.add_edge(START, "writer")
builder.add_edge("writer", "evaluator")

# Conditional Routing based on Evaluation Output
builder.add_conditional_edges(
    "evaluator",
    route_evaluation,
    {
        "approved": END,
        "revise": "writer"  # Self-correction loop!
    }
)

graph = builder.compile()

if __name__ == "__main__":
    inputs = {
        "topic": "Explain how Garbage Collection works in Python (Reference Counting vs Generational GC)",
        "draft": "",
        "feedback": "",
        "quality_score": 0,
        "revision_count": 0
    }
    
    print("=== EXECUTING EVALUATOR-OPTIMIZER WORKFLOW ===")
    for event in graph.stream(inputs):
        for node_name, value in event.items():
            print(f" Completed Node: [{node_name}]")
            
    final_state = graph.get_state({"configurable": {"thread_id": "eval_1"}} if False else {})
    print("\n=== WORKFLOW FINISHED ===")