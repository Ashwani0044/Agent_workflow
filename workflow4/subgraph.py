import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Common LLM setup
llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)
search_tool = DuckDuckGoSearchRun()

#  DEFINE SUB-GRAPH (Research Sub-Pipeline)

# Sub-Graph State: Can have its own isolated fields
class ResearchState(TypedDict):
    search_topic: str
    raw_search_results: str
    curated_facts: str

def search_executor_node(state: ResearchState):
    query = state["search_topic"]
    # Perform web search directly
    results = search_tool.run(query)
    return {"raw_search_results": results}

def fact_curator_node(state: ResearchState):
    prompt = (
        f"Topic: {state['search_topic']}\n"
        f"Raw Search Results:\n{state['raw_search_results']}\n\n"
        "Extract 3-4 bullet points of the most critical facts from this search output."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"curated_facts": response.content}

# Build Sub-Graph
sub_builder = StateGraph(ResearchState)
sub_builder.add_node("search_executor", search_executor_node)
sub_builder.add_node("fact_curator", fact_curator_node)

sub_builder.add_edge(START, "search_executor")
sub_builder.add_edge("search_executor", "fact_curator")
sub_builder.add_edge("fact_curator", END)

# Compile Sub-Graph into an executable object
research_subgraph = sub_builder.compile()



# Parent State
class ParentState(TypedDict):
    topic: str
    outline: str
    search_topic: str      # Passed to/from sub-graph
    curated_facts: str     # Populated by sub-graph
    final_article: str

def outliner_node(state: ParentState):
    prompt = f"Create a concise 3-point outline for an article about: '{state['topic']}'"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # We set 'search_topic' which will feed into the research sub-graph
    return {
        "outline": response.content,
        "search_topic": f"Latest tech developments in {state['topic']}"
    }

def editor_node(state: ParentState):
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Outline:\n{state['outline']}\n\n"
        f"Curated Research Facts:\n{state['curated_facts']}\n\n"
        "Write the final short article synthesizing the outline and research facts."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_article": response.content}

# Build Parent Graph
parent_builder = StateGraph(ParentState)

parent_builder.add_node("outliner", outliner_node)

# MOUNT SUB-GRAPH AS A NODE IN PARENT GRAPH!
# LangGraph automatically maps matching keys between ParentState and ResearchState
parent_builder.add_node("research_module", research_subgraph)

parent_builder.add_node("editor", editor_node)

# Connect Parent Edges
parent_builder.add_edge(START, "outliner")
parent_builder.add_edge("outliner", "research_module") # Steps into sub-graph
parent_builder.add_edge("research_module", "editor")   # Resumes parent flow after sub-graph completes
parent_builder.add_edge("editor", END)

parent_graph = parent_builder.compile()


if __name__ == "__main__":
    inputs = {
        "topic": "FastAPI v0.115 and Python Asynchronous Web Services",
        "outline": "",
        "search_topic": "",
        "curated_facts": "",
        "final_article": ""
    }
    
    print("=== EXECUTING SUB-GRAPH WORKFLOW ===")
    for event in parent_graph.stream(inputs):
        for node_name, value in event.items():
            print(f"\n Executed Node: [{node_name}]")
            if node_name == "outliner":
                print(f"-> Generated Search Topic: {value.get('search_topic')}")
            elif node_name == "research_module":
                print(f"-> Sub-Graph Finished! Curated Facts Preview:\n{value.get('curated_facts')[:200]}...")
            elif node_name == "editor":
                print("\n=== FINAL ARTICLE ===")
                print(value.get("final_article"))