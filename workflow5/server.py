from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
from workflow import app_graph

app = FastAPI(title="LangGraph Workflow Engine")

# request/response model
class StartRequest(BaseModel):
    thread_id: str
    topic: str

class ResumeRequest(BaseModel):
    thread_id: str
    approve: bool
    human_feedback: Optional[str]=None

class WorkflowStatusResponse(BaseModel):
    thread_id: str
    paused: bool
    next_node: Optional[list[str]]
    current_state: dict[str, Any]

# Endpoints
@app.post("/workflow/start", response_model=WorkflowStatusResponse)
def start_workflow(req: StartRequest):
    """Start a new workflow session and run until Human-in-loop interrupt"""
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_input = {"topic": req.topic}

    # run graph until pause point 
    for _ in app_graph.stream(initial_input, config):
        pass

    snapshot = app_graph.get_state(config)

    return {
        "thread_id": req.thread_id,
        "paused": len(snapshot.next) > 0,
        "next_node": list(snapshot.next),
        "current_state": snapshot.values
    }


@app.get("/workflow/status/{thread_id}", response_model=WorkflowStatusResponse)
def get_status(thread_id: str):
    """Fetches current workflow state and next execution targets for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app_graph.get_state(config)
    
    if not snapshot.created_at:
        raise HTTPException(status_code=404, detail="Session thread not found")
        
    return {
        "thread_id": thread_id,
        "paused": len(snapshot.next) > 0,
        "next_node": list(snapshot.next),
        "current_state": snapshot.values
    }


@app.post("/workflow/resume", response_model=WorkflowStatusResponse)
def resume_workflow(req: ResumeRequest):
    """Approve or edit state, then unpause and complete graph execution."""
    config = {"configurable": {"thread_id": req.thread_id}}
    snapshot = app_graph.get_state(config)
    
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="Workflow is already completed or not paused.")
        
    if not req.approve:
        return {
            "thread_id": req.thread_id,
            "paused": False,
            "next_node": [],
            "current_state": {"status": "Cancelled by user"}
        }

    # If human provided feedback, update state before resuming
    if req.human_feedback:
        app_graph.update_state(
            config,
            {"aggregated_insights": f"{snapshot.values.get('aggregated_insights', '')}\n\n[HUMAN EDIT]: {req.human_feedback}"},
            as_node="aggregator"
        )

    # Resume graph execution
    for _ in app_graph.stream(None, config):
        pass

    final_snapshot = app_graph.get_state(config)

    return {
        "thread_id": req.thread_id,
        "paused": len(final_snapshot.next) > 0,
        "next_node": list(final_snapshot.next),
        "current_state": final_snapshot.values
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)