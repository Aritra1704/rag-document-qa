"""FastAPI backend for the social media agent."""

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agent.graph import compile_graph
from src.agent.state import AgentState, Platform, PostStatus


# In-memory storage for pending posts and checkpointer
checkpointer = MemorySaver()
compiled_graph = compile_graph(checkpointer)
pending_threads: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    yield
    # Shutdown
    pending_threads.clear()


app = FastAPI(
    title="Social Media AI Agent",
    description="AI-powered social media post generation with human-in-the-loop approval",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class GeneratePostRequest(BaseModel):
    """Request to generate a new social media post."""

    topic: str = Field(..., description="Topic or theme for the post")
    platform: Platform = Field(default=Platform.TWITTER, description="Target platform")
    tone: str = Field(default="professional", description="Desired tone")
    additional_context: str = Field(default="", description="Additional requirements")


class ApprovalRequest(BaseModel):
    """Request to approve, reject, or edit a post."""

    action: str = Field(..., pattern="^(approve|reject|edit)$")
    edited_text: str | None = Field(None, description="Edited text if action is 'edit'")
    feedback_message: str | None = Field(None, description="Optional feedback")


class PostResponse(BaseModel):
    """Response containing post information."""

    thread_id: str
    status: str
    post_text: str | None = None
    platform: str
    char_count: int | None = None
    published_url: str | None = None
    error_message: str | None = None


class PendingPost(BaseModel):
    """A post awaiting approval."""

    thread_id: str
    topic: str
    platform: str
    post_text: str
    char_count: int
    generation_attempt: int


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy", "service": "social-media-agent"}


@app.post("/posts/generate", response_model=PostResponse)
async def generate_post(request: GeneratePostRequest):
    """Start a new post generation workflow.

    This will:
    1. Generate a post using the LLM
    2. Pause at the approval step
    3. Return the thread_id for tracking
    """
    thread_id = str(uuid.uuid4())

    # Create initial state
    initial_state = AgentState(
        topic=request.topic,
        platform=request.platform,
        tone=request.tone,
        additional_context=request.additional_context,
        thread_id=thread_id,
    )

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run until we hit the interrupt (approval request)
        result = await compiled_graph.ainvoke(initial_state.model_dump(), config)

        # Store in pending threads
        pending_threads[thread_id] = {
            "topic": request.topic,
            "platform": request.platform.value,
            "result": result,
        }

        return PostResponse(
            thread_id=thread_id,
            status=result.get("status", PostStatus.PENDING_APPROVAL.value),
            post_text=result.get("post_content", {}).get("formatted_text")
            if result.get("post_content")
            else None,
            platform=request.platform.value,
            char_count=result.get("post_content", {}).get("char_count")
            if result.get("post_content")
            else None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/posts/pending", response_model=list[PendingPost])
async def list_pending_posts():
    """List all posts awaiting approval."""
    pending = []

    for thread_id, data in pending_threads.items():
        result = data.get("result", {})
        post_content = result.get("post_content")

        if post_content and result.get("status") == PostStatus.PENDING_APPROVAL.value:
            pending.append(
                PendingPost(
                    thread_id=thread_id,
                    topic=data.get("topic", ""),
                    platform=data.get("platform", "twitter"),
                    post_text=post_content.get("formatted_text", "")
                    if isinstance(post_content, dict)
                    else post_content.formatted_text,
                    char_count=post_content.get("char_count", 0)
                    if isinstance(post_content, dict)
                    else post_content.char_count,
                    generation_attempt=result.get("generation_attempts", 1),
                )
            )

    return pending


@app.post("/posts/{thread_id}/approve", response_model=PostResponse)
async def approve_post(thread_id: str, request: ApprovalRequest):
    """Approve, reject, or edit a pending post.

    This will resume the workflow with the human's decision.
    """
    if thread_id not in pending_threads:
        raise HTTPException(status_code=404, detail="Thread not found")

    config = {"configurable": {"thread_id": thread_id}}

    # Create feedback payload
    feedback = {
        "action": request.action,
        "edited_text": request.edited_text,
        "feedback_message": request.feedback_message,
    }

    try:
        # Resume the graph with human feedback using Command
        result = await compiled_graph.ainvoke(
            Command(resume=feedback),
            config,
        )

        # Update or remove from pending
        if result.get("status") in [
            PostStatus.PUBLISHED.value,
            PostStatus.FAILED.value,
        ]:
            pending_threads.pop(thread_id, None)
        else:
            pending_threads[thread_id]["result"] = result

        post_content = result.get("post_content")

        return PostResponse(
            thread_id=thread_id,
            status=result.get("status", "unknown"),
            post_text=post_content.formatted_text
            if post_content and hasattr(post_content, "formatted_text")
            else None,
            platform=pending_threads.get(thread_id, {}).get("platform", "twitter"),
            published_url=result.get("published_url"),
            error_message=result.get("error_message"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/posts/{thread_id}", response_model=PostResponse)
async def get_post_status(thread_id: str):
    """Get the current status of a post workflow."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await compiled_graph.aget_state(config)

        if state.values is None:
            raise HTTPException(status_code=404, detail="Thread not found")

        values = state.values
        post_content = values.get("post_content")

        return PostResponse(
            thread_id=thread_id,
            status=values.get("status", "unknown"),
            post_text=post_content.formatted_text
            if post_content and hasattr(post_content, "formatted_text")
            else None,
            platform=values.get("platform", Platform.TWITTER).value,
            published_url=values.get("published_url"),
            error_message=values.get("error_message"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    from src.config import get_settings

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
