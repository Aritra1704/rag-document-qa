"""LangGraph workflow definition for the social media agent."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    generate_post,
    process_feedback,
    publish_post,
    request_approval,
    should_regenerate,
)
from src.agent.state import AgentState, PostStatus


def create_graph() -> StateGraph:
    """Create the LangGraph workflow for social media post generation.

    The workflow follows this pattern:
    1. generate_post: Use LLM to create post content
    2. request_approval: Pause for human review (interrupt)
    3. process_feedback: Handle approve/reject/edit
    4. publish_post: Post to social media (if approved)

    The graph supports regeneration loops when posts are rejected.
    """
    # Create the graph with our state schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("generate", generate_post)
    graph.add_node("request_approval", request_approval)
    graph.add_node("process_feedback", process_feedback)
    graph.add_node("publish", publish_post)

    # Define the workflow edges
    graph.set_entry_point("generate")

    # After generation, always request approval
    graph.add_edge("generate", "request_approval")

    # After approval request, process the feedback
    graph.add_edge("request_approval", "process_feedback")

    # After processing feedback, decide next step
    graph.add_conditional_edges(
        "process_feedback",
        should_regenerate,
        {
            "publish": "publish",
            "generate": "generate",
            "end": END,
        },
    )

    # After publishing, we're done
    graph.add_edge("publish", END)

    return graph


def compile_graph(checkpointer: MemorySaver | None = None):
    """Compile the graph with optional checkpointing.

    Args:
        checkpointer: Optional checkpointer for state persistence.
                     Required for human-in-the-loop to work across restarts.

    Returns:
        Compiled graph ready for execution.
    """
    graph = create_graph()

    if checkpointer is None:
        # Use in-memory checkpointer by default
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# Pre-compiled graph instance for convenience
def get_compiled_graph():
    """Get a compiled graph with memory checkpointing.

    This is the main entry point for running the agent.
    """
    return compile_graph()
