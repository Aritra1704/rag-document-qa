"""MCP Server exposing social media agent tools."""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from src.agent.graph import get_compiled_graph
from src.agent.state import AgentState, Platform

# Create MCP server instance
server = Server("social-media-agent")

# Store for pending threads
pending_threads: dict[str, dict[str, Any]] = {}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the MCP client."""
    return [
        Tool(
            name="generate_social_post",
            description="Generate a social media post for Twitter or LinkedIn using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or theme for the post",
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["twitter", "linkedin"],
                        "description": "Target social media platform",
                        "default": "twitter",
                    },
                    "tone": {
                        "type": "string",
                        "description": "Desired tone (professional, casual, humorous, etc.)",
                        "default": "professional",
                    },
                    "additional_context": {
                        "type": "string",
                        "description": "Any additional requirements or context",
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="list_pending_posts",
            description="List all posts awaiting human approval",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="approve_post",
            description="Approve a pending post for publishing",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "The thread ID of the post to approve",
                    },
                },
                "required": ["thread_id"],
            },
        ),
        Tool(
            name="reject_post",
            description="Reject a pending post and request regeneration",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "The thread ID of the post to reject",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Optional feedback for regeneration",
                    },
                },
                "required": ["thread_id"],
            },
        ),
        Tool(
            name="edit_and_approve_post",
            description="Edit a pending post and approve it for publishing",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "The thread ID of the post to edit",
                    },
                    "edited_text": {
                        "type": "string",
                        "description": "The edited post text",
                    },
                },
                "required": ["thread_id", "edited_text"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool calls from MCP clients."""

    if name == "generate_social_post":
        return await handle_generate_post(arguments)
    elif name == "list_pending_posts":
        return await handle_list_pending()
    elif name == "approve_post":
        return await handle_approve(arguments, "approve")
    elif name == "reject_post":
        return await handle_approve(arguments, "reject")
    elif name == "edit_and_approve_post":
        return await handle_approve(arguments, "edit")
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")]
        )


async def handle_generate_post(arguments: dict[str, Any]) -> CallToolResult:
    """Generate a new social media post."""
    import uuid

    topic = arguments.get("topic", "")
    platform_str = arguments.get("platform", "twitter")
    tone = arguments.get("tone", "professional")
    context = arguments.get("additional_context", "")

    platform = Platform.TWITTER if platform_str == "twitter" else Platform.LINKEDIN

    thread_id = str(uuid.uuid4())
    graph = get_compiled_graph()

    initial_state = AgentState(
        topic=topic,
        platform=platform,
        tone=tone,
        additional_context=context,
        thread_id=thread_id,
    )

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(initial_state.model_dump(), config)

        pending_threads[thread_id] = {
            "topic": topic,
            "platform": platform_str,
            "result": result,
        }

        post_content = result.get("post_content")
        post_text = post_content.formatted_text if post_content else "Failed to generate"

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "thread_id": thread_id,
                            "status": "pending_approval",
                            "platform": platform_str,
                            "post_text": post_text,
                            "message": "Post generated! Please approve, reject, or edit.",
                        },
                        indent=2,
                    ),
                )
            ]
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error generating post: {str(e)}")]
        )


async def handle_list_pending() -> CallToolResult:
    """List all pending posts."""
    if not pending_threads:
        return CallToolResult(
            content=[TextContent(type="text", text="No posts pending approval.")]
        )

    pending_list = []
    for thread_id, data in pending_threads.items():
        result = data.get("result", {})
        post_content = result.get("post_content")

        if post_content:
            pending_list.append(
                {
                    "thread_id": thread_id,
                    "topic": data.get("topic"),
                    "platform": data.get("platform"),
                    "post_text": post_content.formatted_text
                    if hasattr(post_content, "formatted_text")
                    else str(post_content),
                }
            )

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(pending_list, indent=2),
            )
        ]
    )


async def handle_approve(arguments: dict[str, Any], action: str) -> CallToolResult:
    """Handle approve/reject/edit actions."""
    from langgraph.types import Command

    thread_id = arguments.get("thread_id", "")

    if thread_id not in pending_threads:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Thread not found: {thread_id}")]
        )

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    feedback = {"action": action}
    if action == "edit":
        feedback["edited_text"] = arguments.get("edited_text", "")
    if action == "reject":
        feedback["feedback_message"] = arguments.get("feedback", "")

    try:
        result = await graph.ainvoke(Command(resume=feedback), config)

        status = result.get("status", "unknown")
        published_url = result.get("published_url")

        # Clean up if completed
        if status in ["published", "failed"]:
            pending_threads.pop(thread_id, None)
        else:
            pending_threads[thread_id]["result"] = result

        response = {
            "thread_id": thread_id,
            "status": status,
            "action_taken": action,
        }

        if published_url:
            response["published_url"] = published_url

        if result.get("error_message"):
            response["error"] = result.get("error_message")

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response, indent=2))]
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
