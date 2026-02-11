"""LangGraph node implementations for the social media agent."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from src.agent.llm import get_llm
from src.agent.state import AgentState, HumanFeedback, Platform, PostContent, PostStatus


# Platform-specific prompts
PLATFORM_PROMPTS = {
    Platform.TWITTER: """You are an expert social media copywriter for Twitter/X.
Create engaging tweets that:
- Are concise (max 280 characters including hashtags)
- Use attention-grabbing hooks
- Include relevant hashtags (2-4 max)
- Drive engagement through questions or calls-to-action

Character limit is STRICT: 280 characters maximum.""",
    Platform.LINKEDIN: """You are an expert social media copywriter for LinkedIn.
Create professional posts that:
- Are informative and add value
- Use a professional yet personable tone
- Include relevant hashtags (3-5)
- Encourage professional discussion
- Can be up to 3000 characters but aim for 150-300 words for best engagement""",
}


POST_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{platform_prompt}"),
        (
            "human",
            """Create a {platform} post about the following topic:

Topic: {topic}
Desired Tone: {tone}
{additional_context}

Respond with ONLY the post text. Do not include hashtags in the main text - I will ask for those separately.""",
        ),
    ]
)

HASHTAG_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Given this {platform} post:

"{post_text}"

Suggest {num_hashtags} relevant hashtags. Return ONLY the hashtags, one per line, without the # symbol.""",
        ),
    ]
)


async def generate_post(state: AgentState) -> dict:
    """Generate a social media post using the LLM.

    This node uses the LLM to create post content based on the user's request.
    """
    llm = get_llm()

    # Get platform-specific prompt
    platform_prompt = PLATFORM_PROMPTS.get(
        state.platform, PLATFORM_PROMPTS[Platform.TWITTER]
    )

    # Build additional context string
    additional_context = ""
    if state.additional_context:
        additional_context = f"Additional requirements: {state.additional_context}"

    # Generate post text
    chain = POST_GENERATION_TEMPLATE | llm
    response = await chain.ainvoke(
        {
            "platform_prompt": platform_prompt,
            "platform": state.platform.value,
            "topic": state.topic,
            "tone": state.tone,
            "additional_context": additional_context,
        }
    )

    post_text = response.content.strip()

    # Generate hashtags
    num_hashtags = 3 if state.platform == Platform.TWITTER else 5
    hashtag_chain = HASHTAG_TEMPLATE | llm
    hashtag_response = await hashtag_chain.ainvoke(
        {
            "platform": state.platform.value,
            "post_text": post_text,
            "num_hashtags": num_hashtags,
        }
    )

    hashtags = [
        tag.strip().lstrip("#")
        for tag in hashtag_response.content.strip().split("\n")
        if tag.strip()
    ]

    # Create post content
    post_content = PostContent(
        text=post_text,
        platform=state.platform,
        hashtags=hashtags[:num_hashtags],
    )

    return {
        "post_content": post_content,
        "status": PostStatus.PENDING_APPROVAL,
        "generation_attempts": state.generation_attempts + 1,
        "messages": [
            AIMessage(
                content=f"Generated {state.platform.value} post:\n\n{post_content.formatted_text}"
            )
        ],
    }


async def request_approval(state: AgentState) -> dict:
    """Pause execution and wait for human approval.

    This node uses LangGraph's interrupt() to pause the workflow
    and wait for human feedback.
    """
    if state.post_content is None:
        return {"error_message": "No post content to approve"}

    # Create the approval request payload
    approval_request = {
        "type": "approval_request",
        "post_text": state.post_content.formatted_text,
        "platform": state.platform.value,
        "char_count": state.post_content.char_count,
        "generation_attempt": state.generation_attempts,
        "instructions": (
            "Please review this post and respond with one of:\n"
            "- 'approve' to publish as-is\n"
            "- 'reject' to regenerate\n"
            "- 'edit: <your edited text>' to publish with modifications"
        ),
    }

    # This will pause execution and wait for human input
    # The human's response will be passed back via Command(resume=feedback)
    human_response = interrupt(approval_request)

    # Parse the human response
    feedback = _parse_human_response(human_response)

    return {
        "human_feedback": feedback,
        "messages": [
            HumanMessage(content=f"Human feedback: {feedback.action}")
        ],
    }


def _parse_human_response(response: str | dict) -> HumanFeedback:
    """Parse human response into HumanFeedback object."""
    if isinstance(response, dict):
        return HumanFeedback(**response)

    response_str = str(response).strip().lower()

    if response_str == "approve":
        return HumanFeedback(action="approve")
    elif response_str == "reject":
        return HumanFeedback(action="reject")
    elif response_str.startswith("edit:"):
        edited_text = response_str[5:].strip()
        return HumanFeedback(action="edit", edited_text=edited_text)
    else:
        # Default to approve if unclear
        return HumanFeedback(
            action="approve", feedback_message=f"Unclear response: {response}"
        )


async def process_feedback(state: AgentState) -> dict:
    """Process human feedback and update state accordingly."""
    if state.human_feedback is None:
        return {"status": PostStatus.PENDING_APPROVAL}

    feedback = state.human_feedback

    if feedback.action == "approve":
        return {"status": PostStatus.APPROVED}

    elif feedback.action == "reject":
        if state.generation_attempts >= state.max_attempts:
            return {
                "status": PostStatus.FAILED,
                "error_message": f"Max generation attempts ({state.max_attempts}) reached",
            }
        return {
            "status": PostStatus.DRAFT,
            "human_feedback": None,  # Clear feedback for regeneration
        }

    elif feedback.action == "edit":
        # Update post content with edited text
        if state.post_content and feedback.edited_text:
            updated_content = PostContent(
                text=feedback.edited_text,
                platform=state.platform,
                hashtags=state.post_content.hashtags,
            )
            return {
                "post_content": updated_content,
                "status": PostStatus.APPROVED,
            }
        return {"status": PostStatus.APPROVED}

    return {"status": PostStatus.PENDING_APPROVAL}


async def publish_post(state: AgentState) -> dict:
    """Publish the approved post to the target platform."""
    from src.connectors import get_connector

    if state.post_content is None:
        return {
            "status": PostStatus.FAILED,
            "error_message": "No post content to publish",
        }

    try:
        connector = get_connector(state.platform)
        result = await connector.publish(state.post_content.formatted_text)

        return {
            "status": PostStatus.PUBLISHED,
            "published_url": result.get("url"),
            "messages": [
                AIMessage(
                    content=f"✅ Successfully published to {state.platform.value}!\nURL: {result.get('url', 'N/A')}"
                )
            ],
        }
    except Exception as e:
        return {
            "status": PostStatus.FAILED,
            "error_message": str(e),
            "messages": [
                AIMessage(content=f"❌ Failed to publish: {str(e)}")
            ],
        }


def should_regenerate(state: AgentState) -> str:
    """Determine the next step after processing feedback.

    Returns the name of the next node to execute.
    """
    if state.status == PostStatus.APPROVED:
        return "publish"
    elif state.status == PostStatus.DRAFT:
        return "generate"
    else:
        return "end"
