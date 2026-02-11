"""LangGraph state definitions for the social media agent."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PostStatus(str, Enum):
    """Status of a social media post in the workflow."""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    FAILED = "failed"


class Platform(str, Enum):
    """Supported social media platforms."""

    TWITTER = "twitter"
    LINKEDIN = "linkedin"


class PostContent(BaseModel):
    """Generated post content."""

    text: str = Field(..., description="The post text content")
    platform: Platform = Field(..., description="Target platform")
    hashtags: list[str] = Field(default_factory=list, description="Hashtags for the post")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def formatted_text(self) -> str:
        """Get the post text with hashtags appended."""
        if self.hashtags:
            hashtag_str = " ".join(f"#{tag}" for tag in self.hashtags)
            return f"{self.text}\n\n{hashtag_str}"
        return self.text

    @property
    def char_count(self) -> int:
        """Get character count for the formatted text."""
        return len(self.formatted_text)


class HumanFeedback(BaseModel):
    """Human feedback on a generated post."""

    action: Literal["approve", "reject", "edit"] = Field(
        ..., description="The action taken by the human"
    )
    edited_text: str | None = Field(
        None, description="Edited text if action is 'edit'"
    )
    feedback_message: str | None = Field(
        None, description="Optional feedback message"
    )


class AgentState(BaseModel):
    """State for the LangGraph social media agent workflow.

    This state is passed between nodes and persisted for human-in-the-loop.
    """

    # Conversation messages (LangGraph message handling)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # User request
    topic: str = Field(default="", description="Topic or theme for the post")
    platform: Platform = Field(
        default=Platform.TWITTER, description="Target social media platform"
    )
    tone: str = Field(
        default="professional", description="Desired tone (professional, casual, humorous, etc.)"
    )
    additional_context: str = Field(
        default="", description="Any additional context or requirements"
    )

    # Generated content
    post_content: PostContent | None = Field(
        None, description="The generated post content"
    )

    # Workflow state
    status: PostStatus = Field(
        default=PostStatus.DRAFT, description="Current status in the workflow"
    )
    human_feedback: HumanFeedback | None = Field(
        None, description="Feedback from human review"
    )
    generation_attempts: int = Field(
        default=0, description="Number of generation attempts"
    )
    max_attempts: int = Field(
        default=3, description="Maximum generation attempts before giving up"
    )

    # Result
    published_url: str | None = Field(
        None, description="URL of the published post"
    )
    error_message: str | None = Field(
        None, description="Error message if publishing failed"
    )

    # Thread tracking
    thread_id: str = Field(
        default="", description="Unique thread ID for persistence"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
