"""Base connector interface for social media platforms."""

from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    """Abstract base class for social media platform connectors.

    All platform-specific connectors should inherit from this class
    and implement the required methods.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform."""
        pass

    @property
    @abstractmethod
    def max_length(self) -> int:
        """Return the maximum post length for this platform."""
        pass

    @abstractmethod
    async def publish(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Publish content to the platform.

        Args:
            content: The text content to publish.
            **kwargs: Additional platform-specific parameters.

        Returns:
            A dictionary containing at least:
                - 'success': bool indicating if the post was successful
                - 'url': URL of the published post (if available)
                - 'id': Platform-specific post ID

        Raises:
            ConnectionError: If unable to connect to the platform.
            AuthenticationError: If authentication fails.
            ValueError: If content exceeds platform limits.
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate that the configured credentials are working.

        Returns:
            True if credentials are valid, False otherwise.
        """
        pass

    def validate_content(self, content: str) -> tuple[bool, str]:
        """Validate content before publishing.

        Args:
            content: The content to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if len(content) > self.max_length:
            return False, f"Content exceeds {self.platform_name} limit of {self.max_length} characters"
        if not content.strip():
            return False, "Content cannot be empty"
        return True, ""
