"""Social media connectors package."""

from src.agent.state import Platform
from src.connectors.base import BaseConnector
from src.connectors.twitter import TwitterConnector
from src.connectors.linkedin import LinkedInConnector


def get_connector(platform: Platform) -> BaseConnector:
    """Get the appropriate connector for a platform.

    Args:
        platform: The target social media platform.

    Returns:
        An instance of the platform-specific connector.

    Raises:
        ValueError: If the platform is not supported.
    """
    connectors = {
        Platform.TWITTER: TwitterConnector,
        Platform.LINKEDIN: LinkedInConnector,
    }

    connector_class = connectors.get(platform)
    if connector_class is None:
        raise ValueError(f"Unsupported platform: {platform}")

    return connector_class()


__all__ = [
    "BaseConnector",
    "TwitterConnector",
    "LinkedInConnector",
    "get_connector",
]
