"""LinkedIn connector for posting to LinkedIn profiles."""

from typing import Any

import httpx

from src.config import get_settings
from src.connectors.base import BaseConnector


class LinkedInConnector(BaseConnector):
    """Connector for LinkedIn API.

    Uses OAuth 2.0 with an access token for posting.
    Requires a valid LinkedIn Access Token with w_member_social scope.
    """

    API_BASE_URL = "https://api.linkedin.com/v2"

    def __init__(self):
        """Initialize the LinkedIn connector."""
        self.settings = get_settings()
        self._user_urn: str | None = None

    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "LinkedIn"

    @property
    def max_length(self) -> int:
        """Return LinkedIn's character limit."""
        return 3000

    @property
    def access_token(self) -> str:
        """Get the LinkedIn access token.

        Raises:
            ValueError: If access token is not configured.
        """
        if not self.settings.linkedin_configured:
            raise ValueError(
                "LinkedIn credentials not configured. "
                "Please set LINKEDIN_ACCESS_TOKEN."
            )
        return self.settings.linkedin_access_token

    @property
    def headers(self) -> dict[str, str]:
        """Get the API request headers."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }

    async def _get_user_urn(self) -> str:
        """Get the authenticated user's URN.

        Returns:
            The user URN in format 'urn:li:person:xxx'.
        """
        if self._user_urn is None:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.API_BASE_URL}/userinfo",
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()
                self._user_urn = f"urn:li:person:{data['sub']}"

        return self._user_urn

    async def publish(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Publish a post to LinkedIn.

        Args:
            content: The post text.
            **kwargs: Additional parameters:
                - visibility: 'PUBLIC' or 'CONNECTIONS' (default: 'PUBLIC')

        Returns:
            Dictionary with success status and post ID.

        Raises:
            ValueError: If content exceeds character limit.
            httpx.HTTPStatusError: If API call fails.
        """
        # Validate content
        is_valid, error = self.validate_content(content)
        if not is_valid:
            raise ValueError(error)

        user_urn = await self._get_user_urn()
        visibility = kwargs.get("visibility", "PUBLIC")

        # Build the share payload
        payload = {
            "author": user_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": content,
                    },
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": visibility,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE_URL}/ugcPosts",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        post_id = data.get("id", "")
        # LinkedIn doesn't return a direct URL, construct it
        post_url = f"https://www.linkedin.com/feed/update/{post_id}"

        return {
            "success": True,
            "id": post_id,
            "url": post_url,
            "platform": "linkedin",
        }

    async def validate_credentials(self) -> bool:
        """Validate LinkedIn credentials.

        Returns:
            True if credentials are valid.
        """
        try:
            await self._get_user_urn()
            return True
        except Exception:
            return False
