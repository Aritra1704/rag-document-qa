"""Twitter/X connector using Tweepy."""

import asyncio
from typing import Any

import tweepy

from src.config import get_settings
from src.connectors.base import BaseConnector


class TwitterConnector(BaseConnector):
    """Connector for Twitter/X API using Tweepy.

    Uses OAuth 1.0a User Context for posting tweets.
    Requires API Key, API Secret, Access Token, and Access Secret.
    """

    def __init__(self):
        """Initialize the Twitter connector with credentials from settings."""
        self.settings = get_settings()
        self._client: tweepy.Client | None = None

    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "Twitter/X"

    @property
    def max_length(self) -> int:
        """Return Twitter's character limit."""
        return 280

    @property
    def client(self) -> tweepy.Client:
        """Get or create the Tweepy client.

        Returns:
            Configured Tweepy Client instance.

        Raises:
            ValueError: If Twitter credentials are not configured.
        """
        if self._client is None:
            if not self.settings.twitter_configured:
                raise ValueError(
                    "Twitter credentials not configured. "
                    "Please set TWITTER_API_KEY, TWITTER_API_SECRET, "
                    "TWITTER_ACCESS_TOKEN, and TWITTER_ACCESS_SECRET."
                )

            self._client = tweepy.Client(
                consumer_key=self.settings.twitter_api_key,
                consumer_secret=self.settings.twitter_api_secret,
                access_token=self.settings.twitter_access_token,
                access_token_secret=self.settings.twitter_access_secret,
            )

        return self._client

    async def publish(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Publish a tweet.

        Args:
            content: The tweet text.
            **kwargs: Additional parameters:
                - reply_to: Tweet ID to reply to
                - media_ids: List of media IDs to attach

        Returns:
            Dictionary with success status, tweet URL, and tweet ID.

        Raises:
            ValueError: If content exceeds character limit.
            tweepy.TweepyException: If API call fails.
        """
        # Validate content
        is_valid, error = self.validate_content(content)
        if not is_valid:
            raise ValueError(error)

        # Prepare tweet parameters
        tweet_params: dict[str, Any] = {"text": content}

        if reply_to := kwargs.get("reply_to"):
            tweet_params["in_reply_to_tweet_id"] = reply_to

        if media_ids := kwargs.get("media_ids"):
            tweet_params["media_ids"] = media_ids

        # Run in thread pool since Tweepy is synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.create_tweet(**tweet_params),
        )

        tweet_id = response.data["id"]
        tweet_url = f"https://twitter.com/user/status/{tweet_id}"

        return {
            "success": True,
            "id": tweet_id,
            "url": tweet_url,
            "platform": "twitter",
        }

    async def validate_credentials(self) -> bool:
        """Validate Twitter credentials by fetching the authenticated user.

        Returns:
            True if credentials are valid.
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_me(),
            )
            return response.data is not None
        except Exception:
            return False
