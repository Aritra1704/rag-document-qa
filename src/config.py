"""Social Media AI Agent - Configuration settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o"

    # Twitter/X API
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_secret: str = ""
    twitter_bearer_token: str = ""

    # LinkedIn API
    linkedin_client_id: str = ""
    linkedin_client_secret: str = ""
    linkedin_access_token: str = ""

    # Database
    database_url: str = "sqlite:///./data/agent.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @property
    def twitter_configured(self) -> bool:
        """Check if Twitter credentials are configured."""
        return bool(
            self.twitter_api_key
            and self.twitter_api_secret
            and self.twitter_access_token
            and self.twitter_access_secret
        )

    @property
    def linkedin_configured(self) -> bool:
        """Check if LinkedIn credentials are configured."""
        return bool(self.linkedin_access_token)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
