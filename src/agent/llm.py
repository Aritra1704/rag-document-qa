"""LLM initialization for the social media agent."""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from src.config import get_settings


@lru_cache
def get_llm() -> BaseChatModel:
    """Get the configured LLM instance.

    Returns either OpenAI or Anthropic based on settings.
    """
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.7,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.7,
        )
