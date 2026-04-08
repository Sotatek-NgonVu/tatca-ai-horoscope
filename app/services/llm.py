"""
Claude LLM service.

Wraps the Anthropic SDK to implement LLMService.
Supports adaptive thinking for deeper reasoning on complex prompts.
"""

from __future__ import annotations

import logging

import anthropic

from app.core.exceptions import LLMAuthenticationError, LLMError, LLMRateLimitError
from app.core.interfaces import LLMService

logger = logging.getLogger(__name__)


class ClaudeLLMService(LLMService):
    """LLM service powered by Anthropic Claude."""

    def __init__(self, api_key: str, model: str) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate a response using Claude with adaptive thinking.

        Raises:
            LLMAuthenticationError: If the API key is invalid.
            LLMRateLimitError: If the rate limit is exceeded.
            LLMError: For any other API failure.
        """
        logger.info(
            "Calling Claude (%s) — max_tokens=%d, message_len=%d",
            self._model,
            max_tokens,
            len(user_message),
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Collect only text blocks (skip thinking blocks).
            answer = "".join(
                block.text for block in response.content if block.type == "text"
            )

            logger.info("Claude response: %d characters", len(answer))
            return answer.strip()

        except anthropic.AuthenticationError as exc:
            raise LLMAuthenticationError(
                "Invalid Anthropic API key", cause=exc
            ) from exc

        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(
                "Anthropic API rate limit exceeded", cause=exc
            ) from exc

        except Exception as exc:
            raise LLMError(
                f"Claude API call failed: {exc}", cause=exc
            ) from exc
