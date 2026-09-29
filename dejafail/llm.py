"""Groq JSON client (placeholder until Task 3)."""


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMOutputError(LLMError):
    """The model never produced valid JSON for the schema."""


class LLMUnavailable(LLMError):
    """Groq could not be reached or kept failing."""
