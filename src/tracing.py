"""Langfuse observability — callback-based tracing for LangChain/LangGraph.

All LLM/agent calls go through LangChain (chat model + agent), so tracing is the
Langfuse LangChain CallbackHandler attached to every .invoke(). setup_tracing()
initializes the Langfuse client; langchain_callbacks() supplies the handler.

Docs: https://langfuse.com/integrations/frameworks/langchain
"""

import os
import logging

from src.config import get_settings

log = logging.getLogger(__name__)

_initialized = False


def setup_tracing() -> bool:
    """Initialize the Langfuse client. Idempotent. Safe no-op when keys are missing
    or the package isn't installed, so the app always runs."""
    global _initialized
    if _initialized:
        return True

    s = get_settings()
    if not s.TRACING_ENABLED or not (s.LANGFUSE_PUBLIC_KEY and s.LANGFUSE_SECRET_KEY):
        log.info("Langfuse tracing disabled (TRACING_ENABLED off or keys missing).")
        return False

    # Langfuse reads credentials from os.environ; mirror settings there so one .env
    # drives both. setdefault lets an explicit container/shell env win.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", s.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_BASE_URL", s.LANGFUSE_BASE_URL)
    os.environ.setdefault("LANGFUSE_HOST", s.LANGFUSE_BASE_URL)

    try:
        from langfuse import get_client

        client = get_client()
        if not client.auth_check():
            log.warning("Langfuse auth check failed — verify keys/base URL.")
    except Exception as exc:  # missing package, network, etc. — never block the app
        log.warning("Langfuse tracing setup failed: %s", exc)
        return False

    _initialized = True
    log.info("Langfuse tracing enabled (%s).", s.LANGFUSE_BASE_URL)
    return True


def langchain_callbacks() -> list:
    """Langfuse callback handler(s) for LangChain/LangGraph .invoke(config={"callbacks": ...}).
    Returns [] when tracing is inactive."""
    if not _initialized:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as exc:
        log.warning("Langfuse LangChain callback unavailable: %s", exc)
        return []


def flush() -> None:
    """Force-export buffered traces. Call before a short-lived process exits."""
    if not _initialized:
        return
    from langfuse import get_client

    get_client().flush()
