import asyncio
import logging
import json

import backoff
from fastmcp.client import Client

from evaluation_utils.commons import (
    logger,
    log_handler,
    MCP_MAX_RETRY_TRIES,
    MCP_MAX_RETRY_TIME,
    MCP_BACKOFF_BASE,
    MCP_BACKOFF_MAX_INTERVAL,
    MCP_CALL_TIMEOUT_SECONDS,
)

try:
    from aiohttp import ClientError as AioHttpClientError
except ImportError:  # pragma: no cover - aiohttp ships with fastmcp but guard just in case
    class AioHttpClientError(Exception):
        """Fallback when aiohttp is unavailable."""
        pass


RETRYABLE_MCP_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
    AioHttpClientError,
)


def _describe_operation(details: dict) -> str:
    """Generate a human-readable label for the retried operation."""
    func_name = getattr(details.get("target"), "__name__", "unknown")
    if func_name == "_call_tool":
        args = details.get("args", [])
        if len(args) > 1:
            return f"{func_name}:{args[1]}"
    return func_name


def _log_retry_event(details: dict):
    label = _describe_operation(details)
    tries = details.get("tries", 0)
    exc = details.get("exception")
    # Best-effort extraction of the MCP server URL from the backoff context.
    # For bound methods, args[0] is expected to be the GameEnv instance.
    args = details.get("args") or ()
    mcp_url = None
    if args:
        self_obj = args[0]
        # Prefer an explicit attribute on the GameEnv instance if present.
        mcp_url = getattr(self_obj, "mcp_url", None)
        if mcp_url is None:
            # Fallback to common URL-style attributes on the underlying client.
            client = getattr(self_obj, "client", None)
            if client is not None:
                for attr in ("mcp_url", "url", "base_url", "server_url"):
                    mcp_url = getattr(client, attr, None)
                    if mcp_url:
                        break

    logger.warning(
        "MCP %s retry %s for %s due to %s: %s",
        label,
        tries,
        mcp_url or "unknown-url",
        exc.__class__.__name__ if exc else "UnknownError",
        exc,
    )


def _log_giveup_event(details: dict):
    label = _describe_operation(details)
    exc = details.get("exception")
    # Best-effort extraction of the MCP server URL from the backoff context.
    args = details.get("args") or ()
    mcp_url = None
    if args:
        self_obj = args[0]
        mcp_url = getattr(self_obj, "mcp_url", None)
        if mcp_url is None:
            client = getattr(self_obj, "client", None)
            if client is not None:
                for attr in ("mcp_url", "url", "base_url", "server_url"):
                    mcp_url = getattr(client, attr, None)
                    if mcp_url:
                        break

    logger.error(
        "MCP %s giving up after %s attempts for %s due to %s: %s",
        label,
        details.get("tries", "unknown"),
        mcp_url or "unknown-url",
        exc.__class__.__name__ if exc else "UnknownError",
        exc,
    )

class GameEnv:
    def __init__(self, mcp_url: str):
        # Store the URL so it can be surfaced in retry/backoff logs.
        self.mcp_url = mcp_url
        self.client = Client(mcp_url, log_handler=log_handler)
    
    @backoff.on_exception(
        backoff.expo,
        RETRYABLE_MCP_EXCEPTIONS,
        max_time=MCP_MAX_RETRY_TIME,
        max_tries=MCP_MAX_RETRY_TRIES,
        base=MCP_BACKOFF_BASE,
        max_value=MCP_BACKOFF_MAX_INTERVAL,
        jitter=backoff.full_jitter,
        on_backoff=_log_retry_event,
        on_giveup=_log_giveup_event,
    )
    async def wait_for_ping(self):
        await asyncio.wait_for(self.client.ping(), timeout=MCP_CALL_TIMEOUT_SECONDS)
    
    @backoff.on_exception(
        backoff.expo,
        RETRYABLE_MCP_EXCEPTIONS,
        max_time=MCP_MAX_RETRY_TIME,
        max_tries=MCP_MAX_RETRY_TRIES,
        base=MCP_BACKOFF_BASE,
        max_value=MCP_BACKOFF_MAX_INTERVAL,
        jitter=backoff.full_jitter,
        on_backoff=_log_retry_event,
        on_giveup=_log_giveup_event,
    )
    async def _call_tool(self, tool_name: str, *args, **kwargs):
        logging.getLogger(__name__).debug("Calling tool %s with args=%s kwargs=%s", tool_name, args, kwargs)
        result = await asyncio.wait_for(
            self.client.call_tool(tool_name, *args, **kwargs),
            timeout=MCP_CALL_TIMEOUT_SECONDS,
        )
        logging.getLogger(__name__).debug("Tool %s result: %s", tool_name, result)
        return json.loads(result.structured_content["result"])

    async def load_obs(self):
        return await self._call_tool("load-obs")
    
    async def dispatch_final_action(self, action: str):
        return await self._call_tool("dispatch-final-action", {"action_str": action})
    
    async def get_game_config(self):
        return await self._call_tool("get-game-config")
