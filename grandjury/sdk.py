"""
GrandJury Python SDK — main client.

Design principles:
- Silent failure: all SDK exceptions are caught, logged to stderr only.
  The developer's app must never crash because of GrandJury.
- Sync-first with optional async support (asyncio.run for sync, native async for async).
- Auto-generates gj_inference_id if caller doesn't provide one.
- Decorator (@gj.observe) and context manager (gj.span()) supported.
"""

import functools
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

logger = logging.getLogger("grandjury")

# Default endpoint — override in constructor for self-hosted
DEFAULT_BASE_URL = "https://grandjury-server.onrender.com"


def _generate_inference_id() -> str:
    ts = int(time.time() * 1000)
    rand = uuid.uuid4().hex[:8]
    return f"gj_inf_{ts}_{rand}"


class Span:
    """
    Returned by GrandJury.span() context manager.

    Developer sets output (and optionally metadata) inside the `with` block.
    Trace is submitted on __exit__.
    """

    def __init__(
        self,
        client: "GrandJury",
        name: str,
        input: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._client = client
        self._name = name
        self._input = input
        self._model = model
        self._metadata = metadata or {}
        self._output: Optional[str] = None
        self._start_ms = int(time.time() * 1000)
        self.gj_inference_id: str = _generate_inference_id()

    def set_output(self, output: Any) -> None:
        """Set the model output. Accepts str or will be str()-converted."""
        self._output = output if isinstance(output, str) else str(output)

    def set_metadata(self, **kwargs: Any) -> None:
        self._metadata.update(kwargs)


class GrandJury:
    """
    GrandJury SDK client.

    Args:
        api_key:    Developer API key (gj_sk_live_…). Required.
        project_id: Evaluation/project UUID. Required.
        base_url:   Override API base URL (default: production server).
        timeout:    HTTP timeout in seconds (default: 5).
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 5.0,
    ):
        self._api_key = api_key
        self._project_id = project_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ── Core submit ───────────────────────────────────────────────────────────

    def trace(
        self,
        name: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
        model: Optional[str] = None,
        latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        gj_inference_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Submit one trace synchronously.

        Returns the gj_inference_id on success, None on failure (silent).
        """
        inference_id = gj_inference_id or _generate_inference_id()
        try:
            import requests  # lazy import — only required dep

            resp = requests.post(
                f"{self._base_url}/api/v1/traces/ingest",
                json={
                    "gj_inference_id": inference_id,
                    "name": name,
                    "input": input,
                    "output": output,
                    "model": model,
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "metadata": metadata,
                },
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
            if resp.status_code not in (200, 201):
                logger.debug(
                    "GrandJury: trace submit failed (%s): %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return None
            return inference_id
        except Exception as exc:
            logger.debug("GrandJury: trace submit error (silent): %s", exc)
            print(f"[grandjury] trace submit error: {exc}", file=sys.stderr)
            return None

    async def atrace(
        self,
        name: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
        model: Optional[str] = None,
        latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        gj_inference_id: Optional[str] = None,
    ) -> Optional[str]:
        """Async version of trace()."""
        inference_id = gj_inference_id or _generate_inference_id()
        try:
            import httpx  # lazy import — optional dep for async

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/v1/traces/ingest",
                    json={
                        "gj_inference_id": inference_id,
                        "name": name,
                        "input": input,
                        "output": output,
                        "model": model,
                        "latency_ms": latency_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "metadata": metadata,
                    },
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            if resp.status_code not in (200, 201):
                logger.debug(
                    "GrandJury: async trace submit failed (%s): %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return None
            return inference_id
        except Exception as exc:
            logger.debug("GrandJury: async trace submit error (silent): %s", exc)
            print(f"[grandjury] async trace error: {exc}", file=sys.stderr)
            return None

    # ── Decorator ─────────────────────────────────────────────────────────────

    def observe(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator. Wraps a function and submits a trace on every call.

        Captures:
            input  — str(args[0]) if a single positional arg, else str(kwargs)
            output — str(return_value)
            latency_ms — measured wall time

        Usage:
            @gj.observe(name="my_llm_call", model="gpt-4o")
            def call_llm(prompt: str) -> str:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            op_name = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.time()
                result = fn(*args, **kwargs)
                elapsed_ms = int((time.time() - start) * 1000)

                inp = str(args[0]) if len(args) == 1 else str(kwargs or args)
                out = result if isinstance(result, str) else str(result)

                self.trace(
                    name=op_name,
                    input=inp,
                    output=out,
                    model=model,
                    latency_ms=elapsed_ms,
                    metadata=metadata,
                )
                return result

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                import asyncio
                start = time.time()
                result = await fn(*args, **kwargs)
                elapsed_ms = int((time.time() - start) * 1000)

                inp = str(args[0]) if len(args) == 1 else str(kwargs or args)
                out = result if isinstance(result, str) else str(result)

                import asyncio
                asyncio.ensure_future(
                    self.atrace(
                        name=op_name,
                        input=inp,
                        output=out,
                        model=model,
                        latency_ms=elapsed_ms,
                        metadata=metadata,
                    )
                )
                return result

            import asyncio
            import inspect
            return async_wrapper if inspect.iscoroutinefunction(fn) else wrapper

        return decorator

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        input: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """
        Context manager. Submits trace on exit.

        Usage:
            with gj.span("my_operation", input=prompt) as t:
                response = model.generate(prompt)
                t.set_output(response)
        """
        s = Span(self, name=name, input=input, model=model, metadata=metadata)
        try:
            yield s
        finally:
            elapsed_ms = int(time.time() * 1000) - s._start_ms
            self.trace(
                name=s._name,
                input=s._input,
                output=s._output,
                model=s._model,
                latency_ms=elapsed_ms,
                metadata=s._metadata,
                gj_inference_id=s.gj_inference_id,
            )
