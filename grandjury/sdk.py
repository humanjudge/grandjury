"""
GrandJury Python SDK — main client.

Design principles:
- Zero-config: reads GRANDJURY_API_KEY from env. No-op if missing.
- Silent failure: all SDK exceptions caught, logged to stderr only.
- model_id resolved server-side from API key (no project_id needed).
- Decorator (@gj.observe) and context manager (gj.span()) supported.
- Read path: gj.results() returns evaluation data as DataFrame.
"""

import functools
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Union

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


class _ModelsNamespace:
    """gj.models.list() / gj.models.get(id)"""

    def __init__(self, client: "GrandJury"):
        self._client = client

    def list(self) -> List[Dict[str, Any]]:
        """List all models owned by the current user (requires secret key)."""
        try:
            import requests
            resp = requests.get(
                f"{self._client._base_url}/api/v1/models",
                headers={"Authorization": f"Bearer {self._client._api_key}"},
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("GrandJury: models.list error: %s", exc)
            return []

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model detail including enrollments."""
        try:
            import requests
            resp = requests.get(
                f"{self._client._base_url}/api/v1/models/{model_id}",
                headers={"Authorization": f"Bearer {self._client._api_key}"},
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("GrandJury: models.get error: %s", exc)
            return None


class _BenchmarksNamespace:
    """gj.benchmarks.list() / gj.benchmarks.enroll(...)"""

    def __init__(self, client: "GrandJury"):
        self._client = client

    def list(self) -> List[Dict[str, Any]]:
        """List available benchmarks."""
        try:
            import requests
            resp = requests.get(
                f"{self._client._base_url}/api/v1/challenges",
                headers={"Authorization": f"Bearer {self._client._api_key}"},
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("GrandJury: benchmarks.list error: %s", exc)
            return []

    def enroll(self, benchmark_id: str, model_id: str, endpoint_config: Optional[Dict] = None) -> Optional[Dict]:
        """Enroll a model in a benchmark."""
        try:
            import requests
            resp = requests.post(
                f"{self._client._base_url}/api/v1/models/{model_id}/enroll/{benchmark_id}",
                json={"endpoint_config": endpoint_config},
                headers={"Authorization": f"Bearer {self._client._api_key}"},
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("GrandJury: benchmarks.enroll error: %s", exc)
            return None


class _AnalyticsNamespace:
    """
    gj.analytics.vote_histogram(...) / gj.analytics.population_confidence(...)

    Wraps GrandJuryClient methods. Works on both:
    - Live platform data: gj.analytics.vote_histogram(gj.results(detail='votes'))
    - Offline data: gj.analytics.vote_histogram(pd.read_csv("data.csv"))
    - Auto-fetch: gj.analytics.vote_histogram() — fetches from platform automatically
    """

    def __init__(self, client: "GrandJury"):
        self._client = client
        self._api_client = None

    def _get_api_client(self):
        if self._api_client is None:
            from .api_client import GrandJuryClient
            self._api_client = GrandJuryClient(api_key=self._client._api_key)
        return self._api_client

    def _auto_fetch(self, data):
        """If data is None, auto-fetch from platform."""
        if data is None:
            return self._client.results(detail='votes')
        return data

    def evaluate_model(self, previous_score: float = 0.0, previous_timestamp: str = None,
                       new_votes=None, **kwargs):
        c = self._get_api_client()
        new_votes = self._auto_fetch(new_votes)
        return c.evaluate_model(previous_score, previous_timestamp, new_votes, **kwargs)

    def vote_histogram(self, data=None, duration_minutes: int = 60, gross: bool = True):
        c = self._get_api_client()
        data = self._auto_fetch(data)
        return c.vote_histogram(data, duration_minutes, gross)

    def vote_completeness(self, data=None, voter_list=None, inference_ids=None, **kwargs):
        c = self._get_api_client()
        data = self._auto_fetch(data)
        return c.vote_completeness(data, voter_list or [], inference_ids, **kwargs)

    def population_confidence(self, data=None, voter_list=None, inference_ids=None):
        c = self._get_api_client()
        data = self._auto_fetch(data)
        return c.population_confidence(data, voter_list or [], inference_ids)

    def majority_good_votes(self, data=None, good_vote=True, threshold: float = 0.5):
        c = self._get_api_client()
        data = self._auto_fetch(data)
        return c.majority_good_votes(data, good_vote, threshold)

    def votes_distribution(self, data=None, inference_ids=None):
        c = self._get_api_client()
        data = self._auto_fetch(data)
        return c.votes_distribution(data, inference_ids)


class GrandJury:
    """
    GrandJury SDK client.

    Zero-config: reads GRANDJURY_API_KEY from environment.
    No project_id needed — model resolved server-side from API key.

    Usage:
        from grandjury import GrandJury
        gj = GrandJury()

        # Write traces
        gj.trace(name="chat", input=prompt, output=response, model="gpt-4o")

        # Read results (traces with ≥1 vote only)
        df = gj.results()
        df = gj.results(detail='votes')
        df = gj.results(evaluation='marketing-benchmark')

        # Analytics (works on live or offline data)
        gj.analytics.vote_histogram()
        gj.analytics.population_confidence(voter_list=[...])

        # Browse
        gj.models.list()
        gj.benchmarks.list()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 5.0,
        # Legacy compat — ignored but accepted so old code doesn't break
        project_id: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("GRANDJURY_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._model_id: Optional[str] = None  # resolved lazily from API key

        if not self._api_key:
            print("[grandjury] No API key found. Set GRANDJURY_API_KEY env var. SDK will no-op.", file=sys.stderr)

        # Namespaces
        self.models = _ModelsNamespace(self)
        self.benchmarks = _BenchmarksNamespace(self)
        self.analytics = _AnalyticsNamespace(self)

    def _resolve_model_id(self) -> Optional[str]:
        """Lazily resolve model_id from API key by listing models."""
        if self._model_id:
            return self._model_id
        try:
            models = self.models.list()
            if models:
                self._model_id = models[0]["id"]
        except Exception:
            pass
        return self._model_id

    # ── Read path ─────────────────────────────────────────────────────────────

    def results(
        self,
        detail: Optional[str] = None,
        evaluation: Optional[str] = None,
    ) -> Any:
        """
        Fetch evaluation results for your model.

        Only traces with ≥1 vote are returned (privacy gate).

        Args:
            detail: None for trace-level aggregates, 'votes' for individual votes
            evaluation: filter by evaluation slug or ID

        Returns:
            pandas DataFrame if pandas installed, else list[dict]
        """
        if not self._api_key:
            return []

        model_id = self._resolve_model_id()
        if not model_id:
            logger.debug("GrandJury: could not resolve model_id from API key")
            return []

        try:
            import requests
            params = {}
            if detail:
                params["detail"] = detail
            if evaluation:
                params["evaluation_id"] = evaluation

            resp = requests.get(
                f"{self._base_url}/api/v1/models/{model_id}/evaluations",
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Return as DataFrame if pandas available
            try:
                import pandas as pd
                return pd.DataFrame(data)
            except ImportError:
                return data
        except Exception as exc:
            logger.debug("GrandJury: results() error: %s", exc)
            print(f"[grandjury] results error: {exc}", file=sys.stderr)
            return []

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
        if not self._api_key:
            return None

        inference_id = gj_inference_id or _generate_inference_id()
        try:
            import requests

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
        if not self._api_key:
            return None

        inference_id = gj_inference_id or _generate_inference_id()
        try:
            import httpx

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
