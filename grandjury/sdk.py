"""
GrandJury Python SDK — main client.

Design principles:
- Zero-config: reads GRANDJURY_API_KEY from env. No-op if missing.
- Silent failure: all SDK exceptions caught, logged to stderr only.
- model_id resolved server-side from API key (no project_id needed).
- Decorator (@gj.observe) and context manager (gj.span()) supported.
- Read path: gj.results() returns ResultSet with .to_pandas()/.to_polars()/.to_parquet()/.to_csv()
"""

import functools
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from grandjury.result_set import ResultSet

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
                f"{self._client._base_url}/api/v1/models/me",
                headers={"Authorization": f"Bearer {self._client._auth_key}"},
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
                headers={"Authorization": f"Bearer {self._client._auth_key}"},
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("GrandJury: models.get error: %s", exc)
            return None


class _BenchmarksNamespace:
    """gj.benchmarks.list() / gj.benchmarks.leaderboard(id)"""

    def __init__(self, client: "GrandJury"):
        self._client = client

    def list(self) -> List[Dict[str, Any]]:
        """List available benchmarks."""
        try:
            import requests
            resp = requests.get(
                f"{self._client._base_url}/api/v1/challenges",
                headers={"Authorization": f"Bearer {self._client._auth_key}"},
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("GrandJury: benchmarks.list error: %s", exc)
            return []

    def leaderboard(self, evaluation_id: str) -> "ResultSet":
        """
        Public leaderboard — no auth required. Aggregate stats only.

        Returns per-model: name, slug, emoji, total_votes, pass_rate.
        """
        try:
            import requests
            resp = requests.get(
                f"{self._client._base_url}/api/v1/benchmarks/{evaluation_id}/leaderboard",
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return ResultSet(resp.json())
        except Exception as exc:
            logger.debug("GrandJury: benchmarks.leaderboard error: %s", exc)
            return []

    def votes(
        self,
        evaluation_id: str,
        model: Optional[str] = None,
        detail: str = "votes",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> "ResultSet":
        """
        Premium: vote-level data across all models in a benchmark.

        Requires authentication (PAT) and premium_subscriber role.

        Args:
            evaluation_id: benchmark UUID
            model: optional model slug to filter
            detail: 'votes' for individual votes (default), None for trace-level aggregates
            from_date: ISO date string (e.g. '2026-03-01')
            to_date: ISO date string (e.g. '2026-03-27')
            limit: max results (default 1000)
            offset: pagination offset

        Returns:
            ResultSet with .to_pandas(), .to_polars(), .to_parquet(), .to_csv()
        """
        self._client._require_auth()
        try:
            import requests
            params = {"limit": limit, "offset": offset}
            if detail:
                params["detail"] = detail
            if model:
                params["model"] = model
            if from_date:
                params["from_date"] = from_date
            if to_date:
                params["to_date"] = to_date

            resp = requests.get(
                f"{self._client._base_url}/api/v1/benchmarks/{evaluation_id}/votes",
                params=params,
                headers={"Authorization": f"Bearer {self._client._auth_key}"},
                timeout=self._client._timeout,
            )
            resp.raise_for_status()
            return ResultSet(resp.json())
        except Exception as exc:
            logger.debug("GrandJury: benchmarks.votes error: %s", exc)
            print(f"[grandjury] benchmarks.votes error: {exc}", file=sys.stderr)
            return ResultSet([])

    def enroll(self, benchmark_id: str, model_id: str, endpoint_config: Optional[Dict] = None) -> Optional[Dict]:
        """Enroll a model in a benchmark."""
        try:
            import requests
            resp = requests.post(
                f"{self._client._base_url}/api/v1/models/{model_id}/enroll/{benchmark_id}",
                json={"endpoint_config": endpoint_config},
                headers={"Authorization": f"Bearer {self._client._auth_key}"},
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
        token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 5.0,
        # Legacy compat — ignored but accepted so old code doesn't break
        project_id: Optional[str] = None,
    ):
        # PAT (gj_pat_*) takes priority via token= or GRANDJURY_TOKEN env var
        # Model key (gj_sk_*) via api_key= or GRANDJURY_API_KEY env var
        self._token = token or os.environ.get("GRANDJURY_TOKEN", "")
        self._api_key = api_key or os.environ.get("GRANDJURY_API_KEY", "") if not self._token else ""
        self._auth_key = self._token or self._api_key  # whichever is set
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._model_id: Optional[str] = None  # resolved lazily from API key
        self._models_cache: Optional[List[Dict[str, Any]]] = None

        # Namespaces
        self.models = _ModelsNamespace(self)
        self.benchmarks = _BenchmarksNamespace(self)
        self.analytics = _AnalyticsNamespace(self)

    def _require_auth(self):
        """Raise if no auth key is set."""
        if not self._auth_key:
            raise RuntimeError(
                "Authentication required. Set GRANDJURY_TOKEN (recommended) or GRANDJURY_API_KEY.\n"
                "Get your token at humanjudge.com/profile"
            )

    def _resolve_models(self) -> List[Dict[str, Any]]:
        """Lazily resolve models from token via /models/me endpoint."""
        self._require_auth()
        if self._models_cache is not None:
            return self._models_cache
        try:
            import requests
            resp = requests.get(
                f"{self._base_url}/api/v1/models/me",
                headers={"Authorization": f"Bearer {self._auth_key}"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            self._models_cache = resp.json()
        except Exception:
            self._models_cache = []
        return self._models_cache

    def _resolve_model_id(self, model: Optional[str] = None) -> Optional[str]:
        """
        Resolve model_id.

        - If model= is a UUID, use directly.
        - If model= is a slug, look up from /models/me cache.
        - If not provided and only one model, use that.
        """
        if model:
            # UUID check
            if len(model) == 36 and "-" in model:
                return model
            # Slug lookup
            models = self._resolve_models()
            for m in models:
                if m.get("slug") == model or m.get("name") == model:
                    return m["id"]
            return model  # assume it's an ID
        # No model specified — use single model if only one
        if self._model_id:
            return self._model_id
        models = self._resolve_models()
        if len(models) == 1:
            self._model_id = models[0]["id"]
            return self._model_id
        if len(models) > 1:
            logger.debug("GrandJury: multiple models found. Specify model= parameter.")
            print(f"[grandjury] You have {len(models)} models. Specify model= to choose one:", file=sys.stderr)
            for m in models:
                print(f"  - {m.get('slug') or m['id']}: {m['name']}", file=sys.stderr)
        return None

    # ── Read path ─────────────────────────────────────────────────────────────

    def results(
        self,
        model: Optional[str] = None,
        detail: Optional[str] = None,
        evaluation: Optional[str] = None,
        arena: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Any:
        """
        Fetch evaluation results.

        Two modes:
        1. Per-model (developer): gj.results(model="my-model") — returns your model's traces/votes.
        2. Per-arena (premium): gj.results(arena="eval-id") — returns all models' data in a benchmark.

        Only traces with ≥1 vote are returned (privacy gate).

        Args:
            model: model slug or ID. Required for per-model mode.
            detail: None for trace-level aggregates, 'votes' for individual votes with voter_id
            evaluation: alias for arena — filter by evaluation/benchmark ID
            arena: evaluation/benchmark ID — if provided without model, uses premium benchmark endpoint
            from_date: ISO date string (e.g. '2026-03-01') — only traces created on or after this date
            to_date: ISO date string (e.g. '2026-03-27') — only traces created on or before this date
            limit: max results (default 1000)
            offset: pagination offset

        Returns:
            ResultSet with .to_pandas(), .to_polars(), .to_parquet(), .to_csv(), .to_json()
        """
        self._require_auth()

        # arena= is an alias for evaluation=
        eval_id = arena or evaluation

        # If arena/evaluation provided but no model → premium benchmark endpoint
        if eval_id and not model:
            return self.benchmarks.votes(
                evaluation_id=eval_id,
                model=None,
                detail=detail or "votes",
                from_date=from_date,
                to_date=to_date,
                limit=limit,
                offset=offset,
            )

        # If arena + model → premium benchmark endpoint filtered by model
        if eval_id and model:
            # Check if user owns this model — if so, use the per-model endpoint
            model_id = self._resolve_model_id(model)
            if model_id:
                # User owns this model — use per-model endpoint with evaluation filter
                try:
                    import requests
                    params = {"limit": limit, "offset": offset}
                    if detail:
                        params["detail"] = detail
                    if eval_id:
                        params["evaluation_id"] = eval_id
                    if from_date:
                        params["from_date"] = from_date
                    if to_date:
                        params["to_date"] = to_date
                    resp = requests.get(
                        f"{self._base_url}/api/v1/models/{model_id}/evaluations",
                        params=params,
                        headers={"Authorization": f"Bearer {self._auth_key}"},
                        timeout=self._timeout,
                    )
                    if resp.status_code == 200:
                        return ResultSet(resp.json())
                except Exception:
                    pass
            # Fallback: try premium benchmark endpoint with model filter
            return self.benchmarks.votes(
                evaluation_id=eval_id,
                model=model,
                detail=detail or "votes",
                from_date=from_date,
                to_date=to_date,
                limit=limit,
                offset=offset,
            )

        model_id = self._resolve_model_id(model)
        if not model_id:
            logger.debug("GrandJury: could not resolve model_id")
            return ResultSet([])

        try:
            import requests
            params = {}
            if detail:
                params["detail"] = detail
            if eval_id:
                params["evaluation_id"] = eval_id
            if from_date:
                params["from_date"] = from_date
            if to_date:
                params["to_date"] = to_date

            resp = requests.get(
                f"{self._base_url}/api/v1/models/{model_id}/evaluations",
                params=params,
                headers={"Authorization": f"Bearer {self._auth_key}"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return ResultSet(resp.json())
        except Exception as exc:
            logger.debug("GrandJury: results() error: %s", exc)
            print(f"[grandjury] results error: {exc}", file=sys.stderr)
            return ResultSet([])

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
