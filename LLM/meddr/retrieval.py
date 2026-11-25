from __future__ import annotations

import os
import re
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from typing import Any

from meddr.modeling import parse_exact_json_object
from meddr.prompts import RETRIEVER_SYSTEM, retriever_prompt
from meddr.semantic_validation import repair_retriever_answer


class Retriever:
    """Frozen clinical evidence retriever with runtime-only connection settings."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 3,
        thinking: str | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("MEDDR_RETRIEVER_KEY")
        if not resolved_key:
            raise ValueError("Set MEDDR_RETRIEVER_KEY before running MedDR.")
        resolved_url = base_url or os.getenv("MEDDR_RETRIEVER_ENDPOINT")
        address = urlsplit(resolved_url or "")
        if address.scheme != "https" or not address.hostname or address.username or address.password:
            raise ValueError("Set MEDDR_RETRIEVER_ENDPOINT to a secure chat-completions endpoint.")
        if not model:
            raise ValueError("A Retriever model must be supplied at runtime.")
        if timeout <= 0 or max_retries < 0 or max_tokens < 1:
            raise ValueError("Retriever timeout and token limit must be positive; retries cannot be negative.")
        self._key = resolved_key
        self._endpoint = resolved_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.last_generation: dict[str, Any] | None = None

    def _complete(self, payload: dict[str, Any]) -> str:
        payload = dict(payload)
        payload.update(payload.pop("extra_body", {}))
        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                return str(result["choices"][0]["message"].get("content") or "")
            except HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"Retriever request failed (status {exc.code}).") from None
            except (URLError, TimeoutError):
                pass
            except (KeyError, IndexError, TypeError, ValueError):
                raise RuntimeError("Retriever returned an invalid response.") from None
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Retriever request failed after retries.") from None

    @staticmethod
    def _salvage_answer(content: str) -> str | None:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
        match = re.search(
            r'["\']?answer["\']?\s*:\s*["\']?(.*)',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(
                r'["\']?\s*}\s*$',
                "",
                candidate,
            ).strip()
            candidate = candidate.replace("\\n", " ")
            candidate = candidate.replace('\\"', '"')
            return candidate or None
        if text and not text.startswith(("{", "[")):
            return text
        return None

    def retrieve(self, question: str) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": RETRIEVER_SYSTEM},
                {"role": "user", "content": retriever_prompt(question)},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.thinking:
            request["extra_body"] = {
                "thinking": {"type": self.thinking}
            }
        raw_completions: list[str] = []
        validation_rejections: list[str] = []
        answer: str | None = None
        fallback_used = False
        attempts_made = 0
        for attempt in range(1, 3):
            attempts_made = attempt
            attempt_request = dict(request)
            if attempt > 1:
                attempt_request["messages"] = [
                    *request["messages"],
                    {
                        "role": "user",
                        "content": (
                            "Return a concise valid JSON object now. Keep the "
                            "answer under 250 words and close every string and "
                            "brace."
                        ),
                    },
                ]
            try:
                content = self._complete(attempt_request)
            except Exception as exc:
                validation_rejections.append(
                    f"attempt {attempt}: {type(exc).__name__}"
                )
                continue
            if not isinstance(content, str) or not content.strip():
                validation_rejections.append(
                    f"attempt {attempt}: empty response"
                )
                continue
            raw_completions.append(content)
            try:
                parsed = parse_exact_json_object(content.strip())
                if "answer" not in parsed:
                    raise ValueError(
                        "output must contain the answer field"
                    )
                candidate = str(parsed["answer"]).strip()
                if not candidate:
                    raise ValueError("answer must be non-empty")
                answer = candidate
                break
            except ValueError as exc:
                validation_rejections.append(
                    f"attempt {attempt}: {exc}"
                )
        if answer is None:
            for content in reversed(raw_completions):
                answer = self._salvage_answer(content)
                if answer:
                    fallback_used = True
                    break
        if answer is None:
            answer = (
                "No usable external evidence was returned for this question. "
                "Treat the evidence as missing and retain the frozen prior."
            )
            fallback_used = True
        repaired_answer, semantic_repairs = repair_retriever_answer(answer)
        if fallback_used:
            semantic_repairs = [
                *semantic_repairs,
                "Recovered a usable evidence response after Retriever format "
                "validation failed.",
            ]
        self.last_generation = {
            "request": request,
            "raw_completion": (
                raw_completions[-1] if raw_completions else None
            ),
            "raw_completions": raw_completions,
            "accepted_answer": repaired_answer,
            "semantic_repairs": semantic_repairs,
            "validation_attempts": attempts_made,
            "validation_rejections": validation_rejections,
            "fallback_used": fallback_used,
        }
        return repaired_answer
