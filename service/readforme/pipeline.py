"""transcribe -> extract -> verify -> (explain)

Two VLM passes are used on purpose. A single "read and summarise" pass gives
the model room to invent a deadline that looks plausible; separating a verbatim
transcription pass from the structured pass gives the verifier evidence to check
against, and lets the eval measure OCR error and extraction error separately.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from . import prompts
from .backends import Backend
from .imaging import MAX_PIXELS, estimate_tokens, prepare
from .schema import ActionItem, DocumentReading, DocumentType, FactKind, KeyFact, RawExtraction
from .verify import verify_reading

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def parse_json_object(text: str) -> dict:
    """Tolerant JSON extraction: strips fences, takes the outermost {...}."""
    t = _FENCE.sub("", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(t[start: end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _coerce_kind(k: str | None) -> FactKind:
    try:
        return FactKind(k or "other")
    except ValueError:
        return FactKind.other


def _coerce_doctype(d: str | None) -> DocumentType:
    try:
        return DocumentType(d or "other")
    except ValueError:
        return DocumentType.other


def raw_to_reading(raw: dict, transcription: str, target_language: str, model_id: str) -> DocumentReading:
    r = RawExtraction(**{k: v for k, v in raw.items() if k in RawExtraction.model_fields})
    facts = []
    for f in r.key_facts:
        if not isinstance(f, dict) or not f.get("value"):
            continue
        facts.append(KeyFact(label=str(f.get("label", "")), value=str(f["value"]), kind=_coerce_kind(f.get("kind"))))
    actions = []
    for a in r.actions:
        if not isinstance(a, dict) or not a.get("text"):
            continue
        due = a.get("due")
        actions.append(ActionItem(text=str(a["text"]), due=str(due) if due else None))
    return DocumentReading(
        document_type=_coerce_doctype(r.document_type),
        source_language=r.source_language or "unknown",
        sender=r.sender, recipient=r.recipient, summary=r.summary,
        key_facts=facts, actions=actions,
        transcription=transcription, target_language=target_language, model_id=model_id,
    )


@dataclass
class Timings:
    prepare_s: float = 0.0
    transcribe_s: float = 0.0
    extract_s: float = 0.0
    explain_s: float = 0.0
    visual_tokens_est: int = 0
    extra: dict = field(default_factory=dict)


class ReadForMe:
    def __init__(self, backend: Backend, max_pixels: int = MAX_PIXELS, explain: bool = False):
        self.backend = backend
        self.max_pixels = max_pixels
        self.explain = explain

    def read(self, image: bytes | Image.Image, target_language: str = "en") -> tuple[DocumentReading, Timings]:
        t = Timings()
        t0 = time.perf_counter()
        prep = prepare(image, max_pixels=self.max_pixels)
        t.prepare_s = time.perf_counter() - t0
        t.visual_tokens_est = estimate_tokens(prep.image)

        t0 = time.perf_counter()
        transcription = self.backend.generate(prep.image, prompts.TRANSCRIBE, max_new_tokens=1536).strip()
        t.transcribe_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        raw_text = self.backend.generate(
            prep.image, prompts.EXTRACT.format(target_language=target_language, transcription=transcription), max_new_tokens=1024,
        )
        raw = parse_json_object(raw_text)
        t.extract_s = time.perf_counter() - t0

        reading = raw_to_reading(raw, transcription, target_language, getattr(self.backend, "model_id", ""))
        reading = verify_reading(reading)

        if self.explain:
            t0 = time.perf_counter()
            facts = "\n".join(
                f"- {f.label}: {f.value}" + ("" if f.verified in (True, None) else " [UNVERIFIED]") for f in reading.key_facts
            )
            actions = "\n".join(f"- {a.text}" + (f" (by {a.due})" if a.due else "") for a in reading.actions)
            reading.summary = self.backend.generate(
                None,
                prompts.EXPLAIN.format(target_language=target_language, document_type=reading.document_type,
                                       summary=reading.summary, facts=facts or "(none)", actions=actions or "(none)"),
                max_new_tokens=512,
            ).strip()
            t.explain_s = time.perf_counter() - t0
        return reading, t
