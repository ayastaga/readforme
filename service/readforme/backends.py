"""Model backends.

All backends expose one method:

    generate(image: PIL.Image | None, prompt: str, max_new_tokens: int) -> str

* `TransformersBackend` follows the North-Micro-Vision model card exactly
  (transformers==5.16.0, AutoModelForImageTextToText, processor.apply_chat_template).
* `OpenAICompatBackend` talks to a vLLM server (`vllm serve CohereLabs/North-Micro-Vision-Instruct`)
  or any OpenAI-compatible endpoint. Used by the docker-compose deployment.
* `MockBackend` is deterministic and used by the unit tests and the synthetic eval's
  "oracle" mode; it never loads a model.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Callable, Optional, Protocol

from PIL import Image

DEFAULT_MODEL_ID = "CohereLabs/North-Micro-Vision-Instruct"


class Backend(Protocol):
    model_id: str

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024) -> str: ...


# --------------------------------------------------------------------------- #
class TransformersBackend:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto", attn: str | None = None):
        import torch  # noqa: F401
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        kwargs = {"dtype": "auto", "device_map": device_map}
        if attn:
            kwargs["attn_implementation"] = attn
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024) -> str:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.model.device)
        # Deterministic decoding: extraction must be reproducible for the eval harness.
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# --------------------------------------------------------------------------- #
class OpenAICompatBackend:
    def __init__(self, base_url: str | None = None, model_id: str = DEFAULT_MODEL_ID, api_key: str | None = None, timeout: float = 300):
        import httpx

        self.base_url = (base_url or os.environ.get("READFORME_VLM_URL", "http://localhost:8001/v1")).rstrip("/")
        self.model_id = model_id
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key or os.environ.get('READFORME_VLM_KEY', 'none')}"})

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024) -> str:
        content = []
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": self._data_url(image)}})
        content.append({"type": "text", "text": prompt})
        r = self.client.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model_id, "messages": [{"role": "user", "content": content}],
                  "max_tokens": max_new_tokens, "temperature": 0},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
class MockBackend:
    """Deterministic backend for tests and oracle evals.

    `transcriber(image)` and `extractor(image, transcription)` are injectable so the
    eval can simulate (a) a perfect model, (b) OCR noise, (c) hallucinated numbers.
    """

    model_id = "mock"

    def __init__(self, transcriber: Callable[[Optional[Image.Image]], str] | None = None,
                 extractor: Callable[[Optional[Image.Image], str], dict] | None = None):
        self.transcriber = transcriber or (lambda img: "")
        self.extractor = extractor or (lambda img, t: {"summary": "", "key_facts": [], "actions": []})
        self.calls: list[str] = []

    def generate(self, image, prompt: str, max_new_tokens: int = 1024) -> str:
        self.calls.append(prompt[:40])
        if prompt.startswith("Transcribe"):
            return self.transcriber(image)
        if "TRANSCRIPTION:" in prompt:
            transcription = prompt.split("TRANSCRIPTION:", 1)[1].strip()
            return json.dumps(self.extractor(image, transcription), ensure_ascii=False)
        return "explanation: " + prompt[-200:]


def make_backend(kind: str | None = None, **kw) -> Backend:
    kind = (kind or os.environ.get("READFORME_BACKEND", "openai")).lower()
    if kind == "transformers":
        return TransformersBackend(**kw)
    if kind in ("openai", "vllm"):
        return OpenAICompatBackend(**kw)
    if kind == "mock":
        return MockBackend(**kw)
    raise ValueError(f"unknown backend {kind}")
