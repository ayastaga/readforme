"""Upload-time hook: turn a photographed document into text the toolkit can index.

cohere-toolkit stores only extracted *text* for uploaded files (`File.file_content`).
For images we call the ReadForMe service and store a Markdown rendering of the
verified reading followed by a fenced JSON block, so:

  * `read_file` / `search_file` return something meaningful for a photo, and
  * `read_this_for_me` can recover the structured reading from the stored text.
"""

from __future__ import annotations

import json
import re

import requests

from backend.config.settings import Settings

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "heic", "heif", "bmp", "tif", "tiff"}
READING_MARKER = "<!-- readforme:v1 -->"
_JSON_BLOCK = re.compile(READING_MARKER + r"\s*```json\s*(\{.*\})\s*```", re.S)


def is_image_extension(ext: str) -> bool:
    return (ext or "").lower() in IMAGE_EXTENSIONS


def read_image_via_readforme(file_contents: bytes, filename: str, target_language: str | None = None) -> str:
    url = Settings().get("tools.read_this_for_me.url")
    if not url:
        raise ValueError("ReadForMe service URL not configured (tools.read_this_for_me.url)")
    lang = target_language or Settings().get("tools.read_this_for_me.default_language") or "en"
    timeout = float(Settings().get("tools.read_this_for_me.timeout_s") or 300)
    resp = requests.post(
        f"{url.rstrip('/')}/read",
        files={"image": (filename, file_contents)},
        data={"target_language": lang},
        timeout=timeout,
    )
    resp.raise_for_status()
    return render_reading(resp.json())


def render_reading(reading: dict) -> str:
    """Markdown the chat model can read, plus the JSON block for the tool."""
    lines = [f"# Photographed document: {reading.get('document_type', 'other')}",
             f"Language on page: {reading.get('source_language', 'unknown')}",
             "",
             reading.get("summary", ""),
             "", "## Key facts"]
    for f in reading.get("key_facts", []):
        v = f.get("verified")
        tag = "" if v else (" (NOT CONFIRMED ON PAGE)" if v is False else "")
        lines.append(f"- {f.get('label')}: {f.get('value')}{tag}")
    if reading.get("actions"):
        lines += ["", "## What to do"]
        for a in reading["actions"]:
            lines.append(f"- {a.get('text')}" + (f" — by {a['due']}" if a.get("due") else ""))
    if reading.get("warnings"):
        lines += ["", "## Warnings"] + [f"- {w}" for w in reading["warnings"]]
    lines += ["", "## Page transcription", "", reading.get("transcription", ""), "",
              READING_MARKER, "```json", json.dumps(reading, ensure_ascii=False), "```"]
    return "\n".join(lines)


def extract_reading_json(file_content: str) -> dict | None:
    m = _JSON_BLOCK.search(file_content or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
