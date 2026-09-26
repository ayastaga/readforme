"""ReadThisForMe tool for cohere-toolkit.

Images uploaded to a conversation are read at upload time by the ReadForMe
service (see backend/services/readforme_upload.py); the verified reading is stored
as the file's text content so `read_file` / `search_file` already work on it.

This tool exposes the *structured* reading: each key fact becomes its own tool
result document with a title such as "Amount due — verified" or
"Payment deadline — NOT CONFIRMED ON PAGE", so the Command model can cite facts
individually and the citations in the UI show the verification state.
"""

from __future__ import annotations

import json
from typing import Any

import backend.crud.file as file_crud
from backend.schemas.context import Context
from backend.schemas.tool import ToolCategory, ToolDefinition
from backend.services.readforme_upload import READING_MARKER, extract_reading_json
from backend.tools.base import BaseTool, ToolDefaultPreambleRegistry

ToolDefaultPreambleRegistry.set_preamble(
    "read_this_for_me",
    "When the user uploads a photo of a letter, form, bill, prescription, sign, or notice, use the "
    "read_this_for_me tool with the file as a tuple (filename, file ID). Explain the document in the "
    "user's language. Facts whose title says NOT CONFIRMED ON PAGE must be presented as uncertain and "
    "the user told to check the original. Never state a date, amount, dosage, or deadline that is not in "
    "the tool results.",
)


class ReadThisForMeTool(BaseTool):
    ID = "read_this_for_me"

    def __init__(self):
        pass

    @classmethod
    def is_available(cls) -> bool:
        from backend.config.settings import Settings

        return bool(Settings().get("tools.read_this_for_me.url"))

    @classmethod
    def get_tool_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name=cls.ID,
            display_name="Read This For Me",
            implementation=cls,
            parameter_definitions={
                "file": {
                    "description": "The uploaded photo of a document, as a tuple (filename, file ID)",
                    "type": "tuple[str, str]",
                    "required": True,
                },
            },
            is_visible=True,
            is_available=cls.is_available(),
            error_message=cls.generate_error_message(),
            category=ToolCategory.FileLoader,
            description=(
                "Reads a photographed letter, form, bill, prescription, notice or sign and returns its "
                "type, a summary, and every key fact (amounts, dates, deadlines, dosages, reference numbers) "
                "with a flag saying whether the fact was confirmed on the page."
            ),
        )  # type: ignore

    async def call(self, parameters: dict, ctx: Context, **kwargs: Any) -> list[dict[str, Any]]:
        file = parameters.get("file")
        session = kwargs.get("session")
        user_id = kwargs.get("user_id")
        if not file:
            return self.get_tool_error(details="No file was passed in the tool parameters")
        _, file_id = file
        retrieved = file_crud.get_file(session, file_id, user_id)
        if not retrieved:
            return self.get_tool_error(details="File not found")

        reading = extract_reading_json(retrieved.file_content or "")
        if reading is None:
            return self.get_tool_error(
                details=f"{retrieved.file_name} is not a photographed document read by ReadForMe "
                        f"(missing {READING_MARKER}). Upload a JPG/PNG/WEBP photo of the document."
            )

        name = retrieved.file_name
        results: list[dict[str, Any]] = [{
            "title": f"{name}: overview",
            "url": name,
            "text": (
                f"Document type: {reading.get('document_type')}\n"
                f"Language of the page: {reading.get('source_language')}\n"
                f"Sender: {reading.get('sender') or 'unknown'}\n"
                f"Summary: {reading.get('summary')}\n"
                f"Verification: {reading.get('verification', {}).get('verified', 0)} of "
                f"{reading.get('verification', {}).get('critical_facts', 0)} critical facts confirmed on the page."
            ),
        }]
        for f in reading.get("key_facts", []):
            v = f.get("verified")
            state = "verified" if v else ("NOT CONFIRMED ON PAGE" if v is False else "informational")
            results.append({
                "title": f"{f.get('label')} — {state}",
                "url": name,
                "text": f"{f.get('label')}: {f.get('value')}"
                        + (f"\nEvidence on page: {f['evidence']}" if f.get("evidence") else "")
                        + (f"\nNote: {f['note']}" if f.get("note") else ""),
            })
        for a in reading.get("actions", []):
            results.append({
                "title": "Action for the reader" + (f" (by {a['due']})" if a.get("due") else ""),
                "url": name,
                "text": a.get("text", ""),
            })
        for w in reading.get("warnings", []):
            results.append({"title": "Warning", "url": name, "text": w})
        results.append({"title": f"{name}: page transcription", "url": name, "text": reading.get("transcription", "")})
        return results
