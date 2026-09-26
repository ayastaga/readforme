"""Prompts. Kept as plain strings so the eval harness can version them."""

PROMPT_VERSION = "2026-09-26.1"

TRANSCRIBE = (
    "Transcribe all the text in this image exactly as written, in the original language. "
    "Preserve line breaks, numbers, dates, amounts, and reference numbers exactly. "
    "Do not translate, summarise, or add anything that is not on the page. "
    "If part of the text is unreadable, write [unreadable]."
)

EXTRACT = """You are reading a document for someone who cannot read this language.
Below is the verbatim transcription of the page. Using the image and the transcription, return ONLY a JSON object with this exact shape:

{{
  "document_type": one of ["government_letter","bill_or_invoice","lease_or_contract","medical","prescription","school_notice","bank_or_tax","legal_notice","advertisement","sign","other"],
  "source_language": BCP-47 code of the page language,
  "sender": who sent it or null,
  "recipient": who it is addressed to or null,
  "summary": 2-4 plain sentences in {target_language} explaining what this document is and what it means for the reader,
  "key_facts": [ {{"label": short label in {target_language}, "value": value COPIED VERBATIM from the page, "kind": one of ["money","date","deadline","dosage","reference","name","address","phone","other"]}} ],
  "actions": [ {{"text": what the reader should do, in {target_language}, "due": the deadline COPIED VERBATIM from the page or null}} ]
}}

Rules:
- "value" and "due" must be copied character-for-character from the page. Never compute, convert, or guess a number or date.
- If the page contains no deadline, "due" must be null.
- If you are not sure a fact is on the page, leave it out.
- Output JSON only, no markdown fences, no commentary.

TRANSCRIPTION:
{transcription}
"""

EXPLAIN = """Explain the following document to the reader in {target_language}, in plain, friendly language a non-expert would understand.
Use only the facts listed. Facts marked UNVERIFIED must be introduced with "the page may say" and the reader told to check the original.
Do not add any dates, amounts, or numbers that are not listed.

Document type: {document_type}
Summary: {summary}
Facts:
{facts}
Actions:
{actions}
"""
