"""Data model for a verified document reading.

Every *critical* fact (money, date, deadline, dosage, reference number) carries a
`verified` flag set by `readforme.verify`, plus the exact span of the page
transcription it was matched against. Unverified facts are still returned, but
the UI and the toolkit tool must present them as "not confirmed on the page".
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class FactKind(StrEnum):
    money = "money"
    date = "date"
    deadline = "deadline"          # a date the reader must act by
    dosage = "dosage"              # medication quantity/frequency
    reference = "reference"        # case / account / invoice number
    name = "name"                  # person or organisation
    address = "address"
    phone = "phone"
    other = "other"


# Kinds whose hallucination can hurt someone. These are gated by the verifier.
CRITICAL_KINDS = {FactKind.money, FactKind.date, FactKind.deadline,
                  FactKind.dosage, FactKind.reference, FactKind.phone}


class DocumentType(StrEnum):
    government_letter = "government_letter"
    bill_or_invoice = "bill_or_invoice"
    lease_or_contract = "lease_or_contract"
    medical = "medical"
    prescription = "prescription"
    school_notice = "school_notice"
    bank_or_tax = "bank_or_tax"
    legal_notice = "legal_notice"
    advertisement = "advertisement"
    sign = "sign"
    other = "other"


class KeyFact(BaseModel):
    label: str = Field(description="Short human label, e.g. 'Amount due'")
    value: str = Field(description="Value copied verbatim from the page")
    kind: FactKind = FactKind.other
    verified: Optional[bool] = Field(
        default=None,
        description="True if `value` was found in the page transcription; None if not a critical kind",
    )
    evidence: Optional[str] = Field(
        default=None, description="The transcription span the value was matched to"
    )
    note: Optional[str] = Field(default=None, description="Verifier note, e.g. 'number not on page'")


class ActionItem(BaseModel):
    text: str
    due: Optional[str] = Field(default=None, description="Deadline as written on the page, if any")
    verified: Optional[bool] = None


class DocumentReading(BaseModel):
    """The structured, verified output of the pipeline."""

    document_type: DocumentType = DocumentType.other
    source_language: str = Field(default="unknown", description="BCP-47 code of the page text")
    sender: Optional[str] = None
    recipient: Optional[str] = None
    summary: str = Field(default="", description="2-4 sentence plain-language summary in the target language")
    key_facts: list[KeyFact] = Field(default_factory=list)
    actions: list[ActionItem] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Reader-facing cautions, e.g. 'the deadline could not be confirmed on the page'",
    )
    transcription: str = Field(default="", description="Verbatim page text as read by the model")
    target_language: str = "en"
    model_id: str = ""
    verification: "VerificationReport" = Field(default_factory=lambda: VerificationReport())


class VerificationReport(BaseModel):
    critical_facts: int = 0
    verified: int = 0
    unverified: int = 0
    dropped_actions_due: int = 0

    @property
    def unverified_ratio(self) -> float:
        return self.unverified / self.critical_facts if self.critical_facts else 0.0


DocumentReading.model_rebuild()


class RawExtraction(BaseModel):
    """Loose shape the VLM is asked to emit in pass 2, before verification."""

    document_type: str = "other"
    source_language: str = "unknown"
    sender: Optional[str] = None
    recipient: Optional[str] = None
    summary: str = ""
    key_facts: list[dict] = Field(default_factory=list)
    actions: list[dict] = Field(default_factory=list)
