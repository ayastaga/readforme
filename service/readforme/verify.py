"""Grounding verifier.

The VLM is asked twice: once to transcribe the page verbatim, once to extract
structured facts. The verifier trusts the *transcription* as evidence and checks
every critical extracted fact against it:

* money / dosage / reference / phone  -> canonical digit tokens must appear
* date / deadline                     -> the ISO date must appear (any reading of an
                                         ambiguous numeric date on the page counts)
* everything else                     -> fuzzy substring on canonical text (informational only)

A critical fact whose tokens are not on the page is kept but marked
`verified=False` with a note, a warning is added to the reading, and any
`ActionItem.due` that fails verification is blanked (a fabricated deadline is
worse than a missing one).

This is deliberately conservative: the transcription itself can be wrong, so
"verified" means "consistent with what the model read", not "true". The eval
harness measures how often each of those fails separately.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from .normalize import (
    canon_text, date_tokens, dosage_tokens, money_tokens, number_tokens,
    phone_tokens, reference_tokens,
)
from .schema import (
    CRITICAL_KINDS, ActionItem, DocumentReading, FactKind, KeyFact, VerificationReport,
)

FUZZY_THRESHOLD = 0.85


def _fuzzy_in(needle: str, hay: str) -> tuple[bool, str | None]:
    n, h = canon_text(needle), canon_text(hay)
    if not n:
        return False, None
    if n in h:
        i = h.find(n)
        return True, h[i: i + len(n)]
    # sliding window fuzzy match for OCR-ish noise
    win = len(n)
    best, best_span = 0.0, None
    step = max(1, win // 4)
    for i in range(0, max(1, len(h) - win + 1), step):
        seg = h[i: i + win]
        r = SequenceMatcher(None, n, seg).ratio()
        if r > best:
            best, best_span = r, seg
    return best >= FUZZY_THRESHOLD, best_span


def _token_check(kind: FactKind, value: str, page: str) -> tuple[bool, str | None]:
    """Return (ok, note). ok=True when every claimed canonical token is on the page."""
    if kind == FactKind.money:
        claimed = money_tokens(value) or number_tokens(value)
        on_page = money_tokens(page) | number_tokens(page)
        missing = {t for t in claimed if t not in on_page}
        # accept if any full-amount token matched (cents-less alias is only an alias)
        ok = bool(claimed) and any(t in on_page for t in claimed)
        return ok, None if ok else f"amount {value!r} not found on page (missing digits {sorted(missing)})"
    if kind in (FactKind.date, FactKind.deadline):
        claimed = date_tokens(value)
        if not claimed:
            # e.g. "within 30 days" – fall back to numbers
            claimed_n = number_tokens(value)
            ok = bool(claimed_n) and claimed_n <= number_tokens(page)
            return ok, None if ok else f"date {value!r} could not be parsed or matched"
        on_page = date_tokens(page)
        ok = bool(claimed & on_page)
        return ok, None if ok else f"date {value!r} not found on page"
    if kind == FactKind.phone:
        claimed, on_page = phone_tokens(value), phone_tokens(page)
        ok = bool(claimed & on_page)
        return ok, None if ok else f"phone {value!r} not found on page"
    if kind == FactKind.reference:
        claimed, on_page = reference_tokens(value), reference_tokens(page) | number_tokens(page)
        ok = bool(claimed) and claimed <= on_page
        if not ok:  # tolerate spacing differences in long ids
            ok = canon_text(value).replace(" ", "").replace("-", "") in canon_text(page).replace(" ", "").replace("-", "")
        return ok, None if ok else f"reference {value!r} not found on page"
    if kind == FactKind.dosage:
        claimed, on_page = dosage_tokens(value), dosage_tokens(page)
        ok = bool(claimed) and claimed <= on_page
        return ok, None if ok else f"dosage {value!r} not found on page"
    return False, None


def verify_fact(fact: KeyFact, page: str) -> KeyFact:
    """Return a copy of `fact` with `verified`, `evidence`, `note` populated."""
    f = fact.model_copy()
    found, span = _fuzzy_in(f.value, page)
    f.evidence = span
    if f.kind in CRITICAL_KINDS:
        ok, note = _token_check(f.kind, f.value, page)
        # Fuzzy text similarity must NOT rescue a critical fact: "October 5, 2026" is
        # 94% similar to "October 15, 2026" and exactly the kind of error we exist to catch.
        # Fuzzy is only consulted when the value carries no checkable token at all.
        has_tokens = bool(number_tokens(f.value) or date_tokens(f.value))
        f.verified = ok if has_tokens else found
        f.note = None if f.verified else (note or "value not found on page")
    else:
        f.verified = found if found else None
    return f


def verify_reading(reading: DocumentReading) -> DocumentReading:
    """Verify all critical facts and action deadlines in place; return the reading."""
    page = reading.transcription or ""
    report = VerificationReport()
    facts: list[KeyFact] = []
    for fact in reading.key_facts:
        v = verify_fact(fact, page)
        if v.kind in CRITICAL_KINDS:
            report.critical_facts += 1
            if v.verified:
                report.verified += 1
            else:
                report.unverified += 1
                reading.warnings.append(
                    f"'{v.label}: {v.value}' could not be confirmed on the page. Check the original before acting on it."
                )
        facts.append(v)
    reading.key_facts = facts

    actions: list[ActionItem] = []
    for a in reading.actions:
        a = a.model_copy()
        if a.due:
            ok, _ = _token_check(FactKind.deadline, a.due, page)
            has_tokens = bool(number_tokens(a.due) or date_tokens(a.due))
            a.verified = ok if has_tokens else _fuzzy_in(a.due, page)[0]
            if not a.verified:
                report.dropped_actions_due += 1
                reading.warnings.append(
                    f"A deadline '{a.due}' was suggested for '{a.text}' but is not on the page; it has been removed."
                )
                a.due = None
        actions.append(a)
    reading.actions = actions
    reading.verification = report
    return reading
