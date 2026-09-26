"""Normalisation helpers used by the verifier.

The verifier never trusts a model's paraphrase of a number. It reduces both the
claimed value and the page transcription to canonical tokens (digit strings,
ISO-ish dates, E.164-ish phones) and requires the claimed tokens to appear in
the transcription's tokens. Everything here is deterministic and dependency-free.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

# --- text --------------------------------------------------------------------

_WS = re.compile(r"\s+")
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DEVANAGARI = str.maketrans("०१२३४५६७८९", "0123456789")
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")


def canon_text(s: str) -> str:
    """NFKC-fold, unify digits from other scripts, collapse whitespace, casefold."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.translate(_ARABIC_INDIC).translate(_DEVANAGARI).translate(_FULLWIDTH)
    s = s.replace("\u00a0", " ").replace("\u202f", " ")
    return _WS.sub(" ", s).strip().casefold()


# --- numbers / money -----------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])[+-]?\d[\d.,\u202f ']*\d|\d")


def _digits(token: str) -> str:
    return re.sub(r"\D", "", token)


def number_tokens(s: str) -> set[str]:
    """All numeric tokens as bare digit strings ("1,234.50" -> "123450").

    Also emits the integer part ("1234") so "$1,234.50" on page verifies a claim of
    "1234.50" or "1,234.50" or "1 234,50" and vice-versa. Thousands/decimal separators
    are locale-ambiguous, so we compare on digits only and additionally on the
    integer-part heuristics below.
    """
    out: set[str] = set()
    for m in _NUM.finditer(canon_text(s)):
        tok = m.group(0)
        d = _digits(tok)
        if not d:
            continue
        out.add(d)
        # integer part under both separator conventions
        for sep in (".", ","):
            if sep in tok:
                head = tok.rsplit(sep, 1)[0]
                hd = _digits(head)
                if hd:
                    out.add(hd)
    return out


_CURRENCY = re.compile(r"(\$|€|£|¥|₹|₩|CAD|USD|EUR|GBP|INR|AED|SAR|R\$|C\$|chf|kr)", re.I)


def money_tokens(s: str) -> set[str]:
    """Digit tokens that look like amounts (currency symbol nearby or 2-decimal form)."""
    c = canon_text(s)
    toks: set[str] = set()
    for m in _NUM.finditer(c):
        tok = m.group(0)
        window = c[max(0, m.start() - 6): m.end() + 6]
        two_dec = re.search(r"[.,]\d{2}$", tok) is not None
        if _CURRENCY.search(window) or two_dec:
            d = _digits(tok)
            if d:
                toks.add(d)
                # amount without cents, e.g. "150.00" -> "150"
                if two_dec:
                    toks.add(d[:-2] or "0")
    return toks


# --- dates ---------------------------------------------------------------------

_MONTHS = {
    # en
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # fr
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    # es / pt
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "maio": 5, "junho": 6, "julho": 7,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
    # de
    "januar": 1, "februar": 2, "märz": 3, "marz": 3, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
    # it
    "gennaio": 1, "febbraio": 2, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7,
    "settembre": 9, "ottobre": 10, "dicembre": 12,
}
_MONTH_RE = "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True))

_DATE_PATTERNS = [
    # 2026-09-26, 2026/09/26, 2026.09.26
    re.compile(r"\b(?P<y>(19|20)\d{2})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})\b"),
    # 26/09/2026, 26.09.2026, 26-09-2026  (also 09/26/2026 – ambiguous; we emit both readings)
    re.compile(r"\b(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<y>(19|20)\d{2})\b"),
    # 26 September 2026 / 26 sept. 2026 / 26 de septiembre de 2026
    re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\.?\s+(?:de\s+|of\s+)?(?P<mon>{_MONTH_RE})\.?,?\s+(?:de\s+)?(?P<y>(19|20)\d{{2}})\b"),
    # September 26, 2026 / Sept 26 2026
    re.compile(rf"\b(?P<mon>{_MONTH_RE})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>(19|20)\d{{2}})\b"),
    # 2026年9月26日
    re.compile(r"(?P<y>(19|20)\d{2})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日"),
]


def _mk(y: str, m: int, d: int) -> str | None:
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def date_tokens(s: str) -> set[str]:
    """ISO dates found in `s`. Ambiguous numeric forms emit every valid reading."""
    c = canon_text(s)
    out: set[str] = set()
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(c):
            g = m.groupdict()
            y = g.get("y")
            if "mon" in g and g.get("mon"):
                iso = _mk(y, _MONTHS[g["mon"]], int(g["d"]))
                if iso:
                    out.add(iso)
            elif g.get("a") is not None:
                a, b = int(g["a"]), int(g["b"])
                for mm, dd in ((b, a), (a, b)):
                    iso = _mk(y, mm, dd)
                    if iso:
                        out.add(iso)
            else:
                iso = _mk(y, int(g["m"]), int(g["d"]))
                if iso:
                    out.add(iso)
    return out


# --- phones / references -----------------------------------------------------

_PHONE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")


def phone_tokens(s: str) -> set[str]:
    out = set()
    for m in _PHONE.finditer(canon_text(s)):
        d = _digits(m.group(0))
        if 7 <= len(d) <= 15:
            out.add(d)
            out.add(d[-7:])  # local part
    return out


_REF = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z0-9][a-z0-9-]{4,}\b")


def reference_tokens(s: str) -> set[str]:
    """Alphanumeric identifiers (case numbers, invoice ids). Hyphens removed."""
    return {t.replace("-", "") for t in _REF.findall(canon_text(s))}


# --- dosage --------------------------------------------------------------------

_UNIT_CANON = {
    "mg": "mg", "毫克": "mg", "mcg": "mcg", "µg": "mcg", "ug": "mcg", "微克": "mcg",
    "g": "g", "克": "g", "ml": "ml", "毫升": "ml", "iu": "iu", "unit": "u", "units": "u",
    "tablet": "tab", "tablets": "tab", "tab": "tab", "tabs": "tab", "comprimé": "tab", "comprimés": "tab",
    "comprime": "tab", "comprimes": "tab", "comprimido": "tab", "comprimidos": "tab", "tablette": "tab",
    "tabletten": "tab", "片": "tab", "cap": "cap", "caps": "cap", "capsule": "cap", "capsules": "cap",
    "goutte": "drop", "gouttes": "drop", "drop": "drop", "drops": "drop", "滴": "drop",
}
_UNIT_RE = "|".join(sorted(map(re.escape, _UNIT_CANON), key=len, reverse=True))
_DOSE = re.compile(rf"(\d+(?:[.,]\d+)?)\s*({_UNIT_RE})(?![a-z])", re.I)


def dosage_tokens(s: str) -> set[str]:
    """Dose tokens as '<digits><canonical unit>' plus bare numbers (frequency/duration)."""
    out = set()
    for num, unit in _DOSE.findall(canon_text(s)):
        out.add(f"{_digits(num) or num}{_UNIT_CANON.get(unit.lower(), unit.lower())}")
    out |= number_tokens(s)
    return out
