import json
from readforme.backends import MockBackend
from readforme.pipeline import ReadForMe, parse_json_object
from PIL import Image

PAGE = "Toronto Hydro\nAccount number: 1234567890\nTotal amount due: $88.20\nDue date: 2026-10-01"


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert parse_json_object("not json") == {}


def test_end_to_end_mock_flags_invented_deadline():
    def extractor(img, t):
        return {"document_type": "bill_or_invoice", "source_language": "en", "summary": "A hydro bill.",
                "key_facts": [{"label": "Amount due", "value": "$88.20", "kind": "money"},
                              {"label": "Account", "value": "1234567890", "kind": "reference"},
                              {"label": "Due date", "value": "2026-10-11", "kind": "deadline"}],   # wrong
                "actions": [{"text": "Pay the bill", "due": "2026-10-11"}]}
    engine = ReadForMe(MockBackend(lambda img: PAGE, extractor))
    reading, t = engine.read(Image.new("RGB", (400, 600), "white"), target_language="fr")
    assert reading.document_type.value == "bill_or_invoice"
    by = {f.label: f for f in reading.key_facts}
    assert by["Amount due"].verified and by["Account"].verified
    assert by["Due date"].verified is False
    assert reading.actions[0].due is None
    assert reading.verification.unverified == 1 and reading.verification.dropped_actions_due == 1
    assert len(engine.backend.calls) == 2  # transcribe + extract, no explain pass


def test_pixel_budget_applied():
    from readforme.imaging import prepare, MAX_PIXELS
    p = prepare(Image.new("RGB", (4000, 6000), "white"))
    w, h = p.image.size
    assert w * h <= MAX_PIXELS and w % 32 == 0 and h % 32 == 0


def test_action_deadline_promoted_and_unreadable_dropped():
    # Mirrors the first real-model run: deadline only on the action (translated month),
    # plus a fact whose value is the transcription placeholder.
    page = "Agence du revenu du Canada\nSolde dû : $132.25\nPayez au plus tard le 4 octobre 2026."
    def extractor(img, t):
        return {"document_type": "government_letter", "summary": "Avis.",
                "key_facts": [{"label": "Amount", "value": "$132.25", "kind": "money"},
                              {"label": "Ref", "value": "[unreadable] 4020 2026", "kind": "reference"}],
                "actions": [{"text": "Pay the amount due", "due": "4 October 2026"}]}
    reading, _ = ReadForMe(MockBackend(lambda img: page, extractor)).read(Image.new("RGB", (200, 300), "white"))
    by_kind = {f.kind.value: f for f in reading.key_facts}
    assert "reference" not in by_kind and any("could not be read" in w for w in reading.warnings)
    assert by_kind["deadline"].value == "4 October 2026" and by_kind["deadline"].verified is True
    assert reading.actions[0].due == "4 October 2026" and reading.actions[0].verified is True
