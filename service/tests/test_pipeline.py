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
