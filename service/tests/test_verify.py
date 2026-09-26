from readforme.schema import ActionItem, DocumentReading, FactKind, KeyFact
from readforme.verify import verify_reading

PAGE = """Canada Revenue Agency
Notice of Assessment
Reference: CRA-2026-4417820
You have a balance owing of $1,234.50.
Pay the full amount by October 15, 2026.
Questions? Call 1-800-959-8281."""


def _r(facts, actions=()):
    return DocumentReading(transcription=PAGE, key_facts=facts, actions=list(actions))


def test_verified_facts_pass():
    r = verify_reading(_r([
        KeyFact(label="Amount", value="$1,234.50", kind=FactKind.money),
        KeyFact(label="Deadline", value="October 15, 2026", kind=FactKind.deadline),
        KeyFact(label="Ref", value="CRA-2026-4417820", kind=FactKind.reference),
        KeyFact(label="Phone", value="1-800-959-8281", kind=FactKind.phone),
    ]))
    assert all(f.verified for f in r.key_facts)
    assert r.verification.unverified == 0 and r.warnings == []


def test_model_reformatted_values_still_verify():
    r = verify_reading(_r([
        KeyFact(label="Amount", value="1234.50 CAD", kind=FactKind.money),
        KeyFact(label="Deadline", value="2026-10-15", kind=FactKind.deadline),
    ]))
    assert all(f.verified for f in r.key_facts)


def test_hallucinated_amount_and_date_are_flagged():
    r = verify_reading(_r([
        KeyFact(label="Amount", value="$1,243.50", kind=FactKind.money),   # digit transposition
        KeyFact(label="Deadline", value="October 5, 2026", kind=FactKind.deadline),
    ]))
    assert [f.verified for f in r.key_facts] == [False, False]
    assert r.verification.unverified == 2
    assert len(r.warnings) == 2 and "could not be confirmed" in r.warnings[0]


def test_fabricated_action_deadline_is_removed():
    r = verify_reading(_r([], actions=[ActionItem(text="Pay", due="November 1, 2026"),
                                       ActionItem(text="Object", due="October 15, 2026")]))
    assert r.actions[0].due is None and r.actions[0].verified is False
    assert r.actions[1].due == "October 15, 2026" and r.actions[1].verified is True
    assert r.verification.dropped_actions_due == 1


def test_non_critical_facts_not_gated():
    r = verify_reading(_r([KeyFact(label="Sender", value="Canada Revenue Agency", kind=FactKind.name),
                           KeyFact(label="Sender", value="Some Other Agency", kind=FactKind.name)]))
    assert r.key_facts[0].verified is True and r.key_facts[1].verified is None
    assert r.verification.critical_facts == 0


def test_ocr_noise_tolerated_by_fuzzy():
    noisy = PAGE.replace("Revenue", "Revenu e")
    r = DocumentReading(transcription=noisy, key_facts=[KeyFact(label="Sender", value="Canada Revenue Agency", kind=FactKind.name)])
    assert verify_reading(r).key_facts[0].verified is True
