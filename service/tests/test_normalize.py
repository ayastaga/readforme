from readforme.normalize import date_tokens, money_tokens, number_tokens, phone_tokens, reference_tokens, dosage_tokens, canon_text


def test_money_locale_forms():
    assert "123450" in money_tokens("$1,234.50")
    assert "123450" in money_tokens("1 234,50 €")
    assert "1234" in money_tokens("$1,234.50")          # cents-less alias
    assert money_tokens("no money here") == set()


def test_arabic_indic_digits_fold():
    assert "2026" in number_tokens("٢٠٢٦")
    assert canon_text("Ｆｕｌｌ") == "full"


def test_dates_multilingual():
    assert "2026-09-26" in date_tokens("September 26, 2026")
    assert "2026-09-26" in date_tokens("26 septembre 2026")
    assert "2026-09-26" in date_tokens("26 de septiembre de 2026")
    assert "2026-09-26" in date_tokens("26. September 2026")
    assert "2026-09-26" in date_tokens("2026年9月26日")
    assert "2026-09-26" in date_tokens("2026-09-26")
    # ambiguous numeric: both readings
    toks = date_tokens("03/04/2026")
    assert {"2026-04-03", "2026-03-04"} <= toks


def test_phone_and_reference():
    assert "18009598281" in phone_tokens("Call 1-800-959-8281")
    assert "cra20261234567" in reference_tokens("Reference: CRA-2026-1234567")


def test_dosage():
    t = dosage_tokens("Amoxicillin 500 mg, 3 times a day")
    assert "500mg" in t and "3" in t
