"""Render synthetic civic documents with known ground truth.

Why synthetic: real letters carry personal data and can't be shared. Synthetic
pages give an exact answer key for every amount, date, and reference number, so
we can measure hallucination precisely. Real, consented, redacted photos go in
`eval/real/` with the same JSON schema (see eval/README.md).

Each sample = (PNG, JSON) where the JSON holds the full page text and the list
of critical facts the pipeline should surface. Perturbations mimic phone photos:
rotation, perspective-ish shear, blur, JPEG, glare gradient, low light.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/noto", "/usr/share/fonts/opentype/noto"]


def _font(size: int, cjk: bool = False) -> ImageFont.FreeTypeFont:
    cands = ["NotoSerifCJK-Regular.ttc", "NotoSansCJK-Regular.ttc"] if cjk else ["DejaVuSerif.ttf", "DejaVuSans.ttf"]
    for d in FONT_DIRS:
        for c in cands:
            p = Path(d) / c
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


# ---------------------------------------------------------------- templates ---
# Each template returns (lines, facts). Facts: label, value, kind.
@dataclass
class Fact:
    label: str
    value: str
    kind: str


def _ref(rng):
    return f"{rng.choice(['CRA','IRCC','ON','SR','INV'])}-{rng.randint(2026, 2027)}-{''.join(rng.choices(string.digits, k=7))}"


def _amount(rng, sym="$"):
    v = rng.choice([rng.randint(12, 990) + rng.choice([0, 0.5, 0.25, 0.99]), rng.randint(1000, 12000) + 0.4])
    s = f"{v:,.2f}"
    return (f"{sym}{s}" if sym in "$£" else f"{s} {sym}"), v


def _date(rng, lang):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), 2026
    months = {
        "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
        "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
        "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
    }
    if lang == "en":
        s = rng.choice([f"{months['en'][m-1]} {d}, {y}", f"{y}-{m:02d}-{d:02d}"])
    elif lang == "fr":
        s = rng.choice([f"{d} {months['fr'][m-1]} {y}", f"{d:02d}/{m:02d}/{y}"])
    elif lang == "es":
        s = rng.choice([f"{d} de {months['es'][m-1]} de {y}", f"{d:02d}/{m:02d}/{y}"])
    elif lang == "de":
        s = rng.choice([f"{d}. {months['de'][m-1]} {y}", f"{d:02d}.{m:02d}.{y}"])
    else:  # zh
        s = f"{y}年{m}月{d}日"
    return s, f"{y}-{m:02d}-{d:02d}"


def tax_notice(rng, lang):
    ref = _ref(rng); amt_s, _ = _amount(rng); due_s, due_iso = _date(rng, lang); sent_s, _ = _date(rng, lang)
    name = rng.choice(["A. Okafor", "M. Nguyen", "S. Ahmed", "L. Rossi", "P. Singh"])
    T = {
        "en": [f"Canada Revenue Agency", f"Notice of Assessment", f"Date: {sent_s}", f"Reference: {ref}", f"Dear {name},",
               f"We have assessed your 2025 income tax return. You have a balance owing of {amt_s}.",
               f"To avoid interest charges, pay the full amount by {due_s}.",
               "You can pay online, at your bank, or by mail. If you disagree with this assessment,",
               "you have 90 days from the date of this notice to file an objection.",
               "Questions? Call 1-800-959-8281."],
        "fr": [f"Agence du revenu du Canada", f"Avis de cotisation", f"Date : {sent_s}", f"Référence : {ref}", f"Madame, Monsieur {name},",
               f"Nous avons établi la cotisation de votre déclaration de revenus 2025. Votre solde dû est de {amt_s}.",
               f"Pour éviter les intérêts, payez le montant total au plus tard le {due_s}.",
               "Vous pouvez payer en ligne, à votre banque ou par la poste. Si vous n'êtes pas d'accord,",
               "vous avez 90 jours à compter de la date de cet avis pour déposer une opposition.",
               "Questions ? Composez le 1-800-959-7383."],
        "es": [f"Agencia Tributaria", f"Notificación de liquidación", f"Fecha: {sent_s}", f"Referencia: {ref}", f"Estimado/a {name}:",
               f"Hemos revisado su declaración de la renta de 2025. Tiene un saldo pendiente de {amt_s}.",
               f"Para evitar intereses, pague el importe total antes del {due_s}.",
               "Puede pagar en línea, en su banco o por correo. Si no está de acuerdo,",
               "dispone de 90 días desde la fecha de esta carta para presentar una reclamación.",
               "¿Preguntas? Llame al 900 123 456."],
        "de": [f"Finanzamt", f"Steuerbescheid", f"Datum: {sent_s}", f"Aktenzeichen: {ref}", f"Sehr geehrte/r {name},",
               f"Ihre Einkommensteuererklärung 2025 wurde geprüft. Es ergibt sich eine Nachzahlung von {amt_s}.",
               f"Bitte überweisen Sie den Betrag bis zum {due_s}, um Säumniszuschläge zu vermeiden.",
               "Gegen diesen Bescheid können Sie innerhalb eines Monats Einspruch einlegen.",
               "Rückfragen: 0800 123 4567."],
        "zh": [f"税务局", f"评税通知书", f"日期：{sent_s}", f"档案编号：{ref}", f"{name} 先生/女士：",
               f"您的2025年度所得税申报已完成评税，应缴税款为 {amt_s}。",
               f"请于 {due_s} 前缴清全部款项，以免产生利息。",
               "如对本评税有异议，请在本通知书日期起90天内提出反对。",
               "查询电话：400 123 4567。"],
    }[lang]
    facts = [Fact("Amount owing", amt_s, "money"), Fact("Payment deadline", due_s, "deadline"),
             Fact("Reference number", ref, "reference"), Fact("Objection window", "90", "other")]
    return "government_letter", T, facts, {"deadline_iso": due_iso}


def prescription(rng, lang):
    drug = rng.choice(["Amoxicillin", "Metformin", "Ibuprofen", "Atorvastatin"])
    mg = rng.choice([250, 500, 850, 20, 400]); times = rng.randint(1, 3); days = rng.choice([5, 7, 10, 14, 30])
    rx = f"RX-{''.join(rng.choices(string.digits, k=8))}"; date_s, _ = _date(rng, lang)
    T = {
        "en": ["City Pharmacy", f"Prescription {rx}", f"Date: {date_s}", f"{drug} {mg} mg tablets",
               f"Take 1 tablet {times} times a day for {days} days.", "Take with food. Do not exceed the stated dose.",
               "Refills: 0", "Pharmacist: 416-555-0142"],
        "fr": ["Pharmacie du Centre", f"Ordonnance {rx}", f"Date : {date_s}", f"{drug} {mg} mg comprimés",
               f"Prendre 1 comprimé {times} fois par jour pendant {days} jours.", "À prendre avec de la nourriture.",
               "Renouvellements : 0", "Pharmacien : 514-555-0199"],
        "es": ["Farmacia Central", f"Receta {rx}", f"Fecha: {date_s}", f"{drug} {mg} mg comprimidos",
               f"Tomar 1 comprimido {times} veces al día durante {days} días.", "Tomar con alimentos.",
               "Repeticiones: 0", "Farmacéutico: 91 555 0123"],
        "de": ["Stadt-Apotheke", f"Rezept {rx}", f"Datum: {date_s}", f"{drug} {mg} mg Tabletten",
               f"1 Tablette {times}-mal täglich für {days} Tage einnehmen.", "Zu den Mahlzeiten einnehmen.",
               "Wiederholungen: 0", "Apotheker: 030 555 0177"],
        "zh": ["市中心药房", f"处方 {rx}", f"日期：{date_s}", f"{drug} {mg} 毫克 片剂",
               f"每日 {times} 次，每次 1 片，连服 {days} 天。", "饭后服用。", "续配：0 次", "药剂师：010-5550-1188"],
    }[lang]
    facts = [Fact("Medication", f"{drug} {mg} mg", "dosage"), Fact("Frequency per day", str(times), "dosage"),
             Fact("Duration (days)", str(days), "dosage"), Fact("Prescription number", rx, "reference")]
    return "prescription", T, facts, {}


def utility_bill(rng, lang):
    acct = "".join(rng.choices(string.digits, k=10)); amt_s, _ = _amount(rng); due_s, due_iso = _date(rng, lang)
    kwh = rng.randint(180, 900)
    T = {
        "en": ["Toronto Hydro", "Electricity bill", f"Account number: {acct}", f"Usage this period: {kwh} kWh",
               f"Total amount due: {amt_s}", f"Due date: {due_s}", "Late payments are subject to a 1.5% monthly charge.",
               "Pay online at torontohydro.com or call 416-542-8000."],
        "fr": ["Hydro-Québec", "Facture d'électricité", f"Numéro de compte : {acct}", f"Consommation : {kwh} kWh",
               f"Montant total à payer : {amt_s}", f"Date d'échéance : {due_s}", "Des frais de retard de 1,5 % par mois s'appliquent.",
               "Payez en ligne ou composez le 1 888 385-7252."],
        "es": ["Iberdrola", "Factura de electricidad", f"Número de contrato: {acct}", f"Consumo: {kwh} kWh",
               f"Importe total: {amt_s}", f"Fecha de vencimiento: {due_s}", "Los pagos atrasados tienen un recargo del 1,5 % mensual.",
               "Pague en línea o llame al 900 225 235."],
        "de": ["Stadtwerke", "Stromrechnung", f"Kundennummer: {acct}", f"Verbrauch: {kwh} kWh",
               f"Gesamtbetrag: {amt_s}", f"Fällig am: {due_s}", "Bei Zahlungsverzug fallen 1,5 % Zinsen monatlich an.",
               "Service: 0800 000 1234."],
        "zh": ["市电力公司", "电费账单", f"账户号码：{acct}", f"本期用电量：{kwh} 千瓦时",
               f"应付总额：{amt_s}", f"到期日：{due_s}", "逾期付款每月加收1.5%滞纳金。", "客服电话：95598。"],
    }[lang]
    facts = [Fact("Amount due", amt_s, "money"), Fact("Due date", due_s, "deadline"), Fact("Account number", acct, "reference")]
    return "bill_or_invoice", T, facts, {"deadline_iso": due_iso}


TEMPLATES = [tax_notice, prescription, utility_bill]
LANGS = ["en", "fr", "es", "de", "zh"]


# ---------------------------------------------------------------- rendering ---
def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int, cjk: bool) -> list[str]:
    """Word-wrap (character-wrap for CJK) so nothing is clipped; the answer key must be on the page."""
    units = list(text) if cjk else text.split(" ")
    joiner = "" if cjk else " "
    lines, cur = [], ""
    for u in units:
        cand = (cur + joiner + u) if cur else u
        if font.getlength(cand) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = u
    if cur:
        lines.append(cur)
    return lines


def render(lines: list[str], lang: str, rng: random.Random, w=1240, h=1754) -> Image.Image:
    img = Image.new("RGB", (w, h), (252, 251, 247))
    d = ImageDraw.Draw(img)
    cjk = lang == "zh"
    f_title = _font(44, cjk=cjk); f_body = _font(30, cjk=cjk)
    margin, y = 120, 140
    for i, line in enumerate(lines):
        f = f_title if i < 2 else f_body
        for sub in _wrap(line, f, w - 2 * margin, cjk):
            d.text((margin, y), sub, fill=(20, 20, 24), font=f)
            y += 70 if i < 2 else 48
        if i == 1:
            y += 20
    assert y < h - 100, "page overflow"
    return img


def perturb(img: Image.Image, level: str, rng: random.Random) -> Image.Image:
    """level: clean | photo | hard"""
    if level == "clean":
        return img
    out = img.rotate(rng.uniform(-2.5, 2.5) if level == "photo" else rng.uniform(-6, 6), expand=True, fillcolor=(120, 115, 110))
    # glare gradient
    glare = Image.linear_gradient("L").rotate(rng.uniform(0, 360), expand=False).resize(out.size)
    glare = ImageEnhance.Brightness(glare).enhance(0.35 if level == "photo" else 0.6)
    out = Image.composite(Image.new("RGB", out.size, (255, 255, 255)), out, glare.point(lambda v: min(255, v)))
    if level == "hard":
        out = out.filter(ImageFilter.GaussianBlur(1.2))
        out = ImageEnhance.Brightness(out).enhance(0.75)
    out = ImageEnhance.Contrast(out).enhance(0.9)
    # JPEG round-trip
    import io
    buf = io.BytesIO(); out.save(buf, "JPEG", quality=70 if level == "photo" else 45); buf.seek(0)
    return Image.open(buf).convert("RGB")


def generate(out_dir: Path, n: int = 60, seed: int = 7, levels=("clean", "photo", "hard")) -> list[dict]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        lang = LANGS[i % len(LANGS)]
        tmpl = TEMPLATES[(i // len(LANGS)) % len(TEMPLATES)]
        level = levels[i % len(levels)]
        doc_type, lines, facts, extra = tmpl(rng, lang)
        img = perturb(render(lines, lang, rng), level, rng)
        stem = f"{i:04d}_{tmpl.__name__}_{lang}_{level}"
        img.save(out_dir / f"{stem}.jpg", quality=90)
        gt = {"id": stem, "image": f"{stem}.jpg", "language": lang, "document_type": doc_type, "perturbation": level,
              "page_text": "\n".join(lines), "facts": [asdict(f) for f in facts], **extra}
        (out_dir / f"{stem}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))
        manifest.append(gt)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", default="eval/data/synth"); p.add_argument("--n", type=int, default=60); p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    m = generate(Path(a.out), a.n, a.seed)
    print(f"wrote {len(m)} samples to {a.out}")
