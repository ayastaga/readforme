"""Evaluate the pipeline on a dataset directory of (image, ground-truth JSON) pairs.

Metrics (per sample, aggregated by language / perturbation / document type):

  cer                 character error rate of the transcription vs page_text
  fact_recall         fraction of ground-truth critical facts whose canonical tokens appear
                      among the pipeline's key_facts of the same kind
  hallucinated_facts  critical facts the pipeline returned whose tokens are NOT in the
                      ground-truth page text (i.e. invented numbers/dates)
  verifier_catch      fraction of hallucinated facts the verifier flagged as unverified
  verifier_false_flag fraction of correct critical facts the verifier flagged
  fabricated_deadline 1 if any action `due` was not on the page (before verification)
  latency_s           transcribe + extract wall time

The "before verification" numbers come from the raw extraction; the "after"
numbers are what the user would see. That difference is the point of the project.

Usage:
  python -m eval.run_eval --data eval/data/synth --backend mock --oracle        # sanity: perfect model
  python -m eval.run_eval --data eval/data/synth --backend mock --noise 0.05 --hallucinate 0.2
  python -m eval.run_eval --data eval/data/synth --backend transformers --max-pixels 3868706
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readforme.backends import MockBackend, make_backend  # noqa: E402
from readforme.normalize import (canon_text, date_tokens, dosage_tokens, money_tokens,  # noqa: E402
                                 number_tokens, reference_tokens)
from readforme.pipeline import ReadForMe  # noqa: E402
from readforme.schema import CRITICAL_KINDS, FactKind  # noqa: E402


# ------------------------------------------------------------------ metrics ---
# Several labels are legitimately correct for one document (a tax notice is both a
# government letter and a tax document). Scoring only the generator's label penalised
# correct answers in the first real-model run.
DOC_TYPE_ACCEPT = {
    "government_letter": {"government_letter", "bank_or_tax", "legal_notice"},
    "bill_or_invoice": {"bill_or_invoice"},
    "prescription": {"prescription", "medical"},
}
def cer(ref: str, hyp: str) -> float:
    r, h = canon_text(ref), canon_text(hyp)
    if not r:
        return 0.0
    # Levenshtein (O(n*m); pages are short)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def tokens_for(kind: str, value: str) -> set[str]:
    k = FactKind(kind) if kind in FactKind.__members__ else FactKind.other
    if k == FactKind.money:
        return money_tokens(value) | number_tokens(value)
    if k in (FactKind.date, FactKind.deadline):
        return date_tokens(value) or number_tokens(value)
    if k == FactKind.reference:
        return reference_tokens(value) | number_tokens(value)
    if k == FactKind.dosage:
        return dosage_tokens(value)
    return number_tokens(value) or {canon_text(value)}


def page_tokens(page_text: str) -> set[str]:
    return (money_tokens(page_text) | number_tokens(page_text) | date_tokens(page_text)
            | reference_tokens(page_text) | dosage_tokens(page_text))


def pred_tokens(reading) -> set[str]:
    """Every checkable token the pipeline showed the user: key facts plus action deadlines."""
    got: set[str] = set()
    for kf in reading.key_facts:
        got |= tokens_for(kf.kind.value, kf.value)
    for a in reading.actions:
        if a.due:
            got |= tokens_for("deadline", a.due)
    return got


def score_sample(gt: dict, reading) -> dict:
    page = gt["page_text"]
    ptoks = page_tokens(page)
    out = {"cer": cer(page, reading.transcription)}

    # recall of ground-truth critical facts
    gt_crit = [f for f in gt["facts"] if f["kind"] in CRITICAL_KINDS]
    got = pred_tokens(reading)
    hits = 0
    for f in gt_crit:
        want = tokens_for(f["kind"], f["value"])
        if want and want & got:
            hits += 1
    out["fact_recall"] = hits / len(gt_crit) if gt_crit else 1.0

    # hallucination: returned critical facts not supported by ground-truth page text
    halluc, flagged, correct, false_flag = 0, 0, 0, 0
    for kf in reading.key_facts:
        if kf.kind not in CRITICAL_KINDS:
            continue
        supported = bool(tokens_for(kf.kind.value, kf.value) & ptoks)
        if supported:
            correct += 1
            if kf.verified is False:
                false_flag += 1
        else:
            halluc += 1
            if kf.verified is False:
                flagged += 1
    out["hallucinated_facts"] = halluc
    out["verifier_catch"] = flagged / halluc if halluc else None
    out["verifier_false_flag"] = false_flag / correct if correct else None

    # fabricated deadlines: "shown" = what the user sees after verification;
    # "raw" additionally counts deadlines the verifier blanked before display.
    shown = int(any(a.due and not (tokens_for("deadline", a.due) & ptoks) for a in reading.actions))
    out["fabricated_deadline_shown"] = shown
    out["fabricated_deadline_raw"] = int(shown or reading.verification.dropped_actions_due > 0)
    out["doc_type_ok"] = int(reading.document_type.value in DOC_TYPE_ACCEPT.get(gt["document_type"], {gt["document_type"]}))
    return out


# ------------------------------------------------------------ mock oracle -----
def oracle_backend(data_dir: Path, noise: float, hallucinate: float, seed: int) -> MockBackend:
    """A mock 'model' that knows the answer key, optionally corrupted:
    noise: per-character substitution prob in the transcription (simulated OCR error)
    hallucinate: prob that the extractor invents a wrong amount/deadline
    """
    rng = random.Random(seed)
    gts = {p.stem: json.loads(p.read_text()) for p in data_dir.glob("*.json") if p.name != "manifest.json" and not p.name.startswith("results_")}
    current = {"id": None}

    def transcriber(img):
        gt = gts[current["id"]]
        t = gt["page_text"]
        if noise:
            t = "".join(c if rng.random() > noise or c in "\n " else rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for c in t)
        return t

    def extractor(img, transcription):
        gt = gts[current["id"]]
        facts = [dict(f) for f in gt["facts"]]
        actions = [{"text": "Pay the amount shown", "due": None}]
        for f in facts:
            if f["kind"] == "deadline":
                actions[0]["due"] = f["value"]
        if hallucinate and rng.random() < hallucinate:
            victim = rng.choice([f for f in facts if f["kind"] in ("money", "deadline", "reference")] or facts)
            if victim["kind"] == "money":
                victim["value"] = f"${rng.randint(100, 9000)}.00"
            elif victim["kind"] == "deadline":
                victim["value"] = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
                actions[0]["due"] = victim["value"]
            else:
                victim["value"] = "REF-2026-0000000"
        return {"document_type": gt["document_type"], "source_language": gt["language"], "summary": "…",
                "key_facts": facts, "actions": actions}

    b = MockBackend(transcriber, extractor)
    b.current = current
    return b


# ----------------------------------------------------------------- runner -----
def run(args):
    data = Path(args.data)
    gts = sorted(p for p in data.glob("*.json") if p.name != "manifest.json" and not p.name.startswith("results_"))
    if args.limit:
        gts = gts[: args.limit]
    if args.backend == "mock":
        backend = oracle_backend(data, args.noise, args.hallucinate, args.seed)
    else:
        backend = make_backend(args.backend)
    engine = ReadForMe(backend, max_pixels=args.max_pixels)

    rows = []
    for k, gp in enumerate(gts, 1):
        gt = json.loads(gp.read_text())
        if hasattr(backend, "current"):
            backend.current["id"] = gp.stem
        img = (data / gt["image"]).read_bytes()
        t0 = time.perf_counter()
        reading, timings = engine.read(img, target_language=args.lang)
        s = score_sample(gt, reading)
        s.update({"id": gt["id"], "language": gt["language"], "perturbation": gt["perturbation"],
                  "document_type": gt["document_type"], "latency_s": time.perf_counter() - t0,
                  "visual_tokens_est": timings.visual_tokens_est,
                  "verifier_dropped_due": reading.verification.dropped_actions_due,
                  # what the model actually produced, for diagnosing each metric
                  "pred_document_type": reading.document_type.value,
                  "transcription": reading.transcription,
                  "pred_facts": [{"label": f.label, "value": f.value, "kind": f.kind.value,
                                  "verified": f.verified, "note": f.note} for f in reading.key_facts],
                  "pred_actions": [{"text": a.text, "due": a.due, "verified": a.verified} for a in reading.actions],
                  "gt_facts": gt["facts"],
                  "missed_gt_facts": [f for f in gt["facts"] if f["kind"] in {k.value for k in CRITICAL_KINDS}
                                      and not (tokens_for(f["kind"], f["value"]) & pred_tokens(reading))],
                  "hallucinated": [f.value for f in reading.key_facts if f.kind in CRITICAL_KINDS
                                   and not (tokens_for(f.kind.value, f.value) & page_tokens(gt["page_text"]))]})
        rows.append(s)
        print(f"[{k}/{len(gts)}] {gt['id']}  {s['latency_s']:.1f}s  cer={s['cer']:.3f}  recall={s['fact_recall']:.2f}  "
              f"halluc={s['hallucinated_facts']}", file=sys.stderr, flush=True)
        if args.verbose:
            brief = {k: v for k, v in s.items() if k not in ("transcription", "pred_facts", "pred_actions", "gt_facts")}
            print(json.dumps(brief, ensure_ascii=False))

    report = aggregate(rows)
    report["config"] = vars(args)
    out = Path(args.out) if args.out else data / f"results_{args.backend}.json"
    out.write_text(json.dumps({"summary": report, "rows": rows}, ensure_ascii=False, indent=1))
    print_report(report)
    print(f"\nwrote {out}")




def aggregate(rows: list[dict]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.fmean(xs), 4) if xs else None

    def group(key):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        return {k: summarize(v) for k, v in sorted(g.items())}

    def summarize(rs):
        return {
            "n": len(rs),
            "cer": mean(r["cer"] for r in rs),
            "fact_recall": mean(r["fact_recall"] for r in rs),
            "hallucinated_facts_per_doc": mean(r["hallucinated_facts"] for r in rs),
            "verifier_catch_rate": mean(r["verifier_catch"] for r in rs),
            "verifier_false_flag_rate": mean(r["verifier_false_flag"] for r in rs),
            "fabricated_deadline_rate_raw": mean(r["fabricated_deadline_raw"] for r in rs),
            "fabricated_deadline_rate_shown": mean(r["fabricated_deadline_shown"] for r in rs),
            "doc_type_acc": mean(r["doc_type_ok"] for r in rs),
            "latency_s_p50": round(statistics.median(r["latency_s"] for r in rs), 3),
        }

    return {"overall": summarize(rows), "by_language": group("language"),
            "by_perturbation": group("perturbation"), "by_document_type": group("document_type")}


def print_report(rep: dict):
    def line(name, s):
        print(f"{name:<22} n={s['n']:<3} CER={s['cer']:.3f}  recall={s['fact_recall']:.3f}  "
              f"halluc/doc={s['hallucinated_facts_per_doc']:.2f}  catch={s['verifier_catch_rate']}  "
              f"false_flag={s['verifier_false_flag_rate']}  fab_due raw/shown={s['fabricated_deadline_rate_raw']:.2f}/{s['fabricated_deadline_rate_shown']:.2f}  "
              f"p50={s['latency_s_p50']}s")
    print("== overall"); line("all", rep["overall"])
    for k in ("by_language", "by_perturbation", "by_document_type"):
        print(f"== {k}")
        for name, s in rep[k].items():
            line(name, s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="eval/data/synth")
    p.add_argument("--backend", default="mock", help="mock | transformers | openai")
    p.add_argument("--lang", default="en")
    p.add_argument("--max-pixels", type=int, default=1654 * 2339)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    # mock-only knobs
    p.add_argument("--oracle", action="store_true", help="mock: perfect model")
    p.add_argument("--noise", type=float, default=0.0, help="mock: OCR char-noise prob")
    p.add_argument("--hallucinate", type=float, default=0.0, help="mock: prob of an invented critical fact")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
