# Evaluation

Every sample is an image plus a JSON answer key:

```json
{"id": "...", "image": "....jpg", "language": "fr", "document_type": "bill_or_invoice",
 "perturbation": "photo", "page_text": "full verbatim text of the page",
 "facts": [{"label": "Amount due", "value": "1 234,50 €", "kind": "money"}, ...],
 "deadline_iso": "2026-10-15"}
```

`eval/data/synth/` is generated (`python -m eval.synth.generate`). `eval/data/real/` is for
consented, redacted phone photos of real documents using the same schema; never commit personal
data — redact names, addresses and identifiers on the image *and* in `page_text`, and keep the
redaction consistent so the answer key still matches the page.

Metrics are defined at the top of `run_eval.py`. Results are written to `results_<backend>.json`
next to the data with per-sample rows for plotting.
