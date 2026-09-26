# ReadForMe

**Verified document reading for people who can't read the page**, built on Cohere's
[North Micro Vision Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct)
(2.4B, Apache 2.0) and packaged as a tool for [cohere-toolkit](https://github.com/cohere-ai/cohere-toolkit).

Photograph a government letter, bill, lease, prescription, school notice or sign. Get its type,
a plain-language summary in your language, and every critical fact — amount, date, deadline,
dosage, reference number — **each marked verified or not confirmed on the page**.

Apps that do the first half already exist (WhatLetter, ExplainThisDoc, the `legible` repo,
Google Lens). None of them verify what they tell you, and none report how often they are wrong.
That is what this project adds:

1. **A grounding verifier.** The VLM is asked twice: once to transcribe the page verbatim, once to
   extract structured facts. Every critical fact is then checked against the transcription with
   locale-aware normalisation (`$1,234.50` ≡ `1 234,50 €` ≡ `1234.50`; `26 septembre 2026` ≡
   `2026-09-26` ≡ `2026年9月26日`; `500 毫克` ≡ `500 mg`). Facts that fail are kept but flagged, a
   warning is added, and any *invented deadline* attached to an action is removed before the user
   sees it. Fuzzy text similarity is deliberately not allowed to rescue a critical fact — "October 5"
   is 94% similar to "October 15" and exactly the error this exists to catch.
2. **An evaluation harness with an answer key.** Synthetic civic documents (tax notice, prescription,
   utility bill) in en/fr/es/de/zh, rendered clean and with phone-photo perturbations (rotation,
   glare, blur, JPEG), with every critical fact known. Metrics: transcription CER, critical-fact
   recall, hallucinated facts per document, verifier catch rate, verifier false-flag rate, fabricated
   deadlines *before vs after* verification, and latency vs pixel budget.
3. **A cohere-toolkit integration.** Images become first-class uploads; the verified reading is what
   the Command model sees; each fact is a separately citeable tool result whose title carries its
   verification state.

## Status (2026-09-26)

| Component | State |
|---|---|
| `service/readforme` pipeline, verifier, normalisation | implemented, 16 unit tests passing |
| FastAPI service (`POST /read`) + CLI | implemented, tested with mock backend |
| Transformers backend (model-card API, `transformers==5.16.0`) | implemented, **not yet run on a GPU** |
| vLLM / OpenAI-compatible backend | implemented, not yet run against a live server |
| Synthetic eval set + harness | implemented; 60-sample set generated; oracle and corrupted-model runs pass |
| **Real-model numbers** | **none yet** — see "First experiment" |
| cohere-toolkit integration (tool, upload hook, installer, compose overlay) | implemented; installer tested against the upstream checkout; not yet run end-to-end in Docker |

Nothing here claims a measured accuracy for North Micro Vision. The harness exists so that number
can be produced honestly.

### What the harness already shows (mock backend, 60 synthetic docs)

| model | CER | fact recall | verifier catch | false flag | fabricated deadline raw → shown |
|---|---|---|---|---|---|
| oracle (perfect) | 0.000 | 1.000 | – | 0.000 | 0.00 → 0.00 |
| 5% OCR char noise + 30% invented fact | 0.046 | 0.919 | **1.000** | 0.440 | 0.37 → **0.00** |

Read the second row as the design tension, not a result: with noisy transcription the verifier
catches every invented fact and never shows a fabricated deadline, but it also refuses 44% of
correct facts because their digits were corrupted in the transcription. The real question — where
North Micro Vision sits on that curve for photographed documents, per language and per pixel budget
— is what the first GPU run answers.

## Layout

```
service/               the ML core (pip-installable, no toolkit dependency)
  readforme/
    schema.py          DocumentReading / KeyFact / FactKind (critical kinds are gated)
    normalize.py       digits across scripts, money, dates (en/fr/es/pt/de/it/zh), phones, refs, dosage units
    verify.py          the grounding verifier
    imaging.py         EXIF fix + pixel budget (keeps prompts inside the model's validated 8K tokens)
    prompts.py         transcribe / extract / explain prompts (versioned)
    backends.py        TransformersBackend (model-card exact) · OpenAICompatBackend (vLLM) · MockBackend
    pipeline.py        transcribe -> extract -> verify -> (explain)
    server.py, cli.py
  eval/
    synth/generate.py  synthetic civic documents with answer keys and photo perturbations
    run_eval.py        metrics, per language / perturbation / document type
  tests/
toolkit/               drop-in files for cohere-toolkit + compose overlay
scripts/install_into_toolkit.py
docs/toolkit_integration.md
```

## Quick start (service only, no toolkit)

```bash
cd service
pip install -e ".[dev]"
python -m pytest -q                                  # 16 tests, no model needed

python -m eval.synth.generate --out eval/data/synth --n 60
python -m eval.run_eval --backend mock --oracle       # sanity
python -m eval.run_eval --backend mock --noise 0.05 --hallucinate 0.3

# with the real model (GPU, ~8 GB bf16):
pip install -e ".[transformers]"                     # pins transformers==5.16.0 per the model card
readforme read photo.jpg --lang es --backend transformers
python -m eval.run_eval --backend transformers --max-pixels 3868706

# or serve it:
READFORME_BACKEND=transformers uvicorn readforme.server:app --port 8090
curl -F image=@photo.jpg -F target_language=fr localhost:8090/read
```

vLLM instead of transformers: `vllm serve CohereLabs/North-Micro-Vision-Instruct --port 8001`, then
`READFORME_BACKEND=openai READFORME_VLM_URL=http://localhost:8001/v1`.

## Quick start (inside cohere-toolkit)

```bash
git clone https://github.com/cohere-ai/cohere-toolkit
python scripts/install_into_toolkit.py ../cohere-toolkit
cd ../cohere-toolkit
docker compose -f docker-compose.yml -f docker-compose.readforme.yml up --build
```

Then upload a JPG/PNG/WEBP in a conversation and ask "what is this letter and what do I need to do?".
Details and a manual-patch fallback: [docs/toolkit_integration.md](docs/toolkit_integration.md).

## First experiment (the one that matters)

Run `eval/run_eval.py` with `--backend transformers` on the synthetic set at three pixel budgets
(1.0 MP, 2.0 MP, 3.87 MP). Report, per language and perturbation level:

* CER on digits specifically (add `--verbose` and post-process; digits are what the verifier keys on)
* verifier catch rate vs false-flag rate
* fabricated-deadline rate before/after
* p50 latency and estimated visual tokens

Then collect 30–50 real, consented, redacted phone photos into `eval/data/real/` using the same JSON
schema and rerun. That second run is the number nobody has published for this model.

## Licences

Code: Apache 2.0. Model: North Micro Vision Instruct is Apache 2.0. The optional wider-language
plugins discussed in the design (Cohere's `tiny-aya` and `North-Small-Translate`) are **CC-BY-NC 4.0**
and are not wired in; the VLM-only path covers en, de, fr, es, it, pt, hi, ja, ko, zh, ar.
