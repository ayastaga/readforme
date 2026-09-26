"""readforme read photo.jpg --lang es [--backend transformers|openai|mock]"""
from __future__ import annotations

import argparse
import json
import sys

from .backends import make_backend
from .imaging import MAX_PIXELS
from .pipeline import ReadForMe


def main(argv=None):
    p = argparse.ArgumentParser(prog="readforme")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="read one image")
    r.add_argument("image")
    r.add_argument("--lang", default="en", help="target language (BCP-47)")
    r.add_argument("--backend", default=None, help="transformers | openai | mock (default: $READFORME_BACKEND or openai)")
    r.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    r.add_argument("--explain", action="store_true", help="third pass: rewrite summary in plain language")
    a = p.parse_args(argv)
    engine = ReadForMe(make_backend(a.backend), max_pixels=a.max_pixels, explain=a.explain)
    with open(a.image, "rb") as f:
        reading, t = engine.read(f.read(), target_language=a.lang)
    out = reading.model_dump()
    out["timings"] = t.__dict__
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
