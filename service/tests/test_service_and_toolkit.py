"""API smoke test with the mock backend + toolkit render/parse round trip."""
import importlib.util, io, json, os, sys, types
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

PAGE = "City Pharmacy\nPrescription RX-12345678\nAmoxicillin 500 mg tablets\nTake 1 tablet 3 times a day for 7 days."


def _mock_engine(monkeypatch):
    from readforme import server
    from readforme.backends import MockBackend
    from readforme.pipeline import ReadForMe

    def extractor(img, t):
        return {"document_type": "prescription", "source_language": "en", "summary": "An antibiotic prescription.",
                "key_facts": [{"label": "Medication", "value": "Amoxicillin 500 mg", "kind": "dosage"},
                              {"label": "Frequency", "value": "3 times a day", "kind": "dosage"},
                              {"label": "Duration", "value": "10 days", "kind": "dosage"}],  # wrong: page says 7
                "actions": [{"text": "Take with food", "due": None}]}
    server._engine.cache_clear()
    monkeypatch.setattr(server, "_engine", lambda: ReadForMe(MockBackend(lambda i: PAGE, extractor)))
    return server


def test_read_endpoint(monkeypatch):
    server = _mock_engine(monkeypatch)
    c = TestClient(server.app)
    buf = io.BytesIO(); Image.new("RGB", (300, 400), "white").save(buf, "PNG")
    r = c.post("/read", files={"image": ("rx.png", buf.getvalue(), "image/png")}, data={"target_language": "es"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_type"] == "prescription"
    facts = {f["label"]: f for f in body["key_facts"]}
    assert facts["Medication"]["verified"] is True
    assert facts["Duration"]["verified"] is False and "10 days" in body["warnings"][0]
    assert "timings" in body


def _load_toolkit_upload_module():
    """Import toolkit/src/backend/services/readforme_upload.py with a stubbed backend.config.settings."""
    stub_pkg = types.ModuleType("backend"); stub_cfg = types.ModuleType("backend.config"); stub_settings = types.ModuleType("backend.config.settings")
    class Settings:
        def get(self, k): return {"tools.read_this_for_me.url": "http://x"}.get(k)
    stub_settings.Settings = Settings
    sys.modules.update({"backend": stub_pkg, "backend.config": stub_cfg, "backend.config.settings": stub_settings})
    p = Path(__file__).resolve().parents[2] / "toolkit" / "src" / "backend" / "services" / "readforme_upload.py"
    spec = importlib.util.spec_from_file_location("readforme_upload", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_toolkit_render_roundtrip(monkeypatch):
    server = _mock_engine(monkeypatch)
    c = TestClient(server.app)
    buf = io.BytesIO(); Image.new("RGB", (300, 400), "white").save(buf, "PNG")
    reading = c.post("/read", files={"image": ("rx.png", buf.getvalue(), "image/png")}).json()
    up = _load_toolkit_upload_module()
    text = up.render_reading(reading)
    assert "NOT CONFIRMED ON PAGE" in text and "## Page transcription" in text
    back = up.extract_reading_json(text)
    assert back["key_facts"][2]["verified"] is False
    assert up.is_image_extension("JPG") and not up.is_image_extension("pdf")
