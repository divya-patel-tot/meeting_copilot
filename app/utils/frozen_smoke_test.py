"""Headless smoke test for the PyInstaller bundle (--smoke-test)."""

from __future__ import annotations

import sys
import traceback

from app.utils.paths import BASE_DIR, MODELS_DIR, SILERO_VAD_FILENAME, resource_path

_THRESHOLD = 0.55
_CASES = (
    {
        "label": "sla",
        "query": "What's your SLA for critical incidents?",
        "expect_source": "test_acme_sla.txt",
        "min_score": _THRESHOLD,
    },
    {
        "label": "weather",
        "query": "What do you think about the weather today?",
        "expect_source": None,
        "max_score": _THRESHOLD,
    },
    {
        "label": "office",
        "query": "Where is your office located?",
        "expect_source": "test_office_locations.txt",
        "min_score": _THRESHOLD,
    },
)


def _log(lines: list[str], message: str) -> None:
    lines.append(message)
    print(message)


def run_smoke_test() -> int:
    lines: list[str] = []
    failures = 0

    _log(lines, f"frozen={getattr(sys, 'frozen', False)}")
    _log(lines, f"base_dir={BASE_DIR}")
    _log(lines, f"models_dir={MODELS_DIR}")

    vad_path = MODELS_DIR / SILERO_VAD_FILENAME
    if not vad_path.exists():
        _log(lines, f"FAIL: bundled VAD missing at {vad_path}")
        failures += 1
    else:
        _log(lines, f"OK: bundled VAD found ({vad_path.stat().st_size} bytes)")

    qss_path = resource_path("app", "ui", "styles", "dark_theme.qss")
    if qss_path.exists():
        _log(lines, f"OK: stylesheet found at {qss_path}")
    else:
        _log(lines, f"FAIL: stylesheet missing at {qss_path}")
        failures += 1

    try:
        from app.core.audio.vad import SileroVAD

        vad = SileroVAD(str(vad_path))
        prob = vad.get_speech_probability(
            __import__("numpy").zeros(SileroVAD.CHUNK_SIZE, dtype="float32")
        )
        _log(lines, f"OK: SileroVAD inference prob={prob:.4f}")
    except Exception as exc:
        failures += 1
        _log(lines, f"FAIL: SileroVAD — {exc}")
        _log(lines, traceback.format_exc())

    try:
        from app.core.rag.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        stats = kb.get_stats()
        _log(lines, f"OK: KnowledgeBase loaded chunks={stats['total_chunks']}")

        for case in _CASES:
            matches = kb.query_candidates(case["query"], top_k=4)
            top = matches[0] if matches else None
            top_score = top["score"] if top else 0.0
            top_source = top["source"] if top else None
            _log(
                lines,
                f"  [{case['label']}] top={top_source} score={top_score:.2f}",
            )

            if "min_score" in case:
                if top_score < case["min_score"]:
                    failures += 1
                    _log(
                        lines,
                        f"FAIL: {case['label']} score {top_score:.2f} "
                        f"< {case['min_score']}",
                    )
                elif case.get("expect_source") and top_source != case["expect_source"]:
                    failures += 1
                    _log(
                        lines,
                        f"FAIL: {case['label']} expected source "
                        f"{case['expect_source']}, got {top_source}",
                    )
                else:
                    _log(lines, f"OK: {case['label']} retrieval")

            if "max_score" in case:
                above = [m for m in matches if m["score"] >= case["max_score"]]
                if above:
                    failures += 1
                    _log(
                        lines,
                        f"FAIL: {case['label']} had {len(above)} match(es) "
                        f">= {case['max_score']}",
                    )
                else:
                    _log(lines, f"OK: {case['label']} no relevant retrieval")
    except Exception as exc:
        failures += 1
        _log(lines, f"FAIL: KnowledgeBase — {exc}")
        _log(lines, traceback.format_exc())

    summary = "PASS" if failures == 0 else f"FAIL ({failures} checks)"
    _log(lines, f"SUMMARY: {summary}")

    log_path = BASE_DIR / "smoke_test.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(lines, f"log written to {log_path}")

    return 1 if failures else 0
