#!/usr/bin/env python3
"""Generate read-only Gemini reviews for physical grasp and stack images."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "demo" / "episode_evidence"
OUTPUT_PATH = EVIDENCE_DIR / "gemini_evaluations.json"


def load_settings() -> tuple[str, str]:
    for raw_line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return key, model


def evaluate_image(path: Path, kind: str, *, api_key: str, model: str) -> dict:
    subject = (
        "whether the robot gripper visibly holds a cube after pickup"
        if kind == "grasp"
        else "whether the red cube is visibly and stably placed on top of the blue cube"
    )
    prompt = (
        "You are reviewing one physical FANUC robot side-camera checkpoint. "
        f"Evaluate only {subject}. Do not infer events outside the image. "
        "Return good only when the visual evidence is clear, bad when failure is clear, "
        "and uncertain when the checkpoint cannot prove the result. Keep the reason under "
        "18 words."
    )
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["good", "bad", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "confidence", "reason"],
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseFormat": {
                "text": {"mimeType": "APPLICATION_JSON", "schema": schema}
            },
        },
    }
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + quote(model, safe="")
        + ":generateContent"
    )
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as error:
        raise RuntimeError(f"Gemini returned HTTP {error.code} for {path.name}") from error
    raw = json.loads(body["candidates"][0]["content"]["parts"][0]["text"])
    verdict = str(raw["verdict"]).lower()
    confidence = min(1.0, max(0.0, float(raw["confidence"])))
    reason = " ".join(str(raw["reason"]).split())[:180]
    if verdict not in {"good", "bad", "uncertain"} or not reason:
        raise RuntimeError(f"Gemini returned an invalid review for {path.name}")
    return {"verdict": verdict, "confidence": round(confidence, 3), "reason": reason}


def main() -> int:
    api_key, model = load_settings()
    episodes: dict[str, dict] = {}
    for episode in range(1, 12):
        episodes[str(episode)] = {}
        for kind in ("grasp", "stack"):
            path = EVIDENCE_DIR / f"episode_{episode}_{kind}_side.jpg"
            print(f"Evaluating episode {episode} {kind}...", flush=True)
            episodes[str(episode)][kind] = evaluate_image(
                path, kind, api_key=api_key, model=model
            )
    output = {
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Advisory image review only; human validation remains authoritative.",
        "episodes": episodes,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
