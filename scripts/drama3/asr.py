#!/usr/bin/env python3
"""Actual CPU ASR with pinned Whisper checkpoint; no claimed listening. SPDX-License-Identifier: MIT."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from importlib.metadata import version

MODEL = "Systran/faster-whisper-small"
REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def words(text):
    return re.findall(r"\w+", re.sub(r"\[[^\]]*\]", "", text).casefold())


def word_error_rate(expected, actual):
    left, right = words(expected), words(actual)
    previous = list(range(len(right) + 1))
    for index, source in enumerate(left, 1):
        current = [index]
        for other, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other] + 1, previous[other - 1] + (source != target)))
        previous = current
    return previous[-1] / max(1, len(left))


def tail_matches(expected, actual):
    return words(expected)[-3:] == words(actual)[-3:]


def main():
    from faster_whisper import WhisperModel
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--names", default="english-1,english-2,german-1,german-2,expression,streaming,longer,overlap-a,overlap-b,reference,post-restart")
    args = parser.parse_args()
    model = WhisperModel(str(args.model_dir), device="cpu", compute_type="int8", cpu_threads=6, num_workers=1)
    rows = []
    for name in args.names.split(","):
        receipt = json.loads((args.evidence / (name + ".json")).read_text())
        file = args.evidence / receipt["audio_file"]
        segments, info = model.transcribe(str(file), language="de" if name.startswith("german") else "en", beam_size=5, vad_filter=False, condition_on_previous_text=False)
        text = " ".join(segment.text.strip() for segment in segments)
        wer = word_error_rate(receipt["request"]["text"], text)
        tail = tail_matches(receipt["request"]["text"], text)
        row = dict(name=name, text=text, expected=receipt["request"]["text"], wer=wer, tail_matches=tail, language=info.language, audio_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), passed=wer <= 0.4 and tail)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = dict(model=MODEL, revision=REVISION, faster_whisper=version("faster-whisper"), ctranslate2=version("ctranslate2"), device="cpu", compute_type="int8", beam_size=5, vad_filter=False, rows=rows, passed=all(row["passed"] for row in rows))
    (args.evidence / "asr.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
