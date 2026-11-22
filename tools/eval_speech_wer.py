"""Reproducible multi-seed speech intelligibility evaluator using Whisper ASR.

Scores generated WAV clips against reference dialogue text using normalized
Word Error Rate (WER) for whitespace-delimited languages and Character Error
Rate (CER) for non-spaced scripts (Japanese, Chinese, Korean).

Input manifest JSON format:
[
  {
    "group": "es_seville",
    "variant": "v4.3",
    "seed": 103,
    "language": "es",
    "reference": "¡Qué alegría volver a verte después de tantos años!",
    "wav": "/path/to/es_seville_v43_s103.wav"
  }
]

Usage:
    python tools/eval_speech_wer.py --manifest eval_manifest.json --out wer_report.json
"""

import argparse
import json
import os
import re
import unicodedata
import wave

import numpy as np


def _load_wav_16k(wav_path):
    """Read PCM WAV directly via stdlib wave (or soundfile fallback) and resample to 16 kHz float32.

    Avoids faster-whisper's internal PyAV decode_audio() call, which fails under PyAV >= 14
    where av.open(..., metadata_errors=...) was removed.
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sw == 1:
            audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported WAV sample width {sw}")
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)
    except Exception:
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

    if sr != 16000 and audio.size > 0:
        n_out = max(1, int(round(audio.shape[0] * 16000.0 / float(sr))))
        x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


_NON_SPACED_LANGS = frozenset(("ja", "zh", "yue"))
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(text, language="en"):
    """Lowercase, strip punctuation, normalize Unicode NFKC, and collapse whitespace."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if language in _NON_SPACED_LANGS:
        s = s.replace(" ", "")
    return s


def edit_distance(seq_a, seq_b):
    """Standard Levenshtein edit distance between two token lists."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def compute_error_rate(reference, hypothesis, language="en"):
    """Return (metric_name, error_rate, edits, ref_len, norm_ref, norm_hyp)."""
    norm_ref = normalize_text(reference, language=language)
    norm_hyp = normalize_text(hypothesis, language=language)
    if language in _NON_SPACED_LANGS:
        ref_units = list(norm_ref)
        hyp_units = list(norm_hyp)
        metric = "cer"
    else:
        ref_units = norm_ref.split() if norm_ref else []
        hyp_units = norm_hyp.split() if norm_hyp else []
        metric = "wer"
    edits = edit_distance(ref_units, hyp_units)
    denom = max(1, len(ref_units))
    return metric, edits / float(denom), edits, len(ref_units), norm_ref, norm_hyp


def load_transcriber(model_name, device="cuda"):
    """Load faster-whisper if installed (auto-falling back to CPU int8 on CUDA lib mismatch),
    else openai-whisper."""
    try:
        from faster_whisper import WhisperModel

        def _make_fw(dev):
            compute_type = "float16" if dev == "cuda" else "int8"
            return WhisperModel(model_name, device=dev, compute_type=compute_type)

        state = {"dev": device, "model": _make_fw(device)}

        def _transcribe_fw(wav_path, language):
            lang_arg = None if (not language or language == "auto") else language
            audio = _load_wav_16k(wav_path)
            try:
                segments, _ = state["model"].transcribe(audio, language=lang_arg, beam_size=5, vad_filter=False)
                return " ".join(seg.text.strip() for seg in segments).strip()
            except RuntimeError as err:
                if state["dev"] == "cuda" and ("libcublas" in str(err).lower() or "cublas" in str(err).lower()):
                    print("[WER-EVAL] faster-whisper CUDA library mismatch; falling back to CPU int8", flush=True)
                    state["dev"] = "cpu"
                    state["model"] = _make_fw("cpu")
                    segments, _ = state["model"].transcribe(audio, language=lang_arg, beam_size=5, vad_filter=False)
                    return " ".join(seg.text.strip() for seg in segments).strip()
                raise

        return "faster-whisper:" + model_name, _transcribe_fw
    except ImportError:
        import whisper

        model = whisper.load_model(model_name, device=device)

        def _transcribe_ow(wav_path, language):
            kwargs = {"fp16": (device == "cuda")}
            if language and language != "auto":
                kwargs["language"] = language
            res = model.transcribe(wav_path, **kwargs)
            return str(res.get("text", "")).strip()

        return "openai-whisper:" + model_name, _transcribe_ow


def evaluate_manifest(manifest, transcriber_fn):
    """Run transcription and WER/CER scoring over every manifest entry and aggregate by (group, variant)."""
    rows = []
    agg = {}
    for item in manifest:
        wav_path = item["wav"]
        lang = str(item.get("language", "en")).lower()
        ref = str(item.get("reference", ""))
        hyp = transcriber_fn(wav_path, lang)
        metric, rate, edits, ref_len, nref, nhyp = compute_error_rate(ref, hyp, language=lang)
        row = {
            "group": item.get("group", ""),
            "variant": item.get("variant", ""),
            "seed": item.get("seed"),
            "language": lang,
            "metric": metric,
            "error_rate": round(rate, 4),
            "edits": edits,
            "ref_len": ref_len,
            "reference": ref,
            "hypothesis": hyp,
            "norm_reference": nref,
            "norm_hypothesis": nhyp,
            "wav": os.path.basename(wav_path),
        }
        rows.append(row)
        key = (row["group"], row["variant"], lang, metric)
        bucket = agg.setdefault(key, {"rates": [], "edits": 0, "ref_len": 0})
        bucket["rates"].append(rate)
        bucket["edits"] += edits
        bucket["ref_len"] += ref_len

    summary = []
    for (group, variant, lang, metric), b in sorted(agg.items()):
        n = len(b["rates"])
        mean_rate = sum(b["rates"]) / max(1, n)
        pooled_rate = b["edits"] / float(max(1, b["ref_len"]))
        summary.append({
            "group": group,
            "variant": variant,
            "language": lang,
            "metric": metric,
            "n_seeds": n,
            "mean_error_rate": round(mean_rate, 4),
            "pooled_error_rate": round(pooled_rate, 4),
            "per_seed": [round(r, 4) for r in b["rates"]],
        })
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Score generated WAV speech clips with Whisper WER/CER.")
    parser.add_argument("--manifest", required=True, help="JSON manifest describing WAV clips and references")
    parser.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--out", default="", help="Optional path to save full JSON report")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    backend, fn = load_transcriber(args.model, device=args.device)
    rows, summary = evaluate_manifest(manifest, fn)

    print(f"[WER-EVAL] Backend: {backend} | Clips: {len(rows)}", flush=True)
    for s in summary:
        print(
            f"  {s['group']:16s} | {s['variant']:12s} | {s['language']:4s} ({s['metric'].upper()}) | "
            f"n={s['n_seeds']} | mean={s['mean_error_rate']:.4f} | pooled={s['pooled_error_rate']:.4f} | "
            f"seeds={s['per_seed']}",
            flush=True,
        )

    report = {"backend": backend, "summary": summary, "clips": rows}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[WER-EVAL] Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
