"""Remote external text/vision encoder for ComfyUI (Universal Conditioner).

Holds zero encoder weights in ComfyUI VRAM: forwards prompt text and optional reference
images to a resident external LLM/VLM server exposing POST /v1/hidden_states, aligns
subword spans from the source tokenizer to the target diffusion encoder's vocabulary,
optionally appends zero-VRAM mmapped target token embeddings, and projects into the
target model's conditioning space.
"""

import base64
import http.client
import io
import json
import logging
import os
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import torch
from PIL import Image

try:
    from . import projector
except ImportError:
    import projector

DEFAULT_URL = "http://127.0.0.1:8000"
API_KEY_ENV = "UNICOND_API_KEY"
QI21_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROFILES = ("auto", "minimax_h3", "qwen_image_21")

_HTTP_LOCAL = threading.local()
_TARGET_TOK_CACHE = {}
_EMBED_MMAP_CACHE = {}
_SPECIAL_PRESERVE = frozenset((
    "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|image_pad|>", "<|video_pad|>", "<|endoftext|>",
))


def encode_image_data_url(image, quality=92):
    """Encode a ComfyUI image tensor [H, W, C] or [1, H, W, C] in [0, 1] as a JPEG data URL."""
    t = image[0] if image.dim() == 4 else image
    arr = (t.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.shape[-1] == 4:
        pil = Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(arr[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def build_parts(text, images=(), minimax_ref_items=None, picture_label="<Picture %d>: "):
    """Build OpenAI-style multimodal content parts matching the target prompt layout."""
    parts = []

    def add_text(s):
        if not s:
            return
        if parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += s
        else:
            parts.append({"type": "text", "text": s})

    def add_image(tensor):
        parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(tensor)}})

    if minimax_ref_items:
        counters = {"image": 0, "audio": 0, "video": 0}
        for item in minimax_ref_items:
            kind = item["type"]
            counters[kind] = counters.get(kind, 0) + 1
            if kind == "image":
                add_text(picture_label % counters["image"])
                add_image(item["data"])
            elif kind == "audio":
                add_text("<Audio %d>: " % counters["audio"])
            else:
                frames = item["data"]
                stamps = item.get("timestamps")
                stamps = list(stamps) if stamps is not None else [i / 2.0 for i in range(frames.shape[0])]
                add_text("<Video %d>: " % counters["video"])
                for k in range(frames.shape[0]):
                    add_text("<%.1f seconds>" % stamps[k])
                    add_image(frames[k])
    else:
        for i, img in enumerate(images or ()):
            prefix = " " if (i > 0 and picture_label == "<image%d>") else ""
            add_text(prefix + (picture_label % (i + 1)))
            add_image(img)

    add_text(text or "")
    if not parts:
        parts.append({"type": "text", "text": " "})
    return parts


def decode_hidden_response(raw):
    """Unpack [u64 hdr_len][JSON header][LE BF16 payload] binary response."""
    if len(raw) < 8:
        raise ValueError("hidden_states response truncated before header length (%d bytes)" % len(raw))
    header_len = struct.unpack("<Q", raw[:8])[0]
    if len(raw) < 8 + header_len:
        raise ValueError("hidden_states response truncated inside header (%d < %d)" % (len(raw), 8 + header_len))
    meta = json.loads(raw[8:8 + header_len].decode("utf-8"))
    seq_len = int(meta["seq_len"])
    num_layers = int(meta.get("num_layers", 1))
    hidden_size = int(meta["hidden_size"])
    feat_dim = num_layers * hidden_size
    expected_bytes = seq_len * feat_dim * 2
    payload = raw[8 + header_len:]
    if len(payload) != expected_bytes:
        raise ValueError(
            "hidden_states payload size mismatch: got %d bytes, expected %d (%dx%dx%d bf16)"
            % (len(payload), expected_bytes, seq_len, num_layers, hidden_size)
        )
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).view(1, seq_len, feat_dim).float()
    return meta, tensor


def find_user_span(token_texts, seq_len):
    """Locate (start, end) of user content tokens inside a Qwen chat-templated prompt."""
    if not token_texts or len(token_texts) != seq_len:
        return 0, seq_len
    start = 0
    if token_texts[0] == "<|im_start|>":
        acc = ""
        for idx in range(1, min(seq_len, 8)):
            acc += token_texts[idx]
            if acc.endswith("user\n"):
                start = idx + 1
                break
    end = seq_len
    for idx in range(start, seq_len):
        if token_texts[idx] == "<|im_end|>":
            end = idx
            break
    if end <= start:
        return 0, seq_len
    return start, end


def find_qi21_span(token_texts, seq_len):
    """Locate (start, end) matching Qwen-Image 2.1's encoder output:
    drops the system turn and trailing `<think>...</think>` block, keeping
    `<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n`.
    """
    if not token_texts or len(token_texts) != seq_len:
        return 0, seq_len
    im_starts = [i for i, t in enumerate(token_texts) if t == "<|im_start|>"]
    start = im_starts[1] if len(im_starts) > 1 else 0
    end = seq_len
    for i in range(start, seq_len):
        if token_texts[i] == "<think>":
            end = i
            break
    if end <= start:
        return 0, seq_len
    return start, end


def token_tags_from_types(seq_len, token_types, token_texts=None):
    """Convert server token_types (0=text, non-zero=vision) to H3 tags (0=vision, 1=text)."""
    if not token_types or len(token_types) != seq_len:
        return torch.ones(seq_len, dtype=torch.long)
    tags = torch.tensor([0 if int(t) != 0 else 1 for t in token_types], dtype=torch.long)
    if token_texts and len(token_texts) == seq_len:
        for i, tok in enumerate(token_texts):
            if tok in ("<|vision_start|>", "<|vision_end|>"):
                tags[i] = 0
    else:
        vis = (tags == 0).nonzero(as_tuple=False).flatten().tolist()
        for i in vis:
            if i > 0:
                tags[i - 1] = 0
            if i + 1 < seq_len:
                tags[i + 1] = 0
    return tags


def _spans_from_lens(lens):
    spans = []
    pos = 0
    for n in lens:
        n_int = int(n)
        spans.append((pos, pos + n_int))
        pos += n_int
    return spans


def unpack_by_spans_with_info(src_h, src_lens, tgt_lens, n_to_1="last"):
    """Map source hidden states [L_src, D] onto target token spans [L_tgt, D] by byte/char overlap.

    Returns (out [L_tgt, D], split_pos [L_tgt], split_len [L_tgt], non_1to1 [L_tgt]):
      - For 1:N unpacked groups (one source token covering M >= 2 target tokens),
        split_pos holds 1..M (1-based) and split_len holds M.
      - For 1:1 or N:1 tokens, split_pos = 0 and split_len = 0.
      - non_1to1 is True wherever source and target token boundaries do not match 1:1.
    """
    L_tgt = len(tgt_lens)
    L_src = len(src_lens)
    split_pos = torch.zeros(L_tgt, dtype=torch.long)
    split_len = torch.zeros(L_tgt, dtype=torch.long)
    non_1to1 = torch.zeros(L_tgt, dtype=torch.bool)
    if L_tgt == 0:
        return src_h.new_empty((0, src_h.shape[1])), split_pos, split_len, non_1to1
    if L_src == 0:
        return src_h.new_zeros((L_tgt, src_h.shape[1])), split_pos, split_len, non_1to1

    src_spans = _spans_from_lens(src_lens)
    tgt_spans = _spans_from_lens(tgt_lens)
    out = src_h.new_zeros((L_tgt, src_h.shape[1]))
    src_to_tgts = {}
    j_ptr = 0
    for j, (t0, t1) in enumerate(tgt_spans):
        if t1 <= t0:
            out[j] = src_h[min(j_ptr, L_src - 1)]
            continue
        while j_ptr < L_src and src_spans[j_ptr][1] <= t0:
            j_ptr += 1
        idxs = []
        k = j_ptr
        while k < L_src and src_spans[k][0] < t1:
            if src_spans[k][1] > t0:
                idxs.append(k)
            k += 1
        if idxs:
            if len(idxs) == 1:
                s_idx = idxs[0]
                out[j] = src_h[s_idx]
                src_to_tgts.setdefault(s_idx, []).append(j)
                if src_spans[s_idx] != (t0, t1):
                    non_1to1[j] = True
            else:
                out[j] = src_h[idxs[-1]] if n_to_1 == "last" else src_h[idxs].mean(dim=0)
                non_1to1[j] = True
        else:
            out[j] = src_h[min(j_ptr, L_src - 1)]
            non_1to1[j] = True

    for tgt_group in src_to_tgts.values():
        m = len(tgt_group)
        if m >= 2:
            for rank, j in enumerate(tgt_group):
                split_pos[j] = rank + 1
                split_len[j] = m
                non_1to1[j] = True

    return out, split_pos, split_len, non_1to1


def unpack_by_spans(src_h, src_lens, tgt_lens, n_to_1="last"):
    """Map source hidden states [L_src, D] onto target token spans [L_tgt, D]."""
    out, _, _, _ = unpack_by_spans_with_info(src_h, src_lens, tgt_lens, n_to_1=n_to_1)
    return out


def unpack_by_char_spans(src_h, src_texts, tgt_texts, n_to_1="mean"):
    """Character-span fallback wrapper around `unpack_by_spans`."""
    return unpack_by_spans(
        src_h, [len(s) for s in src_texts], [len(s) for s in tgt_texts], n_to_1=n_to_1
    )


def target_token_byte_lens(target_tok, token_ids):
    """Exact UTF-8 byte length per token id for Qwen2/Qwen3 byte-level BPE tokenizers."""
    added = getattr(target_tok, "_unicond_added_ids", None)
    if added is None:
        added = set(target_tok.get_added_vocab().values())
        target_tok._unicond_added_ids = added
    raw_toks = target_tok.convert_ids_to_tokens(token_ids)
    return [len(t.encode("utf-8")) if int(tid) in added else len(t) for tid, t in zip(token_ids, raw_toks)]


def get_target_tokenizer(profile):
    """Load the lightweight CPU-only target HuggingFace tokenizer for `profile` (0 B VRAM)."""
    tok = _TARGET_TOK_CACHE.get(profile)
    if tok is not None:
        return tok
    if profile == "qwen_image_21":
        from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
        tok = QwenImage21Tokenizer().qwen3vl_8b.tokenizer
    elif profile == "minimax_h3":
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
        tok = MiniMaxH3Tokenizer().qwen3vl_32b.tokenizer
    else:
        raise ValueError("Unsupported target tokenizer profile: %r" % (profile,))
    _TARGET_TOK_CACHE[profile] = tok
    return tok


def lookup_target_embeddings(embed_model_name, token_ids):
    """Zero-VRAM mmap lookup of target embed_tokens rows [L, D] from a safetensors encoder file."""
    from safetensors import safe_open
    import folder_paths

    entry = _EMBED_MMAP_CACHE.get(embed_model_name)
    if entry is None:
        path = (
            embed_model_name
            if os.path.isfile(embed_model_name)
            else folder_paths.get_full_path_or_raise("text_encoders", embed_model_name)
        )
        handle = safe_open(path, framework="pt", device="cpu")
        w_key = None
        for cand in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "embed_tokens.weight",
        ):
            if cand in handle.keys():
                w_key = cand
                break
        if w_key is None:
            raise KeyError("No embed_tokens.weight found in %s" % path)
        q_key = w_key[:-len(".weight")] + ".comfy_quant"
        if q_key in handle.keys():
            q_raw = bytes(handle.get_tensor(q_key).tolist()).decode("utf-8", errors="replace")
            q_meta = json.loads(q_raw)
            fmt = q_meta.get("format", "")
            if fmt not in ("int8_tensorwise", "float8_e4m3fn", ""):
                raise ValueError(
                    "Unsupported embed_tokens comfy_quant format %r in %s" % (fmt, path)
                )
        s_key = w_key + "_scale" if (w_key + "_scale") in handle.keys() else None
        w_slice = handle.get_slice(w_key)
        s_slice = handle.get_slice(s_key) if s_key else None
        vocab_size, dim = int(w_slice.get_shape()[0]), int(w_slice.get_shape()[1])
        entry = (handle, w_slice, s_slice, vocab_size, dim)
        _EMBED_MMAP_CACHE[embed_model_name] = entry

    _, w_slice, s_slice, vocab_size, dim = entry
    rows = []
    zero = torch.zeros(dim, dtype=torch.float32)
    for tid in token_ids:
        itid = int(tid)
        if 0 <= itid < vocab_size:
            r = w_slice[itid].float()
            if s_slice is not None:
                r = r * s_slice[itid].float()
            rows.append(r)
        else:
            rows.append(zero)
    return torch.stack(rows, dim=0) if rows else torch.zeros(0, dim, dtype=torch.float32)


def _extract_orig_text_segments(parts):
    segs = []
    buf = []
    for p in parts or ():
        if isinstance(p, dict) and p.get("type") == "text":
            buf.append(str(p.get("text", "")))
        else:
            if buf:
                segs.append("".join(buf))
                buf = []
    if buf:
        segs.append("".join(buf))
    return segs


def unpack_to_target_vocab(
    hidden_seq,
    src_texts,
    token_types,
    target_tok,
    embed_model=None,
    src_byte_lens=None,
    orig_text_segs=None,
    n_to_1="last",
    return_split_info=False,
):
    """Segment-wise unpack of [L_src, D] from source vocab into target tokenizer's exact tokens.

    Preserves special and vision tokens 1:1 while unpacking plain-text runs via exact UTF-8 byte
    spans (when `src_byte_lens` is returned by the server) or character spans as fallback.
    """
    L_src = hidden_seq.shape[0]
    use_bytes = src_byte_lens is not None and len(src_byte_lens) == L_src
    out_h = []
    out_texts = []
    out_types = []
    out_ids = []
    out_sp = []
    out_sl = []

    runs = []
    i = 0
    while i < L_src:
        is_special = (int(token_types[i]) != 0) or (src_texts[i] in _SPECIAL_PRESERVE)
        if is_special:
            runs.append(("special", i, i + 1))
            i += 1
        else:
            j = i
            while j < L_src and int(token_types[j]) == 0 and src_texts[j] not in _SPECIAL_PRESERVE:
                j += 1
            runs.append(("text", i, j))
            i = j

    text_run_count = sum(1 for kind, _, _ in runs if kind == "text")
    use_orig_segs = orig_text_segs is not None and len(orig_text_segs) == text_run_count
    orig_seg_idx = 0

    for kind, i, j in runs:
        if kind == "special":
            out_h.append(hidden_seq[i:i + 1])
            out_texts.append(src_texts[i])
            out_types.append(int(token_types[i]))
            out_sp.append(torch.zeros(1, dtype=torch.long))
            out_sl.append(torch.zeros(1, dtype=torch.long))
            if int(token_types[i]) != 0:
                out_ids.append(-1)
            else:
                tid = target_tok.convert_tokens_to_ids(src_texts[i])
                out_ids.append(int(tid) if isinstance(tid, int) else -1)
            continue

        seg_texts = src_texts[i:j]
        seg_str = None
        if use_orig_segs:
            cand = orig_text_segs[orig_seg_idx]
            if use_bytes:
                if sum(src_byte_lens[i:j]) == len(cand.encode("utf-8")):
                    seg_str = cand
            elif "".join(seg_texts) == cand:
                seg_str = cand
        orig_seg_idx += 1
        if seg_str is None:
            seg_str = "".join(seg_texts)

        seg_ids = target_tok.encode(seg_str, add_special_tokens=False)
        seg_tgt_texts = [target_tok.decode([tid], skip_special_tokens=False) for tid in seg_ids]
        if use_bytes:
            tgt_blens = target_token_byte_lens(target_tok, seg_ids)
            unpacked, sp, sl, _ = unpack_by_spans_with_info(
                hidden_seq[i:j], src_byte_lens[i:j], tgt_blens, n_to_1=n_to_1
            )
        else:
            unpacked, sp, sl, _ = unpack_by_spans_with_info(
                hidden_seq[i:j], [len(s) for s in seg_texts], [len(s) for s in seg_tgt_texts], n_to_1=n_to_1
            )
        out_h.append(unpacked)
        out_sp.append(sp)
        out_sl.append(sl)
        out_texts.extend(seg_tgt_texts)
        out_types.extend([0] * len(seg_ids))
        out_ids.extend(seg_ids)

    h_cat = torch.cat(out_h, dim=0) if out_h else hidden_seq
    sp_cat = torch.cat(out_sp, dim=0) if out_sp else torch.zeros(h_cat.shape[0], dtype=torch.long)
    sl_cat = torch.cat(out_sl, dim=0) if out_sl else torch.zeros(h_cat.shape[0], dtype=torch.long)
    if embed_model:
        emb = lookup_target_embeddings(embed_model, out_ids).to(device=h_cat.device, dtype=h_cat.dtype)
        h_cat = torch.cat([h_cat, emb], dim=-1)
    if return_split_info:
        return h_cat, out_texts, out_types, out_ids, sp_cat, sl_cat
    return h_cat, out_texts, out_types, out_ids


def project_with_modality(h, proj, tags, cache, split_pos=None, split_len=None):
    """Apply adapter projection, optionally routing vision tokens (tags == 0) through a dedicated branch."""
    if "W_vis" not in proj and not any(k.startswith("mlp_vis.") for k in proj):
        return projector.project(h, proj, cache, split_pos=split_pos, split_len=split_len)

    vis_mask = tags.to(device=h.device) == 0
    if not vis_mask.any():
        return projector.project(h, proj, cache, split_pos=split_pos, split_len=split_len)

    txt_mask = ~vis_mask
    has_seq = any(k.startswith("seq.") for k in proj)
    if txt_mask.any():
        if has_seq:
            h_txt_full = h.clone()
            h_txt_full[:, vis_mask, :] = 0
            txt_full_cond = projector.project(
                h_txt_full,
                proj,
                cache,
                valid_mask=txt_mask.unsqueeze(0).expand(h.shape[0], -1),
                split_pos=split_pos,
                split_len=split_len,
            )
            txt_cond = txt_full_cond[:, txt_mask, :]
        else:
            txt_cond = projector.project(h[:, txt_mask, :], proj, cache)
    else:
        txt_cond = None

    vis_cache = cache.setdefault("_vis_cache", {})
    if "_vis_proj" not in cache:
        vp = {
            "tap": proj["tap"],
            "mean_in": proj.get("mean_in_vis", proj["mean_in"]),
            "std_in": proj.get("std_in_vis", proj["std_in"]),
            "mean_out": proj.get("mean_out_vis", proj["mean_out"]),
            "std_out": proj.get("std_out_vis", proj["std_out"]),
        }
        if "W_vis" in proj:
            vp["W"] = proj["W_vis"].float()
        for k, v in proj.items():
            if k.startswith("mlp_vis."):
                vp["mlp." + k[len("mlp_vis."):]] = v
        for k in ("mean_in", "std_in", "mean_out", "std_out"):
            vp[k] = vp[k].float()
        cache["_vis_proj"] = vp

    vis_h = h[:, vis_mask, :]
    if "W_vis" in proj and vis_h.shape[-1] > proj["W_vis"].shape[0]:
        vis_h = vis_h[..., :proj["W_vis"].shape[0]]
    vis_cond = projector.project(vis_h, cache["_vis_proj"], vis_cache)

    d_out = vis_cond.shape[-1] if txt_cond is None else txt_cond.shape[-1]
    out_dtype = vis_cond.dtype if txt_cond is None else txt_cond.dtype
    cond = torch.empty((h.shape[0], h.shape[1], d_out), device=h.device, dtype=out_dtype)
    if txt_cond is not None:
        cond[:, txt_mask, :] = txt_cond
    cond[:, vis_mask, :] = vis_cond.to(dtype=out_dtype)
    sink = proj.get("sink_out")
    if sink is not None and cond.shape[1] > 0:
        cond[:, 0] = sink.to(device=cond.device, dtype=cond.dtype)
    return cond


def _get_http_conn(parsed, timeout):
    pool = getattr(_HTTP_LOCAL, "conns", None)
    if pool is None:
        pool = {}
        _HTTP_LOCAL.conns = pool
    key = (parsed.scheme, parsed.hostname, parsed.port)
    conn = pool.get(key)
    if conn is None:
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = cls(parsed.hostname, parsed.port, timeout=timeout)
        pool[key] = conn
    else:
        conn.timeout = timeout
        if getattr(conn, "sock", None) is not None:
            try:
                conn.sock.settimeout(timeout)
            except OSError:
                pass
    return key, conn


def fetch_hidden_states(url, parts, model="", layer=-1, layers=None, api_key="", timeout=120, system_prompt=""):
    """POST /v1/hidden_states over persistent HTTP keep-alive and return (meta, hidden_fp32[1, seq, K*H])."""
    endpoint = url.rstrip("/") + "/v1/hidden_states"
    parsed = urllib.parse.urlparse(endpoint)
    path = parsed.path or "/v1/hidden_states"
    if parsed.query:
        path += "?" + parsed.query
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": parts})
    payload = {
        "messages": messages,
        "enable_thinking": False,
    }
    if model:
        payload["model"] = model
    if layers is not None and len(layers) > 0:
        payload["layers"] = [int(x) for x in layers]
    elif layer is not None and int(layer) != -1:
        payload["layer"] = int(layer)
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    token = api_key or os.environ.get(API_KEY_ENV, "")
    if token:
        headers["Authorization"] = "Bearer " + token

    t0 = time.perf_counter()
    raw = None
    last_err = None
    for attempt in range(2):
        key, conn = _get_http_conn(parsed, timeout)
        try:
            conn.request("POST", path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                detail = raw.decode("utf-8", errors="replace")[:500]
                raise RuntimeError("Server /v1/hidden_states HTTP %d from %s: %s" % (resp.status, endpoint, detail))
            break
        except RuntimeError:
            raise
        except Exception as err:
            last_err = err
            try:
                conn.close()
            except Exception:
                pass
            getattr(_HTTP_LOCAL, "conns", {}).pop(key, None)
            if attempt == 1:
                raise RuntimeError("Cannot reach server at %s: %s" % (endpoint, last_err)) from last_err
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    meta, hidden = decode_hidden_response(raw)
    meta["http_ms"] = round(elapsed_ms, 2)
    return meta, hidden


class UniCondRemoteCLIP:
    """Duck-typed ComfyUI CLIP backed by a remote external hidden-states server (0 B encoder VRAM)."""

    def __init__(self, url=DEFAULT_URL, model="", layer=-1, adapter_path=None, api_key="", timeout=120, profile="auto"):
        self.url = url or DEFAULT_URL
        self.model = model or ""
        self.layer = int(layer) if layer is not None else -1
        self.adapter_path = adapter_path
        self.api_key = api_key or ""
        self.timeout = timeout
        self._proj = projector.load_projection(adapter_path) if adapter_path else None
        self._proj_cache = {}
        if profile == "auto" or not profile:
            self.profile = str(self._proj.get("target_profile", "minimax_h3")) if self._proj else "minimax_h3"
        else:
            self.profile = profile
        self.cond_stage_model = None
        self.patcher = None

    def clone(self):
        return UniCondRemoteCLIP(
            url=self.url,
            model=self.model,
            layer=self.layer,
            adapter_path=self.adapter_path,
            api_key=self.api_key,
            timeout=self.timeout,
            profile=self.profile,
        )

    def add_hooks_to_dict(self, pooled_dict):
        return pooled_dict

    def set_clip_options(self, options):
        pass

    def clip_layer(self, layer_idx):
        pass

    def tokenize(self, text, return_word_ids=False, images=(), minimax_ref_items=None,
                 picture_label="<Picture %d>: ", keep_vision=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and not images:
            images = [image[i:i + 1] for i in range(image.shape[0])]
        if self.profile == "qwen_image_21":
            parts = build_parts(text, images=images, picture_label="<image%d>")
            return {"unicond": {"parts": parts, "keep_vision": bool(keep_vision)}}
        parts = build_parts(text, images=images, minimax_ref_items=minimax_ref_items, picture_label=picture_label)
        return {"unicond": {"parts": parts, "keep_vision": True}}

    def _encode(self, tokens):
        try:
            import comfy.model_management as mm
            out_device = mm.intermediate_device()
            torch_dev = mm.get_torch_device()
        except ImportError:
            out_device = torch.device("cpu")
            torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(tokens, dict) and "unicond" in tokens:
            parts = tokens["unicond"]["parts"]
            keep_vision = bool(tokens["unicond"].get("keep_vision", True))
        elif isinstance(tokens, str):
            parts = build_parts(tokens)
            keep_vision = True
        else:
            raise ValueError("UniCondRemoteCLIP received unsupported token payload: %r" % (type(tokens),))

        tap = self.layer
        tap_layers = None
        if tap < 0 and self._proj is not None:
            raw_layers = self._proj.get("layers")
            if isinstance(raw_layers, str) and raw_layers.strip():
                tap_layers = [int(x) for x in json.loads(raw_layers)]
            elif isinstance(raw_layers, (list, tuple)) and raw_layers:
                tap_layers = [int(x) for x in raw_layers]
            elif "tap" in self._proj:
                tap = int(self._proj["tap"])

        sys_prompt = QI21_SYSTEM_PROMPT if self.profile == "qwen_image_21" else ""
        meta, hidden = fetch_hidden_states(
            self.url, parts, model=self.model, layer=tap, layers=tap_layers,
            api_key=self.api_key, timeout=self.timeout, system_prompt=sys_prompt,
        )
        raw_seq = hidden.shape[1]
        token_types = meta.get("token_types") or [0] * raw_seq
        token_texts = meta.get("token_texts")
        raw_blens = meta.get("token_byte_lens")
        token_byte_lens = raw_blens if (isinstance(raw_blens, list) and len(raw_blens) == raw_seq) else None
        proj_dev = (
            torch_dev
            if (self._proj is not None and torch_dev.type != "cpu"
                and (any(k.startswith("seq.") for k in self._proj) or hidden.shape[-1] >= 10240))
            else hidden.device
        )
        unpack_profile = str(self._proj.get("unpack_target_vocab", "")).strip() if self._proj else ""
        embed_model = str(self._proj.get("target_embed_model", "")).strip() if self._proj else ""
        n_to_1 = str(self._proj.get("unpack_n_to_1", "mean")).strip() if self._proj else "mean"
        orig_segs = _extract_orig_text_segments(parts)

        if self.profile == "qwen_image_21":
            start, end = find_qi21_span(token_texts, raw_seq)
            hidden = hidden[:, start:end].contiguous()
            t_types = token_types[start:end]
            sliced_texts = token_texts[start:end] if token_texts and len(token_texts) == raw_seq else None
            sliced_blens = token_byte_lens[start:end] if token_byte_lens is not None else None
            sp, sl = None, None
            if unpack_profile and sliced_texts is not None:
                tgt_tok = get_target_tokenizer(unpack_profile)
                qi_segs = (["user\n" + orig_segs[0]] + orig_segs[1:] + ["\n", "assistant\n"]) if orig_segs else None
                u_h, sliced_texts, t_types, _, sp_1d, sl_1d = unpack_to_target_vocab(
                    hidden[0], sliced_texts, t_types, tgt_tok,
                    embed_model=embed_model or None,
                    src_byte_lens=sliced_blens,
                    orig_text_segs=qi_segs,
                    n_to_1=n_to_1,
                    return_split_info=True,
                )
                hidden = u_h.unsqueeze(0).contiguous()
                sp = sp_1d.unsqueeze(0)
                sl = sl_1d.unsqueeze(0)
            slots = []
            if not keep_vision and any(int(t) != 0 for t in t_types):
                keep_mask = []
                in_vis = False
                kept_count = 0
                for t in t_types:
                    if int(t) != 0:
                        if not in_vis:
                            slots.append(kept_count)
                            in_vis = True
                        keep_mask.append(False)
                    else:
                        in_vis = False
                        keep_mask.append(True)
                        kept_count += 1
                km = torch.tensor(keep_mask, dtype=torch.bool)
                hidden = hidden[:, km, :].contiguous()
                if sp is not None and sl is not None:
                    sp = sp[:, km].contiguous()
                    sl = sl[:, km].contiguous()
            if self._proj is not None:
                with torch.no_grad():
                    cond = projector.project(
                        hidden.to(proj_dev), self._proj, self._proj_cache, split_pos=sp, split_len=sl
                    )
            else:
                cond = hidden
            cond = cond.to(out_device)
            extra = {"pooled_output": None}
            if slots:
                extra["image_slots"] = slots
            logging.info(
                "Universal Conditioner [qwen_image_21]: model=%s seq=%d/%d dim=%d layers=%s slots=%s prefill=%.1fms vision=%.1fms http=%.1fms%s",
                meta.get("model", "?"),
                cond.shape[1],
                raw_seq,
                cond.shape[2],
                meta.get("layers", meta.get("layer", "final")),
                slots,
                float(meta.get("prefill_ms", 0.0)),
                float(meta.get("vision_ms", 0.0)),
                float(meta.get("http_ms", 0.0)),
                " (adapted)" if self._proj is not None else " (raw)",
            )
            return cond, extra

        start, end = find_user_span(token_texts, raw_seq)
        has_sink = self._proj is not None and self._proj.get("sink_out") is not None
        sp, sl = None, None
        if unpack_profile and token_texts and len(token_texts) == raw_seq:
            tgt_tok = get_target_tokenizer(unpack_profile)
            sliced_blens = token_byte_lens[start:end] if token_byte_lens is not None else None
            u_h, sliced_texts, token_types, _, sp_1d, sl_1d = unpack_to_target_vocab(
                hidden[0, start:end], token_texts[start:end], token_types[start:end],
                tgt_tok,
                embed_model=embed_model or None,
                src_byte_lens=sliced_blens,
                orig_text_segs=orig_segs or None,
                n_to_1=n_to_1,
                return_split_info=True,
            )
            hidden = u_h.unsqueeze(0).contiguous()
            sp = sp_1d.unsqueeze(0)
            sl = sl_1d.unsqueeze(0)
        else:
            slice_start = (start - 1) if (has_sink and start > 0) else start
            hidden = hidden[:, slice_start:end].contiguous()
            token_types = token_types[slice_start:end]
            sliced_texts = token_texts[slice_start:end] if token_texts and len(token_texts) == raw_seq else None
        tags = token_tags_from_types(hidden.shape[1], token_types, sliced_texts)

        if self._proj is not None:
            with torch.no_grad():
                cond = project_with_modality(
                    hidden.to(proj_dev), self._proj, tags, self._proj_cache, split_pos=sp, split_len=sl
                )
        else:
            cond = hidden

        cond = cond.to(out_device)
        extra = {"pooled_output": None, "minimax_token_tags": tags}
        logging.info(
            "Universal Conditioner [minimax_h3]: model=%s seq=%d/%d dim=%d layers=%s prefill=%.1fms vision=%.1fms http=%.1fms%s",
            meta.get("model", "?"),
            cond.shape[1],
            raw_seq,
            cond.shape[2],
            meta.get("layers", meta.get("layer", "final")),
            float(meta.get("prefill_ms", 0.0)),
            float(meta.get("vision_ms", 0.0)),
            float(meta.get("http_ms", 0.0)),
            " (adapted)" if self._proj is not None else " (raw)",
        )
        return cond, extra

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond, extra = self._encode(tokens)
        if return_dict:
            out = {"cond": cond}
            out.update(extra)
            return out
        return (cond, extra.get("pooled_output")) if return_pooled else cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        cond, extra = self._encode(tokens)
        if add_dict:
            extra = dict(extra)
            extra.update(add_dict)
        return [[cond, extra]]

    def encode(self, text):
        return self.encode_from_tokens(self.tokenize(text))


def load(url=DEFAULT_URL, model="", layer=-1, adapter="none", profile="auto"):
    adapter_path = None
    if adapter and adapter != "none":
        adapter_path = projector.resolve_adapter_path(adapter)
    return UniCondRemoteCLIP(
        url=url,
        model=model,
        layer=layer,
        adapter_path=adapter_path,
        api_key=os.environ.get(API_KEY_ENV, ""),
        profile=profile,
    )
