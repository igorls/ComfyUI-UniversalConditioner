"""Calibrate a Universal Conditioner adapter for Qwen-Image 2.1.

Usage (set PYTHONPATH to your ComfyUI checkout):
    PYTHONPATH=/path/to/ComfyUI python tools/calibrate_qi21.py \\
        --server-url http://127.0.0.1:8000 \\
        --prompts-json /path/to/prompts.json \\
        --edit-json /path/to/edits.json \\
        --edit-images-dir /path/to/edit_images \\
        --layers=-2,15,47,55,63 \\
        --out-dir /path/to/output \\
        --out-prefix unicond-qi21-v3.1
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encoder  # noqa: E402
import seq_connector  # noqa: E402
from tools.ridge_math import (  # noqa: E402
    compute_gram,
    eval_masked,
    fit_two_block_ridge,
    solve_block_ridge,
)

_CAPS_RE = re.compile(r"[A-Z]{2,}|[0-9]|[\$€£¥]")


def compute_typography_mask(tok_texts):
    """Return bool tensor [L] marking quoted text and ALL-CAPS/alphanumeric signage tokens."""
    L = len(tok_texts)
    mask = torch.zeros(L, dtype=torch.bool)
    in_double = False
    in_single = False
    for i, t in enumerate(tok_texts):
        if t in (
            "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "user", "assistant", "\n",
        ):
            continue
        dq = t.count('"')
        sq = 0
        if "'" in t and not re.search(r"[a-z]'[a-z]", t):
            sq = t.count("'")
        was_in = in_double or in_single
        if dq % 2 == 1:
            in_double = not in_double
        if sq % 2 == 1:
            in_single = not in_single
        if was_in or in_double or in_single or dq > 0 or sq > 0 or bool(_CAPS_RE.search(t)):
            mask[i] = True
    return mask


def load_qi21_image_tensor(path, resolution=1024):
    """Load an RGB image and resize using Qwen-Image 2.1 multiple-of-32 Lanczos rules -> [1, H, W, 3]."""
    pil = Image.open(path).convert("RGB")
    w0, h0 = pil.size
    if resolution > 0:
        ratio = float(w0) / float(max(1, h0))
        width = max(32, int(round(math.sqrt(resolution * resolution * ratio) / 32.0)) * 32)
        height = max(32, int(round(math.sqrt(resolution * resolution / ratio) / 32.0)) * 32)
    else:
        width = max(32, int(round(w0 / 32.0)) * 32)
        height = max(32, int(round(h0 / 32.0)) * 32)
    if (width, height) != (w0, h0):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def extract_8b_qi21_targets_with_ids(clip_name, items, edit_images_dir=None, edit_resolution=1024):
    """Extract Qwen-Image 2.1 native Qwen3-VL-8B targets [seq, 4096] + exact kept token IDs."""
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    print(f"[QI21-8B] Loading {os.path.basename(path)} ...", flush=True)
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
    )
    hf_tok = clip.tokenizer.qwen3vl_8b.tokenizer
    mm.load_models_gpu([clip.patcher])
    print(f"[QI21-8B] Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    img_cache = {}

    def get_img(rel):
        if rel not in img_cache:
            full = rel if os.path.isabs(rel) else os.path.join(edit_images_dir or "", rel)
            img_cache[rel] = load_qi21_image_tensor(full, resolution=edit_resolution)
        return img_cache[rel]

    targets = []
    skipped = 0
    t1 = time.perf_counter()
    try:
        for idx, item in enumerate(items):
            text = item["prompt"]
            img_paths = item.get("images") or []
            imgs = [get_img(p) for p in img_paths]
            tokens = clip.tokenize(text, images=imgs, keep_vision=False, prevent_empty_text=True)
            raw_ids = [t[0] for t in tokens["qwen3vl_8b"][0] if isinstance(t[0], int)]
            im_starts = [i for i, tid in enumerate(raw_ids) if tid == 151644]
            drop = im_starts[1] if len(im_starts) > 1 else 0
            kept_ids = raw_ids[drop:]
            tok_texts = [hf_tok.decode([tid], skip_special_tokens=False) for tid in kept_ids]
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")
            if h.shape[0] != len(kept_ids):
                skipped += 1
                continue
            targets.append({
                "prompt": text,
                "images": img_paths,
                "is_edit": bool(img_paths),
                "category": item.get("category", "t2i"),
                "token_ids": kept_ids,
                "token_texts": tok_texts,
                "image_slots": out.get("image_slots", []),
                "h": h,
            })
            if (idx + 1) % 200 == 0 or idx + 1 == len(items):
                n_ed = sum(1 for x in targets if x["is_edit"])
                print(
                    f"[QI21-8B] Encoded {idx + 1}/{len(items)} (kept={len(targets)}, edits={n_ed}, "
                    f"skip={skipped}, {time.perf_counter() - t1:.1f}s)",
                    flush=True,
                )
    finally:
        del clip
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return targets


def collect_unpacked_sequences(
    url,
    layers,
    targets,
    clip_8b_name=None,
    edit_images_dir=None,
    edit_resolution=1024,
    workers=4,
    n_to_1="last",
):
    """Fetch multi-layer hidden states from the external server, unpack into target 151k tokens,
    drop vision patches (`keep_vision=False`), and optionally append mmapped target embed_tokens."""
    t0 = time.perf_counter()
    tgt_tok = encoder.get_target_tokenizer("qwen_image_21")
    if clip_8b_name:
        encoder.lookup_target_embeddings(clip_8b_name, [0])

    img_url_cache = {}
    unique_imgs = sorted({p for item in targets for p in (item.get("images") or [])})
    if unique_imgs:
        for rel in unique_imgs:
            full = rel if os.path.isabs(rel) else os.path.join(edit_images_dir or "", rel)
            t_img = load_qi21_image_tensor(full, resolution=edit_resolution)
            img_url_cache[rel] = encoder.encode_image_data_url(t_img)

    def build_cached_parts(text, img_paths):
        parts = []
        for i, rel in enumerate(img_paths):
            prefix = " " if i > 0 else ""
            parts.append({"type": "text", "text": f"{prefix}<image{i + 1}>"})
            parts.append({"type": "image_url", "image_url": {"url": img_url_cache[rel]}})
        if text:
            parts.append({"type": "text", "text": text})
        if not parts:
            parts.append({"type": "text", "text": " "})
        return parts

    stats = {"byte_lens_used": 0, "id_mismatches": 0, "len_mismatches": 0, "slot_mismatches": 0}

    def fetch_one(p_idx):
        item = targets[p_idx]
        parts = build_cached_parts(item["prompt"], item.get("images") or [])
        meta, h_full = encoder.fetch_hidden_states(
            url, parts, layers=layers, system_prompt=encoder.QI21_SYSTEM_PROMPT
        )
        raw_seq = h_full.shape[1]
        token_types = meta.get("token_types") or [0] * raw_seq
        tok_texts = meta.get("token_texts") or []
        raw_blens = meta.get("token_byte_lens")
        blens = raw_blens if (isinstance(raw_blens, list) and len(raw_blens) == raw_seq) else None

        start, end = encoder.find_qi21_span(tok_texts, raw_seq)
        u_h_src = h_full[0, start:end]
        u_texts = tok_texts[start:end]
        u_types = token_types[start:end]
        u_blens = blens[start:end] if blens is not None else None

        orig_segs = encoder._extract_orig_text_segments(parts)
        qi_segs = (["user\n" + orig_segs[0]] + orig_segs[1:] + ["\n", "assistant\n"]) if orig_segs else None
        unpacked, out_texts, out_types, out_ids = encoder.unpack_to_target_vocab(
            u_h_src,
            u_texts,
            u_types,
            tgt_tok,
            embed_model=clip_8b_name,
            src_byte_lens=u_blens,
            orig_text_segs=qi_segs,
            n_to_1=n_to_1,
        )

        slots = []
        if any(int(t) != 0 for t in out_types):
            keep_mask = []
            in_vis = False
            kept_count = 0
            for t in out_types:
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
            unpacked = unpacked[km].contiguous()
            out_ids = [tid for tid, k in zip(out_ids, keep_mask) if k]

        id_ok = (out_ids == item["token_ids"])
        slot_ok = (slots == item.get("image_slots", []))
        if unpacked.shape[0] != item["h"].shape[0]:
            return {"prompt_idx": p_idx, "drop": True, "blens_used": blens is not None}
        return {
            "x": unpacked.to(torch.bfloat16).contiguous(),
            "y": item["h"].contiguous(),
            "typo": compute_typography_mask(item["token_texts"]),
            "is_edit": item["is_edit"],
            "category": item.get("category", "t2i"),
            "prompt_idx": p_idx,
            "drop": False,
            "blens_used": blens is not None,
            "id_ok": id_ok,
            "slot_ok": slot_ok,
        }

    results = [None] * len(targets)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(fetch_one, range(len(targets))):
            results[res["prompt_idx"]] = res
            if res["blens_used"]:
                stats["byte_lens_used"] += 1
            if res["drop"]:
                stats["len_mismatches"] += 1
            else:
                if not res["id_ok"]:
                    stats["id_mismatches"] += 1
                if not res["slot_ok"]:
                    stats["slot_mismatches"] += 1
            done += 1
            if done % 250 == 0 or done == len(targets):
                print(
                    f"[SRC] Collected {done}/{len(targets)} ({time.perf_counter() - t0:.1f}s) | "
                    f"blens={stats['byte_lens_used']} drop={stats['len_mismatches']} "
                    f"id_diff={stats['id_mismatches']} slot_diff={stats['slot_mismatches']}",
                    flush=True,
                )

    sequences = [
        {k: r[k] for k in ("x", "y", "typo", "is_edit", "category")}
        for r in results
        if r is not None and not r["drop"]
    ]
    return sequences, round(time.perf_counter() - t0, 2)


def flatten_seqs(seqs):
    X = torch.cat([s["x"] for s in seqs], dim=0)
    Y = torch.cat([s["y"] for s in seqs], dim=0)
    T = torch.cat([s["typo"] for s in seqs], dim=0)
    E = torch.cat([
        torch.full((s["y"].shape[0],), bool(s.get("is_edit", False)), dtype=torch.bool) for s in seqs
    ], dim=0)
    return X, Y, T, E


def compute_oof_base_predictions(tr_seqs, te_seqs, ridge, d_main, n_folds=5, device="cuda"):
    t0 = time.perf_counter()
    lm, le = ridge["lam_main"], ridge["lam_emb"]
    m_out_full = ridge["mean_out"].to(device)
    s_out_full = ridge["std_out"].to(device)

    n = len(tr_seqs)
    fold_size = (n + n_folds - 1) // n_folds
    oof_bases = [None] * n

    for f in range(n_folds):
        v0 = f * fold_size
        v1 = min(n, v0 + fold_size)
        val_s = tr_seqs[v0:v1]
        trn_s = tr_seqs[:v0] + tr_seqs[v1:]
        Xtr, Ytr, _, _ = flatten_seqs(trn_s)
        xtx, xty, m_in, s_in, m_out, s_out = compute_gram(Xtr, Ytr, device=device)
        W_f = solve_block_ridge(xtx, xty, d_main, lm, le, device=device)
        mi_g, si_g = m_in.to(device), s_in.to(device)
        mo_g, so_g = m_out.to(device), s_out.to(device)
        with torch.no_grad():
            for offset, s in enumerate(val_s):
                xn = (s["x"].float().to(device) - mi_g) / si_g
                pred_raw = (xn @ W_f) * so_g + mo_g
                oof_bases[v0 + offset] = ((pred_raw - m_out_full) / s_out_full).to(torch.bfloat16).cpu()
        del Xtr, Ytr, xtx, xty, W_f
        torch.cuda.empty_cache()

    mi_g, si_g = ridge["mean_in"].to(device), ridge["std_in"].to(device)
    W_g = ridge["W"].to(device)
    te_bases = []
    with torch.no_grad():
        for s in te_seqs:
            xn = (s["x"].float().to(device) - mi_g) / si_g
            te_bases.append((xn @ W_g).to(torch.bfloat16).cpu())
    print(f"[OOF {n_folds}-fold] Computed in {time.perf_counter() - t0:.1f}s", flush=True)
    return oof_bases, te_bases


def compute_channel_clamps(tr_seqs, margin=0.15):
    Y = torch.cat([s["y"].float() for s in tr_seqs], dim=0)
    if Y.shape[0] > 120000:
        idx = torch.linspace(0, Y.shape[0] - 1, 120000).long()
        Y = Y[idx]
    q_lo = torch.quantile(Y, 0.001, dim=0)
    q_hi = torch.quantile(Y, 0.999, dim=0)
    span = (q_hi - q_lo).clamp(min=1.0)
    return (q_lo - margin * span).contiguous(), (q_hi + margin * span).contiguous()


def train_oof_refinement_connector(
    tr_seqs,
    te_seqs,
    oof_bases,
    te_bases,
    ridge,
    d_model=768,
    n_layers=2,
    n_heads=8,
    epochs=32,
    batch_size=32,
    lr=4e-4,
    typo_weight=3.0,
    edit_weight=2.5,
    edit_typo_weight=4.0,
    device="cuda",
):
    d_out = ridge["W"].shape[1]
    m_out = ridge["mean_out"].to(device)
    s_out = ridge["std_out"].to(device)
    has_edits = any(s.get("is_edit", False) for s in te_seqs)

    net = seq_connector.BidirectionalSeqConnector(
        d_in=d_out,
        d_out=d_out,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_mult=2,
        conv_kernel=5,
        dropout=0.15,
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.03)
    warmup = 3

    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(warmup)
        prog = float(ep - warmup) / max(1.0, float(epochs - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def make_batch(seqs, bases, idxs):
        B = len(idxs)
        Lmax = max(seqs[k]["y"].shape[0] for k in idxs)
        Base = torch.zeros(B, Lmax, d_out, dtype=torch.bfloat16)
        Y = torch.zeros(B, Lmax, d_out, dtype=torch.bfloat16)
        valid = torch.zeros(B, Lmax, dtype=torch.bool)
        typo = torch.zeros(B, Lmax, dtype=torch.bool)
        edit = torch.zeros(B, Lmax, dtype=torch.bool)
        for i, k in enumerate(idxs):
            L = seqs[k]["y"].shape[0]
            Base[i, :L] = bases[k]
            Y[i, :L] = seqs[k]["y"]
            valid[i, :L] = True
            typo[i, :L] = seqs[k]["typo"]
            if seqs[k].get("is_edit", False):
                edit[i, :L] = True
        return (
            Base.to(device, non_blocking=True),
            Y.to(device, non_blocking=True),
            valid.to(device, non_blocking=True),
            typo.to(device, non_blocking=True),
            edit.to(device, non_blocking=True),
        )

    def evaluate():
        net.eval()
        ps, ts, tm, em = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(te_seqs), batch_size):
                idxs = list(range(i, min(len(te_seqs), i + batch_size)))
                Base, Y, valid, typo, edit = make_batch(te_seqs, te_bases, idxs)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    delta = net(Base, valid_mask=valid)
                pred = (Base.float() + delta.float()) * s_out + m_out
                ps.append(pred[valid])
                ts.append(Y.float()[valid])
                tm.append(typo[valid])
                em.append(edit[valid])
        P, T, M, E = torch.cat(ps, 0), torch.cat(ts, 0), torch.cat(tm, 0), torch.cat(em, 0)
        ac, ar, arel, _ = eval_masked(P, T)
        tc, tr2, trel, tnorm = eval_masked(P, T, M)
        ec, er2, erel, enorm = eval_masked(P, T, E) if has_edits else (0.0, 0.0, 0.0, 1.0)
        return ac, ar, arel, tc, tr2, trel, tnorm, ec, er2, erel, enorm

    def score_fn(m):
        return (0.30 * m[0] + 0.35 * m[3] + 0.35 * m[7]) if has_edits else (0.45 * m[0] + 0.55 * m[3])

    b = evaluate()
    best_score = score_fn(b)
    best_metrics = (b[0], b[1], b[3], b[4], b[7], b[8])
    best_state = None
    rng = random.Random(42)
    t0 = time.perf_counter()

    for ep in range(epochs):
        net.train()
        order = list(range(len(tr_seqs)))
        rng.shuffle(order)
        tr_loss_sum, tr_batches = 0.0, 0
        for i in range(0, len(order), batch_size):
            Base, Y, valid, typo, edit = make_batch(tr_seqs, oof_bases, order[i : i + batch_size])
            yt = Y.float()
            yn_true = (yt - m_out) / s_out
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                delta = net(Base, valid_mask=valid)
                pred_n = Base.float() + delta.float()
                pred = pred_n * s_out + m_out
                pn_a, yn_a = pred_n[valid], yn_true[valid]
                p_a, y_a = pred[valid], yt[valid]
                t_a, e_a = typo[valid], edit[valid]
                w = torch.ones_like(pn_a[:, 0])
                w = torch.where(e_a, edit_weight, w)
                w = torch.where(t_a & ~e_a, typo_weight, w)
                w = torch.where(t_a & e_a, edit_typo_weight, w)
                w = w / w.mean().clamp(min=1e-6)
                tok_mse = ((pn_a - yn_a) ** 2).mean(dim=-1)
                tok_cos = 1.0 - F.cosine_similarity(p_a, y_a, dim=-1)
                tok_rel = (p_a - y_a).norm(dim=-1) / y_a.norm(dim=-1).clamp(min=1e-6)
                loss = (w * (tok_mse + 1.0 * tok_cos + 0.35 * tok_rel)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_loss_sum += float(loss.item())
            tr_batches += 1
        sched.step()

        cur = evaluate()
        score = score_fn(cur)
        if score > best_score:
            best_score = score
            best_metrics = (cur[0], cur[1], cur[3], cur[4], cur[7], cur[8])
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 4 == 0 or ep + 1 == epochs or ep == 0:
            e_str = f" | EDIT cos={cur[7]:.4f} r2={cur[8]:.4f}" if has_edits else ""
            print(
                f"  [REFINE] ep {ep + 1:2d}/{epochs} tr_loss={tr_loss_sum/max(1,tr_batches):.4f} | "
                f"ALL cos={cur[0]:.4f} r2={cur[1]:.4f} | TYPO cos={cur[3]:.4f} r2={cur[4]:.4f}{e_str} "
                f"({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    if best_state is None:
        return None, best_metrics
    net.load_state_dict(best_state)
    return net, best_metrics


def save_qi21_adapter(path, ridge, layers, clip_8b_name, source_model, meta_extra, net=None, clamp_lo=None, clamp_hi=None, n_to_1="last", source_file="", source_sha256=""):
    tensors = {
        "W": ridge["W"].half().contiguous(),
        "mean_in": ridge["mean_in"].half().contiguous(),
        "std_in": ridge["std_in"].half().contiguous(),
        "mean_out": ridge["mean_out"].half().contiguous(),
        "std_out": ridge["std_out"].half().contiguous(),
    }
    if clamp_lo is not None and clamp_hi is not None:
        tensors["clamp_lo"] = clamp_lo.half().contiguous()
        tensors["clamp_hi"] = clamp_hi.half().contiguous()
    meta = {
        "target_profile": "qwen_image_21",
        "unpack_target_vocab": "qwen_image_21",
        "unpack_n_to_1": n_to_1,
        "target_embed_model": os.path.basename(clip_8b_name) if clip_8b_name else "",
        "tap": str(int(layers[-1])),
        "layers": json.dumps([int(x) for x in layers]),
        "d_in": str(int(ridge["W"].shape[0])),
        "d_out": str(int(ridge["W"].shape[1])),
        "lambda": str(ridge["lam_main"]),
        "lambda_emb": str(ridge["lam_emb"]),
        "source_model": source_model,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "target_model": os.path.basename(clip_8b_name) if clip_8b_name else "qwen3vl_8b",
    }
    if net is not None:
        tensors.update(seq_connector.state_dict_prefixed(net, prefix="seq.", dtype=torch.bfloat16))
        meta["connector_type"] = "seq_base_refine_v1"
        meta["seq_input"] = "base"
        meta["seq_config"] = json.dumps(net.config)
    meta.update({k: str(v) for k, v in meta_extra.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata=meta)
    print(f"[SAVE] {path} ({os.path.getsize(path) / (1024 * 1024):.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Calibrate a Universal Conditioner adapter for Qwen-Image 2.1.")
    parser.add_argument("--server-url", default=encoder.DEFAULT_URL)
    parser.add_argument("--source-model", default="neroued/Qwen3.8-27B-nvfp4-NInfer")
    parser.add_argument("--source-file", default="")
    parser.add_argument("--source-sha256", default="")
    parser.add_argument("--clip-8b", default="qwen3vl_8b_int8_convrot.safetensors")
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--n-prompts", type=int, default=2600)
    parser.add_argument("--edit-json", default="")
    parser.add_argument("--edit-images-dir", default="")
    parser.add_argument("--edit-resolution", type=int, default=1024)
    parser.add_argument("--layers", default="-2,15,47,55,63")
    parser.add_argument("--no-target-embed", action="store_true")
    parser.add_argument("--n-to-1", default="last", choices=("last", "mean"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--typo-weight", type=float, default=3.0)
    parser.add_argument("--edit-weight", type=float, default=2.5)
    parser.add_argument("--edit-typo-weight", type=float, default=4.0)
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-prefix", default="unicond-qi21-v3.1")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    embed_model = None if args.no_target_embed else args.clip_8b
    d_main = len(layers) * 5120

    if args.cache_path and os.path.isfile(args.cache_path):
        seqs = torch.load(args.cache_path, weights_only=True)
    else:
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        prompts = [x["prompt"] if isinstance(x, dict) else str(x) for x in loaded]
        random.Random(42).shuffle(prompts)
        prompts = prompts[: args.n_prompts]
        items = [{"prompt": p, "images": [], "category": "t2i"} for p in prompts]
        if args.edit_json and os.path.isfile(args.edit_json):
            with open(args.edit_json, "r", encoding="utf-8") as f:
                for e in json.load(f):
                    items.append({
                        "prompt": str(e.get("prompt", "")),
                        "images": list(e.get("images") or []),
                        "category": str(e.get("category", "edit")),
                    })
        targets = extract_8b_qi21_targets_with_ids(
            args.clip_8b, items, edit_images_dir=args.edit_images_dir, edit_resolution=args.edit_resolution
        )
        seqs, _ = collect_unpacked_sequences(
            args.server_url,
            layers,
            targets,
            clip_8b_name=embed_model,
            edit_images_dir=args.edit_images_dir,
            edit_resolution=args.edit_resolution,
            workers=args.workers,
            n_to_1=args.n_to_1,
        )
        if args.cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)), exist_ok=True)
            torch.save(seqs, args.cache_path)

    t2i_seqs = [s for s in seqs if not s.get("is_edit", False)]
    edit_seqs = [s for s in seqs if s.get("is_edit", False)]
    random.Random(123).shuffle(t2i_seqs)
    random.Random(123).shuffle(edit_seqs)
    nt_tr = int(len(t2i_seqs) * 0.9)
    ne_tr = int(len(edit_seqs) * 0.9)
    tr_seqs = t2i_seqs[:nt_tr] + edit_seqs[:ne_tr]
    te_seqs = t2i_seqs[nt_tr:] + edit_seqs[ne_tr:]
    random.Random(42).shuffle(tr_seqs)

    Xtr, Ytr, _, _ = flatten_seqs(tr_seqs)
    Xte, Yte, Tte, Ete = flatten_seqs(te_seqs)
    clamp_lo, clamp_hi = compute_channel_clamps(tr_seqs)
    ridge = fit_two_block_ridge(Xtr, Ytr, Xte, Yte, Tte, d_main=d_main, E_te=Ete)
    save_qi21_adapter(
        os.path.join(args.out_dir, f"{args.out_prefix}-ridge.safetensors"),
        ridge, layers, embed_model, args.source_model,
        {
            "n_train_prompts": len(tr_seqs),
            "n_train_edits": ne_tr,
            "n_train_tokens": int(Xtr.shape[0]),
            "cos_test": round(ridge["cos"], 6),
            "r2_test": round(ridge["r2"], 6),
            "typo_cos_test": round(ridge["typo_cos"], 6),
            "typo_r2_test": round(ridge["typo_r2"], 6),
            "edit_cos_test": round(ridge["edit_cos"], 6),
            "edit_r2_test": round(ridge["edit_r2"], 6),
        },
        net=None, clamp_lo=clamp_lo, clamp_hi=clamp_hi, n_to_1=args.n_to_1,
        source_file=args.source_file, source_sha256=args.source_sha256,
    )
    del Xtr, Ytr, Xte, Yte
    gc.collect()
    torch.cuda.empty_cache()

    oof_bases, te_bases = compute_oof_base_predictions(tr_seqs, te_seqs, ridge, d_main=d_main, n_folds=5)
    net, (s_ac, s_ar, s_tc, s_tr, s_ec, s_er) = train_oof_refinement_connector(
        tr_seqs, te_seqs, oof_bases, te_bases, ridge,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        epochs=args.epochs, batch_size=args.batch_size,
        typo_weight=args.typo_weight, edit_weight=args.edit_weight, edit_typo_weight=args.edit_typo_weight,
    )
    if net is not None:
        save_qi21_adapter(
            os.path.join(args.out_dir, f"{args.out_prefix}.safetensors"),
            ridge, layers, embed_model, args.source_model,
            {
                "n_train_prompts": len(tr_seqs),
                "n_train_edits": ne_tr,
                "n_train_tokens": int(sum(s["y"].shape[0] for s in tr_seqs)),
                "cos_test": round(s_ac, 6),
                "r2_test": round(s_ar, 6),
                "typo_cos_test": round(s_tc, 6),
                "typo_r2_test": round(s_tr, 6),
                "edit_cos_test": round(s_ec, 6),
                "edit_r2_test": round(s_er, 6),
            },
            net=net, clamp_lo=clamp_lo, clamp_hi=clamp_hi, n_to_1=args.n_to_1,
            source_file=args.source_file, source_sha256=args.source_sha256,
        )


if __name__ == "__main__":
    main()
