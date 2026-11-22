"""Calibrate and diagnose Universal Conditioner adapters for MiniMax H3.

Supports:
- Multi-layer external hidden-state taps with lossless UTF-8 byte-span 248k -> 151k unpacking
- Causal last-token N:1 collapse and zero-VRAM mmapped target embed_tokens lookup
- Two-Block Text Ridge + Dual Multi-Layer Vision Ridge (W_vis)
- 5-Fold Out-of-Fold (OOF) Bidirectional Sequence Connector with:
    * `seq_input="base_plus_emb"` (concatenated Ridge base + standardized target embeddings)
    * `(split_pos, split_len)` learned embeddings for 1:N unpacked tokens
    * Non-sink training filter, uncompacted ref2va vision-gap sequences, and per-channel clamps
    * Stratified ALL / NON_DIAL / EXACT_DIAL / SPLIT_DIAL metrics and train/val loss tracking
- Diagnostic mode (`--diag-adapter`) measuring EXACT_DIAL vs SPLIT_DIAL metrics and per-token
  probe accuracy on an existing adapter
- Greedy forward layer selection (`--layer-sweep-pool`) over candidate source transformer layers

Usage (set PYTHONPATH to your ComfyUI checkout):
    PYTHONPATH=/path/to/ComfyUI python tools/calibrate_h3.py \\
        --server-url http://127.0.0.1:8000 \\
        --prompts-json /path/to/prompts.json \\
        --vision-dir /path/to/vision_samples \\
        --layers=-2,15,43,51,59 \\
        --out-dir /path/to/output \\
        --out-prefix unicond-h3-v4.3
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encoder  # noqa: E402
import projector  # noqa: E402
import seq_connector  # noqa: E402
from tools.ridge_math import (  # noqa: E402
    collect_sample_images,
    compute_gram,
    eval_masked,
    eval_std_r2,
    fit_ridge,
    fit_two_block_ridge,
    solve_block_ridge,
)

SINK_NORM_THRESHOLD = 2000.0


def compute_dialogue_mask(tok_texts):
    """Return bool tensor [L] marking tokens inside <d>...</d> or <|lyrics_start|>...<|lyrics_end|>."""
    L = len(tok_texts)
    mask = torch.zeros(L, dtype=torch.bool)
    in_d = False
    in_l = False
    for i, t in enumerate(tok_texts):
        if t == "<d>":
            in_d = True
            mask[i] = True
            continue
        if t == "</d>":
            mask[i] = True
            in_d = False
            continue
        if t == "<|lyrics_start|>":
            in_l = True
            mask[i] = True
            continue
        if t == "<|lyrics_end|>":
            mask[i] = True
            in_l = False
            continue
        if in_d or in_l:
            mask[i] = True
    return mask


def flatten_clean_seqs(seqs):
    """Flatten sequences while excluding any attention-sink tokens (valid == False)."""
    xs, ys, ds, n1s = [], [], [], []
    for s in seqs:
        m = s["valid"]
        if m.any():
            xs.append(s["x"][m])
            ys.append(s["y"][m])
            ds.append(s["typo"][m])
            n1 = s.get("non_1to1")
            if n1 is None:
                n1 = torch.zeros_like(s["typo"])
            n1s.append(n1[m])
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(ds, dim=0), torch.cat(n1s, dim=0)


def extract_32b_targets_with_ids(clip_name, prompts, vision_items=()):
    """Load MiniMax H3 32B encoder once and extract targets [seq, 5120] + exact token_ids + vision targets."""
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    print(f"[H3-32B] Loading {os.path.basename(path)} ...", flush=True)
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    sm = clip.cond_stage_model.qwen3vl_32b
    hf_tok = clip.tokenizer.qwen3vl_32b.tokenizer
    mm.load_models_gpu([clip.patcher])
    print(f"[H3-32B] Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    captured = {}
    orig_forward = sm.transformer.forward

    def catching(*args, **kwargs):
        captured["embeds_info"] = kwargs.get("embeds_info", [])
        return orig_forward(*args, **kwargs)

    sm.transformer.forward = catching
    text_targets = []
    vision_targets = []
    sinks = []

    try:
        t1 = time.perf_counter()
        for idx, text in enumerate(prompts):
            tokens = clip.tokenize(text)
            raw_ids = [t[0] for t in tokens["qwen3vl_32b"][0] if isinstance(t[0], int)]
            tok_texts = [hf_tok.decode([tid], skip_special_tokens=False) for tid in raw_ids]
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")
            if h.shape[0] != len(raw_ids) or h.shape[0] < 2:
                continue
            if h[0].float().norm().item() > SINK_NORM_THRESHOLD:
                sinks.append(h[0].float())
            text_targets.append({
                "prompt": text,
                "full_token_ids": raw_ids,
                "full_token_texts": tok_texts,
                "token_ids": raw_ids[1:],
                "token_texts": tok_texts[1:],
                "h": h[1:].contiguous(),
            })
            if (idx + 1) % 250 == 0 or idx + 1 == len(prompts):
                print(f"[H3-32B] Text {idx + 1}/{len(prompts)} ({time.perf_counter() - t1:.1f}s)", flush=True)

        t2 = time.perf_counter()
        for idx, (img_name, img_tensor, prompt_text) in enumerate(vision_items):
            captured.clear()
            tokens = clip.tokenize(prompt_text, images=[img_tensor])
            raw_entries = tokens["qwen3vl_32b"][0]
            img_pos = next((k for k, e in enumerate(raw_entries) if not isinstance(e[0], int)), None)
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")
            embeds_info = captured.get("embeds_info", [])
            img_span = None
            for e in embeds_info:
                if e.get("type") == "image":
                    img_span = (int(e["index"]), int(e["size"]))
                    break
            if img_span is not None and img_pos is not None:
                v0 = max(0, img_span[0] - 1)
                v1 = min(h.shape[0], img_span[0] + img_span[1] + 1)
                prefix_ids = [
                    raw_entries[k][0]
                    for k in range(max(0, img_pos - 1))
                    if isinstance(raw_entries[k][0], int)
                ]
                suffix_ids = [
                    raw_entries[k][0]
                    for k in range(min(len(raw_entries), img_pos + 2), len(raw_entries))
                    if isinstance(raw_entries[k][0], int)
                ]
                vision_targets.append({
                    "name": img_name,
                    "prompt": prompt_text,
                    "img_tensor": img_tensor,
                    "vis_size": img_span[1],
                    "vis_block_len": v1 - v0,
                    "v0": v0,
                    "v1": v1,
                    "prefix_ids": prefix_ids,
                    "suffix_ids": suffix_ids,
                    "prefix_texts": [hf_tok.decode([tid], skip_special_tokens=False) for tid in prefix_ids],
                    "suffix_texts": [hf_tok.decode([tid], skip_special_tokens=False) for tid in suffix_ids],
                    "h_vis": h[v0:v1].contiguous(),
                    "h_full": h.contiguous(),
                })
        if vision_items:
            print(f"[H3-32B] Vision {len(vision_targets)}/{len(vision_items)} ({time.perf_counter() - t2:.1f}s)", flush=True)
    finally:
        sm.transformer.forward = orig_forward
        del sm, orig_forward, catching, hf_tok, clip
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sink_out = torch.stack(sinks, dim=0).mean(dim=0) if sinks else torch.zeros(5120, dtype=torch.float32)
    print(f"[H3-32B] Unloaded. sink_out norm={sink_out.norm().item():.2f} (from {len(sinks)} clean sinks)", flush=True)
    return text_targets, vision_targets, sink_out


def _fetch_layers_chunked(url, parts, layers, max_per_call=8):
    """Fetch hidden states across any number of layers, splitting into <= max_per_call chunks."""
    if len(layers) <= max_per_call:
        return encoder.fetch_hidden_states(url, parts, layers=layers)
    metas = []
    hs = []
    for i in range(0, len(layers), max_per_call):
        m, h = encoder.fetch_hidden_states(url, parts, layers=layers[i : i + max_per_call])
        metas.append(m)
        hs.append(h)
    return metas[0], torch.cat(hs, dim=-1)


def collect_h3_multilayer(url, layers, text_targets, vision_targets=(), clip_32b_name=None, workers=4):
    """Fetch multi-layer source states, unpack onto target subwords with split metadata, and align vision."""
    t0 = time.perf_counter()
    tgt_tok = encoder.get_target_tokenizer("minimax_h3")
    if clip_32b_name:
        encoder.lookup_target_embeddings(clip_32b_name, [0])

    byte_mismatches = [0]

    def fetch_text(p_idx):
        item = text_targets[p_idx]
        parts = encoder.build_parts(item["prompt"])
        meta, h_full = _fetch_layers_chunked(url, parts, layers)
        raw_seq = h_full.shape[1]
        tok_texts = meta.get("token_texts") or []
        tok_blens = meta.get("token_byte_lens")
        start, end = encoder.find_user_span(tok_texts, raw_seq)
        u_h = h_full[0, start:end]
        if isinstance(tok_blens, list) and len(tok_blens) == raw_seq:
            src_blens = tok_blens[start:end]
            tgt_blens = encoder.target_token_byte_lens(tgt_tok, item["full_token_ids"])
            if sum(src_blens) != sum(tgt_blens):
                byte_mismatches[0] += 1
            unpacked, sp, sl, non1 = encoder.unpack_by_spans_with_info(u_h, src_blens, tgt_blens, n_to_1="last")
        else:
            u_texts = tok_texts[start:end]
            unpacked, sp, sl, non1 = encoder.unpack_by_spans_with_info(
                u_h,
                [len(s) for s in u_texts],
                [len(s) for s in item["full_token_texts"]],
                n_to_1="last",
            )
        unpacked = unpacked[1:]
        sp = sp[1:]
        sl = sl[1:]
        non1 = non1[1:]
        if clip_32b_name:
            emb = encoder.lookup_target_embeddings(clip_32b_name, item["token_ids"])
            x = torch.cat([unpacked, emb], dim=-1)
        else:
            x = unpacked
        y = item["h"].contiguous()
        non_sink = y.float().norm(dim=-1) < SINK_NORM_THRESHOLD
        return {
            "x": x.to(torch.bfloat16).contiguous(),
            "y": y,
            "typo": compute_dialogue_mask(item["token_texts"]),
            "valid": non_sink,
            "split_pos": sp.to(torch.uint8).contiguous(),
            "split_len": sl.to(torch.uint8).contiguous(),
            "non_1to1": non1.contiguous(),
            "token_texts": item["token_texts"],
            "prompt": item["prompt"],
            "prompt_idx": p_idx,
        }

    text_seqs = [None] * len(text_targets)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(fetch_text, range(len(text_targets))):
            text_seqs[res["prompt_idx"]] = res
            done += 1
            if done % 250 == 0 or done == len(text_targets):
                print(f"[SRC-TXT] Collected {done}/{len(text_targets)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    n_sinks_dropped = sum(int((~s["valid"]).sum().item()) for s in text_seqs)
    n_non1 = sum(int((s["non_1to1"] & s["valid"]).sum().item()) for s in text_seqs)
    n_dial = sum(int((s["typo"] & s["valid"]).sum().item()) for s in text_seqs)
    n_split_dial = sum(int((s["typo"] & s["non_1to1"] & s["valid"]).sum().item()) for s in text_seqs)
    print(
        f"[SRC-TXT] Byte-span mismatches: {byte_mismatches[0]}/{len(text_targets)} | "
        f"pos>=1 sinks masked: {n_sinks_dropped} | non-1:1 toks={n_non1} | "
        f"dial={n_dial} (split_dial={n_split_dial}, {100.0 * n_split_dial / max(1, n_dial):.1f}%)",
        flush=True,
    )

    x_vis_list, y_vis_list, vis_img_ids = [], [], []
    ref_text_seqs = []
    matched_imgs = 0
    t1 = time.perf_counter()
    d_vis = len(layers) * 5120
    d_txt = d_vis + (5120 if clip_32b_name else 0)
    for v_idx, item in enumerate(vision_targets):
        parts = encoder.build_parts(item["prompt"], images=[item["img_tensor"]])
        meta, h_full = _fetch_layers_chunked(url, parts, layers)
        raw_seq = h_full.shape[1]
        t_types = meta.get("token_types") or []
        tok_texts = meta.get("token_texts") or []
        tok_blens = meta.get("token_byte_lens")
        start, end = encoder.find_user_span(tok_texts, raw_seq)
        vis_pad = [i for i, t in enumerate(t_types) if int(t) != 0]
        if len(vis_pad) == item["vis_size"] and vis_pad:
            s0 = (
                vis_pad[0] - 1
                if vis_pad[0] > 0 and (not tok_texts or tok_texts[vis_pad[0] - 1] == "<|vision_start|>")
                else vis_pad[0]
            )
            s1 = (
                vis_pad[-1] + 2
                if vis_pad[-1] + 1 < raw_seq and (not tok_texts or tok_texts[vis_pad[-1] + 1] == "<|vision_end|>")
                else vis_pad[-1] + 1
            )
            if (s1 - s0) == item["vis_block_len"]:
                x_vis_list.append(h_full[0, s0:s1].to(torch.bfloat16))
                y_vis_list.append(item["h_vis"].to(torch.bfloat16))
                vis_img_ids.extend([v_idx] * (s1 - s0))
                matched_imgs += 1

                use_bytes = isinstance(tok_blens, list) and len(tok_blens) == raw_seq
                p_ids, suf_ids = item["prefix_ids"], item["suffix_ids"]
                if (
                    len(p_ids) == item["v0"]
                    and len(suf_ids) == (item["h_full"].shape[0] - item["v1"])
                    and s0 > start
                    and end >= s1
                ):
                    if use_bytes:
                        p_u, p_sp, p_sl, p_n1 = encoder.unpack_by_spans_with_info(
                            h_full[0, start:s0],
                            tok_blens[start:s0],
                            encoder.target_token_byte_lens(tgt_tok, p_ids),
                            n_to_1="last",
                        )
                        s_u, s_sp, s_sl, s_n1 = encoder.unpack_by_spans_with_info(
                            h_full[0, s1:end],
                            tok_blens[s1:end],
                            encoder.target_token_byte_lens(tgt_tok, suf_ids),
                            n_to_1="last",
                        )
                    else:
                        p_u, p_sp, p_sl, p_n1 = encoder.unpack_by_spans_with_info(
                            h_full[0, start:s0],
                            [len(s) for s in tok_texts[start:s0]],
                            [len(s) for s in item["prefix_texts"]],
                            n_to_1="last",
                        )
                        s_u, s_sp, s_sl, s_n1 = encoder.unpack_by_spans_with_info(
                            h_full[0, s1:end],
                            [len(s) for s in tok_texts[s1:end]],
                            [len(s) for s in item["suffix_texts"]],
                            n_to_1="last",
                        )
                    if clip_32b_name:
                        p_u = torch.cat([p_u, encoder.lookup_target_embeddings(clip_32b_name, p_ids)], dim=-1)
                        s_u = torch.cat([s_u, encoder.lookup_target_embeddings(clip_32b_name, suf_ids)], dim=-1)
                    L_full = item["h_full"].shape[0]
                    x_ref = torch.zeros(L_full, d_txt, dtype=torch.bfloat16)
                    x_ref[: item["v0"]] = p_u.to(torch.bfloat16)
                    x_ref[item["v1"] :] = s_u.to(torch.bfloat16)
                    sp_ref = torch.zeros(L_full, dtype=torch.uint8)
                    sl_ref = torch.zeros(L_full, dtype=torch.uint8)
                    n1_ref = torch.zeros(L_full, dtype=torch.bool)
                    sp_ref[: item["v0"]] = p_sp.to(torch.uint8)
                    sl_ref[: item["v0"]] = p_sl.to(torch.uint8)
                    n1_ref[: item["v0"]] = p_n1
                    sp_ref[item["v1"] :] = s_sp.to(torch.uint8)
                    sl_ref[item["v1"] :] = s_sl.to(torch.uint8)
                    n1_ref[item["v1"] :] = s_n1
                    valid_ref = torch.zeros(L_full, dtype=torch.bool)
                    valid_ref[: item["v0"]] = True
                    valid_ref[item["v1"] :] = True
                    y_sub = item["h_full"][1:].contiguous()
                    v_sub = valid_ref[1:] & (y_sub.float().norm(dim=-1) < SINK_NORM_THRESHOLD)
                    d_sub = torch.zeros(L_full - 1, dtype=torch.bool)
                    if item["v1"] - 1 < L_full - 1:
                        d_sub[item["v1"] - 1 :] = compute_dialogue_mask(item["suffix_texts"])
                    ref_text_seqs.append({
                        "x": x_ref[1:].contiguous(),
                        "y": y_sub,
                        "typo": d_sub,
                        "valid": v_sub,
                        "split_pos": sp_ref[1:].contiguous(),
                        "split_len": sl_ref[1:].contiguous(),
                        "non_1to1": n1_ref[1:].contiguous(),
                        "prompt_idx": -1 - v_idx,
                    })

    X_vis = torch.cat(x_vis_list, dim=0) if x_vis_list else torch.empty(0, d_vis, dtype=torch.bfloat16)
    Y_vis = torch.cat(y_vis_list, dim=0) if y_vis_list else torch.empty(0, 5120, dtype=torch.bfloat16)
    vis_ids = torch.tensor(vis_img_ids, dtype=torch.long)
    if vision_targets:
        print(
            f"[SRC-VIS] Matched {matched_imgs}/{len(vision_targets)} imgs "
            f"({X_vis.shape[0]} toks, {len(ref_text_seqs)} ref-gap seqs) in {time.perf_counter() - t1:.1f}s",
            flush=True,
        )
    return text_seqs, (X_vis, Y_vis, vis_ids, matched_imgs), ref_text_seqs


def _eval_stratified(P, T, M_dial, M_non1, m_out, s_out):
    """Compute stratified metrics across ALL, NON_DIAL, DIAL, EXACT_DIAL (1:1), and SPLIT_DIAL (non-1:1)."""
    m_non_dial = ~M_dial
    m_exact_dial = M_dial & ~M_non1
    m_split_dial = M_dial & M_non1

    def block(mask):
        cos, r2, rel, norm = eval_masked(P, T, mask)
        sr2 = eval_std_r2(P, T, m_out, s_out, mask)
        cnt = int(mask.sum().item()) if mask is not None else int(P.shape[0])
        return {"cos": cos, "r2": r2, "std_r2": sr2, "rel": rel, "norm": norm, "n": cnt}

    max_r = (P.norm(dim=-1) / T.norm(dim=-1).clamp(min=1e-6)).max().item()
    return {
        "all": block(None),
        "non_dial": block(m_non_dial),
        "dial": block(M_dial),
        "exact_dial": block(m_exact_dial),
        "split_dial": block(m_split_dial),
        "max_norm_ratio": max_r,
    }


def diagnose_adapter(adapter_path, te_seqs, server_url=None, clip_32b_name=None, probe_prompts=None, device="cuda"):
    """Check B diagnostic: measure EXACT_DIAL vs SPLIT_DIAL metrics on held-out sequences and
    inspect per-token accuracy on specific probe prompts."""
    proj = projector.load_projection(adapter_path)
    cache = {}
    m_out = proj["mean_out"].to(device)
    s_out = proj["std_out"].to(device)
    d_in = proj["mean_in"].shape[0]

    ps, ts, ds, n1s = [], [], [], []
    with torch.no_grad():
        for s in te_seqs:
            x = s["x"].float()
            if x.shape[-1] != d_in:
                raise ValueError(
                    f"Adapter {os.path.basename(adapter_path)} expects d_in={d_in}, cached sequence has {x.shape[-1]}"
                )
            # Prepend 1 dummy sink row so projector.project's sink_out slice [:, 1:] aligns with s['y']
            h_pad = torch.cat([torch.zeros(1, 1, d_in, dtype=torch.float32), x.unsqueeze(0)], dim=1).to(device)
            vm_pad = torch.cat([torch.ones(1, 1, dtype=torch.bool), s["valid"].unsqueeze(0)], dim=1).to(device)
            sp = s.get("split_pos", torch.zeros(x.shape[0], dtype=torch.uint8)).long()
            sl = s.get("split_len", torch.zeros(x.shape[0], dtype=torch.uint8)).long()
            sp_pad = torch.cat([torch.zeros(1, 1, dtype=torch.long), sp.unsqueeze(0)], dim=1).to(device)
            sl_pad = torch.cat([torch.zeros(1, 1, dtype=torch.long), sl.unsqueeze(0)], dim=1).to(device)
            pred = projector.project(h_pad, proj, cache, valid_mask=vm_pad, split_pos=sp_pad, split_len=sl_pad)[0, 1:]
            vm = s["valid"].to(device)
            ps.append(pred[vm])
            ts.append(s["y"].float().to(device)[vm])
            ds.append(s["typo"].to(device)[vm])
            n1 = s.get("non_1to1", torch.zeros_like(s["typo"])).to(device)
            n1s.append(n1[vm])

    P, T, D, N1 = torch.cat(ps, 0), torch.cat(ts, 0), torch.cat(ds, 0), torch.cat(n1s, 0)
    stats = _eval_stratified(P, T, D, N1, m_out, s_out)
    print(f"\n[DIAG-ADAPTER] {os.path.basename(adapter_path)}", flush=True)
    for key in ("all", "non_dial", "dial", "exact_dial", "split_dial"):
        b = stats[key]
        print(
            f"  {key.upper():10s} (n={b['n']:6d}): cos={b['cos']:.4f} R2={b['r2']:.4f} "
            f"stdR2={b['std_r2']:.4f} rel={b['rel']:.4f} norm={b['norm']:.3f}",
            flush=True,
        )
    print(f"  MAX_NORM_RATIO: {stats['max_norm_ratio']:.2f}x", flush=True)

    probe_report = []
    if probe_prompts and server_url and clip_32b_name:
        layers = json.loads(proj["layers"]) if isinstance(proj.get("layers"), str) else [int(proj["tap"])]
        tgt_tok = encoder.get_target_tokenizer("minimax_h3")
        embed_model = str(proj.get("target_embed_model", "")).strip() or clip_32b_name
        n_to_1 = str(proj.get("unpack_n_to_1", "last")).strip()
        text_targets, _, _ = extract_32b_targets_with_ids(clip_32b_name, [p["prompt"] for p in probe_prompts])
        for p_item, t_item in zip(probe_prompts, text_targets):
            parts = encoder.build_parts(t_item["prompt"])
            meta, h_full = encoder.fetch_hidden_states(server_url, parts, layers=layers)
            raw_seq = h_full.shape[1]
            tok_texts = meta.get("token_texts") or []
            tok_blens = meta.get("token_byte_lens")
            t_types = meta.get("token_types") or [0] * raw_seq
            start, end = encoder.find_user_span(tok_texts, raw_seq)
            u_h, out_texts, _, _, sp_1d, sl_1d = encoder.unpack_to_target_vocab(
                h_full[0, start:end],
                tok_texts[start:end],
                t_types[start:end],
                tgt_tok,
                embed_model=embed_model,
                src_byte_lens=tok_blens[start:end] if isinstance(tok_blens, list) and len(tok_blens) == raw_seq else None,
                orig_text_segs=[t_item["prompt"]],
                n_to_1=n_to_1,
                return_split_info=True,
            )
            with torch.no_grad():
                pred_full = projector.project(
                    u_h.unsqueeze(0).to(device),
                    proj,
                    cache,
                    split_pos=sp_1d.unsqueeze(0).to(device),
                    split_len=sl_1d.unsqueeze(0).to(device),
                )[0].cpu()
            pred_sub = pred_full[1 : 1 + t_item["h"].shape[0]]
            true_sub = t_item["h"].float()
            dial_m = compute_dialogue_mask(t_item["token_texts"])
            tok_cos = F.cosine_similarity(pred_sub, true_sub, dim=-1)
            tok_rel = (pred_sub - true_sub).norm(dim=-1) / true_sub.norm(dim=-1).clamp(min=1e-6)
            tok_norm = pred_sub.norm(dim=-1) / true_sub.norm(dim=-1).clamp(min=1e-6)
            sp_sub = sp_1d[1 : 1 + len(t_item["token_texts"])].tolist()
            sl_sub = sl_1d[1 : 1 + len(t_item["token_texts"])].tolist()
            print(f"\n[PROBE] {p_item.get('name', 'probe')}", flush=True)
            rows = []
            for idx_t, (tt, is_d, sp_v, sl_v, cv, rv, nv) in enumerate(
                zip(t_item["token_texts"], dial_m.tolist(), sp_sub, sl_sub, tok_cos.tolist(), tok_rel.tolist(), tok_norm.tolist())
            ):
                if is_d:
                    print(
                        f"  tok[{idx_t:2d}] {tt!r:14s} split=({sp_v}/{sl_v}) cos={cv:.4f} rel={rv:.4f} norm={nv:.3f}",
                        flush=True,
                    )
                    rows.append({
                        "idx": idx_t,
                        "token": tt,
                        "split_pos": sp_v,
                        "split_len": sl_v,
                        "cos": round(cv, 4),
                        "rel": round(rv, 4),
                        "norm_ratio": round(nv, 4),
                    })
            probe_report.append({"name": p_item.get("name", "probe"), "dialogue_tokens": rows})
    return {"stratified": stats, "probes": probe_report}


def greedy_layer_sweep(
    pool_layers,
    k_select,
    tr_seqs,
    te_seqs,
    has_emb=True,
    baseline_layers=(-2, 15, 43, 51, 59),
    device="cuda",
):
    """Greedy forward selection of K layers from `pool_layers` maximizing held-out DIAL stdR2."""
    Xtr_all, Ytr, _, _ = flatten_clean_seqs(tr_seqs)
    Xte_all, Yte, Dte, N1te = flatten_clean_seqs(te_seqs)
    n_pool = len(pool_layers)
    h_dim = 5120
    assert Xtr_all.shape[1] == n_pool * h_dim + (h_dim if has_emb else 0), (
        f"Expected width {n_pool * h_dim + (h_dim if has_emb else 0)}, got {Xtr_all.shape[1]}"
    )

    layer_to_slot = {l: i for i, l in enumerate(pool_layers)}

    def eval_subset(chosen_layers):
        cols = []
        for l in chosen_layers:
            s = layer_to_slot[l]
            cols.append(slice(s * h_dim, (s + 1) * h_dim))
        Xtr_sub = torch.cat([Xtr_all[:, c] for c in cols] + ([Xtr_all[:, -h_dim:]] if has_emb else []), dim=1)
        Xte_sub = torch.cat([Xte_all[:, c] for c in cols] + ([Xte_all[:, -h_dim:]] if has_emb else []), dim=1)
        d_main = len(chosen_layers) * h_dim
        xtx, xty, m_in, s_in, m_out, s_out = compute_gram(Xtr_sub, Ytr, device=device)
        Xte_n = ((Xte_sub.float().to(device) - m_in.to(device)) / s_in.to(device)).float()
        Yte_g = Yte.float().to(device)
        Dte_g = Dte.to(device)
        N1te_g = N1te.to(device)
        mo_g, so_g = m_out.to(device), s_out.to(device)

        best = None
        for lm in (10000.0, 30000.0, 100000.0):
            for le in ((3000.0, 10000.0) if has_emb else (0.0,)):
                W = solve_block_ridge(xtx, xty, d_main, lm, le, device=device)
                pred = (Xte_n @ W) * so_g + mo_g
                st = _eval_stratified(pred, Yte_g, Dte_g, N1te_g, mo_g, so_g)
                obj = st["dial"]["std_r2"]
                if best is None or obj > best["obj"]:
                    best = {"obj": obj, "lm": lm, "le": le, "stats": st}
        del xtx, xty, Xte_n, Xtr_sub, Xte_sub
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best

    print(f"\n=== Greedy Forward Layer Sweep (pool={pool_layers}, K={k_select}) ===", flush=True)
    t0 = time.perf_counter()
    selected = []
    history = []
    for step in range(k_select):
        best_cand = None
        for cand in pool_layers:
            if cand in selected:
                continue
            trial = selected + [cand]
            res = eval_subset(trial)
            st = res["stats"]
            print(
                f"  [SWEEP k={step + 1}] {trial} -> DIAL stdR2={st['dial']['std_r2']:.4f} "
                f"(EXACT={st['exact_dial']['std_r2']:.4f} SPLIT={st['split_dial']['std_r2']:.4f} "
                f"ALL={st['all']['std_r2']:.4f} DIAL_cos={st['dial']['cos']:.4f})",
                flush=True,
            )
            if best_cand is None or res["obj"] > best_cand["obj"]:
                best_cand = {"layer": cand, "trial": trial, **res}
        selected.append(best_cand["layer"])
        history.append({
            "step": step + 1,
            "selected": list(selected),
            "dial_std_r2": round(best_cand["stats"]["dial"]["std_r2"], 6),
            "exact_dial_std_r2": round(best_cand["stats"]["exact_dial"]["std_r2"], 6),
            "split_dial_std_r2": round(best_cand["stats"]["split_dial"]["std_r2"], 6),
            "all_std_r2": round(best_cand["stats"]["all"]["std_r2"], 6),
            "dial_cos": round(best_cand["stats"]["dial"]["cos"], 6),
        })
        print(f"  => Step {step + 1} winner: {selected} (DIAL stdR2={best_cand['obj']:.4f})", flush=True)

    baseline_res = None
    if baseline_layers and all(l in layer_to_slot for l in baseline_layers):
        baseline_res = eval_subset(list(baseline_layers))
        st = baseline_res["stats"]
        print(
            f"  [BASELINE] {list(baseline_layers)} -> DIAL stdR2={st['dial']['std_r2']:.4f} "
            f"(EXACT={st['exact_dial']['std_r2']:.4f} SPLIT={st['split_dial']['std_r2']:.4f} "
            f"ALL={st['all']['std_r2']:.4f} DIAL_cos={st['dial']['cos']:.4f})",
            flush=True,
        )

    # Sort selected layers in ascending depth order (keeping -2 first as embedding layer)
    ordered = sorted(selected, key=lambda x: (-1 if x == -2 else (999 if x == -1 else x)))
    print(f"[SWEEP DONE] {time.perf_counter() - t0:.1f}s | Selected layers (ordered): {ordered}", flush=True)
    return {
        "pool": list(pool_layers),
        "selected": ordered,
        "history": history,
        "baseline_layers": list(baseline_layers) if baseline_res else None,
        "baseline": baseline_res["stats"] if baseline_res else None,
    }


def compute_oof_bases_clean(tr_seqs, te_seqs, ref_seqs, ridge, d_main, n_folds=5, device="cuda"):
    """5-fold OOF Ridge predictions with strict non-sink masking inside every fold."""
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
        Xtr, Ytr, _, _ = flatten_clean_seqs(trn_s)
        xtx, xty, m_in, s_in, m_out, s_out = compute_gram(Xtr, Ytr, device=device)
        W_f = solve_block_ridge(xtx, xty, d_main, lm, le, device=device)
        mi_g, si_g = m_in.to(device), s_in.to(device)
        mo_g, so_g = m_out.to(device), s_out.to(device)
        with torch.no_grad():
            for offset, s in enumerate(val_s):
                xn = (s["x"].float().to(device) - mi_g) / si_g
                pred_raw = (xn @ W_f) * so_g + mo_g
                pred_n = (pred_raw - m_out_full) / s_out_full
                oof_bases[v0 + offset] = pred_n.to(torch.bfloat16).cpu()
        del Xtr, Ytr, xtx, xty, W_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mi_g, si_g = ridge["mean_in"].to(device), ridge["std_in"].to(device)
    W_g = ridge["W"].to(device)
    te_bases = []
    ref_bases = []
    with torch.no_grad():
        for s in te_seqs:
            xn = (s["x"].float().to(device) - mi_g) / si_g
            te_bases.append((xn @ W_g).to(torch.bfloat16).cpu())
        for s in ref_seqs:
            xn = (s["x"].float().to(device) - mi_g) / si_g
            bn = xn @ W_g
            bn[~s["valid"].to(device)] = 0.0
            ref_bases.append(bn.to(torch.bfloat16).cpu())

    oof_p = torch.cat(
        [((b.float().to(device) * s_out_full + m_out_full)[s["valid"].to(device)]).cpu() for b, s in zip(oof_bases, tr_seqs)],
        0,
    )
    oof_y = torch.cat([s["y"][s["valid"]].float() for s in tr_seqs], 0)
    oof_t = torch.cat([s["typo"][s["valid"]] for s in tr_seqs], 0)
    ac, ar, _, _ = eval_masked(oof_p, oof_y)
    dc, dr, _, _ = eval_masked(oof_p, oof_y, oof_t)
    print(
        f"[OOF {n_folds}-fold] {time.perf_counter() - t0:.1f}s | OOF ALL cos={ac:.4f} R2={ar:.4f} | "
        f"OOF DIAL cos={dc:.4f} R2={dr:.4f}",
        flush=True,
    )
    return oof_bases, te_bases, ref_bases


def train_oof_seq_clean(
    tr_seqs,
    te_seqs,
    oof_bases,
    te_bases,
    ridge,
    ref_seqs=(),
    ref_bases=(),
    seq_input="base_plus_emb",
    use_split_emb=True,
    d_model=1024,
    n_layers=4,
    n_heads=16,
    epochs=40,
    batch_size=32,
    lr=4e-4,
    dial_weight=3.5,
    split_dial_weight=5.5,
    device="cuda",
):
    """Train OOF Bidirectional Refinement Connector with optional `base_plus_emb` and `(split_pos, split_len)`
    embeddings, logging train/val loss and stratified ALL/NON_DIAL/EXACT_DIAL/SPLIT_DIAL metrics."""
    d_out = ridge["W"].shape[1]
    m_out = ridge["mean_out"].to(device)
    s_out = ridge["std_out"].to(device)
    emb_m_in = ridge["mean_in"][-d_out:].float()
    emb_s_in = ridge["std_in"][-d_out:].float()

    use_emb = seq_input == "base_plus_emb"
    d_seq_in = (2 * d_out) if use_emb else d_out

    all_tr_seqs = list(tr_seqs) + list(ref_seqs)
    all_tr_bases = list(oof_bases) + list(ref_bases)

    net = seq_connector.BidirectionalSeqConnector(
        d_in=d_seq_in,
        d_out=d_out,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_mult=2,
        conv_kernel=5,
        dropout=0.15,
        use_split_emb=use_split_emb,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"[REFINE-SEQ] Params: {n_params / 1e6:.1f}M ({d_seq_in} -> d_model={d_model} x {n_layers}L "
        f"h={n_heads} -> {d_out}) | seq_input={seq_input} | use_split_emb={use_split_emb} | "
        f"weights=(non_dial=1.0, exact_dial={dial_weight}, split_dial={split_dial_weight}) | "
        f"train_seqs={len(all_tr_seqs)} (incl {len(ref_seqs)} ref-gap)",
        flush=True,
    )

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
        Sin = torch.zeros(B, Lmax, d_seq_in, dtype=torch.bfloat16) if use_emb else None
        Y = torch.zeros(B, Lmax, d_out, dtype=torch.bfloat16)
        valid = torch.zeros(B, Lmax, dtype=torch.bool)
        typo = torch.zeros(B, Lmax, dtype=torch.bool)
        non1 = torch.zeros(B, Lmax, dtype=torch.bool)
        sp = torch.zeros(B, Lmax, dtype=torch.long)
        sl = torch.zeros(B, Lmax, dtype=torch.long)
        for i, k in enumerate(idxs):
            s = seqs[k]
            L = s["y"].shape[0]
            vm = s["valid"]
            b = bases[k].clone()
            b[~vm] = 0
            Base[i, :L] = b
            if use_emb:
                emb_n = (s["x"][..., -d_out:].float() - emb_m_in) / emb_s_in
                emb_n[~vm] = 0.0
                Sin[i, :L, :d_out] = b
                Sin[i, :L, d_out:] = emb_n.to(torch.bfloat16)
            Y[i, :L] = s["y"]
            valid[i, :L] = vm
            typo[i, :L] = s["typo"]
            n1 = s.get("non_1to1")
            if n1 is not None:
                non1[i, :L] = n1
            s_sp = s.get("split_pos")
            s_sl = s.get("split_len")
            if s_sp is not None and s_sl is not None:
                sp[i, :L] = s_sp.long() * vm.long()
                sl[i, :L] = s_sl.long() * vm.long()
        base_g = Base.to(device, non_blocking=True)
        sin_g = Sin.to(device, non_blocking=True) if use_emb else base_g
        return (
            base_g,
            sin_g,
            Y.to(device, non_blocking=True),
            valid.to(device, non_blocking=True),
            typo.to(device, non_blocking=True),
            non1.to(device, non_blocking=True),
            sp.to(device, non_blocking=True),
            sl.to(device, non_blocking=True),
        )

    def compute_loss(pred_n, pred, yt, yn_true, valid, typo, non1):
        pn_a, yn_a = pred_n[valid], yn_true[valid]
        p_a, y_a = pred[valid], yt[valid]
        d_a, n1_a = typo[valid], non1[valid]
        w = torch.ones_like(pn_a[:, 0])
        w = torch.where(d_a & ~n1_a, dial_weight, w)
        w = torch.where(d_a & n1_a, split_dial_weight, w)
        w = w / w.mean().clamp(min=1e-6)
        tok_huber = F.smooth_l1_loss(pn_a, yn_a, reduction="none", beta=1.5).mean(dim=-1)
        tok_cos = 1.0 - F.cosine_similarity(p_a, y_a, dim=-1)
        tok_rel = (p_a - y_a).norm(dim=-1) / y_a.norm(dim=-1).clamp(min=1e-6)
        return (w * (tok_huber + 1.0 * tok_cos + 0.35 * tok_rel)).mean()

    def evaluate():
        net.eval()
        ps, ts, tm, n1m = [], [], [], []
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, len(te_seqs), batch_size):
                idxs = list(range(i, min(len(te_seqs), i + batch_size)))
                Base, Sin, Y, valid, typo, non1, sp, sl = make_batch(te_seqs, te_bases, idxs)
                yt = Y.float()
                yn_true = (yt - m_out) / s_out
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    delta = net(Sin, valid_mask=valid, split_pos=sp, split_len=sl)
                pred_n = Base.float() + delta.float()
                pred = pred_n * s_out + m_out
                val_loss_sum += float(compute_loss(pred_n, pred, yt, yn_true, valid, typo, non1).item())
                val_batches += 1
                ps.append(pred[valid])
                ts.append(yt[valid])
                tm.append(typo[valid])
                n1m.append(non1[valid])
        P, T, M, N1 = torch.cat(ps, 0), torch.cat(ts, 0), torch.cat(tm, 0), torch.cat(n1m, 0)
        stats = _eval_stratified(P, T, M, N1, m_out, s_out)
        stats["val_loss"] = val_loss_sum / max(1, val_batches)
        return stats

    def score_stats(st):
        return (
            0.20 * st["all"]["cos"]
            + 0.20 * st["all"]["std_r2"]
            + 0.25 * st["dial"]["cos"]
            + 0.15 * st["exact_dial"]["std_r2"]
            + 0.20 * st["split_dial"]["std_r2"]
        )

    b = evaluate()
    print(
        f"  [REFINE] ep  0/{epochs} (Ridge) val_loss={b['val_loss']:.4f} | "
        f"ALL cos={b['all']['cos']:.4f} stdR2={b['all']['std_r2']:.4f} | "
        f"NON_DIAL stdR2={b['non_dial']['std_r2']:.4f} | "
        f"EXACT_DIAL cos={b['exact_dial']['cos']:.4f} stdR2={b['exact_dial']['std_r2']:.4f} | "
        f"SPLIT_DIAL cos={b['split_dial']['cos']:.4f} stdR2={b['split_dial']['std_r2']:.4f} | "
        f"max={b['max_norm_ratio']:.2f}x",
        flush=True,
    )
    best_score = score_stats(b)
    best_stats = b
    best_state = None
    rng = random.Random(42)
    t0 = time.perf_counter()

    for ep in range(epochs):
        net.train()
        order = list(range(len(all_tr_seqs)))
        rng.shuffle(order)
        tr_loss_sum = 0.0
        tr_batches = 0
        for i in range(0, len(order), batch_size):
            Base, Sin, Y, valid, typo, non1, sp, sl = make_batch(all_tr_seqs, all_tr_bases, order[i : i + batch_size])
            yt = Y.float()
            yn_true = (yt - m_out) / s_out
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                delta = net(Sin, valid_mask=valid, split_pos=sp, split_len=sl)
                pred_n = Base.float() + delta.float()
                pred = pred_n * s_out + m_out
                loss = compute_loss(pred_n, pred, yt, yn_true, valid, typo, non1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_loss_sum += float(loss.item())
            tr_batches += 1
        sched.step()

        tr_loss = tr_loss_sum / max(1, tr_batches)
        cur = evaluate()
        cur["train_loss"] = tr_loss
        sc = score_stats(cur)
        if sc > best_score:
            best_score = sc
            best_stats = cur
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep + 1 == epochs or ep == 0:
            print(
                f"  [REFINE] ep {ep + 1:2d}/{epochs} tr_loss={tr_loss:.4f} val_loss={cur['val_loss']:.4f} | "
                f"ALL cos={cur['all']['cos']:.4f} stdR2={cur['all']['std_r2']:.4f} | "
                f"NON_DIAL stdR2={cur['non_dial']['std_r2']:.4f} | "
                f"EXACT_DIAL cos={cur['exact_dial']['cos']:.4f} stdR2={cur['exact_dial']['std_r2']:.4f} | "
                f"SPLIT_DIAL cos={cur['split_dial']['cos']:.4f} stdR2={cur['split_dial']['std_r2']:.4f} | "
                f"max={cur['max_norm_ratio']:.2f}x ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    if best_state is None:
        return None, best_stats
    net.load_state_dict(best_state)
    return net, best_stats


def save_h3_adapter(
    path,
    ridge_txt,
    ridge_vis,
    sink_out,
    clamp_lo,
    clamp_hi,
    layers,
    clip_32b_name,
    source_model,
    meta_extra,
    net=None,
    seq_input="base_plus_emb",
    source_file="",
    source_sha256="",
):
    tensors = {
        "W": ridge_txt["W"].half().contiguous(),
        "mean_in": ridge_txt["mean_in"].half().contiguous(),
        "std_in": ridge_txt["std_in"].half().contiguous(),
        "mean_out": ridge_txt["mean_out"].half().contiguous(),
        "std_out": ridge_txt["std_out"].half().contiguous(),
        "sink_out": sink_out.half().contiguous(),
        "clamp_lo": clamp_lo.half().contiguous(),
        "clamp_hi": clamp_hi.half().contiguous(),
    }
    if ridge_vis is not None:
        tensors["W_vis"] = ridge_vis["W"].half().contiguous()
        tensors["mean_in_vis"] = ridge_vis["mean_in"].half().contiguous()
        tensors["std_in_vis"] = ridge_vis["std_in"].half().contiguous()
        tensors["mean_out_vis"] = ridge_vis["mean_out"].half().contiguous()
        tensors["std_out_vis"] = ridge_vis["std_out"].half().contiguous()
    clip_base = os.path.basename(clip_32b_name) if clip_32b_name else ""
    meta = {
        "target_profile": "minimax_h3",
        "unpack_target_vocab": "minimax_h3",
        "unpack_n_to_1": "last",
        "target_embed_model": clip_base,
        "tap": str(int(layers[-1])),
        "layers": json.dumps([int(x) for x in layers]),
        "d_in": str(int(ridge_txt["W"].shape[0])),
        "d_out": str(int(ridge_txt["W"].shape[1])),
        "lambda": str(ridge_txt["lam_main"]),
        "lambda_emb": str(ridge_txt["lam_emb"]),
        "source_model": source_model,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "target_model": clip_base or "qwen3vl_32b_minimax_h3",
    }
    if net is not None:
        tensors.update(seq_connector.state_dict_prefixed(net, prefix="seq.", dtype=torch.bfloat16))
        meta["connector_type"] = "seq_base_emb_refine_v2" if seq_input == "base_plus_emb" else "seq_base_refine_v1"
        meta["seq_input"] = seq_input
        meta["seq_config"] = json.dumps(net.config)
    meta.update({k: str(v) for k, v in meta_extra.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata=meta)
    print(f"[SAVE] {path} ({os.path.getsize(path) / (1024 * 1024):.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Calibrate and diagnose MiniMax H3 Universal Conditioner adapters.")
    parser.add_argument("--server-url", default=encoder.DEFAULT_URL)
    parser.add_argument("--source-model", default="neroued/Qwen3.8-27B-nvfp4-NInfer")
    parser.add_argument("--source-file", default="")
    parser.add_argument("--source-sha256", default="")
    parser.add_argument("--clip-32b", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    parser.add_argument("--prompts-json", default="")
    parser.add_argument("--n-prompts", type=int, default=5200)
    parser.add_argument("--n-vision", type=int, default=80)
    parser.add_argument("--vision-dir", default="")
    parser.add_argument("--layers", default="-2,15,43,51,59")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seq-input", default="base_plus_emb", choices=("base_plus_emb", "base"))
    parser.add_argument("--no-split-emb", action="store_true")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dial-weight", type=float, default=3.5)
    parser.add_argument("--split-dial-weight", type=float, default=5.5)
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--extract-only", action="store_true", help="Save --cache-path and exit before fitting Ridge/connector")
    parser.add_argument("--out-dir", default="./adapters/minimax-h3")
    parser.add_argument("--out-prefix", default="unicond-h3-v4.3")
    parser.add_argument("--diag-adapter", default="", help="Run Check B stratified + probe diagnostics on an existing adapter")
    parser.add_argument("--diag-all", action="store_true", help="Evaluate all loaded sequences in --diag-adapter mode instead of only the last 10%")
    parser.add_argument("--probe-prompts-json", default="", help="Optional JSON list of {name, prompt} for per-token diagnostics")
    parser.add_argument("--diag-out", default="", help="Optional JSON output path for --diag-adapter report")
    parser.add_argument("--layer-sweep-pool", default="", help="Comma-separated candidate layers for greedy forward selection")
    parser.add_argument("--layer-sweep-k", type=int, default=5)
    parser.add_argument("--sweep-prompts", type=int, default=600)
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--layer-sweep-out", default="")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if args.diag_adapter and not (args.cache_path and os.path.isfile(args.cache_path)):
        _diag_proj = projector.load_projection(args.diag_adapter)
        _raw_layers = _diag_proj.get("layers")
        if isinstance(_raw_layers, str) and _raw_layers.strip():
            layers = [int(x) for x in json.loads(_raw_layers)]
        elif isinstance(_raw_layers, (list, tuple)) and _raw_layers:
            layers = [int(x) for x in _raw_layers]
        elif "tap" in _diag_proj:
            layers = [int(_diag_proj["tap"])]
        del _diag_proj

    if args.layer_sweep_pool:
        pool_layers = [int(x.strip()) for x in args.layer_sweep_pool.split(",") if x.strip()]
        if not args.prompts_json:
            raise ValueError("--prompts-json is required for --layer-sweep-pool")
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        prompts = [x["prompt"] if isinstance(x, dict) else str(x) for x in loaded]
        random.Random(42).shuffle(prompts)
        prompts = prompts[: args.sweep_prompts]
        print(f"[SWEEP-CORPUS] {len(prompts)} prompts across pool {pool_layers}", flush=True)
        text_targets, _, _ = extract_32b_targets_with_ids(args.clip_32b, prompts, vision_items=())
        sweep_seqs, _, _ = collect_h3_multilayer(
            args.server_url, pool_layers, text_targets, vision_targets=(), clip_32b_name=args.clip_32b, workers=args.workers
        )
        n_tr_s = int(len(sweep_seqs) * 0.85)
        sweep_report = greedy_layer_sweep(
            pool_layers,
            args.layer_sweep_k,
            sweep_seqs[:n_tr_s],
            sweep_seqs[n_tr_s:],
            has_emb=True,
            baseline_layers=tuple(layers),
        )
        if args.layer_sweep_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.layer_sweep_out)), exist_ok=True)
            with open(args.layer_sweep_out, "w", encoding="utf-8") as f:
                json.dump(sweep_report, f, indent=2)
        if args.sweep_only:
            return
        layers = sweep_report["selected"]
        del sweep_seqs, text_targets
        gc.collect()

    d_main = len(layers) * 5120

    if args.cache_path and os.path.isfile(args.cache_path):
        print(f"[CACHE] Loading {args.cache_path} ...", flush=True)
        blob = torch.load(args.cache_path, weights_only=True)
        text_seqs = blob["text_seqs"]
        X_vis, Y_vis, vis_ids, matched_imgs = blob["vis"]
        ref_text_seqs = blob.get("ref_text_seqs", [])
        sink_out = blob["sink_out"]
        cached_layers = blob.get("layers")
        if cached_layers is not None:
            layers = [int(x) for x in cached_layers]
            d_main = len(layers) * 5120
    else:
        if not args.prompts_json:
            raise ValueError("--prompts-json is required when --cache-path does not exist")
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        prompts = [x["prompt"] if isinstance(x, dict) else str(x) for x in loaded]
        random.Random(42).shuffle(prompts)
        prompts = prompts[: args.n_prompts]
        print(f"[CORPUS] Loaded {len(prompts)} shuffled prompts from {os.path.basename(args.prompts_json)}", flush=True)

        raw_imgs = collect_sample_images(args.n_vision, search_dir=args.vision_dir) if args.vision_dir else []
        vision_items = [(name, img, prompts[i % len(prompts)]) for i, (name, img) in enumerate(raw_imgs)]
        text_targets, vision_targets, sink_out = extract_32b_targets_with_ids(args.clip_32b, prompts, vision_items)
        text_seqs, (X_vis, Y_vis, vis_ids, matched_imgs), ref_text_seqs = collect_h3_multilayer(
            args.server_url,
            layers,
            text_targets,
            vision_targets,
            clip_32b_name=args.clip_32b,
            workers=args.workers,
        )
        if args.cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)), exist_ok=True)
            torch.save(
                {
                    "layers": layers,
                    "text_seqs": text_seqs,
                    "vis": (X_vis, Y_vis, vis_ids, matched_imgs),
                    "ref_text_seqs": ref_text_seqs,
                    "sink_out": sink_out,
                },
                args.cache_path,
            )
            print(f"[CACHE] Saved {args.cache_path}", flush=True)
        if args.extract_only:
            print("[EXTRACT-ONLY] Cache saved; exiting before training.", flush=True)
            return

    n_tr = int(len(text_seqs) * 0.9)
    tr_seqs, te_seqs = text_seqs[:n_tr], text_seqs[n_tr:]

    if args.diag_adapter:
        probe_prompts = None
        if args.probe_prompts_json and os.path.isfile(args.probe_prompts_json):
            with open(args.probe_prompts_json, "r", encoding="utf-8") as f:
                probe_prompts = json.load(f)
        diag_seqs = text_seqs if args.diag_all else te_seqs
        report = diagnose_adapter(
            args.diag_adapter,
            diag_seqs,
            server_url=args.server_url,
            clip_32b_name=args.clip_32b,
            probe_prompts=probe_prompts,
        )
        if args.diag_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.diag_out)), exist_ok=True)
            with open(args.diag_out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        return

    Xtr, Ytr, _, _ = flatten_clean_seqs(tr_seqs)
    Xte, Yte, Dte, N1te = flatten_clean_seqs(te_seqs)
    print(
        f"[SPLIT] train={len(tr_seqs)} seqs ({Xtr.shape[0]} clean toks) | "
        f"test={len(te_seqs)} seqs ({Xte.shape[0]} clean toks, {int(Dte.sum())} dial, "
        f"{int((Dte & N1te).sum())} split_dial) | ref_gap={len(ref_text_seqs)} | d_in={Xtr.shape[1]}",
        flush=True,
    )

    ridge_txt = fit_two_block_ridge(
        Xtr,
        Ytr,
        Xte,
        Yte,
        Dte,
        d_main=d_main,
        lam_mains=(10000.0, 30000.0, 100000.0),
        lam_embs=(3000.0, 10000.0, 30000.0),
    )
    W_diag = ridge_txt["W"]
    blk_names = [f"L{l}" for l in layers] + (["e_tgt"] if W_diag.shape[0] > d_main else [])
    blk_norms = [f"{nm}={W_diag[i * 5120 : (i + 1) * 5120].norm().item():.1f}" for i, nm in enumerate(blk_names)]
    print("[W-BLOCK-NORMS] " + " | ".join(blk_norms), flush=True)

    Ytr_f = Ytr.float()
    clamp_lo = Ytr_f.min(dim=0).values - 2.0 * ridge_txt["std_out"]
    clamp_hi = Ytr_f.max(dim=0).values + 2.0 * ridge_txt["std_out"]
    del Ytr_f

    ridge_vis = None
    if X_vis.shape[0] > 0:
        n_v_tr = max(1, int(matched_imgs * 0.85))
        tr_v = vis_ids < n_v_tr
        te_v = ~tr_v
        if tr_v.any() and te_v.any():
            ridge_vis = fit_ridge(
                X_vis[tr_v],
                Y_vis[tr_v],
                X_vis[te_v],
                Y_vis[te_v],
                lambdas=(100000.0, 300000.0, 1000000.0, 3000000.0),
            )
            print(
                f"[Dual Multi-Layer Vision Ridge] cos={ridge_vis['cos']:.4f} "
                f"r2={ridge_vis['r2']:.4f} (lam={ridge_vis['lambda']:.0f})",
                flush=True,
            )

    save_h3_adapter(
        os.path.join(args.out_dir, f"{args.out_prefix}-dual-ridge.safetensors"),
        ridge_txt,
        ridge_vis,
        sink_out,
        clamp_lo,
        clamp_hi,
        layers,
        args.clip_32b,
        args.source_model,
        {
            "n_train_prompts": len(tr_seqs),
            "n_train_tokens": int(Xtr.shape[0]),
            "cos_test": round(ridge_txt["cos"], 6),
            "r2_test": round(ridge_txt["r2"], 6),
            "dial_cos_test": round(ridge_txt["typo_cos"], 6),
            "dial_r2_test": round(ridge_txt["typo_r2"], 6),
            "cos_test_vis": round(ridge_vis["cos"], 6) if ridge_vis else 0.0,
            "r2_test_vis": round(ridge_vis["r2"], 6) if ridge_vis else 0.0,
        },
        net=None,
        source_file=args.source_file,
        source_sha256=args.source_sha256,
    )
    del Xtr, Ytr, Xte, Yte
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Computing 5-Fold Out-of-Fold Clean Ridge Bases ===", flush=True)
    oof_bases, te_bases, ref_bases = compute_oof_bases_clean(
        tr_seqs, te_seqs, ref_text_seqs, ridge_txt, d_main=d_main, n_folds=5
    )

    print(f"\n=== Training OOF Bidirectional Refinement Connector ({args.epochs} epochs) ===", flush=True)
    net, best_st = train_oof_seq_clean(
        tr_seqs,
        te_seqs,
        oof_bases,
        te_bases,
        ridge_txt,
        ref_seqs=ref_text_seqs,
        ref_bases=ref_bases,
        seq_input=args.seq_input,
        use_split_emb=not args.no_split_emb,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dial_weight=args.dial_weight,
        split_dial_weight=args.split_dial_weight,
    )
    if net is not None:
        save_h3_adapter(
            os.path.join(args.out_dir, f"{args.out_prefix}-dual-seq.safetensors"),
            ridge_txt,
            ridge_vis,
            sink_out,
            clamp_lo,
            clamp_hi,
            layers,
            args.clip_32b,
            args.source_model,
            {
                "n_train_prompts": len(tr_seqs),
                "n_train_tokens": int(sum(int(s["valid"].sum().item()) for s in tr_seqs)),
                "cos_test": round(best_st["all"]["cos"], 6),
                "r2_test": round(best_st["all"]["r2"], 6),
                "std_r2_test": round(best_st["all"]["std_r2"], 6),
                "non_dial_cos_test": round(best_st["non_dial"]["cos"], 6),
                "non_dial_std_r2_test": round(best_st["non_dial"]["std_r2"], 6),
                "dial_cos_test": round(best_st["dial"]["cos"], 6),
                "dial_r2_test": round(best_st["dial"]["r2"], 6),
                "dial_std_r2_test": round(best_st["dial"]["std_r2"], 6),
                "exact_dial_cos_test": round(best_st["exact_dial"]["cos"], 6),
                "exact_dial_std_r2_test": round(best_st["exact_dial"]["std_r2"], 6),
                "split_dial_cos_test": round(best_st["split_dial"]["cos"], 6),
                "split_dial_std_r2_test": round(best_st["split_dial"]["std_r2"], 6),
                "max_norm_ratio_test": round(best_st["max_norm_ratio"], 4),
                "cos_test_vis": round(ridge_vis["cos"], 6) if ridge_vis else 0.0,
                "r2_test_vis": round(ridge_vis["r2"], 6) if ridge_vis else 0.0,
            },
            net=net,
            seq_input=args.seq_input,
            source_file=args.source_file,
            source_sha256=args.source_sha256,
        )
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
