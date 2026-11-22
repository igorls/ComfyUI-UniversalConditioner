"""Shared Ridge math, FP32 regularized Cholesky solver, and masked evaluation helpers."""

import os
import time

import numpy as np
from PIL import Image
import torch


def evaluate_predictions(y_pred, y_true):
    """Return (mean_cosine, pooled_r2) on float32 tensors [N, D]."""
    p = y_pred.float()
    t = y_true.float()
    cos = torch.nn.functional.cosine_similarity(p, t, dim=-1).mean().item()
    ss_res = (t - p).pow(2).sum()
    ss_tot = (t - t.mean(dim=0, keepdim=True)).pow(2).sum().clamp(min=1e-6)
    r2 = 1.0 - (ss_res / ss_tot).item()
    return cos, r2


def eval_masked(y_pred, y_true, mask=None):
    """Return (cos, r2, rel_l2, norm_ratio) on float32 tensors."""
    if mask is not None:
        if not mask.any():
            return 0.0, 0.0, 0.0, 1.0
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    cos, r2 = evaluate_predictions(y_pred, y_true)
    t_norm = y_true.norm(dim=-1).clamp(min=1e-6)
    p_norm = y_pred.norm(dim=-1)
    rel_l2 = ((y_pred - y_true).norm(dim=-1) / t_norm).mean().item()
    norm_ratio = (p_norm / t_norm).mean().item()
    return cos, r2, rel_l2, norm_ratio


def eval_std_r2(y_pred, y_true, m_out, s_out, mask=None):
    """Pooled R2 in standardized output space (immune to single-channel variance skew)."""
    if mask is not None:
        if not mask.any():
            return 0.0
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    pn = (y_pred - m_out) / s_out
    yn = (y_true - m_out) / s_out
    num = (pn - yn).pow(2).sum()
    den = (yn - yn.mean(dim=0, keepdim=True)).pow(2).sum().clamp(min=1e-6)
    return 1.0 - (num / den).item()


def compute_gram(X_train, Y_train, device="cuda"):
    """Compute standardized (xtx, xty, mean_in, std_in, mean_out, std_out) with FP32 chunk matmuls
    accumulated into FP64 Gram matrices."""
    Xt = X_train.float()
    Yt = Y_train.float()
    mean_in = Xt.mean(dim=0)
    std_in = Xt.std(dim=0).clamp(min=1e-6)
    mean_out = Yt.mean(dim=0)
    std_out = Yt.std(dim=0).clamp(min=1e-6)

    d_in, d_out = Xt.shape[1], Yt.shape[1]
    xtx = torch.zeros(d_in, d_in, dtype=torch.float64, device=device)
    xty = torch.zeros(d_in, d_out, dtype=torch.float64, device=device)
    m_in, s_in = mean_in.to(device), std_in.to(device)
    m_out, s_out = mean_out.to(device), std_out.to(device)

    chunk = 8192
    for i in range(0, Xt.shape[0], chunk):
        xb = (Xt[i : i + chunk].to(device) - m_in) / s_in
        yb = (Yt[i : i + chunk].to(device) - m_out) / s_out
        xtx.add_((xb.T @ xb).double())
        xty.add_((xb.T @ yb).double())
    return xtx, xty, mean_in, std_in, mean_out, std_out


def solve_block_ridge(xtx, xty, d_main, lam_main, lam_emb, device="cuda"):
    """Solve SPD (X^T X + diag(lam_main..., lam_emb...)) W = X^T Y via fast FP32 Cholesky
    with automatic FP64 fallback if ill-conditioned."""
    d_in = xtx.shape[0]
    reg = torch.full((d_in,), float(lam_main), dtype=torch.float64, device=device)
    if d_main < d_in:
        reg[d_main:] = float(lam_emb)
    A64 = xtx.clone()
    A64.diagonal().add_(reg)
    A32 = A64.float()
    del A64
    L, info = torch.linalg.cholesky_ex(A32)
    if int(info.item()) == 0:
        W = torch.cholesky_solve(xty.float(), L)
        if torch.isfinite(W).all():
            return W
    A64 = xtx.clone()
    A64.diagonal().add_(reg)
    return torch.linalg.solve(A64, xty).float()


def fit_two_block_ridge(
    X_tr, Y_tr, X_te, Y_te, T_te, d_main, E_te=None, device="cuda", lam_mains=None, lam_embs=None
):
    """Sweep independent (lam_main, lam_emb) regularization on GPU."""
    t0 = time.perf_counter()
    xtx, xty, m_in, s_in, m_out, s_out = compute_gram(X_tr, Y_tr, device=device)
    d_in = X_tr.shape[1]
    has_emb = d_main < d_in
    has_edits = bool(E_te is not None and E_te.any())

    X_te_n = ((X_te.float().to(device) - m_in.to(device)) / s_in.to(device)).float()
    Y_te_g = Y_te.float().to(device)
    T_te_g = T_te.to(device)
    E_te_g = E_te.to(device) if has_edits else None
    mo_g, so_g = m_out.to(device), s_out.to(device)

    if lam_mains is None:
        lam_mains = (3000.0, 10000.0, 30000.0, 100000.0)
    if lam_embs is None:
        lam_embs = (300.0, 1000.0, 3000.0, 10000.0, 30000.0) if has_emb else (0.0,)

    best = None
    for lm in lam_mains:
        for le in lam_embs:
            W = solve_block_ridge(xtx, xty, d_main, lm, le, device=device)
            pred = (X_te_n @ W) * so_g + mo_g
            ac, ar, arel, _ = eval_masked(pred, Y_te_g)
            tc, tr2, trel, tnorm = eval_masked(pred, Y_te_g, T_te_g)
            ec, er2, erel, enorm = eval_masked(pred, Y_te_g, E_te_g) if has_edits else (0.0, 0.0, 0.0, 1.0)
            score = (0.30 * ac + 0.35 * tc + 0.35 * ec) if has_edits else (0.45 * ac + 0.55 * tc)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "W": W.cpu(),
                    "lam_main": lm,
                    "lam_emb": le,
                    "lambda": lm,
                    "cos": ac,
                    "r2": ar,
                    "rel": arel,
                    "typo_cos": tc,
                    "typo_r2": tr2,
                    "typo_rel": trel,
                    "typo_norm": tnorm,
                    "edit_cos": ec,
                    "edit_r2": er2,
                    "edit_rel": erel,
                    "edit_norm": enorm,
                    "mean_in": m_in,
                    "std_in": s_in,
                    "mean_out": m_out,
                    "std_out": s_out,
                }
    edit_str = (
        f" | EDIT cos={best['edit_cos']:.4f} r2={best['edit_r2']:.4f} norm={best['edit_norm']:.3f}"
        if has_edits
        else ""
    )
    print(
        f"[Two-Block Ridge] best lam_main={best['lam_main']:.0f} lam_emb={best['lam_emb']:.0f} "
        f"({time.perf_counter()-t0:.1f}s) | ALL cos={best['cos']:.4f} r2={best['r2']:.4f} rel={best['rel']:.4f} | "
        f"FOCUS cos={best['typo_cos']:.4f} r2={best['typo_r2']:.4f} norm={best['typo_norm']:.3f}{edit_str}",
        flush=True,
    )
    del xtx, xty
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def fit_ridge(
    X_train,
    Y_train,
    X_test,
    Y_test,
    lambdas=(100.0, 500.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0),
    device="cuda",
):
    """Closed-form standardized single-block Ridge regression on GPU via fast FP32 Cholesky."""
    xtx, xty, mean_in, std_in, mean_out, std_out = compute_gram(X_train, Y_train, device=device)
    m_in, s_in = mean_in.to(device), std_in.to(device)
    m_out, s_out = mean_out.to(device), std_out.to(device)
    X_te_n = ((X_test.float().to(device) - m_in) / s_in).float()
    Y_te = Y_test.float().to(device)

    best = None
    for lam in lambdas:
        W = solve_block_ridge(xtx, xty, xtx.shape[0], lam, lam, device=device)
        pred = (X_te_n @ W) * s_out + m_out
        cos, r2 = evaluate_predictions(pred, Y_te)
        if best is None or cos > best["cos"]:
            best = {
                "W": W.cpu(),
                "lambda": float(lam),
                "cos": cos,
                "r2": r2,
                "mean_in": mean_in,
                "std_in": std_in,
                "mean_out": mean_out,
                "std_out": std_out,
            }
    del xtx, xty
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def collect_sample_images(n_images, search_dir, target_size=384):
    """Load up to `n_images` RGB images from `search_dir` as [1, H, W, 3] float32 tensors in [0, 1]."""
    if not search_dir or not os.path.isdir(search_dir):
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp")
    paths = []
    for root, _, files in os.walk(search_dir):
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                paths.append(os.path.join(root, fn))
    out = []
    for p in paths[:n_images]:
        try:
            pil = Image.open(p).convert("RGB")
            w, h = pil.size
            if max(w, h) != target_size:
                scale = float(target_size) / float(max(w, h))
                nw = max(256, int(round(w * scale / 32.0)) * 32)
                nh = max(256, int(round(h * scale / 32.0)) * 32)
                pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            out.append((os.path.basename(p), torch.from_numpy(arr).unsqueeze(0).contiguous()))
        except Exception:
            continue
    return out
