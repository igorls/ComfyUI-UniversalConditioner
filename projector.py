"""Adapter loader and projection math for Universal Conditioner.

Projection file format, standardized linear/MLP projection math, and token-0 attention-sink
handling follow ComfyUI-ClipProj by nicolab28 (MIT, see NOTICE and LICENSES/ClipProj-MIT.txt),
extended with multi-layer sequence refinement (`seq.`), direct target-embedding skip
(`seq_input="base_plus_emb"`), split-offset embeddings, and per-channel clamps.
"""

import os

import torch

try:
    from . import seq_connector
except ImportError:
    import seq_connector

FOLDER = "universal_conditioner"
FALLBACK_FOLDER = "clip_projections"

META_TYPES = {
    "tap": int,
    "d_in": int,
    "d_out": int,
    "n_train_prompts": int,
    "n_train_tokens": int,
    "lambda": float,
    "cos_test": float,
    "r2_test": float,
}

_CACHE = {}


def register_folder():
    """Register models/universal_conditioner (with fallback read from models/clip_projections)."""
    import folder_paths

    primary = os.path.join(folder_paths.models_dir, FOLDER)
    fallback = os.path.join(folder_paths.models_dir, FALLBACK_FOLDER)
    try:
        os.makedirs(primary, exist_ok=True)
    except OSError:
        pass
    dirs = [primary]
    if os.path.isdir(fallback):
        dirs.append(fallback)
    known = folder_paths.folder_names_and_paths.get(FOLDER)
    if known is None:
        folder_paths.folder_names_and_paths[FOLDER] = (dirs, {".safetensors"})
    else:
        for d in dirs:
            if d not in known[0]:
                known[0].append(d)


def list_adapters():
    """Return sorted .safetensors adapter filenames across registered adapter directories."""
    import folder_paths

    register_folder()
    try:
        names = [
            f for f in folder_paths.get_filename_list(FOLDER)
            if f.lower().endswith(".safetensors")
        ]
        return sorted(set(names), key=str.lower)
    except Exception:
        return []


def resolve_adapter_path(name):
    """Resolve an adapter filename or explicit filesystem path."""
    if os.path.isfile(name):
        return name
    import folder_paths

    register_folder()
    return folder_paths.get_full_path_or_raise(FOLDER, name)


def load_projection(path):
    """Load an adapter's tensors and metadata, cached by path and mtime."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    key = (path, os.path.getmtime(path))
    if key in _CACHE:
        return _CACHE[key]
    data = load_file(path)
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    for k, v in meta.items():
        cast = META_TYPES.get(k)
        try:
            data[k] = cast(v) if cast else v
        except (TypeError, ValueError):
            data[k] = v
    for k in ("mean_in", "std_in", "mean_out", "std_out", "tap"):
        if k not in data:
            raise KeyError(
                "%s is missing '%s': not a valid adapter file or incomplete download"
                % (os.path.basename(path), k)
            )
    for k in ("W", "mean_in", "std_in", "mean_out", "std_out", "clamp_lo", "clamp_hi"):
        if k in data:
            data[k] = data[k].float()
    if "W" not in data and not any(k.startswith(("mlp.", "seq.")) for k in data):
        raise ValueError("%s has neither a linear matrix nor a residual network" % os.path.basename(path))
    _CACHE.clear()
    _CACHE[key] = data
    return data


def build_residual(proj, device):
    """Build an optional GELU MLP residual stored as mlp.N.weight / mlp.N.bias."""
    layers = sorted({int(k.split(".")[1]) for k in proj if k.startswith("mlp.")})
    if not layers:
        return None
    dtype = proj["mlp.%d.weight" % layers[0]].dtype
    modules = []
    for n, i in enumerate(layers):
        w = proj["mlp.%d.weight" % i]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=("mlp.%d.bias" % i) in proj)
        lin.weight.data = w.to(device=device, dtype=dtype)
        if lin.bias is not None:
            lin.bias.data = proj["mlp.%d.bias" % i].to(device=device, dtype=dtype)
        modules.append(lin)
        if n < len(layers) - 1:
            modules.append(torch.nn.GELU())
    net = torch.nn.Sequential(*modules).to(device=device, dtype=dtype)
    net.train(False)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def project(h, proj, cache, valid_mask=None, split_pos=None, split_len=None):
    """Map source hidden states [B, seq, d_in] into target conditioning [B, seq, d_out].

    Optional `valid_mask` ([B, seq] bool) zeroes masked slots (e.g. ref2va vision gaps)
    after standardization so Conv1D and attention never bleed across masked spans.
    Optional `split_pos` / `split_len` ([B, seq] long) supply 1:N unpack indices when the
    sequence connector was trained with `use_split_emb=True`.
    """
    dev = h.device
    if cache.get("device") != dev:
        cache.clear()
        cache["device"] = dev
    if "p" not in cache:
        width = proj["mean_in"].shape[0]
        if h.shape[-1] != width:
            raise ValueError(
                "Adapter expects input width %d, got %d" % (width, h.shape[-1])
            )
        cache["p"] = {
            k: proj[k].to(dev)
            for k in ("W", "mean_in", "std_in", "mean_out", "std_out", "clamp_lo", "clamp_hi")
            if k in proj
        }
        cache["mlp"] = build_residual(proj, dev)
        cache["seq"] = seq_connector.build_from_projection(proj, dev)
    p = cache["p"]
    xn = (h.float() - p["mean_in"]) / p["std_in"]
    yn = xn @ p["W"] if "W" in p else None

    net = cache.get("mlp")
    if net is not None:
        out = net(xn.to(net[0].weight.dtype)).float()
        yn = out if yn is None else yn + out

    snet = cache.get("seq")
    sink = proj.get("sink_out")
    if snet is not None:
        s_dtype = snet.in_proj.weight.dtype
        seq_in_mode = proj.get("seq_input")
        if seq_in_mode == "base_plus_emb":
            if yn is None:
                raise ValueError("seq_input='base_plus_emb' requires a base linear matrix W")
            d_emb = yn.shape[-1]
            s_in = torch.cat([yn, xn[..., -d_emb:]], dim=-1)
        elif seq_in_mode == "base":
            s_in = yn
        else:
            s_in = xn

        vm = valid_mask.to(device=dev, dtype=torch.bool) if valid_mask is not None else None
        if vm is not None:
            s_in = s_in * vm.unsqueeze(-1).to(s_in.dtype)

        sp = split_pos.to(device=dev, dtype=torch.long) if split_pos is not None else None
        sl = split_len.to(device=dev, dtype=torch.long) if split_len is not None else None
        if vm is not None and sp is not None and sl is not None:
            sp = sp * vm.long()
            sl = sl * vm.long()

        if sink is not None and s_in.shape[1] > 1:
            vm_sub = vm[:, 1:] if vm is not None else None
            sp_sub = sp[:, 1:] if sp is not None else None
            sl_sub = sl[:, 1:] if sl is not None else None
            out_sub = snet(
                s_in[:, 1:].to(s_dtype),
                valid_mask=vm_sub,
                split_pos=sp_sub,
                split_len=sl_sub,
            ).float()
            if yn is None:
                yn = torch.zeros(
                    (s_in.shape[0], s_in.shape[1], out_sub.shape[-1]),
                    device=s_in.device,
                    dtype=torch.float32,
                )
                yn[:, 1:] = out_sub
            else:
                yn = yn.clone()
                yn[:, 1:] = yn[:, 1:] + out_sub
        else:
            out = snet(
                s_in.to(s_dtype),
                valid_mask=vm,
                split_pos=sp,
                split_len=sl,
            ).float()
            yn = out if yn is None else yn + out

    cond = yn * p["std_out"] + p["mean_out"]
    if "clamp_lo" in p and "clamp_hi" in p:
        cond = torch.clamp(cond, min=p["clamp_lo"], max=p["clamp_hi"])
    if sink is not None and cond.shape[1] > 0:
        cond[:, 0] = sink.to(device=cond.device, dtype=cond.dtype)
    return cond
