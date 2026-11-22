# Universal Conditioner Adapter for MiniMax H3

Maps multi-layer hidden states from an external 27B source encoder into MiniMax H3's `Qwen3-VL-32B` conditioning space (`d_out = 5120`) for text-to-video and spoken dialogue.

## Current status (`v4.2` / `v4.2-stock-control`)

| Field | Value |
| --- | --- |
| Stock-fitted control | `unicond-h3-v4.2-stock-control-dual-seq.safetensors` (`590.09 MB`, not published yet) |
| Uncensored-fitted baseline | `unicond-h3-v4.2-dual-seq.safetensors` (`590.09 MB`, not published yet) |
| Recommended source model | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) (`qwen3_8_27b_nvfp4.ninfer`, sha256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`) |
| Also supported source model | [igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer) (`qwen3_8_27b_orcarouter_nvfp4.ninfer`, sha256 `50f2e407a2a6ee6b049a41de2724119cb4bbcb189ba14445babdd2ff5d56a384`) |
| Tapped source layers | `[-2, 15, 43, 51, 59]` |
| Target space | MiniMax H3 (`qwen3vl_32b`, 151k BPE, `d_out = 5120`) |
| Vocabulary alignment | Token spans aligned by UTF-8 bytes (`unpack_n_to_1 = "last"`) |
| Target embedding lookup | Target encoder's `embed_tokens.weight` read from disk (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, not loaded onto GPU) |
| Architecture | Dual modality: Two-block Text Ridge + 2-layer `d_model = 768` BidirectionalSeqConnector + Multi-layer Vision Ridge (`W_vis`) + `sink_out` + per-channel clamps |

### Calibration data and provenance

- **3,268 prompts** covering visual descriptions, `<d>...</d>` spoken dialogue, `<|lyrics_start|>` singing, and code-switching across 12 languages (PT-BR, EN, ES, FR, IT, DE, JA, KO, ZH, RU, NL, PL), written by the source model.
- **80 reference images** for the dual vision branch (`W_vis`) and uncompacted `ref2va` vision-gap sequence training.
- Target hidden states were extracted using the MiniMax H3 32B text encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`). The adapter file contains only projection weights and normalization statistics; no base model weights or training prompts are bundled.

### Held-out hidden-state metrics (327 shared unseen prompts)

| Adapter / Source | `ALL` cos / R² / stdR² | `DIAL` cos / R² / stdR² | `VIS` raw R² / clean-patch R² | Max norm ratio |
| --- | --- | --- | --- | --- |
| `v4.2-stock-control` on `neroued` (stock) | `0.9431` / `0.7723` / `0.6277` | `0.9282` / `0.7094` / `0.5544` | `0.0044` / `-1.0600` | `1.28x` |
| `v4.2-stock-control` on `orcarouter` | `0.9425` / `0.7697` / `0.6242` | `0.9276` / `0.7075` / `0.5519` | `0.0054` / `-1.0399` | `1.29x` |
| `v4.2` on `neroued` (stock) | `0.9427` / `0.7703` / `0.6250` | `0.9279` / `0.7083` / `0.5530` | `0.0034` / `-1.1917` | `1.25x` |
| `v4.2` on `orcarouter` (fitted) | `0.9431` / `0.7721` / `0.6274` | `0.9282` / `0.7093` / `0.5543` | `0.0065` / `-1.1760` | `1.26x` |

### 5-seed speech evaluation (Whisper `large-v3` on `neroued`)

Acceptance bar: within `+0.03` mean WER (or CER for Japanese) of the native 32B encoder per language across 5 identical seeds, with no outlier seed above `0.30`.

| Language (metric, 5 seeds) | Native 32B mean (per-seed) | `v4.2` mean (per-seed) | Status |
| --- | --- | --- | --- |
| Spanish `es_courtyard` (WER) | `0.0000` (`0, 0, 0, 0, 0`) | `0.0000` (`0, 0, 0, 0, 0`) | Passes WER (subtle `'a'` elision still audible on some seeds) |
| Japanese `ja_kyoto` (CER) | `0.0138` (`0, 0, 0, 0.069, 0`) | `0.1448` (`0.138, 0.241, 0.138, 0.069, 0.138`) | **Fails** (every seed misses native by `> +0.05` CER) |
| Russian `ru_moscow` (WER)* | `0.0833` (`0.083 x 5`) | `0.0833` (`0.083 x 5`) | Matches Whisper WER (minor stutter audible on seed 109) |
| Polish `pl_krakow` (WER) | `0.1167` (`0.083, 0.083, 0.167, 0.083, 0.167`) | `0.1333` (`0.083, 0.167, 0.167, 0.083, 0.167`) | Within bar (`+0.0166`) |

*Russian note: Whisper transcribes `Пойдём` as `Пойдем`, adding a fixed `0.0833` floor to every clip including all 5 native seeds.

### Known limits

- **Japanese speech does not match native yet:** No H3 adapter passes the speech acceptance bar in Japanese (`v4.2` averages `0.1448` CER vs native `0.0138` across all 5 seeds; a `v4.3` experiment with larger connector capacity and dropped `L-2` regressed further to `0.2138` CER despite higher held-out `stdR²` and was rejected). 5-seed scoring for `v4.2-stock-control` and a second Japanese/Russian test sentence are pending.
- **Picture conditioning (`W_vis`) does not work yet:** Vision Ridge R² is near zero overall (`0.0034` to `0.0178`) and **negative (`-1.06` to `-1.19`) on clean image patches**. Roughly `0.83%` of `Qwen3-VL-32B` vision tokens are scattered high-norm register sinks (`2,094` to `23,344` vs max clean patch norm `548.7`); leaving them unfiltered collapses `W_vis` toward a constant mean. Fixing this requires sink filtering, a larger diverse image set, and a seed-locked `<Picture N>` visual A/B against native 32B (planned as a separate `v4.4` revision).
- **Requires target encoder file on disk:** `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` must be present in `ComfyUI/models/text_encoders/` for the `embed_tokens` table lookup; loading fails without it.

## License

This adapter is distributed as a Model Derivative under the **MiniMax H3 Community License Agreement** ([LICENSE](LICENSE)). See [NOTICE](NOTICE).

This license is limited to its **Applicable Territory**; read [LICENSE](LICENSE) for the exact scope and regional restrictions.
