# Universal Conditioner Adapter for MiniMax H3

Maps multi-layer hidden states from an external 27B source encoder into MiniMax H3's `Qwen3-VL-32B` conditioning space (`d_out = 5120`) for text-to-video and spoken dialogue.

## Current status (`v4.4`, 6-layer)

| Field | Value |
| --- | --- |
| Adapter file | `unicond-h3-v4.4-6l-dual-seq.safetensors` (`690.13 MB`, sha256 `43cccbd678831444e30b23638677ff310e8471839d309d34c6660662698cbebe`, not published yet) |
| Recommended source model | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) (`qwen3_8_27b_nvfp4.ninfer`, sha256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`, fitted source for `v4.4`) |
| Also supported source model | [igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer) (`qwen3_8_27b_orcarouter_nvfp4.ninfer`, sha256 `50f2e407a2a6ee6b049a41de2724119cb4bbcb189ba14445babdd2ff5d56a384`) |
| Tapped source layers | `[-2, 15, 27, 43, 51, 59]` |
| Target space | MiniMax H3 (`qwen3vl_32b`, 151k BPE, `d_out = 5120`) |
| Vocabulary alignment | Token spans aligned by UTF-8 bytes (`unpack_n_to_1 = "last"`) |
| Target embedding lookup | Target encoder's `embed_tokens.weight` read from disk (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, not loaded onto GPU) |
| Architecture | Dual modality: Two-block Text Ridge (`35840 -> 5120`, FP16) + 2-layer `d_model = 768` BidirectionalSeqConnector (`20.9M`, `seq_input = "base"`, BF16) + sink-filtered Vision Ridge (`W_vis`, `30720 -> 5120`, FP16) + `sink_out` + per-channel clamps |

### Calibration data and provenance

- **5,095 prompts** (`4,585` train, `510` held-out) covering visual descriptions, `<d>...</d>` spoken dialogue, `<|lyrics_start|>` singing, and code-switching across 12 languages (PT-BR, EN, ES, FR, IT, DE, JA, KO, ZH, RU, NL, PL), written by the source model.
- **370 unique images** (`305` train, `65` strictly held-out) for the refitted vision branch (`W_vis`), filtering scattered high-norm visual register sinks (`||y|| >= 2000`, `1.10%` of test patches) before fitting.
- Target hidden states were extracted using the MiniMax H3 32B text encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`). The adapter file contains only projection weights, normalization statistics, clamps, and the sequence connector; no base model weights or training prompts are bundled.

### Held-out hidden-state metrics (`v4.4`)

Text evaluated on `neroued` (fitted stock source) across the `510`-prompt held-out split (`24,570` tokens, `7,837` dialogue) and the `327`-prompt shared `v4.2` unseen split (`16,303` tokens, `4,431` dialogue). Vision evaluated on `65` unseen images (`7,297` clean patches, `81` native register sinks `>= 2,000`) against both source models.

| Modality / Split (Source) | Cosine | R² | Standardized R² | Max norm ratio |
| --- | --- | --- | --- | --- |
| Text `ALL` (`510` prompts, `neroued`) | `0.9493` | `0.7976` | `0.6657` | `1.31x` |
| Text `DIAL` (`510` prompts, `neroued`) | `0.9364` | `0.7410` | `0.5923` | `1.31x` |
| Text `ALL` (`327` shared prompts, `neroued`) | `0.9462` | `0.7843` | `0.6447` | `1.31x` |
| Text `DIAL` (`327` shared prompts, `neroued`) | `0.9328` | `0.7278` | `0.5768` | `1.31x` |
| Vision clean patches (`65` imgs, `neroued`) | `0.8950` | `+0.4278` | `+0.3635` | `0.898` mean norm ratio |
| Vision clean patches (`65` imgs, `orcarouter`) | `0.8931` | `+0.4185` | `+0.3529` | `0.896` mean norm ratio |

Note on cross-model text transfer: `orcarouter` cross-model text scoring was measured on the `v4.2` 5-layer stock control (`-0.0006` cos / `-0.0026` R² on the 327 shared prompts) and was not separately re-extracted for the 6-layer text head.

### 5-seed speech evaluation (Whisper `large-v3` on `neroued`)

Acceptance bar: within `+0.03` mean WER (or CER for Japanese) of the native 32B encoder per language across 5 identical seeds, with no outlier seed above `0.30`. All test prompts had zero overlap with training prompts.

| Language (prompts x seeds, metric) | Native 32B pooled | `v4.4` (6L) pooled | Per-prompt means (`v4.4` vs Native) |
| --- | --- | --- | --- |
| Spanish (`1 x 5 = 5` clips, WER) | `0.000` (`0/75`) | `0.000` (`0/75`) | `es_courtyard`: `0.000` vs `0.000` (direct listening: clean `'a'` in `alegría` on 5/5 seeds) |
| Russian (`2 x 5 = 10` clips, WER)* | `0.083` / `0.231` | `0.083` / `0.231` | `ru_moscow`: `0.083` vs `0.083`; `ru_tram`: `0.231` vs `0.231` (0 real speech errors on 10/10 clips) |
| Polish (`3 x 5 = 15` clips, WER) | `0.038` (`7/185`) | `0.043` (`8/185`) | `pl_market`: `0.000` vs `0.000`; `pl_train`: `0.000` vs `0.000`; `pl_krakow`: `0.133` vs `0.117` |
| Japanese (`5 x 5 = 25` clips, CER) | `0.013` (`11/830`) | `0.055` (`46/830`) | `ja_bakery`: `0.018` vs `0.018`; `ja_library`: `0.000` vs `0.000`; `ja_station`: `0.043` vs `0.027`; `ja_rain_walk`: `0.106` vs `0.006`; `ja_kyoto`: `0.124` vs `0.014` |

*Russian note: Whisper transcribes `Пойдём` as `Пойдем` (`0.083` fixed floor on `ru_moscow`) and `до Невского` / `пять` as `Даневского` / `5` (`0.231` fixed floor on `ru_tram`) identically on every native and adapted clip.

### Known limits

- **Japanese speech is below native:** Pooled CER across 5 prompts (25 clips, 830 characters) is `0.055` against native `0.013` (`+0.042`, missing the `+0.03` bar). This is a large improvement over `v4.2` (`0.11` to `0.17` CER), with native-matching scores on `ja_bakery` (`0.018`) and `ja_library` (`0.000`), but `v4.4` still misreads specific kanji compounds on repeated seeds: `宇治抹茶` and `用意` in `ja_kyoto` (5/5 seeds), `喫茶店` in `ja_rain_walk` (4/5 seeds), and `列車` in `ja_station` (1/5 seeds).
- **Vision projection (`W_vis`):** Vision projection refitted, held-out clean-patch R² `+0.43` (`+0.42` on the uncensored source), up from `-1.16` before sink filtering, with bounded output norms at native sink positions (`202` to `308`). Not shown to change renders, because the reference picture also reaches the model through the video VAE.
- **Requires target encoder file on disk:** `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` must be present in `ComfyUI/models/text_encoders/` for the `embed_tokens` table lookup; loading fails without it.

## License

This adapter is distributed as a Model Derivative under the **MiniMax H3 Community License Agreement** ([LICENSE](LICENSE)). See [NOTICE](NOTICE).

This license is limited to its **Applicable Territory**; read [LICENSE](LICENSE) for the exact scope and regional restrictions.
