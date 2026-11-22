# Universal Conditioner Adapter for MiniMax H3

Maps multi-layer hidden states from an external 27B source encoder into MiniMax H3's `Qwen3-VL-32B` conditioning space (`d_out = 5120`) for text-to-video, multimodal `<Picture N>` reference video, and multilingual `<d>...</d>` spoken dialogue.

## Current baseline (`v4.2`)

| Field | Value |
| --- | --- |
| Adapter file | `unicond-h3-v4.2-dual-seq.safetensors` (`590.09 MB`, not published yet) |
| Recommended source model | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) (scores pending cross-model check) |
| Also supported source model | [igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer) (fitted source for `v4.2`; exact sha256 recorded on publish) |
| Tapped source layers | `[-2, 15, 43, 51, 59]` |
| Target space | MiniMax H3 (`qwen3vl_32b`, 151k BPE, `d_out = 5120`) |
| Vocabulary alignment | Token spans aligned by UTF-8 bytes (`unpack_n_to_1 = "last"`) |
| Target embedding lookup | Target encoder's `embed_tokens.weight` read from disk (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, not loaded onto GPU) |
| Architecture | Dual modality: Two-block Text Ridge + 2-layer `d_model = 768` BidirectionalSeqConnector + Multi-layer Vision Ridge (`W_vis`) + `sink_out` + per-channel clamps |

### Calibration data and provenance (`v4.2`)

- **3,268 prompts** covering visual descriptions, `<d>...</d>` spoken dialogue, `<|lyrics_start|>` singing, and code-switching across 12 languages (PT-BR, EN, ES, FR, IT, DE, JA, KO, ZH, RU, NL, PL), written by the source model.
- **80 reference images** for the dual vision branch (`W_vis`) and uncompacted `ref2va` vision-gap sequence training.
- Target hidden states were extracted using the MiniMax H3 32B text encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`). The adapter file contains only projection weights and normalization statistics; no base model weights or training prompts are bundled.

### Held-out metrics (`v4.2`)

Evaluated on the 10% held-out split (`327` text sequences + `12` held-out reference images) against the fitted source model:

| Slice | Cosine | R² | Standardized R² | Norm ratio |
| --- | --- | --- | --- | --- |
| All text tokens (`ALL`) | `0.9431` | `0.7722` | `0.6275` | `0.985` |
| Dialogue tokens (`DIAL`) | `0.9282` | `0.7098` | `0.5547` | `0.962` |
| Vision reference tokens (`VIS`) | `0.8386` | `0.1932` | n/a | n/a |

Max held-out norm ratio: `1.26x`.

### Evaluation limits and next revision (`v4.3`)

- **Vision reference path (`W_vis`):** Held-out vision tokens reach cosine `0.8386` at R² `0.1932`. That is a weak linear fit on the path used by `<Picture N>` reference images (such as character faces). Reference-image renders through `v4.2` have not yet been A/B compared against the native 32B encoder (prior `<Picture 1>` visual checks were run on `v4.1` only).
- **Speech checks:** Single-seed listening checks across 10 languages matched native 32B on PT-BR, EN, FR, IT, DE, and bilingual PT/EN, with pronunciation slips remaining on individual words in ES, JA, RU, and PL (often coinciding with 1:N 248k-to-151k subword splits).
- **Revision `v4.3` (in progress):** Adds `seq_input = "base_plus_emb"`, `(split_pos, split_len)` learned embeddings, a 4-layer `d_model = 1024` connector, greedy layer selection, and 5-seed Whisper WER/CER scoring against native 32B. This card will be updated with `v4.3` numbers and multi-seed WER once verification completes.
- Requires `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` on disk in `ComfyUI/models/text_encoders/` for the `embed_tokens` table lookup; loading fails without it.

## License

This adapter is distributed as a Model Derivative under the **MiniMax H3 Community License Agreement** ([LICENSE](LICENSE)). See [NOTICE](NOTICE).

This license is limited to its **Applicable Territory**; read [LICENSE](LICENSE) for the exact scope and regional restrictions.
