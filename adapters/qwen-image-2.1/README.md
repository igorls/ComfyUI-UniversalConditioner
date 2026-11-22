# Universal Conditioner Adapter for Qwen-Image 2.1

Built with Qwen.

Maps multi-layer hidden states from an external 27B source encoder into Qwen-Image 2.1's `Qwen3-VL-8B` conditioning space (`d_out = 4096`) for text-to-image generation, signage/typography, and single- or multi-image editing.

## Specification (`v3.2`)

| Field | Value |
| --- | --- |
| Adapter file | [`unicond-qi21-v3.2-seq.safetensors`](https://huggingface.co/igorls/universal-conditioner/blob/main/qwen-image-2.1/unicond-qi21-v3.2-seq.safetensors) (`268.95 MB`, sha256 `d04509c70357986484f9cb22b13bc61434d50a5275c204b44ed2f5af809aa47c`) |
| Recommended source model | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) (`qwen3_8_27b_nvfp4.ninfer`, sha256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`, fitted source for `v3.2`) |
| Also supported source model | [igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer) (`qwen3_8_27b_orcarouter_nvfp4.ninfer`, sha256 `50f2e407a2a6ee6b049a41de2724119cb4bbcb189ba14445babdd2ff5d56a384`) |
| Tapped source layers | `[-2, 15, 47, 55, 63]` |
| Target space | Qwen-Image 2.1 (`qwen3vl_8b`, 151k BPE, `d_out = 4096`) |
| Vocabulary alignment | Token spans aligned by UTF-8 bytes (`unpack_n_to_1 = "last"`) |
| Target embedding lookup | Target encoder's `embed_tokens.weight` read from disk (`qwen3vl_8b_int8_convrot.safetensors`, not loaded onto GPU) |
| Architecture | Two-block Ridge (`29696 -> 4096`, FP16) + 2-layer `d_model = 768` BidirectionalSeqConnector (`19.3M`, BF16) + per-channel clamps |

## Calibration data and provenance

- **2,495 text-to-image prompts** covering scenes, photography, illustration, typography, signage, quoted strings, and multilingual text, written by the source model.
- **350 multimodal image-conditioned edit sequences** at 1024px (`160` OCR-matched in-place text replacements, `140` attribute-preserving restyles, and `50` two-image edits).
- **Zero overlap** between calibration prompts/images and held-out visual evaluation prompts/images.
- Target hidden states were extracted using the Qwen-Image 2.1 text encoder (`qwen3vl_8b_int8_convrot.safetensors`). The adapter file contains only projection weights, normalization statistics, clamps, and the sequence connector; no base model weights or training prompts are bundled.

## Held-out metrics (`v3.2`)

Evaluated on the 10% held-out split (`283` sequences, `17,123` tokens, `33` edits) against both source models:

| Slice | `neroued` stock (fitted) cos / R² | `orcarouter` uncensored (cross) cos / R² |
| --- | --- | --- |
| All tokens (`ALL`, `n = 17,123`) | `0.9398` / `0.8414` | `0.9366` / `0.8334` |
| Typography / signage (`TYPO`, `n = 4,168`) | `0.9331` / `0.8135` | `0.9313` / `0.8088` |
| Image-conditioned edits (`EDIT`, `n = 2,618`) | `0.9570` / `0.8655` | `0.9506` / `0.8517` |

## Visual A/B checks (`v3.2` on `neroued` vs native `Qwen3-VL-8B`)

Tested across 7 held-out prompts x 2 seeds (`14` seed-locked pairs, `28` images at 1024x1024, 8-step turbo `euler`/`simple`) using the exact same ComfyUI graph (`TextEncodeQwenImage21`): 4 single-image character/storefront edits with text replacement, 2 dense typography/physics T2I prompts, and 1 two-image composition edit.

- **Text and attributes:** All 54 quoted text strings across the 14 `v3.2` renders matched the prompt letter-for-letter, including accented Portuguese/French (`"DISCOS SÃO JOSÉ"`, `"CAFÉ LUMIÈRE"`), symbols (`"TONKOTSU ¥980"`, `"9 BAR • 93°C"`), and in-place storefront sign replacements (`"PADARIA SÃO JOSÉ" -> "DISCOS SÃO JOSÉ"`, `"PÃO QUENTINHO" -> "VINIL & JAZZ"`). Character identity, heterochromia, hair colors, and scene structure matched native `Qwen3-VL-8B` across all 10 edit pairs.
- **Minor visual differences at fixed seeds:** Small stylistic and layout details still differ between `v3.2` and native `Qwen3-VL-8B` on the same seed (for example, two-line vs one-line header wrap and burr grinder drawing details on the espresso poster, subtle framing distance on one two-image edit seed, pocket-watch position on the apothecary macro, and cup lid color on one Paris bookshop seed). On 3 of the 14 pairs (`s601` yen sign, `s602` `"ANCIENNE"`, and `s603` display base), `v3.2` rendered cleaner text than native `Qwen3-VL-8B`.
- **Limits:** Visual verification covered 7 held-out prompts (2 seeds each) rather than a large automated FID or OCR benchmark. Prompts longer than 512 tokens, scripts outside Latin and CJK, and chains of 3 or more reference images have not been systematically evaluated.
- **Target encoder file on disk:** Requires `qwen3vl_8b_int8_convrot.safetensors` in `ComfyUI/models/text_encoders/` for the `embed_tokens` table lookup; loading fails without it.

## License

This adapter is distributed under the **Qwen RESEARCH LICENSE AGREEMENT** ([LICENSE](LICENSE)) for **non-commercial use only**. See [NOTICE](NOTICE). Built with Qwen.
