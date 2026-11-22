# Universal Conditioner Adapter for Qwen-Image 2.1

Built with Qwen.

Maps multi-layer hidden states from an external 27B source encoder into Qwen-Image 2.1's `Qwen3-VL-8B` conditioning space (`d_out = 4096`) for text-to-image generation, signage/typography, and single- or multi-image editing.

## Specification

| Field | Value |
| --- | --- |
| Adapter file | `unicond-qi21-v3.1-seq.safetensors` (`268.95 MB`, not published yet) |
| Recommended source model | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) (scores pending cross-model check) |
| Also supported source model | [igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer) (fitted source for `v3.1`; exact sha256 recorded on publish) |
| Tapped source layers | `[-2, 15, 47, 55, 63]` |
| Target space | Qwen-Image 2.1 (`qwen3vl_8b`, 151k BPE, `d_out = 4096`) |
| Vocabulary alignment | Token spans aligned by UTF-8 bytes (`unpack_n_to_1 = "last"`) |
| Target embedding lookup | Target encoder's `embed_tokens.weight` read from disk (`qwen3vl_8b_int8_convrot.safetensors`, not loaded onto GPU) |
| Architecture | Two-block Ridge (`30720 -> 4096`, FP16) + 2-layer `d_model = 768` BidirectionalSeqConnector (`19.3M`, BF16) + per-channel clamps |

## Calibration data and provenance

- **2,495 text-to-image prompts** covering scenes, photography, illustration, typography, signage, quoted strings, and multilingual text, written by the source model.
- **350 multimodal image-conditioned edit sequences** at 1024px (`160` OCR-matched in-place text replacements, `140` attribute-preserving restyles, and `50` two-image edits).
- **Zero overlap** between calibration prompts/images and held-out visual evaluation prompts/images.
- Target hidden states were extracted using the Qwen-Image 2.1 text encoder (`qwen3vl_8b_int8_convrot.safetensors`). The adapter file contains only projection weights, normalization statistics, clamps, and the sequence connector; no base model weights or training prompts are bundled.

## Held-out metrics

Evaluated on the 10% held-out split (`285` sequences) against the fitted source model:

| Slice | Cosine | R² |
| --- | --- | --- |
| All tokens (`ALL`) | `0.9397` | `0.8411` |
| Typography / signage (`TYPO`) | `0.9329` | `0.8129` |
| Image-conditioned edits (`EDIT`) | `0.9565` | `0.8643` |

## Visual checks and evaluation limits

- 7 held-out prompts were compared with the native `Qwen3-VL-8B` encoder at 1024x1024, one seed each, covering English/Portuguese/Japanese signage, in-place sign replacement (`"PADARIA SÃO JOSÉ" -> "DISCOS SÃO JOSÉ"`, `"PÃO QUENTINHO" -> "VINIL & JAZZ"`), attribute preservation (heterochromia), and two-image composition. All 7 matched the native outputs in composition, attributes, and text spelling.
- **Limits:** Visual checks were limited to 7 held-out prompts (one seed per prompt) rather than a large automated FID or OCR benchmark. Prompts longer than 512 tokens, scripts outside Latin and CJK, and chains of 3 or more reference images have not been systematically evaluated.
- Requires `qwen3vl_8b_int8_convrot.safetensors` on disk in `ComfyUI/models/text_encoders/` for the `embed_tokens` table lookup; loading fails without it.

## License

This adapter is distributed under the **Qwen RESEARCH LICENSE AGREEMENT** ([LICENSE](LICENSE)) for **non-commercial use only**. See [NOTICE](NOTICE). Built with Qwen.
