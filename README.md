# ComfyUI-UniversalConditioner

Use one external LLM/VLM server as the text and vision encoder for multiple ComfyUI diffusion models without loading encoder weights into ComfyUI VRAM.

## Status

Experimental. Two target adapters have been fitted (Qwen-Image 2.1 and MiniMax H3) and evaluated on held-out hidden states plus targeted A/B renders against each model's native encoder.

## Why this exists

Large image and video diffusion models ship with separate 8B to 32B text/vision encoders. Loading a separate encoder per model uses extra GPU memory and slows down switching.

This pack sends the prompt and any reference images to an already running LLM/VLM server over HTTP. The server returns hidden states from multiple layers in one prefill pass. A fitted adapter maps those states into each diffusion model's conditioning space.

## Architecture

Each adapter is a `.safetensors` file with three parts:

1. **Multi-layer Ridge projection** (`W`, `mean_in`, `std_in`, `mean_out`, `std_out`). Maps 5 tapped source layers, plus optional target token embeddings, into the target encoder's standardized space.
2. **Bidirectional sequence connector** (`seq.*`). Applies depthwise Conv1D, bidirectional RoPE self-attention, and SwiGLU FFN as a zero-initialized residual over the Ridge base.
3. **Optional target-specific tensors**. A separate vision projection (`W_vis`, `mean_in_vis`, `std_in_vis`, `mean_out_vis`, `std_out_vis`) for multimodal references, a token-0 attention sink (`sink_out`), and per-channel clamps (`clamp_lo`, `clamp_hi`).

When the source and target models use different vocabularies (for example 248k vs 151k BPE), token spans are aligned by UTF-8 bytes (`unpack_to_target_vocab`). If the adapter uses target token embeddings, the target encoder's embedding table is read from disk via memory mapping rather than loaded onto the GPU.

## Server contract

The node sends `POST <server_url>/v1/hidden_states` with JSON:

```json
{
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "..."}]}
  ],
  "layers": [-2, 15, 43, 51, 59],
  "enable_thinking": false
}
```

The server responds with a binary frame:

- Bytes `0..8`: little-endian `uint64` JSON header byte length `N`.
- Bytes `8..8+N`: UTF-8 JSON header containing `seq_len`, `num_layers`, `hidden_size`, `dtype` (`"bf16"`), `layers`, `token_types`, `token_texts`, and `token_byte_lens`.
- Remaining bytes: contiguous little-endian `bfloat16` payload of shape `[seq_len, num_layers, hidden_size]`.

The default server URL in the node is `http://127.0.0.1:8000`. Set `url` in the node to match your server. If the server requires bearer authentication, set the `UNICOND_API_KEY` environment variable in the ComfyUI process instead of storing keys inside workflow files.

Server availability: the `POST /v1/hidden_states` endpoint is implemented in [NInfer](https://github.com/igorls/ninfer), but it is not yet in a public NInfer release. Any server implementing the binary frame contract above works with this node.

Source models:
- Recommended: [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
- Also supported: [igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer)

Each adapter card records the exact source model and file hash it was fitted on, along with per-source held-out scores.

## Installation

1. Clone this repository into ComfyUI's `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/igorls/ComfyUI-UniversalConditioner.git
```

2. Place adapter `.safetensors` files in `ComfyUI/models/universal_conditioner/` (files in `ComfyUI/models/clip_projections/` are also discovered as a fallback). Adapter weights are not published yet; once released they will be hosted separately on Hugging Face.
3. **On-disk target encoder file requirement:** ComfyUI does not load target encoder weights into GPU VRAM, but adapters that use target token embeddings read rows from the target encoder's `.safetensors` file on disk in `ComfyUI/models/text_encoders/`. Loading fails if that file is missing:
   - Qwen-Image 2.1 adapter: `qwen3vl_8b_int8_convrot.safetensors`
   - MiniMax H3 adapter: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
4. Restart ComfyUI.

## Node usage

Add **Universal Conditioner** (`UniCondEncoder`) under `conditioning/universal_conditioner`.

Inputs:
- `url`: Base URL of your hidden-states server (default `http://127.0.0.1:8000`).
- `adapter`: Adapter `.safetensors` file from `models/universal_conditioner/` (or `none` for raw hidden states).
- `profile`: `auto` (read from adapter metadata), `minimax_h3`, or `qwen_image_21`.
- `model`: Optional model name forwarded in the request payload.
- `layer`: Optional single-layer override (`-1` uses the adapter's multi-layer list).

Output:
- `CLIP`: Standard ComfyUI `CLIP` handle compatible with `CLIPTextEncode`, `TextEncodeQwenImageEdit`, and MiniMax H3 conditioning nodes.

## Prompt enhancer

**Universal Conditioner Prompt Enhancer** (`UniCondPromptEnhancer`) rewrites a short request into a finished prompt with the same language model, before it is encoded. It calls `POST <server_url>/v1/chat/completions` (the OpenAI chat format) on the server you already run for the encoder, so no second model is loaded.

- With pictures connected it writes an edit prompt: what changes, what stays, and any text in quotes exactly as you wrote it. It names the pictures `<image1>`, `<image2>`, the same labels the Qwen-Image 2.1 text encode node uses, so connect them in the same order on both nodes.
- Without pictures it writes an image prompt.
- The prompt comes back in the language of your request.

Inputs: `url`, `prompt`, `enhance` (off passes your prompt through unchanged), `seed` (change it for another wording), `temperature`, `max_tokens`, `model`, `instructions` (your own instructions for the rewrite, in place of the built-in ones), and `images`. Output: `prompt`, a string to connect to a text encode node.

## Workflows

- [workflows/qwen_image_21_edit_prompt_enhancer.json](workflows/qwen_image_21_edit_prompt_enhancer.json): a Qwen-Image 2.1 edit to drop onto the ComfyUI page ([API format](workflows/qwen_image_21_edit_prompt_enhancer_api.json)). One picture and a short request go to the prompt enhancer; the rewritten prompt and the picture go to the Qwen-Image 2.1 text encode node through the Universal Conditioner; 8 steps at cfg 1.

## Adapter cards

Adapter cards and evaluation notes live under `adapters/`. Adapter `.safetensors` weights are not stored in this git repository and are not published yet:

- [Qwen-Image 2.1](adapters/qwen-image-2.1/README.md)
- [MiniMax H3](adapters/minimax-h3/README.md)

## Calibration tools

Scripts in `tools/` fit new adapters against a local ComfyUI checkout and a running `/v1/hidden_states` server:

- `tools/calibrate_qi21.py`: Two-block Ridge and 5-fold OOF sequence connector for Qwen-Image 2.1 (text-to-image and image-conditioned edits).
- `tools/calibrate_h3.py`: Dual text/vision calibration, stratified diagnostics (`--diag-adapter`), greedy forward layer selection (`--layer-sweep-pool`), and OOF sequence connector for MiniMax H3.
- `tools/eval_speech_wer.py`: Multi-seed speech intelligibility scoring using Whisper WER/CER.

## Running tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## License and credits

The adapters carry their target models' terms and are more restricted than the code.

| Component | License | Notes |
| --- | --- | --- |
| Repository code | MIT ([LICENSE](LICENSE)) | ComfyUI node, projection math, calibration scripts, and tests |
| Qwen-Image 2.1 adapter | Qwen Research License ([LICENSE](adapters/qwen-image-2.1/LICENSE)) | Non-commercial use only; see [NOTICE](adapters/qwen-image-2.1/NOTICE) |
| MiniMax H3 adapter | MiniMax H3 Community License ([LICENSE](adapters/minimax-h3/LICENSE)) | Model Derivative; limited to its Applicable Territory; see [NOTICE](adapters/minimax-h3/NOTICE) |

Projection file format, standardized linear/MLP projection math, and token-0 attention-sink handling follow [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) by nicolab28 (MIT). See [NOTICE](NOTICE) and [LICENSES/ClipProj-MIT.txt](LICENSES/ClipProj-MIT.txt).
