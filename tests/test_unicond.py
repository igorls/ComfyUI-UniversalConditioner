"""Unit tests for Universal Conditioner (encoder, projector, seq_connector, ridge_math, eval_speech_wer)."""

import http.server
import json
import os
import struct
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from safetensors.torch import save_file

    import encoder
    import projector
    import seq_connector
    from tools import eval_speech_wer
    from tools import ridge_math
except ImportError:
    torch = None


def _pack_response(meta, tensor_bf16):
    header = json.dumps(meta).encode("utf-8")
    raw_bytes = tensor_bf16.contiguous().view(torch.uint16).cpu().numpy().tobytes()
    return struct.pack("<Q", len(header)) + header + raw_bytes


@unittest.skipIf(torch is None, "needs torch")
class UniversalConditionerTest(unittest.TestCase):
    def test_decode_hidden_response_and_token_tags(self):
        torch.manual_seed(42)
        h = torch.randn(4, 16, dtype=torch.bfloat16)
        meta = {
            "model": "qwen3.8-27b",
            "seq_len": 4,
            "hidden_size": 16,
            "dtype": "bf16",
            "layer": 48,
            "token_types": [0, 1, 1, 0],
        }
        parsed_meta, tensor = encoder.decode_hidden_response(_pack_response(meta, h))
        self.assertEqual(parsed_meta["layer"], 48)
        self.assertEqual(tuple(tensor.shape), (1, 4, 16))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.allclose(tensor[0], h.float(), atol=1e-4))
        tags = encoder.token_tags_from_types(4, parsed_meta["token_types"])
        self.assertEqual(tags.tolist(), [0, 0, 0, 0])
        toks = ["a", "<|vision_start|>", "<|image_pad|>", "<|image_pad|>", "<|vision_end|>", "b"]
        tags2 = encoder.token_tags_from_types(6, [0, 0, 1, 1, 0, 0], toks)
        self.assertEqual(tags2.tolist(), [1, 0, 0, 0, 0, 1])

    def test_dual_modality_projection_routes_vision_and_text_separately(self):
        d = 8
        proj = {
            "tap": 51,
            "W": torch.eye(d),
            "mean_in": torch.zeros(d),
            "std_in": torch.ones(d),
            "mean_out": torch.zeros(d),
            "std_out": torch.ones(d),
            "sink_out": torch.full((d,), 9.0),
            "W_vis": torch.eye(d) * 2.0,
            "mean_in_vis": torch.zeros(d),
            "std_in_vis": torch.ones(d),
            "mean_out_vis": torch.zeros(d),
            "std_out_vis": torch.ones(d),
        }
        h = torch.ones(1, 4, d)
        tags = torch.tensor([1, 0, 0, 1], dtype=torch.long)
        out = encoder.project_with_modality(h, proj, tags, {})
        self.assertTrue(torch.allclose(out[0, 0], torch.full((d,), 9.0)))
        self.assertTrue(torch.allclose(out[0, 1], torch.full((d,), 2.0)))
        self.assertTrue(torch.allclose(out[0, 2], torch.full((d,), 2.0)))
        self.assertTrue(torch.allclose(out[0, 3], torch.full((d,), 1.0)))

    def test_find_user_span_strips_qwen_chat_wrapper(self):
        toks = ["<|im_start|>", "user", "\n", "red", " car", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]
        self.assertEqual(encoder.find_user_span(toks, len(toks)), (3, 5))
        merged = ["<|im_start|>", "user\n", "red", " car", "<|im_end|>"]
        self.assertEqual(encoder.find_user_span(merged, len(merged)), (2, 4))
        raw = ["red", " car"]
        self.assertEqual(encoder.find_user_span(raw, len(raw)), (0, 2))

    def test_find_qi21_span_drops_system_and_trailing_think(self):
        toks = [
            "<|im_start|>", "system", "\n", "Comprehend", "<|im_end|>", "\n",
            "<|im_start|>", "user", "\n", "a", " cat", "<|im_end|>", "\n",
            "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n",
        ]
        self.assertEqual(encoder.find_qi21_span(toks, len(toks)), (6, 16))
        self.assertEqual(
            toks[6:16],
            ["<|im_start|>", "user", "\n", "a", " cat", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"],
        )

    def test_build_parts_text_and_reference_images(self):
        img = torch.zeros(1, 16, 16, 3)
        parts = encoder.build_parts(
            "a cat on a rooftop",
            minimax_ref_items=[{"type": "image", "data": img}, {"type": "audio"}],
        )
        self.assertEqual([p["type"] for p in parts], ["text", "image_url", "text"])
        self.assertEqual(parts[0]["text"], "<Picture 1>: ")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(parts[2]["text"], "<Audio 1>: a cat on a rooftop")

    def test_remote_clip_roundtrip_and_optional_projection(self):
        seen = {}
        expected = torch.arange(3 * 8, dtype=torch.float32).view(3, 8).to(torch.bfloat16)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                seen["path"] = self.path
                seen["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
                meta = {
                    "model": "qwen3.8-27b",
                    "seq_len": 3,
                    "hidden_size": 8,
                    "dtype": "bf16",
                    "layer": seen["body"].get("layer"),
                    "token_types": [0, 0, 0],
                    "prefill_ms": 1.5,
                }
                body = _pack_response(meta, expected)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = "http://127.0.0.1:%d" % port
            clip = encoder.UniCondRemoteCLIP(url=url, model="qwen3.8-27b", layer=48)
            tokens = clip.tokenize("hello world")
            scheduled = clip.encode_from_tokens_scheduled(tokens)
            cond, extra = scheduled[0]
            self.assertEqual(seen["path"], "/v1/hidden_states")
            self.assertEqual(seen["body"]["layer"], 48)
            self.assertEqual(tuple(cond.shape), (1, 3, 8))
            self.assertTrue(torch.allclose(cond[0].cpu(), expected.float(), atol=1e-4))
            self.assertEqual(extra["minimax_token_tags"].tolist(), [1, 1, 1])

            with tempfile.TemporaryDirectory() as folder:
                proj_path = os.path.join(folder, "id.safetensors")
                save_file(
                    {
                        "W": torch.eye(8),
                        "mean_in": torch.zeros(8),
                        "std_in": torch.ones(8),
                        "mean_out": torch.zeros(8),
                        "std_out": torch.ones(8),
                    },
                    proj_path,
                    metadata={"tap": "40"},
                )
                pclip = encoder.UniCondRemoteCLIP(url=url, layer=-1, adapter_path=proj_path)
                pcond = pclip.encode("projected")
                self.assertEqual(seen["body"]["layer"], 40)
                self.assertTrue(torch.allclose(pcond[0].cpu(), expected.float(), atol=1e-4))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_decode_multilayer_hidden_response(self):
        torch.manual_seed(7)
        h = torch.randn(5, 3, 8, dtype=torch.bfloat16)
        meta = {
            "model": "qwen3.8-27b",
            "seq_len": 5,
            "num_layers": 3,
            "hidden_size": 8,
            "dtype": "bf16",
            "layers": [-2, 15, 63],
            "token_types": [0, 0, 0, 0, 0],
        }
        parsed, tensor = encoder.decode_hidden_response(_pack_response(meta, h))
        self.assertEqual(parsed["layers"], [-2, 15, 63])
        self.assertEqual(tuple(tensor.shape), (1, 5, 24))
        self.assertTrue(torch.allclose(tensor[0], h.view(5, 24).float(), atol=1e-4))

    def test_seq_connector_padding_invariance_and_safetensors_roundtrip(self):
        torch.manual_seed(11)
        d_in, d_out = 24, 16
        net = seq_connector.BidirectionalSeqConnector(
            d_in=d_in, d_out=d_out, d_model=32, n_layers=2, n_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0
        )
        net.eval()
        x = torch.randn(1, 5, d_in)
        self.assertTrue(torch.allclose(net(x), torch.zeros(1, 5, d_out)))
        torch.nn.init.normal_(net.out_proj.weight, std=0.1)
        torch.nn.init.normal_(net.out_proj.bias, std=0.1)
        y_single = net(x)
        x_pad = torch.zeros(2, 9, d_in)
        x_pad[0, :5] = x[0]
        x_pad[0, 5:] = 999.0
        x_pad[1, :3] = torch.randn(3, d_in)
        mask = torch.zeros(2, 9, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :3] = True
        y_pad = net(x_pad, valid_mask=mask)
        self.assertTrue(torch.allclose(y_pad[0, :5], y_single[0], atol=1e-5))
        self.assertTrue(torch.allclose(y_pad[0, 5:], torch.zeros(4, d_out), atol=1e-6))

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "seq_test.safetensors")
            tensors = {
                "W": torch.zeros(d_in, d_out, dtype=torch.float16),
                "mean_in": torch.zeros(d_in, dtype=torch.float16),
                "std_in": torch.ones(d_in, dtype=torch.float16),
                "mean_out": torch.zeros(d_out, dtype=torch.float16),
                "std_out": torch.ones(d_out, dtype=torch.float16),
            }
            tensors.update(seq_connector.state_dict_prefixed(net, dtype=torch.float32))
            meta = {
                "tap": "63",
                "layers": json.dumps([-2, 15, 63]),
                "seq_config": json.dumps(net.config),
                "target_profile": "qwen_image_21",
            }
            save_file(tensors, path, metadata=meta)
            proj = projector.load_projection(path)
            out = projector.project(x, proj, {})
            self.assertTrue(torch.allclose(out, y_single, atol=1e-4))

    def test_unpack_by_spans_with_info_returns_exact_split_metadata(self):
        # N:1 collapse ([2, 1] -> [3]), 1:1 exact ([3] -> [3]), 1:3 split ([9] -> [4, 2, 3])
        src_h = torch.tensor(
            [[1.0, 0.0], [3.0, 0.0], [10.0, 20.0], [50.0, 60.0]], dtype=torch.float32
        )
        out, sp, sl, non1 = encoder.unpack_by_spans_with_info(src_h, [2, 1, 3, 9], [3, 3, 4, 2, 3], n_to_1="last")
        self.assertEqual(tuple(out.shape), (5, 2))
        self.assertTrue(torch.allclose(out[0], torch.tensor([3.0, 0.0])))
        self.assertTrue(torch.allclose(out[1], torch.tensor([10.0, 20.0])))
        self.assertTrue(torch.allclose(out[2], torch.tensor([50.0, 60.0])))
        self.assertTrue(torch.allclose(out[3], torch.tensor([50.0, 60.0])))
        self.assertTrue(torch.allclose(out[4], torch.tensor([50.0, 60.0])))
        self.assertEqual(sp.tolist(), [0, 0, 1, 2, 3])
        self.assertEqual(sl.tolist(), [0, 0, 3, 3, 3])
        self.assertEqual(non1.tolist(), [True, False, True, True, True])

    def test_seq_connector_split_embeddings_differentiate_subtokens_and_preserve_1to1(self):
        torch.manual_seed(31)
        d = 16
        net = seq_connector.BidirectionalSeqConnector(
            d_in=d, d_out=d, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0,
            use_split_emb=True,
        )
        self.assertEqual(net.sub_pos_emb.padding_idx, 0)
        self.assertEqual(net.sub_len_emb.padding_idx, 0)
        # Zero-init starts identically at zero
        x = torch.randn(1, 4, d)
        sp = torch.tensor([[0, 1, 2, 3]])
        sl = torch.tensor([[0, 3, 3, 3]])
        net.eval()
        self.assertTrue(torch.allclose(net(x, split_pos=sp, split_len=sl), torch.zeros(1, 4, d)))

        # Activate split embeddings + out_proj: duplicated rows with distinct split_pos produce distinct outputs
        torch.nn.init.normal_(net.out_proj.weight, std=0.2)
        with torch.no_grad():
            net.sub_pos_emb.weight[1:].normal_(std=0.5)
            net.sub_len_emb.weight[1:].normal_(std=0.5)
        dup_x = torch.ones(1, 4, d)
        y_split = net(dup_x, split_pos=sp, split_len=sl)
        self.assertFalse(torch.allclose(y_split[0, 1], y_split[0, 2], atol=1e-4))
        self.assertFalse(torch.allclose(y_split[0, 2], y_split[0, 3], atol=1e-4))
        # When split_pos=0, split_len=0 everywhere, output equals calling without split args
        y_zero_a = net(dup_x, split_pos=torch.zeros_like(sp), split_len=torch.zeros_like(sl))
        y_zero_b = net(dup_x)
        self.assertTrue(torch.allclose(y_zero_a, y_zero_b, atol=1e-5))

    def test_base_plus_emb_projection_and_post_standardization_masking(self):
        torch.manual_seed(43)
        d_out = 8
        d_in = 24  # 16 main + 8 target_emb
        net = seq_connector.BidirectionalSeqConnector(
            d_in=2 * d_out, d_out=d_out, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0,
            use_split_emb=True,
        )
        torch.nn.init.normal_(net.out_proj.weight, std=0.1)
        with torch.no_grad():
            net.sub_pos_emb.weight[1:].normal_(std=0.3)
            net.sub_len_emb.weight[1:].normal_(std=0.3)
        net.eval()

        proj = {
            "tap": 51,
            "W": torch.zeros(d_in, d_out),
            "mean_in": torch.full((d_in,), 3.0),  # non-zero mean tests post-standardization zeroing on masked gaps!
            "std_in": torch.full((d_in,), 2.0),
            "mean_out": torch.zeros(d_out),
            "std_out": torch.ones(d_out),
            "sink_out": torch.full((d_out,), 11.0),
            "W_vis": torch.eye(16, d_out),
            "mean_in_vis": torch.zeros(16),
            "std_in_vis": torch.ones(16),
            "mean_out_vis": torch.zeros(d_out),
            "std_out_vis": torch.ones(d_out),
            "seq_input": "base_plus_emb",
            "seq_config": json.dumps(net.config),
        }
        proj.update(seq_connector.state_dict_prefixed(net, dtype=torch.float32))

        tags = torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.long)
        sp = torch.tensor([[0, 0, 0, 0, 1, 2]])
        sl = torch.tensor([[0, 0, 0, 0, 2, 2]])
        h1 = torch.randn(1, 6, d_in)
        h2 = h1.clone()
        h2[0, 2:4] = 9999.0  # mutating vision gap slots must NOT bleed into text tokens via non-zero mean_in!
        y1 = encoder.project_with_modality(h1, proj, tags, {}, split_pos=sp, split_len=sl)
        y2 = encoder.project_with_modality(h2, proj, tags, {}, split_pos=sp, split_len=sl)
        self.assertTrue(torch.allclose(y1[0, 0], torch.full((d_out,), 11.0)))
        self.assertTrue(torch.allclose(y1[0, tags == 1], y2[0, tags == 1], atol=1e-5))

        # Mutating target_emb tail (last d_out channels) changes base_plus_emb output even though W=0
        h3 = h1.clone()
        h3[0, 4, -d_out:] += 5.0
        y3 = encoder.project_with_modality(h3, proj, tags, {}, split_pos=sp, split_len=sl)
        self.assertFalse(torch.allclose(y1[0, 4], y3[0, 4], atol=1e-4))

    def test_seq_connector_with_sink_excludes_token_zero(self):
        torch.manual_seed(19)
        d = 16
        net = seq_connector.BidirectionalSeqConnector(
            d_in=d, d_out=d, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0
        )
        torch.nn.init.normal_(net.out_proj.weight, std=0.1)
        torch.nn.init.normal_(net.out_proj.bias, std=0.1)
        net.eval()
        proj = {
            "tap": 51,
            "W": torch.eye(d),
            "mean_in": torch.zeros(d),
            "std_in": torch.ones(d),
            "mean_out": torch.zeros(d),
            "std_out": torch.ones(d),
            "sink_out": torch.full((d,), 77.0),
            "seq_input": "base",
            "seq_config": json.dumps(net.config),
        }
        proj.update(seq_connector.state_dict_prefixed(net, dtype=torch.float32))
        x1 = torch.randn(1, 6, d)
        x2 = x1.clone()
        x2[0, 0] = 9999.0
        o1 = projector.project(x1, proj, {})
        o2 = projector.project(x2, proj, {})
        self.assertEqual(tuple(o1.shape), (1, 6, d))
        self.assertTrue(torch.allclose(o1[0, 0], torch.full((d,), 77.0)))
        self.assertTrue(torch.allclose(o1[0, 1:], o2[0, 1:], atol=1e-5))

    def test_clamp_bounds_applied_before_sink_override(self):
        d = 4
        proj = {
            "tap": 51,
            "W": torch.eye(d),
            "mean_in": torch.zeros(d),
            "std_in": torch.ones(d),
            "mean_out": torch.zeros(d),
            "std_out": torch.ones(d),
            "clamp_lo": torch.full((d,), -2.0),
            "clamp_hi": torch.full((d,), 2.0),
            "sink_out": torch.full((d,), 50.0),
        }
        h = torch.tensor([[[10.0, 10.0, 10.0, 10.0], [9.0, -9.0, 0.5, 1.5]]])
        out = projector.project(h, proj, {})
        self.assertTrue(torch.allclose(out[0, 0], torch.full((d,), 50.0)))
        self.assertTrue(torch.allclose(out[0, 1], torch.tensor([2.0, -2.0, 0.5, 1.5])))

    def test_qi21_build_parts_multi_image_spacing_and_empty_prompt(self):
        img = torch.zeros(1, 16, 16, 3)
        p1 = encoder.build_parts("replace X with Y", images=[img], picture_label="<image%d>")
        self.assertEqual([p["type"] for p in p1], ["text", "image_url", "text"])
        self.assertEqual(p1[0]["text"], "<image1>")
        self.assertEqual(p1[2]["text"], "replace X with Y")

        p2 = encoder.build_parts("combine them", images=[img, img], picture_label="<image%d>")
        self.assertEqual([p["type"] for p in p2], ["text", "image_url", "text", "image_url", "text"])
        self.assertEqual(p2[0]["text"], "<image1>")
        self.assertEqual(p2[2]["text"], " <image2>")
        self.assertEqual(p2[4]["text"], "combine them")

        p_empty = encoder.build_parts("", images=[img], picture_label="<image%d>")
        self.assertEqual([p["type"] for p in p_empty], ["text", "image_url"])
        self.assertEqual(p_empty[0]["text"], "<image1>")

    def test_ridge_math_cholesky_matches_fp64_reference(self):
        torch.manual_seed(99)
        X = torch.randn(128, 32)
        W_true = torch.randn(32, 12)
        Y = X @ W_true + 0.05 * torch.randn(128, 12)
        xtx, xty, _, _, _, _ = ridge_math.compute_gram(X, Y, device="cpu")
        W_fast = ridge_math.solve_block_ridge(xtx, xty, d_main=24, lam_main=10.0, lam_emb=2.0, device="cpu")
        reg = torch.full((32,), 10.0, dtype=torch.float64)
        reg[24:] = 2.0
        A = xtx.clone()
        A.diagonal().add_(reg)
        W_ref = torch.linalg.solve(A, xty).float()
        rel_err = (W_fast - W_ref).norm() / W_ref.norm().clamp(min=1e-6)
        self.assertLess(rel_err.item(), 1e-5)

    def test_eval_speech_wer_and_cer_scoring(self):
        metric, rate, edits, ref_len, _, _ = eval_speech_wer.compute_error_rate(
            "¡Qué alegría volver a verte!", "Qué lagría volver a verte", language="es"
        )
        self.assertEqual(metric, "wer")
        self.assertEqual(edits, 1)

        import tempfile
        import wave
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            wp = os.path.join(td, "tone_24k.wav")
            sr = 24000
            samples = (0.25 * np.sin(2.0 * np.pi * 440.0 * np.arange(sr // 2) / sr) * 32767.0).astype(np.int16)
            with wave.open(wp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())
            audio_16k = eval_speech_wer._load_wav_16k(wp)
            self.assertEqual(audio_16k.dtype, np.float32)
            self.assertEqual(audio_16k.shape[0], 8000)
            self.assertLessEqual(float(np.abs(audio_16k).max()), 0.3)
        self.assertEqual(ref_len, 5)
        self.assertAlmostEqual(rate, 0.2, places=4)

        metric_ja, rate_ja, edits_ja, ref_len_ja, _, _ = eval_speech_wer.compute_error_rate(
            "京都の朝、おいしい緑茶を用意しました。", "京都の朝おいしい緑茶を用意しました", language="ja"
        )
        self.assertEqual(metric_ja, "cer")
        self.assertEqual(edits_ja, 0)
        self.assertEqual(rate_ja, 0.0)
        self.assertEqual(ref_len_ja, 17)

    def test_api_key_reads_from_env_var_not_node_input(self):
        seen_auth = {}
        expected = torch.ones(2, 8, dtype=torch.bfloat16)

        class AuthHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                seen_auth["auth"] = self.headers.get("Authorization")
                meta = {"model": "m", "seq_len": 2, "hidden_size": 8, "dtype": "bf16", "token_types": [0, 0]}
                body = _pack_response(meta, expected)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), AuthHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        prev = os.environ.get(encoder.API_KEY_ENV)
        try:
            os.environ[encoder.API_KEY_ENV] = "test-secret-token"
            clip = encoder.load(url="http://127.0.0.1:%d" % port)
            clip.encode("hi")
            self.assertEqual(seen_auth["auth"], "Bearer test-secret-token")
        finally:
            if prev is None:
                os.environ.pop(encoder.API_KEY_ENV, None)
            else:
                os.environ[encoder.API_KEY_ENV] = prev
            srv.shutdown()
            srv.server_close()

        # Verify nodes.py source does not expose an api_key input on the ComfyUI node
        with open(os.path.join(ROOT, "nodes.py"), "r", encoding="utf-8") as f:
            nodes_src = f.read()
        self.assertNotIn("api_key", nodes_src)

    def test_calibration_cache_loads_with_weights_only_true(self):
        blob = {
            "layers": [-2, 15, 43, 51, 59],
            "text_seqs": [
                {
                    "x": torch.randn(4, 16, dtype=torch.bfloat16),
                    "y": torch.randn(4, 8, dtype=torch.bfloat16),
                    "typo": torch.tensor([False, True, True, False]),
                    "valid": torch.tensor([True, True, True, True]),
                    "split_pos": torch.tensor([0, 1, 2, 0], dtype=torch.uint8),
                    "split_len": torch.tensor([0, 2, 2, 0], dtype=torch.uint8),
                    "non_1to1": torch.tensor([False, True, True, False]),
                    "token_texts": ["a", "b", "c", "d"],
                    "prompt": "abcd",
                    "prompt_idx": 0,
                }
            ],
            "vis": (torch.empty(0, 8, dtype=torch.bfloat16), torch.empty(0, 8, dtype=torch.bfloat16), torch.empty(0, dtype=torch.long), 0),
            "ref_text_seqs": [],
            "sink_out": torch.zeros(8, dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as folder:
            p = os.path.join(folder, "cache.pt")
            torch.save(blob, p)
            loaded = torch.load(p, weights_only=True)
            self.assertEqual(loaded["layers"], [-2, 15, 43, 51, 59])
            self.assertEqual(loaded["text_seqs"][0]["token_texts"], ["a", "b", "c", "d"])
            self.assertTrue(torch.equal(loaded["text_seqs"][0]["split_pos"], blob["text_seqs"][0]["split_pos"]))


if __name__ == "__main__":
    unittest.main()
