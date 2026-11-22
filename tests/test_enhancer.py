"""Unit tests for the prompt enhancer (message building, answer cleaning, the chat request)."""

import base64
import http.server
import io
import json
import os
import sys
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import enhancer

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None


class FakeServer:
    """A chat endpoint that records each request and answers with a fixed reply."""

    def __init__(self, reply="A red door.", status=200):
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append((self.path, dict(self.headers), json.loads(body)))
                if status != 200:
                    data = b'{"error": "model is busy"}'
                else:
                    data = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class CleanTest(unittest.TestCase):
    def test_strips_reasoning_labels_fences_and_wrapping_quotes(self):
        self.assertEqual(enhancer.clean("<think>plan\nit</think>\nA red door."), "A red door.")
        self.assertEqual(enhancer.clean("**Enhanced prompt:** A red door."), "A red door.")
        self.assertEqual(enhancer.clean("Prompt: A red door."), "A red door.")
        self.assertEqual(enhancer.clean("```\nA red door.\n```"), "A red door.")
        self.assertEqual(enhancer.clean('"A red door."'), "A red door.")
        self.assertEqual(enhancer.clean("A red\n\n  door."), "A red door.")

    def test_keeps_quoted_text_inside_the_prompt(self):
        text = '"OPEN" replaces the sign "CLOSED"'
        self.assertEqual(enhancer.clean(text), text)
        self.assertEqual(enhancer.clean('Replace the sign with "OPEN"'), 'Replace the sign with "OPEN"')


class MessagesTest(unittest.TestCase):
    def test_no_pictures_uses_the_image_instructions(self):
        messages = enhancer.build_messages("  a red door ", ())
        self.assertEqual(messages[0], {"role": "system", "content": enhancer.IMAGE_SYSTEM})
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "a red door"},
                                                  {"type": "text", "text": enhancer.REMINDER}])

    def test_own_instructions_replace_the_built_in_ones(self):
        messages = enhancer.build_messages("x", (), " Be brief. ")
        self.assertEqual(messages[0]["content"], "Be brief.")
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "x"}])

    @unittest.skipIf(np is None, "needs numpy and Pillow")
    def test_pictures_are_labelled_in_order_before_the_request(self):
        small = np.zeros((8, 12, 3), dtype="float32")
        large = np.ones((1, 2000, 1000, 4), dtype="float32")
        large[..., 3] = 0.0  # fully transparent: shows as white
        messages = enhancer.build_messages("swap them", (small, large))
        self.assertEqual(messages[0]["content"], enhancer.EDIT_SYSTEM)
        content = messages[1]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "text", "image_url", "text", "text"])
        self.assertEqual([content[0]["text"], content[2]["text"], content[4]["text"], content[5]["text"]],
                         ["<image1>", "<image2>", "swap them", enhancer.REMINDER_EDIT])
        pictures = []
        for part in (content[1], content[3]):
            uri = part["image_url"]["url"]
            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
            pictures.append(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
        self.assertEqual(pictures[0].size, (12, 8))
        self.assertEqual(pictures[1].size, (512, 1024))
        self.assertGreater(pictures[1].getpixel((10, 10))[0], 250)


class ChatTest(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop(enhancer.API_KEY_ENV, None)

    def tearDown(self):
        os.environ.pop(enhancer.API_KEY_ENV, None)
        if self.saved is not None:
            os.environ[enhancer.API_KEY_ENV] = self.saved

    def test_request_and_cleaned_answer(self):
        server = FakeServer(reply='Prompt: "A red door, lit from the left."')
        try:
            answer = enhancer.enhance(server.url + "/", "red door", model="m1", temperature=0.5, seed=7, max_tokens=99)
        finally:
            server.close()
        self.assertEqual(answer, "A red door, lit from the left.")
        path, headers, body = server.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertNotIn("Authorization", headers)
        self.assertEqual((body["model"], body["temperature"], body["seed"], body["max_tokens"]), ("m1", 0.5, 7, 99))
        self.assertIs(body["enable_thinking"], False)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["messages"][1]["content"][0]["text"], "red door")

    def test_key_comes_from_the_environment_and_model_is_optional(self):
        os.environ[enhancer.API_KEY_ENV] = "secret"
        server = FakeServer()
        try:
            enhancer.enhance(server.url, "red door")
        finally:
            server.close()
        _, headers, body = server.requests[0]
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertNotIn("model", body)

    def test_empty_request_is_not_sent_and_empty_answer_keeps_the_request(self):
        server = FakeServer(reply="<think>nothing</think>")
        try:
            self.assertEqual(enhancer.enhance(server.url, "   "), "   ")
            self.assertEqual(server.requests, [])
            self.assertEqual(enhancer.enhance(server.url, " red door "), "red door")
        finally:
            server.close()

    def test_server_errors_name_the_endpoint(self):
        server = FakeServer(status=503)
        try:
            with self.assertRaises(RuntimeError) as caught:
                enhancer.enhance(server.url, "red door")
        finally:
            server.close()
        self.assertIn("HTTP 503", str(caught.exception))
        self.assertIn("model is busy", str(caught.exception))
        with self.assertRaises(RuntimeError) as caught:
            enhancer.enhance("http://127.0.0.1:9", "red door", timeout=2)
        self.assertIn("Cannot reach server", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
