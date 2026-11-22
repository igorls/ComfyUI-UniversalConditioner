"""Prompt enhancement over the chat endpoint of the same server that serves the hidden states.

The language model that encodes the prompt can also rewrite it first. A short instruction and the
reference pictures go to POST /v1/chat/completions (the OpenAI chat format) and one finished prompt
comes back as plain text.
"""

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

API_KEY_ENV = "UNICOND_API_KEY"
MAX_SIDE = 1024

EDIT_SYSTEM = """You rewrite a short image-editing request into one precise prompt for an image editing model.

The user's pictures are attached in order. Refer to them in the prompt as <image1>, <image2>, and so on, the way the request does. Look at the pictures before you write.

Rules:
- Keep the user's intent exactly. Do not add changes that were not asked for.
- Say what changes, in concrete visual terms: subject, position, pose, material, colour, lighting.
- Say what must stay the same, naming what you see in the pictures: the person's face, hair and clothes, the framing, the background, the picture's style. Skip this for anything the request replaces.
- Text to put in the picture goes in double quotes, exactly as the user wrote it, in its original language and spelling. Text to replace is quoted exactly as it appears in the picture. Never translate or correct quoted text.
- Write the prompt in the same language as the request: a request in Portuguese gets a prompt in Portuguese, a request in Japanese gets a prompt in Japanese.
- One paragraph, at most 120 words. No lists, no headings, no explanation, no remarks about these rules.

Answer with the prompt only."""

IMAGE_SYSTEM = """You rewrite a short idea into one precise prompt for an image generation model.

Rules:
- Keep the user's idea exactly. Add only detail that makes it concrete: subject, setting, composition, lighting, colour, material, style.
- Text to put in the picture goes in double quotes, exactly as the user wrote it, in its original language and spelling. Never translate or correct quoted text, and never mention the quotation marks themselves.
- Write the prompt in the same language as the idea: an idea in Portuguese gets a prompt in Portuguese, an idea in Japanese gets a prompt in Japanese.
- One paragraph, at most 120 words. No lists, no headings, no explanation, no remarks about these rules.

Answer with the prompt only."""

# Repeated after the request, where the model follows it more reliably than from the instructions alone.
REMINDER = "(Answer with the prompt only, in the language the line above is written in.)"
REMINDER_EDIT = ("(Answer with the prompt only, in the language the line above is written in. "
                 "Name the pictures only as <image1>, <image2>, and so on.)")

_THINK = re.compile(r"<think>.*?</think>", re.S)
_LABEL = re.compile(r"^\s*(?:\*\*)?(?:enhanced |rewritten |final |edit(?:ing)? )?prompt(?:\*\*)?\s*:\s*(?:\*\*)?", re.I)
_QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("`", "`"))


def system_prompt(has_images, override=""):
    """The instructions the model rewrites under: the caller's own, or the built-in ones for the task."""
    if override and override.strip():
        return override.strip()
    return EDIT_SYSTEM if has_images else IMAGE_SYSTEM


def image_data_uri(image, max_side=MAX_SIDE):
    """One picture as a JPEG data URI: a [H, W, C] tensor or array in 0..1, or a PIL image.

    Alpha is laid over white, and the long side is brought down to max_side.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        import numpy as np

        array = image.detach().cpu().float().numpy() if hasattr(image, "detach") else np.asarray(image, dtype="float32")
        if array.ndim == 4:
            array = array[0]
        if array.shape[-1] > 3:
            alpha = array[..., 3:4]
            array = array[..., :3] * alpha + (1.0 - alpha)
        elif array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype("uint8"), "RGB")
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(text, images=(), system=""):
    """The chat messages for one rewrite: the instructions, then each picture under its label, then the request."""
    content = []
    for index, image in enumerate(images):
        content.append({"type": "text", "text": "<image%d>" % (index + 1)})
        content.append({"type": "image_url", "image_url": {"url": image_data_uri(image)}})
    content.append({"type": "text", "text": text.strip()})
    if not (system and system.strip()):
        content.append({"type": "text", "text": REMINDER_EDIT if images else REMINDER})
    return [
        {"role": "system", "content": system_prompt(bool(images), system)},
        {"role": "user", "content": content},
    ]


def clean(text):
    """The model's answer as a bare prompt: no reasoning block, label, code fence or wrapping quotes."""
    text = _THINK.sub("", text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    text = _LABEL.sub("", text).strip()
    for left, right in _QUOTES:
        inner = text[1:-1]
        if len(text) > 1 and text.startswith(left) and text.endswith(right) and left not in inner and right not in inner:
            text = text[1:-1].strip()
            break
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def chat(url, messages, model="", max_tokens=400, temperature=0.3, seed=0, api_key="", timeout=120):
    """POST /v1/chat/completions and return the answer text."""
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "seed": int(seed),
        "stream": False,
        "enable_thinking": False,
    }
    if model:
        payload["model"] = model
    headers = {"Content-Type": "application/json"}
    token = api_key or os.environ.get(API_KEY_ENV, "")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(endpoint, json.dumps(payload).encode("utf-8"), headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError("Server /v1/chat/completions HTTP %d from %s: %s" % (err.code, endpoint, detail)) from err
    except (urllib.error.URLError, OSError) as err:
        raise RuntimeError("Cannot reach server at %s: %s" % (endpoint, err)) from err
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as err:
        raise RuntimeError("Server /v1/chat/completions at %s returned no message" % endpoint) from err


def enhance(url, text, images=(), model="", system="", max_tokens=400, temperature=0.3, seed=0, api_key="", timeout=120):
    """Rewrite one request into a finished prompt. An empty request, or an empty answer, returns the request."""
    if not text or not text.strip():
        return text or ""
    answer = clean(chat(url, build_messages(text, images, system), model=model, max_tokens=max_tokens,
                        temperature=temperature, seed=seed, api_key=api_key, timeout=timeout))
    return answer or text.strip()
