"""ComfyUI node definitions for Universal Conditioner."""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

try:
    from . import encoder, enhancer, projector
except ImportError:
    import encoder
    import enhancer
    import projector


class UniCondEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        projector.register_folder()
        return io.Schema(
            node_id="UniCondEncoder",
            display_name="Universal Conditioner",
            category="conditioning/universal_conditioner",
            description=(
                "Forward prompts and reference images to an external hidden-states server "
                "(POST /v1/hidden_states) and project them into a ComfyUI diffusion model's "
                "conditioning space without loading the encoder into ComfyUI VRAM."
            ),
            inputs=[
                io.String.Input(
                    "url",
                    default=encoder.DEFAULT_URL,
                    tooltip="Base URL of your external hidden-states server (for example http://127.0.0.1:8000).",
                ),
                io.String.Input(
                    "model",
                    default="",
                    optional=True,
                    tooltip="Optional model id sent to the server; leave empty for the server default.",
                ),
                io.Int.Input(
                    "layer",
                    default=-1,
                    min=-2,
                    max=127,
                    optional=True,
                    tooltip="-1 reads the layer list stored in the adapter (or final norm without an adapter).",
                ),
                io.Combo.Input(
                    "adapter",
                    options=["none"] + projector.list_adapters(),
                    default="none",
                    tooltip="Adapter safetensors file from models/universal_conditioner (or models/clip_projections).",
                ),
                io.Combo.Input(
                    "profile",
                    options=list(encoder.PROFILES),
                    default="auto",
                    optional=True,
                    tooltip="Target conditioning layout ('auto' reads target_profile from the adapter).",
                ),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
        )

    @classmethod
    def execute(
        cls,
        url,
        model="",
        layer=-1,
        adapter="none",
        profile="auto",
    ) -> io.NodeOutput:
        return io.NodeOutput(
            encoder.load(
                url=url,
                model=model,
                layer=layer,
                adapter=adapter,
                profile=profile,
            )
        )


class UniCondPromptEnhancer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniCondPromptEnhancer",
            display_name="Universal Conditioner Prompt Enhancer",
            category="conditioning/universal_conditioner",
            description=(
                "Rewrite a short request into a finished prompt with the language model on your server "
                "(POST /v1/chat/completions). With pictures connected it writes an edit prompt that "
                "names them as <image1>, <image2>, and so on; without pictures it writes an image prompt."
            ),
            inputs=[
                io.String.Input(
                    "url",
                    default=encoder.DEFAULT_URL,
                    tooltip="Base URL of your server, the same one the Universal Conditioner node uses.",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="What you want, in a few words. Put text that must appear in the picture in double quotes.",
                ),
                io.Boolean.Input(
                    "enhance",
                    default=True,
                    tooltip="Off passes your prompt through unchanged, to compare with and without.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Change it to get another wording of the same request.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.3,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip="Higher values vary the wording more.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=400,
                    min=32,
                    max=4096,
                    optional=True,
                    tooltip="Upper limit on the length of the rewritten prompt.",
                ),
                io.String.Input(
                    "model",
                    default="",
                    optional=True,
                    tooltip="Optional model id sent to the server; leave empty for the server default.",
                ),
                io.String.Input(
                    "instructions",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Your own instructions for the rewrite. Leave empty for the built-in ones.",
                ),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="The pictures being edited, in the same order as on the text encode node.",
                ),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(
        cls,
        url,
        prompt,
        enhance=True,
        seed=0,
        temperature=0.3,
        max_tokens=400,
        model="",
        instructions="",
        images: io.Autogrow.Type = None,
    ) -> io.NodeOutput:
        if not enhance:
            return io.NodeOutput(prompt)
        images = images or {}
        ordered = [images[name][0] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1]))
                   if images[name] is not None]
        return io.NodeOutput(
            enhancer.enhance(
                url,
                prompt,
                images=ordered,
                model=model,
                system=instructions,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed % (2 ** 31),
            )
        )


class UniCondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [UniCondEncoder, UniCondPromptEnhancer]


async def comfy_entrypoint() -> UniCondExtension:
    projector.register_folder()
    return UniCondExtension()
