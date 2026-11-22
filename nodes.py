"""ComfyUI node definitions for Universal Conditioner."""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

try:
    from . import encoder, projector
except ImportError:
    import encoder
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


class UniCondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [UniCondEncoder]


async def comfy_entrypoint() -> UniCondExtension:
    projector.register_folder()
    return UniCondExtension()
