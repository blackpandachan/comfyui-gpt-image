# New text generation node for ComfyUI using OpenAI
import os
from inspect import cleandoc
import openai
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict


class GPTTextGenerate(ComfyNodeABC):
    """Generate arbitrary text using the OpenAI ChatCompletion API.

    This node is useful for tasks such as short stories, summaries, or
    general purpose text generation. Provide a user prompt and an optional
    system instruction. The assistant's response is returned as text.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "User prompt for the model",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "You are a helpful assistant.",
                        "tooltip": "Optional system message",
                    },
                ),
                "api_key": (
                    IO.STRING,
                    {
                        "default": os.environ.get("OPENAI_API_KEY", ""),
                        "display": "string",
                        "tooltip": "OpenAI API key (fallback to OPENAI_API_KEY)",
                    },
                ),
                "model": (
                    IO.STRING,
                    {
                        "default": "gpt-4o",
                        "display": "string",
                        "tooltip": "OpenAI model for text generation",
                    },
                ),
                "temperature": (
                    IO.FLOAT,
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "number",
                        "tooltip": "Sampling temperature",
                    },
                ),
                "max_tokens": (
                    IO.INT,
                    {
                        "default": 256,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Maximum tokens in the response",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "generate"
    CATEGORY = "Text"
    DESCRIPTION = cleandoc(__doc__ or "")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.8,
        max_tokens: int = 256,
    ):
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise Exception("OpenAI API key required")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.get("content", "")
        return (content,)

NODE_CLASS_MAPPINGS = {"GPTTextGenerate": GPTTextGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"GPTTextGenerate": "GPT Text Generate"}

