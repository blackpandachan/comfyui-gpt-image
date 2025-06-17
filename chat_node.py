import os
from inspect import cleandoc

import openai

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict


class GPTPromptChat(ComfyNodeABC):
    """Generate an advanced prompt using the OpenAI ChatCompletion API.

    This node takes a user instruction and optional system message and
    returns the assistant's response which can be fed into the GPT Image node.
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
                        "tooltip": "Chat message from the user",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "You are an expert prompt engineer for image generation.",
                        "tooltip": "System instructions for the assistant",
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
                        "tooltip": "OpenAI model for chat generation",
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
            },
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "generate"
    CATEGORY = "Prompt"
    DESCRIPTION = cleandoc(__doc__ or "")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert prompt engineer for image generation.",
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.8,
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
            model=model, messages=messages, temperature=temperature
        )
        content = response.choices[0].message.get("content", "")
        return (content,)


NODE_CLASS_MAPPINGS = {"GPTPromptChat": GPTPromptChat}
NODE_DISPLAY_NAME_MAPPINGS = {"GPTPromptChat": "GPT Prompt Chat"}
