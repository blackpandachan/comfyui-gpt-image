import os
from inspect import cleandoc

import openai

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict


class PromptToWorkflowWizard(ComfyNodeABC):
    """Generate a ComfyUI workflow graph based on a textual description.

    This node uses the OpenAI ChatCompletion API to translate natural
    language instructions into a JSON workflow that can be loaded by
    ComfyUI. The exact structure of the workflow is determined by the
    language model. The node returns the raw text from the API, which is
    expected to be a JSON string describing the nodes and their
    connections.
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
                        "tooltip": "Describe the desired workflow",
                    },
                ),
            },
            "optional": {
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
                        "tooltip": "OpenAI model for graph generation",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "generate"
    CATEGORY = "Workflow"
    DESCRIPTION = cleandoc(__doc__ or "")

    def generate(self, prompt: str, api_key: str | None = None, model: str = "gpt-4o"):
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise Exception("OpenAI API key required")

        system = (
            "You are a ComfyUI expert. Return only the JSON for a ComfyUI "
            "workflow graph that satisfies the user's request."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(model=model, messages=messages)
        content = response.choices[0].message.get("content", "")
        return (content,)


NODE_CLASS_MAPPINGS = {"PromptToWorkflowWizard": PromptToWorkflowWizard}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptToWorkflowWizard": "Prompt To Workflow"}
