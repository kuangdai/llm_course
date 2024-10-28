# Introduction


```python
from typing import List, Optional, Any

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult, Generation
from langchain_core.language_models import BaseLLM
from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import (
    StrInput,
    FloatInput,
    IntInput,
)


class CustomLLMComponent(LCModelComponent):
    display_name = "My LLM"
    description = "Generates text using a custom LLM server."
    icon = "Heart"
    name = "MyLLMModel"

    inputs = LCModelComponent._base_inputs + [
        StrInput(
            name="llm_server_url",
            display_name="LLM Server URL",
            advanced=False,
            info="URL for the custom LLM server.",
            value="https://ABC.loca.lt",
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            info="Sampling temperature for text generation.",
            advanced=False,
        ),
        IntInput(
            name="max_new_tokens",
            display_name="Max New Tokens",
            value=50,
            info="Maximum number of tokens to generate.",
            advanced=False,
        ),
    ]

    def build_model(self) -> LanguageModel:
        # Instantiate CustomLLM with the appropriate parameters
        return self.CustomLLM(
            llm_server_url=(self.llm_server_url or "https://ABC.loca.lt") + "/generate_llama3",
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )

    class CustomLLM(BaseLLM):
        """Wrapper class for custom LLM model to comply with the LanguageModel interface."""

        # Define fields as class-level variables
        llm_server_url: str
        temperature: float
        max_new_tokens: int

        def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,  # noqa
                run_manager: Optional[CallbackManagerForLLMRun] = None,  # noqa
                **kwargs: Any,  # noqa
        ) -> str:
            """Generate text from the custom LLM model."""
            payload = {
                "text": prompt,
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens
            }
            headers = {
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(self.llm_server_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result.get("generated_text", "No generated text returned.")
            except requests.RequestException as e:
                return f"Error generating text: {e}"

        def _generate(
                self,
                prompts: List[str],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
        ) -> LLMResult:
            """Implements the required _generate method to handle batch generation."""
            generations = []
            for prompt in prompts:
                text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                generations.append([Generation(text=text)])

            return LLMResult(generations=generations)

        @property
        def _llm_type(self) -> str:
            return "custom_llm"
```

```python
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message

class SingleTurnFilterComponent(Component):
    display_name = "Single-turn Filter"
    description = "Filters the input to keep only the initial response, removing any multi-turn conversation generated (e.g., '\\nUser:', '\\nAI:')."
    icon = "scissors-line-dashed"
    name = "SingleTurnFilter"

    inputs = [
        MessageTextInput(
            name="text",
            display_name="Text",
            info="The input text containing conversation segments.",
        ),
    ]

    outputs = [
        Output(display_name="Filtered Response", name="filtered_response", method="filter"),
    ]

    def filter(self) -> Message:
        # Access the input text directly
        text = self.text
        
        # Remove any multi-turn conversation markers, keeping only the initial response
        filtered_response = text.split("\nUser:")[0].split("\nAI:")[0]
        
        # Set the component status and return the filtered response as a Message
        self.status = filtered_response
        return Message(text=filtered_response)
```