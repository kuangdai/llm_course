from typing import List, Optional, Any

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult, Generation
from langchain_core.language_models import BaseLLM


class LLMInterface:
    def __init__(self, config):
        # Load configurations directly from config dictionary
        self.config = config

    def retrieve_faiss(self, text: str) -> str:
        """Retrieve similar poems using FAISS-based similarity search."""
        payload = {"text": text, "k": self.config.get("retrieve_poem_count", 1)}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{self.config['server_url']}/retrieve_faiss", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("retrieved_poems", "Error: No poems retrieved.")
        except requests.RequestException as e:
            return f"Error in retrieve_faiss: {e}"

    def retrieve_nx_graph(self, text: str) -> str:
        """Retrieve poems based on keyword and graph traversal."""
        payload = {
            "text": text,
            "k": self.config.get("retrieve_poem_count", 1),
            "depth": self.config.get("retrieve_depth", 2),
            "depth_decay": self.config.get("retrieve_depth_decay", 0.5)
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{self.config['server_url']}/retrieve_nx_graph", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("retrieved_poems", "Error: No poems retrieved.")
        except requests.RequestException as e:
            return f"Error in retrieve_nx_graph: {e}"

    def generate(self, text: str) -> str:
        """Generate a response based on input text using the LLM."""
        payload = {
            "text": text,
            "temperature": self.config.get("response_temperature", 0.5),
            "max_new_tokens": self.config.get("response_max_new_tokens", 50)
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{self.config['server_url']}/generate", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("generated_text", "Error: No generated text returned.")
        except requests.RequestException as e:
            return f"Error in generate: {e}"


class CustomLLM(BaseLLM):
    # Define the LLM interface as a class attribute
    llm_interface: LLMInterface

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        """Generate text from the custom LLM model."""
        if not self.llm_interface:
            raise ValueError("LLMInterface is not configured. Call CustomLLM.configure() first.")

        result = self.llm_interface.generate(prompt)
        return result

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
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
