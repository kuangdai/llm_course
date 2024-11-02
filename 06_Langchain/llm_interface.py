from typing import List, Optional, Any

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult, Generation
from langchain_core.language_models import BaseLLM


class LLMInterface:
    @staticmethod
    def retrieve_faiss(text: str, server_url: str, retrieve_poem_count: int = 1) -> str:
        """Retrieve similar poems using FAISS-based similarity search."""
        payload = {"text": text, "k": retrieve_poem_count}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{server_url}/retrieve_faiss", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("retrieved_poems", "Error: No poems retrieved.")
        except requests.RequestException as e:
            return f"Error in retrieve_faiss: {e}"

    @staticmethod
    def retrieve_nx_graph(text: str, server_url: str, retrieve_poem_count: int = 1, depth: int = 2,
                          depth_decay: float = 0.5) -> str:
        """Retrieve poems based on keyword and graph traversal."""
        payload = {
            "text": text,
            "k": retrieve_poem_count,
            "depth": depth,
            "depth_decay": depth_decay
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{server_url}/retrieve_nx_graph", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("retrieved_poems", "Error: No poems retrieved.")
        except requests.RequestException as e:
            return f"Error in retrieve_nx_graph: {e}"

    @staticmethod
    def generate(text: str, server_url: str, temperature: float = 0.5, max_new_tokens: int = 50) -> str:
        """Generate a response based on input text using the LLM."""
        payload = {
            "text": text,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{server_url}/generate", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("generated_text", "Error: No generated text returned.")
        except requests.RequestException as e:
            return f"Error in generate: {e}"


class CustomLLM(BaseLLM):
    server_url: str
    temperature: float
    max_new_tokens: int

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        """Generate text from the custom LLM model."""
        # Directly call generate with class-level server_url, temperature, and max_new_tokens
        result = LLMInterface.generate(
            text=prompt,
            server_url=self.server_url,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )
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
