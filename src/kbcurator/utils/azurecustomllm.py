from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
import requests
from pydantic import PrivateAttr
from typing import List, Optional
import os
from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage, AIMessage
from langchain.messages import HumanMessage, AIMessage
from configparser import ConfigParser

class AzureCustomLLM(LLM):
    """Custom LLM wrapper for Langchain using Azure OpenAI."""

    _llm: AzureChatOpenAI = PrivateAttr()
    stop: Optional[List[str]] = None

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 9000,
        stream: bool = False,
        stop: Optional[List[str]] = None
    ):
        super().__init__()
        config = ConfigParser()
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
        if os.path.exists(config_path) and config.read(config_path):
            if config.has_section("Azure_OpenAI_llm_Model"):
                os.environ.setdefault(
                    "AZURE_OPENAI_API_KEY",
                    config.get("Azure_OpenAI_llm_Model", "api_key", fallback=""),
                )
                os.environ.setdefault(
                    "AZURE_OPENAI_ENDPOINT",
                    config.get("Azure_OpenAI_llm_Model", "api_base", fallback=""),
                )
                os.environ.setdefault(
                    "AZURE_OPENAI_DEPLOYMENT",
                    config.get("Azure_OpenAI_llm_Model", "llm_model", fallback=""),
                )
                os.environ.setdefault(
                    "OPENAI_API_VERSION",
                    config.get("Azure_OpenAI_llm_Model", "api_version", fallback=""),
                )

        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL")
        api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION")

        if not all([api_key, endpoint, deployment, api_version]):
            raise ValueError(
                "Missing Azure OpenAI settings. Set AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_DEPLOYMENT/OPENAI_API_VERSION "
                "or their AZURE_OPENAI_LLM_MODEL_* equivalents."
            )
        
        self._llm = AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )

    def _call(
        self,
        input: str,
        stop: Optional[List[str]] = None,
        sys_prompt: Optional[str] = None,
        history: Optional[List[dict]] = None
    ) -> str:
        messages = [HumanMessage(content=sys_prompt or "You are a helpful AI assistant.")]
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=input))
        try:
            response = self._llm.invoke(messages)
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error during Azure OpenAI API call: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "azure_openai_custom_llm"