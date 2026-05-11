from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
import requests
from pydantic import PrivateAttr
from typing import List, Optional
import os
from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import HumanMessage, AIMessage
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
        config_path = os.path.abspath(os.path.join(os.getcwd(), 'config.ini'))
        if config.read(config_path):
            print(f"Loading config from: {config_path}")
            os.environ["AZURE_OPENAI_API_KEY"] = config.get("Azure_OpenAI_llm_Model", "api_key")
            os.environ["AZURE_OPENAI_ENDPOINT"] = config.get("Azure_OpenAI_llm_Model", "api_base")
            os.environ["AZURE_OPENAI_DEPLOYMENT"] = config.get("Azure_OpenAI_llm_Model", "llm_model")
            os.environ["OPENAI_API_VERSION"] = config.get("Azure_OpenAI_llm_Model", "api_version")
        else:
            # Fall back to environment variables (standard deployment path)
            missing = [v for v in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
                       if not os.getenv(v)]
            if missing:
                raise EnvironmentError(
                    f"config.ini not found at {config_path} and the following env vars are not set: "
                    + ", ".join(missing)
                )
        
        self._llm = AzureChatOpenAI(
            api_key= os.environ["AZURE_OPENAI_API_KEY"],
            api_version= os.environ["OPENAI_API_VERSION"],
            azure_endpoint= os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment= os.environ["AZURE_OPENAI_DEPLOYMENT"],
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