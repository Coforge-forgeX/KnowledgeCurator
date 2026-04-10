
from common_adapters.ai.openai import OpenAIAdapter

class DummyClient:
    def generate(self, prompt: str, model: str) -> str:
        return f"{model}::{prompt}"

def build_llm():
    return OpenAIAdapter(DummyClient(), 'dummy-model')
