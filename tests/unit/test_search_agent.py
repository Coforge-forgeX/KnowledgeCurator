
from agent_search.service.search_agent import SearchAgent

class FakeLLM:
    def generate(self, prompt: str) -> str:
        return f"FAKE::{prompt}"

def test_agent():
    a = SearchAgent(FakeLLM())
    assert a.run('q') == 'FAKE::q'
