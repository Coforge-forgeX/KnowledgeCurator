
class SearchAgent:
    def __init__(self, llm):
        self._llm = llm

    def run(self, query: str) -> str:
        return self._llm.generate(query)
