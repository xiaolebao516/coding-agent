from typing import Protocol

from coding_agent.contracts import ModelResponse


class Model(Protocol):
    provider: str  # e.g. "deepseek"; names only, never credentials or config
    model: str  # e.g. "deepseek-flash"

    def generate(self, history) -> ModelResponse:
        ...


class FakeModel:
    provider = "fake"
    model = "fake"

    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.index = 0

    def generate(self, history) -> ModelResponse:
        response = self.responses[self.index]
        self.index += 1
        return response