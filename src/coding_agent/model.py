from typing import Protocol

from coding_agent.contracts import ModelResponse


class Model(Protocol):
    def generate(self, history) -> ModelResponse:
        ...


class FakeModel:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.index = 0

    def generate(self, history) -> ModelResponse:
        response = self.responses[self.index]
        self.index += 1
        return response