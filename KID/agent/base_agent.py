from abc import ABC, abstractmethod
from typing import Any

from KID.infra import Frame


class BaseAgent(ABC):
    """Base class for Agent that communicates between Env and Policy.

    """

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, action: Frame) -> Frame:
        pass

    def post_func(self, inp: Any) -> Any:
        pass
