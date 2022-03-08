from abc import ABC, abstractmethod
from typing import Dict, Optional

from KID.infra import Frame


class BaseEnv(ABC):
    """Base class for LM envs.

    """

    def __init__(self, **kwargs: Optional[Dict]) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the env."""
        pass

    @abstractmethod
    def step(self, frame: 'Frame') -> 'Frame':
        """Execute one-step interaction.

        :param frame: a tuple (past, last).
        :return: a tuple of (obs, rew, is_done, info)
        """
        pass
