from abc import ABC, abstractmethod

from KID.infra import Frame


class BasePolicy(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update_policy(self, frame: 'Frame', **kwargs) -> 'Frame':
        pass
