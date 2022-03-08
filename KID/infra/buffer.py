from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from KID.infra.frame import Frame

IndexType = Union[slice, int, np.ndarray, List[int]]


class BaseBuffer(ABC):
    """The base class of buffer."""

    def __int__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self._max_size = kwargs.get('size')

        # key attributes for any 'Buffer'
        self.past = None
        self.last = None
        self.kid_past = None
        self.gen_ids = None
        self.reward = None
        self.is_done = None
        self.info = None

    @abstractmethod
    def add(
        self,
        past: Union[torch.tensor, np.ndarray],
        last: Union[torch.tensor, np.ndarray, int],
        kid_past: Union[torch.tensor, np.ndarray],
        gen_ids: Union[torch.tensor, np.ndarray],
        reward: Union[torch.tensor, int, float],
        is_done: bool,
        info: Dict[Any, Any] = {}
    ) -> None:
        """Add data into buffer."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the buffer. Clear the data within the buffer."""
        pass


class ReplayBuffer(BaseBuffer):

    def __init__(self, size: int) -> None:
        super(ReplayBuffer, self).__init__()
        self._maxsize = size  # max size allowed
        self.cur_index = self._size = 0  # current size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: IndexType) -> Optional['Frame']:
        # override get method
        if len(self) == 0:
            return None
        else:
            return Frame(
                past=self.past[index],  # type: ignore
                last=self.last[index],  # type: ignore
                kid_past=self.kid_past[index],  # type: ignore
                gen_ids=self.gen_ids[index],  # type: ignore
                reward=self.reward[index],  # type: ignore
                is_done=self.is_done[index],  # type: ignore
                info=self.info[index]  # type: ignore
            )

    def _add_to_buffer(
        self, name: str, data: Union[torch.Tensor, np.ndarray, str, int, dict]
    ) -> None:
        """
        Helper function to handle some edge cases.

        :param name:
        :param data:
        :return:
        """
        if data is None:
            return

        # if no such key, create a buffer space
        if self.__dict__.get(name, None) is None:
            if isinstance(data, np.ndarray):
                self.__dict__[name] = np.zeros([self._maxsize, *data.shape])
            elif isinstance(data, torch.Tensor):
                self.__dict__[name] = torch.zeros([self._maxsize, *data.shape])
            else:
                self.__dict__[name] = np.zeros([self._maxsize], dtype=type(data))

        self.__dict__[name][self.cur_index] = data

    def update(self, buffer: 'ReplayBuffer') -> None:
        """update the current buffer with input ReplayBuffer type data
        :param buffer:
        :return:
        """
        i = begin = buffer.cur_index % len(buffer)
        while True:
            self.add(
                buffer.last[i],  # type: ignore
                buffer.past[i],  # type: ignore
                buffer.kid_past[i],  # type: ignore
                buffer.gen_ids[i],  # type: ignore
                buffer.reward[i],  # type: ignore
                buffer.is_done[i],  # type: ignore
                buffer.info[i]  # type: ignore
            )
            i = (i + 1) % len(buffer)
            if i == begin:
                break

    def add(
        self,
        past: Union[torch.tensor, np.ndarray],
        last: Union[torch.tensor, np.ndarray, int],
        kid_past: Union[torch.tensor, np.ndarray],
        gen_ids: Union[torch.tensor, np.ndarray],
        reward: Union[torch.tensor, int, float],
        is_done: bool,
        info: Dict[Any, Any] = {}
    ) -> None:

        self._add_to_buffer('past', past)
        self._add_to_buffer('last', last)
        self._add_to_buffer('kid_past', kid_past)
        self._add_to_buffer('gen_ids', gen_ids)
        self._add_to_buffer('reward', reward)
        self._add_to_buffer('is_done', is_done)
        self._add_to_buffer('info', info)

        if self._maxsize > 0:
            self._size = min(self._size + 1, self._maxsize)
            self.cur_index = (self.cur_index + 1) % self._maxsize
        else:
            self._size = self.cur_index = self.cur_index + 1

    def reset(self) -> None:
        self.cur_index = self._size = 0

    def sample(self, batch_size: int) -> Tuple[Optional['Frame'], Any]:
        """
        Sample a batch of data from the buffer
        :param batch_size:
        :return:
        """
        if batch_size > 0:
            indices = np.random.choice(self._size, batch_size)
        else:  # sample all the buffer elements, and keep the time order
            indices = np.concatenate(
                [
                    np.arange(self.cur_index,
                              self._size),  # priority is to fetch the latest
                    np.arange(0, self.cur_index),
                ]
            )
        return self[indices], indices


class ListReplayBuffer(ReplayBuffer):
    """Use list to store frames as buffer."""

    def __init__(self) -> None:
        super().__init__(size=0)

    def _add_to_buffer(self, name: str, data: Any) -> None:
        if data is None:
            return
        if self.__dict__.get(name, None) is None:
            self.__dict__[name] = []  # use a list for now
        self.__dict__[name].append(data)

    def reset(self) -> None:
        self.cur_index = self._size = 0
        for k in list(self.__dict__.keys()):
            if not k.startswith('_'):  # if it is not protected attribute
                self.__dict__[k] = []
