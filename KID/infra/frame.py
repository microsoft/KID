from typing import Generator, Optional, Union

import numpy as np
import torch

ItemType = Union[torch.Tensor, int, list, str, np.ndarray]


class Frame(object):

    def __init__(self, **kwargs: Optional[ItemType]):
        super(Frame, self).__init__()
        self.__dict__.update(kwargs)

    def __getitem__(self, item: str) \
            -> Optional[Union[torch.Tensor, int, list, np.ndarray]]:
        return self.__dict__.get(item, None)

    def __setitem__(self, key: str,
                    value: Optional[ItemType]) \
            -> 'Frame':
        self.__dict__[key] = value
        return self

    def get_by_id(self, index: Union[int, np.ndarray]) -> 'Frame':
        """
        return the data at index position of the whole frames
        :param index: position index
        :return: the data
        """
        cur = Frame()
        for k in self.__dict__.keys():
            if self.__dict__[k] is not None:
                cur.__dict__.update(**{k: self.__dict__[k][index]})
        return cur

    def get_frames(self, size: int) -> Generator:
        """
        return a certain size of frames
        :param size:
        :return:
        """
        assert isinstance(size, int) and size > 0, "Please use a size above zero!"
        size = min(size, self.length(return_max=True))
        i = 0
        index = np.arange(size)
        while i < size:
            yield self.get_by_id(index[i:i + size])
            i += size

    def append(self, frame: 'Frame') -> None:
        """
        append a data to current trajectories
        :param frame:
        :return:
        """
        assert isinstance(frame, Frame)
        for key in frame.__dict__.keys():
            if frame.__dict__[key] is None:
                continue
            if key not in self.__dict__.keys():
                self.__dict__[key] = frame.__dict__[key]
            elif isinstance(frame.__dict__[key], np.ndarray):
                self.__dict__[key] = np.row_stack(
                    (self.__dict__[key], frame.__dict__[key])
                )
            elif isinstance(frame.__dict__[key], torch.Tensor):
                self.__dict__[key] = torch.vstack(
                    (self.__dict__[key], frame.__dict__[key])
                )
            elif isinstance(frame.__dict__[key], list):
                self.__dict__[key] = self.__dict__[key].extend(frame.__dict__[key])
            else:
                s = 'No support type for appending frames' \
                    + str(type(frame.__dict__[key])) \
                    + 'in class Frame.'
                raise TypeError(s)

    def length(self, return_max: bool = True) -> int:
        """
        return the length of the frames
        :param return_max: whether or not return the max length
        :return:
        """
        total_lengths = [
            len(self.__dict__[key]) for key in self.__dict__.keys()
            if self.__dict__[key] is not None
        ]
        return max(total_lengths) if return_max else min(total_lengths)
