import numpy as np
import pytest
import torch

from KID.infra import Frame


def test_frame():
    # test init and assign values
    frame = Frame(obs=torch.tensor(1), nparray=np.zeros([3, 4]))
    frame.obs = torch.tensor(1)
    assert frame.obs == torch.tensor(1)

    # test set and get item
    frame['past'] = [1, 2, 3, 4]
    assert frame['past'] == frame.past

    # test append
    frame.append(frame)
    assert torch.equal(frame.obs, torch.tensor([[1], [1]]))
    assert frame.nparray.shape == (6, 4)
    assert frame.get_by_id(0).obs == frame.get_by_id(1).obs

    # check boundary
    with pytest.raises(IndexError):
        print(frame.get_by_id(2))

    # check iteration
    frame.obs = np.arange(5)
    for i, b in enumerate(frame.get_frames(1)):
        assert b.obs == frame.get_by_id(i).obs


if __name__ == '__main__':
    test_frame()
