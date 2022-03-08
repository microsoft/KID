from dummy_env import DummyEnv

from KID.infra import ReplayBuffer


def test_buffer(size=10, buffer_size=20):
    env = DummyEnv(size)
    buf = ReplayBuffer(buffer_size)
    buf2 = ReplayBuffer(buffer_size)
    past = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10

    for i, a in enumerate(action_list):
        _, last, kid_past, gen_ids, reward, is_done, info = env.step(a)
        buf.add(past, last, kid_past, gen_ids, reward, is_done, info)
        past = kid_past
        assert len(buf) == min(buffer_size, i + 1), print(len(buf), i)

    data, indices = buf.sample(buffer_size * 2)
    assert (indices < len(buf)).all()
    assert (data.past < size).all()
    assert (0 <= data.is_done).all() and (data.is_done <= 1).all()
    assert len(buf) > len(buf2)

    buf2.update(buf)
    assert len(buf) == len(buf2)
    assert buf2[0].past == buf[5].past
    assert buf2[-1].past == buf[4].past


if __name__ == '__main__':
    test_buffer()
