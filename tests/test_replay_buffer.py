import torch

from src.agents.replay_buffer import ReplayBuffer


def _fill(buffer: ReplayBuffer, count: int) -> None:
    for i in range(count):
        image = torch.full((1, 2, 2), fill_value=float(i))
        next_image = torch.full((1, 2, 2), fill_value=float(i + 1))
        buffer.push(image, action=i % 4, reward=float(i), next_image=next_image)


def test_push_and_len_respect_capacity():
    buffer = ReplayBuffer(capacity=5)
    _fill(buffer, count=8)

    assert len(buffer) == 5


def test_sample_returns_expected_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=20)
    _fill(buffer, count=20)

    images, actions, rewards, next_images = buffer.sample(batch_size=4, device="cpu")

    assert images.shape == (4, 1, 2, 2)
    assert next_images.shape == (4, 1, 2, 2)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32


def test_sample_states_returns_only_the_image_half_of_the_transition():
    buffer = ReplayBuffer(capacity=20)
    _fill(buffer, count=20)

    states = buffer.sample_states(batch_size=6, device="cpu")

    assert states.shape == (6, 1, 2, 2)


def test_sample_states_caps_batch_size_to_buffer_length():
    buffer = ReplayBuffer(capacity=20)
    _fill(buffer, count=3)

    states = buffer.sample_states(batch_size=100, device="cpu")

    assert states.shape[0] == 3
