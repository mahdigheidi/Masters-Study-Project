import pytest

from src.environments.classification_mdp import (
    ClassificationMDP,
    ClassificationMDPSpec,
    make_random_labels,
)
from src.environments.easy_mdp import EasyMDP
from src.environments.hard_mdp import HardMDP
from src.environments.sparse_mdp import SparseMDP
from tests.conftest import make_toy_dataset


def test_make_random_labels_are_in_range_and_deterministic_for_seed():
    dataset = make_toy_dataset()
    labels_a = make_random_labels(dataset, num_states=10, seed=0)
    labels_b = make_random_labels(dataset, num_states=10, seed=0)

    assert labels_a == labels_b
    assert len(labels_a) == len(dataset)
    assert all(0 <= label < 10 for label in labels_a)


def test_classification_mdp_rejects_missing_states():
    dataset = make_toy_dataset(num_states=10, samples_per_state=8)
    labels = [0] * len(dataset)  # every image mapped to state 0

    with pytest.raises(ValueError, match="Missing states"):
        ClassificationMDP(dataset, ClassificationMDPSpec(name="broken"), labels=labels)


def test_easy_mdp_reward_is_one_iff_action_matches_state():
    dataset = make_toy_dataset()
    env = EasyMDP(dataset, seed=0)

    env.state = 3
    reward, _ = env.transition(action=3)
    assert reward == 1.0

    env.state = 3
    reward, _ = env.transition(action=7)
    assert reward == 0.0


def test_easy_mdp_step_returns_observation_matching_new_state():
    dataset = make_toy_dataset()
    env = EasyMDP(dataset, seed=0)

    observation, reward, next_state = env.step(action=env.state)
    assert reward == 1.0
    # The toy dataset encodes the label directly in the pixel values.
    assert observation.unique().item() == float(next_state)


def test_hard_mdp_uses_shuffled_labels_not_true_labels():
    dataset = make_toy_dataset(num_states=10, samples_per_state=20)
    env = HardMDP(dataset, seed=0)

    true_labels = dataset.targets
    assert env.labels != true_labels


def test_sparse_mdp_reward_only_at_state_nine_action_nine():
    dataset = make_toy_dataset()
    env = SparseMDP(dataset, seed=0)

    env.state = 9
    reward, next_state = env.transition(action=9)
    assert reward == 1.0
    assert next_state == 0  # wraps around modulo num_states

    env.state = 5
    reward, next_state = env.transition(action=5)
    assert reward == 0.0
    assert next_state == 6

    env.state = 4
    reward, _ = env.transition(action=1)
    assert reward == 0.0
