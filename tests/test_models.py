import torch

from src.models.cnn import CNN
from src.models.mlp import MLP
from src.models.vit import VisionTransformer


def test_mlp_forward_shape_and_feature_dim():
    model = MLP(input_shape=(1, 28, 28), num_actions=10, hidden_dim=32)
    x = torch.randn(5, 1, 28, 28)

    logits = model(x)
    features = model.forward_features(x)

    assert logits.shape == (5, 10)
    assert features.shape == (5, 32)
    assert model.feature_dim == 32


def test_mlp_reset_last_layer_changes_weights():
    torch.manual_seed(0)
    model = MLP(input_shape=(1, 28, 28), num_actions=10, hidden_dim=16)
    before = model.last_layer.weight.clone()

    model.reset_last_layer()

    assert not torch.equal(before, model.last_layer.weight)


def test_cnn_forward_shape_for_mnist_and_cifar_input():
    mnist_model = CNN(input_shape=(1, 28, 28), num_actions=10, conv_channels=8, fc_dim=16)
    cifar_model = CNN(input_shape=(3, 32, 32), num_actions=10, conv_channels=8, fc_dim=16)

    mnist_logits = mnist_model(torch.randn(4, 1, 28, 28))
    cifar_logits = cifar_model(torch.randn(4, 3, 32, 32))

    assert mnist_logits.shape == (4, 10)
    assert cifar_logits.shape == (4, 10)


def test_vision_transformer_forward_shape():
    model = VisionTransformer(
        input_shape=(1, 28, 28),
        patch_size=4,
        num_actions=10,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
    )
    x = torch.randn(3, 1, 28, 28)

    logits = model(x)

    assert logits.shape == (3, 10)


def test_spectral_norm_flag_wraps_linear_layers():
    model = MLP(input_shape=(1, 28, 28), num_actions=10, hidden_dim=8, spectral_norm=True)
    assert hasattr(model.fc1, "weight_orig")
