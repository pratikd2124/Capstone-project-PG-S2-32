import torch

from scrp_toolkit.model import KL, Net


def test_net_forward_shape():
    net = Net(n_features=10, n_layers=3, width=16)
    x = torch.randn(5, 10)
    out = net(x)
    assert out.shape == (5, 2)


def test_kl_loss_is_zero_when_prediction_equals_target():
    shape_rate_min_max = {"shape_min": 0.0, "shape_max": 10.0, "rate_min": 0.0, "rate_max": 10.0}
    # normalised output == normalised target -> predicted Gamma == true Gamma -> KL divergence 0
    same = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
    loss = KL(shape_rate_min_max, same, same)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)


def test_kl_loss_positive_when_prediction_differs():
    shape_rate_min_max = {"shape_min": 0.0, "shape_max": 10.0, "rate_min": 0.0, "rate_max": 10.0}
    pred = torch.tensor([[0.2, 0.8]])
    target = torch.tensor([[0.6, 0.4]])
    loss = KL(shape_rate_min_max, pred, target)
    assert loss.item() > 0
