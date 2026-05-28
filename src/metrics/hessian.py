import torch
from torch.autograd.functional import hessian


def compute_hessian(loss_fn, params):
    return hessian(loss_fn, params)

from pyhessian import hessian

hessian_comp = hessian(
    model,
    criterion,
    data=(inputs, targets),
)

eigenvalues, eigenvectors = hessian_comp.eigenvalues()