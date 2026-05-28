import torch
import torch.nn.functional as F


def flatten_gradients(model):
    """
    Flatten all parameter gradients into one vector.
    """

    grads = []

    for param in model.parameters():

        if param.grad is None:
            continue

        grads.append(param.grad.view(-1))

    return torch.cat(grads)


def compute_gradient_covariance(
    model,
    loss_fn,
    inputs,
    targets,
):
    """
    Computes normalized gradient covariance matrix.

    Returns:
        C: [batch_size, batch_size]
    """

    device = inputs.device

    batch_size = inputs.shape[0]

    gradient_vectors = []

    # compute gradient for each sample individually
    for i in range(batch_size):

        model.zero_grad()

        x = inputs[i].unsqueeze(0)
        y = targets[i].unsqueeze(0)

        output = model(x)

        loss = loss_fn(output, y)

        loss.backward()

        grad_vector = flatten_gradients(model)

        gradient_vectors.append(grad_vector)

    # shape:
    # [batch_size, num_params]
    G = torch.stack(gradient_vectors)

    # normalize gradients
    G = F.normalize(G, p=2, dim=1)

    # covariance / cosine similarity matrix
    C = G @ G.T

    return C.detach().cpu()


C = compute_gradient_covariance(

    model,

    loss_fn,

    inputs,

    targets,

)

C = cluster_covariance_matrix(C)

sns.heatmap(

    C.numpy(),

    cmap="coolwarm",

    center=0,

)