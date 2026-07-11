from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


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
    gradients = torch.stack(gradient_vectors)

    # normalize gradients
    gradients = F.normalize(gradients, p=2, dim=1)

    # covariance / cosine similarity matrix
    covariance = gradients @ gradients.T

    return covariance.detach().cpu()


def sort_by_kmeans(covariance, num_clusters=10):

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=0,
    )

    labels = kmeans.fit_predict(covariance)

    ordering = labels.argsort()

    covariance_clustered = covariance[ordering][:, ordering]

    return covariance_clustered


def plot_gradient_covariance(curves, iter, seed, batch_size=None, num_clusters=None, out_dir="."):
    """Plot one or more named gradient-covariance heatmaps side by side.

    ``curves`` is a sequence of ``(covariance, label)`` tuples -- e.g. one
    for the gradient-descent trajectory and one for the Brownian-motion
    trajectory (Lyle et al. Figure 2 compares exactly these two, reordered
    by KMeans clustering so same-cluster inputs sit adjacent).
    """
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 5))
    if len(curves) == 1:
        axes = [axes]
    fig.suptitle(f'Gradient Covariance (Iteration {iter}, Seed {seed})', fontsize=11)

    subtitle_bits = []
    if batch_size is not None:
        subtitle_bits.append(f"batch k={batch_size}")
    if num_clusters is not None:
        subtitle_bits.append(f"kmeans clusters={num_clusters}")
    subtitle = ", ".join(subtitle_bits)

    for ax, (covariance, label) in zip(axes, curves):
        im = ax.imshow(covariance, cmap="coolwarm_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f"Input index ({subtitle})" if subtitle else "Input index", fontsize=8)
        ax.set_ylabel("Input index", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'gradient_covariance_{iter}_{seed}.png', dpi=300)


if __name__ == "__main__":
    # Here we do a simple test to check for the expected block structure of the covariance matrix on a simple classification task.
    # we train a simple 2-layer MLP on MNIST for a few epochs, and then compute the covariance matrix on a batch of samples from the training set.
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from tqdm import tqdm


    transform = transforms.ToTensor()
    mnist = MNIST(root="data", train=True, download=True, transform=transform)
    train_subset = torch.utils.data.Subset(mnist, range(50000))
    test_subset = torch.utils.data.Subset(mnist, range(10000))
    dataloader = DataLoader(train_subset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=512, shuffle=False)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    EPOCHS = 10
    # we log the loss and accuracy history just for sanity checking, but it's not strictly necessary for the covariance analysis
    loss_history = []
    accuracy_history = []
    for epoch in range(EPOCHS):
        for x, y in tqdm(dataloader):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()
            accuracy_history.append(accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
# compute covariance matrix on a batch of samples
    inputs, targets = next(iter(test_dataloader))
    covariance = compute_gradient_covariance(model, loss_fn, inputs, targets)

    covariance_clustered = sort_by_kmeans(covariance, num_clusters=10)

    plt.figure(figsize=(8, 8))
    plt.imshow(covariance_clustered, cmap="coolwarm_r", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Gradient Covariance Matrix (Clustered)")
    plt.show()
