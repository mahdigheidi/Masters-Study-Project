import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(0)

# -------------------------
# 1. Define MLP model
# -------------------------


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 4)
        )

    def forward(self, x):
        return self.net(x)


model = MLP()

# -------------------------
# 2. Generate 128 samples and quadrant labels
# -------------------------

N = 128
X = torch.randn(N, 2)   # 128 samples, each 2-dimensional

# Labels based on quadrant:
# class 0: x >= 0, y >= 0
# class 1: x <  0, y >= 0
# class 2: x <  0, y <  0
# class 3: x >= 0, y <  0
labels = torch.zeros(N, dtype=torch.long)
labels[(X[:, 0] < 0) & (X[:, 1] >= 0)] = 1
labels[(X[:, 0] < 0) & (X[:, 1] < 0)] = 2
labels[(X[:, 0] >= 0) & (X[:, 1] < 0)] = 3

# Count total number of parameters
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params)


# -------------------------
# 3. Helper function: compute normalized gradient similarity matrix
# -------------------------

def compute_gradient_similarity_matrix(model, X):
    gradients = torch.zeros(N, num_params)

    for i in range(N):
        x_i = X[i].unsqueeze(0)  # shape: [1, 2]

        # Clear old gradients
        model.zero_grad()

        # Forward pass
        y = model(x_i)  # shape: [1, 4]

        # Sample epsilon from N(0, 1)
        eps = torch.randn_like(y)  # shape: [1, 4]

        # Loss:
        # ℓ(θ) = [fθ(X) - stopgrad(fθ(X)) + ε]^2
        loss_vector = (y - y.detach() + eps) ** 2

        # Make scalar loss
        loss = loss_vector.sum()

        # Backward pass
        loss.backward()

        # Save gradients as one long vector
        grad_list = []

        for p in model.parameters():
            grad_list.append(p.grad.reshape(-1))

        grad_vector = torch.cat(grad_list)
        gradients[i] = grad_vector.detach()

    # Normalize gradient vectors
    grad_norms = gradients.norm(dim=1, keepdim=True)
    grad_norms = torch.clamp(grad_norms, min=1e-12)
    gradients_normalized = gradients / grad_norms

    # Sample-by-sample cosine similarity matrix
    similarity_matrix = gradients_normalized @ gradients_normalized.T

    return similarity_matrix


# -------------------------
# 4. Helper function: cluster and plot heatmap
# -------------------------

def plot_clustered_heatmap(similarity_matrix, title):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
    cluster_labels = kmeans.fit_predict(similarity_matrix.detach().numpy())

    ordering = cluster_labels.argsort()
    C_clustered = similarity_matrix[ordering][:, ordering]

    plt.figure(figsize=(8, 8))

    plt.imshow(
        C_clustered.numpy(),
        cmap="coolwarm_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )

    plt.colorbar(label="Cosine Similarity")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()


# -------------------------
# 5. Gradient matrix before training
# -------------------------

similarity_before = compute_gradient_similarity_matrix(model, X)
print("Similarity matrix before training:", similarity_before.shape)

plot_clustered_heatmap(
    similarity_before,
    "Normalized Gradient Similarity Matrix Before Training"
)


# -------------------------
# 6. Train model on quadrant classification
# -------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
num_epochs = 50

for epoch in range(num_epochs):
    optimizer.zero_grad()

    logits = model(X)              # shape: [128, 4]
    train_loss = criterion(logits, labels)

    train_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {train_loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.4f}"
        )


# -------------------------
# 7. Gradient matrix after training
# -------------------------

similarity_after = compute_gradient_similarity_matrix(model, X)
print("Similarity matrix after training:", similarity_after.shape)

plot_clustered_heatmap(
    similarity_after,
    "Normalized Gradient Similarity Matrix After Training"
)

