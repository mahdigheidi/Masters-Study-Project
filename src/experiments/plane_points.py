from torch.nn.utils import parameters_to_vector
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(7)
torch.set_num_threads(1)

# -----------------------------
# 1. Dataset: quadrant classes
# -----------------------------


def make_data(n=2000, lim=5.0):
    X = (torch.rand(n, 2) * 2 - 1) * lim
    y = torch.empty(n, dtype=torch.long)

    y[(X[:, 0] >= 0) & (X[:, 1] >= 0)] = 0  # Q1
    y[(X[:, 0] < 0) & (X[:, 1] >= 0)] = 1  # Q2
    y[(X[:, 0] < 0) & (X[:, 1] < 0)] = 2  # Q3
    y[(X[:, 0] >= 0) & (X[:, 1] < 0)] = 3  # Q4

    return X, y


X, y = make_data()

# -----------------------------
# 2. MLP: 2 -> 7 -> 7 -> 4
# -----------------------------


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 7)
        self.fc2 = nn.Linear(7, 7)
        self.fc3 = nn.Linear(7, 4)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc3(h2)


model = MLP()

# Total parameters:
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params)  # 109

# -----------------------------
# 3. Analytical forward pass
# -----------------------------
x0 = torch.tensor([[1.5, -2.0]])

with torch.no_grad():
    z1 = model.fc1(x0)
    h1 = torch.tanh(z1)
    z2 = model.fc2(h1)
    h2 = torch.tanh(z2)
    logits = model.fc3(h2)
    probs = F.softmax(logits, dim=1)

print("z1:", z1)
print("h1:", h1)
print("z2:", z2)
print("h2:", h2)
print("logits:", logits)
print("probs:", probs)
print("predicted class:", probs.argmax(dim=1).item() + 1)

# -----------------------------
# 4. Plot samples
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="tab10")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.title("Plane samples by quadrant")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -----------------------------
# 5. Train network
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

loss_log = []
acc_log = []

for epoch in range(80):
    perm = torch.randperm(len(X))

    for i in range(0, len(X), 128):
        xb = X[perm[i:i+128]]
        yb = y[perm[i:i+128]]

        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

    loss_log.append(loss.item())
    acc_log.append(acc.item())

print("Final loss:", loss_log[-1])
print("Final accuracy:", acc_log[-1])

plt.figure()
plt.plot(loss_log)
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.show()

plt.figure()
plt.plot(acc_log)
plt.title("Training accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# -----------------------------
# 6. Hessian matrix
# -----------------------------

params = list(model.parameters())
theta0 = parameters_to_vector(params).detach()

shapes = [p.shape for p in params]
numels = [p.numel() for p in params]
names = [name for name, _ in model.named_parameters()]


def unpack(theta):
    out = {}
    idx = 0
    for name, shape, n in zip(names, shapes, numels):
        out[name] = theta[idx:idx+n].view(shape)
        idx += n
    return out


def forward_from_theta(theta, x):
    p = unpack(theta)

    h1 = torch.tanh(x @ p["fc1.weight"].T + p["fc1.bias"])
    h2 = torch.tanh(h1 @ p["fc2.weight"].T + p["fc2.bias"])
    z = h2 @ p["fc3.weight"].T + p["fc3.bias"]

    return z


# Use subset to keep Hessian computation fast
X_hess = X[:200]
y_hess = y[:200]


def loss_theta(theta):
    logits = forward_from_theta(theta, X_hess)
    return F.cross_entropy(logits, y_hess)


theta = theta0.clone().requires_grad_(True)

H = torch.autograd.functional.hessian(loss_theta, theta)
H = H.detach()

print("Hessian shape:", H.shape)
print("Hessian matrix:")
print(H)

eigvals = torch.linalg.eigvalsh(H)

print("Eigenvalues:")
print(eigvals)

print("Smallest eigenvalue:", eigvals.min().item())
print("Largest eigenvalue:", eigvals.max().item())

plt.figure()
plt.hist(eigvals.numpy(), bins=40)
plt.title("Hessian eigenvalue density")
plt.xlabel("Eigenvalue")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 7. Gradient covariance matrix
# -----------------------------


def grad_vector_for_sample(xi, yi):
    loss_i = F.cross_entropy(model(xi.unsqueeze(0)), yi.unsqueeze(0))
    grads = torch.autograd.grad(loss_i, model.parameters())
    return torch.cat([g.reshape(-1) for g in grads]).detach()


k = 128
G = torch.stack([
    grad_vector_for_sample(X[i], y[i])
    for i in range(k)
])

G_norm = G / (G.norm(dim=1, keepdim=True) + 1e-12)

C = G_norm @ G_norm.T

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=20,
    random_state=0,
)

labels = kmeans.fit_predict(C)

ordering = labels.argsort()

C_clustered = C[ordering][:, ordering]

print("Gradient covariance shape:", C.shape)
print("Gradient covariance matrix:")
print(C_clustered)

plt.figure(figsize=(6, 5))
plt.imshow(C_clustered.numpy(), cmap="coolwarm_r", vmin=-5, vmax=5)
plt.colorbar()
plt.title("Gradient covariance matrix")
plt.xlabel("sample index")
plt.ylabel("sample index")
plt.show()
