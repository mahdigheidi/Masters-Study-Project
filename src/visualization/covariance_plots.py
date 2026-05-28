import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

sns.heatmap(
    C.numpy(),
    cmap="coolwarm",
    center=0,
)

plt.title("Gradient Covariance Matrix")

plt.show()