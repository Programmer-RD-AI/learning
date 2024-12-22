from reducto import Reducer
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target
reducer = Reducer(method="pca", n_components=2)
X_reduced = reducer.fit_transform(X)
print("Reduced Data (PCA):", X_reduced)

reducer = Reducer(method="tsne", n_components=2)
X_reduced = reducer.fit_transform(X)
print("Reduced Data (t-SNE):", X_reduced)
reducer = Reducer(method="umap", n_components=2)
X_reduced = reducer.fit_transform(X)
print("Reduced Data (UMAP):", X_reduced)
import matplotlib.pyplot as plt

# Plot the reduced data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="viridis")
plt.colorbar()
plt.title("Dimensionality Reduction Visualization")
plt.show()
