import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import issparse

def hinge_loss(X, y, w):
    y = 2 * y - 1  # convert to -1/+1
    margin = 1 - y * X.dot(w)
    if issparse(margin):
        margin = margin.toarray().reshape(-1)
    return np.mean(np.maximum(0, margin))

def train_svm(X, y, learning_rate=0.01, strength=0.1, iters=1000, regularization="L2"):
    m, n = X.shape
    y = np.array(y).reshape(-1)
    y = 2 * y - 1  # convert 0/1 to -1/+1

    w = np.random.randn(n) * 0.01
    losses = []
    prev_loss = None

    for i in range(iters):
        scores = X.dot(w)
        if issparse(scores):
            scores = scores.toarray().reshape(-1)

        margin = 1 - y * scores
        indicator = (margin > 0).astype(float)

        grad = -X.T.dot(indicator * y) / m  # hinge loss gradient

        # Add regularization
        if regularization == "L1":
            grad += strength * np.sign(w)
        elif regularization == "L2":
            grad += strength * w
        elif regularization == "Elastic Net":
            grad += strength * (0.5 * w + 0.5 * np.sign(w))
        elif regularization == "None":
            pass
        else:
            raise ValueError(f"Unknown regularization type: {regularization}")

        w -= learning_rate * grad

        # Track loss + early stopping
        if i % 1000 == 0 or i == iters - 1:
            loss = hinge_loss(X, y, w)
            losses.append(loss)
            print(f"Iteration {i} - Hinge Loss: {loss:.5f}")
            prev_loss = loss

    # Plotting loss
    plt.plot(range(0, len(losses) * 100, 100), losses)
    plt.title("SVM Hinge Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/figs/svm_hinge_loss_{learning_rate}_{strength}_{iters}_{regularization}.png")
    plt.show()

    return w, prev_loss

def predict_svm(X, w):
    z = X.dot(w)
    if issparse(z):
        z = z.toarray().reshape(-1)
    predictions = np.sign(z)
    return (predictions > 0).astype(int)
