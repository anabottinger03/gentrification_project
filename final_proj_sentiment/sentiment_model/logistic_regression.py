import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import issparse


def sigmoid(z):
    """sigmoid func

    :param z: dot product of optimal theta and our X matrix
    :return: sigmoid
    """
    return 1 / (1 + np.exp(-z))


def logistic_fit(X, y, learning_rate=0.001, strength=0.2, num_iterations=10000, regularization="L1"):
    """
    Logistic regression with regularization and sparse matrix handling
    """
    # Convert to dense if sparse
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Convert labels to proper shape
    y = np.array(y).reshape(-1, 1)

    m, n = X.shape
    theta = np.zeros((n, 1))

    losses = []
    for i in range(num_iterations):
        prev_loss = None
        z = X.dot(theta)
        h = sigmoid(z)
        error = h - y
        gradient = X.T.dot(error)

        if regularization == "L1":
            gradient += strength * np.sign(theta)
        elif regularization == "L2":
            gradient += strength * theta
        elif regularization == "Elastic Net":
            gradient += strength * (0.5 * np.sign(theta) + 0.5 * theta)
        elif regularization == "None":
            pass
        else:
            raise ValueError(f"Unknown regularization type: {regularization}")

        theta -= learning_rate * gradient
        
        if i % 100 == 0 or i == num_iterations - 1:
            loss = compute_loss(h, y)
            losses.append(loss)
            print(f"Iteration {i} - Loss: {loss:.5f}")

            # Early stopping if loss change is small
            if prev_loss is not None and abs(prev_loss - loss) < 1e-6:
                print(f"Stopping early at iteration {i} (loss converged)")
                break
            prev_loss = loss

    plt.plot(range(0, len(losses)*100, 100), losses)
    plt.title("Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"results/figs/logregress_loss_{learning_rate}_{strength}_{num_iterations}_{regularization}.png")

    return prev_loss, theta


def compute_loss(h, y):
    epsilon = 1e-5  # avoid log(0)
    return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))


def log_predict(X, theta):
    """predicts label

    :param X: nxn array of data
    :param theta: nx1 array of theta values from training
    :return: nx1 array of 0 or 1 classification values
    """

    # Use sparse dot product if applicable
    z = X.dot(theta) if issparse(X) else X.dot(X, theta)

    # Make sure z is dense for sigmoid
    z = z.toarray() if hasattr(z, "toarray") else z

    h = sigmoid(z)
    return (h >= 0.5).astype(int)



