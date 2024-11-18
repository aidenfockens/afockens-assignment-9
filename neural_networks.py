import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
import uuid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
        # To store intermediate activations
        self.a1, self.a2, self.z1, self.z2 = None, None, None, None


    def activate(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x), 1 - np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return np.maximum(0, x), (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig, sig * (1 - sig)

    def forward(self, X):
        # Input to hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1, self.grad_a1 = self.activate(self.z1)

        # Hidden to output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid for binary classification
        
        return self.a2

    def backward(self, X, y):
        # Output layer gradient
        dz2 = self.a2 - y  # Binary cross-entropy gradient
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.grad_a1
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        print(f"dz2 mean: {np.mean(dz2)}, dW2 mean: {np.mean(dW2)}")
        print(f"dz1 mean: {np.mean(dz1)}, dW1 mean: {np.mean(dW1)}")
        # Update weights and biases
        print(f"Before Update: W1 mean = {np.mean(self.W1)}")
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        print(f"After Update: W1 mean = {np.mean(self.W1 - self.lr * dW1)}")
        print(f"grad_a1 mean: {np.mean(self.grad_a1)}")

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps (10 per frame)
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden layer features
    hidden_features = mlp.a1

    # Plot 3D Hidden Features with Moving Lines
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )
    for i in range(X.shape[0]):
        ax_hidden.plot(
            [X[i, 0], hidden_features[i, 0]],
            [X[i, 1], hidden_features[i, 1]],
            [0, hidden_features[i, 2]],
            color='gray', alpha=0.5
        )
    ax_hidden.set_title("Hidden Features")

    # Decision Boundary in Input Space with Step Number
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k', alpha=0.7)
    ax_input.set_title(f"Decision Boundary in Input Space (Step {frame * 10})")

    # Update Classification Lines
    for i in range(mlp.W1.shape[1]):  # Loop over each hidden neuron
        weight = mlp.W1[:, i]
        norm = np.linalg.norm(weight)
        if norm > 0:
            weight /= norm  # Normalize for consistent visualization
            ax_input.plot(
                [0, weight[0]],
                [0, weight[1]],
                color='green', alpha=0.5, linestyle='--'
            )
    print(f"Step {frame * 10}: W1 mean = {np.mean(mlp.W1)}, W2 mean = {np.mean(mlp.W2)}")
    # Gradients Visualization
    grads = np.linalg.norm(mlp.W1, axis=0)
    ax_gradient.bar(range(len(grads)), grads, color='blue', alpha=0.7)
    ax_gradient.set_title("Gradient Magnitudes")



def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=step_num // 10,
        repeat=True
    )

    # Generate a unique filename for the GIF
    new_filename = f"visualize_{uuid.uuid4().hex}.gif"
    new_filepath = os.path.join(result_dir, new_filename)

    # Save the animation as a GIF
    ani.save(new_filepath, writer='pillow', fps=10)

    # Clean up old files
    for file in os.listdir(result_dir):
        if file != new_filename and file.endswith(".gif"):
            os.remove(os.path.join(result_dir, file))

    plt.close()
    return new_filename

