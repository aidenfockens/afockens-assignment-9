import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
import uuid

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

        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        

    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    ax_hidden.set_title("Hidden Features")

    # Decision boundary in input space
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k', alpha=0.7)
    ax_input.set_title("Decision Boundary in Input Space")

    # Gradients visualization
    grads = np.linalg.norm(mlp.W1, axis=1)
    for i, grad in enumerate(grads):
        ax_gradient.plot([0, grad], [0, grad], linewidth=grad, color='blue', alpha=0.7)
    ax_gradient.set_title("Gradient Visualization")


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
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)



    new_filename = f"visualize_{uuid.uuid4().hex}.gif"
    new_filepath = os.path.join(result_dir, new_filename)
    # Save the animation as a GIF
    ani.save(new_filepath, writer='pillow', fps=10)
    plt.close()

    # Remove old gifs in the result directory
    for file in os.listdir(result_dir):
        if file != new_filename and file.endswith(".gif"):
            os.remove(os.path.join(result_dir, file))
    plt.close()
    return new_filename

#if __name__ == "__main__":
#    activation = "tanh"
#    lr = 0.1
#    step_num = 1000
#    visualize(activation, lr, step_num)