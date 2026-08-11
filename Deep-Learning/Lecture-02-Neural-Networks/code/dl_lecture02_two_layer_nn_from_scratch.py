"""
DL Lecture 02 - A 2-Layer Neural Network Built From Scratch (NumPy only)
==========================================================================
What this file does (in plain words):
  It builds the EXACT network from the theory file - f = W2 * max(0, W1x+b1) + b2 -
  using nothing but NumPy arrays, trains it with plain gradient descent on a tiny
  toy dataset, and prints the loss going down step by step, so you can SEE
  gradient descent working with your own eyes.

Library used: NumPy only (no PyTorch/TensorFlow) - deliberately, so every single
line of math is visible and traceable back to the theory/numerical README files.
Install: pip install numpy --break-system-packages   (if not already installed)
"""

import numpy as np

# ---------------------------------------------------------------------------
# STEP 1: Make results repeatable
# ---------------------------------------------------------------------------
np.random.seed(1)    # using a fixed "seed" means random numbers are the same
                      # every time you run this file - makes debugging sane.
                      # (Try changing this number - you'll sometimes see the
                      # network get "stuck" with a different seed! That's a
                      # real phenomenon called a local minimum - the loss
                      # stops improving even though a better solution exists.)


# ---------------------------------------------------------------------------
# STEP 2: Create a tiny toy dataset (so we don't need to download anything)
# ---------------------------------------------------------------------------
# 4 training examples, each with 2 input features (D=2).
# This is the classic XOR problem - a famous example that a LINEAR classifier
# (Lecture 2's "f = Wx") can NEVER solve, but a 2-layer network CAN, because
# of the non-linearity discussion in the theory file.
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])                          # shape: (4 examples, 2 features)

y = np.array([[0.0], [1.0], [1.0], [0.0]])   # XOR truth table, shape: (4, 1)


# ---------------------------------------------------------------------------
# STEP 3: Initialise weights randomly (Step 1 of the 5-beat training loop)
# ---------------------------------------------------------------------------
D = 2   # input features
H = 4   # hidden units (our design choice - try changing this number!)
C = 1   # output units (1 = a single "probability of class 1" output)

W1 = np.random.randn(D, H) * 1.0   # shape (2,4) - note: here we store W1 as
                                     # (D,H) instead of (H,D) so we can write
                                     # X @ W1 directly without transposing;
                                     # mathematically equivalent to the
                                     # theory file's (H,D) convention, just a
                                     # common NumPy/PyTorch code layout choice.
b1 = np.zeros((1, H))               # shape (1,4) - one bias per hidden unit
W2 = np.random.randn(H, C) * 1.0    # shape (4,1)
b2 = np.zeros((1, C))               # shape (1,1) - one bias for the output


# ---------------------------------------------------------------------------
# STEP 4: Define the activation functions used in the theory file
# ---------------------------------------------------------------------------
def relu(z):
    """max(0, z), applied elementwise. Negative numbers become exactly 0."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Needed for gradient descent: the 'slope' of ReLU is 1 where z>0, else 0."""
    return (z > 0).astype(float)


def sigmoid(z):
    """Squashes any real number into the range (0, 1) - good for a final
    'probability' output, exactly like Worked Example 3 in the numerical file."""
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# STEP 5: Forward pass - this IS "f = W2 max(0, W1x+b1) + b2" from theory,
#          just written as code, and processing all 4 examples at once.
# ---------------------------------------------------------------------------
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1        # "W1x + b1" for every example at once
    h1 = relu(z1)            # "max(0, ...)" -> the hidden layer's activations
    z2 = h1 @ W2 + b2        # "W2 * (hidden) + b2"
    y_hat = sigmoid(z2)      # final squashing into a 0-1 probability
    return z1, h1, z2, y_hat


# ---------------------------------------------------------------------------
# STEP 6: Training loop - this IS the 5-beat rhythm from theory, repeated
#          many times: Present -> Forward -> Compare -> Adjust
# ---------------------------------------------------------------------------
learning_rate = 0.8
num_steps = 5000

for step in range(num_steps):
    # --- Present + Forward ---
    z1, h1, z2, y_hat = forward(X, W1, b1, W2, b2)

    # --- Compare: compute the loss (how wrong are we, on average?) ---
    # Mean Squared Error, a simple loss: average of (true - predicted)^2
    loss = np.mean((y - y_hat) ** 2)

    # --- Adjust: compute gradients (calculus - how much did each weight
    #     contribute to the error?) and nudge every weight a tiny bit.
    #     This block is "Gradient Descent" written out by hand.
    d_loss_d_yhat = -2 * (y - y_hat) / y.shape[0]         # derivative of MSE
    d_yhat_d_z2 = y_hat * (1 - y_hat)                      # derivative of sigmoid
    d_loss_d_z2 = d_loss_d_yhat * d_yhat_d_z2               # chain rule

    d_loss_d_W2 = h1.T @ d_loss_d_z2                        # gradient for W2
    d_loss_d_b2 = np.sum(d_loss_d_z2, axis=0, keepdims=True)

    d_loss_d_h1 = d_loss_d_z2 @ W2.T                        # push error back to hidden layer
    d_loss_d_z1 = d_loss_d_h1 * relu_derivative(z1)          # chain rule through ReLU

    d_loss_d_W1 = X.T @ d_loss_d_z1                          # gradient for W1
    d_loss_d_b1 = np.sum(d_loss_d_z1, axis=0, keepdims=True)

    # The actual "nudge the weights" step - move OPPOSITE the gradient,
    # scaled by the learning rate (exactly the formula in the numerical file).
    W1 -= learning_rate * d_loss_d_W1
    b1 -= learning_rate * d_loss_d_b1
    W2 -= learning_rate * d_loss_d_W2
    b2 -= learning_rate * d_loss_d_b2

    # Print progress every 1000 steps so we can watch the loss shrink
    if step % 1000 == 0:
        print(f"Step {step:4d} | Loss = {loss:.6f}")


# ---------------------------------------------------------------------------
# STEP 7: Check the final predictions vs the true XOR answers
# ---------------------------------------------------------------------------
print("\nFinal predictions after training:")
_, _, _, final_predictions = forward(X, W1, b1, W2, b2)
for inputs, true_val, pred_val in zip(X, y, final_predictions):
    print(f"  input={inputs}  true={true_val[0]:.0f}  predicted={pred_val[0]:.3f}")

print("\nA LINEAR classifier (f=Wx) can NEVER solve XOR perfectly - this 2-layer")
print("network can, because of the non-linear ReLU activation in the hidden layer.")
print("This is the exact 'ring around center' idea from the theory file, in code form.")
