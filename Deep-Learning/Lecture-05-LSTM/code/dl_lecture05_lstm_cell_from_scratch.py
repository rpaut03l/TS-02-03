"""
DL Lecture 05 - An LSTM Cell Built From Scratch (NumPy only)
===============================================================
What this file does (in plain words):
  Implements all six LSTM equations from the theory file - forget gate,
  input gate, candidate memory, cell state update, output gate, hidden
  state - using only NumPy, and checks the results against the exact
  worked example in numerical/dl_lecture05_lstm_numerical.md (Worked
  Example 2), plus reproduces the forget-gate extremes and the
  RNN-vs-LSTM gradient decay comparison.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


def sigmoid(z):
    """Squashes to (0,1) - used for all three gates ('how much to let through')."""
    return 1.0 / (1.0 + np.exp(-z))


class LSTMCell:
    """
    One LSTM cell, implementing the six equations from the theory file:
        f_t = sigmoid(W_f . [h_{t-1}, x_t] + b_f)
        i_t = sigmoid(W_i . [h_{t-1}, x_t] + b_i)
        C~_t = tanh(W_C . [h_{t-1}, x_t] + b_C)
        C_t = f_t * C_{t-1} + i_t * C~_t
        o_t = sigmoid(W_o . [h_{t-1}, x_t] + b_o)
        h_t = o_t * tanh(C_t)
    """

    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        combined_size = hidden_size + input_size   # [h_{t-1}, x_t] concatenated

        # One weight matrix + bias per gate/component - FOUR total, per theory file
        self.W_f = rng.normal(0, 0.5, (hidden_size, combined_size))
        self.b_f = np.zeros(hidden_size)

        self.W_i = rng.normal(0, 0.5, (hidden_size, combined_size))
        self.b_i = np.zeros(hidden_size)

        self.W_c = rng.normal(0, 0.5, (hidden_size, combined_size))
        self.b_c = np.zeros(hidden_size)

        self.W_o = rng.normal(0, 0.5, (hidden_size, combined_size))
        self.b_o = np.zeros(hidden_size)

        self.hidden_size = hidden_size
        self.input_size = input_size

    def step(self, x_t, h_prev, C_prev):
        """Runs ONE timestep of the LSTM recurrence. Returns (h_t, C_t, gates_dict)."""
        combined = np.concatenate([h_prev, x_t])   # [h_{t-1}, x_t]

        f_t = sigmoid(self.W_f @ combined + self.b_f)          # forget gate
        i_t = sigmoid(self.W_i @ combined + self.b_i)          # input gate
        C_tilde_t = np.tanh(self.W_c @ combined + self.b_c)    # candidate memory
        C_t = f_t * C_prev + i_t * C_tilde_t                   # cell state update
        o_t = sigmoid(self.W_o @ combined + self.b_o)          # output gate
        h_t = o_t * np.tanh(C_t)                               # hidden state

        gates = {"forget": f_t, "input": i_t, "candidate": C_tilde_t, "output": o_t}
        return h_t, C_t, gates

    def param_count(self):
        """Counts total learnable parameters (weights + biases) across all 4 gates."""
        weight_params = 4 * self.hidden_size * (self.hidden_size + self.input_size)
        bias_params = 4 * self.hidden_size
        return weight_params + bias_params


# ---------------------------------------------------------------------------
# Check 1: Reproduce the numerical file's Worked Example 2 EXACTLY
#          (using the SAME hand-picked weights, not random ones)
# ---------------------------------------------------------------------------
def check_worked_example_2():
    print("=" * 65)
    print("Check 1: Worked Example 2 - one LSTM timestep, hand-picked weights")
    print("=" * 65)

    H_prev = np.array([0.2])
    C_prev = np.array([0.5])
    x_t = np.array([1.0])

    f_t = sigmoid(np.array([0.5]) * H_prev + np.array([0.3]) * x_t)
    i_t = sigmoid(np.array([0.6]) * H_prev + np.array([0.4]) * x_t)
    C_tilde = np.tanh(np.array([0.4]) * H_prev + np.array([0.5]) * x_t)
    C_t = f_t * C_prev + i_t * C_tilde
    o_t = sigmoid(np.array([0.3]) * H_prev + np.array([0.6]) * x_t)
    H_t = o_t * np.tanh(C_t)

    print(f"f_t = {f_t[0]:.4f}  (expected ~0.5987)")
    print(f"i_t = {i_t[0]:.4f}  (expected ~0.6271)")
    print(f"C~_t = {C_tilde[0]:.4f}  (expected ~0.5227)")
    print(f"C_t = {C_t[0]:.4f}  (expected ~0.6272)")
    print(f"o_t = {o_t[0]:.4f}  (expected ~0.6593)")
    print(f"H_t = {H_t[0]:.4f}  (expected ~0.3667)")


# ---------------------------------------------------------------------------
# Check 2: Parameter count comparison (LSTM vs vanilla RNN)
# ---------------------------------------------------------------------------
def check_parameter_count():
    print()
    print("=" * 65)
    print("Check 2: LSTM vs vanilla RNN parameter count (D=10, H=20)")
    print("=" * 65)
    D, H = 10, 20
    cell = LSTMCell(input_size=D, hidden_size=H)
    lstm_params = cell.param_count()

    rnn_params = H * D + H * H   # U + W, from Lecture 4's formula
    print(f"Vanilla RNN params (U+W only): {rnn_params}")
    print(f"LSTM params (4 gates + biases): {lstm_params}")
    print(f"Ratio (weights only, ignoring bias): {(4*H*(H+D)) / rnn_params:.2f}x  (expected 4.00x)")


# ---------------------------------------------------------------------------
# Check 3: Run a real forward pass across a short random sequence
# ---------------------------------------------------------------------------
def run_forward_pass_demo():
    print()
    print("=" * 65)
    print("Check 3: Forward pass across a 5-step toy sequence (random weights)")
    print("=" * 65)
    D, H = 3, 4
    cell = LSTMCell(input_size=D, hidden_size=H, seed=42)

    h_t = np.zeros(H)   # starts empty, per theory file
    C_t = np.zeros(H)
    rng = np.random.default_rng(1)

    for t in range(5):
        x_t = rng.normal(0, 1, D)
        h_t, C_t, gates = cell.step(x_t, h_t, C_t)
        print(f"t={t} | forget_gate_mean={gates['forget'].mean():.3f} | "
              f"input_gate_mean={gates['input'].mean():.3f} | "
              f"cell_state={np.round(C_t, 3)}")


# ---------------------------------------------------------------------------
# Check 4: Gradient decay comparison - RNN (factor 0.5) vs LSTM (factor 0.98)
# ---------------------------------------------------------------------------
def check_gradient_decay():
    print()
    print("=" * 65)
    print("Check 4: Gradient decay - vanilla RNN vs LSTM forget-gate pathway")
    print("=" * 65)
    for n in [5, 10, 20, 50]:
        rnn_decay = 0.5 ** n
        lstm_decay = 0.98 ** n
        print(f"  n={n:2d} | vanilla RNN (0.5^n) = {rnn_decay:.8f} | "
              f"LSTM (0.98^n) = {lstm_decay:.4f}")


if __name__ == "__main__":
    check_worked_example_2()
    check_parameter_count()
    run_forward_pass_demo()
    check_gradient_decay()
