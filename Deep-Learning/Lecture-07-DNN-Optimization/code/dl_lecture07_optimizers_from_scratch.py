"""
DL Lecture 07 - SGD, Momentum, RMSProp, and Adam Built From Scratch
======================================================================
What this file does (in plain words):
  Implements all four optimizer update rules from the theory file as small
  Python classes, checks each one against the numerical README's hand-worked
  examples, and then runs all four side-by-side on the SAME toy loss function
  so you can literally watch which one converges fastest.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


class SGD:
    """theta = theta - eta * g"""
    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, theta, grad):
        return theta - self.lr * grad


class Momentum:
    """v = gamma*v + eta*g ;  theta = theta - v"""
    def __init__(self, lr=0.1, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.v = 0.0

    def update(self, theta, grad):
        self.v = self.gamma * self.v + self.lr * grad
        return theta - self.v


class RMSProp:
    """E[g^2] = beta*E[g^2] + (1-beta)*g^2 ;  theta = theta - eta/sqrt(E[g^2]+eps) * g"""
    def __init__(self, lr=0.1, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.E_g2 = 0.0

    def update(self, theta, grad):
        self.E_g2 = self.beta * self.E_g2 + (1 - self.beta) * grad ** 2
        step = self.lr / (np.sqrt(self.E_g2) + self.eps) * grad
        return theta - step


class Adam:
    """Combines momentum (m) and RMSProp (v), with bias correction."""
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def update(self, theta, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return theta - step


# ---------------------------------------------------------------------------
# Check 1: Reproduce every numerical README worked example EXACTLY
# ---------------------------------------------------------------------------
def check_worked_examples():
    print("=" * 65)
    print("Check 1: SGD (Worked Example 1)")
    print("=" * 65)
    sgd = SGD(lr=0.1)
    theta = sgd.update(1.0, 4.0)
    print(f"theta = {theta}  (expected 0.6)")

    print()
    print("=" * 65)
    print("Check 2: Momentum, two steps (Worked Example 2)")
    print("=" * 65)
    mom = Momentum(lr=0.1, gamma=0.9)
    theta = 1.0
    for g in [4.0, 3.0]:
        theta = mom.update(theta, g)
        print(f"  v={mom.v:.4f}  theta={theta:.4f}")
    print("(expected: v=0.4,theta=0.6  then  v=0.66,theta=-0.06)")

    print()
    print("=" * 65)
    print("Check 3: RMSProp, two steps (Worked Example 3)")
    print("=" * 65)
    rms = RMSProp(lr=0.1, beta=0.9)
    theta = 1.0
    for g in [4.0, -3.0]:
        theta = rms.update(theta, g)
        print(f"  E[g^2]={rms.E_g2:.4f}  theta={theta:.4f}")
    print("(expected: E=1.6,theta=0.6838  then  E=2.34,theta=0.8799)")

    print()
    print("=" * 65)
    print("Check 4: Adam, two steps (Worked Example 4)")
    print("=" * 65)
    adam = Adam(lr=0.1, beta1=0.9, beta2=0.999)
    theta = 1.0
    for g in [4.0, 3.0]:
        theta = adam.update(theta, g)
        print(f"  m={adam.m:.4f}  v={adam.v:.4f}  theta={theta:.4f}")
    print("(expected: theta=0.9  then  theta=0.8017)")


# ---------------------------------------------------------------------------
# Check 2: Race all four optimizers on a simple toy loss function
#          L(theta) = (theta - 3)^2   ->  minimum is at theta=3, gradient=2*(theta-3)
# ---------------------------------------------------------------------------
def toy_loss_gradient(theta):
    """Gradient of L(theta) = (theta - 3)^2, i.e. dL/dtheta = 2*(theta-3)"""
    return 2 * (theta - 3)


def race_optimizers():
    print()
    print("=" * 65)
    print("Check 5: Racing all 4 optimizers toward the minimum of (theta-3)^2")
    print("=" * 65)
    optimizers = {
        "SGD": SGD(lr=0.1),
        "Momentum": Momentum(lr=0.1, gamma=0.9),
        "RMSProp": RMSProp(lr=0.3),
        "Adam": Adam(lr=0.3),
    }
    thetas = {name: -5.0 for name in optimizers}   # everyone starts far from the minimum (theta=3)

    print(f"{'step':>4s} | " + " | ".join(f"{name:>10s}" for name in optimizers))
    for step in range(15):
        row = [f"{step:4d}"]
        for name, opt in optimizers.items():
            grad = toy_loss_gradient(thetas[name])
            thetas[name] = opt.update(thetas[name], grad)
            row.append(f"{thetas[name]:10.4f}")
        print(" | ".join(row))

    print("\nTrue minimum is at theta=3.0 for all optimizers.")


if __name__ == "__main__":
    check_worked_examples()
    race_optimizers()
