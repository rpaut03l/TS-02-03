"""
DL Lecture 13 (Bonus) - Loss Functions, Gradients, and a Mini GAN From Scratch
=================================================================================
What this file does (in plain words):
  Implements the Discriminator loss, both versions of the Generator loss
  (minimax and non-saturating), their gradient magnitudes, the global
  optimum check, a toy FID calculation, and a genuinely trainable (though
  tiny) 1D GAN that learns to generate samples matching a simple target
  distribution - all checked against the numerical README.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


# ---------------------------------------------------------------------------
# PART 1: The core GAN loss functions
# ---------------------------------------------------------------------------
def discriminator_loss(D_real, D_fake):
    """Implements: L_D = -[log(D(x)) + log(1-D(G(z)))]"""
    return -(np.log(D_real) + np.log(1 - D_fake))


def generator_loss_minimax(D_fake):
    """Implements: L_G = log(1 - D(G(z)))  (to be MINIMIZED)"""
    return np.log(1 - D_fake)


def generator_loss_nonsaturating(D_fake):
    """Implements: L_G = -log(D(G(z)))  (to be MINIMIZED)"""
    return -np.log(D_fake)


def check_losses():
    print("=" * 65)
    print("Check 1: Discriminator and Generator losses (Worked Examples 1 & 2)")
    print("=" * 65)
    L_D = discriminator_loss(D_real=0.9, D_fake=0.2)
    print(f"L_D (D(x)=0.9, D(G(z))=0.2): {L_D:.4f}  (expected ~0.3285)")

    L_G_mm = generator_loss_minimax(D_fake=0.2)
    L_G_ns = generator_loss_nonsaturating(D_fake=0.2)
    print(f"L_G minimax:        {L_G_mm:.4f}  (expected ~-0.2231)")
    print(f"L_G non-saturating: {L_G_ns:.4f}  (expected ~1.6094)")


# ---------------------------------------------------------------------------
# PART 2: Gradient magnitude comparison (the vanishing gradient proof)
# ---------------------------------------------------------------------------
def minimax_grad_magnitude(D_fake):
    return 1 / (1 - D_fake)


def nonsaturating_grad_magnitude(D_fake):
    return 1 / D_fake


def check_gradient_comparison():
    print()
    print("=" * 65)
    print("Check 2: Gradient magnitude comparison (Worked Example 3)")
    print("=" * 65)
    print(f"{'D(G(z))':>10s} | {'minimax grad':>14s} | {'non-sat grad':>14s} | {'ratio':>8s}")
    for d in [0.01, 0.10, 0.50, 0.90]:
        mm = minimax_grad_magnitude(d)
        ns = nonsaturating_grad_magnitude(d)
        print(f"{d:>10.2f} | {mm:>14.4f} | {ns:>14.4f} | {ns/mm:>8.2f}")
    print("At D(G(z))=0.01, expect non-saturating grad ~100x larger than minimax grad.")


# ---------------------------------------------------------------------------
# PART 3: Global optimum verification
# ---------------------------------------------------------------------------
def check_global_optimum():
    print()
    print("=" * 65)
    print("Check 3: Global optimum Discriminator loss (Worked Example 4)")
    print("=" * 65)
    L_D_optimum = discriminator_loss(D_real=0.5, D_fake=0.5)
    expected = 2 * np.log(2)
    print(f"L_D at D(x)=D(G(z))=0.5: {L_D_optimum:.4f}  (expected 2*ln(2) = {expected:.4f})")


# ---------------------------------------------------------------------------
# PART 4: Toy 1D FID
# ---------------------------------------------------------------------------
def toy_fid_1d(mu1, sigma1, mu2, sigma2):
    """Simplified 1D version of Frechet Inception Distance."""
    return (mu1 - mu2) ** 2 + sigma1 ** 2 + sigma2 ** 2 - 2 * np.sqrt(sigma1 ** 2 * sigma2 ** 2)


def check_fid():
    print()
    print("=" * 65)
    print("Check 4: Toy 1D FID (Worked Example 5)")
    print("=" * 65)
    fid = toy_fid_1d(0.0, 1.0, 0.5, 1.2)
    print(f"FID(real N(0,1), fake N(0.5,1.2)): {fid:.4f}  (expected 0.29)")
    fid_identical = toy_fid_1d(0.0, 1.0, 0.0, 1.0)
    print(f"FID(identical distributions): {fid_identical:.4f}  (expected 0.0)")


# ---------------------------------------------------------------------------
# PART 5: A genuinely trainable tiny 1D GAN
#          Real data ~ N(4, 1.5).  Generator learns to map noise z~N(0,1)
#          to this target distribution, purely via adversarial training.
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TinyGAN1D:
    """A minimal linear Generator and Discriminator for a 1D toy problem."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # Generator: x_fake = g_w * z + g_b  (learns to shift/scale noise)
        self.g_w = rng.normal(1, 0.1)
        self.g_b = rng.normal(0, 0.1)
        # Discriminator: score = sigmoid(d_w * x + d_b)
        self.d_w = rng.normal(0, 0.1)
        self.d_b = rng.normal(0, 0.1)

    def generate(self, z):
        return self.g_w * z + self.g_b

    def discriminate(self, x):
        return sigmoid(self.d_w * x + self.d_b)

    def train_step(self, real_batch, z_batch, lr=0.01):
        # --- Discriminator step ---
        fake_batch = self.generate(z_batch)
        D_real = self.discriminate(real_batch)
        D_fake = self.discriminate(fake_batch)

        # Gradients of L_D w.r.t. d_w, d_b (derived from L_D = -[log(D_real)+log(1-D_fake)])
        grad_d_real = (D_real - 1) * real_batch      # d/d(d_w) contribution from real term
        grad_d_fake = D_fake * fake_batch             # d/d(d_w) contribution from fake term
        grad_dw = np.mean(grad_d_real + grad_d_fake)
        grad_db = np.mean((D_real - 1) + D_fake)
        self.d_w -= lr * grad_dw
        self.d_b -= lr * grad_db

        # --- Generator step (non-saturating loss: maximize log(D(G(z)))) ---
        fake_batch = self.generate(z_batch)           # recompute with updated D implicitly next
        D_fake = self.discriminate(fake_batch)
        # d(-log(D_fake))/d(fake) = -(1-D_fake)*d_w ; then chain into g_w, g_b via fake=g_w*z+g_b
        grad_fake = -(1 - D_fake) * self.d_w
        grad_gw = np.mean(grad_fake * z_batch)
        grad_gb = np.mean(grad_fake)
        self.g_w -= lr * grad_gw
        self.g_b -= lr * grad_gb

        return np.mean(discriminator_loss(D_real, D_fake))


def run_tiny_gan_training_demo():
    print()
    print("=" * 65)
    print("Check 5: Training a tiny 1D GAN to match a real N(4, 1.5) distribution")
    print("=" * 65)
    rng = np.random.default_rng(42)
    gan = TinyGAN1D(seed=7)

    true_mean, true_std = 4.0, 1.5
    batch_size = 64

    for step in range(2001):
        real_batch = rng.normal(true_mean, true_std, batch_size)
        z_batch = rng.normal(0, 1, batch_size)
        loss = gan.train_step(real_batch, z_batch, lr=0.02)
        if step % 500 == 0:
            fake_sample = gan.generate(rng.normal(0, 1, 1000))
            print(f"Step {step:4d} | D_loss={loss:.4f} | "
                  f"Generator output mean={fake_sample.mean():.3f}, std={fake_sample.std():.3f} "
                  f"(target mean={true_mean}, std={true_std})")

    print("\nIf training worked, the Generator's mean/std should approach 4.0 / 1.5,")
    print("purely from adversarial feedback - no direct access to the real data's statistics.")


if __name__ == "__main__":
    check_losses()
    check_gradient_comparison()
    check_global_optimum()
    check_fid()
    run_tiny_gan_training_demo()
