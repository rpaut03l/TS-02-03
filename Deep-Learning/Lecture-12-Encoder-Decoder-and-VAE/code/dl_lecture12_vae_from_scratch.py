"""
DL Lecture 12 - A Mini Variational Autoencoder Built From Scratch (NumPy only)
=================================================================================
What this file does (in plain words):
  Implements a tiny end-to-end VAE pipeline in NumPy: an encoder producing
  mu and sigma, the reparameterization trick, a decoder, the closed-form
  Gaussian KL divergence formula, and a full "sample from prior, then
  decode" generation demo. Every formula is checked against the numerical
  README's worked examples.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


# ---------------------------------------------------------------------------
# PART 1: Compression ratio + VAE vs plain-AE output size
# ---------------------------------------------------------------------------
def compression_ratio(input_size, latent_size):
    return input_size / latent_size


def vae_encoder_output_size(latent_dim):
    """A VAE outputs BOTH a mean vector AND a std-dev vector -> 2 x N."""
    return 2 * latent_dim


def check_compression_and_sizes():
    print("=" * 65)
    print("Check 1: Compression ratios + VAE output size (Worked Examples 1 & 2)")
    print("=" * 65)
    print(f"16384px -> 20 latent: ratio = {compression_ratio(16384, 20)}  (expected 819.2)")
    print(f"1300pt  -> 32 latent: ratio = {compression_ratio(1300, 32):.3f}  (expected ~40.625)")
    print(f"N=64 VAE encoder output size: {vae_encoder_output_size(64)}  (expected 128)")


# ---------------------------------------------------------------------------
# PART 2: KL divergence, closed form for Gaussian vs standard normal
# ---------------------------------------------------------------------------
def kl_divergence_gaussian(mu, sigma):
    """
    Implements: KL = 0.5 * sum(sigma^2 + mu^2 - 1 - ln(sigma^2))
    Measures how far N(mu, sigma^2) is from the standard normal N(0,1).
    """
    mu = np.array(mu, dtype=float)
    sigma = np.array(sigma, dtype=float)
    return 0.5 * np.sum(sigma ** 2 + mu ** 2 - 1 - np.log(sigma ** 2))


def check_kl_divergence():
    print()
    print("=" * 65)
    print("Check 2: KL divergence (Worked Example 3)")
    print("=" * 65)
    kl = kl_divergence_gaussian(mu=[0.5, -0.2, 1.0], sigma=[1.2, 0.8, 0.5])
    print(f"KL(learned || prior): {kl:.4f}  (expected ~1.044)")

    kl_at_prior = kl_divergence_gaussian(mu=[0, 0, 0], sigma=[1, 1, 1])
    print(f"KL when encoder output EQUALS the prior: {kl_at_prior:.6f}  (expected exactly 0)")


# ---------------------------------------------------------------------------
# PART 3: The reparameterization trick
# ---------------------------------------------------------------------------
def reparameterize(mu, sigma, epsilon):
    """Implements: z = mu + sigma * epsilon"""
    return mu + sigma * epsilon


def check_reparameterization():
    print()
    print("=" * 65)
    print("Check 3: Reparameterization trick (Worked Example 4)")
    print("=" * 65)
    z = reparameterize(mu=2.0, sigma=0.5, epsilon=1.0)
    print(f"z = mu + sigma*epsilon = {z}  (expected 2.5)")


# ---------------------------------------------------------------------------
# PART 4: A full mini VAE - encoder, reparameterize, decoder, generation
# ---------------------------------------------------------------------------
class MiniVAE:
    def __init__(self, input_dim, latent_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.latent_dim = latent_dim
        # Encoder: maps input -> (mu, log_sigma)
        self.W_mu = rng.normal(0, 0.3, (latent_dim, input_dim))
        self.W_sigma = rng.normal(0, 0.3, (latent_dim, input_dim))
        # Decoder: maps latent z -> reconstructed input
        self.W_dec = rng.normal(0, 0.3, (input_dim, latent_dim))

    def encode(self, x):
        mu = self.W_mu @ x
        log_sigma = self.W_sigma @ x
        sigma = np.exp(0.5 * log_sigma)   # ensures sigma is always positive
        return mu, sigma

    def decode(self, z):
        return self.W_dec @ z

    def forward(self, x, rng):
        mu, sigma = self.encode(x)
        epsilon = rng.normal(0, 1, self.latent_dim)   # epsilon ~ N(0,1)
        z = reparameterize(mu, sigma, epsilon)         # the reparameterization trick
        x_reconstructed = self.decode(z)
        kl = kl_divergence_gaussian(mu, sigma)
        return x_reconstructed, mu, sigma, kl

    def generate_new_sample(self, rng):
        """The two-step generation process: sample z from the prior, then decode."""
        z = rng.normal(0, 1, self.latent_dim)   # Step 1: sample from prior p(z) = N(0,1)
        return self.decode(z)                    # Step 2: decode into a new data sample


def run_mini_vae_demo():
    print()
    print("=" * 65)
    print("Check 4: A full mini VAE - encode, reparameterize, decode, generate")
    print("=" * 65)
    input_dim, latent_dim = 6, 3
    vae = MiniVAE(input_dim, latent_dim, seed=1)
    rng = np.random.default_rng(2)

    x = rng.normal(0, 1, input_dim)
    print(f"Original input x: {np.round(x, 3)}")

    x_recon, mu, sigma, kl = vae.forward(x, rng)
    print(f"Encoder mu:    {np.round(mu, 3)}")
    print(f"Encoder sigma: {np.round(sigma, 3)}")
    print(f"Reconstructed x: {np.round(x_recon, 3)}")
    print(f"KL divergence for this input: {kl:.4f}")

    print("\nGenerating 3 brand-new synthetic samples (sample z from prior, then decode):")
    for i in range(3):
        new_sample = vae.generate_new_sample(rng)
        print(f"  Sample {i+1}: {np.round(new_sample, 3)}")


if __name__ == "__main__":
    check_compression_and_sizes()
    check_kl_divergence()
    check_reparameterization()
    run_mini_vae_demo()
