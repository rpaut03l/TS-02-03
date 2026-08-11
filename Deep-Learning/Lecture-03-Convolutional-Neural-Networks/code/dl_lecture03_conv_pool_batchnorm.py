"""
DL Lecture 03 - Convolution, Max Pooling, and Batch Normalization From Scratch
================================================================================
What this file does (in plain words):
  Implements the three core CNN operations from the theory file - convolution,
  max pooling, and batch normalization - using only NumPy, and checks each one
  against the exact worked examples in numerical/dl_lecture03_cnn_numerical.md
  so you can see the formulas AND the code agree, number for number.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


# ---------------------------------------------------------------------------
# PART 1: Output volume size formula (the "how big will my output be" math)
# ---------------------------------------------------------------------------
def conv_output_size(input_size: int, filter_size: int, stride: int, padding: int) -> int:
    """
    Implements: W2 = (W1 - F + 2P) / S + 1

    input_size  : W1 (or H1) - the input's width or height
    filter_size : F - the filter's spatial size
    stride      : S - how many pixels the filter jumps each step
    padding     : P - how many pixels of zero-border are added
    """
    return (input_size - filter_size + 2 * padding) // stride + 1


# ---------------------------------------------------------------------------
# PART 2: A simple 2D convolution (single channel, single filter, for clarity)
# ---------------------------------------------------------------------------
def convolve2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Slides `kernel` over `image` and computes the sum-of-elementwise-products
    at every position - this IS the "cookie cutter stamped everywhere" idea
    from the theory file, in code.
    """
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode="constant", constant_values=0)

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    out_h = conv_output_size(img_h, k_h, stride, 0)   # padding already applied above
    out_w = conv_output_size(img_w, k_w, stride, 0)

    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            row_start = i * stride
            col_start = j * stride
            patch = image[row_start:row_start + k_h, col_start:col_start + k_w]
            output[i, j] = np.sum(patch * kernel)   # elementwise multiply, then sum
    return output


# ---------------------------------------------------------------------------
# PART 3: Max Pooling (verified against Worked Example 6 in the numerical file)
# ---------------------------------------------------------------------------
def max_pool2d(feature_map: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """Keeps only the largest value in each pool_size x pool_size window."""
    fm_h, fm_w = feature_map.shape
    out_h = (fm_h - pool_size) // stride + 1
    out_w = (fm_w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            row_start = i * stride
            col_start = j * stride
            window = feature_map[row_start:row_start + pool_size, col_start:col_start + pool_size]
            output[i, j] = np.max(window)
    return output


# ---------------------------------------------------------------------------
# PART 4: Batch Normalization (verified against Worked Example 5)
# ---------------------------------------------------------------------------
def batch_norm(x: np.ndarray, gamma: float = 1.0, beta: float = 0.0, epsilon: float = 0.0) -> np.ndarray:
    """
    Implements: x_hat = (x - mean) / sqrt(variance + epsilon)
                y = gamma * x_hat + beta
    """
    mean = np.mean(x)
    variance = np.var(x)                # NumPy's default: divides by N (matches this course's convention)
    x_hat = (x - mean) / np.sqrt(variance + epsilon)
    y = gamma * x_hat + beta
    return y


# ---------------------------------------------------------------------------
# PART 5: Run every check against the numerical README's worked examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("Check 1: Output volume formula (32x32x3, F=5, S=1, P=0)")
    print("=" * 65)
    size = conv_output_size(32, 5, 1, 0)
    print(f"Computed output size: {size}x{size}  (expected 28x28)")

    print()
    print("=" * 65)
    print("Check 2: The 64x64x3 in-class example (F=3, S=1, P=1)")
    print("=" * 65)
    size2 = conv_output_size(64, 3, 1, 1)
    print(f"Computed output size: {size2}x{size2}  (expected 64x64 - 'same' padding)")

    print()
    print("=" * 65)
    print("Check 3: AlexNet CONV1 (227x227x3, F=11, S=4, P=0)")
    print("=" * 65)
    size3 = conv_output_size(227, 11, 4, 0)
    print(f"Computed output size: {size3}x{size3}  (expected 55x55)")
    params_per_filter = 11 * 11 * 3 + 1
    total_params = params_per_filter * 96
    print(f"Params per filter: {params_per_filter}  | Total CONV1 params: {total_params:,}  (expected 34,944)")

    print()
    print("=" * 65)
    print("Check 4: Max Pooling on the numerical file's 4x4 example")
    print("=" * 65)
    feature_map = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [8, 2, 9, 0],
        [3, 7, 4, 5],
    ])
    pooled = max_pool2d(feature_map, pool_size=2, stride=2)
    print("Input:\n", feature_map)
    print("Max-pooled output (expected [[6,4],[8,9]]):\n", pooled)

    print()
    print("=" * 65)
    print("Check 5: Batch Normalization on the numerical file's example")
    print("=" * 65)
    x = np.array([2.0, 4.0, 4.0, 8.0])
    y = batch_norm(x, gamma=2.0, beta=1.0, epsilon=0.0)
    print("Input:", x)
    print("BatchNorm output (expected approx [-1.294, 0.541, 0.541, 4.212]):")
    print(np.round(y, 3))

    print()
    print("=" * 65)
    print("Check 6: A tiny real convolution (Sobel-like vertical edge filter)")
    print("=" * 65)
    toy_image = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ], dtype=float)
    vertical_edge_kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=float)
    edges = convolve2d(toy_image, vertical_edge_kernel, stride=1, padding=0)
    print("Toy image (bright left half, dark right half):\n", toy_image)
    print("Convolution output (large values = strong vertical edge detected):\n", edges)
