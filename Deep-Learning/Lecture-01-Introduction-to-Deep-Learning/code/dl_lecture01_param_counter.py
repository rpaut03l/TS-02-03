"""
DL Lecture 01 - Parameter Counter Demo
========================================
What this file does (in plain words):
  It is a tiny calculator that answers the exact question Lecture 1 asks:
  "How many weights does a Fully Connected layer need vs a Convolutional
  layer, for the same image?" It prints the numbers AND the ratio, so you
  can reproduce every number in numerical/dl_lecture01_introduction_numerical.md
  yourself, and try your own image sizes.

No deep learning library is required to run this file - it is pure Python
arithmetic, so it works identically on your laptop, Google Colab, or Kaggle
with zero setup.
"""

# ---------------------------------------------------------------------------
# STEP 1: A function to count parameters in a Fully Connected (Dense) layer
# ---------------------------------------------------------------------------
def fc_layer_params(height: int, width: int, channels: int, hidden_units: int) -> int:
    """
    Counts parameters in a fully connected layer that takes a flattened
    image as input.

    height, width, channels : the shape of the input image (e.g. 256, 256, 3)
    hidden_units            : how many neurons are in this fully connected layer

    Formula: (H * W * C) inputs, each connected to every hidden unit,
             plus one bias per hidden unit.
    """
    num_inputs = height * width * channels        # flatten the image into one long vector
    weights = num_inputs * hidden_units            # every input connects to every neuron
    biases = hidden_units                           # each neuron gets its own bias number
    total = weights + biases
    return total


# ---------------------------------------------------------------------------
# STEP 2: A function to count parameters in a Convolutional layer
# ---------------------------------------------------------------------------
def conv_layer_params(kernel_size: int, in_channels: int, num_filters: int) -> int:
    """
    Counts parameters in a convolutional layer.

    kernel_size  : the filter's spatial size (e.g. 3 means a 3x3 filter)
    in_channels  : how many channels the INPUT has (e.g. 3 for RGB)
    num_filters  : how many different filters this layer learns (its OUTPUT depth)

    Formula: each filter has (kernel_size * kernel_size * in_channels) weights
             plus 1 bias, and we have num_filters such filters.
    Notice: this formula does NOT depend on the image's height or width at all -
    that independence is exactly why CNNs scale so well to large images.
    """
    weights_per_filter = kernel_size * kernel_size * in_channels
    params_per_filter = weights_per_filter + 1      # +1 bias for this filter
    total = params_per_filter * num_filters
    return total


# ---------------------------------------------------------------------------
# STEP 3: Reproduce every worked example from the numerical README
# ---------------------------------------------------------------------------
def run_lecture_examples():
    print("=" * 60)
    print("Example A: 1-megapixel grayscale image, 1000 hidden units")
    print("=" * 60)
    # 1 megapixel ~ 1000 x 1000 x 1 (grayscale, so channels=1)
    a = fc_layer_params(height=1000, width=1000, channels=1, hidden_units=1000)
    print(f"Fully Connected parameters: {a:,}")   # should print ~1,001,000,000

    print()
    print("=" * 60)
    print("Example B: In-class question - 256x256x3 image")
    print("=" * 60)
    fc = fc_layer_params(height=256, width=256, channels=3, hidden_units=32)
    conv = conv_layer_params(kernel_size=3, in_channels=3, num_filters=32)
    print(f"Fully Connected (32 units): {fc:,} parameters")
    print(f"Convolutional (3x3, 32 filters): {conv:,} parameters")
    print(f"FC needs {fc / conv:,.0f}x more parameters than Conv")

    print()
    print("=" * 60)
    print("Example C: Try your own image size (edit the numbers below!)")
    print("=" * 60)
    my_height, my_width, my_channels = 512, 512, 3
    my_hidden_units, my_kernel, my_filters = 64, 5, 64
    fc2 = fc_layer_params(my_height, my_width, my_channels, my_hidden_units)
    conv2 = conv_layer_params(my_kernel, my_channels, my_filters)
    print(f"Fully Connected: {fc2:,} parameters")
    print(f"Convolutional:   {conv2:,} parameters")
    print(f"FC needs {fc2 / conv2:,.0f}x more parameters than Conv")


# ---------------------------------------------------------------------------
# STEP 4: Only run the examples if this file is executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # This "if" guard means: if someone IMPORTS this file into another script
    # (to reuse the two functions above), the examples will NOT auto-run.
    # It only runs when you execute this file directly, e.g. `python dl_lecture01_param_counter.py`
    run_lecture_examples()
