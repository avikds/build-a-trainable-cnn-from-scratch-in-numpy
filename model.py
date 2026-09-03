"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np

def argmax_rows(matrix):
    # Return the column index of the largest element in each row.
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
def row_max(matrix):
    # Return the maximum value of each row, preserving the row dimension.
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    return np.exp(logits - row_max(logits))

# Step 5 - stable_softmax
def stable_softmax(logits):
    # Compute a numerically stable softmax row-wise over (N, C) logits.
    shifted_exp = exp_shifted(logits)
    return shifted_exp / row_sum(shifted_exp)

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # Convert integer labels into a (N, num_classes) one-hot float matrix.
    labels = np.asarray(labels)
    result = np.zeros((labels.shape[0], num_classes), dtype=float)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # Return the probability assigned to the true class for each sample.
    return probs[np.arange(probs.shape[0]), labels]

# Step 8 - cross_entropy_loss
def cross_entropy_loss(probs, labels, eps=1e-12):
    # Return the mean negative log-likelihood of the true-class probabilities.
    true_class_probs = gather_true_class_probs(probs, labels)
    true_class_probs = np.clip(true_class_probs, eps, None)
    return -np.mean(np.log(true_class_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # Return the fraction of rows whose argmax matches the integer label.
    predictions = argmax_rows(logits_or_probs)
    return np.mean(predictions == labels)

# Step 10 - he_std
import math

def he_std(fan_in):
    # Return the He initialization standard deviation sqrt(2 / fan_in).
    return float(math.sqrt(2.0 / fan_in))

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # Seed NumPy's global RNG for reproducibility.
    np.random.seed(seed)

    # Sample weights from a zero-mean normal distribution
    # with the He initialization standard deviation.
    return np.random.normal(
        loc=0.0,
        scale=he_std(fan_in),
        size=shape
    ).astype(np.float64)

# Step 12 - init_zero_bias
def init_zero_bias(length):
    # Return a 1D float64 array of zeros with the given length.
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # Zero-pad the spatial (H, W) dimensions on both sides.
    if pad == 0:
        return images

    return np.pad(
        images,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0
    )

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # Return the output spatial dimension for a valid integer-producing
    # convolution or pooling configuration.
    return int((input_size + 2 * padding - kernel) / stride + 1)

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    # Zero-pad the input spatial dimensions first.
    padded = pad_2d(images, padding)

    n, c, h, w = images.shape

    # Compute output spatial dimensions.
    out_h = output_spatial_size(h, kernel_h, stride, padding)
    out_w = output_spatial_size(w, kernel_w, stride, padding)

    # Allocate the column matrix.
    cols = np.empty(
        (n * out_h * out_w, c * kernel_h * kernel_w),
        dtype=images.dtype
    )

    row = 0

    # Order patches by sample, then output row, then output column.
    # Within each patch, values are stored channel-major.
    for i in range(n):
        for oh in range(out_h):
            h_start = oh * stride
            h_end = h_start + kernel_h

            for ow in range(out_w):
                w_start = ow * stride
                w_end = w_start + kernel_w

                patch = padded[
                    i,
                    :,
                    h_start:h_end,
                    w_start:w_end
                ]

                cols[row] = patch.reshape(-1)
                row += 1

    return cols

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # Unpack the original input shape.
    n, c, h, w = input_shape

    # Compute the output spatial dimensions.
    out_h = output_spatial_size(h, kernel_h, stride, padding)
    out_w = output_spatial_size(w, kernel_w, stride, padding)

    # Accumulate into a padded tensor so that overlapping patches
    # contribute to the same spatial locations.
    padded_h = h + 2 * padding
    padded_w = w + 2 * padding

    images_padded = np.zeros(
        (n, c, padded_h, padded_w),
        dtype=cols.dtype
    )

    row = 0

    # Patches are ordered by sample, then output row, then output column,
    # matching the ordering used by im2col.
    for i in range(n):
        for oh in range(out_h):
            h_start = oh * stride
            h_end = h_start + kernel_h

            for ow in range(out_w):
                w_start = ow * stride
                w_end = w_start + kernel_w

                patch = cols[row].reshape(c, kernel_h, kernel_w)

                # Accumulate because overlapping patches contribute
                # to the same elements of the image.
                images_padded[
                    i,
                    :,
                    h_start:h_end,
                    w_start:w_end
                ] += patch

                row += 1

    # Remove the padding before returning.
    if padding > 0:
        return images_padded[:, :, padding:-padding, padding:-padding]

    return images_padded

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    # Unpack dimensions.
    n, c_in, h, w = x.shape
    c_out, _, kernel_h, kernel_w = weights.shape

    # Unroll input patches using im2col.
    cols = im2col(x, kernel_h, kernel_w, stride, padding)

    # Reshape weights so each output channel corresponds to one row.
    weights_2d = weights.reshape(c_out, -1)

    # Perform the convolution as a matrix multiplication and add bias.
    output_2d = cols @ weights_2d.T + bias

    # Compute output spatial dimensions.
    out_h = output_spatial_size(h, kernel_h, stride, padding)
    out_w = output_spatial_size(w, kernel_w, stride, padding)

    # Restore the expected 4D feature-map layout.
    output = output_2d.reshape(n, out_h, out_w, c_out)
    output = output.transpose(0, 3, 1, 2)

    # Cache values required for the backward pass.
    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
    }

    return output, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    # Retrieve cached values from the forward pass.
    weights = cache["weights"]
    x_shape = cache["x_shape"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    n, c_out, out_h, out_w = d_out.shape
    c_in = x_shape[1]

    # Rearrange upstream gradients to match the im2col row ordering:
    # sample, output row, output column, output channel.
    d_out_2d = d_out.transpose(0, 2, 3, 1).reshape(n * out_h * out_w, c_out)

    # Each input patch receives gradients from every output channel.
    weights_2d = weights.reshape(c_out, c_in * kernel_h * kernel_w)
    d_cols = d_out_2d @ weights_2d

    # Fold the patch gradients back into the original image layout.
    dx = col2im(
        d_cols,
        x_shape,
        kernel_h,
        kernel_w,
        stride,
        padding
    )

    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # Retrieve cached values.
    cols = cache["cols"]
    weights = cache["weights"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    n, c_out, out_h, out_w = d_out.shape

    # Match the row ordering used by im2col:
    # sample, output row, output column, output channel.
    d_out_2d = d_out.transpose(0, 2, 3, 1).reshape(
        n * out_h * out_w,
        c_out
    )

    # Forward uses:
    # output_2d = cols @ weights_2d.T + bias
    # Therefore:
    # dW_2d = d_out_2d.T @ cols
    d_weights_2d = d_out_2d.T @ cols

    # Restore the original weight tensor shape.
    return d_weights_2d.reshape(weights.shape)

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # Sum over the batch and spatial dimensions, leaving one value per output channel.
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # Compute all convolution gradients using the corresponding helpers.
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)

    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # Unpack input dimensions.
    n, c, h, w = x.shape

    # Max pooling uses no padding.
    out_h = output_spatial_size(h, kernel, stride, 0)
    out_w = output_spatial_size(w, kernel, stride, 0)

    # Allocate output and cache the flat argmax index for each window.
    out = np.empty((n, c, out_h, out_w), dtype=x.dtype)
    argmax = np.empty((n, c, out_h, out_w), dtype=np.int64)

    # Apply the pooling window independently to each sample and channel.
    for i in range(n):
        for ch in range(c):
            for oh in range(out_h):
                h_start = oh * stride
                h_end = h_start + kernel

                for ow in range(out_w):
                    w_start = ow * stride
                    w_end = w_start + kernel

                    window = x[i, ch, h_start:h_end, w_start:w_end]

                    # Flatten the window so argmax is the required
                    # in-window index in [0, kernel * kernel).
                    flat_window = window.reshape(-1)
                    idx = np.argmax(flat_window)

                    out[i, ch, oh, ow] = flat_window[idx]
                    argmax[i, ch, oh, ow] = idx

    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride,
    }

    return out, cache

# Step 23 - scatter_grad_window
def scatter_grad_window(grad_value, argmax_index, kernel):
    # Create a zero-filled pooling window.
    grad_window = np.zeros((kernel, kernel), dtype=float)

    # Convert the flat row-major index into 2D coordinates.
    row = argmax_index // kernel
    col = argmax_index % kernel

    # Place the upstream gradient at the argmax position.
    grad_window[row, col] = grad_value

    return grad_window

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    # Retrieve the original pooling configuration.
    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]

    n, c, h, w = x_shape
    _, _, out_h, out_w = d_out.shape

    # Initialize the input gradient.
    dx = np.zeros(x_shape, dtype=d_out.dtype)

    # Route each upstream gradient to the cached maximum position.
    for i in range(n):
        for ch in range(c):
            for oh in range(out_h):
                h_start = oh * stride
                h_end = h_start + kernel

                for ow in range(out_w):
                    w_start = ow * stride
                    w_end = w_start + kernel

                    grad_window = scatter_grad_window(
                        d_out[i, ch, oh, ow],
                        argmax[i, ch, oh, ow],
                        kernel
                    )

                    # Accumulate because pooling windows may overlap.
                    dx[i, ch, h_start:h_end, w_start:w_end] += grad_window

    return dx

# Step 25 - relu_forward
def relu_forward(x):
    # Compute the elementwise ReLU and cache the original input.
    out = np.maximum(0, x)

    cache = {
        "x": x
    }

    return out, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # Propagate gradients only where the cached input is strictly positive.
    x = cache["x"]
    return d_out * (x > 0)

# Step 27 - flatten_forward
def flatten_forward(x):
    # Cache the original shape for the backward pass.
    cache = {
        "x_shape": x.shape
    }

    # Flatten each sample while preserving the batch dimension.
    out = x.reshape(x.shape[0], -1)

    return out, cache

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

