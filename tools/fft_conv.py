from functools import partial
from typing import Iterable, Tuple, Union

import sys
from fxpmath import Fxp
import numpy as np
import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn


def quantize_fi(x, signed, int_bits, frac_bits, outfile=None, folding=1):

    # print(str(x.dtype), x[-1, 0, 0], 'complex' in str(x.dtype))
    width = int_bits + frac_bits
    print('quantize fi shape:', len(x.shape), x.shape)
    batches, channels, h, w = x.shape
    in_x_r = np.real(x)
    channels_per_fold = channels // folding
    x = x.reshape((batches, folding, channels_per_fold, h, w))
    x_r = np.real(x)
    x = Fxp(x, signed, width, frac_bits)
    
    if outfile:
        content = 'memory_initialization_radix=2;\n'
        content += 'memory_initialization_vector=\n'
        for batch in range(batches):
            for fold in range(folding):  
                for row in range(h):
                    for col in range(w):
                        for channel in range(channels_per_fold):
                            el = x[batch, fold, channel, row, col]
                            if 'complex' in str(x.dtype):
                                # print(type(el.bin()), el.bin(), el.shape)
                                # print(str(el.bin()))
                                bin_repr = str(el.bin())[:-1]
                                bin_repr = bin_repr.replace('+', '')
                                bin_repr = bin_repr.replace('-', '')
                                bin_real = bin_repr[0:width]
                                bin_imag = bin_repr[width:]
                                content += bin_imag + bin_real
                            else:
                                content += el.bin()
                        content += '\n'
        content += ';'

        with open(outfile, 'w') as result_file:
            result_file.write(content)
        sys.exit()


# x in NCHW shape
def to_binarray(x, bitwidth, folding=8, reverse_endian=True, reverse_inner=True, outpath='bytearray.bin'):

    # fold channels
    x = x.astype('int32')
    N, C, H, W = x.shape
    channels_per_fold = C // folding
    x = x.reshape((N, folding, channels_per_fold, H, W))

    # change each number to u2 integer
    result = np.zeros(x.shape, dtype=int)
    mask = x < 0
    result[mask] = x[mask] + (1 << bitwidth)
    result[~mask] = x[~mask]

    # divide each number to bytes
    bytes_per_num = bitwidth // 8
    result_bytes = np.zeros(result.shape + (bytes_per_num,), dtype=int)
    print('result_bytes:', result_bytes.shape)
    for byte in range(bytes_per_num):
        result_bytes[..., byte] = (result >> byte*8) & 255

    # reverse bytes
    if reverse_endian:
        result_bytes = np.flip(result_bytes, axis=-1)
    # reverse channels in each fold
    if reverse_inner:
        result_bytes = np.flip(result_bytes, axis=2)

    # result_bytes in [batch, fold, channels_per_fold, h, w, bytes_per_num] shape
    # transpose to [batch, h, w, fold, channel_per_fold, bytes_per_num]
    result_bytes = result_bytes.transpose((0, 3, 4, 1, 2, 5))
    result_bytes = result_bytes.astype(np.uint8).tobytes()
    # result_bytes = bytearray(result_bytes)
    with open(outpath, "wb") as outfile:
        outfile.write(result_bytes)


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.

    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    # print("tofft signal, kernel", signal_.shape, padded_kernel.shape)
    # print(padded_kernel[0, 0])
    MYPAD_signal_ = f.pad(signal_, [0, 10, 0, 10])
    # print('signal range:', torch.min(signal_), torch.max(signal_), np.unique(signal_.cpu().numpy()).shape)
    # to_binarray(MYPAD_signal_.cpu().numpy(), bitwidth=16, folding=1, outpath="test_inputs/signal32x32_w16.bin")
    MYPAD_padded_kernel = f.pad(padded_kernel, [0, 10, 0, 10])
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))
    kernel_fr_torch_r = np.real(rfftn(MYPAD_padded_kernel, dim=tuple(range(2, signal.ndim)))/256)
    kernel_fr_torch_i = np.imag(rfftn(MYPAD_padded_kernel, dim=tuple(range(2, signal.ndim)))/256)


    signal_fr_np = np.fft.fft2(MYPAD_signal_)
    kernel_fr_np = np.fft.fft2(MYPAD_padded_kernel)
    # NOPAD_kernel_fr = rfftn(kernel, dim=tuple(range(2, signal.ndim)))

    # NOPAD_kernel_fr_r = np.real(NOPAD_kernel_fr)
    # NOPAD_kernel_fr_i = np.imag(NOPAD_kernel_fr)
    # signal_fr_r_np = np.real(np.fft.fft2(signal_.cpu().numpy()))

    kernel_fr.imag *= -1
    kernel_fr_np.imag *= -1

    kernel_fr_np /= 256
    # quantize_fi(kernel_fr_np[:, :, :, :17], True, 16, 0, folding=8, outfile='memory_inits/crossing_template_32x17_s16_0.coe')
    signal_fr_np /= 256
    output_fr_np = signal_fr_np * kernel_fr_np
    signal_fr_r = np.real(signal_fr_np)
    signal_fr_i = np.imag(signal_fr_np)
    kernel_fr_r = np.real(kernel_fr_np)
    kernel_fr_i = np.imag(kernel_fr_np)
    # print('template range r', np.min(kernel_fr_r), np.max(kernel_fr_r), 'imag:', np.min(kernel_fr_i), np.max(kernel_fr_i))
    output_fr_np_r = np.real(output_fr_np)
    output_fr_np_i = np.imag(output_fr_np)
    # print('product range range r', np.min(output_fr_np_r), np.max(output_fr_np_r), 'imag:', np.min(output_fr_np_i), np.max(output_fr_np_i))

    output_fr_np_16_REAL = np.real(output_fr_np[:, :16, :, :])
    output_fr_np_16sum = np.sum(output_fr_np[:, :16, :, :], axis=1)
    output_fr_np_16sum_r = np.real(output_fr_np_16sum)
    output_fr_np_32sum_r = np.real(np.sum(output_fr_np[:, :32, :, :], axis=1))
    output_fr_np_sum = np.sum(output_fr_np, axis=1)
    output_fr_np_sum_r = np.real(output_fr_np_sum)
    # print('max output_fr_np_sum_r:', np.min(np.real(output_fr_np_sum)), np.max(np.real(output_fr_np_sum)))

    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
    output_np = np.real(np.fft.ifft2(output_fr_np_sum))
    output_np_r = output_np/256
    # print('output:', output.shape)
    # Remove extra padded values
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    # print('crop slices', crop_slices)
    output = output[crop_slices].contiguous()
    # print('output cropped', output.shape)
    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        ndim: int = 1,
    ):
        """
        Args:
            in_channels: (int) Number of channels in input tensors
            out_channels: (int) Number of channels in output tensors
            kernel_size: (Union[int, Iterable[int]) Square radius of the kernel
            padding: (Union[int, Iterable[int]) Number of zero samples to pad the
                input on the last dimension.
            stride: (Union[int, Iterable[int]) Stride size for computing output values.
            bias: (bool) If True, includes bias, which is added after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


FFTConv1d = partial(_FFTConv, ndim=1)
FFTConv2d = partial(_FFTConv, ndim=2)
FFTConv3d = partial(_FFTConv, ndim=3)
