"""Minimal tensor-alignment helper used by the private split-K combine kernel."""

import cutlass
import cutlass.cute as cute


def assume_tensor_aligned(tensor):
    if tensor is None:
        return None
    divby = 128 // tensor.element_type.width
    strides = tuple(
        stride
        if isinstance(stride, int)
        else cute.assume(stride, divby=divby)
        for stride in tensor.stride[:-1]
    ) + (tensor.stride[-1],)
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(tensor.shape, stride=strides),
    )
