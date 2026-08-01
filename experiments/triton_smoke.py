"""Minimal correctness check for the Linux Triton kernel environment."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, count: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(axis=0) * block_size + tl.arange(0, block_size)
    mask = offsets < count
    tl.store(output_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask), mask=mask)


def main() -> None:
    x = torch.randn(65_537, device="cuda")
    y = torch.randn_like(x)
    output = torch.empty_like(x)
    block_size = 256
    add_kernel[(triton.cdiv(x.numel(), block_size),)](x, y, output, x.numel(), block_size=block_size)
    torch.testing.assert_close(output, x + y)
    print(f"Triton smoke test passed on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
