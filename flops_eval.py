"""
GPU FLOPS / TOPS benchmark across FP32 / FP16 / BF16 / FP8 / FP4 / INT8.

Fixes (v2):
1. INT8 probe uses 32x32 to satisfy `torch._int_mm` M>16 / 8-align constraints.
2. FP8/FP4 do NOT go through torch.matmul (unsupported). Use torch._scaled_mm
   with explicit scale_a/scale_b/out_dtype, gated by compute capability
   (>=8.9 for FP8, >=10.0 for FP4).
3. Hopper `_scaled_mm` disallows e5m2 x e5m2; when measuring e5m2 we pair it
   with e4m3fn on the B side. Detection uses the same mixed pairing.
4. `out=` support for torch._int_mm / torch._scaled_mm is detected by an
   actual call (try/except TypeError), NOT by inspect.signature — those are
   C++ builtins with no Python signature.
5. Preallocate outputs and pass `out=` when supported, removing per-iteration
   allocation noise (especially bad for INT8 with INT32 accumulator).
6. Clear skip reasons for every un-runnable dtype.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


# --------------------------------------------------------------------------- #
# Capability helpers
# --------------------------------------------------------------------------- #
def _cc() -> tuple[int, int]:
    return torch.cuda.get_device_capability(torch.device("cuda"))


def _has_scaled_mm() -> bool:
    return hasattr(torch, "_scaled_mm")


# --------------------------------------------------------------------------- #
# Support probes
# --------------------------------------------------------------------------- #
def is_matmul_dtype_supported(dtype: torch.dtype) -> bool:
    """Probe torch.matmul support for FP32/FP16/BF16."""
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    try:
        A = torch.randn(32, 32, dtype=torch.float32, device=device).to(dtype)
        B = torch.randn(32, 32, dtype=torch.float32, device=device).to(dtype)
        torch.matmul(A, B)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _fp8_probe(a_dtype: torch.dtype, b_dtype: torch.dtype) -> bool:
    """Try a small _scaled_mm with given operand dtypes."""
    if not _has_scaled_mm():
        return False
    device = torch.device("cuda")
    try:
        A = torch.randn(32, 32, device=device).to(a_dtype)
        # B must be column-major for _scaled_mm
        B = torch.randn(32, 32, device=device).to(b_dtype).t().contiguous().t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        torch._scaled_mm(
            A, B,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def pick_fp8_b_dtype(a_dtype: torch.dtype) -> Optional[torch.dtype]:
    """
    Given the FP8 dtype we want to benchmark as A, pick a valid B dtype.
    On Hopper, e5m2 x e5m2 is disallowed; pair e5m2 with e4m3fn instead.
    Returns None if no supported pairing exists on this stack.
    """
    if not torch.cuda.is_available() or _cc() < (8, 9) or not _has_scaled_mm():
        return None

    e4m3 = getattr(torch, "float8_e4m3fn", None)
    e5m2 = getattr(torch, "float8_e5m2", None)

    # Preferred: same dtype on both sides.
    if _fp8_probe(a_dtype, a_dtype):
        return a_dtype

    # Fallback: mix with e4m3fn (the "reference" FP8 dtype on Hopper).
    if e4m3 is not None and a_dtype is not e4m3 and _fp8_probe(a_dtype, e4m3):
        return e4m3

    # Try e5m2 on B side as a last resort.
    if e5m2 is not None and a_dtype is not e5m2 and _fp8_probe(a_dtype, e5m2):
        return e5m2

    return None


def is_fp4_matmul_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if getattr(torch, "float4_e2m1fn_x2", None) is None:
        return False
    return _cc() >= (10, 0)


def get_int8_tensor_core_support_status() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available."
    if not hasattr(torch, "_int_mm"):
        return False, "torch._int_mm is not available in this PyTorch version."

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    capability = _cc()
    if capability < (7, 5):
        return False, f"{gpu_name} compute capability {capability} is lower than 7.5."

    try:
        A = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device=device)
        B = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device=device)
        torch._int_mm(A, B)
        torch.cuda.synchronize()
        return True, "supported"
    except Exception as e:
        return False, f"torch._int_mm probe failed: {e}"


# --------------------------------------------------------------------------- #
# Runtime `out=` support detection (works on C++ builtins)
# --------------------------------------------------------------------------- #
def _int_mm_supports_out(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> bool:
    try:
        torch._int_mm(A, B, out=C)
        return True
    except TypeError:
        return False


def _scaled_mm_supports_out(A, B, scale_a, scale_b, out_dtype, C) -> bool:
    try:
        torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b,
                         out_dtype=out_dtype, out=C)
        return True
    except TypeError:
        return False


# --------------------------------------------------------------------------- #
# Timing helper
# --------------------------------------------------------------------------- #
def _time_kernel(fn: Callable[[], None], iterations: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def measure_matmul_flops(dtype: torch.dtype, size: int = 8192, iterations: int = 100) -> None:
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    try:
        A = torch.randn(size, size, dtype=torch.float32, device=device).to(dtype)
        B = torch.randn(size, size, dtype=torch.float32, device=device).to(dtype)
        C = torch.empty(size, size, dtype=dtype, device=device)

        def step() -> None:
            torch.matmul(A, B, out=C)

        elapsed_s = _time_kernel(step, iterations)
        total_flops = 2.0 * (size ** 3) * iterations
        tflops = total_flops / elapsed_s / 1e12
        print(f"[{gpu_name}] [{dtype}] Matrix Size: {size}x{size}, TFLOPS: {tflops:.2f}")
    except Exception as e:
        print(f"[{gpu_name}] [{dtype}] Evaluation failed. Error: {e}")


def measure_fp8_flops(a_dtype: torch.dtype, b_dtype: torch.dtype,
                      size: int = 8192, iterations: int = 100) -> None:
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    label = f"{a_dtype} x {b_dtype}" if a_dtype is not b_dtype else str(a_dtype)
    try:
        A = torch.randn(size, size, device=device).to(a_dtype)
        B = torch.randn(size, size, device=device).to(b_dtype).t().contiguous().t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        C = torch.empty(size, size, dtype=torch.bfloat16, device=device)

        use_out = _scaled_mm_supports_out(A, B, scale_a, scale_b, torch.bfloat16, C)

        def step_out() -> None:
            torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b,
                             out_dtype=torch.bfloat16, out=C)

        def step_alloc() -> None:
            torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b,
                             out_dtype=torch.bfloat16)

        step = step_out if use_out else step_alloc
        elapsed_s = _time_kernel(step, iterations)
        total_flops = 2.0 * (size ** 3) * iterations
        tflops = total_flops / elapsed_s / 1e12
        print(f"[{gpu_name}] [{label}] Matrix Size: {size}x{size}, TFLOPS: {tflops:.2f}")
    except Exception as e:
        print(f"[{gpu_name}] [{label}] Evaluation failed. Error: {e}")


def measure_int8_tensor_core(size: int = 8192, iterations: int = 100) -> None:
    is_supported, reason = get_int8_tensor_core_support_status()
    if not is_supported:
        print(f"[INT8 Tensor Core] Skipped. Reason: {reason}")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    try:
        A = torch.randint(-128, 128, (size, size), dtype=torch.int8, device=device)
        B = torch.randint(-128, 128, (size, size), dtype=torch.int8, device=device)
        C = torch.empty(size, size, dtype=torch.int32, device=device)

        use_out = _int_mm_supports_out(A, B, C)

        def step_out() -> None:
            torch._int_mm(A, B, out=C)

        def step_alloc() -> None:
            torch._int_mm(A, B)

        step = step_out if use_out else step_alloc
        elapsed_s = _time_kernel(step, iterations)
        total_ops = 2.0 * (size ** 3) * iterations
        tops = total_ops / elapsed_s / 1e12
        note = "" if use_out else "  (no out= support; alloc-in-loop overhead included)"
        print(f"[{gpu_name}] [INT8 Tensor Core] Matrix Size: {size}x{size}, TOPS: {tops:.2f}{note}")
    except Exception as e:
        print(f"[{gpu_name}] [INT8 Tensor Core] Evaluation failed. Error: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== GPU FLOPS Evaluation ===")
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.device("cuda")
    print(f"Device       : {torch.cuda.get_device_name(device)}")
    print(f"Capability   : {_cc()}")
    print(f"Torch version: {torch.__version__}")
    print()

    # ---- Float matmul path (torch.matmul) ----
    float_dtypes: list[torch.dtype] = [torch.float32, torch.float16]
    if torch.cuda.is_bf16_supported():
        float_dtypes.append(torch.bfloat16)

    for dt in float_dtypes:
        if is_matmul_dtype_supported(dt):
            measure_matmul_flops(dt)
        else:
            print(f"[{dt}] Skipped. Reason: torch.matmul does not support this dtype on current stack.")

    # ---- FP8 path (torch._scaled_mm) ----
    if _cc() < (8, 9):
        for dtype_name in ("float8_e4m3fn", "float8_e5m2"):
            if getattr(torch, dtype_name, None) is not None:
                print(f"[torch.{dtype_name}] Skipped. Reason: compute capability {_cc()} < 8.9 (needs Ada/Hopper).")
    elif not _has_scaled_mm():
        for dtype_name in ("float8_e4m3fn", "float8_e5m2"):
            if getattr(torch, dtype_name, None) is not None:
                print(f"[torch.{dtype_name}] Skipped. Reason: torch._scaled_mm not available in this PyTorch version.")
    else:
        for dtype_name in ("float8_e4m3fn", "float8_e5m2"):
            a_dtype = getattr(torch, dtype_name, None)
            if a_dtype is None:
                print(f"[torch.{dtype_name}] Skipped. Reason: dtype not exposed in this PyTorch version.")
                continue
            b_dtype = pick_fp8_b_dtype(a_dtype)
            if b_dtype is None:
                print(f"[torch.{dtype_name}] Skipped. Reason: no supported _scaled_mm operand pairing on this stack.")
                continue
            measure_fp8_flops(a_dtype, b_dtype)

    # ---- FP4 path ----
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        print("[torch.float4_e2m1fn_x2] Skipped. Reason: dtype not exposed in this PyTorch version.")
    elif not is_fp4_matmul_supported():
        print(f"[torch.float4_e2m1fn_x2] Skipped. Reason: compute capability {_cc()} < 10.0 (Blackwell required).")
    else:
        print("[torch.float4_e2m1fn_x2] Detected. Skipping timing: FP4 GEMM needs blockwise scales; wire up your CUTLASS/cuBLASLt pipeline to benchmark.")

    # ---- INT8 Tensor Core ----
    measure_int8_tensor_core()


if __name__ == "__main__":
    main()
