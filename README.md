# GPU Evaluation Toolkit

A PyTorch-based toolkit for evaluating GPU performance and bandwidth.

## Scripts

1. **`flops_eval.py`**\
   Evaluates GPU TFLOPS across different precisions (FP32, FP16, BF16) using `torch.matmul`.
2. **`bandwidth_eval.py`**\
   Measures PCIe transfer speeds (Host-to-Device and Device-to-Host) using pinned memory, as well as GPU internal memory bandwidth (Device-to-Device).
3. **`nvlink_eval.py`**\
   Tests pairwise P2P communication bandwidth between multiple GPUs (requires $\ge$ 2 GPUs).

## Usage

```bash
python flops_eval.py
python bandwidth_eval.py
python nvlink_eval.py
```

## Test Results
### H20
```bash
=== GPU FLOPS Evaluation ===
[NVIDIA H20] \[torch.float32] Matrix Size: 8192x8192, TFLOPS: 23.71
[NVIDIA H20] \[torch.float16] Matrix Size: 8192x8192, TFLOPS: 142.10
[NVIDIA H20] \[torch.bfloat16] Matrix Size: 8192x8192, TFLOPS: 139.3

=== CPU-GPU & GPU-GPU Memory Bandwidth Evaluation ===
[NVIDIA H20] Data Size per transfer: 1024 MB, Iterations: 100
[NVIDIA H20] Host to Device (H2D) Bandwidth: 46.62 GB/s
[NVIDIA H20] Device to Host (D2H) Bandwidth: 50.92 GB/s
[NVIDIA H20] Device to Device (D2D) Bandwidth: 1949.40 GB/s

=== Multi-GPU NVLink/P2P Bandwidth Evaluation ===
Found 2 GPUs. Evaluating pairwise P2P bandwidth (Data Size: 1024 MB, Iterations: 100)...

[NVIDIA H20 (GPU 0)] -> [NVIDIA H20 (GPU 1)] Bandwidth: 367.82 GB/s
[NVIDIA H20 (GPU 1)] -> [NVIDIA H20 (GPU 0)] Bandwidth: 367.65 GB/s
```
### A30
```
=== GPU FLOPS Evaluation ===
[NVIDIA A30] [torch.float32] Matrix Size: 8192x8192, TFLOPS: 9.17
[NVIDIA A30] [torch.float16] Matrix Size: 8192x8192, TFLOPS: 89.31
[NVIDIA A30] [torch.bfloat16] Matrix Size: 8192x8192, TFLOPS: 114.33

=== CPU-GPU & GPU-GPU Memory Bandwidth Evaluation ===
[NVIDIA A30] Data Size per transfer: 1024 MB, Iterations: 100
[NVIDIA A30] Host to Device (H2D) Bandwidth: 23.67 GB/s
[NVIDIA A30] Device to Host (D2H) Bandwidth: 24.26 GB/s
[NVIDIA A30] Device to Device (D2D) Bandwidth: 734.02 GB/s

=== Multi-GPU NVLink/P2P Bandwidth Evaluation ===
Found 2 GPUs. Evaluating pairwise P2P bandwidth (Data Size: 1024 MB, Iterations: 100)...

[NVIDIA A30 (GPU 0)] -> [NVIDIA A30 (GPU 1)] Bandwidth: 17.18 GB/s
[NVIDIA A30 (GPU 1)] -> [NVIDIA A30 (GPU 0)] Bandwidth: 17.16 GB/s
```
### V100
```bash
=== GPU FLOPS Evaluation ===
[Tesla V100-SXM2-32GB] [torch.float32] Matrix Size: 8192x8192, TFLOPS: 13.97
[Tesla V100-SXM2-32GB] [torch.float16] Matrix Size: 8192x8192, TFLOPS: 93.16
[Tesla V100-SXM2-32GB] [torch.bfloat16] Matrix Size: 8192x8192, TFLOPS: 10.22

=== CPU-GPU & GPU-GPU Memory Bandwidth Evaluation ===
[Tesla V100-SXM2-32GB] Data Size per transfer: 1024 MB, Iterations: 100
[Tesla V100-SXM2-32GB] Host to Device (H2D) Bandwidth: 11.55 GB/s
[Tesla V100-SXM2-32GB] Device to Host (D2H) Bandwidth: 12.27 GB/s

=== Multi-GPU NVLink/P2P Bandwidth Evaluation ===
Found 4 GPUs. Evaluating pairwise P2P bandwidth (Data Size: 1024 MB, Iterations: 100)...

[Tesla V100-SXM2-32GB (GPU 0)] -> [Tesla V100-SXM2-32GB (GPU 1)] Bandwidth: 22.59 GB/s
[Tesla V100-SXM2-32GB (GPU 0)] -> [Tesla V100-SXM2-32GB (GPU 2)] Bandwidth: 9.16 GB/s
[Tesla V100-SXM2-32GB (GPU 0)] -> [Tesla V100-SXM2-32GB (GPU 3)] Bandwidth: 45.15 GB/s
[Tesla V100-SXM2-32GB (GPU 1)] -> [Tesla V100-SXM2-32GB (GPU 0)] Bandwidth: 22.59 GB/s
[Tesla V100-SXM2-32GB (GPU 1)] -> [Tesla V100-SXM2-32GB (GPU 2)] Bandwidth: 45.15 GB/s
[Tesla V100-SXM2-32GB (GPU 1)] -> [Tesla V100-SXM2-32GB (GPU 3)] Bandwidth: 8.85 GB/s
[Tesla V100-SXM2-32GB (GPU 2)] -> [Tesla V100-SXM2-32GB (GPU 0)] Bandwidth: 8.72 GB/s
[Tesla V100-SXM2-32GB (GPU 2)] -> [Tesla V100-SXM2-32GB (GPU 1)] Bandwidth: 45.15 GB/s
[Tesla V100-SXM2-32GB (GPU 2)] -> [Tesla V100-SXM2-32GB (GPU 3)] Bandwidth: 22.59 GB/s
[Tesla V100-SXM2-32GB (GPU 3)] -> [Tesla V100-SXM2-32GB (GPU 0)] Bandwidth: 45.14 GB/s
[Tesla V100-SXM2-32GB (GPU 3)] -> [Tesla V100-SXM2-32GB (GPU 1)] Bandwidth: 8.72 GB/s
[Tesla V100-SXM2-32GB (GPU 3)] -> [Tesla V100-SXM2-32GB (GPU 2)] Bandwidth: 22.59 GB/s
```
