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
Device       : NVIDIA H20
Capability   : (9, 0)
Torch version: 2.6.0+cu124

[NVIDIA H20] [torch.float32] Matrix Size: 8192x8192, TFLOPS: 24.15
[NVIDIA H20] [torch.float16] Matrix Size: 8192x8192, TFLOPS: 142.08
[NVIDIA H20] [torch.bfloat16] Matrix Size: 8192x8192, TFLOPS: 139.68
[NVIDIA H20] [torch.float8_e4m3fn] Matrix Size: 8192x8192, TFLOPS: 281.01
[NVIDIA H20] [torch.float8_e5m2 x torch.float8_e4m3fn] Matrix Size: 8192x8192, TFLOPS: 281.01
[torch.float4_e2m1fn_x2] Skipped. Reason: dtype not exposed in this PyTorch version.
[NVIDIA H20] [INT8 Tensor Core] Matrix Size: 8192x8192, TOPS: 56.07

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

### L20
```bash
=== GPU FLOPS Evaluation ===
Device       : NVIDIA L20
Capability   : (8, 9)
Torch version: 2.7.0a0+git3cc1095.aml

[NVIDIA L20] [torch.float32] Matrix Size: 8192x8192, TFLOPS: 35.99
[NVIDIA L20] [torch.float16] Matrix Size: 8192x8192, TFLOPS: 108.42
[NVIDIA L20] [torch.bfloat16] Matrix Size: 8192x8192, TFLOPS: 114.34
[NVIDIA L20] [torch.float8_e4m3fn] Matrix Size: 8192x8192, TFLOPS: 216.70
[NVIDIA L20] [torch.float8_e5m2 x torch.float8_e4m3fn] Matrix Size: 8192x8192, TFLOPS: 219.52
[torch.float4_e2m1fn_x2] Skipped. Reason: dtype not exposed in this PyTorch version.
[NVIDIA L20] [INT8 Tensor Core] Matrix Size: 8192x8192, TOPS: 109.11

=== CPU-GPU & GPU-GPU Memory Bandwidth Evaluation ===
[NVIDIA L20] Data Size per transfer: 1024 MB, Iterations: 100
[NVIDIA L20] Host to Device (H2D) Bandwidth: 25.12 GB/s
[NVIDIA L20] Device to Host (D2H) Bandwidth: 18.77 GB/s
[NVIDIA L20] Device to Device (D2D) Bandwidth: 608.26 GB/s
```

### A30
```bash
=== GPU FLOPS Evaluation ===
Device       : NVIDIA A30
Capability   : (8, 0)
Torch version: 2.7.1+cu126

[NVIDIA A30] [torch.float32] Matrix Size: 8192x8192, TFLOPS: 9.30
[NVIDIA A30] [torch.float16] Matrix Size: 8192x8192, TFLOPS: 90.25
[NVIDIA A30] [torch.bfloat16] Matrix Size: 8192x8192, TFLOPS: 115.61
[torch.float8_e4m3fn] Skipped. Reason: compute capability (8, 0) < 8.9 (needs Ada/Hopper).
[torch.float8_e5m2] Skipped. Reason: compute capability (8, 0) < 8.9 (needs Ada/Hopper).
[torch.float4_e2m1fn_x2] Skipped. Reason: dtype not exposed in this PyTorch version.
[NVIDIA A30] [INT8 Tensor Core] Matrix Size: 8192x8192, TOPS: 37.00

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
Device       : Tesla V100-SXM2-32GB
Capability   : (7, 0)
Torch version: 2.6.0+cu124

[Tesla V100-SXM2-32GB] [torch.float32] Matrix Size: 8192x8192, TFLOPS: 13.87
[Tesla V100-SXM2-32GB] [torch.float16] Matrix Size: 8192x8192, TFLOPS: 92.81
[Tesla V100-SXM2-32GB] [torch.bfloat16] Matrix Size: 8192x8192, TFLOPS: 10.07
[torch.float8_e4m3fn] Skipped. Reason: compute capability (7, 0) < 8.9 (needs Ada/Hopper).
[torch.float8_e5m2] Skipped. Reason: compute capability (7, 0) < 8.9 (needs Ada/Hopper).
[torch.float4_e2m1fn_x2] Skipped. Reason: dtype not exposed in this PyTorch version.
[INT8 Tensor Core] Skipped. Reason: Tesla V100-SXM2-32GB compute capability (7, 0) is lower than 7.5.

=== CPU-GPU & GPU-GPU Memory Bandwidth Evaluation ===
[Tesla V100-SXM2-32GB] Data Size per transfer: 1024 MB, Iterations: 100
[Tesla V100-SXM2-32GB] Host to Device (H2D) Bandwidth: 11.56 GB/s
[Tesla V100-SXM2-32GB] Device to Host (D2H) Bandwidth: 12.26 GB/s
[Tesla V100-SXM2-32GB] Device to Device (D2D) Bandwidth: 727.44 GB/s

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


## Bench Device Results
  
### MLU580
```bash
================================================================
  设备检测信息 (Device Info)
================================================================
  Python       : 3.11.10
  PyTorch      : 2.7.0a0+gite0d27cd.aml
  torch_mlu    : 已安装
  选定后端     : Cambricon MLU  (device='mlu')
  设备数量     : 1
----------------------------------------------------------------
  [卡 0]
    型号       : MLU580-X6
    显存总量   : 47.28 GiB
    计算核簇   : 16
    显存占用   : 0 MiB / 47.28 GiB  (可用 47.28 GiB)
================================================================

[1] 浮点峰值算力  (GEMM 8192x8192, iters=50)
----------------------------------------------------------------
  dtype           TFLOPS
----------------------------------------------------------------
  fp32             34.46
  fp16            131.20
  bf16            140.17

[2] 显存带宽  (block=512 MB, iters=50)
----------------------------------------------------------------
  H2D (Host->Device, pinned) :      50.01 GB/s
  D2D (Device->Device)       :    1175.35 GB/s
```

### V100
```bash
================================================================
  设备检测信息 (Device Info)
================================================================
  Python       : 3.10.21
  PyTorch      : 2.4.0+cu121
  torch_mlu    : 未安装
  选定后端     : NVIDIA CUDA  (device='cuda')
  设备数量     : 1
----------------------------------------------------------------
  [卡 0]
    型号       : Tesla V100-SXM2-32GB
    算力(SM)   : 7.0
    多处理器数 : 80
    显存总量   : 31.74 GiB
    显存占用   : 310 MiB / 31.74 GiB  (可用 31.44 GiB)
================================================================

[1] 浮点峰值算力  (GEMM 8192x8192, iters=50)
----------------------------------------------------------------
  dtype           TFLOPS
----------------------------------------------------------------
  fp32             13.78
  fp16             89.43
  bf16             10.13

[2] 显存带宽  (block=512 MB, iters=50)
----------------------------------------------------------------
  H2D (Host->Device, pinned) :      12.41 GB/s
  D2D (Device->Device)       :     801.65 GB/s
```

### T4
```bash
================================================================
  设备检测信息 (Device Info)
================================================================
  Python       : 3.11.10
  PyTorch      : 2.7.0a0+git3cc1095.aml
  torch_mlu    : 未安装
  选定后端     : NVIDIA CUDA  (device='cuda')
  设备数量     : 1
----------------------------------------------------------------
  [卡 0]
    型号       : Tesla T4
    算力(SM)   : 7.5
    多处理器数 : 40
    显存总量   : 14.75 GiB
    显存占用   : 103 MiB / 14.75 GiB  (可用 14.65 GiB)
================================================================

[1] 浮点峰值算力  (GEMM 8192x8192, iters=50)
----------------------------------------------------------------
  dtype           TFLOPS
----------------------------------------------------------------
  fp32              3.76
  fp16             22.66
  bf16              2.06

[2] 显存带宽  (block=512 MB, iters=50)
----------------------------------------------------------------
  H2D (Host->Device, pinned) :      12.41 GB/s
  D2D (Device->Device)       :     233.59 GB/s
```

### L20
```bash
================================================================
  设备检测信息 (Device Info)
================================================================
  Python       : 3.11.10
  PyTorch      : 2.7.0a0+git3cc1095.aml
  torch_mlu    : 未安装
  选定后端     : NVIDIA CUDA  (device='cuda')
  设备数量     : 1
----------------------------------------------------------------
  [卡 0]
    型号       : NVIDIA L20
    算力(SM)   : 8.9
    多处理器数 : 92
    显存总量   : 44.53 GiB
    显存占用   : 290 MiB / 44.53 GiB  (可用 44.24 GiB)
================================================================

[1] 浮点峰值算力  (GEMM 8192x8192, iters=50)
----------------------------------------------------------------
  dtype           TFLOPS
----------------------------------------------------------------
  fp32             36.28
  fp16            109.70
  bf16            114.71

[2] 显存带宽  (block=512 MB, iters=50)
----------------------------------------------------------------
  H2D (Host->Device, pinned) :      26.97 GB/s
  D2D (Device->Device)       :     653.23 GB/s
```

### H20
```bash
================================================================
  设备检测信息 (Device Info)
================================================================
  Python       : 3.11.10
  PyTorch      : 2.7.0a0+git3cc1095.aml
  torch_mlu    : 未安装
  选定后端     : NVIDIA CUDA  (device='cuda')
  设备数量     : 1
----------------------------------------------------------------
  [卡 0]
    型号       : NVIDIA H20
    算力(SM)   : 9.0
    多处理器数 : 78
    显存总量   : 95.00 GiB
    显存占用   : 325 MiB / 95.00 GiB  (可用 94.69 GiB)
================================================================

[1] 浮点峰值算力  (GEMM 8192x8192, iters=50)
----------------------------------------------------------------
  dtype           TFLOPS
----------------------------------------------------------------
  fp32             31.99
  fp16            136.75
  bf16            137.05

[2] 显存带宽  (block=512 MB, iters=50)
----------------------------------------------------------------
  H2D (Host->Device, pinned) :      53.89 GB/s
  D2D (Device->Device)       :    3260.64 GB/s
```
