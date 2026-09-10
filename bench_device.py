#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备性能评测脚本：浮点峰值算力 + 显存带宽 (H2D / D2D)
兼容寒武纪 MLU (torch_mlu) 与 英伟达 CUDA，自动检测后端。

用法:
    python bench_device.py                 # 默认参数
    python bench_device.py --dtypes fp32,tf32,fp16,bf16
    python bench_device.py --gemm-size 8192 --iters 50 --bw-mb 512
"""
import argparse
import time
import sys

import torch

# ---- 尝试导入寒武纪扩展；失败则忽略（走 CUDA / CPU 分支） ----
try:
    import torch_mlu  # noqa: F401
    _HAS_MLU = True
except Exception:
    _HAS_MLU = False


# ------------------------- 设备后端探测 -------------------------
def detect_backend():
    """返回 (device_str, backend_name, sync_fn, empty_cache_fn)。"""
    if _HAS_MLU and hasattr(torch, "mlu") and torch.mlu.is_available():
        return (
            "mlu",
            "Cambricon MLU",
            torch.mlu.synchronize,
            getattr(torch.mlu, "empty_cache", lambda: None),
        )
    if torch.cuda.is_available():
        return (
            "cuda",
            "NVIDIA CUDA",
            torch.cuda.synchronize,
            torch.cuda.empty_cache,
        )
    return ("cpu", "CPU (no accelerator found)", lambda: None, lambda: None)


DEVICE, BACKEND, SYNC, EMPTY_CACHE = detect_backend()


def device_name():
    try:
        if DEVICE == "mlu":
            return torch.mlu.get_device_name(0)
        if DEVICE == "cuda":
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _fmt_mb(nbytes):
    """字节数转成人类可读 (MiB / GiB)。"""
    try:
        gib = nbytes / (1024 ** 3)
        if gib >= 1:
            return f"{gib:.2f} GiB"
        return f"{nbytes / (1024 ** 2):.0f} MiB"
    except Exception:
        return "N/A"


def print_device_info():
    """在评测开始前打印检测到的设备详细信息。"""
    print("=" * 64)
    print("  设备检测信息 (Device Info)")
    print("=" * 64)
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  torch_mlu    : {'已安装' if _HAS_MLU else '未安装'}")
    print(f"  选定后端     : {BACKEND}  (device='{DEVICE}')")

    if DEVICE == "cpu":
        print("  加速卡       : 未检测到 MLU / CUDA 设备")
        print("=" * 64)
        return

    # ---- 设备数量 ----
    try:
        if DEVICE == "mlu":
            count = torch.mlu.device_count()
        else:
            count = torch.cuda.device_count()
    except Exception:
        count = 1
    print(f"  设备数量     : {count}")

    # ---- 逐卡详情 ----
    for idx in range(count):
        print("-" * 64)
        print(f"  [卡 {idx}]")
        # 型号
        try:
            if DEVICE == "mlu":
                nm = torch.mlu.get_device_name(idx)
            else:
                nm = torch.cuda.get_device_name(idx)
        except Exception:
            nm = "unknown"
        print(f"    型号       : {nm}")

        # CUDA 专有：算力 (compute capability)
        if DEVICE == "cuda":
            try:
                cap = torch.cuda.get_device_capability(idx)
                print(f"    算力(SM)   : {cap[0]}.{cap[1]}")
            except Exception:
                pass
            try:
                props = torch.cuda.get_device_properties(idx)
                print(f"    多处理器数 : {props.multi_processor_count}")
                print(f"    显存总量   : {_fmt_mb(props.total_memory)}")
            except Exception:
                pass

        # MLU 专有：属性(不同版本字段可能不同, 容错读取)
        if DEVICE == "mlu":
            try:
                props = torch.mlu.get_device_properties(idx)
                total = getattr(props, "total_memory", None)
                if total:
                    print(f"    显存总量   : {_fmt_mb(total)}")
                mc = getattr(props, "multi_processor_count", None)
                if mc:
                    print(f"    计算核簇   : {mc}")
            except Exception:
                pass

        # ---- 显存占用 (mem_get_info: (free, total)) ----
        try:
            if DEVICE == "mlu":
                free, total = torch.mlu.mem_get_info(idx)
            else:
                free, total = torch.cuda.mem_get_info(idx)
            used = total - free
            print(f"    显存占用   : {_fmt_mb(used)} / {_fmt_mb(total)}"
                  f"  (可用 {_fmt_mb(free)})")
        except Exception:
            pass

    print("=" * 64)


# ------------------------- 数据类型映射 -------------------------
DTYPE_MAP = {
    "fp32": torch.float32,
    "tf32": torch.float32,   # 用 float32 张量 + 开启 TF32 后端
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _set_tf32(enable: bool):
    """仅对 CUDA 有效；MLU 上此开关无副作用。"""
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
    except Exception:
        pass


# ------------------------- 浮点峰值 (GEMM) -------------------------
def bench_gemm(dtype_key: str, n: int, iters: int, warmup: int):
    """
    用方阵矩阵乘 C = A @ B 估算峰值算力。
    单次 GEMM 的浮点运算量 ~ 2 * n^3 (FLOPs)。
    返回 TFLOPS。
    """
    torch_dtype = DTYPE_MAP[dtype_key]
    _set_tf32(dtype_key == "tf32")

    try:
        a = torch.randn(n, n, device=DEVICE, dtype=torch_dtype)
        b = torch.randn(n, n, device=DEVICE, dtype=torch_dtype)
    except Exception as e:
        return None, f"分配/初始化失败: {e}"

    # 预热
    try:
        for _ in range(warmup):
            c = a @ b
        SYNC()
    except Exception as e:
        return None, f"warmup 失败(可能不支持该 dtype): {e}"

    # 计时
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    SYNC()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    flops = 2.0 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12

    del a, b, c
    EMPTY_CACHE()
    return tflops, None


# ------------------------- 显存带宽 -------------------------
def bench_h2d(nbytes: int, iters: int, warmup: int):
    """Host(pinned) -> Device 拷贝带宽 (GB/s)。"""
    n_elem = nbytes // 4  # float32
    try:
        host = torch.empty(n_elem, dtype=torch.float32).pin_memory()
    except Exception:
        host = torch.empty(n_elem, dtype=torch.float32)  # 退化为非 pinned
    dev = torch.empty(n_elem, dtype=torch.float32, device=DEVICE)

    for _ in range(warmup):
        dev.copy_(host, non_blocking=True)
    SYNC()

    t0 = time.perf_counter()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
    SYNC()
    t1 = time.perf_counter()

    gbps = (nbytes * iters) / (t1 - t0) / 1e9
    del host, dev
    EMPTY_CACHE()
    return gbps


def bench_d2d(nbytes: int, iters: int, warmup: int):
    """Device -> Device 拷贝带宽 (GB/s)。读+写各计一次，故 *2。"""
    n_elem = nbytes // 4
    src = torch.empty(n_elem, dtype=torch.float32, device=DEVICE)
    dst = torch.empty(n_elem, dtype=torch.float32, device=DEVICE)

    for _ in range(warmup):
        dst.copy_(src)
    SYNC()

    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    SYNC()
    t1 = time.perf_counter()

    # 拷贝需读 src + 写 dst，实际访存 = 2 * nbytes
    gbps = (2 * nbytes * iters) / (t1 - t0) / 1e9
    del src, dst
    EMPTY_CACHE()
    return gbps


# ------------------------------ main ------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLU/CUDA 浮点峰值与显存带宽评测")
    parser.add_argument("--dtypes", default="fp32,fp16,bf16",
                        help="逗号分隔: fp32,tf32,fp16,bf16")
    parser.add_argument("--gemm-size", type=int, default=8192, help="GEMM 方阵边长 n")
    parser.add_argument("--iters", type=int, default=50, help="GEMM 计时迭代次数")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--bw-mb", type=int, default=512, help="带宽测试单块大小(MB)")
    parser.add_argument("--bw-iters", type=int, default=50, help="带宽计时迭代次数")
    args = parser.parse_args()

    # ---- 先打印设备检测信息 ----
    print_device_info()

    if DEVICE == "cpu":
        print("\n[警告] 未检测到 MLU / CUDA 加速卡，结果仅供参考。\n")

    # ---- 浮点峰值 ----
    print(f"\n[1] 浮点峰值算力  (GEMM {args.gemm_size}x{args.gemm_size}, "
          f"iters={args.iters})")
    print("-" * 64)
    print(f"  {'dtype':<8}{'TFLOPS':>14}")
    print("-" * 64)
    for dk in [d.strip().lower() for d in args.dtypes.split(",") if d.strip()]:
        if dk not in DTYPE_MAP:
            print(f"  {dk:<8}{'跳过(未知dtype)':>14}")
            continue
        tflops, err = bench_gemm(dk, args.gemm_size, args.iters, args.warmup)
        if err:
            print(f"  {dk:<8}{'N/A':>14}   ({err})")
        else:
            print(f"  {dk:<8}{tflops:>14.2f}")

    # ---- 显存带宽 ----
    nbytes = args.bw_mb * 1024 * 1024
    print(f"\n[2] 显存带宽  (block={args.bw_mb} MB, iters={args.bw_iters})")
    print("-" * 64)
    if DEVICE == "cpu":
        print("  CPU 模式跳过带宽测试。")
    else:
        try:
            h2d = bench_h2d(nbytes, args.bw_iters, args.warmup)
            print(f"  H2D (Host->Device, pinned) : {h2d:>10.2f} GB/s")
        except Exception as e:
            print(f"  H2D : N/A ({e})")
        try:
            d2d = bench_d2d(nbytes, args.bw_iters, args.warmup)
            print(f"  D2D (Device->Device)       : {d2d:>10.2f} GB/s")
        except Exception as e:
            print(f"  D2D : N/A ({e})")

    print("\n完成。\n")


if __name__ == "__main__":
    main()
