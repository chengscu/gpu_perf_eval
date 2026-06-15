import torch
import time

def measure_p2p_pair(src_id, dst_id, size_mb, iterations):
    device_src = torch.device(f"cuda:{src_id}")
    device_dst = torch.device(f"cuda:{dst_id}")

    # 分配内存资源 (根据 float32 = 4 bytes 计算元素个数)
    num_elements = (size_mb * 1024 * 1024) // 4
    tensor_src = torch.randn(num_elements, dtype=torch.float32, device=device_src)
    tensor_dst = torch.empty(num_elements, dtype=torch.float32, device=device_dst)

    # 预热兵马 (Warm-up)
    for _ in range(5):
        # 使用 non_blocking=True 确保走底层异步 CUDA 流，模拟真实环境
        tensor_dst.copy_(tensor_src, non_blocking=True)
    torch.cuda.synchronize(device_src)
    torch.cuda.synchronize(device_dst)

    # 核心计时阶段 (弃用极易跨卡错乱的 Event，改用 CPU 高精度钟表)
    start_time = time.perf_counter()

    for _ in range(iterations):
        tensor_dst.copy_(tensor_src, non_blocking=True)

    # 必须下达死命令：令主程序等待两张显卡彻底完成队列中的所有搬运任务
    torch.cuda.synchronize(device_src)
    torch.cuda.synchronize(device_dst)

    end_time = time.perf_counter()

    # 计算耗时与带宽
    p2p_time_s = end_time - start_time
    # 总数据量(GB) = (单次MB * 次数) / 1024
    p2p_bandwidth = (size_mb * iterations / 1024) / p2p_time_s # GB/s
    
    return p2p_bandwidth

def measure_all_p2p_bandwidth(size_mb=1024, iterations=100):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("At least 2 GPUs are required for NVLink P2P evaluation.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Evaluating pairwise P2P bandwidth (Data Size: {size_mb} MB, Iterations: {iterations})...\n")

    bandwidth_matrix = [[0.0] * num_gpus for _ in range(num_gpus)]

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i == j:
                continue
            
            gpu_src_name = torch.cuda.get_device_name(i)
            gpu_dst_name = torch.cuda.get_device_name(j)
            
            bw = measure_p2p_pair(i, j, size_mb, iterations)
            bandwidth_matrix[i][j] = bw
            print(f"[{gpu_src_name} (GPU {i})] -> [{gpu_dst_name} (GPU {j})] Bandwidth: {bw:.2f} GB/s")

if __name__ == "__main__":
    print("=== Multi-GPU NVLink/P2P Bandwidth Evaluation ===")
    measure_all_p2p_bandwidth(size_mb=1024, iterations=100)