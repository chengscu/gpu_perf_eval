import torch

def measure_bandwidth(size_mb=1024, iterations=100):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    
    # Allocate memory
    # Float32 takes 4 bytes, so the number of elements is size_mb * 1024 * 1024 / 4
    num_elements = (size_mb * 1024 * 1024) // 4
    
    # Enable pin_memory to maximize PCIe transfer bandwidth
    cpu_tensor = torch.randn(num_elements, dtype=torch.float32).pin_memory()
    gpu_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)

    # Warm-up
    for _ in range(5):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        cpu_tensor.copy_(gpu_tensor)
    torch.cuda.synchronize()

    # --- Test Host to Device (H2D) Bandwidth ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    end_event.record()
    torch.cuda.synchronize()

    h2d_time_s = start_event.elapsed_time(end_event) / 1000.0
    h2d_bandwidth = (size_mb * iterations) / h2d_time_s / 1024 # GB/s

    # --- Test Device to Host (D2H) Bandwidth ---
    start_event.record()
    for _ in range(iterations):
        cpu_tensor.copy_(gpu_tensor)
    end_event.record()
    torch.cuda.synchronize()

    d2h_time_s = start_event.elapsed_time(end_event) / 1000.0
    d2h_bandwidth = (size_mb * iterations) / d2h_time_s / 1024 # GB/s

    # --- Test Device to Device (D2D) Bandwidth ---
    gpu_tensor_2 = torch.empty(num_elements, dtype=torch.float32, device=device)
    
    # Warm-up D2D
    for _ in range(5):
        gpu_tensor_2.copy_(gpu_tensor)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iterations):
        gpu_tensor_2.copy_(gpu_tensor)
    end_event.record()
    torch.cuda.synchronize()

    d2d_time_s = start_event.elapsed_time(end_event) / 1000.0
    # D2D involves 1 read and 1 write, so data size is 2x
    d2d_bandwidth = (2 * size_mb * iterations) / d2d_time_s / 1024 # GB/s

    print(f"[{gpu_name}] Data Size per transfer: {size_mb} MB, Iterations: {iterations}")
    print(f"[{gpu_name}] Host to Device (H2D) Bandwidth: {h2d_bandwidth:.2f} GB/s")
    print(f"[{gpu_name}] Device to Host (D2H) Bandwidth: {d2h_bandwidth:.2f} GB/s")
    print(f"[{gpu_name}] Device to Device (D2D) Bandwidth: {d2d_bandwidth:.2f} GB/s")

if __name__ == "__main__":
    print("=== CPU-GPU & GPU-GPU Memory Bandwidth Evaluation ===")
    measure_bandwidth(size_mb=1024, iterations=100)