import torch

def measure_flops(dtype, size=8192, iterations=100):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    try:
        # Initialize matrices
        A = torch.randn(size, size, dtype=torch.float32, device=device).to(dtype)
        B = torch.randn(size, size, dtype=torch.float32, device=device).to(dtype)

        # Warm-up and test
        for _ in range(10):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Record time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iterations):
            C = torch.matmul(A, B)
        end_event.record()
        torch.cuda.synchronize()

        # Calculate elapsed time
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0

        # Floating point operations for matrix multiplication: 2 * M * N * K
        total_flops = 2.0 * (size ** 3) * iterations
        tflops = (total_flops / elapsed_time_s) / 1e12

        print(f"[{gpu_name}] [{dtype}] Matrix Size: {size}x{size}, TFLOPS: {tflops:.2f}")
    except Exception as e:
        print(f"[{gpu_name}] [{dtype}] Evaluation failed or not supported on this device/PyTorch version. Error: {e}")

if __name__ == "__main__":
    print("=== GPU FLOPS Evaluation ===")
    dtypes = [torch.float32, torch.float16]
    
    # Check if the current GPU supports bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
        
    for dt in dtypes:
        measure_flops(dt)