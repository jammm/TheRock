import torch
import time
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend

###############################################################################
# Check for GPU
###############################################################################
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not available. Please run on a CUDA-enabled device.")

device = torch.device("cuda")
torch.cuda.init()  # Initialize CUDA context (optional, helps measure baseline)

###############################################################################
# Helper function for measuring one run
###############################################################################
def measure_op(op_func, warmup=3, total_runs=10):
    """
    op_func: a callable that runs the operation (including memory measurement)
             and returns (time_ms, peak_mem_MB, gflops_s).
    warmup: number of warm-up runs to discard.
    total_runs: total runs to do, including warmup.
    Returns: average_time_ms, average_peak_mem_MB, average_gflops_s over the runs after warm-up.
    """
    times = []
    mems = []
    flops = []

    for run_idx in range(total_runs):
        # Reset peak memory stats at the start of each run
        torch.cuda.reset_peak_memory_stats(device)
        
        t_ms, peak_mb, gf_s = op_func()
        
        if run_idx >= warmup:
            times.append(t_ms)
            mems.append(peak_mb)
            flops.append(gf_s)

    avg_time_ms = sum(times) / len(times)
    avg_mem_mb = sum(mems) / len(mems)
    avg_gf_s = sum(flops) / len(flops)
    return avg_time_ms, avg_mem_mb, avg_gf_s


###############################################################################
# 1) Define the Scaled Dot-Product Attention test
###############################################################################
def run_sdpa():
    # Configuration
    B, heads = 1, 8
    L = 8192
    E = 128
    S = L

    # Create random Q, K, V in half precision
    # We place them inside the function so each run re-allocates 
    # new memory (to measure peak memory usage properly).
    q = torch.randn(B, heads, L, E, device=device, dtype=torch.float16)
    k = torch.randn(B, heads, S, E, device=device, dtype=torch.float16)
    v = torch.randn(B, heads, S, E, device=device, dtype=torch.float16)

    # Start timing
    torch.cuda.synchronize()
    start_time = time.time_ns()

    # Run scaled dot-product attention (Flash Attention backend)
    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = scaled_dot_product_attention(q, k, v)

    # Synchronize & end timing
    torch.cuda.synchronize()
    end_time = time.time_ns()

    # Measure time
    time_ms = (end_time - start_time) / 1e6  # Convert nanoseconds to milliseconds

    # Peak memory usage (MB)
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_mb = peak_mem_bytes / (1024**2)

    # Compute FLOPs for scaled dot-product attention:
    # Q*K^T -> 2 * B * heads * L * S * E
    # Attn*V -> 2 * B * heads * L * S * E
    # Total = 4 * B * heads * L * S * E
    flops = 4.0 * B * heads * L * S * E
    # Convert to GFLOPs/s
    time_s = (end_time - start_time) / 1e9  # Convert nanoseconds to seconds
    flops_s = flops / time_s
    gflops_s = flops_s / 1e9

    return time_ms, peak_mem_mb, gflops_s

###############################################################################
# 2) Define the Conv2d test
###############################################################################
def run_conv2d():
    # Configuration
    N, Cin, Cout = 1, 3, 64
    H, W = 4096, 4096
    kernel_size = 7
    stride = 1
    padding = 3

    # Create Conv2d layer in half precision
    conv = torch.nn.Conv2d(Cin, Cout, kernel_size=kernel_size, 
                           stride=stride, padding=padding).to(device, dtype=torch.float16)
    x = torch.randn(N, Cin, H, W, device=device, dtype=torch.float16)

    # Start timing
    torch.cuda.synchronize()
    start_time = time.time_ns()

    # Forward pass
    y = conv(x)

    # Synchronize & end timing
    torch.cuda.synchronize()
    end_time = time.time_ns()

    # Measure time
    time_ms = (end_time - start_time) / 1e6  # Convert nanoseconds to milliseconds

    # Peak memory usage (MB)
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_mb = peak_mem_bytes / (1024**2)

    # Compute FLOPs for Conv2d:
    # 2 * N * Cout * H_out * W_out * Cin * kH * kW
    N_out, Cout_out, H_out, W_out = y.shape
    Cin_out = conv.in_channels
    kH, kW = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
    flops = 2.0 * N_out * Cout_out * H_out * W_out * Cin_out * kH * kW
    time_s = (end_time - start_time) / 1e9  # Convert nanoseconds to seconds
    flops_s = flops / time_s
    gflops_s = flops_s / 1e9

    return time_ms, peak_mem_mb, gflops_s


###############################################################################
# 3) Define the GEMM test
# Inspired from https://github.com/shisa-ai/mamf-finder/blob/main/mamf-finder.py
###############################################################################
def benchmark_mm(m, n, k, dtype, device, warmup=3, total_runs=10):
    """
    Basic GEMM benchmark. Allocates A, B, C, C_rand; re-randomizes C before each run
    to force writes, but does not emulate L2/LLC cache flushing.
    Supports float32 matmul and torch._scaled_mm for float8_e4m3fn if available.
    Returns average_time_ms, average_peak_mem_MB, average_gflops_s.
    """
    # Prepare output & random target to force writes
    C = torch.empty(m, n, dtype=dtype, device=device).contiguous()
    C_rand = torch.randn(m, n, dtype=dtype, device=device).contiguous()

    A = torch.randn(m, k, dtype=dtype, device=device).contiguous()
    B = torch.randn(k, n, dtype=dtype, device=device).contiguous()
    mm_op = lambda: torch.mm(A, B)

    flops_per_run = 2.0 * m * n * k

    times = []
    mems = []
    gflops = []

    for run_idx in range(total_runs):
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(device)
        # Re-randomize C to force a write
        C.copy_(C_rand)
        # Time the GEMM
        torch.cuda.synchronize()
        start_time = time.time_ns()
        _ = mm_op()
        torch.cuda.synchronize()
        end_time = time.time_ns()

        if run_idx >= warmup:
            t_ms = (end_time - start_time) / 1e6  # Convert nanoseconds to milliseconds
            times.append(t_ms)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            mems.append(peak_mb)
            time_s = (end_time - start_time) / 1e9  # Convert nanoseconds to seconds
            gflops.append((flops_per_run / time_s) / 1e9)

    avg_time = sum(times) / len(times)
    avg_mem  = sum(mems)  / len(mems)
    avg_gflops = sum(gflops) / len(gflops)
    return avg_time, avg_mem, avg_gflops


###############################################################################
# Run the measurements
###############################################################################
print("Benchmarking Scaled Dot-Product Attention (Flash) in FP16 ...")
sdpa_time, sdpa_mem, sdpa_gflops = measure_op(run_sdpa, warmup=3, total_runs=10)
print(f"Average time: {sdpa_time:.2f} ms")
print(f"Average peak memory: {sdpa_mem:.2f} MB")
print(f"Average throughput: {sdpa_gflops:.2f} GFLOP/s\n")

print("Benchmarking Conv2d in FP16 ...")
conv_time, conv_mem, conv_gflops = measure_op(run_conv2d, warmup=3, total_runs=10)
print(f"Average time: {conv_time:.2f} ms")
print(f"Average peak memory: {conv_mem:.2f} MB")
print(f"Average throughput: {conv_gflops:.2f} GFLOP/s\n")

print("Benchmarking GEMM in FP16 ...")
mm_time32, mm_mem32, mm_gflops32 = benchmark_mm(4352, 13568, 3840, torch.bfloat16, device, warmup=3, total_runs=10)
print(f"Average time: {mm_time32:.2f} ms")
print(f"Average peak memory: {mm_mem32:.2f} MB")
print(f"Average throughput: {mm_gflops32:.2f} GFLOP/s")

###############################################################################
# Check PyTorch build configuration
###############################################################################
print("\nPyTorch preferred CUDA BLAS backend:")
print(torch.backends.cuda.preferred_blas_library())

# Start Generation Here
import sys
import platform
import subprocess

print("\nPython environment:")
print(f"Executable     : {sys.executable}")
print(f"Implementation : {platform.python_implementation()}")
print(f"Version        : {platform.python_version()}")
print(f"Build          : {platform.python_build()}")
print(f"Compiler       : {platform.python_compiler()}")
print(f"Platform       : {platform.platform()}")
print(f"Processor      : {platform.processor()}")

print(f"\nPyTorch        : {torch.__version__}")
print(f"HIP available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    hip_version = getattr(torch.version, 'hip', None)
    print(f"HIP version    : {hip_version}")
    print(f"HIP devices    ({torch.cuda.device_count()}):")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"Device [{idx}] - {props.name}")
        print("  Compute Capability       :", f"{props.major}.{props.minor}")
        print("  Multiprocessor Count     :", props.multi_processor_count)
        print("  Total Memory             :", f"{props.total_memory / (1024**2):.2f} MB")
        print("  Integrated               :", bool(props.is_integrated))
        print("  Multi-GPU Board          :", bool(props.is_multi_gpu_board))
        print("  Max Threads / SM         :", props.max_threads_per_multi_processor)
        print("  GCN Arch Name            :", props.gcnArchName)
        print("  Warp Size                :", props.warp_size)
        print("  UUID                     :", props.uuid)
        print("  L2 Cache Size            :", f"{props.L2_cache_size / 1024:.2f} KB")


# Newer PyTorch versions expose the config as a string
config_str = torch._C._show_config()
print("\nPyTorch build arguments:")
for line in config_str.strip().splitlines():
    print("    " + line)
