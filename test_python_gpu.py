import sys
import time

import torch
from torch import nn


def heading(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def main():
    heading("1. PYTORCH INSTALLATION")

    print(f"Python version:       {sys.version.split()[0]}")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"PyTorch CUDA build:   {torch.version.cuda}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    print(f"cuDNN available:      {torch.backends.cudnn.is_available()}")
    print(f"cuDNN version:        {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("\nFAIL: PyTorch cannot access CUDA.")
        sys.exit(1)

    heading("2. GPU INFORMATION")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    print(f"GPU:                  {torch.cuda.get_device_name(device)}")
    print(f"CUDA device count:    {torch.cuda.device_count()}")
    print(f"Compute capability:   {props.major}.{props.minor}")
    print(f"GPU memory:           {props.total_memory / 1024**3:.2f} GiB")

    heading("3. BASIC GPU TENSOR TEST")

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)

    z = x * y

    print(f"x device:             {x.device}")
    print(f"y device:             {y.device}")
    print(f"x * y:                {z.cpu().tolist()}")

    expected = torch.tensor([4.0, 10.0, 18.0])

    assert torch.allclose(z.cpu(), expected)

    print("PASS: GPU tensor arithmetic works.")

    heading("4. CPU/GPU NUMERICAL AGREEMENT")

    torch.manual_seed(12345)

    a_cpu = torch.randn(1024, 1024)
    b_cpu = torch.randn(1024, 1024)

    cpu_result = a_cpu @ b_cpu

    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    gpu_result = a_gpu @ b_gpu
    torch.cuda.synchronize()

    max_error = (cpu_result - gpu_result.cpu()).abs().max().item()
    mean_error = (cpu_result - gpu_result.cpu()).abs().mean().item()

    print(f"Maximum absolute difference: {max_error:.8f}")
    print(f"Mean absolute difference:    {mean_error:.8f}")

    if torch.allclose(cpu_result, gpu_result.cpu(), rtol=1e-3, atol=1e-3):
        print("PASS: CPU and GPU calculations agree.")
    else:
        print("WARNING: Results differ more than expected.")

    heading("5. GPU AUTOGRAD TEST")

    x = torch.randn(2048, 2048, device=device, requires_grad=True)

    loss = (x ** 2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print(f"Loss:                 {loss.item():.6f}")
    print(f"Gradient mean:        {x.grad.mean().item():.8e}")
    print(f"Gradient device:      {x.grad.device}")
    print("PASS: GPU autograd/backpropagation works.")

    del x
    torch.cuda.empty_cache()

    heading("6. NEURAL NETWORK TRAINING TEST")

    torch.manual_seed(12345)

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(4096, 128, device=device)
    targets = torch.randint(0, 10, (4096,), device=device)

    initial_loss = None

    for step in range(10):
        optimizer.zero_grad(set_to_none=True)

        output = model(inputs)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()

        print(f"Step {step + 1:2d}: loss = {loss.item():.6f}")

    final_loss = loss.item()

    print(f"\nInitial loss:         {initial_loss:.6f}")
    print(f"Final loss:           {final_loss:.6f}")
    print(f"Model device:         {next(model.parameters()).device}")

    assert next(model.parameters()).is_cuda
    assert torch.isfinite(loss)

    print("PASS: Neural network forward/backward/update works on GPU.")

    del model, optimizer, inputs, targets
    torch.cuda.empty_cache()

    heading("7. CPU vs GPU MATRIX MULTIPLICATION BENCHMARK")

    # Large enough that GPU acceleration should be clearly visible,
    # while remaining easily within a 16-GB GPU's memory.
    size = 4096
    iterations = 10

    print(f"Matrix size:          {size} x {size}")
    print(f"Iterations:           {iterations}")

    torch.manual_seed(12345)

    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    # CPU benchmark
    start = time.perf_counter()

    for _ in range(iterations):
        c_cpu = a_cpu @ b_cpu

    cpu_time = time.perf_counter() - start

    # GPU copies are performed before timing so PCIe transfer time does not
    # contaminate the compute benchmark.
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    # GPU warm-up
    for _ in range(3):
        c_gpu = a_gpu @ b_gpu

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        c_gpu = a_gpu @ b_gpu

    # CUDA operations are asynchronous, so synchronization is essential
    # before stopping the timer.
    torch.cuda.synchronize()

    gpu_time = time.perf_counter() - start

    speedup = cpu_time / gpu_time

    print(f"\nCPU time:             {cpu_time:.3f} s")
    print(f"GPU time:             {gpu_time:.3f} s")
    print(f"GPU speedup:          {speedup:.2f}x")

    heading("8. CUDA MEMORY")

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"Currently allocated:  {allocated:.1f} MiB")
    print(f"Currently reserved:   {reserved:.1f} MiB")
    print(f"Peak allocated:       {peak:.1f} MiB")

    heading("RESULT")

    print("PASS: PyTorch CUDA installation appears to be functioning correctly.")
    print(f"PyTorch:              {torch.__version__}")
    print(f"CUDA runtime:         {torch.version.cuda}")
    print(f"GPU:                  {torch.cuda.get_device_name(0)}")
    print(f"Benchmark speedup:    {speedup:.2f}x")


if __name__ == "__main__":
    main()