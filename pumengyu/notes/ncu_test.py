"""
Attention 对比：普通 vs Flash Attention
用 torch.profiler 查看 kernel 时间，用 torch.cuda.Event 精确计时
"""
import torch
import torch.nn.functional as F
import time

B, H, N, d = 2, 8, 2048, 64
Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)

def naive_attention(Q, K, V):
    scale = d ** -0.5
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    A = torch.softmax(S, dim=-1)
    return torch.matmul(A, V)

def flash_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)

# 预热
for _ in range(10):
    naive_attention(Q, K, V)
    flash_attention(Q, K, V)
torch.cuda.synchronize()

# ---- 精确计时 ----
REPEAT = 100

def benchmark(fn, name):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        fn(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / REPEAT
    print(f"{name:25s}  {ms:.3f} ms/iter")
    return ms

t1 = benchmark(naive_attention, "naive attention")
t2 = benchmark(flash_attention, "flash attention")
print(f"\nspeedup: {t1/t2:.2f}x")

# ---- torch.profiler：看具体 kernel 名字和时间 ----
print("\n=== torch.profiler (naive) ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(3):
        naive_attention(Q, K, V)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))

print("\n=== torch.profiler (flash) ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(3):
        flash_attention(Q, K, V)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
