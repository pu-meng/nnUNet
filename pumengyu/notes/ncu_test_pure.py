"""给 NCU 用的干净脚本，不含 torch.profiler"""
import torch
import torch.nn.functional as F

B, H, N, d = 2, 8, 2048, 64
Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)

def naive_attention(Q, K, V):
    S = torch.matmul(Q, K.transpose(-2, -1)) * (d ** -0.5)
    A = torch.softmax(S, dim=-1)
    return torch.matmul(A, V)

def flash_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)

# 预热（NCU 默认跳过预热）
for _ in range(5):
    naive_attention(Q, K, V)
    flash_attention(Q, K, V)
torch.cuda.synchronize()

# NCU 会 capture 这两个 kernel
out1 = naive_attention(Q, K, V)
torch.cuda.synchronize()

out2 = flash_attention(Q, K, V)
torch.cuda.synchronize()

print("done")
