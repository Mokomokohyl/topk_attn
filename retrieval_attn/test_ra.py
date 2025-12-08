import torch
import math
from torch.utils.cpp_extension import load
import os

# Constants from the kernel
NUM_HEADS = 32
HEAD_DIM = 128
SEQ_LEN = 8192
CLUSTER_SIZE = 5
QK_BLOCKS_PER_CLUSTER = 4

# Load the extension
print("Compiling and loading extension...")
ra_ops = load(
    name='retrieval_attention_cpp',
    sources=['bind.cpp', 'retrieval_attention.cu'],
    extra_cuda_cflags=['-O3', '-arch=sm_90a', '--ptxas-options=-v', '-lineinfo'],
    verbose=True
)
print("Extension loaded.")

def reference_retrieval_attention(q, k, v, topk_per_block):
    # q: [B, NUM_HEADS, HEAD_DIM]
    # k: [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    # v: [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    
    batch_size = q.size(0)
    
    # Reshape Q to [B, H, 1, D]
    q_h = q.unsqueeze(2) # [B, H, 1, D]
    
    # Reshape K, V to [1, H, N, D] (broadcast over batch)
    k_h = k.permute(1, 0, 2).unsqueeze(0) # [1, H, N, D]
    v_h = v.permute(1, 0, 2).unsqueeze(0) # [1, H, N, D]
    
    # Split into blocks
    keys_per_block = SEQ_LEN // QK_BLOCKS_PER_CLUSTER
    
    all_topk_scores = []
    all_topk_indices = []
    
    for i in range(QK_BLOCKS_PER_CLUSTER):
        start_idx = i * keys_per_block
        end_idx = start_idx + keys_per_block
        
        # Get block K
        k_block = k_h[:, :, start_idx:end_idx, :] # [1, H, BlockSize, D]
        
        # Compute scores: Q @ K.T
        # [B, H, 1, D] @ [1, H, D, BlockSize] -> [B, H, 1, BlockSize]
        scores = torch.matmul(q_h, k_block.transpose(2, 3))
        scores = scores.squeeze(2) # [B, H, BlockSize]
        
        # Select TopK
        topk_scores, topk_indices = torch.topk(scores, topk_per_block, dim=-1)
        
        # Adjust indices to global
        topk_indices = topk_indices + start_idx
        
        all_topk_scores.append(topk_scores)
        all_topk_indices.append(topk_indices)
        
    # Concatenate all topk
    # [B, H, TotalTopK]
    gathered_scores = torch.cat(all_topk_scores, dim=2)
    gathered_indices = torch.cat(all_topk_indices, dim=2)
    
    # Softmax
    # Using raw exp to match kernel
    exp_scores = torch.exp(gathered_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    probs = exp_scores / sum_exp
    
    # Gather Values
    # gathered_indices: [B, H, TotalTopK]
    # v_h: [1, H, N, D]
    
    # We need to gather vectors from v_h using gathered_indices
    # Expand indices to [B, H, TotalTopK, D]
    indices_expanded = gathered_indices.unsqueeze(-1).expand(-1, -1, -1, HEAD_DIM)
    
    # Expand v_h to [B, H, N, D]
    v_h_expanded = v_h.expand(batch_size, -1, -1, -1)
    
    v_selected = torch.gather(v_h_expanded, 2, indices_expanded.long()) # [B, H, TotalTopK, D]
    
    # Weighted sum
    # probs: [B, H, TotalTopK] -> [B, H, 1, TotalTopK]
    probs_expanded = probs.unsqueeze(2)
    
    # [B, H, 1, TotalTopK] @ [B, H, TotalTopK, D] -> [B, H, 1, D]
    output = torch.matmul(probs_expanded, v_selected)
    
    return output.squeeze(2) # [B, H, D]

def run_test(topk=128, batch_size=4):
    print(f"\nTesting with TOPK={topk}, BATCH_SIZE={batch_size}...")
    torch.manual_seed(0)
    device = torch.device("cuda")
    
    # Initialize tensors
    q = torch.randn(batch_size, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    
    output_kernel = torch.zeros(batch_size, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    
    # Run Kernel
    if topk == 32:
        print("Running DSM Kernel...")
        ra_ops.retrieval_attention_32(q, k, v, output_kernel)
    elif topk == 128:
        print("Running DSM Kernel...")
        ra_ops.retrieval_attention_128(q, k, v, output_kernel)
    elif topk == 256:
        print("Running DSM Kernel...")
        ra_ops.retrieval_attention_256(q, k, v, output_kernel)
    elif topk == 512:
        print("Running DSM Kernel...")
        ra_ops.retrieval_attention_512(q, k, v, output_kernel)
    elif topk == 1024:
        print("Running DSM Kernel...")
        ra_ops.retrieval_attention_1024(q, k, v, output_kernel)
        
    # Run Reference
    output_ref = reference_retrieval_attention(q, k, v, topk)
    
    # Compare
    # output_kernel is [B, H, D], output_ref is [B, H, D]
    
    diff = (output_kernel - output_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"DSM Kernel - Max diff: {max_diff}")
    print(f"DSM Kernel - Mean diff: {mean_diff}")
    
    if max_diff < 1e-2:
        print("DSM Kernel: PASSED")
    else:
        print("DSM Kernel: FAILED")

    # Run Global Kernel
    output_kernel_global = torch.zeros(batch_size, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    if topk == 32:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_32(q, k, v, output_kernel_global)
    elif topk == 128:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_128(q, k, v, output_kernel_global)
    elif topk == 256:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_256(q, k, v, output_kernel_global)
    elif topk == 512:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_512(q, k, v, output_kernel_global)
    elif topk == 1024:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_1024(q, k, v, output_kernel_global)

    diff_global = (output_kernel_global - output_ref).abs()
    max_diff_global = diff_global.max().item()
    mean_diff_global = diff_global.mean().item()

    print(f"Global Kernel - Max diff: {max_diff_global}")
    print(f"Global Kernel - Mean diff: {mean_diff_global}")

    if max_diff_global < 1e-2:
        print("Global Kernel: PASSED")
    else:
        print("Global Kernel: FAILED")

    # Run Pipelined Kernel
    output_kernel_pipelined = torch.zeros(batch_size, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    if topk == 32:
        print("Running Pipelined Kernel...")
        ra_ops.retrieval_attention_pipelined_32(q, k, v, output_kernel_pipelined)
    elif topk == 128:
        print("Running Pipelined Kernel...")
        ra_ops.retrieval_attention_pipelined_128(q, k, v, output_kernel_pipelined)
    elif topk == 256:
        print("Running Pipelined Kernel...")
        ra_ops.retrieval_attention_pipelined_256(q, k, v, output_kernel_pipelined)
    elif topk == 512:
        print("Running Pipelined Kernel...")
        ra_ops.retrieval_attention_pipelined_512(q, k, v, output_kernel_pipelined)
    elif topk == 1024:
        print("Running Pipelined Kernel...")
        ra_ops.retrieval_attention_pipelined_1024(q, k, v, output_kernel_pipelined)

    diff_pipelined = (output_kernel_pipelined - output_ref).abs()
    max_diff_pipelined = diff_pipelined.max().item()
    mean_diff_pipelined = diff_pipelined.mean().item()

    print(f"Pipelined Kernel - Max diff: {max_diff_pipelined}")
    print(f"Pipelined Kernel - Mean diff: {mean_diff_pipelined}")

    if max_diff_pipelined < 1e-2:
        print("Pipelined Kernel: PASSED")
    else:
        print("Pipelined Kernel: FAILED")

if __name__ == "__main__":
    run_test(32, batch_size=4)
    run_test(128, batch_size=4)
    run_test(256, batch_size=4)
    run_test(512, batch_size=4)
    run_test(1024, batch_size=4)
