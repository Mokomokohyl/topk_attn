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
    # q: [1, NUM_HEADS, HEAD_DIM]
    # k: [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    # v: [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    
    # Reshape Q to [NUM_HEADS, 1, HEAD_DIM]
    q_h = q.squeeze(0).unsqueeze(1) # [H, 1, D]
    
    # Reshape K, V to [NUM_HEADS, SEQ_LEN, HEAD_DIM]
    k_h = k.permute(1, 0, 2) # [H, N, D]
    v_h = v.permute(1, 0, 2) # [H, N, D]
    
    # Split into blocks
    keys_per_block = SEQ_LEN // QK_BLOCKS_PER_CLUSTER
    
    all_topk_scores = []
    all_topk_indices = []
    
    for i in range(QK_BLOCKS_PER_CLUSTER):
        start_idx = i * keys_per_block
        end_idx = start_idx + keys_per_block
        
        # Get block K
        k_block = k_h[:, start_idx:end_idx, :] # [H, BlockSize, D]
        
        # Compute scores: Q @ K.T
        # [H, 1, D] @ [H, D, BlockSize] -> [H, 1, BlockSize]
        scores = torch.matmul(q_h, k_block.transpose(1, 2))
        scores = scores.squeeze(1) # [H, BlockSize]
        
        # Select TopK
        topk_scores, topk_indices = torch.topk(scores, topk_per_block, dim=-1)
        
        # Adjust indices to global
        topk_indices = topk_indices + start_idx
        
        all_topk_scores.append(topk_scores)
        all_topk_indices.append(topk_indices)
        
    # Concatenate all topk
    # [H, TotalTopK]
    gathered_scores = torch.cat(all_topk_scores, dim=1)
    gathered_indices = torch.cat(all_topk_indices, dim=1)
    
    # Softmax
    # Note: The kernel uses hexp directly without subtracting max, 
    # but for numerical stability in python we usually do.
    # However, to match kernel exactly we might want to check if we should skip max subtraction.
    # The kernel does: exp_val = hexp(s_all_scores[i]); sum += exp_val;
    # So it does NOT subtract max.
    # But float16 exp can easily overflow. 
    # Let's try standard softmax first, if it fails we can try raw exp.
    # Actually, let's try to match the kernel behavior: raw exp.
    
    # probs = torch.softmax(gathered_scores, dim=-1) 
    # Using raw exp to match kernel
    exp_scores = torch.exp(gathered_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    probs = exp_scores / sum_exp
    
    # Gather Values
    # gathered_indices: [H, TotalTopK]
    # v_h: [H, N, D]
    
    # We need to gather vectors from v_h using gathered_indices
    # Expand indices to [H, TotalTopK, D]
    indices_expanded = gathered_indices.unsqueeze(-1).expand(-1, -1, HEAD_DIM)
    v_selected = torch.gather(v_h, 1, indices_expanded.long()) # [H, TotalTopK, D]
    
    # Weighted sum
    # probs: [H, TotalTopK] -> [H, 1, TotalTopK]
    probs_expanded = probs.unsqueeze(1)
    
    # [H, 1, TotalTopK] @ [H, TotalTopK, D] -> [H, 1, D]
    output = torch.matmul(probs_expanded, v_selected)
    
    return output.squeeze(1) # [H, D]

def run_test(topk=128):
    print(f"\nTesting with TOPK={topk}...")
    torch.manual_seed(0)
    device = torch.device("cuda")
    
    # Initialize tensors
    q = torch.randn(1, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    
    output_kernel = torch.zeros(1, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    
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
        
    # Run Reference
    output_ref = reference_retrieval_attention(q, k, v, topk)
    
    # Compare
    # output_kernel is [1, H, D], output_ref is [H, D]
    output_kernel_s = output_kernel.squeeze(0)
    
    diff = (output_kernel_s - output_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"DSM Kernel - Max diff: {max_diff}")
    print(f"DSM Kernel - Mean diff: {mean_diff}")
    
    if max_diff < 1e-2:
        print("DSM Kernel: PASSED")
    else:
        print("DSM Kernel: FAILED")

    # Run Global Kernel
    output_kernel_global = torch.zeros(1, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    if topk == 32:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_32(q, k, v, output_kernel_global)
    elif topk == 128:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_128(q, k, v, output_kernel_global)
    elif topk == 256:
        print("Running Global Kernel...")
        ra_ops.retrieval_attention_global_256(q, k, v, output_kernel_global)

    output_kernel_global_s = output_kernel_global.squeeze(0)
    diff_global = (output_kernel_global_s - output_ref).abs()
    max_diff_global = diff_global.max().item()
    mean_diff_global = diff_global.mean().item()

    print(f"Global Kernel - Max diff: {max_diff_global}")
    print(f"Global Kernel - Mean diff: {mean_diff_global}")

    if max_diff_global < 1e-2:
        print("Global Kernel: PASSED")
    else:
        print("Global Kernel: FAILED")

if __name__ == "__main__":
    run_test(32)
    run_test(128)
    run_test(256)
