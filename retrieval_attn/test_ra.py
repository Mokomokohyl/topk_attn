import torch
import math
from torch.utils.cpp_extension import load
import os
import sys
import re
import tempfile

# Constants from the kernel
NUM_HEADS = 32
HEAD_DIM = 128
SEQ_LEN = 8192
CLUSTER_SIZE = 5
QK_BLOCKS_PER_CLUSTER = 4

import ctypes

# Global results list
results = []

class CaptureOutput:
    def __init__(self):
        self.captured = ""
        self.stdout_fd = sys.stdout.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.temp_fd, self.temp_path = tempfile.mkstemp()

    def __enter__(self):
        sys.stdout.flush()
        # Redirect stdout to the temp file
        os.dup2(self.temp_fd, self.stdout_fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Flush Python's stdout
        sys.stdout.flush()
        
        # Flush C stdout to ensure data is written to file
        try:
            libc = ctypes.CDLL(None)
            c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
            libc.fflush(c_stdout)
        except Exception:
            pass

        # Restore stdout
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.close(self.saved_stdout_fd)
        
        # Close the temp file handle
        os.close(self.temp_fd)
        
        # Read content
        with open(self.temp_path, 'r') as f:
            self.captured = f.read()
            
        # Clean up
        os.remove(self.temp_path)
        
        # Print captured output to real stdout
        print(self.captured, end='')

def parse_and_record(output, topk, kernel_type):
    # Look for lines like: "Retrieval Attention ... - Avg time: X ms, Throughput: Y iter/s"
    # We use a flexible regex to catch different prefixes
    match = re.search(r"Avg time:\s+([\d\.]+)\s+ms,\s+Throughput:\s+([\d\.]+)\s+iter/s", output)
    if match:
        avg_time = float(match.group(1))
        throughput = float(match.group(2))
        results.append({
            "TopK": topk,
            "Kernel": kernel_type,
            "Avg Time (ms)": avg_time,
            "Throughput (iter/s)": throughput
        })


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
    print("Running DSM Kernel...")
    with CaptureOutput() as capturer:
        if topk == 32:
            ra_ops.retrieval_attention_32(q, k, v, output_kernel)
        elif topk == 128:
            ra_ops.retrieval_attention_128(q, k, v, output_kernel)
        elif topk == 256:
            ra_ops.retrieval_attention_256(q, k, v, output_kernel)
        elif topk == 512:
            ra_ops.retrieval_attention_512(q, k, v, output_kernel)
        elif topk == 1024:
            ra_ops.retrieval_attention_1024(q, k, v, output_kernel)
    parse_and_record(capturer.captured, topk, "DSM")
        
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
    print("Running Global Kernel...")
    with CaptureOutput() as capturer:
        if topk == 32:
            ra_ops.retrieval_attention_global_32(q, k, v, output_kernel_global)
        elif topk == 128:
            ra_ops.retrieval_attention_global_128(q, k, v, output_kernel_global)
        elif topk == 256:
            ra_ops.retrieval_attention_global_256(q, k, v, output_kernel_global)
        elif topk == 512:
            ra_ops.retrieval_attention_global_512(q, k, v, output_kernel_global)
        elif topk == 1024:
            ra_ops.retrieval_attention_global_1024(q, k, v, output_kernel_global)
    parse_and_record(capturer.captured, topk, "Global")

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
    print("Running Pipelined Kernel...")
    with CaptureOutput() as capturer:
        if topk == 32:
            ra_ops.retrieval_attention_pipelined_32(q, k, v, output_kernel_pipelined)
        elif topk == 128:
            ra_ops.retrieval_attention_pipelined_128(q, k, v, output_kernel_pipelined)
        elif topk == 256:
            ra_ops.retrieval_attention_pipelined_256(q, k, v, output_kernel_pipelined)
        elif topk == 512:
            ra_ops.retrieval_attention_pipelined_512(q, k, v, output_kernel_pipelined)
        elif topk == 1024:
            ra_ops.retrieval_attention_pipelined_1024(q, k, v, output_kernel_pipelined)
    parse_and_record(capturer.captured, topk, "Pipelined")

    diff_pipelined = (output_kernel_pipelined - output_ref).abs()
    max_diff_pipelined = diff_pipelined.max().item()
    mean_diff_pipelined = diff_pipelined.mean().item()

    print(f"Pipelined Kernel - Max diff: {max_diff_pipelined}")
    print(f"Pipelined Kernel - Mean diff: {mean_diff_pipelined}")

    if max_diff_pipelined < 1e-2:
        print("Pipelined Kernel: PASSED")
    else:
        print("Pipelined Kernel: FAILED")

    # Run Two Kernels (TopK + Reduce)
    output_kernel_two = torch.zeros(batch_size, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    print("Running TopK+Reduce kernels...")
    with CaptureOutput() as capturer:
        if topk == 32:
            ra_ops.retrieval_attention_two_kernels_32(q, k, v, output_kernel_two)
        elif topk == 128:
            ra_ops.retrieval_attention_two_kernels_128(q, k, v, output_kernel_two)
        elif topk == 256:
            ra_ops.retrieval_attention_two_kernels_256(q, k, v, output_kernel_two)
        elif topk == 512:
            ra_ops.retrieval_attention_two_kernels_512(q, k, v, output_kernel_two)
        elif topk == 1024:
            ra_ops.retrieval_attention_two_kernels_1024(q, k, v, output_kernel_two)
    parse_and_record(capturer.captured, topk, "TwoKernels")

    diff_two = (output_kernel_two - output_ref).abs()
    max_diff_two = diff_two.max().item()
    mean_diff_two = diff_two.mean().item()

    print(f"TopK+Reduce kernels - Max diff: {max_diff_two}")
    print(f"TopK+Reduce kernels - Mean diff: {mean_diff_two}")

    if max_diff_two < 1e-2:
        print("Two Kernels: PASSED")
    else:
        print("Two Kernels: FAILED")

    # Run Gather+Baseline
    output_kernel_gather = torch.zeros(batch_size, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    print("Running TopK+Attn...")
    with CaptureOutput() as capturer:
        if topk == 32:
            ra_ops.retrieval_attention_gather_baseline_32(q, k, v, output_kernel_gather)
        elif topk == 128:
            ra_ops.retrieval_attention_gather_baseline_128(q, k, v, output_kernel_gather)
        elif topk == 256:
            ra_ops.retrieval_attention_gather_baseline_256(q, k, v, output_kernel_gather)
        elif topk == 512:
            ra_ops.retrieval_attention_gather_baseline_512(q, k, v, output_kernel_gather)
        elif topk == 1024:
            ra_ops.retrieval_attention_gather_baseline_1024(q, k, v, output_kernel_gather)
    parse_and_record(capturer.captured, topk, "TopK+Attn")

    diff_gather = (output_kernel_gather - output_ref).abs()
    max_diff_gather = diff_gather.max().item()
    mean_diff_gather = diff_gather.mean().item()

    print(f"TopK+Attn - Max diff: {max_diff_gather}")
    print(f"TopK+Attn - Mean diff: {mean_diff_gather}")

    if max_diff_gather < 1e-2:
        print("TopK+Attn: PASSED")
    else:
        print("TopK+Attn: FAILED")

if __name__ == "__main__":
    run_test(32, batch_size=4)
    run_test(128, batch_size=4)
    run_test(256, batch_size=4)
    run_test(512, batch_size=4)
    run_test(1024, batch_size=4)

    print("\nPerformance Results (Batch Size = 4):")
    print(f"| {'TopK':<5} | {'Kernel Type':<10} | {'Avg Time (ms)':<15} | {'Throughput (iter/s)':<20} |")
    print(f"|{'-'*7}|{'-'*12}|{'-'*17}|{'-'*22}|")
    for res in results:
        print(f"| {res['TopK']:<5} | {res['Kernel']:<10} | {res['Avg Time (ms)']:<15.3f} | {res['Throughput (iter/s)']:<20.2f} |")

