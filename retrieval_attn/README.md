# 计算流程
4 : 1 的 `q@k + topk` 和 `softmax + P@V`。分组做topk  
seqlen = 8192，分一组2048四组做gemv + 组内topk，将score和indices传给一个block做softmax + P@V