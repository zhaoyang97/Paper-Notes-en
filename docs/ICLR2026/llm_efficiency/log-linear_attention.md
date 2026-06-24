---
title: >-
  [Paper Note] Log-Linear Attention
description: >-
  [ICLR 2026][LLM Efficiency][Linear Attention] The authors replace the "fixed-size hidden state" in linear attention with a set of multi-scale hidden states that grow **logarithmically with sequence length**. This maintains matrix-multiply-friendly parallel training ($O(T \log T)$ computation, $O(\log T)$ decoding memory) while pushing the expressivity of linear attention toward that of softmax attention.
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "Linear Attention"
  - "Sub-quadratic Attention"
  - "State Space Models"
  - "Hierarchical Matrices"
  - "Fenwick Tree"
  - "Long Context"
date: 2026-05-08
content_hash: 6e1758cc9cfb71a7
---

# Log-Linear Attention

**Conference**: ICLR 2026  
**arXiv**: [https://openreview.net/forum?id=mOJgZWkXKW](https://openreview.net/forum?id=mOJgZWkXKW)  
**Code**: [https://github.com/HanGuo97/log-linear-attention](https://github.com/HanGuo97/log-linear-attention)  
**Area**: Efficient LLM Architectures / Sequence Modeling  
**Keywords**: Linear Attention, Sub-quadratic Attention, State Space Models, Hierarchical Matrices, Fenwick Tree, Long Context  

## TL;DR
The authors replace the "fixed-size hidden state" in linear attention with a set of multi-scale hidden states that grow **logarithmically with sequence length**. This maintains matrix-multiply-friendly parallel training ($O(T \log T)$ computation, $O(\log T)$ decoding memory) while pushing the expressivity of linear attention toward that of softmax attention.

## Background & Motivation
**Background**: Softmax attention in Transformers offers high accuracy and parallelizability but suffers from $O(T^2)$ computation and $O(T)$ memory bottlenecks for long sequences. Linear attention and State Space Models (e.g., Mamba-2, Gated DeltaNet) rewrite attention as a linear RNN with matrix-valued hidden states, achieving $O(T)$ training and $O(1)$ decoding, while preserving GPU-friendly matmul density through chunkwise parallelism.

**Limitations of Prior Work**: These efficient models are essentially **RNNs**—compressing the entire history into a **fixed-size** hidden state. Regardless of context length, the state capacity remains constant. This is a fundamental flaw for tasks requiring fine-grained memory, such as associative recall within a given context: distant information is either overwritten or blurred.

**Key Challenge**: Softmax attention maintains independent KV pairs for every token ($t$ "bins" of size 1, infinite expressivity but $O(T)$ memory); linear attention squeezes the entire prefix into 1 bin ($O(1)$ memory but limited expressivity). The "middle ground" in this spectrum has remained vacant.

**Goal**: Find an intermediate regime between these extremes that achieves sub-quadratic training and sub-linear decoding while retaining more long-range information than a fixed state.

**Key Insight**: **Use a Fenwick tree to partition the prefix into multiple exponentially growing bins**—using fine-grained small bins for nearby tokens (high-resolution preservation) and coarse-grained large bins for distant tokens (rough summarization). The number of bins $L = O(\log T)$, with each bin maintaining an independent recurrent state. During decoding, the query only needs to attend to $O(\log T)$ states. **Crucially**, this is equivalent to replacing the causal mask $M$ in linear attention with a data-dependent **hierarchical matrix (HODLR-type H-matrix)**, which supports $O(T \log T)$ matmul-rich parallel training.

## Method

### Overall Architecture
The paper establishing a unified perspective: almost all efficient attention mechanisms can be written as $P = A \odot M, O = PV$, where $A$ is an attention-like interaction matrix (e.g., $QK^\top$ for linear attention) and $M$ is a lower-triangular causal mask. **The structure of $M$ determines the algorithmic complexity** (all 1s $\to$ linear attention $O(T)$; 1-semiseparable $\to$ gated Mamba-2; Toeplitz $\to$ long convolution $O(T \log T)$). The innovation of Log-linear attention lies in designing $M$ as a **quasi-H (hierarchical) matrix** $M^H$: during training, it enables $O(T \log T)$ chunkwise parallel scans; during decoding, it corresponds to $O(\log T)$ recurrent states. This is a **general framework** where any linear attention with structured memory and chunkwise parallel primitives can be "lifted" into a log-linear variant.

```mermaid
flowchart TD
    X[Input Sequence Q,K,V] --> P[Fenwick Tree Prefix Binning<br/>Fine close/Coarse far, L=O(log T) Bins]
    P --> S["Independent Recurrent States S^(ℓ)_t per Bin"]
    S --> L["Data-dependent Multi-scale Weights λ^(ℓ)_t<br/>(Linear function of xt)"]
    L --> M["Hierarchical mask M^H (HODLR/quasi-H)"]
    M --> Train["Training: Chunkwise Parallel Scan<br/>O(T log T) Computation"]
    M --> Decode["Decoding: O(log T) States<br/>O(log T) Time & Memory"]
    Train --> Inst["Instantiation: Log-Linear Mamba-2 / Gated DeltaNet<br/>M = M^S ⊙ M^H"]
    Decode --> Inst
```

### Key Designs

**1. Fenwick Tree Binning: Partitioning the "Prefix" into Exponentially Growing Multi-scale Memory.** From a decoding perspective, attention partitions the prefix $[0,t)$ into several bins and summarizes the history in each. This work utilizes the Fenwick tree (binary indexed tree) logic: greedily subtracting the "largest power of 2 in the remaining segment" from the current position, determined by $\mathrm{lssb}(t)$ (the least significant bit of $t$). This results in bins $B^{(\ell)}_t$ of size $2^{\ell-1}$ ($\ell\ge1$), plus a size-1 sentinel bin $B^{(0)}_t$, totaling at most $L=\lceil\log_2 t+1\rceil+1$ disjoint bins. This introduces a natural inductive bias: **nearer tokens are kept at finer granularity, while distant tokens are summarized more coarsely**—neither keeping one for every token (too expensive like softmax) nor compressing all into one (too blurry like linear attention).

**2. Multi-scale States + Data-dependent Scale Weights $\lambda$: Letting the Model Decide Which Time Scale to Focus On.** Each bin maintains an independent recurrent memory $S^{(\ell)}_t \in \mathbb{R}^{d\times d}$. The output at time $t$ is the weighted sum of contributions from each bin:
$$o_t = \sum_{\ell=0}^{L-1} \lambda^{(\ell)}_t\, q_t^\top \sum_{s\in B^{(\ell)}_t} v_s k_s^\top = \sum_{\ell=0}^{L-1} \lambda^{(\ell)}_t\, q_t^\top S^{(\ell)}_t.$$
The non-negative coefficients $\lambda^{(\ell)}_t$ are parameterized as linear functions of the current input $x_t$, allowing the model to adaptively emphasize different time scales. A key property: **when all $\lambda^{(\ell)}_t$ are identical (or linearly dependent across time), log-linear attention degrades to standard linear attention**. Allowing $\lambda$ to vary across levels is essential for capturing multi-scale structures.

**3. Parallel Form of Hierarchical Mask: Rewriting Multi-scale Recurrence as Matmul-rich H-matrix Multiplication.** The recursive form above is conceptually clear but GPU-unfriendly. The authors rewrite it in parallel form $O = (QK^\top \odot M^H)V$, where:
$$M^H_{ts} = \begin{cases}\lambda^{(\ell(t,s))}_t & s\le t \\ 0 & \text{otherwise}\end{cases}$$
Here, $\ell(t,s)$ is the bin level of token $s$ relative to $t$ under Fenwick partitioning. $M^H$ is a lower-triangular **HODLR (Hierarchical Off-Diagonal Low-Rank) matrix**—its diagonal blocks are fine-grained and off-diagonal blocks are low-rank, sitting between a general H-matrix and a semi-separable matrix. The authors term this a quasi-H matrix. This recursive low-rank structure enables $O(T \log T)$ training.

**4. Chunkwise Parallel Scan + Plug-and-Play.** The training algorithm generalizes the classic parallel prefix sum (scan) to a hierarchical setting: running independent chunkwise scans for each memory level, totaling $O(\log T)$ times. Each scan takes $O(T)$ time/memory, resulting in an overall $O(\frac{T\log T}{C})$ complexity. Compared to token-level scans limited by memory bandwidth, the chunked form reorganizes updates into parallel operations. For implementation, the authors keep the original model's $A$ form and simply element-wise multiply its temporal mask with $M^H$, yielding **Log-Linear Mamba-2** ($O=(QK^\top\odot M^S\odot M^H)V$) and **Log-Linear Gated DeltaNet**. The entire chunkwise parallel scan is implemented in Triton, optimized via "level fusion" (fusing 4 levels into one kernel) for acceleration.

## Key Experimental Results

### Main Results
Academic-scale pre-training: 50B tokens, 16K sequence length, 21 layers, hidden size 1536, trained from scratch. Log-linear variants only increase parameters by <3% (Mamba-2) / <0.4% (Gated DeltaNet).

| Model | WikiText ppl ↓ | LAMBADA ppl ↓ | LM-Eval Avg ↑ |
|---|---|---|---|
| Transformer (21L) | 21.56 | 22.14 | 44.0 |
| Transformer (24L, param matched) | 21.13 | 21.17 | 45.6 |
| Hyena (log-linear computation but linear memory) | 29.50 | / | / |
| Mamba-2 | 22.44 | 24.14 | 44.8 |
| **w/ Log-Linear (Ours)** | **22.11** | **21.86** | **44.9** |
| Gated DeltaNet | 21.73 | 19.71 | 45.0 |
| **w/ Log-Linear (Ours)** | **21.45** | **18.09** | **45.5** |

Log-Linear Gated DeltaNet outperforms the linear version on perplexity and all common-sense reasoning benchmarks except one, and exceeds the layer-matched Transformer on all metrics.

### Ablation Study

| Task | Conclusion |
|---|---|
| MQAR (Multi-Query Associative Recall, 64 KV pairs) | Mamba-2 89.6→92.9; Gated DeltaNet 79.0→84.4 (dim=32). Improvements even in models already optimized for recall. |
| NIAH Single/Multi-needle (RULER) | Log-Linear Mamba-2 improved in 8/9 cases; Log-Linear Gated DeltaNet improved in all multi-needle tasks. |
| Per-position loss (Book-3) | Log-linear versions of both models show lower smoothed loss at all positions, indicating better long-range context utilization. |
| Training Throughput (H100, 48 heads, dim 64) | Custom Triton kernels exceed FlashAttention-2 at >8K sequence lengths; Log-Linear Mamba-2 (+MLP) exceeds Transformer throughput at 32K. |

### Key Findings
- **Expressivity gain at almost "zero cost"**: Adding <3% parameters leads to general improvements over fixed-state linear baselines in recall, long-range, and NIAH tasks, without degradation in short-context tasks.
- **Recall-heavy tasks benefit most**: MQAR, NIAH, and per-position loss—tasks testing long-range memory—show the most significant gains, confirming that "fixed state" is the bottleneck for recall.
- **Throughput outpaces attention with length**: As sequences lengthen, the $O(T \log T)$ advantage over $O(T^2)$ becomes pronounced, surpassing FlashAttention-2 after 8K tokens.

## Highlights & Insights
- **Unified Perspective First**: $P=A\odot M$ fits linear attention, Mamba-2, DeltaNet, and long convolutions into one framework, noting that "the structure of $M$ (rather than removing softmax) is the source of efficiency." This framing is highly valuable.
- **Bringing Data Structures to Attention**: Classical numerical algebra tools like Fenwick trees and HODLR hierarchical matrices are elegantly mapped to attention masks. The $O(T \log T)/O(\log T)$ complexities arise naturally from "structure" rather than engineering tricks.
- **Orthogonal to Existing Linear Attention**: It is an "uplift operator" rather than a new model—any linear attention with chunkwise primitives can be lifted to a log-linear version with low implementation cost and high transferability.
- **Elegant Degradation**: When $\lambda$ is synchronized, it degrades to linear attention, making it a strict superset of linear attention with no theoretical downside.

## Limitations & Future Work
- **Still trails behind Transformer**: A significant gap remains relative to Transformers on all benchmarks; log-linear only narrows but does not eliminate the expressivity gap of fixed/finite states.
- **Exploration of $\lambda$ Parameterization**: Due to compute constraints, 700M–800M models were only run once; different parameterizations/hyperparameters for $\lambda$ were not searched. Better $\lambda$ designs might yield further gains.
- **Constant Factors and Kernel Complexity**: While $O(\log T)$ bins are asymptotically superior, naive implementations are slower than Mamba-2 primitives (requiring custom optimizations like level fusion).
- **Limited Scale**: Experiments were conducted at academic scales (50B tokens, <1B parameters); performance at larger scales and longer contexts remains to be verified.

## Related Work & Insights
- **Linear Attention / SSM Lineage**: Katharopoulos Linear Attention, RetNet, Mamba-2, DeltaNet—this paper treats these as special cases of $M$ structures and generalizes them.
- **Long Convolution Models**: Hyena/MultiHyena/Toeplitz networks also use $O(T \log T)$ computation, but memory remains linear, highlighting the unique value of log-linear's "sub-linear memory."
- **Practical Inspirations**: (1) The diagnosis that "fixed state = recall bottleneck" applies to all RNN-style efficient models; multi-scale states are a promising path. (2) Introducing classic data structures (Fenwick trees, H-matrices) into sequence mask design provides a new set of tools between "pure RNN" and "pure attention." (3) The "uplift operator" philosophy—adding orthogonal capabilities to existing efficient architectures—is highly suitable for industrial adoption.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Using Fenwick trees/hierarchical matrices to give "logarithmically growing multi-scale states" a clean parallel training form creates a genuine new intermediate point between linear and softmax attention.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic (MQAR), LM, NIAH, per-position loss, and throughput with consistent findings; however, scale is small, and hyperparameters for $\lambda$ were not fully explored.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative from unified perspective → motivation → method → instantiation is exceptionally clear; the structured comparison in Table 1 and discussion of degradation properties are highlights.
- **Value**: ⭐⭐⭐⭐⭐ A plug-and-play uplift operator with elegant complexity and empirical improvements in long-range recall, offering strong extensibility for the efficient sequence modeling community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention Layer for Training on Outrageously Large Contexts](race_attention_a_strictly_linear-time_attention_layer_for_training_on_outrageous.md)
- [\[ICML 2026\] Dynamic Linear Attention](../../ICML2026/llm_efficiency/dynamic_linear_attention.md)
- [\[ICLR 2026\] Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention for Test-Time Regression](local_linear_attention_an_optimal_interpolation_of_linear_and_softmax_attention_.md)
- [\[ICLR 2026\] FlexLinearAttention: Compiling a Unified Abstraction into Scalable Kernels for Linear Attention](flexlinearattention_compiling_a_unified_abstraction_into_scalable_kernels_for_li.md)
- [\[ICLR 2026\] MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](mhla_restoring_expressivity_of_linear_attention_via_token-level_multi-head.md)

</div>

<!-- RELATED:END -->
