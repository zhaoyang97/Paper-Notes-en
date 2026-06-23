---
title: >-
  [Paper Note] LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error
description: >-
  [ICLR 2026][Model Compression][Paper Note] LSA directly characterizes the redundancy of Transformer layers using the "minimal linear reconstruction error assuming 50% least important weights are pruned." This approach bypasses Wanda-style weight scoring and manual reduce functions, enabling non-uniform sparsity allocation across layers (and even blocks/projecti
tags:
  - ICLR 2026
  - Model Compression
date: 2026-05-08
content_hash: dfb2058ed5934703
---
# LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=xq3lza5IjN](https://openreview.net/forum?id=xq3lza5IjN)  
**Code**: [https://github.com/BeiYazi0/LSA](https://github.com/BeiYazi0/LSA)  
**Area**: Model Compression / LLM Pruning  
**Keywords**: Layer-wise Sparsity Allocation, Linear Reconstruction Error, Non-uniform Sparsity, Training-free Pruning, Large Language Models

## TL;DR
LSA directly characterizes the redundancy of Transformer layers using the "minimal linear reconstruction error assuming 50% least important weights are pruned." This approach bypasses Wanda-style weight scoring and manual reduce functions, enabling non-uniform sparsity allocation across layers (and even blocks/projections), outperforming methods like OWL and DLP under 70% high sparsity.

## Background & Motivation
**Background**: One-shot pruning methods such as SparseGPT and Wanda compress LLMs without retraining but apply a **uniform sparsity rate** across all layers. Recent studies indicate that contributions of different layers to model performance are imbalanced, leading methods like OWL and DLP to explore **layer-wise non-uniform sparsity allocation**.

**Limitations of Prior Work**: Both OWL and DLP rely on Wanda-style weight scoring (weight magnitude $\times$ activation norm) combined with a manually designed reduce function (mean multiplier for OWL, median for DLP) to estimate layer importance. This introduces two issues: (1) The selection of reduce functions is empirical and lacks theoretical optimality, with OWL's mean multiplier $m$ failing to generalize across models; (2) They are restricted to **layer-level granularity**, as performance collapses significantly when sparsity allocation is refined to the block (attention/FFN) or projection (Q/K/V, etc.) level.

**Key Challenge**: To characterize "which layer should be pruned more" more precisely and robustly without being bottlenecked by the "per-weight scoring + manual reduce function" path, while enabling finer granularity.

**Goal**: To answer two overlooked questions: Is there a need for a superior reduce function to evaluate layer importance? Can non-uniform sparsity be applied at a finer granularity without degrading performance?

**Key Insight**: **Bypassing weight scoring**, the authors use the **minimal linear reconstruction error (LRE)** of each layer under the hypothesis of "pruning 50% least important weights" to quantify redundancy. Layers with low error are more likely to contain critical outliers (low redundancy, should be pruned less), while layers with high error exhibit a uniform error distribution (high redundancy, can be pruned more). This logic naturally extends to block/projection levels.

## Method

### Overall Architecture
LSA consists of three steps: first, a greedy algorithm calculates the "minimal linear reconstruction error when $p=50\%$ of input channels are pruned" for each linear layer as its redundancy; second, these errors are normalized into relative importance scores, and a hyperparameter $\beta$ maps importance to sparsity rates within the $[p_r-\beta, p_r+\beta]$ interval; finally, the same logic is applied at the block/projection granularity for finer non-uniform allocation. Throughout the process, only errors are calculated—weights are not actually pruned until the final unified execution.

```mermaid
flowchart LR
    A[Calibration Data + Layer Weights W] --> B[Algorithm 1<br/>Greedy search for p=50% Minimal LRE]
    B --> C[Layer-wise Redundancy E_l]
    C --> D[Normalize to Importance<br/>I_l = 1 − E_l/ΣE_i]
    D --> E[β Scaling to Sparsity<br/>s = p_r + mean(d) − d]
    E --> F{Granularity Choice}
    F --> G[layer-wise]
    F --> H[block-wise]
    F --> I[projection-wise]
    G & H & I --> J[Execute Pruning<br/>Error itself also serves as weight importance metric]
```

### Key Designs

**1. Minimal Linear Reconstruction Error as a Redundancy Metric: Converting "Output Difference After Pruning" into Submatrix Summation.** For a fully connected layer $W\in\mathbb{R}^{c_o\times c_i}$ and input activations $X$, the linear reconstruction error under pruning mask $M$ is $E=\lVert WX^T-(M\odot W)X^T\rVert_2^2$. The authors derive that: letting $H=X^TX$ and $S=H\odot(W^TW)$, if $P$ is the set of pruned input channels, the error simplifies to $E=\sum_{i\in P}\sum_{j\in P}S_{i,j}$. This transforms selecting "which weights to prune" into "selecting a $c_s \times c_s$ ($c_s=\lfloor c_i\cdot p\rfloor$) submatrix in $S$ to minimize its element sum," providing a basis for efficient solving.

**2. Efficient Solving via Greedy + Grouping: Avoiding Combinatorial Explosion and Full Matrix Storage.** Exhausting all $C(c_i,c_s)$ submatrices is infeasible. LSA uses a vector $\epsilon$ to track "incremental error if a certain channel is further pruned" (initialized with the diagonal of $S$). Each step greedily selects the channel with minimal $\epsilon_j$, updates $\epsilon$ using the corresponding row/column of $S$, and sets the error of the selected channel to $\infty$. To support unstructured pruning, $S^{(k)}=H\odot(W_{k,:}^TW_{k,:})$ is defined for each output channel $k$, and rows are computed on-demand during updates: $e_{k,:}\leftarrow e_{k,:}+2W_{k,i}(W_{k,:}\odot H_{i,:})$. By **vectorizing parallel** processing of all output channels and handling $c_i$ input channels in **groups of size $B$** (Algorithm 1), a balance between accuracy and speed is achieved. Note that this process only yields error values without actual pruning.

**3. $\beta$ Scaling Mapping from Error to Importance to Sparsity: Preventing Collapse from Over-pruning.** The importance of the $l$-th layer is defined as $I_l=1-E_l/\sum_i E_i$. Layers with lower importance receive higher sparsity. Since the original score scale must adjust with the target sparsity $p_r$, the hyperparameter $\beta$ is introduced to compress importance into $[0,2\beta]$. Let $d=I\times2\beta$, and allocate according to $s=p_r+\mathrm{mean}(d)-d$. This constrains each layer's sparsity within $[p_r-\beta, p_r+\beta]$, preventing extreme pruning and performance collapse. The authors also found that the $p$ used for error calculation is insensitive (results are similar below 70%), so $p$ is fixed at 50%.

**4. Fine-grained (Block / Projection) Allocation: The First Fine-grained Sparsity Scheme That Does Not Collapse.** OWL/DLP can only share one importance score at the layer level; attempting to go down to the block or projection level destroys the information flow between layers, causing collapse. LSA extends the same logic to finer granularities, ensuring the overall sparsity remains constant via $s=(p_r\times N+(\mathrm{mean}(d)-d)\times\mathrm{mean}(N))/N$ (where $N$ is the number of weights in each projection). This is the first method capable of non-uniform allocation at the projection level without catastrophic degradation; block-wise even frequently outperforms layer-wise.

## Key Experimental Results

### Main Results: Comparison of Three Granularities (WikiText PPL, 70% Sparsity)

| Method | LLaMA1-7B Layer/Block/Proj | LLaMA2-7B Layer/Block/Proj |
|------|---------------------|---------------------|
| Dense | 5.68 | 5.47 |
| SparseGPT+OWL | 18.98 / 25.93 / 29.87 | 20.68 / 27.39 / 29.91 |
| SparseGPT+DLP | 17.78 / 24.78 / 23.26 | 18.68 / 29.64 / 28.05 |
| SparseGPT+**Ours** | **17.57 / 18.25 / 19.46** | **18.63 / 20.40 / 21.15** |
| Wanda+OWL | 24.85 / 57.91 / 85.39 | 30.03 / 52.57 / 80.32 |
| Wanda+DLP | 20.89 / 40.81 / 52.69 | 22.85 / 59.87 / 117.84 |
| Wanda+**Ours** | **20.66 / 21.60 / 24.82** | **22.89 / 25.55 / 34.56** |

Key Observation: When OWL/DLP move from layer to block/projection level, PPL explodes (e.g., Wanda+DLP at projection level reaches 117.84), while LSA remains nearly stable across all three granularities.

### Ablation Study

| Setting | Conclusion |
|------|------|
| block-wise on LLaMA3 (70%, PPL) | LSA(B) is universally optimal: 3-8B 32.94 vs DLP 40.12; 3.2-1B 87.08 vs DLP 112.29; 3.2-3B 46.24 vs DLP 54.86 |
| LOD and PPL (LLaMA1/2-13B, 70%) | LSA is most effective at preserving outliers, showing highest LOD and lowest PPL (e.g., LLaMA2-13B: LSA 246.05 / 12.56 vs DLP 237.37 / 13.39) |
| $\beta$ Robustness Range (block-wise) | OWL/DLP are only effective in $[0,0.07]$, whereas LSA remains stable throughout $[0,0.17]$ |
| $p$ for Error Calculation | Results are similar for $p<70\%$; insensitive to $p$, fixed at 50% |

### Key Findings
- LSA outperforms SOTA on language modeling and seven zero-shot tasks at 70% high sparsity.
- Shallow layers have low redundancy (less pruning) and deep layers have high redundancy (more pruning), consistent with recent findings that "deep layers are often less important than expected."
- Error itself is a better weight importance metric than Wanda scores—it accounts for the cumulative additional error when multiple weights are pruned simultaneously.

## Highlights & Insights
- **Paradigm Shift**: Moves from "per-weight scoring + manual reduce function" to "directly measuring redundancy with layer-level reconstruction error," removing a difficult-to-tune and non-generalizable empirical step.
- **Elegant Mathematical Simplification**: The pruning error is reduced to the sum of submatrix elements in $S$, and the infeasible combinatorial search is compressed into an algorithm runnable on LLMs via greedy increments, grouping, and vectorization.
- **First Exploration of Projection-level Non-uniform Sparsity**: Refutes the assertion that "fine-grained allocation inevitably collapses," showing that block-wise is often superior to layer-wise, opening paths for even finer allocation.
- **Strong Robustness**: Insensitive to both $\beta$ and the $p$ used for error calculation, making it easier to deploy in engineering.

## Limitations & Future Work
- Remains within unstructured sparsity; actual inference acceleration depends on specialized sparse kernels/hardware support, and end-to-end speedup data is not directly provided.
- Requires calibration data to compute $H=X^TX$, creating some dependence on the calibration set distribution.
- Although $\beta$ is robust, it remains a hyperparameter requiring tuning based on target sparsity; fully adaptive allocation is not yet achieved.
- Primarily validated on LLaMA series and a few 7B models; performance on ultra-large scale (70B+) and more architectures needs further verification.

## Related Work & Insights
- **One-shot Pruning**: SparseGPT (magnitude $\times$ inverse Hessian) and Wanda (magnitude $\times$ activation L2) established the baseline for training-free pruning using uniform sparsity.
- **Layer-wise Non-uniform Sparsity**: OWL (allocation by outlier distribution LOD) and DLP (redundancy via median of weight scores) are direct competitors; LSA identifies the "Wanda scoring + reduce function" bottleneck shared by both.
- **Reconstruction Error Pruning**: Methods like ThiNet and Zhuang et al. minimize channel reconstruction error in CNNs but require iterative forward passes with complexity proportional to the square of channel counts; LSA makes this applicable to LLMs through precomputed errors and greedy incremental updates.
- **Insights**: Shifting "importance measurement" from heuristic scoring to a derivable error objective is a strategy worth reusing in other LLM compression tasks like quantization and KV-cache compression.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Uses minimal linear reconstruction error instead of weight scoring to measure layer redundancy and is the first to achieve projection-level non-uniform sparsity without collapse.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers LLaMA1/2/3, Vicuna, Mistral, and Qwen; includes multi-dimensional comparisons of three granularities, LOD, $\beta$ robustness, and zero-shot tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear, with complete mathematical derivations, algorithm pseudocode, and well-coordinated figures/tables.
- **Value**: ⭐⭐⭐⭐ Provdes a superior and robust layer-wise sparsity allocation scheme for training-free LLM pruning while opening up finer-grained directions; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Large Language Model Compression with Global Rank and Sparsity Optimization](large_language_model_compression_with_global_rank_and_sparsity_optimization.md)
- [\[ICLR 2026\] RCPU: Rotation-Constrained Error Compensation for Structured Pruning of Large Language Models](rcpu_rotation-constrained_error_compensation_for_structured_pruning_of_large_lan.md)
- [\[ICLR 2026\] MaskPro: Linear-Space Probabilistic Learning for Strict (N:M)-Sparsity on LLMs](maskpro_linear-space_probabilistic_learning_for_strict_nm-sparsity_on_llms.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ICLR 2026\] SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization](serq_saliency-aware_low-rank_error_reconstruction_for_llm_quantization.md)

</div>

<!-- RELATED:END -->
