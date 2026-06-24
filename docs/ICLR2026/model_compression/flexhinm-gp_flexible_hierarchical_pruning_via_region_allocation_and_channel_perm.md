---
title: >-
  [Paper Note] FlexHiNM-GP: Flexible Hierarchical Pruning via Region Allocation and Channel Permutation
description: >-
  [ICLR 2026][Model Compression][N:M Sparsity] Weight layers are adaptively partitioned into "dense (4:4)", "N:M (2:4)", and "fully pruned (0:4)" regions. This is coupled with a structure-aware Gyro-Permutation and differentiable 2:4 mask learning, enabling structured pruning to approach unstructured accuracy while maintaining GPU Sparse Tensor Core compatibility.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "N:M Sparsity"
  - "Hierarchical Sparsity"
  - "Channel Permutation"
  - "Sparse Tensor Core"
  - "Hard Concrete"
  - "Gradual Pruning"
date: 2026-05-08
content_hash: 46c4b2099dce3280
---

# FlexHiNM-GP: Flexible Hierarchical Pruning via Region Allocation and Channel Permutation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=YaZraqRsbB](https://openreview.net/forum?id=YaZraqRsbB)  
**Code**: To be confirmed  
**Area**: Model Compression / Structured Pruning  
**Keywords**: N:M Sparsity, Hierarchical Sparsity, Channel Permutation, Sparse Tensor Core, Hard Concrete, Gradual Pruning  

## TL;DR
Weight layers are adaptively partitioned into "dense (4:4)", "N:M (2:4)", and "fully pruned (0:4)" regions. This is coupled with a structure-aware Gyro-Permutation and differentiable 2:4 mask learning, enabling structured pruning to approach unstructured accuracy while maintaining GPU Sparse Tensor Core compatibility.

## Background & Motivation
- **Background**: N:M sparsity (typically 2:4) is natively supported by NVIDIA Ampere’s Sparse Tensor Core (STC) via the `mma.sp` instruction, eliminating software indexing overhead. It is currently the most hardware-friendly pruning paradigm.
- **Limitations of Prior Work**: Hardware only supports fixed N:M modes (e.g., 2:4 is fixed at 50%), preventing flexible adjustment of pruning granularity by layer or weight importance. Venom proposed Hierarchical N:M (HiNM), which prunes some column vectors entirely (0:4) and applies 2:4 to the remainder to exceed 50% sparsity—but the uniform 2:4 step for remaining vectors is overly coarse.
- **Key Challenge**: The importance distribution variance among retained vectors is high. Even if all four elements in a vector are significant, 2:4 constraints force the removal of two. This mismatch between fixed patterns and weight distributions leads to significant info loss.
- **Goal**: Add "fine-grained regional control" and "significant weight alignment" to HiNM without sacrificing STC compatibility, pushing structured pruning accuracy toward unstructured levels.
- **Core Idea**: A trio of **triple-level region allocation**, **structure-aware channel permutation**, and **differentiable masks**. A closed-form formula splits the target sparsity into boundaries for dense/2:4/fully-pruned regions; Gyro-Permutation rearranges important elements; and Hard Concrete allows 2:4 masks to be learnable during gradual pruning.

## Method

### Overall Architecture
FlexHiNM-GP partitions the weight matrix into tiles by output channels. Within each tile: **Boundary Search** determines the ratios of the three regions; **Gyro-Permutation** rearranges input/output channels to align significant weights; **Gradual Pruning** uses Hard Concrete masks to jointly optimize weights and the 2:4 mask. At inference, a custom GPU kernel distributes dense and sparse tiles across two CUDA streams using `mma` and `mma.sp`, merging via `atomicAdd`.

```mermaid
flowchart LR
    W[Pre-trained Weight W] --> OP[Output Channel Gyro-Permutation]
    OP --> BS[Boundary Search<br/>Solve for vs, ps]
    BS --> VP[Vector Pruning 0:4]
    VP --> IP[Input Channel Gyro-Permutation]
    IP --> NM[2:4 Pruning + Hard Concrete Mask Learning]
    NM --> GP[Gradual Pruning<br/>Four-stage Fine-tuning]
    GP --> K[Dual-stream GPU Kernel<br/>mma + mma.sp Merged]
```

### Key Designs

**1. Boundary Search for Triple Regions: Simplifying 1D Sparsity Search.** Given a target sparsity $t_s$, the method introduces two boundary parameters: the vector sparsity boundary $v_s$ (percentage of vectors pruned entirely) and the partial sparsity boundary $p_s$ (percentage of remaining vectors assigned to the 2:4 zone). Since 2:4 contributes $(1-v_s)\,p_s \times 0.5$ sparsity, the total is $t_s = v_s + (1-v_s)\,p_s \times 0.5$, yielding $p_s = \frac{2(t_s - v_s)}{1 - v_s}$. This constraint collapses a 2D search into a single curve $v_s$. The objective is to maximize the sum of second-order importance $R_{total}=R_{dense}+R_{24}$. Leveraging the concavity of the objective (proven in Appendix), the search starts at $v_s=t_s$ and decreases by step $\alpha$ until the objective drops. The step size is adaptive: $\alpha = \alpha - \beta v_s - \gamma\,\text{RMSProp}(v_s,p_s)$, where $\beta$ penalizes excessive $v_s$ and $\gamma$ applies a soft constraint.

**2. Gyro-Permutation: Rotating Significant Weights to the Right Indices.** Channel permutation rearranges channels without changing the computation result to place important weights in retained positions. It is formulated as $\arg\max_{\Lambda_O,\Lambda_I}\|M\odot D[\Lambda_O;\Lambda_I]\|$ where $D$ is the importance matrix and $M$ satisfies valid constraints. To avoid combinatorial explosion, the process is decoupled into "Output Permutation → Vector Pruning → Intra-tile Input Permutation." Each iteration involves: **sampling** M vectors per tile, **clustering** into N clusters via balanced K-means, and **assignment** via the Hungarian algorithm based on a pruning cost matrix. This captures coupling between input/output structures and integrates index translation into HiNM's native indexing during memory transfer to avoid Tetris-like overhead.

**3. Hard Concrete Differentiable 2:4 Mask: Co-evolving Masks and Weights.** During gradual pruning, as sparsity increases from a% to b%, boundaries update—some 2:4 vectors are pruned, and some dense vectors become 2:4. Static greedy selection becomes sub-optimal after fine-tuning. Hard Concrete distributions parameterize the binary mask for each 4-element group as learnable logits $\alpha_i$: noise $\epsilon_i\sim U(0,1)$ is sampled, soft masks $s_i = \sigma\!\big(\frac{1}{\tau}(\log\epsilon_i - \log(1-\epsilon_i) + \log\alpha_i)\big)$ are calculated, stretched to $(-0.1, 1.1)$, and clipped to $[0,1]$ for the final mask $z_i$. The loss is $L = L_{task} + \lambda_s\,\text{mean}(z_i) + \lambda_c\,\text{mean}(|\sum_{i=1}^4 z_i - 2|)$, where $L_{hard}$ enforces exactly 2 elements per group. Temperature $\tau$ anneals every 5 epochs, and masks are hardened at a 0.5 threshold every 20 epochs. Masks are only updated for newly introduced weight zones to ensure monotonic pruning.

**4. Dual-stream Custom GPU Kernel: Parallel Dense and Sparse Execution.** Weight tiles are categorized as sparse or dense and assigned to Stream0 and Stream1. Stream0 loads 2:4 pruned weights and selected input vectors (via pre-computed N:M indices) into shared memory for STC `mma.sp`. Stream1 loads full dense tiles for standard GEMM `mma`. Partial results from both streams are accumulated via `atomicAdd`. The kernel dynamically rearranges input channels to ensure cross-layer consistency without offline permutation.

## Key Experimental Results

### Main Results (LLaMA2-7B Accuracy/%)

| Method | 75% OBQA | 75% ARC-E | 75% PIQA | 75% HellaS | 87.5% PIQA | 87.5% HellaS |
|------|------|------|------|------|------|------|
| Dense | 32.07 | 76.39 | 79.62 | 57.03 | 79.62 | 57.03 |
| Unstructured | 23.87 | 62.04 | 71.75 | 45.07 | 67.84 | 40.96 |
| OVW | 19.67 | 53.77 | 65.13 | 39.86 | 61.32 | 34.28 |
| HiNM-V (≈Venom) | 20.33 | 54.96 | 66.16 | 40.63 | 62.45 | 35.17 |
| HiNM-GP | 22.07 | 57.84 | 68.87 | 42.72 | 64.18 | 36.82 |
| **Ours** | **23.13** | **59.22** | **70.14** | **44.58** | **65.72** | **38.11** |

FlexHiNM-GP outperforms HiNM-GP by +1.39% on average at 75% sparsity and is the closest structured method to the unstructured baseline. On Deit-Base, it achieves 81.13%/79.46% top-1 at 75%/80% sparsity. On Bert-Base (75% sparsity), it reaches 88.55 F1 (SQuAD) and 91.65% (SST-2), nearly matching unstructured performance (89.04 / 91.86%).

### Ablation Study (Deit-Base, Accuracy/%)

| Variant | Flexible | Hard Concrete | Gumbel | 75% | 87.5% | 95% |
|------|------|------|------|------|------|------|
| ① FlexHiNM-GP | ✓ | ✓ | – | **81.13** | **75.23** | **61.43** |
| ② | ✓ | – | ✓ | 81.10 | 75.55 | 59.77 |
| ③ | ✓ | – | – | 81.08 | 75.34 | 59.64 |
| ⑥ HiNM-GP | – | – | – | 81.04 | 74.35 | 58.94 |

Both Flexible region allocation and Hard Concrete contribute positively. At high sparsity (95%), Hard Concrete significantly outperforms Gumbel-Softmax (61.43 vs 59.77) because Gumbel is limited to 6 fixed patterns per group, whereas Hard Concrete generates near-binary masks for each element independently, offering a larger search space and stable gradients.

### Key Findings
- **Inference Speed**: Latency falls between OVW and HiNM-GP. Deit-Base achieves 1.96×/2.22×/2.65× speedup at 75%/87.5%/90% sparsity. The slight latency cost of retaining dense regions is justified by significant accuracy gains.
- **Permutation is Critical**: Comparing HiNM-V (no permutation) with HiNM-GP shows that without Gyro-Permutation, significant weights misalign with retained tiles, worsening degradation as sparsity increases.

## Highlights & Insights
- **Boundary Optimization**: The closed-form $p_s = \frac{2(t_s - v_s)}{1-v_s}$ reduces complex 2D allocation to a 1D scan, making it engineering-friendly.
- **Physical Intuition**: High-importance vectors stay dense (protected from 2:4 damage), medium-importance use 2:4, and low-importance are pruned. This restricts the side effects of 2:4 to non-essential weights.
- **Zero Runtime Overhead**: Integrating index translation into native indexing and memory staging avoids the high GPU runtime overhead seen in methods like Tetris.

## Limitations & Future Work
- **Specific to 2:4**: The method is centered on current STC 2:4 support; future N:M hardware would require re-deriving boundary constraints.
- **Training Cost**: The four-stage gradual pruning with joint optimization of weights and Hard Concrete masks, plus iterative permutation, involves high training overhead.
- **Hyperparameter Sensitivity**: $\beta, \gamma$, loss weights $\lambda_s, \lambda_c$, and annealing schedules require tuning; cross-architecture generalization needs more extensive validation.
- **Latency vs. Accuracy**: Retaining dense regions remains a trade-off; it may not be optimal for pure throughput-driven scenarios.

## Related Work & Insights
- **Venom (HiNM)**: Direct predecessor. This work refines the "uniform 2:4 for remainder" limitation.
- **Pool & Yu / Tetris**: Source of the channel permutation idea; Gyro-Permutation adds HiNM-awareness and zero-overhead indexing.
- **MaskLLM / S2HPruner**: Representative learnable semi-structured sparsity. Replacing Gumbel with Hard Concrete provides a larger search space and smoother gradients.
- **Insight**: When hardware provides fixed patterns, a "software-side multi-level region + permutation alignment + differentiable mask" approach is a universal paradigm for bridging the gap between structured and unstructured accuracy.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of triple-region boundaries, Gyro-Permutation, and Hard Concrete is a solid innovation within the HiNM framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers Deit, Bert, and LLaMA2 across high sparsity ratios with performance and latency metrics, though end-to-end comparisons with more recent learnable methods would be beneficial.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear evolution of variants and boundary derivations, though some notation is dense.
- **Value**: ⭐⭐⭐⭐ — Directly targets STC hardware with high practical value for large model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](../../NeurIPS2025/model_compression/permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)
- [\[ICLR 2026\] FlexLoRA: Entropy-Guided Flexible Low-Rank Adaptation](flexlora_entropy-guided_flexible_low-rank_adaptation.md)
- [\[ICLR 2026\] LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error](lsa_layer-wise_sparsity_allocation_for_large_language_model_pruning_based_on_min.md)
- [\[ICLR 2026\] Navigating the Accuracy-Size Trade-Off with Flexible Model Merging](navigating_the_accuracy-size_trade-off_with_flexible_model_merging.md)
- [\[ICML 2026\] ToaSt: Token Channel Selection and Structured Pruning for Efficient ViT](../../ICML2026/model_compression/toast_token_channel_selection_and_structured_pruning_for_efficient_vit.md)

</div>

<!-- RELATED:END -->
