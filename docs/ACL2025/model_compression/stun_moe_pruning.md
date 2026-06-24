---
title: >-
  [Paper Note] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning
description: >-
  [ACL 2025][Model Compression][MoE Pruning] STUN proposes a two-stage MoE pruning paradigm of "structured-then-unstructured": the first stage utilizes routing weight behavioral similarity to cluster redundant experts, completing expert-level pruning with $O(1)$ GPU forward passes; the second stage performs unstructured weight pruning within the remaining experts. Co-designing these two stages allows 40% sparsification on the 480B Snowflake Arctic with almost no performance los…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "MoE Pruning"
  - "Structured Pruning"
  - "Unstructured Pruning"
  - "Expert Clustering"
  - "Sparsification"
date: 2026-05-08
content_hash: d28440da2700c186
---

# STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning

**Conference**: ACL 2025  
**arXiv**: [2409.06211](https://arxiv.org/abs/2409.06211)  
**Code**: [https://github.com/thnkinbtfly/STUN](https://github.com/thnkinbtfly/STUN)  
**Area**: Model Compression  
**Keywords**: MoE Pruning, Structured Pruning, Unstructured Pruning, Expert Clustering, Sparsification

## TL;DR

STUN proposes a two-stage MoE pruning paradigm of "structured-then-unstructured": the first stage utilizes routing weight behavioral similarity to cluster redundant experts, completing expert-level pruning with $O(1)$ GPU forward passes; the second stage performs unstructured weight pruning within the remaining experts. Co-designing these two stages allows 40% sparsification on the 480B Snowflake Arctic with almost no performance loss.

## Background & Motivation

**Background**: MoE (Mixture of Experts) has become a mainstream architecture choice for large models by sparsely activating subsets of experts to reduce inference costs. However, the total number of parameters has significantly increased—Mixtral 56B (8 experts), DBRX 132B (16 experts), Snowflake Arctic 480B (128 experts)—meaning service deployment still requires substantial GPU memory.

**Limitations of Prior Work**: Unstructured pruning (such as Wanda and SparseGPT) performs element-wise sparsification within weight tensors and cannot leverage the natural expert structure of MoE. Row/column-level structured pruning (such as LLM-Pruner) severely damages model capability. Existing expert-level pruning methods (such as Lu et al.) require exhaustive search over expert combinations, with a complexity of $O(k^n/\sqrt{n})$ ($k>1$, where $n$ is the number of experts), which is completely intractable for 128 experts.

**Key Challenge**: Expert-level pruning can exploit the inherent structure of MoE and yields good performance, but its search space scales exponentially with the number of experts. Unstructured pruning is scalable but suffers from catastrophic performance collapse on generative tasks (like GSM8K). Relying on either method alone fails to preserve performance at high sparsity levels.

**Key Insight**: Re-framing the pruning problem as an interpolation between structured and unstructured pruning reveals that the optimal performance lies at an intermediate interpolation point (Figure 1). Furthermore, an $O(1)$ clustering algorithm based on expert behavior similarity is proposed to solve the scalability bottleneck.

**Core Idea**: First, expert-level pruning is performed in $O(1)$ complexity using routing weight similarity to eliminate inter-expert redundancy. Then, unstructured pruning is executed within the remaining experts to capture intra-expert sparsity. The two stages are complementary and synergistic.

## Method

### Overall Architecture

Original MoE $\to$ **Stage 1: Expert-Level Structured Pruning** (clustering experts based on behavioral similarity of routing weights, keeping one representative expert per cluster, requiring $O(1)$ GPU calls) $\to$ **Stage 2: Unstructured Weight Pruning** (performing element-wise sparsification within the remaining experts using Wanda/OWL) $\to$ final compressed model. The entire process is training-free, requiring no fine-tuning or backpropagation.

### Key Designs

1. **$O(1)$ Expert Clustering and Pruning Based on Behavioral Similarity**

    - **Function**: Clusters experts with similar behaviors in each MoE layer and keeps only the representative expert closest to the cluster center, while pruning the rest.
    - **Mechanism**: Extracts the behavioral similarity between expert pairs $b_{i,j} = -\|W_i - W_j\|_F + \lambda_2 a_{i,j}$ (where $a_{i,j}$ represents co-activation statistics) from the routing layer weight matrix $W$. Similar routing weights imply similar activation probabilities for the same inputs, suggesting similar specialized functions. Based on this similarity, hierarchical clustering is performed to reformulate greedy step-by-step pruning into step-by-step probability distribution maximization (Eq. 5-6), making greedy decisions approximately equivalent to combinatorial search.
    - **Design Motivation**: The exhaustive expert combination search by Lu et al. requires $>10^{33}$ forward passes for 128 experts, which is completely infeasible. The proposed method only requires routing weight calculations (taking < 1 minute on CPU) + an optional single forward pass to collect co-activation statistics.

2. **Taylor Approximation + Selective Expert Reconstruction**

    - **Function**: Decides which expert to keep in each cluster after clustering, and whether to reconstruct the representative expert using the cluster mean.
    - **Mechanism**: Approximates the reconstruction loss of each expert using a first-order Taylor expansion $\mathcal{E}_i = \|M(x;\theta) - M(x;\theta - \theta_i)\|_F$, selecting the expert closest to the cluster center parameter $\bar{\theta_i}$ as the representative. For clusters with fewer experts than a threshold $\kappa$ (set as 3), the representative expert is reconstructed using the mean of all member parameters to further reduce reconstruction error.
    - **Design Motivation**: Simply choosing the nearest expert discards the knowledge of pruned experts, whereas mean reconstruction for small clusters partially recovers this information. Computing the mean is avoided for large clusters to prevent blurriness.

3. **Unstructured Pruning Under Kurtosis Preservation**

    - **Function**: Performs unstructured pruning (such as Wanda or OWL) on the model after expert-level pruning.
    - **Mechanism**: The kurtosis of weight distribution is a surrogate indicator for the robustness of unstructured pruning, where high kurtosis denotes a large number of near-zero weights that can be safely pruned. Expert-level pruning does not change the internal weight distribution of the remaining experts (which remains approximately Gaussian), so the kurtosis increases rather than decreases (rising from 14,248 to 15,623 in experiments). Conversely, performing unstructured pruning directly removes near-zero weights, pushing the distribution toward a bimodal shape (reducing kurtosis) and leaving less headroom for further sparsification.
    - **Design Motivation**: Theoretical guarantees demonstrate that the sequence of "structured-then-unstructured" is superior to the reverse order or using either in isolation.

### Loss & Training

The entire pipeline is training-free, requiring no backpropagation or fine-tuning. Only a small amount of C4 calibration data is used for the optional collection of co-activation statistics. For the largest Arctic model, setting $(\lambda_1, \lambda_2) = (1, 0)$ (meaning pure routing weight similarity) completes expert pruning with zero GPU calls.

## Key Experimental Results

### Main Results: STUN vs. Unstructured Pruning Baselines

| Model | Sparsity | Method | GSM8K | ARC-c | HellaSwag | MMLU | Average |
|------|--------|------|-------|-------|-----------|------|------|
| Arctic (480B) | 0% | Unpruned | 70.74 | 56.91 | 66.94 | 64.86 | 68.33 |
| Arctic (480B) | 40% | **STUN (OWL)** | **70.28** | **57.68** | **64.94** | **64.75** | **67.66** |
| Arctic (480B) | 40% | OWL | 63.76 | 56.74 | 65.08 | 63.47 | 67.35 |
| Arctic (480B) | 65% | **STUN (OWL)** | **43.97** | **51.54** | **59.91** | **59.24** | **62.67** |
| Arctic (480B) | 65% | OWL | 13.42 | 44.37 | 53.69 | 52.02 | 56.68 |
| Mixtral-8x7B-Inst | 65% | **STUN (OWL)** | **25.09** | **48.12** | **54.05** | **60.39** | **60.34** |
| Mixtral-8x7B-Inst | 65% | OWL | 1.29 | 24.15 | 49.27 | 57.60 | 45.20 |

### Comparison of Expert Pruning Methods (Mixtral-8x7B, 25% Sparsity)

| Method | Computational Complexity | ARC-C | BoolQ | HellaSwag | MMLU | WinoGrande | Average |
|------|-----------|-------|-------|-----------|------|------------|------|
| Unpruned | - | 59.4 | 84.2 | 84.0 | 67.9 | 75.6 | 71.5 |
| **Ours** | **$O(1)$** | **55.6** | **83.1** | **81.1** | **63.3** | **72.7** | **70.7** |
| Expert Drop | $O(n)$ | 53.2 | 77.7 | 80.5 | 52.2 | 76.8 | 66.0 |
| SEER-MoE | - | - | - | - | 56.7 | - | - |
| Lu et al. | $O(k^n/\sqrt{n})$ | - | - | - | - | - | 64.22 |

### Key Findings

- **Optimal Interpolation**: For a fixed 50% overall sparsity on Mixtral, pure structured pruning ($x=1$) and pure unstructured pruning ($x=0$) both exhibit massive performance degradation, whereas intermediate interpolation points yield the highest performance (Figure 1), validating the necessity of the hybrid two-stage paradigm.
- **Significant Advantage on GSM8K Generation Task**: At 40% sparsity on Arctic, STUN drops only 0.46 points on GSM8K, while OWL drops 6.98 points; at 65% sparsity, STUN retains a score of 43.97, while OWL drops to 13.42.
- **No Performance Gap Between $O(1)$ and $O(n)$**: At 25% sparsity on Mixtral, their averages are 64.34 and 63.97, respectively, with $O(1)$ being slightly superior.
- **Larger MoEs Benefit More**: The performance gap between STUN and pure unstructured pruning widens as the number of experts increases (Arctic 128 experts > Mixtral-8x22B > Mixtral-8x7B), which aligns well with the scaling trend of MoEs.
- **Extremely High Efficiency**: Running STUN+OWL on Arctic 480B takes only a single H100 GPU and 1.12 hours, while the method of Lu et al. is completely unviable for 128 experts (requiring $>10^{33}$ forward passes).

## Highlights & Insights

- The key insight behind $O(1)$ expert pruning is leveraging the implicit structure in routing weights. Similar routing weights indicate similar activation patterns, and greedy pruning post-clustering is approximately equivalent to combinatorial search. This logic can be transferred to other models with natural modular structures.
- The theoretical analysis of kurtosis preservation elegantly explains why the pruning sequence (structured first, then unstructured) cannot be reversed: performing unstructured pruning first reduces kurtosis, which hurts the capacity for subsequent pruning.
- The effectiveness of the STUN paradigm is also verified on dense models (Llama-2), indicating that the "coarse-to-fine" two-stage pruning paradigm is a general strategy.

## Limitations & Future Work

- Unstructured pruning finds it difficult to directly accelerate inference on current GPUs (requiring hardware support for sparsity), meaning actual deployment speedups depend heavily on hardware.
- Expert behavior similarity is only computed based on routing weights and co-activation statistics, without considering semantic divergence in expert outputs.
- Performance drops noticeably beyond 40% sparsity (on Arctic at 65% sparsity, GSM8K drops from 70 to 44); strategies for maintaining performance under extreme sparsity levels require adaptation.
- Experiments on dense models only utilize LLM-Surgeon with 5% structured pruning as the first stage, leaving the design space under-explored.

## Related Work & Insights

- **vs. Wanda/OWL**: Purely unstructured methods, whereas STUN removes redundant experts in the first stage before executing them, making them complementary. The primary distinction is that STUN additionally exploits inter-expert redundancy within MoEs.
- **vs. Lu et al.**: Both are expert-level pruning methods, but Lu et al. has a complexity of $O(k^n/\sqrt{n})$. STUN reduces the search space from exponential to constant via behavioral clustering while achieving superior results. Furthermore, Lu et al. relies heavily on calibration data, whereas STUN utilizes the model's self-structure.
- **vs. LLM-Pruner**: A general structured pruning framework (row/column-level) that suffers from catastrophic performance collapse on high-sparsity MoEs (obtaining only 1.29 on GSM8K at 65% sparsity). STUN preserves crucial capabilities by pruning at the expert-level granularity.
- **vs. SEER-MoE / Expert Drop**: Alternative expert-level pruning methods. STUN outperforms them significantly on MMLU (63.3 vs. 56.7 vs. 52.2) and further structures compression by stacking unstructured pruning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Redefining the pruning problem as a structured-unstructured interpolation and achieving $O(1)$ expert pruning via routing weight clustering is clean and elegant. The kurtosis preservation analysis provides solid theoretical backing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 480B Arctic + the Mixtral series + dense Llama-2 across multiple tasks and sparsity levels, supported by extensive ablation studies (such as $O(1)$ vs. $O(n)$ and interpolation ratios).
- **Writing Quality**: ⭐⭐⭐⭐ Figure 1 intuitively presents the advantage of interpolation, while Table 1 clearly positions the papers' contributions. The logical structure flows effortlessly from motivation $\to$ method $\to$ theory $\to$ experiments.
- **Value**: ⭐⭐⭐⭐⭐ Performing near-lossless 40% compression on a 480B class MoE using only 1 GPU within approximately 1 hour possesses immediate engineering value for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](../../ICLR2026/model_compression/mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)
- [\[ACL 2025\] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](../../ICML2025/model_compression/slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
