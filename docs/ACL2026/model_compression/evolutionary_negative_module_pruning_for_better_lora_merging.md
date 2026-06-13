---
title: >-
  [Paper Note] Evolutionary Negative Module Pruning for Better LoRA Merging
description: >-
  [ACL 2026][Model Compression][LoRA Merging] Ours proposes the ENMP method, which discovers and prunes "negative modules" that degrade performance during LoRA merging through an evolutionary search strategy. As a plug-and…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "LoRA Merging"
  - "Negative Module Pruning"
  - "Evolutionary Search"
  - "Multi-task Deployment"
  - "CMA-ES"
date: 2026-05-08
content_hash: b31d71d13cabcc86
---

# Evolutionary Negative Module Pruning for Better LoRA Merging

**Conference**: ACL 2026  
**arXiv**: [2604.17753](https://arxiv.org/abs/2604.17753)  
**Code**: [github](https://github.com/CaoAnda/ENMP-LoRAMerging)  
**Area**: Model Merging / LoRA Fusion  
**Keywords**: LoRA Merging, Negative Module Pruning, Evolutionary Search, Multi-task Deployment, CMA-ES

## TL;DR

Ours proposes the ENMP method, which discovers and prunes "negative modules" that degrade performance during LoRA merging through an evolutionary search strategy. As a plug-and-play enhancement, it consistently improves the performance of existing merging algorithms across NLP and vision domains.

## Background & Motivation

**Background**: LoRA has become the mainstream method for fine-tuning large models due to its parameter efficiency and good convergence. In practical deployment, it is often necessary to merge LoRA adapters from multiple tasks into a single backbone to achieve efficient multi-task inference.

**Limitations of Prior Work**: Existing merging methods (e.g., Task Arithmetic, TIES, DARE, KnOTS, CoreSpace, etc.) implicitly assume that all LoRA matrices contribute positively to the merged model. However, the authors discovered that LoRA modules in certain layers actually degrade global performance during merging—referred to as "negative modules."

**Key Challenge**: The impact of negative modules is interdependent: a module that appears "negative" in a full set might become beneficial after other harmful modules are removed, and vice-versa. This conditional dependency prevents greedy strategies from capturing high-order interactions, and a $2^N$ search space makes exhaustive search infeasible.

**Goal**: Design a method capable of automatically locating and pruning these negative modules to serve as a universal enhancement plugin for existing merging algorithms.

**Key Insight**: Model the module selection problem as a combinatorial optimization problem and utilize evolutionary strategies to efficiently search for the optimal pruning configuration in a continuous latent space.

**Core Idea**: Utilize the covariance matrix of the CMA-ES evolutionary strategy to model inter-module dependencies. After searching in a continuous space, results are mapped to discrete pruning masks to precisely remove harmful modules.

## Method

### Overall Architecture

The ENMP framework consists of two core phases: (1) sampling candidate pruning masks in a continuous latent space via CMA-ES evolutionary search; (2) applying masks to LoRA adapters to prune negative modules before completing the merge using existing methods (e.g., TIES, DARE). The search process iteratively optimizes distribution parameters based on validation set performance.

### Key Designs

1.  **Negative Module Pruning Mechanism**:

    - **Function**: Selectively removes performance-degrading LoRA layers before merging.
    - **Mechanism**: Defines a binary pruning mask $\mathbf{m} \in \{0,1\}^{L \times T}$, using all attention projections (q/k/v/out_proj) within a Transformer layer as the minimum pruning unit to maintain internal semantic consistency of the attention mechanism.
    - **Design Motivation**: Leave-one-out analysis experiments revealed that merging performance actually improves after removing LoRA modules from certain layers, confirming the existence of negative modules.

2.  **CMA-ES Evolutionary Search Optimization**:

    - **Function**: Efficiently finds the optimal pruning configuration in a $2^N$ discrete search space.
    - **Mechanism**: Introduces a continuous latent vector $\mathbf{z} \in \mathbb{R}^N$ as a learnable negative score, mapping continuous values to binary masks through a dynamic threshold strategy. A conservative initialization (mean $-1$) is used to ensure the search starts from a full merging state.
    - **Design Motivation**: The covariance matrix of CMA-ES can model inter-module dependencies and capture high-order interactions ignored by greedy methods.

3.  **Dynamic Threshold Mask Mapping**:

    - **Function**: Translates continuous latent space search results into discrete binary pruning masks.
    - **Mechanism**: Sets a maximum pruning ratio $k$, selecting the $\lfloor k \cdot N \rfloor$ largest positive elements from $\mathbf{z}$ to be 1 (pruned), with the rest being 0 (retained).
    - **Design Motivation**: Achieves adaptive sparsity via an upper-bound constraint—experiments show the algorithm autonomously converges to the optimal sparsity level without fine-tuning.

### Loss & Training

Evolutionary search is a one-time offline calculation. Population size $N_{\text{pop}}=16$, iterated for 60 generations, initial step size $\sigma=0.5$, and maximum pruning ratio $k=0.2$. Candidate solutions are evaluated in parallel on 8 RTX 4090 GPUs, converging in approximately 2.3 hours, with most gains obtained within the first 10 generations.

## Key Experimental Results

### Main Results (NLP Benchmark - Llama-3-8B)

| Method | Average Normalized Accuracy | Gain |
|------|-----------------|------|
| TA | 90.25% | - |
| TA + ENMP | 93.49% | +3.24% |
| TIES | 89.99% | - |
| TIES + ENMP | 96.39% | +6.40% |
| DARE | 89.20% | - |
| DARE + ENMP | 96.17% | +6.97% |
| KnOTS | 92.47% | - |
| KnOTS + ENMP | 97.29% | +4.82% |
| CoreSpace | 94.18% | - |
| CoreSpace + ENMP | 96.73% | +2.55% |

### Ablation Study

| Configuration | Average Normalized Accuracy | Description |
|------|-----------------|------|
| TA + Random Pruning | 89.10% | Random pruning actually degrades performance |
| TA + ENMP | 93.49% | Precise localization is key |
| k=0.0 | 90.25% | No pruning |
| k=0.1 | 93.37% | Significant gains even with minimal pruning |
| 64 samples/task | 91.17% | Effective with small amounts of validation data |

### Key Findings
- ENMP brings consistent improvements across all baseline methods, indicating that negative modules are a universal bottleneck in LoRA merging.
- Over +20% recovery is achieved on the sensitive QNLI task, suggesting that task interference is unevenly distributed.
- Prune-then-Align outperforms Align-then-Prune, preventing negative modules from "polluting" the shared subspace.
- Effective in the vision domain as well (KnOTS +5.54%), demonstrating cross-modal generality.

## Highlights & Insights
- Systematically reveals the "negative module" phenomenon in LoRA merging for the first time, challenging the implicit assumption that "all modules contribute positively."
- Plug-and-play design: can be combined with any existing merging algorithm without modifying the algorithm itself.
- Adaptive sparsity: the search algorithm automatically determines the optimal number of pruned modules without manual tuning.
- The merged model maintains the original backbone structure with zero additional inference overhead.

## Limitations & Future Work
- Evolutionary search requires one-time offline computation (~2.3 hours), posing challenges for scaling to extremely large models (70B+).
- Relies on a validation set to calculate fitness, making it inapplicable to strict data-free merging scenarios.
- Future work can explore more efficient sampling strategies and data-free pruning methods.

## Related Work & Insights
- **vs Task Arithmetic/TIES/DARE**: These methods handle interference at the parameter level; ENMP eliminates interference at the module level, making them complementary.
- **vs KnOTS/CoreSpace**: Subspace alignment methods assume all modules contribute positively; ENMP yields better results by removing harmful modules before alignment.
- **vs Greedy Pruning**: Greedy strategies ignore cross-layer dependencies, leading to performance degradation (55.76%), while evolutionary search captures high-order interactions.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically reveals negative modules and proposes an evolutionary search solution with a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across NLP and CV domains with 6 baselines and extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, logical flow from phenomenon to method to experiments.
- Value: ⭐⭐⭐⭐ High practical plug-and-play value with important insights for the LoRA merging field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)
- [\[CVPR 2026\] Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy](../../CVPR2026/model_compression/preference-aligned_lora_merging_preserving_subspace_coverage_and_addressing_dire.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/model_compression/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](../../ICML2026/model_compression/saliency-aware_model_merging.md)

</div>

<!-- RELATED:END -->
