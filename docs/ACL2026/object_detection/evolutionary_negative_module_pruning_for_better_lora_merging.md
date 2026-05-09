---
title: >-
  [Paper Note] Evolutionary Negative Module Pruning for Better LoRA Merging
description: >-
  [ACL 2026][Object Detection][LoRA Merging] Proposes ENMP, a method that uses evolutionary search to identify and prune "negative modules" in LoRA merging that degrade performance. As a plug-and-play enhancement, it consistently improves existing merging algorithms across NLP and vision tasks.
tags:
  - ACL 2026
  - Object Detection
  - LoRA Merging
  - Negative Module Pruning
  - Evolutionary Search
  - Multi-Task Deployment
  - CMA-ES
date: 2026-05-08
content_hash: 23bd3406323b6510
---
# Evolutionary Negative Module Pruning for Better LoRA Merging

**Conference**: ACL 2026
**arXiv**: [2604.17753](https://arxiv.org/abs/2604.17753)
**Code**: [github](https://github.com/CaoAnda/ENMP-LoRAMerging)
**Area**: Model Merging / LoRA Fusion
**Keywords**: LoRA merging, negative module pruning, evolutionary search, multi-task deployment, CMA-ES

## TL;DR

This paper proposes ENMP, a method that leverages evolutionary search to identify and prune "negative modules" that degrade performance during LoRA merging. Designed as a plug-and-play enhancement, ENMP consistently improves existing merging algorithms across both NLP and vision domains.

## Background & Motivation

**Background**: LoRA has become the dominant fine-tuning paradigm for large models owing to its parameter efficiency and favorable convergence properties. In practical deployment, it is often necessary to merge multiple task-specific LoRA adapters into a single backbone to enable efficient multi-task inference.

**Limitations of Prior Work**: Existing merging methods—such as Task Arithmetic, TIES, DARE, KnOTS, and CoreSpace—implicitly assume that all LoRA matrices contribute positively to the merged model. However, the authors find that LoRA modules at certain layers can actually degrade overall performance upon merging, giving rise to the concept of "negative modules."

**Key Challenge**: The impact of negative modules is mutually dependent: a module that appears harmful in the full set may become beneficial once other detrimental modules are removed, and vice versa. This conditional dependency prevents greedy strategies from capturing higher-order interactions, while the $2^N$ search space renders exhaustive enumeration infeasible.

**Goal**: To design a method that automatically identifies and removes negative modules, serving as a general-purpose enhancement plugin compatible with existing merging algorithms.

**Key Insight**: The module selection problem is formulated as a combinatorial optimization problem, and an evolutionary strategy is employed to efficiently search for the optimal pruning configuration in a continuous latent space.

**Core Idea**: CMA-ES is used to model inter-module dependencies via its covariance matrix, enabling search in a continuous space that is subsequently mapped to discrete pruning masks to precisely eliminate harmful modules.

## Method

### Overall Architecture

The ENMP framework consists of two core stages: (1) sampling candidate pruning masks in a continuous latent space via CMA-ES evolutionary search; and (2) applying the masks to LoRA adapters to remove negative modules before performing merging with existing methods (e.g., TIES, DARE). The search process iteratively optimizes distribution parameters using validation set performance as the fitness signal.

### Key Designs

1. **Negative Module Pruning Mechanism**:

    - **Function**: Selectively removes LoRA layers that degrade performance prior to merging.
    - **Mechanism**: A binary pruning mask $\mathbf{m} \in \{0,1\}^{L \times T}$ is defined, with all attention projections within a Transformer layer (q/k/v/out_proj) treated as the minimal pruning unit to preserve the internal semantic consistency of the attention mechanism.
    - **Design Motivation**: Leave-one-out analysis reveals that removing LoRA modules at certain layers improves merging performance, empirically confirming the existence of negative modules.

2. **CMA-ES Evolutionary Search Optimization**:

    - **Function**: Efficiently identifies the optimal pruning configuration within the $2^N$ discrete search space.
    - **Mechanism**: A continuous latent vector $\mathbf{z} \in \mathbb{R}^N$ is introduced as a learnable negativity score, which is mapped to a binary mask via a dynamic thresholding strategy. A conservative initialization (mean $-1$) ensures that the search begins from the fully merged state.
    - **Design Motivation**: The covariance matrix of CMA-ES models inter-module dependencies, capturing higher-order interactions that greedy methods overlook.

3. **Dynamic Threshold Mask Mapping**:

    - **Function**: Converts continuous latent search results into discrete binary pruning masks.
    - **Mechanism**: A maximum pruning ratio $k$ is set; the top $\lfloor k \cdot N \rfloor$ positive entries of $\mathbf{z}$ are set to 1 (pruned) and the remainder to 0 (retained).
    - **Design Motivation**: The upper-bound constraint enables adaptive sparsity—experiments show that the algorithm autonomously converges to the optimal sparsity level without fine-grained hyperparameter tuning.

### Loss & Training

The evolutionary search is a one-time offline computation. Population size is $N_{\text{pop}}=16$, run for 60 generations with initial step size $\sigma=0.5$ and maximum pruning ratio $k=0.2$. Candidate configurations are evaluated in parallel on 8 RTX 4090 GPUs, converging in approximately 2.3 hours, with the majority of gains achieved within the first 10 generations.

## Key Experimental Results

### Main Results (NLP Benchmark — Llama-3-8B)

| Method | Avg. Normalized Accuracy | Gain |
|--------|--------------------------|------|
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

| Configuration | Avg. Normalized Accuracy | Note |
|---------------|--------------------------|------|
| TA + Random Pruning | 89.10% | Random pruning hurts performance |
| TA + ENMP | 93.49% | Precise localization is critical |
| $k=0.0$ | 90.25% | No pruning |
| $k=0.1$ | 93.37% | Significant gains with minimal pruning |
| 64 samples/task | 91.17% | Effective with very limited validation data |

### Key Findings
- ENMP yields consistent improvements across all baseline methods, indicating that negative modules constitute a universal bottleneck in LoRA merging.
- Recoveries exceeding +20% are observed on the sensitivity-prone QNLI task, suggesting that task interference is unevenly distributed.
- Prune-then-Align outperforms Align-then-Prune, preventing negative modules from "contaminating" the shared subspace.
- ENMP is equally effective in the vision domain (KnOTS +5.54%), demonstrating cross-modal generality.

## Highlights & Insights
- This work is the first to systematically expose the "negative module" phenomenon in LoRA merging, challenging the implicit assumption that all modules contribute positively.
- The plug-and-play design is compatible with any existing merging algorithm without requiring modifications to the merging procedure itself.
- Adaptive sparsity: the search algorithm automatically determines the optimal number of modules to prune, eliminating the need for fine-grained hyperparameter tuning.
- The merged model retains the original backbone architecture, incurring zero additional inference overhead.

## Limitations & Future Work
- The evolutionary search requires a one-time offline computation (~2.3 hours), and scaling to very large models (70B+) remains challenging.
- The method relies on a validation set to compute fitness scores, precluding its use in strictly data-free merging scenarios.
- Future work may explore more efficient sampling strategies and data-free pruning approaches.

## Related Work & Insights
- **vs. Task Arithmetic / TIES / DARE**: These methods address interference at the parameter level, whereas ENMP eliminates interference at the module level; the two are complementary.
- **vs. KnOTS / CoreSpace**: Subspace alignment methods assume all modules contribute positively; removing harmful modules prior to alignment yields superior results.
- **vs. Greedy Pruning**: Greedy strategies ignore cross-layer dependencies, leading to performance collapse (55.76%); evolutionary search captures higher-order interactions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic exposure of the negative module phenomenon with an evolutionary search solution; novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validated across NLP and CV domains, with six baseline comparisons and extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, and the logical flow from phenomenon to method to experiments is coherent.
- **Value**: ⭐⭐⭐⭐ High practical value as a plug-and-play tool; provides important insights for the LoRA merging community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/object_detection/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[CVPR 2026\] SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras](../../CVPR2026/object_detection/spiraldiff_spiral_diffusion_with_lora_for_rgb-to-raw_conversion_across_cameras.md)
- [\[AAAI 2026\] T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection](../../AAAI2026/object_detection/t-rex-omni_integrating_negative_visual_prompt_in_generic_object_detection.md)
- [\[ECCV 2024\] SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](../../ECCV2024/object_detection/shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)
- [\[ECCV 2024\] Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer](../../ECCV2024/object_detection/portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer.md)

<!-- RELATED:END -->
