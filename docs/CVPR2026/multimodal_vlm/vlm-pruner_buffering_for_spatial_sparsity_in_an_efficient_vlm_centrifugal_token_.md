---
title: >-
  [Paper Note] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm
description: >-
  [CVPR 2026][Multimodal VLM][visual token pruning] This paper proposes VLM-Pruner, a training-free centrifugal token pruning method that balances redundancy elimination and local detail preservation through a Buffering fo…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "visual token pruning"
  - "inference acceleration"
  - "spatial sparsity"
  - "training-free"
  - "VLM efficiency"
date: 2026-05-08
content_hash: 08e0b9c33b076a28
---

# VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm

**Conference**: CVPR 2026
**arXiv**: [2512.02700](https://arxiv.org/abs/2512.02700)
**Code**: [https://github.com/Casey-bit/VLMPruner](https://github.com/Casey-bit/VLMPruner)
**Area**: Multimodal VLM
**Keywords**: visual token pruning, inference acceleration, spatial sparsity, training-free, VLM efficiency

## TL;DR
This paper proposes VLM-Pruner, a training-free centrifugal token pruning method that balances redundancy elimination and local detail preservation through a Buffering for Spatial Sparsity (BSS) criterion. At an 88.9% pruning rate, it consistently outperforms existing methods across 5 VLMs while achieving end-to-end inference acceleration.

## Background & Motivation
**Background**: VLMs combine visual encoders with LLMs and achieve strong performance on image understanding tasks, but the large number of visual tokens generated from high-resolution images introduces substantial computational overhead due to the quadratic complexity of attention. Training-free token pruning has emerged as a mainstream solution.

**Limitations of Prior Work**: The two dominant strategies each exhibit distinct shortcomings: (a) **importance-driven** methods (e.g., FastV) retain tokens based on attention scores but tend to cluster selections within similar local regions, leading to redundancy; (b) **redundancy elimination** methods (e.g., DivPrune/DART) greedily select tokens with the lowest mutual similarity but ignore spatial relationships, resulting in overly scattered selections that fail to provide complete coverage of target object details.

**Key Challenge**: There is a fundamental tension between reducing redundancy (selecting highly diverse tokens) and maintaining local completeness (selecting spatially contiguous tokens). Excessive pursuit of diversity causes selections to alternate between foreground and background regions.

**Goal**: To design a token pruning method that simultaneously balances redundancy elimination and spatial continuity.

**Key Insight**: The observation that fine-grained details of target objects require spatially adjacent tokens for complete coverage motivates a "centrifugal" selection strategy that expands outward from a core region.

**Core Idea**: The BSS criterion biases token selection toward spatially proximate low-redundancy tokens, enabling an ordered centrifugal expansion from near to far.

## Method

### Overall Architecture
VLM-Pruner operates at the second layer of the LLM decoder in three stages: (1) selecting a small set of high-diversity pivot tokens as initial seeds; (2) expanding the selection set from near to far via a greedy algorithm guided by the BSS criterion; and (3) merging information from discarded tokens back into retained tokens via Similarity-Weighted Aggregation (SWA).

### Key Designs

1. **Pivot Initialization**:

    - Function: Select $\kappa$ semantically diverse seed tokens.
    - Mechanism: Iteratively apply a max-min strategy in the token key space: $j_t = \arg\max_{j \in \mathcal{C}} \min_{j' \in \mathcal{S}_{t-1}} \|\mathbf{K}_j - \mathbf{K}_{j'}\|_2$
    - Design Motivation: Using keys (low-dimensional, refined semantic representations) rather than hidden states reduces interference from redundant information; max-min ensures that the initial seeds cover diverse semantic regions.

2. **Buffering for Spatial Sparsity (BSS) Criterion**:

    - Function: Modify the similarity metric so that tokens spatially distant from the current selection set are "deferred" in selection order.
    - Mechanism: Define the BSS-modulated similarity $\widetilde{M}_{ij} = M_{ij}(1+\lambda\bar{\delta}_i(\mathcal{S}))$, where $\bar{\delta}_i(\mathcal{S}) = \min_{j \in \mathcal{S}} D_{ij}^{(sp)} / D_{max}$ is the normalized minimum spatial distance from candidate token $i$ to the current selection set. Tokens farther from the selection set have their similarity amplified, making them appear "more redundant" and thus less likely to be selected.
    - Design Motivation: Pure redundancy elimination methods tend to select peripheral background tokens (which have low similarity to the main subject). BSS avoids this scattered selection through a spatial proximity-first mechanism. As iterations proceed, $\bar{\delta}_i$ continuously decreases, preserving the overall submodular property.

3. **Similarity-Weighted Aggregation (SWA) Recovery**:

    - Function: Recover useful information from discarded tokens.
    - Mechanism: Each discarded token $u$ is assigned to the most similar retained token $j^*(u) = \arg\max_{j \in \mathcal{S}} M_{uj}$, followed by weighted aggregation: $\mathbf{H}_j = \beta\mathbf{H}_j + (1-\beta)\sum_{u} \alpha_{u\to j}\mathbf{H}_u$ (with $\beta=0.3$).
    - Design Motivation: Centrifugal selection inevitably discards the outermost tokens; SWA compensates for this loss through information recovery.

### Acceleration Strategies
- Only the $q=256$ highest-variance channels are retained for similarity matrix computation, reducing complexity from $O(N^2 d)$ to $O(N^2 q)$.
- Candidate tokens are processed in batches (batch size $B$) to avoid per-token evaluation.
- The acceptance threshold is scheduled from strict to lenient ($\tau^{(0)}=0.8$, step size 0.1) to prevent premature acceptance of isolated distant tokens.

## Key Experimental Results

### Main Results — LLaVA-1.5-7B at Different Pruning Rates

| Method | Retain 192 (66.7%) Avg. | Retain 128 (77.8%) Avg. | Retain 64 (88.9%) Avg. |
|--------|------------------------|------------------------|------------------------|
| FastV | 96.45% | 92.95% | - |
| DART | 98.40% | 97.00% | - |
| DivPrune | 97.80% | 96.40% | - |
| **VLM-Pruner** | **98.85%** | - | **Best** |

### Ablation Study — Effect of Individual Components (inferred from paper description)

| Configuration | Effect |
|---------------|--------|
| w/o BSS (pure redundancy elimination) | Scattered selection, similar to DivPrune |
| w/o SWA | Loss of peripheral information, performance degradation |
| w/o pivot (random initialization) | Uneven initial coverage |
| Full VLM-Pruner | Optimal balance |

### Key Findings
- VLM-Pruner consistently achieves top performance across 5 VLMs (including LLaVA-1.5-7B) on 13 benchmarks, with the advantage growing as the pruning rate increases.
- Competitive performance is maintained even at the extreme pruning rate of 88.9%, confirming that centrifugal selection more effectively preserves fine-grained object details.
- Visualizations clearly show that tokens selected by VLM-Pruner are more densely and orderly distributed over target objects (e.g., 4 consecutive tokens on a fork), whereas DivPrune/DART selections alternate between foreground and background.
- Compared to importance-based methods, VLM-Pruner permits moderate local clustering to preserve detail integrity rather than penalizing all forms of spatial aggregation.

## Highlights & Insights
- **Centrifugal pruning paradigm**: This work transcends the existing binary of "importance-driven vs. diversity-driven" methods by proposing a novel paradigm of locally dense coverage expanding outward, with clear intuition and thorough empirical validation.
- **Mathematical elegance of BSS**: A simple spatial distance modulation achieves the "proximity-first" effect while preserving the submodular property and effectively reordering token selection.
- **Training-free and generalizable**: The method requires no modification to VLM weights, is plug-and-play, and demonstrates consistent effectiveness across diverse model architectures.

## Limitations & Future Work
- The pruning is applied at the second decoder layer; whether this placement is optimal has not been systematically studied.
- The method introduces several hyperparameters ($\lambda$, $\tau^{(0)}$, $\Delta\tau$, $\beta$), and sensitivity analyses remain incomplete.
- The method assumes that tokens corresponding to target objects are spatially contiguous, which may not hold for scattered or occluded objects.
- Precomputation of the similarity matrix $M \in \mathbb{R}^{N \times N}$ still incurs overhead and may become a bottleneck for high-resolution inputs.
- No comparison is made against training-based pruning methods (e.g., ATP-LLaVA).

## Related Work & Insights
- **vs. FastV/SparseVLM** (importance-driven): These methods can underperform random pruning in certain settings, as importance-based criteria lead to redundant selection.
- **vs. DivPrune/DART** (redundancy elimination): The pursuit of global diversity results in scattered selections; VLM-Pruner achieves ordered expansion through BSS.
- **vs. MustDrop/FiCoCo** (methods with locality penalties): These methods penalize all spatial aggregation, whereas moderate clustering is in fact beneficial for preserving fine-grained details.

## Rating
- Novelty: ⭐⭐⭐⭐ Centrifugal pruning combined with the BSS criterion represents a creative and well-motivated design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 VLMs × 13 benchmarks × 3 pruning rates.
- Writing Quality: ⭐⭐⭐⭐ The motivation–method–experiment logical chain is clear, with excellent illustrations.
- Value: ⭐⭐⭐⭐ High practical utility as a training-free method; the design philosophy has strong transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)
- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](../../ICCV2025/multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)

</div>

<!-- RELATED:END -->
