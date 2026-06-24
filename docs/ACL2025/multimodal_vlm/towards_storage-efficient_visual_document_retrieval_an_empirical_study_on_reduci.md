---
title: >-
  [Paper Note] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings
description: >-
  [ACL 2025][Multimodal VLM][Visual Document Retrieval] This paper systematically studies compression strategies for patch-level embeddings in Visual Document Retrieval (VDR). It finds that pruning is inherently unsuitable for VDR (where simple random pruning unexpectedly performs best), whereas token merging combined with fine-tuning preserves 94.6% of retrieval performance while retaining only 2.8% of storage (Light-ColPali/ColQwen2).
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Visual Document Retrieval"
  - "ColPali"
  - "Token Merging"
  - "Token Pruning"
  - "Storage Efficiency"
date: 2026-05-08
content_hash: 9194643f42788d95
---

# Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings

**Conference**: ACL 2025  
**arXiv**: [2506.04997](https://arxiv.org/abs/2506.04997)  
**Code**: None (mentioned but not explicitly linked)  
**Area**: Information Retrieval  
**Keywords**: Visual Document Retrieval, ColPali, Token Merging, Token Pruning, Storage Efficiency

## TL;DR

This paper systematically studies compression strategies for patch-level embeddings in Visual Document Retrieval (VDR). It finds that pruning is inherently unsuitable for VDR (where simple random pruning unexpectedly performs best), whereas token merging combined with fine-tuning preserves 94.6% of retrieval performance while retaining only 2.8% of storage (Light-ColPali/ColQwen2).

## Background & Motivation

Visual Document Retrieval (VDR) encodes document pages as images into embedding vectors for retrieval, preserving layout structures and visual elements. Current state-of-the-art retrievers like ColPali/ColQwen2 generate a large number of patch-level embeddings per page (1024 for ColPali, 768 for ColQwen2). Although this achieves fine-grained perception and superior retrieval performance, it incurs massive storage overhead:

- A medium-sized document of 50 pages requires approximately 10 MB to store embedding vectors.
- In large-scale real-world deployment scenarios, storage bottlenecks severely constrain the scalability of VDR systems.

Core Problem: **How to significantly reduce the number of patch embeddings per page while minimizing the decline in retrieval performance?**

## Method

### Overall Architecture

This work systematically explores token reduction strategies from two dimensions:
1. **Token Pruning** (directly discarding some embeddings) $\rightarrow$ proven unsuitable for VDR.
2. **Token Merging** (merging multiple embeddings into one) $\rightarrow$ searching for the optimal combination across three dimensions $\rightarrow$ Light-ColPali/ColQwen2.

### Key Designs

1. **Token Pruning Experiments (Proving Infeasibility)**:

    - Three strategies are evaluated: random pruning, score-guided pruning (ranking based on response potential for synthetic queries), and attention-guided pruning (based on [EOS] token attention weights).
    - **Counter-intuitive Finding**: Random pruning unexpectedly outperforms carefully designed strategies.
    - **Root Cause Analysis**: (a) The patches of the same page activated by different queries barely overlap (only slightly more than random), making it impossible to predict which patches are important offline; (b) patch embeddings exhibit redundant grouping, and targeted pruning strategies discard entire groups, leading to informational gaps.
    - Core Conclusion: Pruning is inherently infeasible during the offline stage without query information.

2. **Three-Dimensional Exploration of Token Merging**:

    - **Merging Method** (three types): 1D spatial pooling, 2D spatial pooling, and semantic clustering (hierarchical clustering).
    - **Fine-tuning (Yes/No)**: Whether to use merged embeddings for fine-tuning during training.
    - **Merging Location** (four locations): Pre-Encoder / Post-Encoder / Post-LLM / Post-Projector.
    - Each dimension is evaluated independently before combining them into the optimal solution.

3. **Final Schema of Light-ColPali/ColQwen2**:

    - Merging Method: Semantic clustering (slightly superior to spatial pooling).
    - Fine-tuning: Yes (recovers 67% of performance loss at a merging factor of 49).
    - Merging Location: Post-Projector (merging as late as possible to preserve maximum visual information).
    - Design Motivation: The VDR scenario prioritizes storage over inference latency, allowing merging to occur at the final stage.

### Loss & Training

- Fine-tuning employs the same contrastive learning loss as ColPali/ColQwen2.
- Both training and inference stages utilize merged embeddings to calculate relevance scores.
- Fine-tuning consumes approximately 72 A100-GPU hours.

## Key Experimental Results

### Main Results (Performance of Light-ColQwen2 across Three Benchmarks)

| Method | Merging Factor | Relative Storage | ViDoRE-Info | ViDoRE-Doc | ViDoRE-Avg | Relative Perf. |
|------|---------|---------|-------------|------------|------------|---------|
| ColQwen2 (Original) | - | 64.4x | 91.5 | 55.4 | 81.4 | 100% |
| ColQwen2+Pruning | 9x | 7.6x | 85.6 | 48.3 | 74.0 | 90.9% |
| Light-ColQwen2 | 4x | 16.4x | 89.5 | 56.6 | 80.6 | **99.0%** |
| Light-ColQwen2 | 9x | 7.6x | 90.4 | 56.1 | 79.9 | **98.2%** |
| Light-ColQwen2 | 25x | 3.0x | 88.9 | 54.6 | 78.4 | 96.3% |
| Light-ColQwen2 | 49x | 1.8x | 86.9 | 52.6 | 77.0 | **94.6%** |

### Ablation Study on Merging Location (Merging Factor = 9)

| Location | Info | Doc | Arxiv | TabF | TAT | Shift | Average |
|------|------|-----|-------|------|-----|-------|------|
| Pre-Encoder | 70.2 | 29.8 | 80.0 | 74.1 | 50.5 | 49.7 | 59.1 |
| Post-Encoder | 79.5 | 41.7 | 81.9 | 80.8 | 54.1 | 54.4 | 65.4 |
| Post-LLM | 89.7 | 55.2 | 87.6 | 88.6 | 79.5 | 85.7 | **81.0** |
| Post-Projector | 90.4 | 56.1 | 86.7 | 88.8 | 79.1 | 87.3 | **81.4** |

### Key Findings

1. **Pruning is Infeasible in VDR**: Random pruning surprisingly outperforms carefully designed strategies, retaining only 58–85% of performance at a 95% pruning rate.
2. **Semantic Clustering Slightly Outperforms Spatial Pooling**: It preserves 97.5% and 92.6% of performance at merging factors of 9 and 25, respectively.
3. **Fine-tuning is Crucial**: At a merging factor of 49, fine-tuning recovers 67% of the performance loss (an absolute improvement of 8.4%).
4. **Later Merging is Better**: The average score at the Post-Projector location (81.4) significantly outperforms the Pre-Encoder location (59.1).
5. **Extreme Compression is Feasible**: Light-ColQwen2 maintains 94.6% of retrieval performance with only 2.8% of the storage overhead.

## Highlights & Insights

- **Counter-intuitive & Deep Analysis**: The discovery that random pruning outperforms elaborately designed strategies is surprising, and the root causes (unpredictability and redundancy of activated patches) are thoroughly analyzed.
- **Systematic Experimental Design**: The grid search methodology over three dimensions (merging method $\times$ fine-tuning $\times$ location) provides a valuable blueprint.
- **Key Insights on VDR vs. Generation**: Generation tasks focus on latency and FLOPs, demanding token reduction in earlier layers; conversely, VDR focuses on storage, allowing merging to be deferred to the final stage.
- **High Practical Utility**: Light-ColPali/ColQwen2 can directly serve as baseline schemes for storage-efficient VDR.

## Limitations & Future Work

- Fine-tuning still incurs a computational overhead of 72 A100-GPU hours.
- Evaluated only on ColPali and ColQwen2 retrievers; generalizability remains to be verified.
- The online computational overhead of semantic clustering (hierarchical clustering) is not thoroughly discussed.
- Joint optimization with dimensional compression (e.g., quantization) is not explored.
- Real-world deployment performance on larger-scale document corpora has not been tested.

## Related Work & Insights

- Compared to the TokenPooling work by Clavié et al. (2024), this study introduces systematic pruning analysis and fine-tuning validation.
- The key difference from LVLM generation efficiency works (such as FastV, ToMe, etc.) is that the VDR scenario is indifferent to latency but highly sensitive to storage.
- This approach is orthogonal to the product quantization dimensional compression used in ColBERTv2, and the two can be combined.
- Insight: Token reduction in VDR requires a completely different mindset from generation tasks, where the offline, query-agnostic condition serves as the core constraint.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The discovery of pruning's infeasibility is highly insightful, and the systematic three-dimensional exploration methodology is robust.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Tested on three benchmarks and two base models, comparing multiple strategies with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic; the narrative from the failure of pruning to the success of merging flows naturally.
- **Value**: ⭐⭐⭐⭐ — Provides directly applicable solutions for the real-world deployment of VDR systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](../../ACL2026/multimodal_vlm/sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](../../NeurIPS2025/multimodal_vlm/what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[ACL 2025\] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](../../CVPR2025/multimodal_vlm/docopilot_improving_multimodal_models_for_document-level_understanding.md)

</div>

<!-- RELATED:END -->
