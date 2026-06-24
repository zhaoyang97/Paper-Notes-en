---
title: >-
  [Paper Note] NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval
description: >-
  [CVPR 2025][Multimodal VLM][Cross-Modal Retrieval] This paper proposes NeighborRetr, which addresses the hubness problem (where a few samples dominate nearest neighbors) in cross-modal retrieval through a triple mechanism: centrality-weighted loss (reducing training weights of hub samples), neighborhood adjustment loss (distinguishing between good/bad hubs), and uniform regularization (ensuring each sample is retrieved fairly). It achieves 49.5% (+0.9% over SOTA) R@1 on MSR-V…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Cross-Modal Retrieval"
  - "Hubness Problem"
  - "Centrality Weighting"
  - "Neighborhood Adjustment"
  - "Uniform Regularization"
date: 2026-05-08
content_hash: b279098ed4961bb2
---

# NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval

**Conference**: CVPR 2025  
**arXiv**: [2503.10526](https://arxiv.org/abs/2503.10526)  
**Code**: [https://github.com/zzezze/NeighborRetr](https://github.com/zzezze/NeighborRetr)  
**Area**: Information Retrieval  
**Keywords**: Cross-Modal Retrieval, Hubness Problem, Centrality Weighting, Neighborhood Adjustment, Uniform Regularization

## TL;DR

This paper proposes NeighborRetr, which addresses the hubness problem (where a few samples dominate nearest neighbors) in cross-modal retrieval through a triple mechanism: centrality-weighted loss (reducing training weights of hub samples), neighborhood adjustment loss (distinguishing between good/bad hubs), and uniform regularization (ensuring each sample is retrieved fairly). It achieves 49.5% (+0.9% over SOTA) R@1 on MSR-VTT text-to-video retrieval.

## Background & Motivation

### Background

**Background**: Cross-modal retrieval (e.g., text-to-video, text-to-image) maps data from different modalities into a shared embedding space. However, high-dimensional embedding spaces suffer from the hubness problem—where a small number of samples become nearest neighbors (hubs) for many queries, while the majority of samples are rarely retrieved (anti-hubs).

**Limitations of Prior Work**: (1) Hub samples contain a mix of "good hubs" (genuinely semantically relevant) and "bad hubs" (located in specific spatial positions but irrelevant), meaning one cannot simply suppress all hubs. (2) The existence of anti-hubs causes a large number of relevant samples to never be retrieved. (3) Existing contrastive learning approaches ignore the disparity between hubs and anti-hubs.

**Key Challenge**: Simply suppressing hubness erroneously penalizes good hubs (genuinely relevant high-frequency samples), whereas failing to suppress it allows bad hubs to dominate the retrieval results.

**Key Insight**: Utilize a memory bank to online estimate the centrality (retrieval frequency) of each sample, and subsequently handle good hubs, bad hubs, and anti-hubs distinctively through different mechanisms.

**Core Idea**: Centrality weighting + Good/bad hub distinction + Anti-hub uniform regularization = Solving cross-modal hubness.

### Proposed Approach

**Goal**: ### Key Designs

1. **Centrality Weighted Loss**: $w(x_i) = \exp(C(x_i)/\kappa)$, reducing the weight of high-centrality (hub) samples in the contrastive loss to decrease their dominance on learning.

2. **Neighborhood Adjustment Loss**: Differentiates good/bad hubs using "de-centralized similarity"—good hubs have high de-centralized similarity (remaining relevant after subtracting centrality), while bad hubs have low.

3. **Uniform Regularization**: $\mathcal{L}_{Opt}$ forces retrieval.


## Method

### Key Designs

1. **Centrality Weighted Loss**: $w(x_i) = \exp(C(x_i)/\kappa)$, where high-centrality (hub) samples are assigned lower weights in the contrastive loss, reducing their dominant effect on learning.

2. **Neighborhood Adjustment Loss**: Uses "de-centralized similarity" to distinguish good/bad hubs—good hubs exhibit high de-centralized similarity (remaining relevant after subtracting centrality), while bad hubs exhibit low.

3. **Uniform Regularization**: $\mathcal{L}_{Opt}$ forces the retrieval probability distribution to tend toward uniformity, ensuring anti-hubs also have opportunities to be retrieved.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{Wti} + \mathcal{L}_{Nbi} + \mathcal{L}_{Opt}$ + fine-grained WTI module. A memory bank is used for online centrality estimation.

## Key Experimental Results

| Benchmark | R@1 | Rsum | Description |
|------|-----|------|------|
| MSR-VTT T→V | **49.5%** | **207.7** | +0.9 vs HBI |
| MSR-VTT V→T | **48.7%** | **207.5** | +1.9 |
| MSVD T→V | SOTA | — | Consistently optimal across multiple benchmarks |

### Ablation Study
- Bad hubs are reduced, good hubs are enhanced, and anti-hubs are minimized—these three work in synergy.
- Decoupling intra-modal and cross-modal weighting improves stability.
- Uniform regularization yields the most significant improvement for low-frequency samples.

### Key Findings
- The hubness problem is systemic in cross-modal retrieval; without addressing it, there exists a performance ceiling of 3-5%.
- Distinguishing between good and bad hubs is a key innovation—achieving 1-2% better performance than simply suppressing all hubs.

## Highlights & Insights
- **First systematic solution to cross-modal hubness**—from theory (centrality measurement) to practice (triple loss).
- **Distinction between good and bad hubs**—not all high-frequency samples are bad; semantically relevant hubs are valuable.

## Limitations & Future Work
- The hyperparameter $\kappa$ needs to be tuned on a dataset-by-dataset basis.
- Memory bank size affects efficiency.
- Assumes single positive sample queries.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic solution to cross-modal hubness is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Provides a solution to an overlooked problem in cross-modal retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](../../ACL2025/multimodal_vlm/maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)
- [\[AAAI 2026\] Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval](../../AAAI2026/multimodal_vlm/neighbor-aware_instance_refining_with_noisy_labels_for_cross-modal_retrieval.md)
- [\[ACL 2025\] CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](../../ACL2025/multimodal_vlm/cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)
- [\[CVPR 2025\] Cross-modal Information Flow in Multimodal Large Language Models](cross-modal_information_flow_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] Intra-class Distribution-guided Generative Hashing with Neighbor Refinement for Cross-modal Retrieval](../../CVPR2026/multimodal_vlm/intra-class_distribution-guided_generative_hashing_with_neighbor_refinement_for_.md)

</div>

<!-- RELATED:END -->
