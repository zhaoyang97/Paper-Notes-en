---
title: >-
  [Paper Note] Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels
description: >-
  [ICCV 2025][3D Vision][Semantic Correspondence] This paper proposes DIY-SC, a 3D-aware pseudo-label generation strategy (chained propagation + relaxed cycle consistency + spherical prototype filtering) to train a lightweight adapter that improves semantic correspondence using foundation model features, achieving a 4.5% gain over the previous SOTA on SPair-71k (PCK@0.1 per-keypoint) without any manual keypoint annotations.
tags:
  - ICCV 2025
  - 3D Vision
  - Semantic Correspondence
  - Pseudo-Labels
  - 3D Awareness
  - Foundation Models
  - Self-Training
date: 2026-05-08
content_hash: 8122e851c81a1c86
---

# Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels

**Conference**: ICCV 2025
**arXiv**: [2506.05312](https://arxiv.org/abs/2506.05312)
**Code**: [https://genintel.github.io/DIY-SC](https://genintel.github.io/DIY-SC)
**Area**: 3D Vision / Semantic Correspondence
**Keywords**: Semantic Correspondence, Pseudo-Labels, 3D Awareness, Foundation Models, Self-Training

## TL;DR

This paper proposes DIY-SC, a 3D-aware pseudo-label generation strategy (chained propagation + relaxed cycle consistency + spherical prototype filtering) to train a lightweight adapter that improves semantic correspondence using foundation model features, achieving a 4.5% gain over the previous SOTA on SPair-71k (PCK@0.1 per-keypoint) without any manual keypoint annotations.

## Background & Motivation

**Background**: Semantic correspondence is a classical computer vision problem — finding semantically corresponding points across different instances. Recent large-scale pretrained visual models (DINO, Stable Diffusion) have demonstrated surprisingly strong zero-shot semantic matching capabilities.

**Limitations of Prior Work**:
   - Foundation model features are ambiguous on symmetric objects and repeated parts (e.g., the left and right wheels of a car are indistinguishable).
   - Supervised methods (e.g., TLR) rely on manual keypoint annotations, which are scarce and difficult to scale to larger datasets.
   - Weakly supervised methods (e.g., SphMap) address symmetry via spherical mapping, but the spherical prior performs poorly on objects with complex topology (e.g., animals) and requires manual weight tuning.

**Key Challenge**: Foundation models encode rich semantic knowledge, but simple feature concatenation or weighted averaging cannot leverage it optimally; yet effectively exploiting this knowledge requires supervision signals, while manual annotation is costly and non-scalable.

**Goal**: Without relying on manual keypoint annotation, leverage only weak 3D supervision signals (category, mask, coarse pose) to train an adapter via self-generated pseudo-labels for improved semantic correspondence.

**Key Insight**: Zero-shot matching performs well under small viewpoint differences but degrades significantly under large viewpoint gaps. Leveraging this observation, high-quality pseudo-labels are first generated on image pairs with small viewpoint differences, then propagated to large-viewpoint-gap pairs via chained propagation.

**Core Idea**: Generate pseudo-labels with foundation models → improve pseudo-label quality via 3D-aware chained propagation + cycle consistency + spherical filtering → train a lightweight adapter with high-quality pseudo-labels.

## Method

### Overall Architecture

DIY-SC consists of two stages: (1) pseudo-label generation and filtering — producing high-quality matching pairs via azimuth-based sampling, chained propagation, relaxed cycle consistency, and spherical prototype filtering; (2) supervised training — using pseudo-labels to train a lightweight adapter $f_p$ to refine the foundation model features $\tilde{\mathcal{F}} = [\mathcal{F}^{DINO}, \mathcal{F}^{SD}]$.

### Key Designs

1. **3D-Aware Image Pair Sampling and Chained Propagation**

    - Function: Generate high-quality pseudo-labels for image pairs with large viewpoint differences.
    - Mechanism: Zero-shot matching performs well (PCK@0.1 of 75.9%) when the viewpoint gap is $< 45°$, but degrades sharply (54.0%) beyond $90°$. Accordingly, a $K$-tuple $(I_1, ..., I_K)$ is constructed such that each consecutive pair has a viewpoint gap $< 90°$, and matches are propagated from "easy pairs" to "hard pairs" via recursive nearest neighbor: $\mathcal{P}^{k+1} = \text{NN}^{k \to k+1}(\mathcal{P}^k)$. $K=4$ is selected to cover a full $180°$ viewpoint range.
    - Design Motivation: Direct NN matching on large-viewpoint-gap pairs yields high error rates; chained propagation adopts a "small-step accumulation" strategy in which each step is individually reliable.

2. **Relaxed Cycle Consistency Constraint**

    - Function: Filter spurious matches introduced during chained propagation.
    - Mechanism: Standard cycle consistency requires $\text{NN}^{t \to s}(\text{NN}^{s \to t}(p_i^s)) = p_i^s$ (strict equality), but zero-shot matching rarely returns to the exact original location. This is relaxed to $\|\hat{p}_i^s - p_i^s\|_2 < r_{max}$, permitting a deviation of one feature patch, and applied iteratively at each segment of the chain.
    - Design Motivation: Strict cycle consistency rejects too many valid matches; the relaxed version retains more correct correspondences while maintaining effective filtering.

3. **Canonical Spherical 3D Prior Filtering**

    - Function: A spherical mapper $f_s$ maps DINO features onto a canonical sphere $\mathcal{S}^2$, rejecting match pairs that map to different spherical regions.
    - Mechanism: For each match pair, the spherical positions $\Psi^s = f_s(\mathcal{F}^{DINO}(\mathcal{P}^s))$ and $\Psi^t = f_s(\mathcal{F}^{DINO}(\mathcal{P}^t))$ are computed; a match is rejected if $\text{sim}(\psi_i^s, \psi_i^t) < \theta_{th}$ (with $\theta_{th} < 0.15\pi$).
    - Design Motivation: Unlike SphMap, which directly fuses spherical features into matching (degrading localization precision), this work uses spherical information solely for deletion — removing incorrect pseudo-labels without interfering with the original zero-shot matching quality. This avoids performance degradation caused by the spherical prior being ill-suited to certain object categories.

4. **Adapter Supervised Training**

    - Function: Train a 4-layer bottleneck adapter (5M parameters) with pseudo-labels to refine foundation model features.
    - Mechanism: Two loss functions are employed — a sparse contrastive loss $\mathcal{L}_{sparse} = CL(\mathcal{F}^s(\mathcal{P}^s), \mathcal{F}^t(\mathcal{P}^t))$ that maximizes similarity at matched points while minimizing it at non-matched points; and a dense loss $\mathcal{L}_{dense} = \sum \|\hat{p}_i^t - (p_i^t + \epsilon)\|_2$ that propagates gradients to unlabeled regions via WindowSoftArgmax.
    - Design Motivation: The sparse loss directly optimizes feature discriminability; the dense loss ensures that unlabeled regions of the feature map are also optimized, jointly achieving global feature improvement.

### Loss & Training

- Training uses AdamW optimizer with weight decay 0.001, learning rate $5 \times 10^{-3}$, and a one-cycle scheduler for 200K steps.
- 30K pseudo-label image pairs are generated per category, with up to 50 randomly sampled keypoints per pair.
- Input resolution: $960^2$ for SD and $840^2$ for DINOv2; feature map resolution $60 \times 60$.

## Key Experimental Results

### Main Results: PCK@0.1 on SPair-71k (per-keypoint)

| Method | Supervision | PCK@0.1 (avg) |
|--------|-------------|---------------|
| SD + DINOv2 zero-shot | None | 64.0 |
| DistillDIFT (U.S.) | Unsupervised | 65.1 |
| SphMap† | 3D weak supervision | 67.8 |
| TLR | Keypoint labels | 69.6 |
| DistillDIFT (W.S.) | Keypoint labels | 70.6 |
| **DIY-SC (Ours)** | **Pseudo-labels + 3D weak supervision** | **74.4** |
| **DIY-SC (IN3D→SPair)** | **Pseudo-labels + 3D weak supervision** | **75.1** |

The largest improvements are observed on symmetric/repeated-part categories: bus +15.7%, car +14.0%.

### Ablation Study

| Pseudo-label | Cycle Cons. | Relaxed CC | Chain Prop. | Sphere Filter | PCK@0.1 |
|--------------|-------------|------------|-------------|---------------|---------|
| | | | | | 65.0 (zero-shot) |
| ✓ | | | | | 67.2 |
| ✓ | ✓ | | | | 66.9 |
| ✓ | | ✓ | | | 68.4 |
| ✓ | | ✓ | ✓ | | 70.0 |
| ✓ | | | | ✓ | 72.9 |
| ✓ | | ✓ | ✓ | ✓ | **74.4** |

### Key Findings
- **Naive pseudo-labels are already effective** (+2.2%): Training the adapter with NN-based pseudo-labels alone improves performance, as learning to combine SD+DINO features is superior to simple concatenation.
- **Relaxed cycle consistency outperforms the strict version** (68.4 vs. 66.9): The strict version rejects too many valid matches.
- **Spherical filtering contributes the most** (+5.7% standalone, +4.4% on top of chained propagation): It effectively resolves symmetry and repeated-part ambiguity.
- **Scaling to a larger dataset (ImageNet-3D) further improves performance**: The method surpasses the previous SOTA without ever observing SPair-71k, demonstrating strong generalization.
- Cross-dataset evaluation on AP-10k also surpasses SOTA, without the severe overfitting exhibited by supervised methods (PCK@0.1: 70.6 vs. supervised 68.3).

## Highlights & Insights

- **A general paradigm of pseudo-labels with quality control**: Generate pseudo-labels using zero-shot methods, improve quality through multi-stage filtering, and then use them for supervised training. This paradigm is not limited to semantic correspondence and is transferable to other annotation-intensive tasks.
- **"Delete only, do not modulate" usage of spherical priors**: Unlike SphMap, which blends spherical and original features with weighted fusion (harming localization precision), this work uses spherical information solely to remove incorrect pseudo-labels, elegantly avoiding adverse side effects.
- **The chained propagation idea**: A hard problem is decomposed into multiple easier subproblems, each step quality-assured, cumulatively solving the globally difficult task — this divide-and-conquer strategy has broad applicability across many domains.

## Limitations & Future Work

- The spherical prior remains suboptimal for objects with complex topology, though its impact is mitigated by using it only for filtering; more flexible 3D priors could be considered.
- The method relies on coarse-grained azimuth annotations for sampling; further reducing supervision requirements remains an open direction.
- Chained propagation may accumulate errors over very long chains, limiting performance under extreme viewpoint variations.
- Validation is restricted to object-level semantic correspondence; extension to scene-level correspondence has not been explored.

## Related Work & Insights

- **vs. SphMap**: SphMap resolves symmetry by weighting spherical and original features, but performs poorly on non-rigid objects and requires manual weight tuning; DIY-SC uses the sphere only for filtering without contaminating original features, yielding better and more robust results.
- **vs. TLR**: TLR uses dataset-specific keypoint label definitions to distinguish left from right, which is non-transferable to other datasets; DIY-SC surpasses it without requiring any keypoint labels.
- **vs. DistillDIFT**: DistillDIFT fine-tunes features on 3D instance data but generalizes poorly to cross-instance matching; DIY-SC trains on cross-instance pseudo-labels, resulting in stronger generalization.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of chained propagation + relaxed cycle consistency + spherical filtering is novel, though each individual component largely builds on existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablations are highly comprehensive (per-component, per-category, cross-dataset) with in-depth qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is rigorously developed, method description is clear, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐⭐ New SOTA + scalability to larger datasets + deep methodological insights into the pseudo-label paradigm.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MARCO: Navigating the Unseen Space of Semantic Correspondence](../../CVPR2026/3d_vision/marco_semantic_correspondence.md)
- [\[ICCV 2025\] CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance](casp_improving_semi-dense_feature_matching_pipeline_leveraging_cascaded_correspo.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](monocular_semantic_scene_completion_via_masked_recurrent_networks.md)
- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)

<!-- RELATED:END -->
