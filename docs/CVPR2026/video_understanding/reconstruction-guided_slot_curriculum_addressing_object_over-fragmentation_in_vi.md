---
title: >-
  [Paper Note] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning
description: >-
  [CVPR 2026][Video Understanding][Object-centric representation] This paper proposes SlotCurri, a reconstruction-guided slot-count curriculum learning strategy that begins training with very few slots and progressively ex…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Object-centric representation"
  - "over-fragmentation"
  - "curriculum learning"
  - "Slot Attention"
  - "video segmentation"
date: 2026-05-08
content_hash: 408e6ce6253bdd6f
---

# Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning

**Conference**: CVPR 2026
**arXiv**: [2603.22758](https://arxiv.org/abs/2603.22758)  
**Code**: [GitHub](https://github.com/wjun0830/SlotCurri)  
**Area**: Video Understanding / Object-Centric Learning
**Keywords**: Object-centric representation, over-fragmentation, curriculum learning, Slot Attention, video segmentation

## TL;DR

This paper proposes SlotCurri, a reconstruction-guided slot-count curriculum learning strategy that begins training with very few slots and progressively expands slot capacity only in regions with high reconstruction error. Combined with structure-aware loss and cyclic inference, SlotCurri effectively addresses the over-fragmentation problem — where a single object is erroneously split across multiple slots — in video object-centric learning, achieving a +6.8 FG-ARI improvement on YouTube-VIS.

## Background & Motivation

Video object-centric learning (VOCL) aims to decompose raw videos into compact object slot representations, providing foundations for downstream tasks such as scene understanding and video segmentation. However, existing methods suffer from severe **over-fragmentation**:

- **Key Challenge**: Models are implicitly encouraged to exploit all available slots to minimize the reconstruction objective — larger slot budgets generally yield better reconstruction quality, causing multiple slots to collaboratively represent a single object.
- **Practical Harm**: A single object is split across multiple slots, breaking the one-to-one correspondence between slots and objects, and degrading interpretability and computational efficiency.
- **Limitations of Prior Work**: SOLV adopts a produce-then-merge strategy, but the merging stage can fail (contrastive learning has already pushed slots toward encoding distinct representations).

**Core Idea**: Rather than patching fragmentation after the fact, prevent it at the source — treat slot count as a curriculum variable, increasing it progressively from few to many, ensuring new slots are assigned only to regions that genuinely require greater representational capacity.

## Method

### Overall Architecture

Built upon the SlotContrast baseline, comprising four core components:
1. Reconstruction-guided slot curriculum learning (SlotCurri)
2. Structure-aware reconstruction loss (SSIM3D)
3. Cyclic inference
4. Temporal consistency contrastive learning (inherited from SlotContrast)

### Key Designs

1. **Reconstruction-Guided Slot Curriculum Learning**: Training begins with $K_{\text{init}}=2$ slots and expands progressively over $M=3$ stages. At each stage transition:
    - The weighted reconstruction error per slot is computed: $\delta^{(k)} = \sum_{t,h,w} \alpha^{(k,t,h,w)} \cdot \mathcal{L}_{\text{MSE}}^{(t,h,w)}$
    - Child slot counts are allocated proportionally to error magnitude (higher error → more children), with deterministic rounding to ensure the total count is exact.
    - Child slots are initialized via **distance-aware noise perturbation**: $\hat{\mathbf{s}}^{(k^*)} = \hat{\mathbf{s}}^{(k)} + \beta \cdot d_{\text{nearest}}^{(k)} \cdot \frac{\|\hat{\mathbf{s}}^{(k)}\|}{\mu_{\text{norm}}} \cdot \mathbf{v}$, where the noise magnitude is proportional to the parent slot's distance to its nearest neighbor and its relative feature norm, ensuring children explore under-represented regions rather than duplicating the parent.

2. **Structure-Aware Reconstruction Loss (SSIM3D)**: MSE processes each pixel independently, blurring spatial details and object boundaries — a problem especially pronounced in early stages with few slots. SSIM is computed over $3 \times 3 \times 3$ spatiotemporal windows, explicitly preserving local contrast and edge information. The final loss is: $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{SSC}} \mathcal{L}_{\text{SSC}} + \lambda_{\text{SSIM3D}} \mathcal{L}_{\text{SSIM3D}}$

3. **Cyclic Inference**: Applied only during inference — slots are first propagated forward to the last frame and then backward to the first frame. The backward-propagated slot representations are used for mask decoding, allowing early frames to leverage future contextual information. Inference time increases by only 0.3% (286s → 287s).

### Loss & Training

- Total loss: MSE reconstruction + SlotContrast contrastive + SSIM3D structural
- Curriculum schedule: slots are expanded at 10% and 25% of total iterations
- Accelerated slot growth rule: $K^{(m)} = K_{\text{init}} + m \cdot \sigma + 3m(m-1)/2$
- $\sigma$ is adjusted per dataset (YouTube-VIS: 1, MOVi-C: 3, MOVi-E: 5) to ensure the final slot count matches the baseline
- Hyperparameters: $\beta=0.2$, $\lambda_{\text{SSIM3D}}=0.05$, consistent across datasets
- Hardware: 2 × NVIDIA RTX A6000

## Key Experimental Results

### Main Results

| Method | YouTube-VIS FG-ARI↑ | YouTube-VIS mBO↑ | MOVi-C FG-ARI↑ | MOVi-E FG-ARI↑ |
|--------|---------------------|-------------------|-----------------|-----------------|
| STEVE | 15.0 | 19.1 | 36.1 | 50.6 |
| VideoSAUR | 28.9 | 26.3 | 64.8 | 73.9 |
| SlotContrast | 38.0 | 33.7 | 69.3 | 82.9 |
| **SlotCurri** | **44.8±1.2** | **35.5±2.2** | **77.6±0.9** | **83.7±0.2** |

Comparison with anti-fragmentation methods (Image FG-ARI):

| Method | MOVi-C | MOVi-E |
|--------|--------|--------|
| AdaSlot | 75.6 | 76.7 |
| SOLV | — | 80.8 |
| **SlotCurri** | **81.6** | **84.9** |

### Ablation Study

Component contributions on YouTube-VIS:

| Simple Curriculum | Recon.-Guided | SSIM | Cyclic Inference | FG-ARI | mBO |
|-------------------|---------------|------|------------------|--------|-----|
| — | — | — | — | 36.1 | 32.7 |
| ✓ | — | — | — | 38.8 | 32.3 |
| — | ✓ | — | — | 42.6 | 33.7 |
| — | ✓ | ✓ | — | 43.6 | 35.2 |
| — | ✓ | ✓ | ✓ | **44.8** | **35.5** |

Hyperparameter sensitivity:
- Number of curriculum stages $M$: $M=3$ is optimal (44.8); $M=2$ is insufficient (41.5); $M=4$ yields a slight regression (44.7).
- Perturbation coefficient $\beta$: $\beta=0.2$ is best (44.8); too small ($0.1$: 42.8) causes children to closely resemble parents; too large ($0.3$: 40.2) introduces destructive noise.
- SSIM coefficient $\lambda$: 0.05 is optimal; higher values (0.07) are harmful.

### Key Findings

- A simple curriculum alone (random initialization of new slots) yields a +2.7 FG-ARI gain, confirming the effectiveness of progressive expansion itself.
- Reconstruction-guided allocation contributes a further +3.8 over the simple curriculum, demonstrating that purposeful slot assignment is significantly superior to random allocation.
- The degree of over-fragmentation metric (DOF@0.5) decreases from 1.38 to 1.26, directly validating the reduction in fragmentation.
- Object identification recall (OIR@0.5) improves from 24.9% to 30.3% while simultaneously reducing fragmentation.
- Gains on MOVi-E are smaller, as the primary challenge in this dataset is under-fragmentation (too many small objects) rather than over-fragmentation.

## Highlights & Insights

- **Elegant design philosophy**: "Prevention over remediation" — rather than fragmenting first and merging later, slot allocation is controlled at the source.
- **Distance-aware noise initialization** is carefully designed: noise magnitude is proportional to the nearest-neighbor distance, ensuring children inherit parent information while being directed to explore new regions.
- **Synergy between SSIM3D and curriculum learning**: SSIM sharpens semantic boundaries during early low-slot stages, so subsequent slot expansions build upon already well-separated semantic foundations.
- **Cyclic inference is extremely lightweight** (+0.3% inference time) yet effectively compensates for the lack of future context in early frames.

## Limitations & Future Work

- Gains are limited in scenarios requiring fine-grained segmentation of many small objects (e.g., MOVi-E), as the approach does not address under-fragmentation.
- The number of curriculum stages and expansion timing are currently fixed manually; scene-adaptive scheduling strategies remain unexplored.
- The initial slot count is fixed at 2, which may be insufficient for scenes with a very large number of objects.
- Validation is limited to the DINOv2 backbone; compatibility with other visual foundation models is unknown.
- The mBO metric on synthetic datasets does not surpass VideoSAUR, possibly because the latter directly models motion patterns and thus holds an advantage in synthetic settings.

## Related Work & Insights

- **vs. SOLV**: SOLV over-produces then merges; SlotCurri constrains then expands — the latter is more fundamental in preventing fragmentation.
- **Curriculum learning tradition**: Treating sample difficulty as a curriculum variable (Bengio 2009); this paper innovatively treats slot count as the curriculum variable.
- **vs. AdaSlot**: AdaSlot adaptively adjusts slot count but does not consider where to allocate; SlotCurri performs targeted allocation guided by reconstruction error.
- **Broader inspiration**: The reconstruction-guided capacity expansion strategy may generalize to other structured representation learning paradigms (e.g., node expansion in capsule networks or graph neural networks).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Slot-count curriculum learning combined with reconstruction-guided expansion is a novel and effective solution to over-fragmentation, though individual component designs are relatively intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on three real and synthetic datasets with comprehensive ablations; introduces new OIR and DOF metrics to quantitatively verify fragmentation reduction.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is articulated with exceptional clarity; method derivation is progressive and well-structured; visualizations are rich and intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a practical training paradigm for the VOCL community, though the application scope is relatively specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [\[ICLR 2026\] From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](../../ICLR2026/video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)
- [\[AAAI 2026\] Predicting Video Slot Attention Queries from Random Slot-Feature Pairs](../../AAAI2026/video_understanding/predicting_video_slot_attention_queries_from_random_slot-feature_pairs.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)
- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)

</div>

<!-- RELATED:END -->
