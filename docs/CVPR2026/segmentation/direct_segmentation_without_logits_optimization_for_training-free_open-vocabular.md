---
title: >-
  [Paper Note] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Open-vocabulary semantic segmentation] This paper proposes an open-vocabulary semantic segmentation method that bypasses the logits optimization process entirely. Based on the assumption that ho…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Open-vocabulary semantic segmentation"
  - "training-free"
  - "distributional discrepancy"
  - "optimal transport"
  - "Markov process"
date: 2026-05-08
content_hash: 484f1a9cf8335ab5
---

# Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2604.07723](https://arxiv.org/abs/2604.07723)
**Code**: [GitHub](https://github.com/liblacklucy/DSLO)
**Area**: Image Segmentation
**Keywords**: Open-vocabulary semantic segmentation, training-free, distributional discrepancy, optimal transport, Markov process

## TL;DR

This paper proposes an open-vocabulary semantic segmentation method that bypasses the logits optimization process entirely. Based on the assumption that homogeneous regions exhibit consistent distributional discrepancies from their logits to a degenerate distribution, the method directly constructs segmentation maps via either the optimal transport path or the analytical solution of maximum transport velocity. The approach achieves state-of-the-art performance on 8 benchmarks without requiring training or model-specific modulation.

## Background & Motivation

Open-vocabulary semantic segmentation (OVSS) requires pixel-level vision-language alignment. The dominant paradigm in existing methods can be characterized as **logits optimization**: computing cosine similarity (logits) between visual and linguistic features, minimizing the discrepancy between the logits distribution and the ground-truth (GT) distribution to obtain optimal logits, and then applying argmax to produce the segmentation map. This paradigm is realized in two ways:

**Iterative training paradigm**: Requires GT annotations and time-consuming training.

**Attention modulation paradigm** (training-free): Calibrates self-attention computation to correct fine-grained alignment, but the denoising operation is **data-agnostic yet model-specific** (e.g., CLIP-specific attention substitution), limiting generalizability.

Both approaches **first derive optimal logits and then construct the segmentation map**. The authors' core insight is: can we **entirely bypass logits optimization** and directly obtain segmentation maps from distributional discrepancies themselves?

Key assumption: **Homogeneous regions exhibit consistent distributional discrepancies, while heterogeneous regions exhibit distinct ones.** If this holds, distributional discrepancies inherently encode semantic information, rendering explicit logits optimization unnecessary.

## Method

### Overall Architecture

1. Compute cosine similarity between visual and linguistic features using CLIP to obtain logits.
2. Apply non-maximum suppression (NMS) and normalization to the logits.
3. Compute the distributional discrepancy from the normalized logits to a degenerate distribution (uniform distribution $\frac{1}{N}\mathbf{1}_N$).
4. Restore the original resolution via joint bilateral upsampling (JBU).
5. Apply argmax to produce the final segmentation map.

The optimization formulation $\mathcal{Q}^* = \arg\min_\mathcal{Q} \mathbf{D}(\mathcal{P}\|\mathcal{Q})$ is reformulated as an analytical solution $\mathbf{M} = \arg\max_{N_c} \mathbf{D}(\mathcal{S}\|\mathcal{Q})$, where $\mathcal{S}$ is a degenerate distribution substituting the GT distribution.

### Key Designs

1. **Degenerate Distribution as GT Substitute (§3.3)**:

    - At inference time, the GT distribution is unavailable and must be approximated. The authors propose using a degenerate (uniform) distribution as a substitute.
    - Experimental validation demonstrates that KL divergence from logits to GT ($\mathbf{D}(\mathcal{P}\|\mathcal{Q})$) and from logits to the degenerate distribution ($\mathbf{D}(\mathcal{S}\|\mathcal{Q})$) yield highly consistent performance across 5 datasets.
    - Visualizations show that $\mathcal{S}$ and $\mathcal{P}$ occupy antipodal positions in feature space — logits optimization moves toward the GT endpoint, while the proposed method computes discrepancies to the degenerate endpoint.
    - **Design Motivation**: The degenerate distribution is the only distribution determinable at inference time without additional information.

2. **Optimal Transport Path (§3.4)**:

    - Intuition: Homogeneous regions should share consistent degeneration paths; thus, the path itself quantifies discrepancy.
    - The problem is formulated as Sinkhorn optimal transport:
    $$\boldsymbol{\pi}^* = \min_{\boldsymbol{\pi}} \sum_{i,j} \mathbf{C}_{i,j}\boldsymbol{\pi}_{i,j} - \epsilon\sum_{i,j}\boldsymbol{\pi}_{i,j}(\ln\boldsymbol{\pi}_{i,j} - 1)$$
    - The cost matrix $\mathbf{C}$ uses hierarchically averaged self-attention tensors from Stable Diffusion v2.
    - Via Lagrange multipliers, the analytical solution is: $\boldsymbol{\pi}^* = \text{diag}(\boldsymbol{\mu})\mathbf{K}\text{diag}(\boldsymbol{\nu})$, where the Gibbs kernel $\mathbf{K} = \exp(-\mathbf{C}/\epsilon)$.
    - $\boldsymbol{\mu}$ and $\boldsymbol{\nu}$ are updated via Sinkhorn iterations (50 iterations, $\epsilon=0.1$).

3. **Maximum Transport Velocity (§3.5)**:

    - Intuition: Transport velocity also quantifies discrepancy — given the same path, slower velocity implies greater discrepancy.
    - The convergence of logits to a stationary distribution is modeled as a Markov process: $\mathbf{f}^{c(l)} = \mathbf{f}^{c(0)} \cdot \mathbf{T}^l$
    - The transition matrix $\mathbf{T}$ is obtained by transforming the self-attention tensor into a doubly stochastic matrix via iterative proportional fitting (IPF, 15 iterations).
    - The maximum transport velocity for each patch is defined as the reciprocal of the number of steps to convergence: $\mathbf{v}_i^c = \max\{1/l : |\mathbf{f}_i^{c(l)} - \mathbf{f}_i^{c(l-1)}| \leq \tau\}$
    - $\tau=0.3$ is the convergence threshold.

4. **Source of Self-Attention Tensors**:

    - Self-attention from Stable Diffusion v2 is used rather than from CLIP.
    - Noise-free latent features are directly encoded; self-attention is extracted via single-step unconditional denoising.
    - Combining tensors from the $\text{up}_0$ and $\text{up}_1$ blocks yields the best results.

### Loss & Training

The method is entirely **training-free**. No training or fine-tuning is involved. Off-the-shelf CLIP (ViT-B/16 or ViT-L/14) and Stable Diffusion v2 weights are used. Inference is performed in 16-bit floating point precision on whole images without sliding windows.

## Key Experimental Results

### Main Results

**CLIP ViT-B/16 Backbone:**

| Method | Paradigm | VOC21 | Context60 | COCO-Stuff | Cityscapes | ADE20K | Avg |
|--------|----------|-------|-----------|------------|------------|--------|-----|
| SCLIP | M.M. | 59.1 | 30.4 | 22.4 | 32.2 | 16.1 | 38.2 |
| NACLIP | M.M. | 58.9 | 32.2 | 23.3 | 35.5 | 17.4 | 39.4 |
| CASS | M.M. | 65.8 | 36.7 | 26.7 | 39.4 | 20.4 | 44.4 |
| **Ours (O.P.)** | - | 66.9 | 37.6 | 28.6 | 41.7 | 22.8 | **46.2** |
| **Ours (M.V.)** | - | **67.8** | **38.3** | **28.9** | **43.3** | **23.0** | **46.9** |

**CLIP ViT-L/14 Backbone:**

| Method | VOC21 | Context60 | COCO-Stuff | Cityscapes | ADE20K | Avg |
|--------|-------|-----------|------------|------------|--------|-----|
| SC-CLIP | 65.0 | 36.9 | 26.9 | 41.3 | 21.7 | 45.2 |
| **Ours (M.V.)** | **68.9** | **38.7** | **29.2** | **43.9** | **23.4** | **47.8** |

### Ablation Study

| Configuration | VOC21 | COCO-Stuff | Cityscapes | ADE20K | Avg |
|---------------|-------|------------|------------|--------|-----|
| (I) Baseline (raw logits) | 18.6 | 7.2 | 6.7 | 3.2 | 8.9 |
| (II) +KL Divergence | 44.2 | 12.1 | 8.6 | 6.4 | 17.8 |
| (III) +NMS | 45.9 | 13.0 | 9.6 | 7.7 | 19.1 |
| (IV) +JBU | 46.3 | 13.3 | 10.1 | 8.8 | 19.6 |
| (V) +Optimal Transport Path | 66.9 | 28.6 | 41.7 | 22.8 | 40.0 |
| (VI) +Maximum Transport Velocity | **67.8** | **28.9** | **43.3** | **23.0** | **40.8** |
| (VII) Fusion of (V)+(VI) | 64.9 | 26.8 | 41.4 | 20.5 | 38.4 |

### Key Findings

1. **Distributional discrepancy can replace logits optimization**: Simple KL divergence alone yields a +8.9% mIoU gain; optimal transport/Markov process further contributes +22%.
2. **Maximum velocity slightly outperforms optimal path**: +0.7% average on B/16 and +0.6% on L/14.
3. **Fusing both modes degrades performance**: The two discrepancy measures capture different aspects (high-frequency textures vs. inter-class boundaries); naive fusion introduces interference.
4. **SD2 self-attention outperforms ViT-based models**: SD2 self-attention tensors are more effective for constructing transition matrices.
5. **Fewer denoising steps are preferable**: Encoding without noise injection ensures deterministic feature extraction.
6. **$\tau=0.3$ is the optimal threshold**: Higher thresholds cause premature degeneration before the logits distribution reaches the optimal degenerate state.

## Highlights & Insights

- **Paradigm shift**: Moving from "optimize logits then construct segmentation map" to "directly obtain segmentation map from distributional discrepancies," eliminating the need for training and model-specific modulation.
- **Theoretical elegance**: Connecting the segmentation problem to optimal transport and Markov processes provides dual geometric and probabilistic interpretations.
- **Degenerate distribution as GT substitute**: The antipodal relationship between GT and degenerate distributions in feature space is cleverly exploited, obviating the need for GT at inference time.
- **Triple freedom**: No GT annotations, no time-consuming training, and no model-specific modulation are required.
- **Complementarity of optimal path vs. maximum velocity**: The former is sensitive to high-frequency textures; the latter to inter-class boundaries.
- **Stable Diffusion as a feature extractor**: SD2 self-attention tensors are better suited for constructing inter-patch transition probabilities than those from CLIP or DINO.

## Limitations & Future Work

1. **Dependency on Stable Diffusion**: Loading the SD2 model for self-attention extraction adds memory and computational overhead at inference time.
2. **Computational cost of Sinkhorn iterations**: 50 iterations of optimal transport computation may be slow for high-resolution images.
3. **Manual tuning of $\tau$ and $\epsilon$**: Although experiments suggest relative robustness to these hyperparameters, empirical selection is still required.
4. **Fusing the two modes fails to accumulate gains**: While this is an interesting finding, it also implies a missed potential performance ceiling.
5. **Validation limited to semantic segmentation**: Applicability to more complex tasks such as panoptic and instance segmentation remains unexplored.
6. **Limited theoretical guarantees for the degenerate distribution substitute**: Feasibility is empirically validated, but rigorous theoretical analysis is absent.

## Related Work & Insights

- Contrasted with attention substitution methods such as ClearCLIP, SCLIP, and NACLIP — these remain within the logits optimization paradigm.
- VFM proxy methods such as ProxyCLIP and CASS incorporate DINO features; this work innovatively introduces SD2 self-attention.
- The application of optimal transport (Sinkhorn algorithm) to segmentation provides a geometric perspective on distributional discrepancy measurement.
- Using Markov process convergence speed as a semantic measure is a novel contribution.
- The model-agnostic nature of the method (not bound to a specific CLIP architecture) gives it potential to generalize to future vision-language models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The paradigm of bypassing logits optimization is original and convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 benchmarks, two CLIP scales, detailed ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear, though some notation is dense.
- Value: ⭐⭐⭐⭐⭐ — New SOTA for training-free OVSS with a concise and generalizable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](insid3_training-free_in-context_segmentation_with_dinov3.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)

</div>

<!-- RELATED:END -->
