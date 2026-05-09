---
title: >-
  [Paper Note] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective
description: >-
  [AAAI 2026][Image Generation][classifier guidance] Through systematic empirical study, this paper reveals the essential mechanism underlying both classifier guidance and classifier-free guidance — both steer denoising trajectories away from the classifier's decision boundary to achieve conditional generation — and proposes a flow matching-based post-processing method that validates this "classifier-centric" perspective on high-dimensional data.
tags:
  - AAAI 2026
  - Image Generation
  - classifier guidance
  - classifier-free guidance
  - decision boundary
  - flow matching post-processing
  - theoretical analysis of diffusion models
date: 2026-05-08
content_hash: 58d3ce60dbef75c8
---

# Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective

**Conference**: AAAI 2026
**arXiv**: [2503.10638](https://arxiv.org/abs/2503.10638)
**Code**: None
**Area**: Diffusion Models / Conditional Generation / Classifier-Free Guidance
**Keywords**: classifier guidance, classifier-free guidance, decision boundary, flow matching post-processing, theoretical analysis of diffusion models

## TL;DR

Through systematic empirical study, this paper reveals the essential mechanism underlying both classifier guidance and classifier-free guidance — both steer denoising trajectories away from the classifier's decision boundary to achieve conditional generation — and proposes a flow matching-based post-processing method that validates this "classifier-centric" perspective on high-dimensional data.

## Background & Motivation

Classifier-free guidance (CFG) has become the de facto standard for conditional generation in diffusion models — from text-to-image (Stable Diffusion) to text-to-3D (DreamFusion), CFG is indispensable in large-scale generation. Yet the community lacks a deep understanding of why CFG works.

Recent theoretical work has established a widely misunderstood fact: CFG sampling is **not** equivalent to sampling from a sharpened distribution. So what is CFG actually doing?

The core insight of this paper is: **rather than focusing solely on CFG, one should trace back to its origin — classifier guidance (CG)**. CG decomposes conditional denoising into unconditional denoising plus a classifier prediction:

$$p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1}, c) = Z \cdot p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1}) \cdot p_\theta(c|\mathbf{x}_t)$$

CFG implicitly realizes the role of the classifier by randomly dropping conditioning information. However, this decomposition relies on a key assumption (conditional forward diffusion = unconditional forward diffusion), which **does not always hold**.

## Method

### Overall Architecture

This paper presents a systematic empirical study conducted at three levels:
1. Visualizing complete denoising trajectories of CG and CFG on 1D synthetic data
2. Validating the generalizability of findings on 2D fractal data
3. Proposing a flow matching post-processing method to indirectly validate the classifier-centric perspective on high-dimensional data (MNIST, CIFAR-10)

### Key Design 1: Analysis of the Key Assumption in Classifier Guidance

The mathematical derivation of the CG decomposition relies on defining $\hat{q}(\mathbf{x}_{t+1}|\mathbf{x}_t, c) \triangleq q(\mathbf{x}_{t+1}|\mathbf{x}_t)$, i.e., assuming that the conditional forward diffusion is identical to the unconditional forward diffusion. Experiments on 1D Gaussian data ($\mathcal{N}(\pm 1.0, 0.05)$, $\mathcal{N}(\pm 0.5, 0.05)$, $\mathcal{N}(\pm 0.1, 0.05)$) show that:

- The original conditional model (left-hand side of the equation) produces **straight denoising paths**
- The CG decomposition version (right-hand side) produces **curved trajectories** that are pushed away from the decision boundary
- The greater the class distribution overlap ($\pm 0.1$), the more pronounced the discrepancy

Key finding: **CG behavior is dominated by the classifier's properties** — different classifiers (linear vs. nonlinear) produce entirely different trajectories, even when using the same initial noise and unconditional model.

### Key Design 2: CFG Also Steers Away from Decision Boundaries

The noise prediction formula for CFG is:

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t, c) = \epsilon_\theta(\mathbf{x}_t, t) + w \cdot (\epsilon_\theta(\mathbf{x}_t, t, c) - \epsilon_\theta(\mathbf{x}_t, t))$$

When $w > 1$ (the commonly used guidance scale), the additional conditioning signal is amplified. Experiments confirm that CFG likewise steers denoising trajectories away from the data's decision boundaries. This explains why high guidance scales produce high-fidelity images — they more forcefully push generations away from ambiguous regions where different conditioning signals intersect.

### Key Design 3: Flow Matching Post-Processing Validation Method

Decision boundaries cannot be directly visualized in high-dimensional data. To address this, the authors design a proxy validation method:

A rectified flow post-processing model is trained:

$$\min_{v_\theta} \int_0^1 \mathbb{E}_{\mathcal{X}} [\|(\hat{\mathbf{x}} - \text{NN}(\hat{\mathbf{x}}, \mathcal{X}_{\text{real}})) - v_\theta(\hat{\mathbf{x}}_t, c, t)\|^2] dt$$

Core Idea: if low-quality generations are indeed concentrated near decision boundaries, then a post-processing step that pushes generated samples toward their nearest real data neighbors should consistently improve quality. **The use of nearest neighbors (NN) automatically focuses on low-quality generations** — high-quality generations lie close to real data, yielding nearly zero learning signal.

### Loss & Training

- The denoising models for CG/CFG use the standard noise prediction loss
- The post-processing model uses a rectified flow objective, randomly selecting one of the top-$k$ nearest neighbors as the target ($k=20$) to avoid local optima

## Key Experimental Results

### Main Results: CIFAR-10 Post-Processing Performance (Table 2)

| CFG Scale | FID Before | FID After | Gain |
|-----------|-----------|-----------|------|
| 2.25 | 8.016 | **5.821** | -27.4% |
| 2.50 | 9.402 | **5.936** | -36.9% |
| 2.75 | 10.75 | **6.176** | -42.6% |

Post-processing consistently reduces FID across multiple guidance scales, validating the classifier-centric perspective.

### Ablation Study: NN Distance Metric Comparison (Table S1)

| Post-proc. | NN Space | CFG 2.25 | CFG 2.50 | CFG 2.75 |
|------------|----------|----------|----------|----------|
| ✗ | - | 35.77 | 41.58 | 46.37 |
| ✓ | Pixel | 22.55 | 25.96 | 28.95 |
| ✓ | DINOv2 CLS | 19.37 | 22.97 | 26.48 |
| ✓ | DINOv2 Patch | **17.27** | **20.19** | **23.32** |

Different NN distance metrics lead to significantly different post-processing outcomes, with DINOv2 patch feature space performing best.

### Key Findings

1. **CG decomposition is inexact**: The conditional diffusion model (left-hand side) and the CG decomposition (right-hand side) exhibit systematic differences in denoising trajectories, especially in class-overlapping regions
2. **Classifiers dominate CG behavior**: Different classifiers produce entirely different denoising trajectories — the effectiveness of CG depends almost entirely on the nature of the classifier
3. **CFG also steers away from decision boundaries**: Although CFG contains no explicit classifier, its behavior is consistent with CG — higher guidance scales push more forcefully away from boundaries
4. **Consistency of post-processing**: From 2D fractals to MNIST to CIFAR-10, the post-processing method consistently improves generation quality, validating the hypothesis that low-quality generations concentrate near decision boundaries
5. **A one-for-all-scales model is feasible**: A single post-processing model trained on mixed samples from multiple guidance scales generalizes to unseen scales with performance comparable to scale-specific models

## Highlights & Insights

1. **Root-cause research approach**: Rather than broadly analyzing CFG, the paper traces back to the mathematical foundations of CG, identifies the imprecision of its key assumption, and establishes a unified understanding from CG to CFG
2. **Intuitive "push away from decision boundaries" explanation**: Provides a clear geometric intuition for the guidance mechanism — conditional generation is achieved by avoiding regions of classifier uncertainty
3. **Elegant indirect validation method**: When decision boundaries cannot be directly observed in high-dimensional spaces, flow matching post-processing serves as a cleverly designed proxy for validation
4. **Complete experimental chain from 1D to CIFAR-10**: Multi-scale, multi-dataset systematic validation is highly convincing

## Limitations & Future Work

1. **Unsuccessful on ImageNet**: Defining NN distance in high-dimensional spaces is itself an open problem; different distance metrics produce drastically different nearest neighbors and post-processing outcomes
2. **Doubled inference cost**: Post-processing requires an additional round of flow matching diffusion, increasing inference time by approximately 1×
3. **Primarily empirical work**: Rigorous theoretical proofs are lacking; the "push away from decision boundaries" interpretation is largely a conjecture based on visualization and experiments
4. **Limited to the DDPM framework**: Applicability to other generative frameworks such as flow matching and consistency models has not been verified
5. **Limited practical utility of the post-processing method**: Nearest-neighbor queries require access to a real dataset, which may be restricted by privacy and copyright concerns in production settings

## Related Work & Insights

| Work | Type | Core Claim |
|------|------|------------|
| Bradley & Nakkiran (2024) | Theoretical analysis | CFG is equivalent to predictor-corrector |
| Xia et al. (2024) | Theoretical analysis | CFG does not sample from a tilted distribution |
| Chung et al. (2024) CFG++ | Method improvement | Small scale + manifold constraint |
| Lin & Yang (2024) | Training improvement | CFG is essentially a perceptual loss |
| Karras et al. (2024) Autoguidance | Method improvement | Guide with a weaker version of itself |
| **Ours** | **Empirical analysis** | **Both CG/CFG steer away from decision boundaries** |

This paper is complementary to the above theoretical works — theoretical works establish "what CFG is not," while this paper uses experiments to demonstrate "what CFG is doing."

## Rating

- **Novelty**: ⭐⭐⭐⭐ (classifier-centric perspective is novel and intuitively compelling)
- **Technical Contribution**: ⭐⭐⭐ (leans more toward empirical observation; theoretical depth is limited)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (complete gradient of validation from 1D to CIFAR-10)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (excellent visualizations, clear logical chain)
- **Practical Impact**: ⭐⭐⭐ (understanding-oriented work; direct application value is limited)
- **Overall Recommendation**: ⭐⭐⭐⭐ (3.5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](../../CVPR2026/image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](../../ICCV2025/image_generation/teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](../../ICLR2026/image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)

</div>

<!-- RELATED:END -->
