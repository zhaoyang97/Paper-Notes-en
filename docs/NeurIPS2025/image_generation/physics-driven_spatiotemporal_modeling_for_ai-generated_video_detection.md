---
title: >-
  [Paper Note] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection
description: >-
  [NeurIPS 2025 Spotlight][Image Generation][AI-generated video detection] A physics conservation law-based paradigm for AI-generated video detection is proposed. A normalized spatiotemporal gradient (NSG) statistic is defined to capture the ratio of spatial probability gradients to temporal density changes. Pre-trained diffusion models are used to estimate NSG, and detection is performed via MMD. The method surpasses the state of the art by 16% in Recall and 10.75% in F1.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Image Generation"
  - "AI-generated video detection"
  - "probability flow conservation"
  - "normalized spatiotemporal gradient"
  - "diffusion models"
  - "MMD"
date: 2026-05-08
content_hash: 56db4a861f77ed1e
---

# Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.08073](https://arxiv.org/abs/2510.08073)  
**Authors**: Shuhai Zhang, Zihao Lian, Jiahao Yang, Daiyuan Li (SCUT), Guoxuan Pang (USTC), Feng Liu (U Melbourne), Bo Han (HKBU), Shutao Li (HNU), Mingkui Tan (SCUT)
**Code**: [ZSHsh98/NSG-VD](https://github.com/ZSHsh98/NSG-VD)  
**Area**: Image Generation
**Keywords**: AI-generated video detection, probability flow conservation, normalized spatiotemporal gradient, diffusion models, MMD

## TL;DR

A physics conservation law-based paradigm for AI-generated video detection is proposed. A normalized spatiotemporal gradient (NSG) statistic is defined to capture the ratio of spatial probability gradients to temporal density changes. Pre-trained diffusion models are used to estimate NSG, and detection is performed via MMD. The method surpasses the state of the art by 16% in Recall and 10.75% in F1.

## Background & Motivation

### State of the Field
AI video generation technologies (e.g., Sora) have reached near-perfect visual realism, making the detection of AI-generated videos an urgent need for maintaining trust in digital media. The core challenges are: (1) videos contain complex spatial texture structures and temporal motion trajectories, requiring a joint modeling framework; and (2) differences between AI-generated and real videos in visual appearance and temporal evolution are increasingly subtle.

### Limitations of Prior Work
- **Artifact-based methods** (optical flow modeling, appearance consistency analysis) rely on generator-specific artifact features and fail against high-quality generative models such as Sora.
- **DeMamba** achieves only 40.60% Recall on HotShot and 48.21% Recall on Sora.
- **STIL** collapses completely in critical scenarios (HotShot 1.40% Recall, Sora 1.79% Recall).
- **TALL** achieves only 25.00% Recall on Sora.
- Existing methods neglect the physics-constrained spatiotemporal evolution dynamics inherent in natural videos.

### Root Cause
Natural videos inherently obey physical laws such as motion coherence and texture continuity, whereas AI-generated videos frequently exhibit systematic inconsistencies that violate physical constraints. This paper asks: can the intrinsic spatiotemporal dynamics of natural videos be modeled through physical conservation laws, thereby exposing synthetic anomalies?

## Method

### Probability Flow Velocity Field Modeling
Video evolution is modeled as a fluid-dynamics-like process. The probability flux density is defined as $\mathbf{J}(\mathbf{x},t) = p(\mathbf{x},t) \cdot \mathbf{v}(\mathbf{x},t)$, where $p(\mathbf{x},t)$ is the probability density and $\mathbf{v}(\mathbf{x},t)$ is the velocity field guiding the flow of probability mass. Conservation of probability mass implies the continuity equation:

$$\frac{\partial p(\mathbf{x},t)}{\partial t} + \nabla_{\mathbf{x}} \cdot \mathbf{J}(\mathbf{x},t) = 0$$

Substituting $\mathbf{J}$ and taking the logarithm, and applying the incompressible flow approximation (the divergence term $\nabla_\mathbf{x} \cdot \mathbf{v}$ is a sub-dominant term), yields:

$$\mathbf{v}(\mathbf{x},t) \cdot \nabla_{\mathbf{x}} \log p(\mathbf{x},t) \approx -\partial_t \log p(\mathbf{x},t)$$

### Normalized Spatiotemporal Gradient (NSG)
Because the solution for the velocity field $\mathbf{v}$ is non-unique, its dual field—the normalized spatiotemporal gradient—is defined as:

$$\mathbf{g}(\mathbf{x},t) = \frac{\nabla_{\mathbf{x}} \log p(\mathbf{x},t)}{-\partial_t \log p(\mathbf{x},t) + \lambda}$$

where $\lambda > 0$ prevents numerical instability. NSG satisfies $\mathbf{v} \cdot \mathbf{g} \approx 1$, circumventing the ill-posed inversion of $\mathbf{v}$ while retaining the key information of spatiotemporal gradient dynamics.

**Physical Interpretation**: NSG quantifies the sensitivity of the probability flow direction per unit of temporal change, simultaneously capturing spatial irregularities (via $\nabla_\mathbf{x} \log p$) and temporal inconsistencies (via $\partial_t \log p$).

### NSG Estimation via Diffusion Models
The gradient estimation capability of pre-trained diffusion models is exploited:

- **Spatial gradient**: The score network $\mathbf{s}_\theta$ directly approximates $\nabla_\mathbf{x} \log p(\mathbf{x},t) \approx \mathbf{s}_\theta(\mathbf{x}_t)$.
- **Temporal derivative**: Based on the brightness constancy assumption (optical flow constraint), $\partial_t \log p(\mathbf{x},t) \approx -\nabla_\mathbf{x} \log p(\mathbf{x},t) \cdot \frac{\Delta\mathbf{x}}{\Delta t}$.

The resulting NSG estimator is:

$$\mathbf{g}(\mathbf{x},t) \approx \frac{\mathbf{s}_\theta(\mathbf{x}_t)}{\mathbf{s}_\theta(\mathbf{x}_t) \cdot \frac{\mathbf{x}_{t+\Delta t} - \mathbf{x}_t}{\Delta t} + \lambda}$$

No explicit optical flow computation is required; only a single forward pass of the diffusion model plus frame differencing suffices.

### NSG-VD Detection Method
1. Aggregate NSG features $\mathbf{G}(\mathbf{x}) = \{\mathbf{g}(\mathbf{x},t)\}_{t=1}^T$ across all frames of the video.
2. Compute the distributional discrepancy between the test video's NSG and the NSG of a reference set of real videos using a deep kernel MMD.
3. Apply threshold $\tau$: if $\widehat{\text{MMD}}_b^2 > \tau$, classify as Fake.

The core deep kernel combines a learnable feature mapping $\phi_\mathbf{G}$ with a Gaussian kernel, optimized via multi-population perception (MPP) to maximize detection power.

### Theoretical Guarantees
Assuming real videos $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \sigma(t)^2\mathbf{I}_d)$ and generated videos $\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma(t)^2\mathbf{I}_d)$, it is proven that the upper bound on NSG feature distance grows with the distributional shift $\varphi = \|\boldsymbol{\mu}\|^2/\sigma(t)^2$. This guarantees that the MMD between real videos is smaller than the MMD between real and generated videos, providing the theoretical foundation for NSG-VD.

## Key Experimental Results

### Experiment 1: Standard Evaluation (Trained on Pika)

Evaluated on the GenVideo benchmark, trained on 10,000 videos each from Kinetics-400 (real) and Pika (generated).

| Method | Avg Recall | Avg Accuracy | Avg F1 | Avg AUROC |
|--------|-----------|-------------|--------|-----------|
| DeMamba | 72.02 | 84.21 | 80.12 | 93.88 |
| NPR | 57.35 | 77.96 | 68.39 | 93.02 |
| TALL | 60.78 | 79.85 | 72.63 | 95.67 |
| STIL | 27.02 | 63.51 | 35.82 | 93.49 |
| **NSG-VD** | **88.02** | **91.46** | **90.87** | **96.14** |

Key comparisons: NSG-VD achieves 78.57% Recall on Sora (vs. DeMamba 48.21%) and 92.50% on HotShot (vs. DeMamba 40.60%).

### Experiment 2: Class-Imbalanced Setting

Trained on only 1,000 generated videos (SEINE) + 10,000 real videos, simulating real-world scarcity of generated samples.

| Method | Avg Recall | Avg Accuracy | Avg F1 | Avg AUROC |
|--------|-----------|-------------|--------|-----------|
| DeMamba | 64.09 | 81.60 | 76.44 | 94.85 |
| NPR | 32.71 | 66.09 | 46.54 | 87.10 |
| TALL | 36.08 | 67.95 | 51.40 | 91.96 |
| STIL | 46.78 | 73.21 | 61.43 | 90.20 |
| **NSG-VD** | **93.21** | **89.16** | **89.48** | **94.91** |

With only 1/10 of the generated training data, NSG-VD still achieves 93.21% Recall, surpassing DeMamba by 29.12%, and reaches 82.14% Recall on Sora (vs. DeMamba 33.93%).

### Ablation Study: Spatial Gradient vs. Temporal Derivative

| Component | Recall | Accuracy | F1 | AUROC |
|-----------|--------|----------|-----|-------|
| Spatial gradient only | 87.99 | 82.84 | 83.40 | 91.85 |
| Temporal derivative only | 60.35 | 71.09 | 66.97 | 78.95 |
| **NSG-VD (both combined)** | **88.02** | **91.46** | **90.87** | **96.14** |

The spatial gradient is the primary contributor, but combining it with the temporal derivative improves F1 from 83.40% to 90.87% (+7.47%), validating the necessity of their synergy under the physical conservation principle.

## Highlights & Insights

- **A novel physics-driven paradigm**: For the first time, probability flow conservation laws are introduced into AI-generated video detection. NSG statistics model the intrinsic spatiotemporal dynamics of natural videos rather than relying on generator-specific artifacts.
- **Elegant estimator design**: Spatial gradients are estimated via the diffusion model score function, and temporal derivatives via the brightness constancy constraint, avoiding complex optical flow computation and requiring only a single forward pass.
- **Strong generalization**: Significantly outperforms the state of the art across 10 diverse generators (including closed-source Sora), maintaining 93%+ Recall even under class imbalance (1/10 generated data).
- **Solid theoretical grounding**: The quantitative relationship between NSG feature distances of real/generated videos and distributional shift is rigorously proven, providing theoretical justification for detection efficacy.
- **Threshold robustness**: Performance is stable for $\tau \in [0.7, 1.1]$, requiring no fine-grained hyperparameter tuning.

## Limitations & Future Work

- **Gaussian distribution assumption**: The theoretical analysis (Theorem 1) relies on a Gaussian distribution assumption, whereas real video distributions are far more complex, and the theoretical bound may not be tight.
- **Incompressible flow approximation**: Neglecting the divergence term is heuristic and may not hold for rapid scene transitions or large motion.
- **Dependence on diffusion models**: A pre-trained diffusion model is required as the score estimator, entailing greater computational cost than traditional methods.
- **Brightness constancy assumption**: This assumption may fail under strong illumination changes or occlusion, degrading temporal derivative estimation.
- **Reference set dependency**: Detection requires maintaining a reference set of real videos; the choice and size of this set in deployment will affect performance.
- **Accuracy slightly lower than Recall**: Under the SEINE training setting, Accuracy (86.05%) falls below that of some baselines, indicating a non-negligible false positive rate.

## Related Work & Insights

- **DeMamba (Chen et al., 2024)**: Mamba-based spatiotemporal relationship modeling relying on large-scale supervised training, with insufficient generalization to unseen generators (Sora 48.21% Recall vs. NSG-VD 78.57%).
- **TALL (Xu et al., 2023)**: Spatiotemporal modeling via thumbnail layouts, but unstable on closed-source models (Sora 25.00% Recall).
- **STIL (Gu et al., 2021)**: Separately models spatial and temporal inconsistencies, but collapses completely on novel generators (HotShot 1.40% Recall).
- **NPR (Tan et al., 2024)**: Deepfake detection based on CNN upsampling operations, with large performance variance (Accuracy 57.20%–98.20%).
- **DIRE (Wang et al., 2023)**: Detects generated images via diffusion model reconstruction error, but does not address spatiotemporal dynamics modeling.
- **Score-based detection (Song et al., 2025; Zhang et al., 2024)**: Uses score statistics to detect AI-generated text/images; this paper extends the approach to the video domain and introduces physical constraints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of fluid-mechanics probability flow conservation to video detection; the NSG statistic is elegantly defined with strong physical intuition.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 10 generators, 3 training settings, and comprehensive ablations, though ablations over additional backbones and diffusion models are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from physical modeling to statistic definition, estimator derivation, and theoretical guarantees is complete and coherent.
- Value: ⭐⭐⭐⭐⭐ — Opens a new physics-driven direction for AI-generated video detection with substantial practical performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content](../../CVPR2025/image_generation/wheres_the_liability_in_the_generative_era_recovery-based_black-box_detection_of.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](../../ECCV2024/image_generation/zero-shot_detection_of_ai-generated_images.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](../../CVPR2025/image_generation/any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[NeurIPS 2025\] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)

</div>

<!-- RELATED:END -->
