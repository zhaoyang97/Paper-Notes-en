---
title: >-
  [Paper Note] VeCoR — Velocity Contrastive Regularization for Flow Matching
description: >-
  [CVPR 2026][Image Generation][Flow Matching] This paper proposes VeCoR (Velocity Contrastive Regularization), which introduces a "negative velocity" contrastive signal into standard Flow Matching training. By simultaneously guiding the model on "where to go" and "where not to go," VeCoR achieves more stable trajectory evolution and higher perceptual fidelity—yielding relative FID reductions of 22% and 35% for SiT-XL/2 and REPA-SiT-XL/2, respectively, on ImageNet-1K.
tags:
  - CVPR 2026
  - Image Generation
  - Flow Matching
  - Contrastive Learning
  - Velocity Field Regularization
  - Negative Sample Guidance
date: 2026-05-08
content_hash: ea05b1737c72b5e4
---

# VeCoR — Velocity Contrastive Regularization for Flow Matching

**Conference**: CVPR 2026
**arXiv**: [2511.18942](https://arxiv.org/abs/2511.18942)
**Code**: Available (project page)
**Area**: Image Generation
**Keywords**: Flow Matching, Contrastive Learning, Velocity Field Regularization, Negative Sample Guidance, Image Generation

## TL;DR
This paper proposes VeCoR (Velocity Contrastive Regularization), which introduces a "negative velocity" contrastive signal into standard Flow Matching training. By simultaneously guiding the model on "where to go" and "where not to go," VeCoR achieves more stable trajectory evolution and higher perceptual fidelity—yielding relative FID reductions of 22% and 35% for SiT-XL/2 and REPA-SiT-XL/2, respectively, on ImageNet-1K.

## Background & Motivation
**Background**: Flow Matching (FM) has emerged as a powerful alternative to diffusion models, learning time-dependent velocity fields to transport a prior distribution to the data distribution. FM offers both theoretical elegance and computational efficiency.

**Limitations of Prior Work**: Standard FM provides only unilateral positive supervision—training the model to move "in the right direction" while offering no feedback on "avoiding wrong directions." Under lightweight model configurations or low-step settings, minor inconsistencies in the velocity field accumulate errors, causing samples to deviate from the data manifold.

**Key Challenge**: FM supervision is directionally asymmetric (attractive only, no repulsive force). Under limited data or model capacity, the learned flow in certain regions lacks sufficient regularization, leading to artifacts such as color shifts, geometric distortions, blurriness, and noise.

**Key Insight**: Inspired by contrastive learning—if positive pairs can be constructed to align representations, why not also construct "negative velocities" to repel undesirable flow directions?

**Core Idea**: Extend FM from a purely attractive objective to a two-sided "attraction–repulsion" training signal, regularizing the velocity field by constructing augmentation-based negative samples in image, latent, and velocity spaces.

## Method

### Overall Architecture
VeCoR is a plug-and-play training scheme that does not alter the core ODE formulation of FM. A contrastive regularization term is added on top of the standard FM objective: for each training sample, a positive velocity $\hat{v}_+$ (standard GT velocity) and a set of negative velocities $\hat{v}_-$ (obtained via augmentation perturbations) are constructed, and training simultaneously attracts the positive velocity while repelling the negative velocities.

### Key Designs

1. **VeCoR Contrastive Loss**:

    - Function: Provides two-sided velocity-space supervision for FM.
    - Mechanism: A repulsion term is added to the standard FM MSE loss:
    $\hat{\mathcal{L}}^{(\text{VeCoR})} = \frac{1}{N}\sum_{i=1}^N \left[\|v_\theta - \hat{v}_+^{(i)}\|_2^2 - \lambda \sum_{j=1}^K \|v_\theta - \hat{v}_-^{(ij)}\|_2^2\right]$
      The first term attracts the predicted velocity toward the GT, while the second term repels it from negative sample directions. $\lambda \in (0,1)$ controls repulsion strength.
    - Design Motivation: The attractive objective alone specifies "where to go"; the repulsion term specifies "where not to go"—supplying additional regularization for under-constrained regions under limited data or model capacity.

2. **Negative Velocity Candidate Construction**:

    - Function: Constructs negative velocity samples that are semantically consistent but dynamically inconsistent.
    - Mechanism: Augmentation-based perturbations are applied at three levels:
        - **Image Space (I)**: Apply random cropping, color jitter, etc. to training images → encode into perturbed latent representations → compute negative velocities.
        - **Latent Space (II)**: Apply perturbations directly to latent representations → compute negative velocities.
        - **Velocity Space (III)**: Apply operations such as channel shuffling or additive noise directly to positive velocities → obtain negative velocities.
    - Design Motivation: Inspired by the SimCLR augmentation taxonomy (spatial/geometric vs. appearance transforms), adapted to velocity space. Experiments show that spatial/geometric transforms (especially channel shuffling) are more effective than appearance transforms, as structural perturbations yield more informative dynamic negative samples.

3. **Default Configuration and Scalability**:

    - Function: Provides an efficient default setup and a CFG integration scheme.
    - Mechanism: The default configuration uses Random Channel Shuffle (RCS) in velocity space with $K=1$ and $\lambda=0.05$. When integrating with CFG, a conflict must be resolved—the VeCoR contrastive objective can be interpreted as moving predicted velocities away from the mean of negative trajectories, which may conflict with the unconditional guidance direction of CFG; the correction strategy from ΔFM is adopted.
    - Design Motivation: Channel shuffling disrupts the structural correspondence between feature channels, producing velocities that are structurally incorrect yet broadly plausible—making it the most effective and lowest-overhead negative sample strategy.

### Loss & Training
- Fully plug-and-play: no additional data or architectural modifications required.
- Only one negative velocity computation is added; the overhead relative to standard FM training is minimal.
- Compatible with arbitrary FM variants (SiT, REPA-SiT, MMDiT).

## Key Experimental Results

### Main Results — ImageNet-1K 256×256 (50 NFEs)

| Model | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|------|------|-----|-------|--------|-------|
| SiT-XL/2 | 20.01 | 74.15 | 8.45 | 0.63 | 0.63 |
| +ΔFM | 16.32 | 78.07 | 5.08 | 0.66 | 0.63 |
| **+VeCoR** | **15.56** | **80.96** | **4.70** | **0.67** | 0.62 |
| REPA-SiT-XL/2 | 11.14 | 115.83 | 8.25 | 0.67 | 0.65 |
| **+VeCoR** | **7.28** | **127.90** | **5.17** | **0.71** | 0.64 |

### MS-COCO T2I Experiments

| Method | ODE (Heun) CFG=2.0 | SDE (E-M) CFG=2.0 |
|------|---------------------|---------------------|
| M+R (baseline) | 5.03 | 6.03 |
| +ΔFM | 5.16 | 4.78 |
| **+VeCoR (RCR)** | **4.82** | **4.55** |

### Ablation Study — Perturbation Space and Operation Type

| Perturbation Space | Best Operation | FID (SiT-S/2) |
|---------|---------|--------------|
| Velocity Space | Channel Shuffle | **55.13** (baseline 64.26) |
| Latent Space | Random Crop | ~57 |
| Image Space | Random Crop | ~58 |

### Key Findings
- The largest gains are observed on smaller models (SiT-S/2: FID 64→55, relative −14%), indicating particular effectiveness for capacity-constrained models.
- Spatial/geometric perturbations outperform appearance perturbations—color jitter and similar operations introduce only shallow variations, insufficient to provide effective dynamic contrastive signals.
- $K=2$ negative samples yields the optimal count; additional negatives produce diminishing returns.
- Combined with CFG, the method achieves a state-of-the-art FID of 1.94, demonstrating that VeCoR learns a more robust velocity field.

## Highlights & Insights
- **Transferring contrastive learning to velocity field space**: Rather than performing contrastive learning in representation space (as in SimCLR), VeCoR applies contrastive objectives to ODE velocity fields—resulting in learned flows that more stably and compactly adhere to the data manifold.
- **Minimalist design**: By default, a single channel shuffling operation serves as the negative sample generator and a single hyperparameter $\lambda=0.05$ suffices, yet consistently yields substantial improvements.
- **Complementarity with ΔFM**: ΔFM focuses on inter-condition semantic discriminability, while VeCoR focuses on the geometric stability of individual trajectories—the two approaches are complementary.

## Limitations & Future Work
- Recall decreases slightly (0.63→0.62), suggesting that contrastive repulsion may marginally constrain generation diversity.
- Current negative samples are entirely heuristic augmentations; learned negative sample mining warrants exploration.
- Validation is limited to ImageNet and COCO; high-resolution (512+) and video generation experiments are absent.
- $\lambda$ is fixed at 0.05 and may require adjustment across different datasets and model scales.

## Related Work & Insights
- **vs. ΔFM (Contrastive FM)**: ΔFM applies inter-condition contrastive objectives to enhance semantic discriminability; VeCoR applies intra-trajectory contrastive objectives to enhance geometric stability—the two are complementary.
- **vs. REPA**: REPA accelerates convergence through representation alignment; VeCoR further improves upon REPA (FID 11.14→7.28), demonstrating that velocity field regularization and representation alignment are orthogonal.
- **Transferability**: The VeCoR framework generalizes to any ODE/SDE-based generative model, including diffusion models and continuous normalizing flows.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of introducing contrastive learning into the velocity field is novel yet intuitively natural; the augmentation-based negative sample design is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple backbones, multiple scales, detailed ablations, and T2I and CFG combination experiments are included.
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis is clear, illustrations are intuitive, and mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐ A general plug-and-play method with broad applicability to the FM community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)

<!-- RELATED:END -->
