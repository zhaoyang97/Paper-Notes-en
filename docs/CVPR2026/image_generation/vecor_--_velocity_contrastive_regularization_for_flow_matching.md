---
title: >-
  [Paper Note] VeCoR — Velocity Contrastive Regularization for Flow Matching
description: >-
  [CVPR 2026][Image Generation][Flow Matching] This paper proposes VeCoR (Velocity Contrastive Regularization), which introduces "negative velocity" contrastive signals into standard Flow Matching training. By simultaneously guiding the model on "where to go" and "where not to go," it achieves more stable trajectory evolution and higher perceptual fidelity—obtainin
tags:
  - CVPR 2026
  - Image Generation
  - Flow Matching
date: 2026-05-08
content_hash: 7012e94f8a680f29
---
# VeCoR — Velocity Contrastive Regularization for Flow Matching

**Conference**: CVPR 2026 Findings  
**arXiv**: [2511.18942](https://arxiv.org/abs/2511.18942)  
**Code**: Yes (Project Page)  
**Area**: Image Generation  
**Keywords**: Flow Matching, Contrastive Learning, Velocity Field Regularization, Negative Sample Guidance, Image Generation

## TL;DR
This paper proposes VeCoR (Velocity Contrastive Regularization), which introduces "negative velocity" contrastive signals into standard Flow Matching training. By simultaneously guiding the model on "where to go" and "where not to go," it achieves more stable trajectory evolution and higher perceptual fidelity—obtaining 22% and 35% relative FID reductions for SiT-XL/2 and REPA-SiT-XL/2 on ImageNet-1K, respectively.

## Background & Motivation
**Background**: Flow Matching (FM) has become a powerful alternative to diffusion models, transporting a prior distribution to the data distribution by learning a time-dependent velocity field. FM offers theoretical elegance and computational efficiency.

**Limitations of Prior Work**: Standard FM only provides one-sided positive supervision—training the model to "move in the right direction"—but lacks feedback on "not moving in the wrong direction." In lightweight models or low-step configurations, small inconsistencies in the velocity field accumulate errors, causing samples to deviate from the data manifold.

**Key Challenge**: The supervision of FM is directionally asymmetric (only attraction, no repulsion). When data or model capacity is limited, the learned flows in certain regions lack sufficient regularization, leading to issues like color shifts, geometric distortion, blurring, and artifacts.

**Key Insight**: Inspired by contrastive learning—since positive samples can be constructed to align representations, why not also construct "negative velocities" to repel undesirable flow directions?

**Core Idea**: Extend the FM objective from a pure attractive target to an "attraction-repulsion" bilateral training signal, regularizing the velocity field by constructing augmented negative samples in image/latent/velocity spaces.

## Method

### Overall Architecture
VeCoR aims to solve the problem where standard Flow Matching only has "attraction" but no "repulsion": the model is only told which direction to take at each step, but never which directions are wrong, making it prone to deviating in under-constrained regions when capacity is limited. The approach involves appending a contrastive regularization term to the standard FM objective—for each training sample, in addition to the original positive velocity $\hat{v}_+$ (GT velocity), a set of "seemingly reasonable but dynamically incorrect" negative velocities $\hat{v}_-$ is created. During training, the predicted velocity is pulled towards the positive velocity and pushed away from the negatives. This scheme does not modify the ODE formulation of FM and serves as a plug-and-play training add-on, compatible with various backbones such as SiT, REPA-SiT, and MMDiT.

### Key Designs

**1. Attraction-Repulsion Contrastive Loss in Velocity Space: Completing One-Sided Supervision**

The MSE of standard FM only contains an attraction term, telling the model "where to go" without any signal telling it "where not to go." VeCoR adds a repulsion term directly to the original loss:

$$\hat{\mathcal{L}}^{(\text{VeCoR})} = \frac{1}{N}\sum_{i=1}^N \left[\|v_\theta - \hat{v}_+^{(i)}\|_2^2 - \lambda \sum_{j=1}^K \|v_\theta - \hat{v}_-^{(ij)}\|_2^2\right]$$

The first term still pulls the predicted velocity $v_\theta$ towards the GT velocity, while the second term pushes it away from $K$ negative velocity directions, with $\lambda \in (0,1)$ controlling the repulsion strength. This term is significant because when data or model capacity is limited, a pure attraction target cannot regulate under-constrained regions, whereas the repulsion term provides regularization—ensuring the velocity field not only aligns with the GT but also actively avoids directions that would lead samples off the data manifold.

**2. Three Levels of Negative Velocity Candidate Sets: Generating "Reasonable but Incorrect" Negative Samples**

For the repulsion term to be effective, negative velocities cannot be pure noise—they must be semantically consistent with the current sample but dynamically incorrect to make repelling them meaningful. Drawing from SimCLR's augmentation ideas, VeCoR generates negative velocities by applying perturbations at three levels: image space (random cropping, color jittering on training images, then encoding into perturbed latent representations to calculate velocity), latent space (directly perturbing latent representations), and velocity space (directly performing channel shuffling or adding noise to the positive velocity). Experiments found that structural spatial/geometric perturbations are far more effective than appearance perturbations—color jittering only changes shallow appearance, resulting in negative velocities too similar to the positive ones to provide informative contrast; whereas structural perturbations (especially channel shuffling) disrupt feature correspondences, producing negative velocities that are "wrong in just the right way," yielding the strongest contrastive signal.

**3. Default Use of Channel Shuffle + CFG Conflict Resolution: Minimizing Overhead**

Overall, the default configuration uses Random Channel Shuffle (RCS) in the velocity space, with $K=1$ and $\lambda=0.05$. Channel shuffling changes the structural correspondence between feature channels, producing velocity directions that are structurally incorrect but globally reasonable, and it requires no extra encoding or forward passes, making it the most cost-effective source of negative samples. One detail requiring separate handling is integration with CFG: the contrastive objective of VeCoR can be understood as "pushing the predicted velocity away from the mean direction of negative trajectories," which might conflict with the unconditional guidance direction of CFG. Therefore, the correction strategy from ΔFM is adopted during sampling to resolve this conflict and prevent the two types of guidance from canceling each other out.

### Loss & Training
The method is plug-and-play, requiring no extra data or architectural changes. Compared to standard FM training, it only adds one negative velocity calculation, resulting in minimal extra overhead. It can be directly applied to any FM variant such as SiT, REPA-SiT, or MMDiT.

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
- Gains are largest on small models (SiT-S/2: FID 64→55, relative -14%), indicating particular effectiveness for capacity-constrained models.
- Spatial/geometric perturbations outperform appearance perturbations—transformations like color jittering introduce only shallow changes, insufficient for effective dynamical contrast.
- $K=2$ negative samples is the optimal number; beyond this, marginal returns diminish.
- Combined with CFG, it achieves a SOTA FID of 1.94, proving that VeCoR learns a more robust velocity field.

## Highlights & Insights
- **Transferring Contrastive Learning to Velocity Space**: Instead of performing contrast in representation space (like SimCLR), it is performed on the ODE velocity field—resulting in learned flows that are more stable and tightly fit to the data manifold.
- **Minimalist Design**: The default only requires one channel shuffle operation for negative samples and one hyperparameter $\lambda=0.05$ to bring significant and consistent improvements.
- **Complementarity with ΔFM**: ΔFM focuses on semantic discriminative power between conditions, while VeCoR focuses on the geometric stability of individual trajectories—the two are complementary.

## Limitations & Future Work
- Recall slightly decreases (0.63→0.62), suggesting contrastive repulsion might marginally limit generative diversity.
- Current negative samples are based on heuristic augmentations; learned negative sample mining could be explored.
- Verification is limited to ImageNet and COCO; high-resolution (512+) and video generation experiments are missing.
- $\lambda$ is fixed at 0.05, which may require tuning for different datasets and model scales.

## Related Work & Insights
- **vs ΔFM (Contrastive FM)**: ΔFM performs contrast between conditions to enhance semantic discrimination; VeCoR performs contrast within trajectories to enhance geometric stability—a complementary relationship.
- **vs REPA**: REPA accelerates convergence through representation alignment; VeCoR further improves upon REPA (11.14→7.28 FID), showing that velocity field regularization and representation alignment are orthogonal.
- **Transferability**: The VeCoR framework is generalizable to any ODE/SDE-based generative model—diffusion models, continuous normalizing flows, etc.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of introducing contrastive learning into velocity fields is novel yet intuitive, and the augmented negative sample design is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Detailed ablations across multiple backbones, scales, and inclusion of T2I and CFG combinations.
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis is clear, illustrations are intuitive, and formula derivations are rigorous.
- Value: ⭐⭐⭐⭐ A general plug-and-play method with broad applicability to the FM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](../../ICML2026/image_generation/stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[CVPR 2026\] From Navigation to Refinement: Revealing the Two-Stage Nature of Flow-based Diffusion Models through Oracle Velocity](from_navigation_to_refinement_revealing_the_two-stage_nature_of_flow-based_diffu.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)

</div>

<!-- RELATED:END -->
