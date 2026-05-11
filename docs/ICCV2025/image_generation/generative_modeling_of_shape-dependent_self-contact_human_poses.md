---
title: >-
  [Paper Note] Generative Modeling of Shape-Dependent Self-Contact Human Poses
description: >-
  [ICCV 2025][Image Generation][Self-Contact] This work constructs Goliath-SC, the first large-scale self-contact pose dataset with accurate shape annotations (383K poses / 130 subjects)…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Self-Contact"
  - "Body Shape"
  - "diffusion model"
  - "SMPL-X"
  - "Pose Generation"
  - "Pose Refinement"
date: 2026-05-08
content_hash: d2639d0c12c86643
---

# Generative Modeling of Shape-Dependent Self-Contact Human Poses

**Conference**: ICCV 2025
**arXiv**: [2509.23393](https://arxiv.org/abs/2509.23393)
**Code**: [https://tkhkaeio.github.io/projects/25-scgen](https://tkhkaeio.github.io/projects/25-scgen)
**Area**: Image Generation / Human Pose Modeling / Self-Contact Pose
**Keywords**: Self-Contact, Body Shape, diffusion model, SMPL-X, Pose Generation, Pose Refinement

## TL;DR
This work constructs Goliath-SC, the first large-scale self-contact pose dataset with accurate shape annotations (383K poses / 130 subjects), proposes PAPoseDiff—a shape-conditioned part-aware latent diffusion model for modeling body-shape-dependent self-contact pose distributions—and leverages the learned diffusion prior for monocular pose refinement, outperforming SOTA methods such as BUDDI and SMPLer-X on unseen subjects.

## Background & Motivation

### Problem Definition
Self-contact poses (e.g., touching the face, crossing arms, placing hands on the body) are extremely common in daily life and are closely related to the expression of psychological states and sign language communication. A key challenge is that self-contact poses are inherently constrained by body shape—the same action of "rubbing the belly" manifests in entirely different poses and contact regions for individuals with different body types.

### Limitations of Prior Work

**Insufficient datasets**:
- HumanSC3D: only 1K self-contact poses / 6 subjects
- MTP: 1.6K poses / 148 subjects, but lacks paired RGB images, leading to inaccurate annotations
- InterHand2.6M / Decaf: focus on local hand-hand or hand-face interactions, neglecting the influence of whole-body pose on contact

**Methodological limitations**:
- Regression-based methods (SMPLer-X): directly regress poses via ViT without contact priors
- BUDDI: learns the joint distribution of two bodies, but does not operate in latent space and lacks part-aware attention
- Existing methods model the joint pose-shape distribution without explicitly modeling the conditional relationship "pose depends on shape"

### Core Insight
Body shape (BMI, skeletal proportions, etc.) determines the feasible space of self-contact poses: a lean individual rubbing the belly adopts entirely different arm angles than a heavier individual. It is necessary to explicitly model $p(\theta | \beta)$ rather than $p(\theta, \beta)$.

## Method

### Overall Architecture
PAPoseDiff consists of three stages:
1. **Data construction**: Multi-camera dome capture from Goliath → SMPL-X fitting → contact map analysis → self-contact frame filtering
2. **Generative training**: Part-wise SMPL-X parameters → per-part autoencoder encoding into latent space → shape-conditioned and timestep-conditioned self-attention Transformer → denoising prediction
3. **Monocular refinement**: Initial SMPL-X estimate → noising to an intermediate step → conditional denoising guided by 2D keypoint fitting → refined pose output

### Key Design 1: Part-Aware Latent Diffusion (PAPoseDiff)

**Data representation**:
- Target (denoising object): $\mathbf{X} = [\theta_f, \theta_{rh}, \theta_{lh}, \theta_b]$, representing pose parameters for the face+expression, right hand, left hand, and body, respectively
- Condition: SMPL-X shape parameters $\mathbf{I} \in \mathbb{R}^{300}$

**Latent diffusion**:
- An autoencoder is trained per part to encode high-dimensional pose parameters into low-dimensional latent representations
- Diffusion is performed in latent space, as self-contact poses are constrained to a low-dimensional manifold near the body surface

**Part-aware self-attention**:
- Embeddings of face / right_hand / left_hand / body / time / shape are concatenated into a sequence
- Self-attention enables the model to learn inter-part interactions (e.g., contact relationships between hands and body)
- Learnable part embeddings are added to facilitate part-aware relational learning

### Key Design 2: Shape Condition Perturbation

Directly applying condition dropout ($c=\varnothing$, shape=0) corresponds to an "average body shape," lacking individual-specific signal. An alternative—shape parameter micro-perturbation—is adopted:

$$\mathbf{c} = \mathbf{I} + s_I \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)$$

Applied with 30% probability, with $s_I = 10^{-4}$. The underlying assumption is that individuals with similar body shapes perform similar self-contact poses. This better enhances subject diversity compared to conditional dropout.

### Key Design 3: Training Loss

$$\mathcal{L}_D = \lambda_\theta L_\theta + \lambda_v L_v + \lambda_{col} L_{col}$$

- $L_\theta$: L1 loss on pose parameters
- $L_v$: L1 loss on SMPL-X mesh vertices
- $L_{col}$: collision loss—computed using a ray-tracing collision detector (restricted to hand-related collisions to avoid false positives in regions such as the armpit)

### Key Design 4: Monocular Pose Refinement (Algorithm 1)

Given an initial estimate $\mathbf{X}_0^{init}$, denoising begins only from the last 10% of timesteps:
1. Add noise to the initial pose up to step $n_r=100$
2. Incorporate 2D keypoint fitting gradient guidance at each denoising step
3. Optional: Blended Pose Denoising—refine only the body parts of interest (e.g., upper body) while retaining the initial values for the remainder

Key advantage: no additional fine-tuning is required (in contrast to the SDS approach of InterHandGen), and the method can be directly applied to the output of any 2D/3D estimator.

## Key Experimental Results

### Dataset: Goliath-SC
- 383K self-contact poses, 130 subjects (70F / 56M / 4NB)
- Multi-camera dome with 220 RGB cameras + 3D full-body scans
- Scripted action instructions (e.g., "rub the neck"), captured at 30 Hz
- Training set: 313K; evaluation set: 9.7K (unseen subjects)

### Self-Contact Pose Generation Results

| Method | FID↓ | KID(×10⁻³)↓ | Diversity↑ | Precision↑ | Recall↑ | Col.ratio↓ |
|--------|------|-------------|------------|------------|---------|------------|
| VPoser*(shape-cond) | 9.16 | 0.882 | 3.20 | 1.0 | 0.005 | 1.37 |
| BUDDI*(shape-cond) | 2.66 | 1.12 | 5.59 | 0.995 | 0.488 | 1.47 |
| **PAPoseDiff(Ours)** | **1.25** | **0.430** | 5.98 | 0.985 | **0.708** | 1.52 |

FID is reduced by 53% relative to BUDDI*, and Recall (coverage) improves from 0.488 to 0.708.

### Ablation Study

| Model Variant | FID↓ | Diversity↑ | Col.ratio↓ |
|---------------|------|------------|------------|
| w/o Shape cond. | 2.18 | 5.52 | 1.92 |
| w/o PASA | 1.42 | 5.74 | 1.62 |
| w/o Shape rand. | 1.27 | 5.89 | 1.41 |
| w/o Anti-col. | 1.28 | 6.01 | 1.85 |
| **Full model** | **1.25** | 5.98 | 1.52 |

Key findings:
1. Shape conditioning is the most critical factor: removing it increases FID from 1.25 to 2.18 (+74%)
2. Part-aware self-attention (PASA) contributes substantially (FID: 1.42 → 1.25)
3. Anti-collision guidance effectively reduces the collision ratio (1.85 → 1.52) while preserving FID

### Monocular Pose Estimation Refinement (MPJPE mm)

| Initial Estimator | Initial | +2D fitting | +BUDDI* | +Ours(w/o shape) | +Ours |
|-------------------|---------|-------------|---------|------------------|-------|
| Hand4Whole | 126.3 | 89.5 | 74.5 | 37.9 | **35.3** |
| HybrIK-X | 82.3 | 51.8 | 65.0 | 45.9 | **32.4** |
| SMPLer-X | 58.0 | 41.7 | 71.7 | 33.7 | **31.8** |
| SMPLer-X† | 42.0 | 41.7 | - | - | - |

Key findings:
1. The PAPoseDiff prior yields significant improvements across all initial estimators, with larger gains for weaker initializations
2. Shape-conditioned refinement is particularly effective in reducing hand estimation error (e.g., HybrIK-X: 85.5 → 58.7 mm)
3. BUDDI* can even degrade performance when the initial estimate is already strong (SMPLer-X: 58.0 → 71.7), indicating that fine-grained contact priors require more precise modeling
4. The shape-conditioned prior outperforms the unconditional prior, especially for hand and body parts

## Highlights & Insights

1. **Milestone dataset**: Goliath-SC is two orders of magnitude larger than existing datasets (383K vs. 1–4K) and provides, for the first time, full-body self-contact data with accurate body shape annotations across 130 subjects of diverse body types.
2. **Conditional vs. joint generation**: This work is the first to model self-contact poses as $p(\theta|\beta)$ rather than $p(\theta,\beta)$—body shape serves as a "constraint" rather than a "degree of freedom," a hypothesis strongly validated by the experiments.
3. **Shape perturbation > Conditional dropout**: For continuous conditional variables such as SMPL-X shape parameters, micro-perturbation generalizes better than zeroing out, since zero shape corresponds to a "template body," whereas micro-perturbation corresponds to "similar body shapes."
4. **Efficient refinement**: Only the last 10% of timesteps (100 steps) are used for denoising; no additional training is required, and the approach is plug-and-play for any SMPL-X estimator.
5. **Practical optimization of collision detection**: Restricting detection to hand-related collisions avoids false positives in anatomically atypical regions such as the armpit.

## Limitations & Future Work

1. Fine-grained hand-hand interaction collisions remain difficult to fully eliminate due to the high degrees of freedom in that region.
2. The method only handles scenarios with contact; extending the framework to general pose modeling without contact (e.g., MTP) remains an open direction.
3. The dataset covers only scripted actions; spontaneous self-contact in natural settings (e.g., unconscious face-touching) is not included.
4. Only local pose is modeled (global orientation and translation are excluded), which limits applicability in certain scenarios.
5. Incorporating temporal dynamics and language-based conditioning is left for future work.

## Related Work & Insights

- The concept of **Blended Latent Diffusion** is cleverly applied to pose refinement—only the body parts of interest are refined while the remaining parts retain their original estimates.
- Core differences from BUDDI: (1) latent-space diffusion vs. parameter-space diffusion; (2) part-level self-attention vs. holistic processing; (3) conditional generation vs. joint generation.
- The ray-tracing collision detector (following Müller et al.) offers computational efficiency suitable for online loss computation during training.
- Shape interpolation experiments (Fig. 4) demonstrate a smooth shape-pose manifold—generated poses maintain plausible contact continuously as body shape transitions from heavier to leaner.

## Rating ⭐⭐⭐⭐
- **Novelty**: ⭐⭐⭐⭐ (The combination of shape-conditioned diffusion, part-aware attention, and efficient refinement constitutes a novel design)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Covers generation evaluation, estimation refinement, ablation studies, and qualitative shape interpolation analysis; the dataset itself is a significant contribution)
- **Writing Quality**: ⭐⭐⭐⭐ (Logical structure is clear, problem motivation is well-articulated, and Algorithm 1 provides a concise description of the refinement procedure)
- **Value**: ⭐⭐⭐⭐ (The prior model is plug-and-play after any SMPL-X estimator without retraining, though reproducibility is limited as the dataset has not been publicly released)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DreamDance: Animating Human Images by Enriching 3D Geometry Cues from 2D Poses](dreamdance_animating_human_images_by_enriching_3d_geometry_cues_from_2d_poses.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](aether_geometric-aware_unified_world_modeling.md)
- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](../../ICLR2026/image_generation/direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)

</div>

<!-- RELATED:END -->
