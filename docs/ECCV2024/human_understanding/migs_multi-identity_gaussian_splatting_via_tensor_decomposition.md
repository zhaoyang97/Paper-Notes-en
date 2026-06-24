---
title: >-
  [Paper Note] MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition
description: >-
  [ECCV 2024][Human Understanding][3D Gaussian Splatting] MIGS is proposed to unify the 3DGS parameters of multiple human identities into a single low-rank tensor via CP tensor decomposition, significantly reducing parameter size while achieving robust animation for unseen poses.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "3D Gaussian Splatting"
  - "Multi-identity Representation"
  - "Tensor Decomposition"
  - "Human Animation"
  - "Monocular Video"
date: 2026-05-08
content_hash: 8f4a05b929ae469a
---

# MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition

**Conference**: ECCV 2024  
**arXiv**: [2407.07284](https://arxiv.org/abs/2407.07284)  
**Code**: [Project Page](https://aggelinacha.github.io/MIGS/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Multi-identity Representation, Tensor Decomposition, Human Animation, Monocular Video

## TL;DR

MIGS is proposed to unify the 3DGS parameters of multiple human identities into a single low-rank tensor via CP tensor decomposition, significantly reducing parameter size while achieving robust animation for unseen poses.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has been successfully applied to human avatar modeling, enabling real-time rendering with high visual quality. Existing methods like 3DGS-Avatar and GauHuman combine 3DGS with the SMPL human prior to learn animatable human representations from monocular videos.

**Limitations of Prior Work**: Currently, all 3DGS human avatar methods rely on **per-identity optimization**—each individual requires a independently trained model. This leads to: (a) a linear explosion of parameters in multi-person scenes, where $N_i$ individuals require $N_i \times N_g \times M$ parameters; (b) limited training data for a single identity, resulting in a sharp decline in animation quality under out-of-distribution (OOD) poses.

**Key Challenge**: Per-identity models can only learn limited human deformation patterns, yet real-world applications require animation robustness under extreme poses (e.g., highly challenging dances). Joint multi-identity learning can share deformation knowledge, but how can this be achieved without exploding the parameter count?

**Goal**: Learn a **unified multi-identity 3DGS representation** from monocular videos to simultaneously compress parameters and enhance animation robustness under OOD poses through cross-identity knowledge sharing.

**Key Insight**: Drawing inspiration from the classic TensorFaces, the Gaussian parameters of all identities can be organized into a high-order tensor, and low-rank approximation can be achieved using CP decomposition.

**Core Idea**: Different human bodies share similar structural features; therefore, the multi-identity Gaussian parameter tensor exhibits a low-rank structure that can be efficiently represented using CP decomposition.

## Method

### Overall Architecture

The pipeline of MIGS consists of three steps: (1) defining 3D Gaussians in the canonical space for each identity $i$, with parameters including position $\boldsymbol{\mu}$, scale $\boldsymbol{s}$, rotation quaternion $\boldsymbol{q}$, feature vector $\boldsymbol{f}$, and opacity $\alpha$; (2) stacking the Gaussian parameters of all identities into a third-order tensor $\boldsymbol{\mathcal{W}} \in \mathbb{R}^{N_i \times N_g \times M}$; (3) performing CP decomposition on this tensor to learn only the decomposed factor matrices. During animation, a non-rigid deformation network $f_d$ and an LBS-based rigid transformation are used to transform the canonical Gaussians into the observation space.

### Key Designs

1. **High-Order Tensor Construction**: For $N_i$ identities, each having $N_g$ Gaussians, and each Gaussian containing $M=43$ dimensional parameters (3 for position, 3 for scale, 4 for rotation, 32 for features, 1 for opacity), the tensor is constructed as:

$$\boldsymbol{\mathcal{W}} \in \mathbb{R}^{N_i \times N_g \times M}, \quad \boldsymbol{w}_{i,g,:} = [\boldsymbol{\mu}^{(i,g)}; \boldsymbol{s}^{(i,g)}; \boldsymbol{q}^{(i,g)}; \boldsymbol{f}^{(i,g)}; \alpha^{(i,g)}]$$

This naturally decouples the three dimensions: "identity", "Gaussian index", and "parameter type".

2. **CP Tensor Decomposition**: CANDECOMP/PARAFAC (CP) decomposition is applied to the tensor $\boldsymbol{\mathcal{W}}$. First, the tensor is unfolded along the second dimension to obtain $\boldsymbol{W}_{(2)} \in \mathbb{R}^{N_g \times (N_i M)}$, which is then approximated as:

$$\boldsymbol{W}_{(2)} \approx \boldsymbol{U}_3 (\boldsymbol{U}_2 \odot \boldsymbol{U}_1)^T$$

where $\boldsymbol{U}_1 \in \mathbb{R}^{M \times R}$, $\boldsymbol{U}_2 \in \mathbb{R}^{N_i \times R}$, $\boldsymbol{U}_3 \in \mathbb{R}^{N_g \times R}$, and $\odot$ represents the Khatri-Rao product. In practice, only $(M + N_i + N_g)R$ parameters need to be learned instead of $M \cdot N_i \cdot N_g$. When $R=100, N_g=5 \times 10^4, N_i=30$, the parameter count is reduced from $6.5 \times 10^7$ to $5 \times 10^6$, **decreasing by an order of magnitude**.

3. **Non-Rigid and Rigid Deformation**: The non-rigid deformation network $f_d$ outputs offsets for position, scale, and rotation: $(\delta\boldsymbol{\mu}, \delta\boldsymbol{s}, \delta\boldsymbol{q}, \boldsymbol{z}) = f_d(\boldsymbol{\mu}_c; \boldsymbol{z}_p)$. The rigid transformation is based on SMPL's LBS: $\boldsymbol{T} = \sum_{b=1}^{B} f_r(\boldsymbol{\mu}_d)_b \boldsymbol{B}_b$. Color is predicted from the feature vectors and spherical harmonics using an MLP $f_c$.

4. **Initialization Strategy**: Points are initialized by sampling $N_g$ points from the SMPL mesh of the first identity. The CP decomposition of their parameter matrix is computed using the CPPower algorithm from TensorLy, yielding $\boldsymbol{U}_1$, $\boldsymbol{U}_3$, and the first row of $\boldsymbol{U}_2$. The first row of $\boldsymbol{U}_2$ is then copied to all other rows.

5. **Personalization and New Identities**: (a) Personalization: Other parameters are frozen while only the color MLP $f_c$ is fine-tuned to recover high-frequency details. (b) New Identities: A new row is added to $\boldsymbol{U}_2$, and only this new row and $f_c$ are optimized, preventing disruption to the learned multi-identity deformation knowledge.

### Loss & Training

The loss functions from 3DGS-Avatar are adopted: RGB photometric loss + mask loss + skinning weight regularization + as-isometric-as-possible regularization. During training, frames from different identities are sampled alternately for rendering optimization. Notably, **no per-frame latent codes are used** to prevent overfitting to the training frames.

## Key Experimental Results

### Main Results

**ZJU-MoCap Novel View Synthesis (Trained on 6 identities)**:

| Method | 377 PSNR↑ | 386 PSNR↑ | 392 PSNR↑ | 394 PSNR↑ |
|------|-----------|-----------|-----------|-----------|
| HumanNeRF | 30.41 | 33.20 | 31.04 | 30.31 |
| 3DGS-Avatar | 30.64 | 33.63 | 31.66 | 30.54 |
| **MIGS (Ours)** | **32.85** | **34.98** | **33.88** | **32.28** |

**AIST++ Dance Dataset (Trained on 30 identities)**:

| Method | Basic PSNR↑ | Basic LPIPS*↓ | Advanced PSNR↑ | Advanced LPIPS*↓ |
|------|-------------|---------------|----------------|------------------|
| HumanNeRF | 24.58 | 29.20 | 22.01 | 39.01 |
| 3DGS-Avatar | 28.89 | 18.20 | 25.51 | 28.86 |
| **MIGS** | **29.82** | **17.73** | **26.54** | **26.02** |

### Ablation Study

**Impact of CP Decomposition Rank R (AIST++ Advanced Test, LPIPS*↓)**:

| Number of Identities | R=10 | R=100 | R=200 |
|--------|------|-------|-------|
| 10 | ~28 | ~26 | ~26 |
| 20 | ~32 | ~27 | ~27 |
| 30 | ~38 | ~28 | ~27 |

$R=10$ is insufficient to capture the diversity of multiple identities, $R=100$ is already sufficient, and $R=200$ yields no significant improvement. After personalization fine-tuning, the results of $R=100$ and $R=200$ are almost identical.

### Key Findings

- Increasing the number of training identities $\rightarrow$ improves OOD pose robustness (decreases LPIPS), but results become smoother $\rightarrow$ personalization fine-tuning can recover details.
- On the highly challenging dance poses in AIST++, MIGS significantly outperforms all per-identity methods, especially under extreme poses like crossed limbs.
- New identity learning requires only a 10-second short video + optimizing the new row of $\boldsymbol{U}_2$.

## Highlights & Insights

- **Physical Intuition of Low-Rank Assumption**: Different human bodies share skeletal structures and motion patterns, which provides strong prior support for the low-rank structure of the parameter tensor.
- **Extremely High Parameter Efficiency**: 30 identities require only $1/13$ of the parameters required by a single-identity model.
- **Scalable Design**: Introducing a new identity only requires adding a single row, avoiding the need to retrain the entire model.
- This work elegantly ports the classic tensor decomposition approach (the TensorFaces concept) into the 3DGS era.

## Limitations & Future Work

- Results tend to become smoother when the number of identities is large, relying heavily on personalization fine-tuning.
- All identities share the same number of Gaussians $N_g$, which cannot adapt to variations in different body sizes.
- The current scale has only been validated up to 30 identities; scalability to thousands of identities remains unverified.
- The non-rigid deformation network is still a shared MLP, which may limit the expression of extreme deformations.

## Related Work & Insights

- **TensorFaces** (2002): A pioneering work using multilinear tensor decomposition to represent facial variation patterns, serves as a direct inspiration for MIGS.
- **3DGS-Avatar**: The single-identity baseline for MIGS, which removes the per-frame latent code to enhance generalization.
- **SNARF**: A differentiable forward skinning method, used in the rigid transformation module of MIGS.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce CP tensor decomposition to multi-identity 3DGS modeling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on two datasets with detailed ablation and new identity generalization experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivations and intuitive explanations.
- **Value**: ⭐⭐⭐⭐ — Highly efficient multi-identity representation holds practical potential for applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification](multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](../../CVPR2025/human_understanding/gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)

</div>

<!-- RELATED:END -->
