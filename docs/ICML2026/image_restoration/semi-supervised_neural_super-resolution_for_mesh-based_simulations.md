---
title: >-
  [Paper Note] Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations
description: >-
  [ICML 2026][Image Restoration][mesh super-resolution] SuperMeshNet utilizes two complementary MPNNs—a primary model predicting LR→HR and an auxiliary model predicting HR-HR differences corresponding to LR-LR pairs—to gen…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "mesh super-resolution"
  - "semi-supervised regression"
  - "complementary learning"
  - "message passing inductive bias"
  - "PDE simulation acceleration"
date: 2026-05-08
content_hash: 9428a4b8d6cd9741
---

# Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations

**Conference**: ICML 2026  
**arXiv**: [2605.09284](https://arxiv.org/abs/2605.09284)  
**Code**: <https://github.com/jykim-git/SuperMeshNet.git>  
**Area**: 3D Vision / Physics Simulation / Graph Neural Networks  
**Keywords**: mesh super-resolution, semi-supervised regression, complementary learning, message passing inductive bias, PDE simulation acceleration

## TL;DR
SuperMeshNet utilizes two complementary MPNNs—a primary model predicting LR→HR and an auxiliary model predicting HR-HR differences corresponding to LR-LR pairs—to generate mutual pseudo-labels on samples without paired HR. Combined with two lightweight inductive biases, node-level and message-level centering, it enables PDE mesh super-resolution to outperform the 100% HR fully supervised baseline using only 10% HR data, consistently reducing RMSE across 6 MPNN architectures.

## Background & Motivation

**Background**: Mesh-based PDE simulations such as FEM and FVM are governed by mesh size in terms of solution accuracy and computational cost; fine meshes are accurate but expensive. Neural network super-resolution aims to use cheap LR simulations to predict HR solutions. Existing works roughly fall into two categories: CNN-based (requiring inefficient interpolation of irregular meshes onto regular grids) and MPNN-based (directly processing graphs but requiring large amounts of paired HR supervision).

**Limitations of Prior Work**: The acquisition of HR data itself is the bottleneck that super-resolution seeks to avoid—fine-mesh simulation is precisely what is expensive—making "full supervision" inherently contradictory. Existing unsupervised solutions like PhySRNet embed PDE residuals into the loss but are limited to finite difference on regular grids; MAgNet performs zero-shot interpolation but suffers from prediction errors significantly higher than supervised versions.

**Key Challenge**: HR data scarcity vs. hungry MPNN training. Conventional semi-supervised regression methods (Mean Teacher, UCVME, TNNR) almost all default to two models predicting the "same target," leading to highly correlated pseudo-labels that reinforce errors, which fails in MPNN super-resolution scenarios.

**Goal**: (1) Introduce semi-supervised learning to mesh-based super-resolution for the first time, compatible with any MPNN; (2) Design a mechanism where "two models predict different but related targets" to decorrelate pseudo-label errors through complementarity; (3) Systematically summarize MPNN inductive biases beneficial for super-resolution.

**Key Insight**: From a physical perspective, two HR solutions governed by the same PDE but differing only in parameter $\mu$ characterize the system's response to parametric perturbations. If a model specifically learns this difference, the pseudo-labels it provides are orthogonal in dimension to "direct HR prediction," thereby breaking pseudo-label collapse.

**Core Idea**: Use the primary model $F_\theta$ to learn the inter-resolution map $u_l \to u_h$, and the auxiliary model $G_\phi$ to learn the intra-resolution difference $(u_l^r, u_l^s) \to (u_h^r - u_h^s)$. The two serve as mutual pseudo-label sources, providing complementary supervision on unpaired LR data.

## Method

### Overall Architecture
The dataset is divided into two parts: a small paired LR–HR set $\mathcal{D}_a=\{(u_l^q, u_h^q)\}_{q=1}^{N_h}$ ($N_h \ll N$) and an unpaired LR set $\mathcal{D}_b=\{u_l^q\}_{q=N_h+1}^{N}$. Each batch selects two paired LR samples $\alpha, \beta$ and one unpaired LR sample $\gamma$. The primary model $F_\theta(u_l^q)=\hat{u}_h^q$ is used for inference; the auxiliary model $G_\phi(u_l^r, u_l^s)=\hat{u}_h^{rs}$ is used only during training to predict HR differences. The supervised part is trained using the ground truth HR labels of $\alpha, \beta$; the unsupervised part treats the sum or difference of one model's prediction and a known HR as a pseudo-label to train the other model. Both models share an LR encoder to save cost. The primary model is based on SRGNN, using a dual-path fusion of a kNN-upsampler and a latent-space upsampler.

### Key Designs

1.  **Complementary Learning**:
    - **Function**: Allows the two models to predict targets that are "different but physically related," completely breaking pseudo-label collapse.
    - **Mechanism**: The supervised losses are $\mathcal{L}_{F,sup} = \ell(\hat{u}_h^\alpha, u_h^\alpha) + \ell(\hat{u}_h^\beta, u_h^\beta)$ and $\mathcal{L}_{G,sup} = \ell(\hat{u}_h^{\alpha\beta}, u_h^\alpha - \text{kNN}(u_h^\beta;P_h^\beta\to P_h^\alpha))$. For the unsupervised part: $\mathcal{L}_{F,unsup}$ uses $\hat{u}_h^{\gamma\alpha} + u_h^\alpha$ as a pseudo-label to supervise $F_\theta(u_l^\gamma)$; $\mathcal{L}_{G,unsup}$ uses $\hat{u}_h^\gamma - u_h^\alpha$ as a pseudo-label to supervise $G_\phi(u_l^\gamma, u_l^\alpha)$.
    - **Design Motivation**: Mean Teacher / UCVME use two isomorphic networks to predict the same target, where errors are reinforced after pseudo-labels converge to the same mode (confirmation bias). In this work, the prediction spaces of the two models differ (HR solution vs. HR difference), naturally decorrelating errors and providing physical priors regarding parametric sensitivity.

2.  **kNN Interpolation for Mesh Mismatch**:
    - **Function**: Ensures the "HR difference" remains defined even across two different HR meshes.
    - **Mechanism**: Different $\mu$ correspond to different geometries; hence node positions $P_h^r \ne P_h^s$ for $u_h^r$ and $u_h^s$ prevent direct subtraction. kNN distance weighting is used to project one onto the node positions of the other, $\text{kNN}(u_h^s; P_h^s \to P_h^r)$, before subtraction. All difference terms in unsupervised losses must follow the corresponding kNN projection direction.
    - **Design Motivation**: The primary difference between mesh-based simulation and CNNs or regular grids is the irregular structure; kNN interpolation is a PointNet-style lightweight, differentiable solution that avoids learning additional alignment networks.

3.  **MPNN Inductive Bias: Node-level / Message-level Centering**:
    - **Function**: An MPNN-agnostic training trick that consistently improves super-resolution performance across architectures.
    - **Mechanism**: After updating node embeddings, each MPNN layer performs $x_i \leftarrow x_i - \frac{1}{n}\sum_i x_i$. For architectures that explicitly aggregate messages (e.g., MGN), it further performs $agg_i \leftarrow agg_i - \frac{1}{n}\sum_i agg_i$, effectively removing the global mean from intermediate representations.
    - **Design Motivation**: The authors argue that super-resolution relies primarily on local relative structures rather than absolute means. Centering can smooth the loss landscape (analogous to BatchNorm) but is only beneficial in tasks that do not depend on the global mean. Ablations show consistent RMSE reductions across GCN/SAGE/GAT/GTR/GIN/MGN (e.g., MGN 0.0269→0.0226).

### Loss & Training
The total losses are $\mathcal{L}_F = \mathcal{L}_{F,sup} + \mathcal{L}_{F,unsup}$ and $\mathcal{L}_G = \mathcal{L}_{G,sup} + \mathcal{L}_{G,unsup}$, with both weights set to 1 without scheduling. When outputting multiple physical quantities (velocity + pressure), a weighted MSE is used: 99:1 for time-dependent PDE Dataset 1, and $10^{-8}:1$ for real geometry datasets to handle magnitude differences. Optimization uses Adam ($\text{lr}=10^{-3}$) and PyTorch AMP on hardware comprising an i9-10920X and RTX A6000.

## Key Experimental Results

### Main Results

Dataset 1 (Linear elasticity von Mises stress, FEM), RMSE↓ across 6 MPNNs:

| Method | $N_h$, $N$ | GCN | SAGE | GAT | GTR | GIN | MGN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fully Supervised (no bias) | 20, 20 | 0.0874 | 0.0876 | 0.0826 | 0.0758 | 0.0819 | 0.0655 |
| Fully Supervised (no bias) | 200, 200 | 0.0575 | 0.0544 | 0.0512 | 0.0450 | 0.0381 | 0.0228 |
| SuperMeshNet-O (no bias) | 20, 200 | 0.0613 | 0.0589 | 0.0544 | 0.0451 | 0.0404 | 0.0269 |
| **SuperMeshNet (with bias)** | 20, 200 | **0.0431** | **0.0450** | **0.0457** | **0.0385** | **0.0277** | **0.0226** |

Real Geometry (Motorbike + rider incompressible Navier-Stokes) Drag / Lift coefficients (relative error):

| Method | $N_h$,$N$ | Drag (rel. err) | Lift (rel. err) |
| :--- | :--- | :--- | :--- |
| Ground truth HR | — | 0.3724 | 0.0368 |
| SuperMeshNet | 40, 200 | 0.3778 (0.014) | 0.0433 (0.177) |
| Fully Supervised | 200, 200 | 0.3653 (0.019) | 0.0380 (0.033) |

### Ablation Study

Dataset 1, MGN, $N_h=20, N=200$, Inductive Bias Ablation:

| Configuration | RMSE | Description |
| :--- | :--- | :--- |
| No bias (O) | 0.0269 | Complementary learning only |
| + Node centering (N) | 0.0237 | N alone captures the majority of the gain |
| + Message centering (M) | 0.0247 | M alone is slightly weaker than N |
| N + M | **0.0226** | Best performance when combined |

Semi-supervised Regression Baselines (Dataset 1, $N_h=20, N=200$, MGN):

| Method | RMSE | Training Time (s) |
| :--- | :--- | :--- |
| Mean-Teacher | 0.0325 | 693.84 |
| TNNR | 0.0624 | 477.48 |
| UCVME | 0.0293 | 1122.62 |
| SuperMeshNet-O | 0.0269 | 503.2 |
| **SuperMeshNet** | **0.0226** | **421** |

### Key Findings
- Achieving better performance than the 100% HR fully supervised baseline using only 10% HR (20 vs 200)—a 90% HR saving—is the core practical conclusion. Especially on fine meshes where data generation costs scale exponentially with resolution, the total training cost is inverted into a net decrease.
- Complementary learning achieves the lowest RMSE with the shortest training time (421 s vs. UCVME 1122 s) because other semi-supervised methods use isomorphic dual networks requiring redundant computations for the same target; this work reuses a shared encoder.
- On time-dependent PDE Dataset 2, where the difference between HR and LR vorticity is massive (128x node ratio), full supervision fails while SuperMeshNet still recovers HR. This proves the HR-HR relationship provided by $G_\phi$ offers a stronger learning signal than the pure LR→HR mapping.

## Highlights & Insights
- Coupling two models through a common HR quantity while they predict different physical variables is an elegant paradigm that combines co-training with PDE physical symmetry. This can be extended to any problem with a "parametric solution family" structure (climate, biomechanics, lattice simulation).
- The universal inductive bias of node/message centering requires only one line of code yet uniformly improves 6 MPNN architectures, indicating that removing the mean for "relative structure" tasks is a very cheap and robust trick.
- The experimental design emphasizes that the "primary value lies in the 90% saving of HR data, rather than absolute RMSE reduction," maintaining a pragmatic stance toward the actual bottleneck in mesh-based super-resolution.

## Limitations & Future Work
- The training time for complementary learning is longer than full supervision (though shorter than similar semi-supervised methods). The paper admits a net benefit is only achieved on sufficiently fine meshes where HR data generation costs explode; it may not be cost-effective for medium-to-small scale meshes.
- There is a lack of theoretical guarantees for training stability, with only empirical studies provided in the appendix. Theoretically, error amplification could still occur if the auxiliary model $G_\phi$ has high intrinsic error; failure is expected on PDEs with strong non-linearity or bifurcations.
- The selection of HR samples is also crucial (empirical exploration in Appendix I.12), but currently uses random sampling. An active learning style HR sampling strategy would likely further reduce $N_h$.

## Related Work & Insights
- **vs. PhySRNet (Arora, 2022)**: Completely unsupervised but requires finite difference, limiting it to regular grids. This work requires small amounts of HR but handles irregular meshes.
- **vs. MAgNet (Boussif et al., 2022)**: Zero-shot MPNN interpolation with prediction errors much higher than the supervised version. This work significantly reduces error using minimal HR.
- **vs. UCVME / Mean Teacher / TNNR**: These semi-supervised regression methods use "same-target dual networks" prone to pseudo-label collapse. This work uses "different-target dual networks" to fundamentally decorrelate.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combining "different-target dual models + physical differences" for mesh super-resolution is a genuine first, well-adapted for MPNNs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Very detailed coverage across 6 MPNNs × 3 FEM datasets + 3 CFD datasets + semi-supervised baselines + inductive bias ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous physical and mathematical notation, clear pipeline diagrams, and a very rich appendix.
- **Value**: ⭐⭐⭐⭐ The 90% HR saving directly addresses real pain points in industrial CAE and climate simulation, with open-source code ready for use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)
- [\[ICML 2026\] Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution](coloring_the_noise_adversarial_sobolev_alignment_for_faithful_image_super_resolu.md)
- [\[NeurIPS 2025\] Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks](../../NeurIPS2025/image_restoration/spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)
- [\[ICML 2026\] PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)
- [\[ICML 2026\] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)

</div>

<!-- RELATED:END -->
