---
title: >-
  [Paper Note] TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning
description: >-
  [AAAI 2026][Medical Imaging][Tooth alignment] This paper proposes TAlignDiff, a unified framework that integrates a geometry-constrained point cloud regression network (PRN) with a diffusion-based transformation matrix denoising module (DTMD) under a joint training paradigm. Through a bidirectional feedback mechanism, the framework achieves superior automatic tooth alignment on small-scale clinical datasets compared to existing methods.
tags:
  - AAAI 2026
  - Medical Imaging
  - Tooth alignment
  - orthodontic treatment
  - diffusion model
  - transformation matrix
  - point cloud regression
date: 2026-05-08
content_hash: 4a90a43decaaa873
---

# TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning

**Conference**: AAAI 2026
**arXiv**: [2508.04565](https://arxiv.org/abs/2508.04565)
**Code**: Unavailable (to be released upon acceptance)
**Area**: Medical Imaging / 3D Vision
**Keywords**: Tooth alignment, orthodontic treatment, diffusion model, transformation matrix, point cloud regression

## TL;DR

This paper proposes TAlignDiff, a unified framework that integrates a geometry-constrained point cloud regression network (PRN) with a diffusion-based transformation matrix denoising module (DTMD) under a joint training paradigm. Through a bidirectional feedback mechanism, the framework achieves superior automatic tooth alignment on small-scale clinical datasets compared to existing methods.

## Background & Motivation

**Background**: Automatic tooth alignment is a core task in orthodontic treatment, requiring prediction of the optimal motion (rotation + translation) for each tooth from a malocclusion state to a normal occlusion. Current deep learning approaches primarily encode 3D dental models as point clouds and employ regression networks to predict 6-DoF transformation matrices ($4 \times 4$ rotation + translation matrices), supervised by point-wise geometric losses such as reconstruction error. Representative methods include TANet (graph convolution-based feature propagation) and TAPoseNet (multi-scale GCN).

**Limitations of Prior Work**: Existing methods rely solely on point-to-point geometric constraints to supervise transformation matrix prediction. However, transformation matrices possess intrinsic distributional characteristics—reasonable ranges for valid rotations/translations, inter-tooth correlations, etc.—which deterministic geometric constraints cannot capture, causing predicted matrices to fall outside plausible ranges.

**Key Challenge**: Although TADPM was the first to introduce diffusion models for learning the transformation matrix distribution, it conditions the diffusion process directly on high-dimensional point cloud and mesh features, resulting in high computational complexity and heavy dependence on large datasets, with poor performance in small-sample clinical settings.

**Goal**: How can both explicit geometric constraints and implicit distributional modeling be leveraged simultaneously on small-scale clinical data? Specific sub-problems: (1) How to design a lightweight diffusion module for learning the transformation matrix distribution? (2) How to establish bidirectional feedback between geometric regression and diffusion-based distribution modeling?

**Key Insight**: Rather than applying diffusion to high-dimensional geometric features, diffusion is applied exclusively to the transformation matrices themselves—reducing the diffusion input dimensionality from high-dimensional point cloud features to a $32 \times 16$ transformation matrix space, substantially lowering the learning difficulty and data requirements of the diffusion model.

**Core Idea**: A lightweight diffusion model performs noise estimation in the transformation matrix space, implicitly constraining the regression network outputs to conform to the true distribution by contrasting noise estimates between predicted and ground-truth matrices.

## Method

### Overall Architecture

TAlignDiff comprises two modules: (1) **PRN** (Point cloud Regression Network): takes malocclusion tooth point clouds as input, extracts global and local features, and regresses transformation matrices for 32 teeth; (2) **DTMD** (Diffusion Transformation Matrix Denoising module): applies forward noising and reverse denoising to transformation matrices, learning their latent distribution in clinical data. The two modules are coupled via a joint training strategy and a contrastive denoising loss. During inference, only PRN is used; DTMD is excluded, preserving inference efficiency.

### Key Designs

1. **Point Cloud Regression Network (PRN)**:

    - **Function**: Predicts transformation matrices for 32 teeth from 3D dental point clouds, where each matrix comprises a $3 \times 3$ rotation matrix and a $3 \times 1$ translation vector.
    - **Mechanism**: Two PointNet encoders $\epsilon_g$ and $\epsilon_l$ extract global features (entire dental arch) and local features (individual teeth), respectively. Their concatenated outputs are decoded by a three-layer MLP (channels $[512, 256, 16]$) to regress the transformation matrices: $T^* = \phi(\epsilon_g(P_{in}) \oplus \epsilon_l(P_{in}))$. Each PointNet encoder consists of three 1D convolutional layers (channels $[64, 128, 1024]$).
    - **Design Motivation**: PointNet processes raw point clouds directly and extracts hierarchical features via symmetric functions. The dual global–local encoder design enables the model to perceive both the overall dental arch arrangement and the geometric details of individual teeth.

2. **Diffusion Transformation Matrix Denoising Module (DTMD)**:

    - **Function**: Serves as a training auxiliary module that learns the latent distribution of transformation matrices in clinical data, indirectly constraining PRN outputs through noise estimation.
    - **Mechanism**: The target transformation matrix $M_0$ (reshaped $M_{gt}$) is progressively noised via a forward diffusion chain: $q(M_t | M_0) = \mathcal{N}(M_t | \sqrt{\gamma_t} M_0, (1-\gamma_t)I)$. The diffusion model $\epsilon_{\theta_d}$ learns to predict the added noise. The key innovation is the contrastive denoising loss: $L_{denoi} = \mathbb{E}[\|\epsilon_{\theta_d}(M_{gt}^t, t) - \epsilon_{\theta_d}(M_{pre}^t, t)\|_1]$, which compares noise estimates of the ground-truth and predicted matrices after injecting identical noise at the same timestep—the closer the predicted matrix is to the true distribution, the smaller the discrepancy between their noise estimates.
    - **Design Motivation**: Using transformation matrices (rather than high-dimensional point cloud features) as diffusion inputs reduces the input dimensionality from thousands of dimensions to $32 \times 16$, substantially lowering data requirements. DTMD participates only in training and not in inference, leaving inference efficiency unaffected.

3. **Joint Training Strategy**:

    - **Function**: Achieves cooperative optimization of PRN and DTMD through a staged training procedure.
    - **Mechanism**: During the first 200 epochs, PRN and DTMD are jointly trained with all four losses optimized simultaneously. During the subsequent 200 epochs, DTMD parameters are frozen and only PRN is trained, continuing to leverage the pretrained DTMD (via $L_{denoi}$) to refine PRN outputs. The total loss is $L_{total} = L_{rec} + \lambda_1 L_{center} + \lambda_2 L_{denoi} + \lambda_3 L_{diffusion}$.
    - **Design Motivation**: The first stage allows DTMD to learn the distributional characteristics of transformation matrices; the second stage freezes DTMD so that PRN can focus on exploiting the learned distributional prior. This "learn-then-use" strategy avoids training instability between the two modules.

### Loss & Training

Four loss functions are employed: (1) point cloud reconstruction loss $L_{rec} = \frac{1}{N}\sum\|T^* \cdot P_{in} - T \cdot P_{in}\|_1$; (2) tooth centroid displacement loss $L_{center} = \frac{1}{M}\sum\|C_{predict} - C_{target}\|_1$, constraining the collective displacement direction of the teeth; (3) diffusion training loss $L_{diffusion}$, a standard noise prediction MSE; (4) contrastive denoising loss $L_{denoi}$, measuring the noise estimation discrepancy between PRN outputs and ground-truth matrices. Optimal weights: $\lambda_1=0.1, \lambda_2=0.01, \lambda_3=0.1$. Data augmentation includes multi-tooth random rotation (5–10 teeth) and single-tooth random translation.

## Key Experimental Results

### Main Results

| Method | Dataset | TRE (mm) ↓ | AAE (mm) ↓ |
|--------|---------|-----------|-----------|
| PointNet++ | Test | 0.791±0.927 | 0.717±0.833 |
| PointMLP | Test | 0.819±0.935 | 0.743±0.844 |
| TADPM | Test | 0.890±0.963 | 0.821±0.883 |
| PSTN | Test | 0.779±0.917 | 0.705±0.821 |
| **TAlignDiff (Ours)** | Test | **0.725±0.834** | **0.646±0.734** |

### Ablation Study

| Config ($\lambda_1, \lambda_2, \lambda_3$) | TRE (Test) | AAE (Test) | Note |
|---------------------------------------------|-----------|-----------|------|
| (0, 0, 0) baseline | 0.784±0.927 | 0.711±0.831 | Reconstruction loss only |
| (0.1, 0, 0) | 0.748±0.873 | 0.670±0.778 | + centroid loss |
| (0.1, 0.01, 0.1) best | **0.725±0.834** | **0.646±0.734** | Full model |
| (0.2, 0.01, 0.1) | 0.766±0.864 | 0.692±0.766 | Centroid weight too large |
| (0.1, 0.05, 0.1) | 0.739±0.850 | 0.663±0.755 | Denoising weight too large |

### Key Findings

- Incorporating the DTMD module reduces TRE from 0.784 to 0.725 (−7.5%) and AAE from 0.711 to 0.646 (−9.1%), validating the effectiveness of diffusion-based distribution modeling.
- The joint training strategy outperforms a stepwise training variant in which DTMD and PRN are trained independently, lacking bidirectional feedback.
- TADPM performs worst on this small-sample dataset (TRE 0.890 vs. 0.725), as it conditions the diffusion process on high-dimensional features and requires substantially more data than TAlignDiff.
- 3D scatter plot visualizations show that transformation matrices predicted by TAlignDiff exhibit tighter clustering, indicating improved model stability.
- In challenging cases such as deep overbite malocclusion, TAlignDiff's mesh reconstruction results most closely approximate the target.

## Highlights & Insights

- **Reducing the diffusion input space from high-dimensional geometric features to the transformation matrix space** is the most practically impactful design decision. It confines the diffusion model to learning distributions in a $32 \times 16$-dimensional space rather than thousands of point cloud feature dimensions, enabling training on a dataset of only 124 patients.
- **The contrastive denoising loss** is an elegant design: rather than having the diffusion model directly correct predictions, it transmits distributional information to PRN indirectly by comparing noise estimates of the predicted and ground-truth matrices under identical noise levels. This "contrastive learning in noise space" paradigm is generalizable to other regression tasks.
- **DTMD is excluded at inference time**—the diffusion model serves solely as a regularizer/prior during training, incurring zero additional overhead at inference, making it highly practical.

## Limitations & Future Work

- The dataset is extremely small (124 patients, split 74:20:30), and the model's generalization ability has not been validated at scale.
- Only PointNet is used as the feature extractor; stronger point cloud encoders (e.g., PointNet++, Point Transformer) may yield further improvements.
- The sequential nature of orthodontic treatment is not modeled—actual orthodontics proceeds in stages (e.g., per-step aligners in clear aligner therapy), whereas the current method directly predicts the final state.
- Comparative evaluation against clinical expert judgments is absent; quantitative metrics such as TRE/AAE may not fully reflect clinical acceptability.

## Related Work & Insights

- **vs. TADPM (Lei et al. 2024)**: TADPM conditions the diffusion process on high-dimensional point cloud and mesh features, yielding the worst performance on this small-sample dataset (TRE 0.890 vs. 0.725). TAlignDiff drastically reduces data dependency by projecting into the transformation matrix space.
- **vs. TANet/TAPoseNet**: These methods capture inter-tooth relationships via GCN/multi-scale graph networks but rely solely on geometric constraints. TAlignDiff introduces complementary distributional constraints.
- **vs. PSTN (Li et al. 2020)**: PSTN predicts transformations directly via spatial transformer networks and achieves performance close to TAlignDiff (TRE 0.779 vs. 0.725), but lacks the regularization effect of distributional priors.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of applying diffusion models to transformation matrix distribution learning is novel, and the contrastive denoising loss is a creative contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ The dataset is too small (124 cases); large-scale validation and clinical evaluation are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed method descriptions, though visualizations could be more comprehensive.
- **Value**: ⭐⭐⭐⭐ Directly applicable to orthodontic AI with clear clinical relevance; the small-sample diffusion-assisted regression paradigm is transferable to other medical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](../../ICLR2026/medical_imaging/discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)
- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[AAAI 2026\] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation](divide_conquer_and_unite_hierarchical_style-recalibrated_prototype_alignment_for.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2026/medical_imaging/semitooth_a_generalizable_semisupervised_framework.md)

</div>

<!-- RELATED:END -->
