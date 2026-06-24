---
title: >-
  [Paper Note] Learning to Generate Conditional Tri-Plane for 3D-Aware Expression Controllable Portrait Animation
description: >-
  [ECCV 2024][3D Vision][Portrait Animation] This paper proposes Export3D, which learns appearance-decoupled expression representations (CLeBS) via contrastive pre-training and directly generates conditional tri-planes integrated with Expression-Adaptive Layer Normalization (EAdaLN), achieving cross-identity 3D-aware portrait expression animation without identity leakage.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Portrait Animation"
  - "Tri-plane"
  - "Expression Control"
  - "Contrastive Learning"
  - "3DMM"
date: 2026-05-08
content_hash: 40d82356cf0c8af4
---

# Learning to Generate Conditional Tri-Plane for 3D-Aware Expression Controllable Portrait Animation

**Conference**: ECCV 2024  
**arXiv**: [2404.00636](https://arxiv.org/abs/2404.00636)  
**Code**: [Project Page](https://export3d.github.io)  
**Area**: 3D Vision  
**Keywords**: Portrait Animation, Tri-plane, Expression Control, Contrastive Learning, 3DMM

## TL;DR

This paper proposes Export3D, which learns appearance-decoupled expression representations (CLeBS) via contrastive pre-training and directly generates conditional tri-planes integrated with Expression-Adaptive Layer Normalization (EAdaLN), achieving cross-identity 3D-aware portrait expression animation without identity leakage.

## Background & Motivation

- **Core demand of portrait animation**: Virtual human services (cross-lingual dubbing, avatars, video conferencing, etc.) require transferring driving expressions while maintaining the source identity, but cross-identity expression transfer is highly challenging.
- **Limitations of image warping methods**: Mainstream 2D methods (e.g., FOMM, TPSMM) rely on image warping to estimate motion, but facial expressions as local motions are easily overwhelmed by global head movements, and expressions and appearances are highly entangled.
- **Defects of 3DMM expression parameters**: 3DMM expression parameters $\beta \in \mathbb{R}^{64}$ contain implicit appearance information. Using them directly leads to identity leakage (leakage of eye shape, face outline, etc.) during cross-identity transfer.
- **Issues with 3D deformation fields**: Methods like HiDe-NeRF and NOFA control expressions by predicting point-wise deformation fields, which can cause video-level artifacts (flickering, lighting instability).
- **Instability of decoupled learning**: DPE decouples pose and expression via cycle consistency, but the training is unstable, leading to temporal inconsistency.
- **Identity loss in GAN inversion**: EG3D-based methods control expressions through latent space inversion, but style latent codes lack the capacity to encode spatial information and individual details.

## Method

### Overall Architecture

Export3D consists of two stages: (1) Contrastive pre-training stage: a CLeBS expression encoder is trained on a video dataset to obtain expression representations decoupled from appearance; (2) Main model training stage: a hybrid tri-plane generator takes the source image and driving expression parameters as inputs, injects expression information through EAdaLN, and generates the expression-transferred tri-plane. This tri-plane is then rendered via volume rendering and upsampled via super-resolution to output the final image.

### Key Designs

#### Module 1: Contrastive Learning-based Basis Scaling (CLeBS)

Positive and negative sample pairs are sampled from the same video, and contrastive learning is utilized to make the expression encoder $f_e(\cdot)$ learn appearance-independent expression representations. The contrastive loss is as follows:

$$\mathcal{L}_{cl} = -\log\left(\frac{\exp(\cos(f_I(X_k), f_e(\beta_k))/\tau)}{\sum_{j \neq k} \exp(\cos(f_I(X_j), f_e(\beta_k))/\tau)}\right)$$

where $f_I(\cdot)$ is the image encoder and $\tau$ is the temperature parameter. Since all samples originate from the same video (sharing the same appearance), this objective function forces the encoder to discard appearance information.

The key innovation lies in the design of an expression subspace with an orthogonal basis structure. Through QR decomposition, an orthogonal basis $V = \{v_1, v_2, \ldots, v_n\} \subseteq \mathbb{R}^d$ is obtained. The expression parameters are then converted into low-dimensional coefficients to scale the orthogonal basis:

$$\beta' = f_e(\beta) = \lambda_1 v_1 + \lambda_2 v_2 + \cdots + \lambda_n v_n \in \mathbb{R}^d$$

where $\lambda = (\lambda_1, \ldots, \lambda_n) \in \mathbb{R}^n$ ($n \ll 64$), and $\langle v_i, v_j \rangle = \delta_{ij}$, allowing different expressions to be scaled and controlled independently along orthogonal directions.

#### Module 2: Expression-Adaptive Layer Normalization (EAdaLN)

Expression information is injected before the multi-head self-attention and Mix-FFN blocks of the ViT, modulating the visual tokens via scaling and shifting:

$$\text{EAdaLN}(x, \beta'_D) = \sigma(\beta'_D) \times \text{LN}(x) + \mu(\beta'_D) \in \mathbb{R}^{h \times \frac{HW}{2^8}}$$

where $\sigma(\beta'_D)$ and $\mu(\beta'_D)$ are the scaling and shifting factors computed from the refined expression parameters. Compared to cross-attention, EAdaLN treats expression as a global modulation signal rather than a token-wise query, which is more suitable for global semantic attributes like facial expressions.

#### Module 3: Hybrid Tri-plane Generator

The generator is constructed by combining ViT and convolutional layers. The source image is encoded into visual features through convolutional blocks, converted into tokens via patch merging, and processed by EAdaLN-ViT blocks. Finally, pixel shuffle is used for upsampling, and a Gaussian low-pass filter is applied to eliminate grid artifacts to produce the final expression-transferred tri-plane:

$$\text{T}_{\beta_D}(S) = \mathbf{G}(S, \beta_D) \in \mathbb{R}^{3 \times 32 \times \frac{H}{2} \times \frac{W}{2}}$$

The tri-plane is projected and aggregated to obtain features: $F_{\beta_D}(S) = \frac{1}{3}(F_{\beta_D,xy} + F_{\beta_D,yz} + F_{\beta_D,zx})$, which are decoded by an MLP into color and density for volume rendering.

### Loss & Training

- CLeBS is frozen after pre-training and does not participate in subsequent training; the image encoder $f_I$ is discarded after pre-training.
- Online EMA is employed to stabilize the tri-planes, adding $T_{EMA}$ to the generated tri-planes.
- A low-resolution image $\hat{D}_{raw} \in \mathbb{R}^{3 \times H/4 \times W/4}$ is rendered first, and then passed through a super-resolution module to obtain the final resolution.
- Plain convolutional blocks, instead of style-modulated convolutions, are used for super-resolution (as style latent codes are not employed).
- A Gaussian low-pass filter is applied after pixel shuffle to eliminate grid artifacts caused by token patching.

## Key Experimental Results

### Main Results

Quantitative comparison on the VFHQ dataset:

| Method | PSNR↑ | SSIM↑ | AKD↓ | CSIM↑(Same ID) | AED↓(Same ID) | CSIM↑(Cross ID) | AED↓(Cross ID) |
|------|-------|-------|------|------------|----------|------------|----------|
| StyleHEAT | 14.23 | 0.428 | 30.41 | 0.464 | 0.161 | 0.505 | 0.242 |
| DPE | 23.24 | 0.750 | 3.66 | 0.831 | 0.083 | 0.586 | 0.253 |
| HiDe-NeRF† | 21.23 | 0.728 | 8.25 | 0.867 | 0.106 | 0.707 | 0.255 |
| **Ours** | **23.56** | 0.704 | **3.45** | **0.811** | **0.082** | **0.694** | **0.208** |

(† indicates evaluation restricted to the foreground facial region)

### Ablation Study

Ablation study on expression encoding methods:

| Method | PSNR↑ | CSIM↑(Same ID) | AED↓(Same ID) | CSIM↑(Cross ID) | AED↓(Cross ID) |
|------|-------|------------|----------|------------|----------|
| Direct 3DMM | 23.08 | 0.789 | 0.105 | 0.648 | 0.209 |
| E2E LeBS (n=25) | 23.11 | 0.745 | 0.109 | 0.670 | 0.218 |
| E2E LeBS (n=10) | 23.24 | 0.751 | 0.110 | 0.672 | 0.238 |
| E2E LeBS (n=5) | 22.63 | 0.658 | 0.140 | 0.632 | 0.246 |
| **CLeBS (Full)** | **23.56** | **0.811** | **0.082** | **0.694** | **0.208** |

EAdaLN vs Cross-Attention: EAdaLN consistently outperforms cross-attention on CSIM (0.811 vs 0.678) and AED (0.082 vs 0.125).

### Key Findings

- t-SNE visualization clearly demonstrates that raw 3DMM expression parameters cluster by identity (entangled with appearance), whereas this clustering disappears after processing with CLeBS.
- LeBS alone (without contrastive pre-training) fails to decouple appearance and expression; reducing the number of basis vectors only degrades both simultaneously.
- Contrastive pre-training is key—the same-video sampling strategy enables effective discrimination of expressions sharing the same appearance.
- The orthogonal basis structure allows independent and linear control over different expression directions (e.g., eye blinking, lip movement).
- EAdaLN outperforms cross-attention since expressions act as global modulation signals, making position-level fine-grained attention unnecessary.

## Highlights & Insights

- **A New Paradigm for Appearance-Expression Decoupling**: Instead of relying on cycle consistency or explicit annotations, decoupling is naturally achieved through same-video contrastive learning, resulting in stable training and remarkable effectiveness.
- **Orthogonal Basis Design**: The concept of orthogonal structures from 3DMM is introduced into the learned expression space. The orthogonality of the basis is guaranteed via QR decomposition, enabling interpretable control of expression directions.
- **Design Philosophy of EAdaLN**: Expression is injected as a global semantic signal through normalization layers, which fits the demand for "modulation" rather than "selection" much better than cross-attention.

## Limitations & Future Work

- Dependence on 3DMM for extracting expression parameters, which can be inaccurate in cases of occlusion or extreme angles.
- The super-resolution module relies on convolutional upsampling, which may introduce blurriness; stronger super-resolution schemes like diffusion models have not been explored.
- The scale and diversity of the VFHQ training dataset are limited, potentially leading to suboptimal generalization on extreme expressions or non-frontal faces.
- Audio-driven scenarios have not been explored; CLeBS could be extended to integrate speech-to-expression mapping.

## Related Work & Insights

- **Conditional Injection from DiT**: EAdaLN directly borrows the adaptive normalization design from Diffusion Transformers (DiTs), proving its efficacy within GAN frameworks as well.
- **Orthogonal Basis Idea from LIA**: The orthogonal motion dictionary of LIA is extended to the expression space and further refined via contrastive learning.
- **Insights for the Virtual/Digital Human Field**: Appearance-expression decoupling is a core requirement for real-time virtual communication. CLeBS provides a lightweight, yet effective pre-training solution.

## Rating

⭐⭐⭐⭐ — This work presents an elegant solution to the appearance-expression entanglement problem. The combined design of contrastive pre-training, orthogonal basis, and EAdaLN is novel, showing excellent results in cross-identity expression transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis](../../NeurIPS2025/3d_vision/hyplanehead_rethinking_tri-plane-like_representations_in_full-head_image_synthes.md)
- [\[ECCV 2024\] Learning 3D-Aware GANs from Unposed Images with Template Feature Field](learning_3d-aware_gans_from_unposed_images_with_template_feature_field.md)
- [\[ECCV 2024\] SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](sceneverse_scaling_3d_vision-language_learning_for_grounded_scene_understanding.md)
- [\[CVPR 2026\] EG-3DVG: Expression and Geometry Aware Grounding Decoder for 3D Visual Grounding](../../CVPR2026/3d_vision/eg-3dvg_expression_and_geometry_aware_grounding_decoder_for_3d_visual_grounding.md)
- [\[ECCV 2024\] Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting](repaint123_fast_and_high-quality_one_image_to_3d_generation_with_progressive_con.md)

</div>

<!-- RELATED:END -->
