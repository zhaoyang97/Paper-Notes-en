---
title: >-
  [Paper Note] FG-Portrait: 3D Flow Guided Editable Portrait Animation
description: >-
  [CVPR 2026][Image Generation][Portrait Animation] FG-Portrait introduces "3D optical flow" — directly computed from the FLAME parametric 3D head model without any learning — as a geometry-driven motion correspondence sig…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Portrait Animation"
  - "3D Optical Flow"
  - "Parametric Head Model"
  - "Diffusion Model"
  - "Expression Editing"
date: 2026-05-08
content_hash: bf4f80184809b2de
---

# FG-Portrait: 3D Flow Guided Editable Portrait Animation

**Conference**: CVPR 2026
**arXiv**: [2603.23381](https://arxiv.org/abs/2603.23381)  
**Code**: Unavailable  
**Area**: Diffusion Models / Image Generation
**Keywords**: Portrait Animation, 3D Optical Flow, Parametric Head Model, Diffusion Model, Expression Editing

## TL;DR

FG-Portrait introduces "3D optical flow" — directly computed from the FLAME parametric 3D head model without any learning — as a geometry-driven motion correspondence signal. Combined with depth-guided sampling for 3D flow encoding as the motion condition for a diffusion model ControlNet, the method achieves substantially improved motion transfer accuracy (APD reduced by 22%+) and supports inference-time expression and head pose editing.

## Background & Motivation

1. **Background**: Portrait animation aims to transfer the expression and head pose of a driving portrait to a source image subject. Current mainstream approaches fall into two categories: (a) diffusion model-based methods (X-Portrait, Face-Adapter, HunyuanPortrait), which use landmark or latent representations of the driving image as motion conditions; (b) motion field prediction-based methods (FOMM, EMOPortrait), which learn dense motion correspondences between source and driving frames to warp source features.
2. **Limitations of Prior Work**: Diffusion model-based methods condition generation only on the driving motion without explicit source-driving motion correspondences, resulting in imprecise motion transfer (notable pose/expression errors). Motion field prediction methods require large-scale self-supervised learning of 2D dense motion, yet estimating 3D motion from 2D images is inherently ill-posed and frequently fails under large pose variations or significant appearance discrepancies.
3. **Key Challenge**: How to introduce accurate, robust, learning-free source-driving motion correspondences into a diffusion model framework?
4. **Goal**: (a) Establish accurate 3D spatial motion correspondences; (b) effectively encode 3D motion priors as 2D conditional signals compatible with diffusion models; (c) support user-specified editing at inference time.
5. **Key Insight**: Parametric 3D head models such as FLAME naturally provide per-vertex semantic correspondences — the same vertex index corresponds to the same facial structure location across different poses and expressions. This geometric property enables direct computation of learning-free 3D displacements as motion guidance.
6. **Core Idea**: Replace learned motion field prediction with per-point displacements (3D flow) computed on a FLAME 3D head model, serving as the motion condition for a diffusion model ControlNet, thereby enabling accurate and editable portrait animation.

## Method

### Overall Architecture

FG-Portrait follows a typical three-branch diffusion model portrait animation framework: (1) a Stable Diffusion U-Net as the image generator; (2) an appearance network $A$ (isomorphic to the U-Net) that extracts appearance features from the source image and injects them into the generator's self-attention layers; (3) a ControlNet $G$ responsible for motion control. The key contribution lies in the third branch — replacing conventional landmarks or driving images with the proposed **3D flow encoding** $F_{src \leftarrow tgt}$ as the motion condition. During training, $I_{dri}$ and $I_{src}$ are sampled from different frames of the same video, with the objective of reconstructing $I_{dri}$.

### Key Designs

1. **3D Flow**

    - **Function**: Establishes per-point 3D displacement correspondences between the source and target (driving) portraits without any learning.
    - **Mechanism**: FLAME parameters $(β_{src}, θ_{src}, ψ_{src})$ and $(β_{dri}, θ_{dri}, ψ_{dri})$ are first estimated from the source and driving images respectively. The target head $M_{tgt}$ is then assembled using the source identity shape $β_{src}$, driving pose $θ_{dri}$, and driving expression $ψ_{dri}$. A Surface Field (SF) function is used to find per-point correspondences between $M_{tgt}$ and $M_{src}$: given a target space point $p_{tgt}$, the corresponding source space point is found as $p_{src} = \text{SF}(p_{tgt}; M_{tgt}, M_{src})$, and the 3D flow is defined as $f_{src \leftarrow tgt} = p_{src} - p_{tgt}$. **Backward search** (from target to source) is adopted to ensure every target point finds a valid correspondence.
    - **Design Motivation**: Compared to learned motion fields, 3D flow directly leverages FLAME's topological semantic correspondences, requiring no training data, remaining unaffected by appearance differences, and maintaining accuracy under large pose variations.

2. **3D Flow Encoding with Depth-Guided Sampling**

    - **Function**: Converts the 3D motion prior into a 2D pixel-level motion conditioning signal for the ControlNet.
    - **Mechanism**: For each pixel $(u,v)$ in the target image, $N=20$ 3D points $p_{tgt}^n = H[d_n(K^{-1}q_{tgt})^\top, 1]^\top$ are sampled along the back-projected ray, and the 3D flow is queried at each point, yielding $F_{src \leftarrow tgt} \in \mathbb{R}^{H \times W \times 3N}$ as input to the ControlNet. The key innovation is **depth-guided sampling**: the target head depth map $\tilde{D}_{tgt} = \text{Render}(M_{tgt}; H, K)$ is first rendered; pixels within the head region are sampled in the range $[\tilde{D}_{tgt}[u,v] - \delta, \tilde{D}_{tgt}[u,v] + \delta]$ (with $\delta = 0.01m$), while pixels outside the head region use a predefined range $[d_{near}, d_{far}]$.
    - **Design Motivation**: Uniformly sampled 3D points may lie far from the actual 3D position corresponding to each pixel, causing the 3D flow to fail to accurately reflect 2D motion changes. Depth-guided sampling concentrates the sampled points near the actual surface, ensuring consistency between the 3D flow encoding and 2D motion. Experiments demonstrate that this reduces APD from 9.659 to 2.682.

3. **User-Specified Expression and Pose Editing**

    - **Function**: Supports feedforward facial expression and head pose editing at inference time.
    - **Mechanism**: The user specifies a FLAME expression increment $\Delta\psi_{usr}$ or pose increment $\Delta\theta_{usr}$, which is directly added to the driving parameters: $\psi_{dri} \leftarrow \psi_{dri} + \Delta\psi_{usr}$. The target mesh $M_{tgt}$ and corresponding 3D flow encoding are then recomputed. Since the motion condition is entirely driven by the parametric model, editing is immediate and controllable.
    - **Design Motivation**: Conventional diffusion-based portrait animation methods accept only driving image inputs and cannot directly control specific expression or pose parameters. The 3D flow framework in FG-Portrait inherently supports parameter-level editing without additional training.

### Loss & Training

The standard diffusion model denoising loss is used: $\mathcal{L} = \mathbb{E}_{z_0,c,\epsilon,t}[\|\epsilon - U(z_t, t, c)\|_2^2]$. SD 1.5 serves as the backbone with its weights frozen. The appearance network is initialized from X-Portrait, and the additional input layers of the ControlNet are randomly initialized. The AdamW optimizer is used with a learning rate of $1e^{-5}$. After training the image diffusion pipeline, temporal layers are inserted and fine-tuned on video sequences to achieve temporal consistency. Training data consists of 1K videos sampled from the VFHQ dataset.

## Key Experimental Results

### Main Results

VFHQ self-reenactment at 512×512:

| Method | LPIPS↓ | CSIM↑ | APD↓ | AED↓ |
|--------|--------|-------|------|------|
| EMOPortrait | 0.235 | 0.729 | 3.047 | 0.371 |
| X-Portrait | 0.195 | 0.777 | 3.660 | 0.357 |
| Follow-Your-Emoji | 0.162 | 0.774 | 3.570 | 0.402 |
| HunyuanPortrait | 0.162 | 0.781 | 3.440 | 0.341 |
| **Ours** | **0.158** | **0.807** | **2.682** | **0.327** |

VFHQ cross-reenactment:

| Method | FID↓ | CSIM↑ | APD↓ | AED↓ |
|--------|------|-------|------|------|
| EMOPortrait | 100.6 | 0.386 | 7.860 | 0.660 |
| Face-Adapter | 94.6 | 0.424 | 7.785 | 0.688 |
| HunyuanPortrait | 92.7 | 0.455 | 9.220 | 0.658 |
| **Ours** | **87.0** | 0.462 | **7.764** | **0.652** |

FFHQ cross-dataset generalization: FID 99.4 (best), APD 9.297 (best), AED 0.714 (best).

### Ablation Study

Motion condition comparison (Tab. 4):

| Motion Condition | S-APD↓ | S-AED↓ | C-APD↓ | C-AED↓ |
|------------------|--------|--------|--------|--------|
| Driving Landmark | 4.001 | 0.373 | 8.588 | 0.688 |
| Predicted Flow | 4.232 | 0.384 | 12.430 | 0.778 |
| **Ours (3D Flow)** | **2.682** | **0.327** | **7.764** | **0.652** |

Depth-guided sampling ablation (Tab. 5):

| Configuration | LPIPS↓ | CSIM↑ | APD↓ | AED↓ |
|---------------|--------|-------|------|------|
| w/o Depth (uniform sampling) | 0.213 | 0.770 | 9.659 | 0.730 |
| **w/ Depth** | **0.158** | **0.807** | **2.682** | **0.327** |

Hyperparameter ablation: $N=20$ and $\delta=0.01m$ yield stable performance across configurations; $N=10$ results in a slight performance drop.

### Key Findings

- **3D Flow yields decisive improvements in motion transfer**: APD is reduced by 33% compared to landmark conditioning (4.001→2.682) and by 37% compared to predicted flow, demonstrating the substantial advantage of geometry-driven, learning-free motion correspondences.
- **Depth-guided sampling is critical for 3D flow encoding**: Without depth guidance, APD reaches 9.659; with depth guidance, it drops to 2.682, an improvement of approximately 72%, underscoring the importance of querying flow at the correct 3D location.
- In cross-reenactment, CSIM is slightly lower than Follow-Your-Emoji (0.462 vs. 0.484), which represents a reasonable trade-off: more accurate motion transfer inherently involves a balance between identity preservation and motion precision.
- Temporal consistency (FVD) is 412.1, second only to Follow-Your-Emoji (382.6), indicating the model is highly competitive in temporal coherence.

## Highlights & Insights

- The **learning-free 3D motion correspondence** is the paper's most significant contribution: FLAME's topological correspondences directly provide per-vertex semantic matching without self-supervised training, without dependence on data volume, and with robustness under extreme poses. This paradigm is transferable to body animation (via SMPL), hand animation, or any domain with an available parametric model.
- The approach of **encoding 3D information as 2D conditioning signals** is elegant: stacking the flow values sampled at multiple 3D points along each ray preserves 3D information while remaining compatible with the 2D diffusion model input format. Depth guidance further focuses sampling on the effective surface region.
- Inference-time parameter-level editing is a significant practical advantage — users can precisely control expression intensity and head rotation angles, a capability that is not achievable in most diffusion-based animation methods.

## Limitations & Future Work

- The mesh resolution of the FLAME model is limited, which may hinder the representation of fine-grained expressions such as micro-expressions and wrinkle dynamics, a limitation acknowledged by the authors.
- The model is trained only on real human portraits and performs poorly on cartoon or stylized characters (e.g., eyelid closure artifacts), requiring fine-tuning on cartoon data.
- Using SD 1.5 as the backbone is relatively dated; upgrading to more advanced DiT architectures (e.g., SD3, FLUX) could further improve generation quality.
- The accuracy of FLAME fitting directly affects the quality of the 3D flow; fitting may be unreliable under occlusion or extreme illumination conditions.

## Related Work & Insights

- **vs. X-Portrait**: X-Portrait uses the driving image itself as the motion condition, implicitly relying on the model to learn motion correspondences. This fails under large pose variations (APD 3.660 vs. 2.682). FG-Portrait provides explicit geometric guidance, substantially reducing the learning burden.
- **vs. HunyuanPortrait**: HunyuanPortrait employs a stronger DiT backbone and video diffusion model, yet still underperforms FG-Portrait in motion accuracy (APD 3.440 vs. 2.682), suggesting that the design of the motion condition is more critical than the choice of backbone architecture.
- **vs. EMOPortrait**: EMOPortrait is a representative learned motion field method based on GANs rather than diffusion models. It is substantially outperformed by FG-Portrait in both motion accuracy and image quality, validating the generalization limitations of learning-based motion estimation approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — 3D flow as a learning-free motion correspondence is an elegant and original contribution; the depth-guided encoding strategy is also novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers both self-reenactment and cross-reenactment settings, evaluated on VFHQ and FFHQ datasets with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem motivation is clearly articulated, the method is described concisely and elegantly, and the figures are highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Introduces a new motion guidance paradigm for portrait animation; the 3D flow concept is broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ExpPortrait: Expressive Portrait Generation via Personalized Representation](expportrait_expressive_portrait_generation_via_personalized_representation.md)
- [\[CVPR 2026\] Ani3DHuman: Photorealistic 3D Human Animation with Self-guided Stochastic Sampling](ani3dhuman_photorealistic_3d_human_animation_with_self-guided_stochastic_samplin.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](../../ICCV2025/image_generation/domain_generalizable_portrait_style_transfer.md)
- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[ICCV 2025\] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](../../ICCV2025/image_generation/structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)

</div>

<!-- RELATED:END -->
