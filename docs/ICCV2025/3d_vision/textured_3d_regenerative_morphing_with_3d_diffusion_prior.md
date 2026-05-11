---
title: >-
  [Paper Note] Textured 3D Regenerative Morphing with 3D Diffusion Prior
description: >-
  [ICCV 2025][3D Vision][3D Morphing] This paper proposes a regenerative 3D morphing method based on a 3D diffusion prior. By performing interpolation at three levels — initial noise, model parameters…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Morphing"
  - "Diffusion Models"
  - "Textured 3D Representation"
  - "Attention Fusion"
  - "Frequency-Domain Enhancement"
date: 2026-05-08
content_hash: 6b83b5b73a859d4c
---

# Textured 3D Regenerative Morphing with 3D Diffusion Prior

**Conference**: ICCV 2025
**arXiv**: [2502.14316](https://arxiv.org/abs/2502.14316)
**Code**: None
**Area**: 3D Vision
**Keywords**: 3D Morphing, Diffusion Models, Textured 3D Representation, Attention Fusion, Frequency-Domain Enhancement

## TL;DR

This paper proposes a regenerative 3D morphing method based on a 3D diffusion prior. By performing interpolation at three levels — initial noise, model parameters, and conditioning features — and combining three strategies (Attention Fusion, Token Reordering, and Low-Frequency Enhancement), it is the first to achieve smooth and semantically plausible morphing sequences for textured 3D objects across categories.

## Background & Motivation

3D morphing aims to generate smooth and plausible interpolation sequences between two 3D objects, which is critical for creative applications such as visual effects in film and television. Compared to image morphing, 3D morphing is more challenging as it requires holistic interpolation of 3D objects (image morphing can be viewed as a special case from a fixed viewpoint).

**Limitations of Prior Work**:

**Shape-only morphing**: Prior methods primarily rely on establishing point-to-point correspondences and determining smooth deformation trajectories, restricting them to texture-free, topologically aligned datasets (e.g., FAUST human body shapes, Shrec'20 quadrupeds), with no support for texture.

**Labor-intensive preprocessing**: Morphing new data requires cumbersome registration and matching steps.

**Limited morphing capability**: Constrained by insufficient object diversity and small-scale datasets, resulting in blurry and implausible interpolations.

The authors raise two key questions: (a) Are explicit point-to-point correspondences truly necessary? (b) Can generalization in textured 3D morphing be enhanced via a general generative prior?

**Mechanism**: The method leverages the implicit correspondence capacity and generative capability of 3D diffusion models to fuse source and target information, regenerating interpolated textured 3D representations — termed "Regenerative Morphing."

## Method

### Overall Architecture

The method builds on Gaussian Anything as the 3D diffusion prior, a two-stage native 3D diffusion model: the first stage produces structured point cloud representations via a geometry diffusion model $\epsilon_G$, and the second stage generates texture features via a texture diffusion model $\epsilon_T$. The overall pipeline consists of three steps: Basic Interpolation, smoothness improvement via Attention Fusion, and plausibility improvement via Token Reordering and Low-Frequency Enhancement.

### Key Designs

1. **Three-Level Basic Interpolation**:

   Interpolation between source and target is performed with weights $(1-\alpha)$ and $\alpha$ at three levels:
   - **Initial noise interpolation**: The input noise of the source and target is obtained via diffusion inversion; spherical linear interpolation (SLERP) is applied to generate intermediate noise $[\mathbf{z}_T^\alpha, \mathbf{z}_G^\alpha]$ to preserve Gaussian noise properties.
   - **Model parameter interpolation**: LoRA fine-tuning is performed separately for the source and target; the two sets of LoRA parameters are linearly interpolated to obtain morphing models $\epsilon_G^\alpha$ and $\epsilon_T^\alpha$.
   - **Conditioning feature interpolation**: Text prompts for the source and target are encoded via a CLIP encoder as $\mathbf{c}^{src}$ and $\mathbf{c}^{tgt}$, and linearly interpolated to obtain $\mathbf{c}^\alpha$.

   However, basic interpolation suffers from two issues: abrupt transitions (due to mapping variability from nonlinear multi-step denoising) and artifacts (from misalignment between the conditioning space and the diffusion space).

2. **Attention Fusion**:

   The source, target, and interpolated noises are simultaneously fed into the morphing model to obtain three sets of $(Q, K, V)$; fused attention is then applied to improve smoothness:

   $\text{Fused-Attn}(Q^\alpha, K^\alpha, V^\alpha) = \text{Attn}(Q^\alpha, [(1-\alpha)K^{src} + \alpha K^{tgt}, K^\alpha], [(1-\alpha)V^{src} + \alpha V^{tgt}, V^\alpha])$

   This strategy combines self-attention and cross-attention fusion, leveraging unified attention features from the fine-tuned model to enhance smoothness. However, excessive Attention Fusion can cause structural collapse and surface quality degradation.

3. **Token Reordering**:

   **Design Motivation**: 3D objects are tokenized into sequences $\{h_j\}_{j=1}^M$; relying solely on implicit correspondence via the attention mechanism may lead to semantically implausible associations (e.g., matching chair legs to donut icing). Analysis reveals that 3D diffusion features do capture semantic correspondences.

   **Implementation**: Token sequences of source and target are reordered between DiT blocks so that semantically similar tokens are aligned to the same index positions:

   $\text{minimize} \sum_{j=1}^{M} \|h_j^{src} - h_{\sigma(j)}^{tgt}\|$

   Different strategies are applied based on $\alpha$: when $\alpha \in [0, 0.5)$, the target is reordered relative to the source; when $\alpha \in [0.5, 1]$, the source is reordered relative to the target.

4. **Low-Frequency Enhancement**:

   Frequency-domain analysis reveals that in 3D generation, low-frequency noise governs global layout while high-frequency noise governs surface details. Excessive attention fusion amplifies high-frequency components and disrupts low-frequency components, thereby degrading 3D surface generation quality.

   **Implementation**: Tokens are transformed to the frequency domain via FFT; low-frequency signals are enhanced and then transformed back via IFFT:

   $F'_{\omega < \omega_0}(h) = F_{\omega < \omega_0}(h) \odot scale$
   $h' = \text{IFFT}([F'_{\omega < \omega_0}(h), F_{\omega \geq \omega_0}(h)])$

   where $scale = 5$ and $\omega_0 = 0.1\pi$.

### Loss & Training

- LoRA fine-tuning parameters: rank=16, alpha=20, target layers=['to_k', 'to_q', 'to_v', 'qkv'], trained for 500 steps.
- 250 denoising timesteps are used.
- $\alpha$ is sampled from a Beta distribution at 10 interpolation points.
- Attention Fusion range: geometry diffusion steps 1–120~180; texture diffusion steps 1–5.
- Token Reordering range: steps 80–200.
- Low-Frequency Enhancement range: steps 200–230.

## Key Experimental Results

### Main Results

| Method | FID↓ | STP-GPT↑ | SEP-GPT↑ | PPL↓ | PDV↓ | STP-U↑ | SEP-U↑ |
|---|---|---|---|---|---|---|---|
| DiffMorpher | 218.07 | 0.23 | 0.13 | 5.23 | 0.0535 | 0.435 | 0.300 |
| AID | 115.72 | 0.67 | 0.70 | 4.68 | 0.0118 | 0.380 | 0.505 |
| MV-Adapter | 120.93 | 0.63 | 0.57 | 7.29 | 0.0152 | 0.225 | 0.350 |
| Luma | 95.49 | 0.83 | 0.77 | 7.37 | 0.0007 | 0.415 | 0.330 |
| MorphFlow | 147.70 | 0.87 | 0.90 | 3.10 | 0.0001 | 0.555 | 0.505 |
| **Ours** | **6.36** | **1.00** | **1.00** | **3.02** | **0.0001** | **0.915** | **0.950** |

The proposed method leads across all metrics: FID is reduced to 6.36 (vs. 95.49 for the second-best), GPT-evaluated structural and semantic plausibility both achieve a perfect score of 1.0, and user study scores of STP-U and SEP-U reach 0.915 and 0.950, respectively.

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Basic Interpolation | Baseline | Three-level interpolation provides basic fusion |
| + Attention Fusion (few steps) | Smoothness↑ | Improves transitions; excessive application causes collapse |
| + Token Reordering | Plausibility↑ | Alleviates structural collapse; quality still degrades at distant timesteps |
| + Low-Frequency Enhancement | Smoothness + plausibility balanced | Frequency-domain enhancement preserves surface quality |

### Key Findings

1. 2D diffusion methods (DiffMorpher, AID) suffer from mode collapse and lack 3D consistency.
2. Multi-view diffusion is limited by pixel-level alignment and produces interpolation errors at large spatial distances.
3. Video generation models (Luma) offer limited controllability and poor structural consistency.
4. Regenerative methods using a 3D diffusion prior naturally avoid discontinuity artifacts, as the fusion accounts for the full latent space distribution.

## Highlights & Insights

- **First textured 3D regenerative morphing**: Breaks the limitation of prior methods to shape-only morphing without requiring explicit correspondences.
- **Elegant multi-level fusion strategy**: Interpolation at the noise, parameter, and conditioning levels covers distinct control dimensions of the diffusion model.
- **Frequency-domain analysis guides improvement**: Low/high-frequency signal analysis provides principled understanding of quality degradation, motivating targeted enhancement strategies.
- **Impressive cross-category morphing**: Semantically plausible transitions are generated between highly dissimilar objects such as boots and teddy bears, or pumpkins and mushrooms.

## Limitations & Future Work

- Performance is bounded by the capabilities of the underlying 3D generative model; morphing complex textured 3D objects remains challenging.
- Future work may integrate more advanced 3D generative models (e.g., Trellis) to improve fidelity and diversity.
- Extension to 4D content morphing (e.g., transitions between animation sequences) is a promising direction.
- Hyperparameters such as timestep ranges require tuning for different scenarios.

## Related Work & Insights

- 2D morphing methods such as DiffMorpher and AID inspired the Attention Fusion strategy, though this paper identifies their shortcomings in 3D settings.
- Works such as DIFT demonstrate that diffusion features encode semantic information; this paper extends the finding to the 3D diffusion space.
- The frequency-domain analysis methodology can be applied to other 3D generation tasks to improve surface quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of a 3D diffusion prior to textured 3D morphing; paradigm-level innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive quantitative metrics (FID / GPT evaluation / user study) with clear ablations; large-scale quantitative evaluation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and in-depth analysis; frequency-domain and semantic analyses are convincing.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for textured 3D morphing, though the application domain is relatively specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)

</div>

<!-- RELATED:END -->
