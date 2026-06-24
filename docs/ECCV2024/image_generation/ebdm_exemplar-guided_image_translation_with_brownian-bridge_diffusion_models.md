---
title: >-
  [Paper Note] EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models
description: >-
  [ECCV 2024][Image Generation][Exemplar-guided image translation] This paper proposes the EBDM framework, which models exemplar-guided image translation as a stochastic Brownian-bridge diffusion process, directly translating structural controls into realistic images. By integrating a Global Encoder, an Exemplar Network, and an Exemplar Attention Module, the framework effectively incorporates both the global style and detailed texture information of the exemplar image.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Exemplar-guided image translation"
  - "Brownian-bridge diffusion models"
  - "texture transfer"
  - "conditional image generation"
  - "style control"
date: 2026-05-08
content_hash: 43501d56a33d9bc0
---

# EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2410.09802](https://arxiv.org/abs/2410.09802)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Exemplar-guided image translation, Brownian-bridge diffusion models, texture transfer, conditional image generation, style control

## TL;DR

This paper proposes the EBDM framework, which models exemplar-guided image translation as a stochastic Brownian-bridge diffusion process, directly translating structural controls into realistic images. By integrating a Global Encoder, an Exemplar Network, and an Exemplar Attention Module, the framework effectively incorporates both the global style and detailed texture information of the exemplar image.

## Background & Motivation

Exemplar-guided image translation aims to generate realistic images that conform to both structural controls (such as semantic segmentation masks, edge maps, and pose keypoints) and style exemplars. This has significant application value in user-controllable style manipulation.

Existing methods face three major challenges:

**1. Limitations of Dense Correspondence**: Mainstream methods (e.g., CoCosNet, RABIT) rely on establishing dense correspondences between cross-domain inputs. This incurs quadratic memory and computational costs and yields poor matching quality in sparse correspondence scenarios (e.g., semantic masks to real images), leading to local distortion and semantic inconsistency.

**2. Inadequacy of Text Prompts**: Although diffusion models excel at text-to-image generation, accurately describing every detail of an image (especially visual attributes such as texture and color) using text is difficult, and CLIP embeddings are insufficient to capture all visual details.

**3. Sensitivity to Multi-condition Injection**: When existing diffusion-based methods (such as the combination of ControlNet and IP-Adapter) use both structural control and style conditions simultaneously, they are highly sensitive to hyperparameters like guidance scales, making stable generation difficult.

The core innovation of EBDM lies in utilizing the Brownian-bridge diffusion process to treat the structural control as the fixed starting point of the diffusion, directly translating it into a realistic image without needing an additional structural condition injection mechanism. This allows the network to focus on learning the integration of exemplar style information, resulting in more robust training and inference.

## Method

### Overall Architecture

EBDM is based on the Brownian-bridge Diffusion Model (BBDM) within the Stable Diffusion framework, consisting of three core components:

1. **Denoising U-Net**: Learns the direct translation from structural control to real images based on the Brownian-bridge process.
2. **Global Encoder**: Extracts the global style information of the exemplar image using DINOv2.
3. **Exemplar Network + Exemplar Attention Module**: Extracts and integrates the detailed texture information of the exemplar image.

**Key Differences between Brownian-bridge Diffusion and Standard Diffusion**:
- Standard DDPM: $x_T \sim \mathcal{N}(0, I)$ (the endpoint is pure Gaussian noise)
- Brownian-bridge: $(x_T, x_0) \sim q_{\text{data}}(\mathcal{X}, \mathcal{Y})$ (both endpoints are fixed data points)

Specifically, $x_T = z_\mathcal{X}$ is the latent code of the structural control, and $x_0 = z_{\mathcal{X} \to \mathcal{Y}}$ is the latent code of the target image. The forward process is defined as:

$$q(x_t | x_0, y) = \mathcal{N}(x_t; (1-m_t)x_0 + m_t y, \delta_t I)$$

where $m_t = t/T$ and $\delta_t = 2(m_t - m_t^2)$. This means the Denoising U-Net directly learns the translation from structural control to the image, **without requiring explicit structural condition injection**.

### Key Designs

**1. Global Encoder (Global Style Encoding)**

Choosing DINOv2 (instead of CLIP) as the global style encoder because:
- The self-supervised learning strategy of DINOv2 makes it superior to CLIP in capturing semantic features.
- This method does not require text-image alignment, rendering CLIP's text-alignment capability advantageous-free in this scenario.

Processing method:
$$\tau_\theta(I_\mathcal{Y}) = \text{Linear}(\text{DINO}(I_\mathcal{Y})_{[\text{CLS}]}) \in \mathbb{R}^c$$

The [CLS] token extracted from DINO is mapped through a linear layer and injected into the denoising process as global style information via the cross-attention mechanism.

**2. Exemplar Network (Detail Texture Network)**

The Global Encoder is limited by the input resolution ($224^2$) and cannot preserve fine-grained texture details. Therefore, the Exemplar Network $\psi_\theta$ is introduced:

- It adopts a siamese structure similar to the denoising U-Net, with redundant layers removed to improve efficiency.
- It encodes the exemplar image $z_\mathcal{Y}$ into multi-layer feature maps $\{F_1^l\}_{l=0}^N$.
- It receives global style information via cross-attention in each block.

**3. Exemplar Attention Module (Exemplar Attention Module)**

Since the exemplar image and target control are not spatially aligned, simple concatenation or addition is inapplicable. A spatial attention fusion scheme is proposed:

- Concat the exemplar feature $F_1^l$ and the denoising feature $F_2^l$ along the spatial dimension: $F_{\text{in}}^l = \text{concat}(F_1^l, F_2^l) \in \mathbb{R}^{C \times H \times 2W}$
- Apply self-attention to the concatenated feature, enabling the denoising feature to query relevant textures from the exemplar.
- Extract the corresponding denoising feature portion as the output via a Chunk operation.

$$Q = \phi_q^l(F_{\text{in}}^l), \quad K = \phi_k^l(F_{\text{in}}^l), \quad V = \phi_v^l(F_{\text{in}}^l)$$

$$F_{\text{EA}}^l = W^l \text{Softmax}(QK^T / \sqrt{V}) V + F_{\text{in}}^l$$

This design avoids the high overhead of dense correspondence matching while allowing the model to adaptively select relevant textures from the exemplar.

### Loss & Training

**Two-stage Training**:

- **First Stage**: Trains the cross-attention of the denoising U-Net and the Global Encoder, learning the translation from control to image along with rough style fusion. A reconstruction task (using the target image itself as the exemplar) is used, and the pretrained parameters of the VAE and the Global Encoder are frozen.
- **Second Stage**: Incorporates the Exemplar Network and the Exemplar Attention Module, freezes the first-stage parameters, and focuses on training detail texture integration. Predefined exemplar-target pairs are used.

**Training Objective**:

$$\mathbb{E}_{x_0, y, I_\mathcal{Y}, \epsilon}[c_{\epsilon t} \| m_t(x_T - x_0) + \sqrt{\delta_t}\epsilon - \epsilon_\theta(x_t, t, \tau_\theta(I_\mathcal{Y}), \psi_\theta(z_\mathcal{Y}, \tau_\theta(I_\mathcal{Y}))) \|^2]$$

**Inference**: Uses a deterministic ODE sampler to step-by-step denoise starting from the structural control origin, requiring only a single exemplar condition.

## Key Experimental Results

### Main Results

Image quality comparison (FID ↓ / SWD ↓ / LPIPS ↑, three tasks):

| Method | DeepFashion FID | CelebA-HQ Edge FID | CelebA-HQ Mask FID |
|------|----------------|--------------------|--------------------|
| CoCosNet | 14.40 | 14.30 | 21.83 |
| CoCosNetv2 | 12.81 | 12.85 | 20.64 |
| RABIT | 12.58 | 11.67 | 20.44 |
| MIDMs | 10.89 | 15.67 | N/A |
| **EBDM (Ours)** | **10.62** | **11.84** | **12.21** |

Comparison with SOTA diffusion methods (CelebA-HQ Edge):

| Method | SSIM ↑ | PSNR ↑ |
|------|--------|--------|
| ControlNet | 0.882 | 35.30 |
| ControlNet+CLIP | 0.894 | 35.94 |
| **EBDM (Ours)** | **0.901** | **36.40** |

### Ablation Study

Global Encoder selection comparison (CelebA-HQ Edge):

| Configuration | SSIM ↑ | FID ↓ | Sem. ↑ |
|------|--------|-------|--------|
| Baseline (w/o Global Encoding) | 0.831 | 16.31 | 0.531 |
| + CLIP | 0.632 | 23.42 | 0.752 |
| + DINO | 0.754 | 21.32 | 0.786 |
| **Full Method (EBDM)** | **0.901** | **11.84** | **0.920** |

Using CLIP as the global encoder instead significantly degrades the SSIM (from 0.831 to 0.632), as its text-alignment property is not advantageous for this task. DINOv2 combined with the full framework achieves the best performance.

### Key Findings

1. **Fundamental Advantage of Brownian-bridge Diffusion**: Treating structural control as a diffusion endpoint rather than an additional condition naturally maintains structural consistency and frees up conditional capacity for style fusion.
2. **DINOv2 Outperforms CLIP as a Visual Style Encoder**: Self-supervised learning features are significantly superior to contrastive learning features in representing fine-grained visual similarity.
3. **Largest Improvement on the Mask-to-Photo Task** (FID 12.21 vs. second best 20.44): Since matching methods struggle to establish effective correspondences on semantic segmentation masks, while diffusion methods naturally handle this through iterative denoising.
4. **Robustness of Single Condition**: Compared to multi-condition combinations like ControlNet + IP-Adapter, EBDM achieves better results using only the exemplar condition, free from hyperparameter sensitivity issues.

## Highlights & Insights

1. **Ingenious Application of Brownian-bridge Diffusion**: Formulating image translation naturally as a stochastic process between two fixed endpoints avoids the complexity of structural condition injection, offering a more elegant structure-preserving solution than ControlNet.
2. **Departure from the Dense Correspondence Matching Paradigm**: It demonstrates that the diffusion framework can entirely replace the traditional "matching-then-generation" pipeline, simultaneously improving both computational efficiency and generation quality.
3. **Complementary Global + Local Design**: The DINOv2 [CLS] token captures global style, while the Exemplar Network captures local textures, establishing a complementary dual-path design.
4. **Spatial Self-Attention Fusion in Exemplar Attention**: This successfully addresses the core challenge of integrating unaligned features.

## Limitations & Future Work

1. Validation is limited to face (CelebA-HQ) and fashion (DeepFashion) datasets; more complex scene-level tasks have not been tested.
2. Semantic consistency scores (Tab. 4) underperform methods like DynaST on certain metrics, potentially due to variations introduced by the stochastic nature of the Brownian bridge.
3. The two-stage training increases implementation complexity.
4. The spatial concatenation in the Exemplar Attention Module may incur high GPU memory overhead at higher resolutions.
5. Extension to temporal consistency for video sequences has not been explored.

## Related Work & Insights

- **BBDM**: This work is the first to apply the Brownian-bridge Diffusion Model to exemplar-guided image translation, demonstrating the advantages of this diffusion paradigm in cross-domain translation.
- **ControlNet / IP-Adapter**: Although flexible, the hyperparameter sensitivity of multi-condition combinations acts as a bottleneck for practical deployment. EBDM's single-condition design proves much more robust.
- **CoCosNet Series**: These established the "matching-then-generation" paradigm, but EBDM demonstrates that comparable or even superior performance can be achieved without explicit matching.
- **Insights**: The Brownian-bridge diffusion process can be extended to other cross-domain translation tasks (e.g., semantic-to-real, sketch-to-photo, day-to-night).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of Brownian bridge to exemplar-guided translation, featuring an elegant framework design.
- Technical Depth: ⭐⭐⭐⭐ — The three components are well-designed, and the ablation study clearly validates each design choice.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three tasks + comparisons with multiple baselines + ablation studies.
- Value: ⭐⭐⭐⭐ — Direct applications in scenarios like virtual try-on and face editing.
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[CVPR 2026\] DBMSolver: A Training-free Diffusion Bridge Sampler for High-Quality Image-to-Image Translation](../../CVPR2026/image_generation/dbmsolver_a_training-free_diffusion_bridge_sampler_for_high-quality_image-to-ima.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ICCV 2025\] TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation](../../ICCV2025/image_generation/tlb-vfi_temporal-aware_latent_brownian_bridge_diffusion_for_video_frame_interpol.md)

</div>

<!-- RELATED:END -->
