---
title: >-
  [Paper Note] AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation
description: >-
  [AAAI 2026][3D Vision][Text-to-3D] This paper reveals a key issue in SDS: the source distribution is dynamically evolving rather than static. To address this, it proposes AnchorDS, which anchors the source distribution by feeding the current rendered image as an image condition into a dual-conditioned diffusion model. This resolves semantic over-smoothing and multi-view inconsistency in SDS, comprehensively outperforming SDS/VSD/SDS-Bridge on T3Bench.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Text-to-3D"
  - "Score Distillation Sampling"
  - "Diffusion Models"
  - "3DGS"
  - "Dynamic Source Distribution"
  - "Semantic Consistency"
date: 2026-05-08
content_hash: 8d8eb16ef3246473
---

# AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation

**Conference**: AAAI 2026  
**arXiv**: [2511.11692](https://arxiv.org/abs/2511.11692)  
**Code**: [https://github.com/viridityzhu/AnchorDS](https://github.com/viridityzhu/AnchorDS)  
**Area**: 3D Vision  
**Keywords**: Text-to-3D, Score Distillation Sampling, Diffusion Models, 3DGS, Dynamic Source Distribution, Semantic Consistency

## TL;DR
This paper reveals a key issue in SDS: the source distribution is dynamically evolving rather than static. To address this, it proposes AnchorDS, which anchors the source distribution by feeding the current rendered image as an image condition into a dual-conditioned diffusion model. This resolves semantic over-smoothing and multi-view inconsistency in SDS, comprehensively outperforming SDS/VSD/SDS-Bridge on T3Bench.

## Background & Motivation
**Background**：Optimization-based text-to-3D methods distill gradients from pretrained 2D diffusion models using SDS to train NeRF/3DGS, generating 3D content without requiring 3D data.

**Limitations of Prior Work**：SDS suffers from two typical limitations: (1) **Semantic over-smoothing**: unique semantic features of objects degrade into blurry, uniform representations (e.g., a swan merging into the lake water); (2) **Multi-view inconsistency**: incoherent geometry and appearance across different viewpoints (e.g., the multi-head/Janus problem).

**Key Challenge**：Through mathematical analysis, the authors identify the root cause: the CFG gradient in SDS can be decomposed into a "target term $m_1$" (pushing towards the text-conditioned distribution) and a "variance term $m_2$" (pushing away from the source distribution). Crucially, the source distribution is approximated by an unconditional prior $p(z_t; t, \emptyset)$, completely ignoring the dynamic changes of the 3D model during optimization. Therefore, $\hat{z}_{t \to 0}^{\text{source}}$ neither encodes the semantics of the current 3D state nor reflects the existing geometry, causing the gradient direction to mismatch the actual 3D state.

**Key Insight**：Re-interpreting SDS as a **dynamic editing process**—each step acts as a progressive edit based on the current 3D state, meaning the source distribution should evolve together with the 3D model.

**Core Idea**：Utilizing the current rendered image as an image condition to input into the diffusion model, replacing the unconditional prior to estimate the source distribution, thus achieving "state-anchored" gradient guidance.

## Method

### Overall Architecture
The core modification of AnchorDS lies in the SDS gradient computation: replacing the unconditional noise prediction $\hat{\epsilon}_\phi(z_t; t, \emptyset)$ in the source distribution estimation with an image-conditioned noise prediction $\hat{\epsilon}_\phi(z_t; t, \emptyset, I^{(\tau)})$, where $I^{(\tau)}$ is the rendered image at the current optimization step $\tau$.

The new guidance gradient is: $g_t^{(\tau)} = \hat{\epsilon}_\phi(z_t; t, y) - \hat{\epsilon}_\phi(z_t; t, \emptyset, I^{(\tau)})$

The target term remains unchanged (text-conditioned), but the source term now encodes the structural and semantic information of the current 3D state.

Overall pipeline: Render the current 3D model at each step $\to$ encode to latent + add noise $\to$ query the diffusion model twice (text condition for target prediction, image condition for source prediction) $\to$ backpropagate the difference gradient to update 3D parameters.

### Key Designs

1. **Dynamic Source Distribution Anchoring (Core)**:

    - **Function**: Replacing the unconditional prior with an image-conditioned diffusion model to estimate the source distribution.
    - **Mechanism**: Utilizing IP-Adapter or ControlNet to feed the current rendered image $I^{(\tau)}$ as an image condition into the diffusion model. The image condition does not constrain the output content but acts as a **contextual anchor** to guide the generation, preserving the structural information of the current 3D state.
    - **Design Motivation**: Pretrained image-conditioned diffusion models naturally possess image inversion capabilities—directly utilizing the model's intrinsic image $\to$ latent mapping to achieve precise source anchoring without extra inversion steps. This only requires one extra U-Net forward pass (which can be run in parallel with the original pass), maintaining the same runtime as standard SDS.

2. **Pseudo-source Reconstruction and Quality Assessment**:

    - **Function**: Explicitly reconstructing source images and providing a quantitative metric for the estimation quality of the source distribution.
    - **Mechanism**: Deriving the one-step denoised latent $\hat{z}_{t \to 0}^{\text{anchored}}$ from the image-conditioned noise prediction, decoding it to obtain a reconstructed image, and computing the L2 distance from the original rendered image: $\mathcal{L}_{\text{rec}} = \| \varepsilon(\hat{z}_{t \to 0}^{\text{anchored}}) - I^{(\tau)} \|_2^2$
    - **Design Motivation**: This reconstruction loss serves as both a quality metric and the foundation for two enhancement strategies.

3. **Filtering Strategy**:

    - **Function**: Setting a threshold $\gamma$ based on $\mathcal{L}_{\text{rec}}$ to discard source estimations with excessively large reconstruction errors.
    - **Mechanism**: Zeroing out the AnchorDS loss when $\mathcal{L}_{\text{rec}} > \gamma$ to skip unreliable gradient updates.
    - **Design Motivation**: Simply and effectively filtering out abnormal predictions caused by domain shifts in the image condition, thereby improving training stability.

4. **Fine-tuning Strategy**:

    - **Function**: Lightly fine-tuning a single layer of IP-Adapter to bridge the domain gap between real and rendered image distributions.
    - **Mechanism**: Updating the parameters of a single layer of the image adapter using the gradient of $\mathcal{L}_{\text{rec}}$, allowing it to "see" data from the rendering domain. The overhead is minimal (training time increases from ~25 minutes to ~30 minutes).
    - **Design Motivation**: Pretrained 2D models are trained on real images, causing distribution discrepancies when processing synthetic rendered images.

### Loss & Training
- AnchorDS gradient: $\nabla_\Theta \mathcal{L}_{\text{AnchorDS}} = w(t) \cdot g_t^{(\tau)} \cdot \frac{\partial z_t}{\partial \Theta}$
- Source reconstruction loss: $\mathcal{L}_{\text{rec}} = \| \varepsilon(\hat{z}_{t \to 0}^{\text{anchored}}) - I^{(\tau)} \|_2^2$
- Filtering and Fine-tuning are mutually exclusive options (complementary strategies).
- Default image conditioner: IP-Adapter (SD 1.5) or ControlNet (SD 2.1).
- 3D representation: Supports 3DGS (GaussianDreamer) and NeRF.

## Key Experimental Results

### Main Results
T3Bench benchmark (300 prompts, consisting of three categories: single object, single object with environment, and multi-object):

| Method | All↑ | Single↑ | Surr↑ | Multi↑ |
|------|------|---------|-------|--------|
| SDS (DreamFusion) | 20.5 | 24.9 | 19.3 | 17.3 |
| SDS (GaussianDreamer) | 29.7 | 42.3 | 26.1 | 20.6 |
| AnchorDS (IP-Adapter) + Finetune | **33.3** | **45.3** | **29.0** | 25.7 |
| AnchorDS (ControlNet) + Filter | 33.2 | **46.1** | 29.4 | 24.0 |

User Study (912 participants):

| Method | CLIP↑ | 3D Consistency Q1↓ | Text Alignment Q2↓ | Visual Quality Q3↓ |
|------|-------|------------|-----------|-----------|
| VSD (SD 2.1) | 0.352 | 1.84 | 1.85 | 1.79 |
| **Ours (ControlNet, SD 2.1)** | **0.369** | **1.16** | **1.15** | **1.21** |
| VSD (SD 1.5) | 0.281 | 1.99 | 2.00 | 2.08 |
| SDS-Bridge (SD 1.5) | 0.233 | 2.38 | 2.35 | 2.29 |
| **Ours (IP-Adapter, SD 1.5)** | **0.334** | **1.63** | **1.66** | **1.63** |

### Ablation Study

| Configuration | All↑ | Description |
|------|------|------|
| SDS baseline | 29.7 | Baseline |
| AnchorDS (IP-Adapter) | 30.7 | Source anchoring only +1.0 |
| + Filter | 32.8 | Filter unstable predictions +3.1 |
| + Finetune | **33.3** | Fine-tune adapter +3.6 |

### Key Findings
- **Source anchoring itself is effective** (+1.0), with Filter and Finetune providing additional individual gains.
- **Largest improvement in multi-object scenes**: The Multi category jumps from 20.6 to 25.7 (+24.8%) because source anchoring effectively prevents semantic blending between different objects.
- **Insensitive to the choice of image-conditioning model**: Both IP-Adapter and ControlNet are effective, demonstrating the generalizability of the method.
- **VSD, although more elaborate in distribution modeling, still exhibits unnatural colors and structures** due to neglecting source dynamics.
- **SDS-Bridge introduces new biases** (such as materials and texture biases) via manual negative prompts, showing less flexibility.

## Highlights & Insights
- **Precise mathematical and analytical grounding**: Decomposing the SDS gradient into $m_1$ and $m_2$, and further expanding it into a pseudo-editing formulation (Eq.8), clearly exposes how unconditional source estimation discards current 3D state details. This analytical framework is far more convincing than intuitive explanations.
- **Extremely lightweight**: The core modification simply replaces the unconditional branch of the CFG with an image-conditioned branch—a one-line code change requiring no extra networks or training data. The Filter/Finetune strategies are also inherently simple.
- **"Dynamic source distribution" perspective is generalizable to other distillation scenarios**: Such as 2D editing or video generation variants of SDS. Any context employing SDS for iterative optimization could benefit from similar source anchoring.

## Limitations & Future Work
- **Still reliant on the SDS paradigm**: Inherits the native slowness of SDS (requiring thousands of optimization steps), making it less suitable for real-time applications.
- **Upper bound of image-conditioned model capacity**: If IP-Adapter/ControlNet has poor inversion performance on specific rendering styles, the quality of source estimation degrades (which is why the Filter strategy is necessary).
- **Lack of comparison against feed-forward 3D generation methods**: Evaluation on T3Bench only compares against SDS variants, lacking comparisons with native 3D generative models.

## Related Work & Insights
- **vs VSD (ProlificDreamer)**: VSD trains a particle distribution model using LoRA to approximate the source distribution, incurring high computational costs (running 4 models). AnchorDS directly uses the rendered image as a condition, requiring zero additional models while performing better.
- **vs SDS-Bridge**: SDS-Bridge manually describes the 3D state using negative prompts to correct source deviations, which introduces brand new biases. AnchorDS lets the model directly "see" the current rendering, resulting in zero bias.
- **vs DDS**: DDS also uses reference images, but requires paired reference prompts and serves a different purpose. In AnchorDS, the image condition is automatically acquired (as the current rendering).

## Rating
- Novelty: ⭐⭐⭐⭐ Deep mathematical analysis of the SDS source distribution issue; elegant and simple method.
- Experimental Thoroughness: ⭐⭐⭐⭐ T3Bench + User Study + Ablation + comparisons across multiple baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, intuitive illustrations, and a complete logical chain.
- Value: ⭐⭐⭐⭐ Significant improvement within the SDS framework; simple and highly reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](../../ECCV2024/3d_vision/dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](../../ECCV2024/3d_vision/dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](../../CVPR2026/3d_vision/text-image_conditioned_3d_generation.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[ICML 2026\] RelaxFlow: Text-Driven Amodal 3D Generation](../../ICML2026/3d_vision/relaxflow_text-driven_amodal_3d_generation.md)

</div>

<!-- RELATED:END -->
