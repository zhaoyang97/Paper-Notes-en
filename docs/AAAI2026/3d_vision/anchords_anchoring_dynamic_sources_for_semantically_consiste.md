---
title: >-
  [Paper Note] AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation
description: >-
  [AAAI 2026][3D Vision][Text-to-3D] This paper identifies a critical yet overlooked issue in SDS: the source distribution is dynamically evolving rather than static. AnchorDS is proposed to anchor the source distribution…
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
content_hash: 2c57753713ec1167
---

# AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation

**Conference**: AAAI 2026  
**arXiv**: [2511.11692](https://arxiv.org/abs/2511.11692)  
**Code**: [https://github.com/viridityzhu/AnchorDS](https://github.com/viridityzhu/AnchorDS)  
**Area**: 3D Vision  
**Keywords**: Text-to-3D, Score Distillation Sampling, Diffusion Models, 3DGS, Dynamic Source Distribution, Semantic Consistency

## TL;DR
This paper identifies a critical yet overlooked issue in SDS: the source distribution is dynamically evolving rather than static. AnchorDS is proposed to anchor the source distribution by feeding the current rendered image as an image condition into a dual-conditioned diffusion model, thereby resolving semantic over-smoothing and multi-view inconsistency in SDS. The method comprehensively outperforms SDS, VSD, and SDS-Bridge on T3Bench.

## Background & Motivation
**Background**: Optimization-based text-to-3D methods distill gradients from pretrained 2D diffusion models via SDS to train NeRF/3DGS, enabling 3D content generation without 3D data.

**Limitations of Prior Work**: SDS suffers from two well-known failure modes — (1) **semantic over-smoothing**: object-specific semantic features degrade into blurry, uniform representations (e.g., swans and lake surfaces blending together); (2) **multi-view inconsistency**: geometry and appearance are incoherent across viewpoints (e.g., the multi-head/Janus problem).

**Key Challenge**: Through mathematical analysis, the authors identify the underlying cause — the CFG gradient in SDS can be decomposed into a "target term $m_1$" (pushing toward the text-conditioned distribution) and a "variance term $m_2$" (pushing away from the source distribution). The source distribution is approximated by the unconditional prior $p(z_t; t, \emptyset)$, which entirely ignores the dynamic evolution of the 3D model during optimization. As a result, $\hat{z}_{t \to 0}^{\text{source}}$ encodes neither the semantics of the current 3D state nor the existing geometric structure, causing the gradient direction to be inconsistent with the actual 3D state.

**Key Insight**: SDS is reinterpreted as a **dynamic editing process** — each step is a progressive edit conditioned on the current 3D state, and the source distribution should co-evolve with the 3D model.

**Core Idea**: The current rendered image is used as an image condition for the diffusion model, replacing the unconditional prior to estimate the source distribution, thereby achieving state-anchored gradient guidance.

## Method

### Overall Architecture
The core modification of AnchorDS lies in the SDS gradient computation: the source distribution estimate is replaced from the unconditional noise prediction $\hat{\epsilon}_\phi(z_t; t, \emptyset)$ to an image-conditioned noise prediction $\hat{\epsilon}_\phi(z_t; t, \emptyset, I^{(\tau)})$, where $I^{(\tau)}$ is the rendered image at the current optimization step $\tau$.

The new guidance gradient is: $g_t^{(\tau)} = \hat{\epsilon}_\phi(z_t; t, y) - \hat{\epsilon}_\phi(z_t; t, \emptyset, I^{(\tau)})$

The target term remains unchanged (text-conditioned), while the source term now encodes the structural and semantic information of the current 3D state.

Overall pipeline: render the current 3D model at each step → encode to latent and add noise → query the diffusion model twice (text-conditioned for target prediction, image-conditioned for source prediction) → backpropagate the difference gradient to update 3D parameters.

### Key Designs

1. **Dynamic Source Distribution Anchoring (Core)**:

    - Function: Replace the unconditional prior with an image-conditioned diffusion model to estimate the source distribution.
    - Mechanism: Using IP-Adapter or ControlNet, the current rendered image $I^{(\tau)}$ is fed as an image condition into the diffusion model. The image condition does not constrain the output content but serves as a **contextual anchor** that guides generation while preserving the structural information of the current 3D state.
    - Design Motivation: Pretrained image-conditioned diffusion models inherently possess image inversion capability — the model's intrinsic image-to-latent mapping is directly leveraged for accurate source anchoring without additional inversion steps. Only one extra U-Net forward pass is required (parallelizable with the original pass), keeping runtime identical to standard SDS.

2. **Pseudo-Source Reconstruction and Quality Assessment**:

    - Function: Explicitly reconstruct the source image and provide a quantitative metric for source distribution estimation quality.
    - Mechanism: The one-step denoised latent $\hat{z}_{t \to 0}^{\text{anchored}}$ is recovered from the image-conditioned noise prediction and decoded to a reconstructed image; the L2 distance to the original rendered image is computed: $\mathcal{L}_{\text{rec}} = \| \varepsilon(\hat{z}_{t \to 0}^{\text{anchored}}) - I^{(\tau)} \|_2^2$
    - Design Motivation: This reconstruction loss serves both as a quality metric and as the foundation for two complementary enhancement strategies.

3. **Filtering Strategy**:

    - Function: A threshold $\gamma$ is applied based on $\mathcal{L}_{\text{rec}}$ to discard source estimates with excessively large reconstruction errors.
    - Mechanism: When $\mathcal{L}_{\text{rec}} > \gamma$, the AnchorDS loss for that step is set to zero, skipping unreliable gradient updates.
    - Design Motivation: A simple yet effective mechanism to filter out anomalous predictions caused by domain shift in the image condition, improving training stability.

4. **Fine-tuning Strategy**:

    - Function: Lightweight fine-tuning of a single layer in the IP-Adapter to bridge the domain gap between real image distributions and rendered image distributions.
    - Mechanism: Gradient updates from $\mathcal{L}_{\text{rec}}$ are used to update the parameters of one layer in the image adapter, exposing it to rendered-domain data. The overhead is minimal (training time increases from ~25 min to ~30 min).
    - Design Motivation: Pretrained 2D models are trained on real images and exhibit a distributional bias when processing synthetic rendered images.

### Loss & Training
- AnchorDS gradient: $\nabla_\Theta \mathcal{L}_{\text{AnchorDS}} = w(t) \cdot g_t^{(\tau)} \cdot \frac{\partial z_t}{\partial \Theta}$
- Source reconstruction loss: $\mathcal{L}_{\text{rec}} = \| \varepsilon(\hat{z}_{t \to 0}^{\text{anchored}}) - I^{(\tau)} \|_2^2$
- Filtering and Fine-tuning are used as alternative complementary strategies.
- Default image conditioners: IP-Adapter (SD 1.5) or ControlNet (SD 2.1)
- 3D representations: supports 3DGS (GaussianDreamer) and NeRF

## Key Experimental Results

### Main Results
T3Bench benchmark (300 prompts, covering single object / single object with surroundings / multiple objects):

| Method | All↑ | Single↑ | Surr↑ | Multi↑ |
|------|------|---------|-------|--------|
| SDS (DreamFusion) | 20.5 | 24.9 | 19.3 | 17.3 |
| SDS (GaussianDreamer) | 29.7 | 42.3 | 26.1 | 20.6 |
| AnchorDS (IP-Adapter) + Finetune | **33.3** | **45.3** | **29.0** | 25.7 |
| AnchorDS (ControlNet) + Filter | 33.2 | **46.1** | 29.4 | 24.0 |

Human evaluation (912 participants):

| Method | CLIP↑ | 3D Consistency Q1↓ | Text Alignment Q2↓ | Visual Quality Q3↓ |
|------|-------|------------|-----------|-----------|
| VSD (SD 2.1) | 0.352 | 1.84 | 1.85 | 1.79 |
| **Ours (ControlNet, SD 2.1)** | **0.369** | **1.16** | **1.15** | **1.21** |
| VSD (SD 1.5) | 0.281 | 1.99 | 2.00 | 2.08 |
| SDS-Bridge (SD 1.5) | 0.233 | 2.38 | 2.35 | 2.29 |
| **Ours (IP-Adapter, SD 1.5)** | **0.334** | **1.63** | **1.66** | **1.63** |

### Ablation Study

| Configuration | All↑ | Notes |
|------|------|------|
| SDS baseline | 29.7 | Baseline |
| AnchorDS (IP-Adapter) | 30.7 | Source anchoring only, +1.0 |
| + Filter | 32.8 | Filter unstable predictions, +3.1 |
| + Finetune | **33.3** | Fine-tune adapter, +3.6 |

### Key Findings
- **Source anchoring alone is effective** (+1.0); Filter and Finetune each provide additional gains.
- **Largest improvement on multi-object scenes**: the Multi category improves from 20.6 to 25.7 (+24.8%), as source anchoring effectively prevents semantic mixing between different objects.
- **Method is insensitive to the choice of image conditioner**: both IP-Adapter and ControlNet are effective, demonstrating the generality of the approach.
- **VSD, despite more refined distribution modeling, still produces unnatural colors and structural artifacts due to neglect of source dynamics.**
- **SDS-Bridge's hand-crafted negative prompts introduce new biases** (e.g., material and texture artifacts), limiting flexibility.

## Highlights & Insights
- **Precise problem analysis with mathematical grounding**: Decomposing the SDS gradient into $m_1$ and $m_2$, and further expanding it into a pseudo-editing formulation (Eq. 8), clearly exposes the information loss caused by the unconditional source estimate. This analytical framework is far more convincing than intuitive explanations alone.
- **Extremely lightweight method**: The core modification amounts to replacing the unconditional branch in CFG with an image-conditioned branch — a change on the order of a single line of code, requiring no additional networks or training data. The Filter/Finetune strategies are equally simple.
- **The "dynamic source distribution" perspective may generalize to other distillation scenarios**: e.g., SDS variants in 2D editing and video generation — any iterative optimization setting using SDS could benefit from analogous source anchoring.

## Limitations & Future Work
- **Still reliant on the SDS paradigm**: inherits SDS's inherently slow optimization (requiring thousands of steps), making it unsuitable for real-time applications.
- **Upper-bounded by image conditioner capability**: if IP-Adapter/ControlNet exhibits poor inversion capability for certain rendering styles, source estimation quality degrades (which motivates the Filter strategy).
- **No comparison with feed-forward 3D generation methods**: the T3Bench evaluation only compares against SDS variants, lacking comparison with 3D-native generative models.

## Related Work & Insights
- **vs. VSD (ProlificDreamer)**: VSD trains a LoRA-based particle distribution model to approximate the source distribution, incurring high computational overhead (4 models). AnchorDS directly conditions on the rendered image with zero additional models and achieves superior performance.
- **vs. SDS-Bridge**: SDS-Bridge corrects source bias via hand-crafted negative prompts describing the 3D state, introducing new biases. AnchorDS lets the model directly "see" the current rendering, introducing no additional bias.
- **vs. DDS**: DDS also uses a reference image but requires a paired reference prompt and serves a different purpose. AnchorDS acquires its image condition automatically (the current rendering).

## Rating
- Novelty: ⭐⭐⭐⭐ In-depth analysis of the source distribution problem in SDS; method is elegant and concise.
- Experimental Thoroughness: ⭐⭐⭐⭐ T3Bench + human evaluation + ablation + multi-baseline comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Analysis and derivations are clear, figures are intuitive, and the logical chain is complete.
- Value: ⭐⭐⭐⭐ A significant improvement within the SDS framework; the method is simple and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](../../CVPR2026/3d_vision/text-image_conditioned_3d_generation.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[ICML 2026\] RelaxFlow: Text-Driven Amodal 3D Generation](../../ICML2026/3d_vision/relaxflow_text-driven_amodal_3d_generation.md)
- [\[NeurIPS 2025\] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](../../NeurIPS2025/3d_vision/walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)
- [\[AAAI 2026\] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)

</div>

<!-- RELATED:END -->
