---
title: >-
  [Paper Note] Boosting Generative Image Modeling via Joint Image-Feature Synthesis
description: >-
  [NeurIPS 2025 **Spotlight**][Image Generation][Joint Image-Feature Generation] This paper proposes ReDi (Representation Diffusion), a framework that jointly models VAE image latents and DINOv2 semantic features within a…
tags:
  - "NeurIPS 2025 **Spotlight**"
  - "Image Generation"
  - "Joint Image-Feature Generation"
  - "Diffusion Model"
  - "DINOv2"
  - "Representation Guidance"
  - "DiT"
date: 2026-05-08
content_hash: fe32742a9d1be61b
---

# Boosting Generative Image Modeling via Joint Image-Feature Synthesis

**Conference**: NeurIPS 2025 **Spotlight**
**arXiv**: [2504.16064](https://arxiv.org/abs/2504.16064)  
**Code**: [GitHub](https://representationdiffusion.github.io/)  
**Area**: Image Generation
**Keywords**: Joint Image-Feature Generation, Diffusion Model, DINOv2, Representation Guidance, DiT

## TL;DR

This paper proposes ReDi (Representation Diffusion), a framework that jointly models VAE image latents and DINOv2 semantic features within a diffusion model — both are simultaneously denoised from pure noise within a single diffusion process. With minimal modifications to the DiT architecture, ReDi achieves a 23× training convergence speedup and state-of-the-art FID, while unlocking a novel Representation Guidance inference strategy.

## Background & Motivation

**Background**: Latent diffusion models (LDMs) are the dominant approach for high-quality image generation, while self-supervised representation learning methods (e.g., DINOv2) excel at semantic understanding. Although each paradigm has its strengths, they have long remained separate — LDM internal features lack semantic grounding, while DINOv2 has no generative capability.

**Limitations of Prior Work**: REPA (Yu et al., 2025) first demonstrated that aligning the internal representations of a diffusion model with DINOv2 can simultaneously improve generation quality and training efficiency. However, REPA requires additional distillation losses (contrastive/MSE) to align intermediate features, resulting in a complex training objective.

**Key Challenge**: How can one elegantly achieve both high-quality image generation and semantic representation learning within a single model, without introducing complex distillation mechanisms?

**Goal**: To propose a more direct alternative to REPA — rather than aligning representations, the diffusion model is trained to jointly generate both images and semantic features.

**Key Insight**: DINOv2 semantic features are treated as a "second modality" on par with VAE latents, jointly denoised within the same diffusion process.

**Core Idea**: Instead of indirectly aligning the internal representations of a diffusion model, the model is trained to directly learn to generate semantic features — joint modeling naturally forces the model to integrate low-level visual and high-level semantic information during generation.

## Method

### Overall Architecture

Given an input image $I$, both the VAE latent $\mathbf{x}_0 = \mathcal{E}_x(I) \in \mathbb{R}^{L \times C_x}$ and the DINOv2 feature $\mathbf{z}_0 = \mathcal{E}_z(I) \in \mathbb{R}^{L \times C_z}$ are extracted simultaneously. The same forward diffusion noise schedule is applied to both, and a single Transformer is used for joint denoising. The training objective is a simple joint denoising loss; at inference time, both image and semantic features are generated simultaneously from pure noise.

### Key Designs

1. **Joint Forward-Reverse Diffusion Process**:
    - **Function**: Independent noise is added to both the image latent and semantic features using the same noise schedule, followed by joint denoising.
    - **Mechanism**: The forward process is defined as $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}_x$ and $\mathbf{z}_t = \sqrt{\bar\alpha_t}\mathbf{z}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}_z$. The model simultaneously predicts two sets of noise: $\boldsymbol{\epsilon}_\theta^x$ and $\boldsymbol{\epsilon}_\theta^z$. The joint loss is $\mathcal{L} = \|\boldsymbol{\epsilon}_\theta^x - \boldsymbol{\epsilon}_x\|^2 + \lambda_z\|\boldsymbol{\epsilon}_\theta^z - \boldsymbol{\epsilon}_z\|^2$, with default $\lambda_z=1$.
    - **Design Motivation**: Forcing the model to learn the joint distribution of image details and semantic structure, where each modality provides complementary information to the other, naturally leads to improved generation quality.

2. **Token Fusion Strategy (Merged vs. Separate)**:
    - **Function**: Two approaches for feeding VAE tokens and semantic tokens into the Transformer.
    - **Mechanism**: The **Merged** approach projects both token sets via separate linear projections and adds them channel-wise — $\mathbf{h}_t = \mathbf{x}_t\mathbf{W}_{emb}^x + \mathbf{z}_t\mathbf{W}_{emb}^z$ — preserving the original token count $L$. The **Separate** approach concatenates both sets along the sequence dimension, yielding $2L$ tokens. Merged is used by default to maintain computational efficiency.
    - **Design Motivation**: Merged enables tight early-stage interaction between the two information streams without increasing computational cost; Separate offers greater expressive capacity at the cost of doubled computation.

3. **PCA-Reduced Semantic Representations + Representation Guidance**:
    - **Function**: DINOv2's 768-dimensional features are reduced to 8 dimensions via PCA to balance computation; during inference, the semantic branch is used to guide image generation.
    - **Mechanism**: PCA dimensionality reduction addresses the capacity imbalance caused by $C_z \gg C_x$. Representation Guidance, analogous to CFG, is formulated as $\hat{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + w_r(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{z}_t, t) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t))$. During training, $\mathbf{z}_t$ is randomly dropped with probability $p_{drop}$ so the model learns denoising both with and without semantic conditioning.
    - **Design Motivation**: PCA prevents high-dimensional semantic features from consuming disproportionate model capacity. Representation Guidance leverages the model's own learned semantics to guide generation without requiring any additional model.

### Loss & Training

The joint denoising loss is $\mathcal{L}_{joint} = \|\boldsymbol{\epsilon}_\theta^x - \boldsymbol{\epsilon}_x\|^2 + \lambda_z\|\boldsymbol{\epsilon}_\theta^z - \boldsymbol{\epsilon}_z\|^2$ ($\lambda_z=1$). During training, $\mathbf{z}_t$ is zeroed out and the semantic loss is disabled with probability $p_{drop}=0.2$. Images are encoded using SD-VAE-FT-EMA ($32\times32\times4$), and semantics are extracted via DINOv2-B+Registers. The PCA projection matrix is precomputed on 76,800 randomly sampled ImageNet images.

## Key Experimental Results

### Main Results

| Model | Method | Iterations | FID↓ | Notes |
|-------|--------|-----------|------|-------|
| DiT-XL/2 | Baseline | 7M | 9.6 | Original DiT convergence |
| DiT-XL/2 | REPA | 400K | 12.3 | Distillation alignment |
| DiT-XL/2 | **ReDi** | **400K** | **8.7** | Joint modeling, surpasses 7M-step baseline |
| SiT-XL/2 | Baseline | 7M | 8.3 | Original SiT convergence |
| SiT-XL/2 | REPA | 4M | 5.9 | Requires 10× iterations |
| SiT-XL/2 | **ReDi** | **700K** | **5.6** | 6× faster convergence |
| SiT-XL/2+CFG | **ReDi** | 350 epochs | **1.72** | SOTA unconditional diffusion |
| SiT-XL/2+CFG | **ReDi** | 800 epochs | **1.61** | Current best |

### Ablation Study

| Configuration | FID↓ | Notes |
|--------------|------|-------|
| Merged tokens (default) | 8.7 | Efficient and effective |
| Separate tokens | 8.2 | Stronger but doubles computation |
| No PCA (768-dim) | Degraded | Capacity imbalance |
| PCA to 8-dim (default) | 8.7 | Optimal balance |
| $\lambda_z=0$ (no semantic loss) | ~DiT baseline | Semantic branch is necessary |
| ReDi + REPA | 3.3 (4M iters) | Two methods are complementary |

### Key Findings
- ReDi accelerates convergence of DiT-XL/2 and SiT-XL/2 by approximately **23×**.
- Compared to REPA, ReDi converges **6× faster** with better FID.
- ReDi and REPA are complementary — combined, they reach FID 3.6 at 1M steps (REPA alone requires 4M steps to reach 5.9).
- Representation Guidance improves generation quality without relying on any external classifier.

## Highlights & Insights
- An elegantly designed Spotlight contribution — substantial gains with minimal architectural modification.
- The paradigm debate of joint modeling vs. distillation alignment: directly modeling the joint distribution proves more effective than indirect alignment.
- Representation Guidance is a fully self-contained inference strategy, complementary to CFG.
- The finding that the two methods are complementary suggests that joint modeling and alignment capture different aspects of information.

## Limitations & Future Work
- PCA dimensionality reduction is linear and may discard nonlinear semantic structure.
- The semantic encoder (DINOv2) is frozen; joint end-to-end fine-tuning may yield further improvements.
- Validation is limited to ImageNet 256×256; higher resolutions and text-to-image settings remain unexplored.
- The Separate tokens approach doubles computation, motivating the need for more efficient attention mechanisms.

## Related Work & Insights
- **vs. REPA**: REPA aligns intermediate features via distillation, requiring additional losses and yielding weaker results; ReDi directly models the joint distribution in a simpler and more effective manner.
- **vs. MT-Diffusion**: MT-Diffusion also introduces CLIP representations but does not quantify their impact on generation; ReDi systematically evaluates the benefits of joint modeling.
- **vs. VideoJam**: Inspires the Representation Guidance design — analogous to the motion guidance strategy employed in VideoJam.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Joint image-feature diffusion is a genuinely new paradigm; Representation Guidance is an original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-scale models, multiple frameworks, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Method is clearly described with well-formalized mathematical notation.
- **Value**: ⭐⭐⭐⭐⭐ Spotlight recognition is well-deserved; opens a new direction for representation-aware generative modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](../../ICCV2025/image_generation/polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[NeurIPS 2025\] OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data](osmgen_highly_controllable_satellite_image_synthesis_using_openstreetmap_data.md)
- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)
- [\[NeurIPS 2025\] Conditional Panoramic Image Generation via Masked Autoregressive Modeling](conditional_panoramic_image_generation_via_masked_autoregres.md)

</div>

<!-- RELATED:END -->
