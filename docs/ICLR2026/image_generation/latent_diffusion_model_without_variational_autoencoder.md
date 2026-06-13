---
title: >-
  [Paper Note] Latent Diffusion Model without Variational Autoencoder
description: >-
  [ICLR 2026][Image Generation][Self-supervised representation] This paper proposes SVG, which replaces the VAE latent space with frozen DINOv3 self-supervised features for diffusion model training. A lightweight residual…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Self-supervised representation"
  - "DINOv3"
  - "VAE-free latent diffusion"
  - "unified feature space"
  - "few-step generation"
date: 2026-05-08
content_hash: ab395bb67b83bc5d
---

# Latent Diffusion Model without Variational Autoencoder

**Conference**: ICLR 2026
**arXiv**: [2510.15301](https://arxiv.org/abs/2510.15301)  
**Code**: [GitHub](https://github.com/shiml20/SVG)  
**Area**: Diffusion Models / Visual Representation
**Keywords**: Self-supervised representation, DINOv3, VAE-free latent diffusion, unified feature space, few-step generation

## TL;DR
This paper proposes SVG, which replaces the VAE latent space with frozen DINOv3 self-supervised features for diffusion model training. A lightweight residual encoder supplements fine-grained details, enabling faster training, more efficient inference, and a unified visual representation applicable across tasks.

## Background & Motivation
- **Background**: The VAE + Diffusion paradigm suffers from three fundamental limitations: training/inference inefficiency, poor quality under few-step sampling, and limited semantic discriminability of VAE features.
- **Limitations of Prior Work**: VAE latent spaces exhibit severe semantic entanglement (t-SNE visualizations show heavy class overlap), causing contradictory velocity field directions and necessitating more sampling steps. Existing acceleration methods (REPA, VA-VAE) improve training by aligning with VFM features but only address symptoms without fundamentally restructuring the latent space.
- **Key Challenge**: The lack of semantic discriminability in VAE latent spaces is the fundamental cause of slow convergence and inefficient sampling.
- **Goal**: To demonstrate that a semantically well-structured latent space can substantially accelerate diffusion training and reduce the required number of sampling steps.

## Method

### Overall Architecture
SVG consists of three components: a frozen DINOv3 encoder and a lightweight residual encoder jointly produce the SVG feature space, which is then passed to an SVG Decoder for image reconstruction. The diffusion model is trained directly within the SVG feature space.

### Key Designs

1. **SVG Autoencoder**: The frozen DINOv3-ViT-S/16+ encoder produces a $16 \times 16 \times 384$ feature map (for 256×256 images). A residual encoder (ViT) captures fine-grained details absent from DINO features; its output is concatenated with DINO features to form the complete SVG representation. The residual distribution is aligned to the DINO feature distribution via batch statistics:
$$\hat{F}_R = \frac{F_R - \mu(F_R)}{\sigma(F_R)} \cdot \sigma(F_D) + \mu(F_D)$$

2. **SVG Diffusion**: Unlike the $16 \times 16 \times 4$ latent space of VAE, SVG trains the diffusion model in the $16 \times 16 \times 384$ high-dimensional feature space. Although high-dimensional training is typically unstable, the strong semantic dispersion of SVG features enables stable and efficient optimization. Training follows the flow matching objective under the SiT configuration.

3. **Semantic Dispersion Analysis**: Via t-SNE visualizations and a toy example, the paper demonstrates that a semantically well-separated feature space yields consistent intra-class velocity directions and clearly distinct inter-class directions, simplifying optimization and reducing the required sampling steps.

### Loss & Training
Two-stage training:
- **Stage 1**: Train only the residual encoder and SVG Decoder (reconstruction loss + distribution alignment), with DINOv3 frozen.
- **Stage 2**: Train SVG Diffusion (SiT configuration, QK-Norm, per-channel normalization).

## Key Experimental Results

### Main Results (ImageNet 256×256)

| Method | Tokenizer | Training Epochs | Steps | gFID w/o CFG | gFID w/ CFG |
|--------|-----------|-----------------|-------|--------------|-------------|
| DiT-XL | SD-VAE | 1400 | 250 | 9.62 | 2.27 |
| SiT-XL | SD-VAE | 1400 | 250 | 9.35 | 2.15 |
| REPA-XL | SD-VAE | 800 | 250 | 5.90 | 1.42 |
| SiT-XL (SD-VAE) | SD-VAE | 80 | **25** | 22.58 | 6.06 |
| SiT-XL (VA-VAE) | VA-VAE | 80 | **25** | 7.29 | 4.13 |
| **SVG-XL** | **SVGTok** | **80** | **25** | **6.57** | **3.54** |
| **SVG-XL** | **SVGTok** | **500** | **25** | **3.94** | **2.10** |

### Few-Step Generation Comparison

| Method | Steps | FID w/o CFG | FID w/ CFG |
|--------|-------|-------------|------------|
| SiT-XL (SD-VAE) | 5 | 69.38 | 29.48 |
| SiT-XL (VA-VAE) | 5 | 74.46 | 35.94 |
| **SVG-XL** | **5** | **12.26** | **9.03** |
| SiT-XL (SD-VAE) | 10 | 32.81 | 10.26 |
| **SVG-XL** | **10** | **9.39** | **6.49** |

### Key Findings
- SVG-XL at 25 steps (80 epochs) achieves FID=6.57, substantially outperforming SiT-XL at the same budget (FID=22.58).
- Only 5 sampling steps are required to reach FID=12.26, a level that SiT requires approximately 250 steps to match.
- The SVG feature space preserves the semantic discriminability of DINOv3 (linear probing accuracy close to original DINO).
- The residual encoder is critical for reconstructing color fidelity and high-frequency details.
- Among all VFMs evaluated, DINOv3 is best suited as the foundation for a unified feature space.

## Highlights & Insights
- This work is the first to demonstrate that self-supervised features can be directly used for generative modeling, breaking the convention that VAE is the only viable latent space for latent diffusion.
- The causal analysis linking semantic dispersion to training efficiency is insightful, and the toy example provides an intuitive illustration.
- The approach enables a unified feature space applicable to generation, perception, and understanding tasks.
- The exceptional performance under 5-step generation highlights the dimensionality-reduction effect of a semantically structured latent space.

## Limitations & Future Work
- Validation is currently limited to ImageNet 256×256; the approach has not been extended to text-guided generation or high-resolution synthesis.
- The SVG feature dimensionality is substantially higher (384 vs. 4 for VAE), incurring greater memory overhead.
- The method relies on a specific DINOv3 model; alternative self-supervised approaches (e.g., MAE, SigLIP) yield inferior results.
- Reconstruction quality (rFID=0.65) is slightly below that of the best VAE baselines.

## Related Work & Insights
- Alignment-based methods such as REPA and VA-VAE motivated this work; however, SVG more fundamentally replaces the feature space rather than merely aligning to it.
- SVG is complementary to autoregressive methods such as MAR, providing a superior latent space for continuous diffusion models.
- This work suggests that future visual generative models may no longer require a dedicated VAE.

## Technical Details
- DINOv3-ViT-S/16+ produces $16 \times 16 \times 384$ features (vs. $16 \times 16 \times 4$ for SD-VAE).
- The residual encoder adopts a ViT architecture (implemented via the timm library) and its output is channel-concatenated with DINOv3 features.
- The SVG Decoder follows the decoder architecture design of VA-VAE.
- Per-channel normalization is applied to the SVG feature space to stabilize high-dimensional diffusion training.
- The patch embedding layer in DiT is replaced with a simple linear projection (384 → model dimension).
- The hidden state channel count is typically larger than 384 (e.g., 1152 in DiT-XL), so SVG does not incur inference inefficiency.
- Linear probing accuracy: DINOv3 original 86.4%, SVG (frozen DINO component) 85.2%, demonstrating that semantic capability is largely preserved.
- MAE and SigLIP encoders lack sufficient reconstruction capacity to support high-quality generation.
- SVG-XL at 1400 epochs and 25 steps achieves FID=3.36 (w/o CFG) / 1.92 (w/ CFG), approaching state-of-the-art performance.
- The approach scales effectively across model sizes from SVG-B (130M) to SVG-XL (675M).
- SVG features are validated for perception and understanding tasks via diagnostic probing experiments.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First work to eliminate VAE and use self-supervised features directly for diffusion; the idea is original and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablations are comprehensive, but large-scale and text-guided experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is thoroughly analyzed and visualizations are compelling.
- **Value**: ⭐⭐⭐⭐⭐ — Has the potential to reshape the design paradigm of latent diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] T-LoRA: Single Image Diffusion Model Customization Without Overfitting](../../AAAI2026/image_generation/t-lora_single_image_diffusion_model_customization_without_overfitting.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[ICLR 2026\] MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion](mvcustom_multi-view_customized_diffusion_via_geometric_latent_rendering_and_comp.md)
- [\[AAAI 2026\] Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model](../../AAAI2026/image_generation/structure-based_rna_design_by_step-wise_optimization_of_latent_diffusion_model.md)
- [\[CVPR 2026\] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](../../CVPR2026/image_generation/one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
