---
title: >-
  [Paper Note] MARBLE: Material Recomposition and Blending in CLIP-Space
description: >-
  [CVPR 2025][Image Generation][material editing] Operates material embeddings solely in CLIP space, achieving material transfer and blending via targeted injection into material-responsive layers of UNet, and achieves parametric control of roughness, metallic, transparency, and glow by predicting attribute editing directions with a lightweight MLP, without fine-tuning the diffusion model.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "material editing"
  - "CLIP-space"
  - "material blending"
  - "parametric control"
  - "diffusion model"
date: 2026-05-08
content_hash: 226f9bfeee73688c
---

# MARBLE: Material Recomposition and Blending in CLIP-Space

**Conference**: CVPR 2025  
**arXiv**: [2506.05313](https://arxiv.org/abs/2506.05313)  
**Code**: [Project Page](https://marblecontrol.github.io/)  
**Area**: Image Generation  
**Keywords**: material editing, CLIP-space, material blending, parametric control, diffusion model

## TL;DR

Operates material embeddings solely in CLIP space, achieving material transfer and blending via targeted injection into material-responsive layers of UNet, and achieves parametric control of roughness, metallic, transparency, and glow by predicting attribute editing directions with a lightweight MLP, without fine-tuning the diffusion model.

## Background & Motivation

**Background**: Material editing is a core problem in computer graphics. Traditional methods require explicit estimation of geometry, lighting, and material properties. Recent diffusion model methods are divided into: ZeST (zero-shot material transfer but with coarse granularity) and Alchemist (fine-tunes SD for fine-grained control but risks breaking priors).

**Limitations of Prior Work**:
1. **Insufficient Granularity in ZeST**: Only supports high-level material transfer, unable to perform fine-grained attribute control such as roughness.
2. **Overfitting Risk of Alchemist**: Fine-tuning the entire diffusion model on synthetic data may destroy the object priors embedded in the model.
3. **Geometry Leakage**: ZeST injects CLIP features into all UNet blocks, causing non-material information (such as identity features) to also be transferred, leading to geometric deformation.

**Key Challenge**: Need a unified framework that can achieve both coarse-grained material transfer and fine-grained attribute tuning, while keeping the base diffusion model intact.

**Key Insight**: Keep the SD model frozen and operate material embeddings only in CLIP space—finding the specific layers in UNet responsible for material attribution to perform targeted injection, and learning attribute editing directions within CLIP space.

## Method

### Overall Architecture

Improved based on the ZeST architecture, using the SDXL inpainting model $\mathcal{S}$:
1. Input: Target image $I$ (with foreground mask $F_I$ and depth map $D_I$) + material reference image $I_m$
2. Encode $I_m$ with CLIP to obtain the material embedding $z_m$
3. Inject $z_m$ via IP-Adapter only into the "material layers" of the UNet (specific blocks near the bottleneck layer)
4. Generate output: $I_{gen} = \mathcal{S}(I_{init}, F_I, D_I, f(z_m))$

### Key Designs

**1. Targeted Material Layer Injection (Targeted Material Block Injection)**
- **Function**: Through exhaustive visualization experiments, the attention blocks (near the bottleneck layer) specifically responsible for material attribution in UNet are identified, and the CLIP material embedding is injected only into these layers instead of all layers.
- **Mechanism**: Inspired by InstantStyle, injecting layer-by-layer and observing the output revealed that material attribution and style attribution are managed by the same block. Injecting only this block preserves object geometry and lighting much better.
- **Design Motivation**: Injecting into all layers leaks non-material information (such as object identity semantics) from CLIP features, causing geometric deformation. Targeted injection achieves decoupling of material information from geometry and lighting.

**2. CLIP-Space Material Blending (Material Blending)**
- **Function**: Given CLIP embeddings $z_{m_1}$ and $z_{m_2}$ of two material reference images, linear interpolation is performed to generate a blended material: $I_{gen} = \mathcal{S}(I_{init}, F_I, D_I, f(\alpha z_{m_1} + (1-\alpha) z_{m_2}))$.
- **Mechanism**: Leveraging the interpretable directionality in CLIP space (similar to the interpretability of GAN latent spaces), interpolating embeddings produces semantically continuous blended materials. Three configurations are all effective: (1) different objects with different materials, (2) the same material with different attribute values, (3) the same object and material with different attributes.
- **Design Motivation**: The linear structure of CLIP space naturally supports semantic interpolation without requiring extra training.

**3. Parametric Control via MLP**
- **Function**: For each attribute (roughness, metallic, transparency, and glow), a 2-layer MLP $p_\theta$ is trained to predict the editing direction in CLIP space using the material reference image and editing intensity $\delta$. During inference: $z_{m_{a+\delta}} = \text{CLIP}(I_m) + p_\theta(I_m, \delta)$.
- **Mechanism**: 300 synthetic objects are rendered using Blender (250 for training / 50 for validation), other attributes held constant, with each object rendered under different attribute values. The differences in CLIP features are calculated and denoised via Singular Value Decomposition (SVD) low-rank approximation, then the MLP is trained to predict these low-rank directions. Training objective: $\arg\min_\theta [\text{cossim}(s_{m_{a+\delta}}, p_\theta) + \text{MSE}(s_{m_{a+\delta}}, p_\theta)]$.
- **Design Motivation**: (1) Retain the original priors by avoiding fine-tuning of the diffusion model; (2) Solve CLIP feature noise issues with SVD denoising; (3) Train MLPs for each attribute independently to enable simultaneous editing of multiple attributes when combined.

### Loss & Training

- **MLP Training**: Joint optimization with cosine similarity loss + MSE loss
- **Minimal Data**: Requires only 250 synthetic objects for training, and can yield decent results with as few as 16 objects
- **SVD Low-rank Approximation**: SVD is performed on the stacked editing direction matrices, keeping the principal components that explain 67%-80% of the variance
- **No Fine-tuning of SD Model**: All attribute controllers are trained independently and can be combined arbitrarily

## Key Experimental Results

### Main Results

| Method | PSNR↑ | LPIPS↓ | CLIP↑ | DreamSim↓ |
|---|---|---|---|---|
| **Roughness** | | | | |
| Concept Slider (Images) | 18.87 | 0.356 | 0.597 | 0.567 |
| **MARBLE** | **26.56** | **0.056** | **0.931** | **0.129** |
| **Metallic** | | | | |
| Concept Slider (Images) | 19.45 | 0.317 | 0.655 | 0.479 |
| **MARBLE** | **26.82** | **0.053** | **0.928** | **0.121** |
| **Transparency** | | | | |
| Concept Slider (Images) | 19.85 | 0.346 | 0.639 | 0.525 |
| **MARBLE** | **26.99** | **0.070** | **0.905** | **0.163** |
| **Glow** | | | | |
| Concept Slider (Images) | 16.92 | 0.301 | 0.661 | 0.509 |
| **MARBLE** | **19.73** | **0.111** | **0.890** | **0.213** |

### User Study

87.5% of the participants (14 out of 16) preferred the results of MARBLE over Image Concept Slider.

### Data Efficiency Ablation

| Number of Training Objects | PSNR↑ | DreamSim↓ |
|---|---|---|
| 8 | ~25 | ~0.18 |
| 16 | ~26 | ~0.15 |
| 250 (Full) | 26.99 | 0.163 |

Only 16 objects are needed to achieve performance close to the full dataset.

### Key Findings

1. **Targeted vs. Full Injection**: Injecting only into the material layers significantly improves geometric preservation—full injection can lead to geometric deformation of the object (e.g., "hands" growing on a toy).
2. **Attribute Decoupling**: Attributes like roughness and metallic can be controlled independently without interfering with each other, supporting multi-attribute grid editing.
3. **Style Generalization**: Since the SD model weights are not modified, the learned editing directions can generalize to different styles such as anime and oil paintings.
4. **Baseline Failure Modes**: InstructPix2Pix often causes geometric and color changes; Concept Slider requires DDIM inversion, causing inaccurate reconstruction and failing to capture concepts like transparency and glow.

## Highlights & Insights

- **Minimally Invasive Design**: Operates only in CLIP space + target-injects one layer, avoiding modification of any pre-trained weights to maximally preserve the prior knowledge of the diffusion model
- **Unified Multi-functional Framework**: Material transfer, blending, and fine-grained attribute control are completed within the same framework
- **High Data Efficiency**: A usable attribute controller can be trained with just 16 synthetic objects, which is highly friendly for actual deployment
- **Material Interpretability of CLIP Space**: Unveils that CLIP embeddings contain decouplable material attribute directions, a finding that has its own independent value

## Limitations & Future Work

- Parametric control sometimes changes object texture patterns (e.g., the pattern of a leather backpack changing with roughness)
- May produce artifacts on objects that already meet the attribute conditions (e.g., increasing transparency on glass)
- Relies on the encoder-decoder process of SDXL, leading to high-frequency detail loss
- Supports only 4 attributes (roughness/metallic/transparency/glow), and does not cover more PBR parameters

## Related Work & Insights

- **ZeST**: A zero-shot material transfer framework on which MARBLE improves through targeted injection
- **InstantStyle**: Methodology discovering style attribution layers in UNet; MARBLE confirms that material attribution and style attribution share the same layer
- **Concept Slider**: A LoRA adapter method for continuous concept control, which requires DDIM inversion and struggles to capture material concepts well
- **Alchemist**: Pioneering work fine-tuning SD for fine-grained material control, which modifies SD weights

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐ A new paradigm for material editing in CLIP space, with an ingenious combination of targeted injection + SVD denoising  
**Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive quantitative, qualitative, user studies, and data efficiency analysis  
**Writing Quality**: ⭐⭐⭐⭐ Expressive and intuitive diagrams, clear description of methods  
**Practical Value**: ⭐⭐⭐⭐⭐ Extremely low training data requirements + plug-and-play design, high practical application value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[CVPR 2025\] LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending](latexblend_scaling_multi-concept_customized_generation_with_latent_textual_blend.md)
- [\[CVPR 2025\] Latent Space Imaging](latent_space_imaging.md)
- [\[NeurIPS 2025\] LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation](../../NeurIPS2025/image_generation/llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)
- [\[CVPR 2025\] CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)

</div>

<!-- RELATED:END -->
