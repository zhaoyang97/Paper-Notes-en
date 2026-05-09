---
title: >-
  [Paper Note] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?
description: >-
  [Image Generation] Through systematic analysis of the behavior of $W_{\{q,k,v,o\}}$ components during LoRA fine-tuning, this work reveals that $W_v$ and $W_o$ are responsible for learning panoramic spherical structure while $W_q$ and $W_k$ retain perspective-domain shared knowledge. Based on this finding, the paper proposes UniPano, an efficient single-branch panorama generation framework.
tags:
  - Image Generation
date: 2026-05-08
content_hash: 306ce6eccd26b4cc
---

# What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2505.22129](https://arxiv.org/abs/2505.22129)
- **Code**: [GitHub](https://github.com/jinhong-ni/UniPano)
- **Area**: Diffusion Models · Panorama Generation
- **Keywords**: Panorama Generation, LoRA Fine-tuning, Attention Mechanism Analysis, Stable Diffusion, Spherical Distortion

## TL;DR
Through systematic analysis of the behavior of $W_{\{q,k,v,o\}}$ components during LoRA fine-tuning, this work reveals that $W_v$ and $W_o$ are responsible for learning panoramic spherical structure while $W_q$ and $W_k$ retain perspective-domain shared knowledge. Based on this finding, the paper proposes UniPano, an efficient single-branch panorama generation framework.

## Background & Motivation

Panoramic images (360°) exhibit spherical distortion due to equirectangular projection and a 2:1 aspect ratio, fundamentally differing from standard perspective images. Given the scarcity of panoramic data (e.g., Matterport3D contains only 10,800 images), training generative models from scratch is infeasible.

Existing approaches fall into two categories:

**Multi-view stitching methods**: Generate multiple perspective images and stitch them into a panorama, but the pipeline is complex.

**End-to-end fine-tuning methods**: e.g., PanFusion uses a dual-branch framework but incurs enormous memory and training overhead (60.12 GB peak GPU memory).

An intriguing observation is that simple LoRA fine-tuning of pretrained diffusion models can already produce reasonable panoramas. **What is the underlying mechanism?** This serves as the core motivation: which components of a pretrained perspective diffusion model play a critical role in panoramic adaptation?

## Method

### Overall Architecture

The analysis is conducted under the LoRA fine-tuning paradigm, systematically investigating the roles of four trainable components $W_{\{q,k,v,o\}}$ in the cross-attention module. Based on the findings, the paper proposes UniPano: an efficient single-branch panorama generation framework.

### Key Design 1: Isolated Training Experiments

Each of $W_q$, $W_k$, $W_v$, $W_o$ is trained independently with a LoRA adapter while the remaining components are frozen:

- **$W_q$ and $W_k$ trained in isolation fail**: They cannot capture the spherical distortion structure.
- **$W_v$ and $W_o$ trained in isolation succeed**: They can learn the equirectangular projection characteristics of panoramas.

Quantitative results show that $W_v$ and $W_o$ achieve significantly better FAED and FID scores than $W_q$ and $W_k$.

### Key Design 2: Decomposition Experiments after Joint Training

$W_{\{q,k,v,o\}}$ LoRA adapters are jointly trained, and selected adapters are selectively disabled at inference time:

- Disabling $W_v$ and $W_o$ LoRA → the model reverts to generating perspective images.
- Disabling $W_q$ and $W_k$ LoRA → the model still generates high-quality panoramic images.

**Conclusion**: $W_q$ and $W_k$ encode cross-domain shared perspective knowledge (without spherical distortion information), whereas $W_v$ and $W_o$ are dedicated to adapting perspective knowledge to the spherical structure of the panoramic domain.

### UniPano Design

Based on the above insights, UniPano incorporates the following key designs:

1. **Freeze $W_q$ and $W_k$**: As they carry no panorama-specific information.
2. **Enhance the expressive capacity of $W_o$**: Replace the LoRA on $W_o$ with a Mixture of Experts (MoE) module.

$$\text{MoE}(x) = \sum_{i=1}^{E} g_i(x) \cdot \text{LoRA}_i(x)$$

where $g_i$ denotes the weights learned by the routing network, and each expert is an independent LoRA module.

### Loss & Training

Training employs the standard diffusion denoising objective:

$$\mathcal{L} = \mathbb{E}_{z_t, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

An auxiliary routing load-balancing loss is additionally incorporated to stabilize MoE training.

## Experiments

### Main Results: Comparison with SOTA Methods (512×1024 Panorama Generation)

| Method | Peak Memory (GB) | Training Time (hrs) | FAED↓ | FID↓ (Pano) | FID↓ (20-view) | FID↓ (8-view) |
|--------|-----------------|---------------------|-------|-------------|----------------|---------------|
| SD+LoRA | 31.69 | 2.26 | 7.19 | 51.69 | 19.32 | 20.68 |
| PanFusion | 60.12 | 6.61 | 6.04 | 46.47 | 17.04 | 19.88 |
| **UniPano** | **32.59** | **3.43** | **5.90** | **46.47** | **17.09** | **17.74** |

UniPano achieves the best FAED and 8-view FID scores, while consuming only 54% of PanFusion's GPU memory and 52% of its training time.

### Ablation Study: Comparison of $W_o$ Enhancement Strategies

| Strategy | FAED↓ | FID↓ (Pano) | FID↓ (20-view) | FID↓ (8-view) |
|----------|-------|-------------|----------------|---------------|
| Pano Only baseline | 7.90 | 50.40 | 20.10 | 20.56 |
| LoRA (r=8) | 8.34 | 48.58 | 16.94 | 19.42 |
| Deformable Attn | 9.78 | 46.41 | 16.56 | 18.77 |
| Local Window Attn | 7.65 | 50.39 | 18.24 | 19.41 |
| **MoE** | **7.21** | **48.83** | 19.50 | 20.05 |

MoE achieves the best FAED score, demonstrating its superior effectiveness in capturing panoramic spherical structure.

### Key Findings

1. Query and Key in the attention mechanism encode cross-domain general knowledge, whereas Value and Output are the key components for domain adaptation.
2. This finding provides generalizable guidance for domain adaptation in other settings: enhancing the capacity of $W_v$ and $W_o$ is more efficient than full LoRA fine-tuning.
3. UniPano scales to higher resolutions (end-to-end generation at 1024×2048).

## Highlights & Insights

1. **Mechanistic discovery**: This is the first work to systematically explain how a pretrained perspective diffusion model adapts to the panoramic domain via LoRA fine-tuning.
2. **Clear design principle**: The dichotomy — $W_q$/$W_k$ retaining general knowledge and $W_v$/$W_o$ learning domain-specific knowledge — is both concise and compelling.
3. **Significant efficiency advantage**: Only a 2.8% increase in GPU memory overhead, while matching or surpassing the dual-branch approach that doubles memory consumption.

## Limitations & Future Work

- The choice of MoE is heuristic; the paper acknowledges it may not be the optimal design.
- The analysis is limited to LoRA fine-tuning in cross-attention modules and does not extend to self-attention.
- Evaluation is conducted solely on the Matterport3D dataset, limiting scene diversity.
- Generated resolution still falls short of practical application requirements.

## Related Work & Insights

- **Diffusion Models**: Stable Diffusion, DDPM, LoRA fine-tuning
- **Panorama Generation**: PanFusion, StitchDiffusion, CubeDiff, PanoFree
- **LoRA Analysis**: This paper is the first to systematically analyze the role of individual LoRA components in domain adaptation.

## Rating
- Novelty: ★★★★☆ — Novel perspective through mechanistic analysis
- Technical Depth: ★★★★☆ — Systematic experimental design with convincing conclusions
- Practicality: ★★★★☆ — Concise and efficient framework, easy to generalize

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

<!-- RELATED:END -->
