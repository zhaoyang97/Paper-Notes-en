---
title: >-
  [Paper Note] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation
description: >-
  [CVPR 2026][Image Restoration][Dataset Distillation] This paper proposes EVLF, a plug-and-play early vision-language fusion method operating at the encoder-backbone interface, addressing the problem of text dominance and degraded visual fidelity caused by late-stage semantic injection in diffusion-based dataset distillation.
tags:
  - CVPR 2026
  - Image Restoration
  - Dataset Distillation
  - Diffusion Models
  - Vision-Language Fusion
  - Early Fusion
  - Plug-and-Play
date: 2026-05-08
content_hash: f4ccf8c709913878
---

# EVLF: Early Vision-Language Fusion for Generative Dataset Distillation

**Conference**: CVPR 2026
**arXiv**: [2603.07476](https://arxiv.org/abs/2603.07476)
**Code**: [GitHub](https://github.com/wenqi-cai297/earlyfusion-for-dd/)
**Area**: Image Restoration
**Keywords**: Dataset Distillation, Diffusion Models, Vision-Language Fusion, Early Fusion, Plug-and-Play

## TL;DR

This paper proposes EVLF, a plug-and-play early vision-language fusion method operating at the encoder-backbone interface, addressing the problem of text dominance and degraded visual fidelity caused by late-stage semantic injection in diffusion-based dataset distillation.

## Background & Motivation

Dataset Distillation (DD) aims to synthesize compact training sets that enable models to achieve high accuracy with few samples. Diffusion-based DD methods (e.g., D4M, MGD3) have become mainstream, yet share a core structural problem:

**Semantic dominance from late fusion**: In standard diffusion pipelines, textual semantics are injected via cross-attention during the denoising stage (late fusion), causing text signals to excessively dominate the generation trajectory.

**Degraded visual fidelity**: Since encoder-derived visual latents carry only visual information, late-injected semantics operate in a "corrective" rather than "co-evolutionary" manner, producing samples that match class labels but suffer from visual distortion.

**Manifestations include**: unnatural shapes, text-like textures, and overly simplified contours in generated samples.

Core insight: Moving semantic fusion from the denoising stage to the encoder output stage (encoder-backbone interface) allows visual and semantic signals to co-evolve from the very beginning of the diffusion process.

## Method

### Overall Architecture

The EVLF pipeline:
1. The VAE encoder produces visual latents $z_{\text{img}} = \mathcal{E}(x)$
2. The text encoder produces class embeddings $e_{\text{text}} = \mathcal{T}(y)$
3. A **lightweight cross-attention module** fuses both at the encoder output: $z_{\text{fused}} = \text{CA}(z_{\text{img}}, e_{\text{text}})$
4. The fused $z_{\text{fused}}$ serves as the initial condition for subsequent diffusion generation
5. Optionally, the denoiser is fine-tuned to adapt to the fused latent distribution

### Key Designs

1. **Early-fusion cross-attention module**: Image tokens serve as Query and text tokens as Key/Value, ensuring semantics are injected with vision as the anchor rather than allowing text to dominate:
$$Q = \tilde{z}W_Q, \quad K = \tilde{e}W_K, \quad V = \tilde{e}W_V$$
$$z_{\text{fused}} = \psi(\text{LN}(\tilde{z} + \text{softmax}(\frac{QK^\top}{\sqrt{d}})V))$$
**Design Motivation**: Using image features as Query ensures semantics "guide without overwriting" visual structure.

2. **Dual-loss training objective**:

    - **Visual preservation loss** $\mathcal{L}_{\text{MSE}} = \|z_{\text{fused}} - z_{\text{img}}\|_2^2$: ensures the fused latent does not deviate from the original visual structure.
    - **Semantic alignment loss** $\mathcal{L}_{\text{InfoNCE}}$: maps $z_{\text{fused}}$ to the text embedding space via a learnable projector and performs in-batch contrastive learning to align same-class samples.
    - Total loss: $\mathcal{L}_{\text{CA}} = \lambda_1 \mathcal{L}_{\text{InfoNCE}} + \lambda_2 \mathcal{L}_{\text{MSE}}$

3. **Plug-and-play design**: EVLF inserts only at the encoder-backbone interface, independent of any specific training schedule or loss function, and can be seamlessly integrated into arbitrary encoder-based diffusion DD pipelines such as D4M and MGD3.

### Loss & Training

- Cross-attention module trained for 4 epochs, batch size 16, AdamW optimizer
- $\lambda_1 = 0.1$ (fixed); $\lambda_2$ linearly increases from 0.05 to 1.0 over the first 2 epochs
- Optional denoiser fine-tuning using standard diffusion loss on $z_{\text{fused}}$
- Denoiser fine-tuning required when integrating with D4M; frozen when integrating with MGD3
- Training feasible on a single NVIDIA A5000 GPU

## Key Experimental Results

### Main Results

| Dataset | IPC | Metric | D4M | D4M+EVLF | MGD3 | MGD3+EVLF |
|--------|-----|------|-----|----------|------|-----------|
| ImageWoof | 10 | ResNetAP-10 | 33.2 | 37.3 | 36.6 | **39.3** |
| ImageWoof | 50 | ResNetAP-10 | 51.7 | 55.8 | 55.6 | **59.0** |
| ImageNette | 20 | ResNetAP-10 | 66.3 | 71.7 | 69.2 | **72.5** |
| CIFAR-10 | 10 | Accuracy | 37.6 | **45.7** | - | - |
| Tiny-ImageNet | 10 | Accuracy | 42.5 | **49.2** | - | - |
| ImageNet-1K | 50 | Accuracy | 60.1 | 60.6 | 60.3 | **61.9** |

### Ablation Study

| Configuration | IPC=10 | IPC=20 | IPC=50 | Notes |
|------|--------|--------|--------|------|
| D4M baseline | 47.7 | 56.3 | 67.8 | ResNetAP-10 on ImageIDC |
| +Denoiser fine-tuning | 54.1 | 61.1 | 70.3 | Fine-tuning is effective |
| +Cross-attention | 51.1 | 57.5 | 69.1 | Cross-attention is effective |
| +Both combined | **57.3** | **62.0** | **72.1** | Complementary effect is optimal |

### Key Findings

- EVLF consistently improves performance across all IPC settings and datasets, with larger gains at low IPC (8.1% on CIFAR-10 at IPC=10)
- t-SNE visualizations show that EVLF-generated samples exhibit broader distribution coverage and improved diversity
- Enabling EVLF ($\lambda_1 > 0$) yields significant improvements in both accuracy and coverage, with results being insensitive to the specific value of $\lambda_1$
- Transfer learning experiments indicate that datasets distilled with EVLF exhibit superior feature transferability

## Highlights & Insights

1. **Precise diagnosis**: The paper accurately identifies late-stage semantic injection as the root cause of text over-correction in diffusion-based DD, and illustrates this with intuitive visualizations (Fig. 1).
2. **Elegant design**: A single lightweight cross-attention module achieves plug-and-play improvement without modifying any other part of the pipeline.
3. **Comprehensive experiments**: Evaluations span CIFAR-10/100, Tiny-ImageNet, ImageNet-1K and its subsets, across multiple IPC settings and architectures.
4. **Conceptual inspiration**: The comparison between early and late fusion reveals the importance of semantic injection timing in conditional generation.

## Limitations & Future Work

1. Currently supports only class-level conditioning; instance-level or multi-label scenarios are not addressed.
2. Gains on the large-scale ImageNet-1K setting are relatively modest (~0.5–1.6%).
3. More sophisticated fusion mechanisms (e.g., multi-layer fusion, adaptive fusion weights) remain unexplored.
4. Future directions include instance-aware and compositional prompting, as well as extension to finer-grained control.

## Related Work & Insights

- **D4M**: A prototype-driven LDM DD method; EVLF serves as a plug-and-play enhancement for it.
- **MGD3**: A multimodal-guided DD method; EVLF integrates seamlessly into it as well.
- **MinimaxDiffusion**: A DiT-based DD method using minimax optimization, focusing on discriminability and representativeness.
- Inspiration: The early-fusion concept of EVLF generalizes to optimizing semantic injection timing in other conditional generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of replacing late injection with early fusion is simple yet insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 7 datasets, multiple IPC settings, multiple architectures, ablation/visualization/transfer experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem diagnosis, excellent framework diagram, and coherent logical flow.
- **Value**: ⭐⭐⭐⭐ — A general plug-and-play method with practical value for the DD community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)
- [\[NeurIPS 2025\] Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](../../NeurIPS2025/image_restoration/enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)

</div>

<!-- RELATED:END -->
