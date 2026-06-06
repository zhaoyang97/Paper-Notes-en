---
title: >-
  [Paper Note] LuVe: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts
description: >-
  [ICML 2026][Video Generation][Ultra-High-Resolution Video Generation] LuVe redefines UHR video generation from "passive detail enhancement" to "active content completion." Through a three-stage cascade (low-resolution mo…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Ultra-High-Resolution Video Generation"
  - "Diffusion Models"
  - "Dual Frequency Experts"
  - "Latent Space Upsampling"
  - "Cascaded Architecture"
date: 2026-05-08
content_hash: 2a7f9226c6d01e10
---

# LuVe: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts

**Conference**: ICML 2026  
**arXiv**: [2602.11564](https://arxiv.org/abs/2602.11564)  
**Code**: To be confirmed  
**Area**: Video Generation / Ultra-High Resolution  
**Keywords**: Ultra-High-Resolution Video Generation, Diffusion Models, Dual Frequency Experts, Latent Space Upsampling, Cascaded Architecture

## TL;DR
LuVe redefines UHR video generation from "passive detail enhancement" to "active content completion." Through a three-stage cascade (low-resolution motion → latent space upsampling → high-resolution refinement) and frequency-domain-driven Dual Frequency Experts (Low-Frequency Expert for global semantic consistency, High-Frequency Expert for texture refinement), it achieves a total score of 84.03 on VBench 4K, surpassing UltraWan-4K's 83.75.

## Background & Motivation

**Background**: Significant progress has been made in low-resolution video diffusion, but quality degrades severely at Ultra-High-Resolution (UHR). Existing solutions fall into three categories: training-free (modifying inference strategies without retraining), fine-tuning strategies (adapting to UHR datasets), and video super-resolution (generating at low resolution first, then upsampling frame-by-frame).

**Limitations of Prior Work**:
- Training-free methods suffer from over-smoothed textures and missing high-frequency information, as the base T2V model has not seen UHR data and lacks inherent UHR capabilities.
- VSR methods improve clarity but only perform low-level texture enhancement, failing to complete missing semantic structures and content.
- Direct training of UHR models faces a triple coupling challenge: (1) **Motion modeling difficulty**—high resolution limits the capacity of temporal modules; (2) **Semantic planning errors**—spatial expansion leads to global/local repetition or inconsistency; (3) **Insufficient detail synthesis**—motion blur, texture degradation, and high-frequency loss.

**Key Challenge**: The existing cascade paradigm (e.g., FlashVideo, LaVie, Waver) restricts the high-resolution stage to being a "detail enhancer," which only improves low-level visual attributes and cannot perform true content and semantic completion.

**Goal**: Redefine the cascade paradigm for UHR generation—not only to enhance details but also to strengthen global semantic coherence and content fidelity.

**Key Insight**: Observing the phased nature of the diffusion process through Power Spectral Density (PSD) analysis—low frequencies (global structures) are captured at high noise levels, while high frequencies (details) are synthesized at low noise levels. This motivates the design of specialized expert modules with clear division of labor.

**Core Idea**: Replace the traditional two-stage cascade with an LMG → VLU → HCR three-stage pipeline. By deploying Low-Frequency and High-Frequency experts at different diffusion stages, frequency-domain constraints are applied. This realizes a complete process of establishing motion priors → intelligent latent upsampling → joint semantic-detail completion.

## Method

### Overall Architecture
Three collaborative stages:
1. **Low-resolution Motion Generation (LMG)**: Uses a pre-trained T2V (Wan2.1-1.3B) to generate low-resolution video latents, establishing reliable temporal consistency and motion priors.
2. **Video Latent Upsampling (VLU)**: A dedicated VLUer performs continuous upsampling at arbitrary resolutions directly in the latent space, avoiding the high overhead of VAE encoding/decoding.
3. **High-resolution Content Refinement (HCR)**: Integrates Dual Frequency Experts—the Low-Frequency Expert (LFE) enhances global semantic coherence and content fidelity, while the High-Frequency Expert (HFE) refines texture and detail richness.

### Key Designs

1. **Video Latent Upsampler (VLUer) based on Implicit Representation**:
    - **Function**: Parameter-efficient continuous upsampling of any resolution on the latent manifold, avoiding manifold shift and codec overhead.
    - **Mechanism**: An encoder extracts features $F$ from the low-resolution latent $z_0^L$; a video INR upsampler maps features via an implicit function based on 3D coordinates $Q(x, y, t)$; a decoder reconstructs spatio-temporal representations $\hat{z}$ in the high-resolution latent domain, where $\hat{z}(x, y, t) = \text{Decoder}(U(F, Q(x, y, t)))$. Two-stage loss: Phase 1 uses latent L1 loss $\mathcal{L}_{\text{latent}} = \mathcal{L}_1(z_{sr}, z_{hr})$; Phase 2 adds pixel supervision and frame difference loss $\mathcal{L}_{\text{pixel}} = \mathcal{L}_1(x_{sr}, x_{hr}) + \mathcal{L}_{\text{frame}}$, where $\mathcal{L}_{\text{frame}} = \frac{1}{n-1} \sum_{t=2}^n \|\Delta x_{sr}^{(t)} - \Delta x_{hr}^{(t)}\|_1$.
    - **Design Motivation**: To avoid manifold shifts common in traditional latent interpolation and VAE bottlenecks in RGB interpolation. Pixel-level loss eliminates blocking artifacts, while frame difference loss explicitly constrains temporal coherence.

2. **Dual Frequency Experts (PSD Analysis Driven)**:
    - **Function**: Deploying frequency-specialized, parameter-efficient modules at different denoising stages of the diffusion process to reinforce low-frequency semantics and high-frequency details respectively.
    - **Mechanism**: PSD analysis shows Wan2.1 captures low frequencies (global structure) during high-noise stages and focuses on high frequencies (details) during low-noise stages. **LFE (Low-Frequency Expert)**: Trained during high-noise stages ($t \in [t_{\text{switch}}, 1]$), integrated via LoRA into DiT attention modules as $y = \text{Attention}(x) + \text{LoRA}(\text{LowPass}(x))$. **HFE (High-Frequency Expert)**: Trained during low-noise stages ($t \in [0, t_{\text{switch}}]$), integrated via LoRA into FFN layers as $y = \text{FFN}(x) + \text{LoRA}(\text{HighPass}(x))$. The switching point is $t_{\text{switch}} = 0.417$.
    - **Design Motivation**: (1) Parameter efficiency via LoRA; (2) Functional division—Attention handles global semantics (low-freq) while FFN handles local details (high-freq); (3) Explicit frequency constraints—low/high-pass filtering ensures experts focus on target bands; (4) Natural alignment with the diffusion process phases.

3. **Data Selection and Augmentation Strategy**:
    - **Function**: Providing targeted, high-quality training data distributions for the two frequency experts.
    - **Mechanism**: **LFE Data**: Filtering the UltraVideo dataset using HPS v3 scores, keeping only samples > 6.5 to ensure clean data with strong semantic alignment. **HFE Data**: Applying Unsharp Masking to the LFE-filtered subset to amplify high-frequency components and boundary sharpness.
    - **Design Motivation**: Task-specific data distribution is crucial for robust UHR generation. LFE requires clean data for global consistency, while HFE requires augmented data for rich texture details.

## Key Experimental Results

### Main Results (VBench)

| Model | SC ↑ | BC ↑ | TF ↑ | IQ ↑ | AQ ↑ | Average ↑ |
|------|------|------|------|------|------|--------|
| Wan2.1-720p | 95.70 | 96.05 | 98.45 | 68.28 | 56.46 | 82.98 |
| UltraWan-1K | 95.40 | 96.45 | 98.98 | 58.26 | 49.89 | 79.79 |
| UltraWan-4K | 95.81 | 96.11 | 97.71 | 71.44 | 57.69 | 83.75 |
| CineScale-4K | 95.16 | 95.95 | 97.80 | 67.74 | 57.82 | 82.89 |
| **Ours-2K** | **95.83** | **96.76** | **98.18** | **71.15** | **59.78** | **84.34** |
| **Ours-4K** | **95.36** | **96.46** | **98.09** | **71.33** | **58.91** | **84.03** |

The 4K composite score of 84.03 surpasses UltraWan-4K (83.75) and CineScale-4K (82.89).

### Ablation Study

| Configuration | Mode | FID_patch ↓ | Realism ↑ | AQ ↑ |
|------|------|------------|--------|------|
| UHR scaling only | End-to-end | 54.10 | 6.72 | 57.04 |
| LoRA Experts | Cascade | 47.03 | 7.28 | 58.65 |
| w/o Experts | Cascade | 46.48 | 7.00 | 58.57 |
| w/o LF Expert | Cascade | 43.86 | 7.08 | 59.10 |
| w/o HF Expert | Cascade | 44.44 | 7.36 | 59.34 |
| w/o Data Selection | Cascade | 43.77 | 7.40 | 58.80 |
| w/o Unsharp Masking | Cascade | 42.96 | 7.52 | 59.53 |
| **Full Model** | Cascade | **41.03** | **7.64** | **59.78** |

### Comparison with VSR methods (VSR applied on VBench outputs)

| Method | MUSIQ ↑ | MANIQA ↑ | NIQE ↓ | DOVER ↑ |
|------|---------|---------|--------|---------|
| RealBasicVSR | 55.90 | 0.401 | 4.15 | 0.712 |
| FlashVSR | 56.54 | 0.402 | 3.20 | 0.755 |
| **Ours** | **58.01** | **0.410** | **3.16** | **0.784** |

### Key Findings
- **Cruciality of LFE**: Removing the LF expert caused FID_patch to rise from 41.03 to 43.86 (+6.9%). Qualitative analysis showed scattered attention maps, semantic planning errors, and content artifacts.
- **Contribution of HFE**: Removing the HF expert increased FID_patch to 44.44 (+8.3%), with visual evidence of texture blur and loss of detail.
- **Data Strategy**: Removing Unsharp Masking resulted in an FID_patch of 42.96 vs. 41.03 (-4.7%), proving data augmentation for the high-frequency expert is indispensable.
- **Human Evaluation**: In a study with 60 videos and 20 reviewers, the proposed method led significantly in all dimensions (> 60% preference rate)—Overall Quality 63.5% / Details 60.3% / Temporal Consistency 62.3% / Text Alignment 61.1%.

## Highlights & Insights
- **Strategic Value of Paradigm Shift**: Moving from passive "detail enhancement" to "active content completion" redefines the role of the high-resolution generation stage. It upgrades the UHR problem from "how to be clearer" to "how to be more realistic and rich."
- **Elegant Frequency Decomposition**: Utilizing the inherent frequency structure of the diffusion process identified via PSD analysis. Low/high-pass filtering + task-specific LoRA experts provide a precise mapping to these stages, reflecting a deep understanding of internal diffusion mechanisms.
- **Self-Consistency of Three-Tier Design**: High self-consistency in module selection (Attention → Global → LFE, FFN → Local → HFE), temporal division (High noise → Low freq → LFE, Low noise → High freq → HFE), and data strategy (HPS filtering + Unsharp Masking).
- **Parameter Efficiency**: Implemented via LoRA, total trainable parameters are far fewer than full fine-tuning.
- **Transferable Design**: The concept of frequency decomposition + phased experts can be extended to other multi-stage generation tasks (e.g., T2I SR, multimodal generation).

## Limitations & Future Work
- VLUer inference latency (0.922s/frame) remains significantly higher than latent interpolation (0.004s), requiring further acceleration for industrial real-time applications.
- The method depends on high-quality UHR video data (UltraVideo dataset) and is sensitive to data distribution.
- Future improvements: Exploring more efficient latent upsampling operators (distillation/knowledge transfer); researching adaptive frequency switching instead of a fixed $t_{\text{switch}} = 0.417$; extending to more tasks and architectures.

## Related Work & Insights
- **vs Training-free methods** (Demofusion / LSRNA): These extend pre-trained models to high resolution by modifying inference. While computationally efficient, they are limited by the base model's generative capacity. LuVe actively enhances generative capacity via frequency experts.
- **vs Traditional VSR** (RealBasicVSR / VEnhancer): VSR modules are trained independently and cannot complete semantic information lost in the low-resolution stage. LuVe's tight cascade + frequency experts enable joint semantic-detail optimization.
- **vs Existing Cascade methods** (FlashVideo / LaVie / Waver): Existing solutions limit the high-resolution stage to passive enhancement. LuVe breaks this paradigm bottleneck, allowing the high-resolution stage to participate in content completion and semantic fidelity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Paradigm innovation (detail enhancement to content completion) + frequency decomposition design with both theoretical depth and engineering value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive multi-dimensional comparison (VBench / FID_patch / Custom metrics vs. T2V / VSR / Human Evaluation) + detailed ablation study.
- Writing Quality: ⭐⭐⭐⭐ Rigorous logic; PSD analysis deeply motivates the design; method description is clear and reproducible.
- Value: ⭐⭐⭐⭐⭐ Addresses actual UHR generation bottlenecks (semantic consistency + detail fidelity), holding significant value for both academia and industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](../../ICCV2025/video_generation/magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[AAAI 2026\] SphereDiff: Tuning-free Omnidirectional Panoramic Image and Video Generation via Spherical Latent Representation](../../AAAI2026/video_generation/spherediff_tuning-free_360_static_and_dynamic_panorama_generation_via_spherical_.md)
- [\[ICML 2026\] Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation](quantized_keys_steal_attention_bias_correction_for_kv-cache_compression_in_video.md)

</div>

<!-- RELATED:END -->
