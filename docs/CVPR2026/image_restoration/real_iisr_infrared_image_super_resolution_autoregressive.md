---
title: >-
  [Paper Note] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset
description: >-
  [CVPR 2026][Image Restoration][Infrared image super-resolution] This paper proposes Real-IISR, a unified autoregressive framework that addresses the unique challenges of real-world infrared image super-resolution via a T…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Infrared image super-resolution"
  - "visual autoregression"
  - "thermal-structure guidance"
  - "conditional adaptive codebook"
  - "thermal order consistency"
date: 2026-05-08
content_hash: 6177be9da2908251
---

# Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset

**Conference**: CVPR 2026
**arXiv**: [2603.04745](https://arxiv.org/abs/2603.04745)  
**Code**: [https://github.com/JZD151/Real-IISR](https://github.com/JZD151/Real-IISR)  
**Area**: Image Restoration / Infrared Super-Resolution
**Keywords**: Infrared image super-resolution, visual autoregression, thermal-structure guidance, conditional adaptive codebook, thermal order consistency

## TL;DR

This paper proposes Real-IISR, a unified autoregressive framework that addresses the unique challenges of real-world infrared image super-resolution via a Thermal-Structure Guidance (TSG) module, a Conditional Adaptive Codebook (CAC), and a Thermal Order Consistency loss. It also introduces the FLIR-IISR dataset comprising 1,457 real LR-HR infrared image pairs.

## Background & Motivation

1. **Background**: Visible-image super-resolution has seen remarkable progress, yet infrared imaging exhibits unique degradations—spatially varying blur, unstable thermal boundaries, and temperature-dependent radiometric drift—caused by longer wavelengths and weaker atmospheric scattering.
2. **Limitations of Prior Work**: Existing IISR methods are trained on synthetic datasets (downsampled IVIF datasets) and thus fail to capture real infrared degradations. Diffusion models are further limited in IISR by stochastic sampling and the absence of infrared-specific degradation priors.
3. **Key Challenge**: (1) The lack of a real infrared degradation dataset; (2) the absence of infrared-aware degradation modeling—thermal radiation intensity does not correspond to structural edges, and non-uniform degradation introduces quantization bias.
4. **Goal**: Simultaneously address both the dataset and methodological gaps in real-world IISR.
5. **Key Insight**: Exploit the temperature–brightness monotonicity of infrared imaging as a physical constraint.
6. **Core Idea**: Dual thermal-structure guidance + degradation-adaptive codebook + thermal order preservation loss.

## Method

### Overall Architecture

The TSG module fuses thermal priors for degradation-aware encoding → the VAR backbone generates outputs via scale-by-scale prediction → CAC dynamically adjusts quantized embeddings → the Thermal Order Consistency loss enforces physical consistency.

### Key Designs

1. **Thermal-Structure Guidance Module (TSG)**:

    - **Function**: Mitigates the mismatch between thermal radiation and structural edges.
    - **Mechanism**: A thermal map $I_{\text{Heat}}$ and an edge map $I_{\text{Edge}}$ are constructed from the LR input and encoded by DINOv3 encoders, respectively. A learnable attention gate $W = \sigma(L(A) + G(A))$ adaptively balances the contributions of both features. The fused features guide LR features via cross-attention.
    - **Design Motivation**: A car engine as a strong heat source produces a thermal radiation region that often deviates from the vehicle's actual contour. Direct training would cause the model to overfit thermal peaks while neglecting true structural edges.

2. **Conditional Adaptive Codebook (CAC)**:

    - **Function**: Dynamically corrects quantization bias to enhance texture fidelity.
    - **Mechanism**: Each code embedding is dynamically modulated via low-rank perturbation: $Z'(g)[i] = Z[i] + \tanh(\alpha)[(U_i \odot h(g))V^\top]$, where the conditioning vector $h(g)$ is derived from TSG features. The same discrete index can be decoded into different embedding vectors under different degradation conditions.
    - **Design Motivation**: Discretization errors introduced by standard VQ-VAE quantization are exacerbated under non-uniform infrared degradation, and a static codebook cannot adapt to spatially varying degradation patterns.

3. **Thermal Order Consistency Loss $\mathcal{L}_{\text{TOC}}$**:

    - **Function**: Preserves the monotonic physical relationship between temperature and brightness.
    - **Mechanism**: Applied over adjacent patch pairs: $\mathcal{L}_{\text{TOC}} = \text{ReLU}(-[(I_{\text{SR}}^p(i) - I_{\text{SR}}^p(j)) \times (I_{\text{HR}}^p(i) - I_{\text{HR}}^p(j))])$. It penalizes cases where brightness ordering between SR and HR is inconsistent.
    - **Design Motivation**: In infrared images, higher temperature corresponds to higher pixel brightness (monotonicity). Degradations such as defocus and motion blur cause local thermal compression and peak shifts. MSE constrains only absolute values and cannot guarantee relative ordering.

### Loss & Training

$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{TOC}}$, with $\lambda_1=0.2$, $\lambda_2=0.8$. Training uses 4 × A800 GPUs, AdamW optimizer, and 10K fine-tuning iterations.

## Key Experimental Results

### Main Results

| Dataset | Metric | Real-IISR | DifIISR (Prev. SOTA) | Gain |
|--------|------|-----------|-------------------|------|
| FLIR-IISR@Set5 | MUSIQ↑ | 59.90 | 54.79 | +5.11 |
| FLIR-IISR@Set15 | MUSIQ↑ | 57.06 | 53.16 | +3.90 |
| FLIR-IISR@Set5 | LPIPS↓ | 0.1615 | 0.2525 | −0.091 |

### Ablation Study

| Configuration | PSNR | MUSIQ | Note |
|------|------|-------|------|
| w/o TSG | Drop | Drop | Edge blurring, inaccurate thermal distribution |
| w/o CAC | Drop | Drop | Unstable texture |
| w/o $\mathcal{L}_{\text{TOC}}$ | Drop | Drop | Thermal peak drift |
| VAR vs. diffusion baseline | VAR better | VAR better | Deterministic generation better suited for infrared |

### Key Findings

- Real-IISR has the highest parameter count (1144.6M) yet achieves the fastest inference (2.45 FPS), as autoregression requires no multi-step denoising.
- Iterative denoising in diffusion-based methods blurs high-frequency thermal details and disrupts structure–thermal correspondence.
- $\mathcal{L}_{\text{TOC}}$ effectively prevents thermal peak drift and maintains physical consistency.

## Highlights & Insights

- **Domain-specific constraint design**: The thermal order consistency loss elegantly exploits the physical monotonicity of infrared imaging.
- **Real dataset construction**: Real degradations are simulated via autofocus variation and motion blur, filling the gap in real-world infrared SR data.
- **Low-rank perturbation in CAC**: A low-rank structure controls the magnitude of codebook modulation, balancing flexibility and stability.

## Limitations & Future Work

- The FLIR-IISR dataset contains only 1,457 pairs, limiting its scale.
- Only 4× super-resolution is supported; generalization to other scale factors is unverified.
- The quality of thermal and edge maps depends on the LR input, which may be unreliable under extreme degradation.
- Future work could extend the framework to infrared video super-resolution to leverage temporal information.

## Related Work & Insights

- **vs. VARSR**: VARSR is designed for visible images without infrared constraints; Real-IISR introduces thermal priors.
- **vs. DifIISR**: DifIISR employs diffusion with gradient alignment, but multi-step denoising is slow and blurs infrared details.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Thermal-structure guidance and thermal order constraints are innovative designs tailored for infrared imaging, filling the gap in real-world infrared SR.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, multiple competing methods, comprehensive ablation studies, and complete efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; thermal map visualizations and grayscale fluctuation plots intuitively demonstrate physical consistency.
- **Value**: ⭐⭐⭐⭐ — Addresses both the data and methodological gaps in real-world infrared SR, providing a benchmark for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution](finpercep_rm_a_fine_grained_reward_model_and_co_evolutionary_curriculum_for_rl_ba.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)

</div>

<!-- RELATED:END -->
