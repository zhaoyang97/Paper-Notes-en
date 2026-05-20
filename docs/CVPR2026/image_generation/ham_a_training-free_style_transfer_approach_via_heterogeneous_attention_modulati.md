---
title: >-
  [Paper Note] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Style Transfer] This paper proposes HAM, a training-free style transfer method that achieves high-quality stylization without sacrificing content identity. HAM applies heterogeneous modulati…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Style Transfer"
  - "Attention Modulation"
  - "Training-Free"
  - "Diffusion Models"
  - "Identity Preservation"
date: 2026-05-08
content_hash: 1c7648c752ddabe0
---

# HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2603.24043](https://arxiv.org/abs/2603.24043)  
**Code**: None  
**Area**: Diffusion Models / Image Generation
**Keywords**: Style Transfer, Attention Modulation, Training-Free, Diffusion Models, Identity Preservation

## TL;DR
This paper proposes HAM, a training-free style transfer method that achieves high-quality stylization without sacrificing content identity. HAM applies heterogeneous modulation (GAR + LAT) to self-attention and cross-attention layers of diffusion models, complemented by style-injected noise initialization (SINI), attaining state-of-the-art performance across multiple metrics.

## Background & Motivation

**Background**: Diffusion model-based style transfer methods fall into two main categories: fine-tuning methods (training style control modules via LoRA/ControlNet) and training-free methods (manipulating attention features at inference time). Fine-tuning methods are computationally expensive and lack robustness; training-free methods such as StyleID and DiffArtist achieve stylization by injecting the style image's keys and values into self-attention layers.

**Limitations of Prior Work**: Existing training-free methods rely solely on self-attention manipulation to simultaneously inject style and preserve content. However, the Q/K/V projections in self-attention jointly encode spatial positional relationships and semantic representations, making it difficult to balance style expression and content preservation through a single channel, often leading to insufficient stylization or content distortion.

**Key Challenge**: Injecting style through self-attention inevitably corrupts content identity, as Q/K/V inherently couples spatial structure with semantic content. Existing methods are therefore trapped in a style–content trade-off.

**Goal**: To simultaneously capture complex style references and preserve content identity (structure, texture, text, etc.) under a training-free setting.

**Key Insight**: Decouple style injection and content protection across different attention mechanisms—using self-attention for global style–content fusion control, and cross-attention for precise local style transplantation and content preservation, thereby realizing heterogeneous modulation.

**Core Idea**: By applying heterogeneous attention modulation strategies to self-attention (GAR for global fusion) and cross-attention (LAT for local transplantation), HAM decouples style injection and content protection into distinct attention channels.

## Method

### Overall Architecture
HAM consists of three core modules: Global Attention Regulation (GAR), Local Attention Transplantation (LAT), and Style-Injected Noise Initialization (SINI). The system employs three parallel diffusion model branches: a content teacher (processing the content image), a style teacher (processing the style reference), and a student generator (producing the stylized output). SINI first constructs an initial noise that fuses style and content information. During denoising, GAR operates on self-attention layers for macro-level style–content fusion, while LAT operates on cross-attention layers for precise style/content control. The method is compatible with both SD2.1 (DDIM-based) and SD3.5 (DiT-based) architectures.

### Key Designs

1. **Global Attention Regulation (GAR)**:

    - **Function**: Achieves macro-level style introduction and content preservation in self-attention layers.
    - **Mechanism**: AdaIN is first applied to align and fuse the self-attention projections of the content teacher $(Q^c, K^c, V^c)$ and the style teacher $(Q^s, K^s, V^s)$, producing composite projections $(Q^{cs}, K^{cs}, V^{cs})$ by normalizing the content features and rescaling them with the mean and variance of the style features. The composite projections are then blended with the student generator's own self-attention projections via hyperparameter $\alpha$: $\hat{Q} = \alpha \cdot Q^m + (1-\alpha) \cdot Q^{cs}$, ensuring that the statistical distribution of self-attention projections remains aligned with the main branch throughout generation.
    - **Design Motivation**: Directly replacing K/V in self-attention severely disrupts content structure. AdaIN-based fusion followed by weighted blending introduces style statistics while preserving the spatial-semantic structure of the main branch. The optimal performance is achieved at $\alpha=0.75$, which favors retaining main-branch information.

2. **Local Attention Transplantation (LAT)**:

    - **Function**: Precisely controls style injection and content protection in cross-attention layers.
    - **Mechanism**: The cross-attention K/V from the style teacher are directly transplanted into the student generator's cross-attention, replacing the original K/V for localized style injection. To prevent content identity degradation, the query is protected by blending the content teacher's $Q^c_{cross}$ with the main branch's $Q^m_{cross}$ via hyperparameter $\beta$: $\hat{Q}^m_{cross} = \beta \cdot Q^m_{cross} + (1-\beta) \cdot Q^c_{cross}$.
    - **Design Motivation**: Unlike prior methods that operate exclusively in self-attention, LAT leverages cross-attention for style transplantation. Since cross-attention was originally designed for text-conditioned injection and does not carry spatial structural information, replacing K/V in this channel does not disrupt content structure as self-attention manipulation does. The optimal balance is achieved at $\beta=0.25$.

3. **Style-Injected Noise Initialization (SINI)**:

    - **Function**: Incorporates style and content information at the diffusion starting point, providing a better initialization for generation.
    - **Mechanism**: AdaIN is applied to fuse the content initial noise $z^c_T$ and the style initial noise $z^s_T$ into a stylized noise. A content residual term (the difference between the original content noise and the fused noise) is then introduced, with its weight controlled by hyperparameter $\gamma$: $z^m_T = \gamma \cdot \text{ContentResidual} + \text{StylizedNoise}$.
    - **Design Motivation**: Using content noise alone cannot incorporate style; pure AdaIN fusion loses content identity. The dual-component design allows the initial noise to carry both style statistics and content structural information via residual connection. The optimal value is $\gamma=0.5$.

### Loss & Training
The method requires no training; all operations are performed at inference time. SD2.1 uses 50-step DDIM denoising at a resolution of 512×512. The three hyperparameters $\alpha=0.75$, $\beta=0.25$, and $\gamma=0.5$ are determined via ablation studies.

## Key Experimental Results

### Main Results

| Method | ArtFID↓ | LPIPS↓ | DINO↑ | CLIP-I↑ | CLIP-T↑ | DC↑ | CC↑ |
|--------|---------|--------|-------|---------|---------|-----|-----|
| StyleID (CVPR'24) | 15.161 | 0.635 | 0.544 | 0.619 | 0.213 | 1.873 | 1.964 |
| DiffArtist (MM'25) | 16.174 | 0.520 | 0.629 | 0.626 | 0.220 | 1.987 | 1.984 |
| AttDistillation (CVPR'25) | 16.170 | 0.629 | 0.541 | 0.615 | 0.219 | 1.878 | 1.969 |
| **HAM (Ours)** | **15.151** | **0.479** | **0.728** | **0.682** | **0.223** | **2.113** | **2.057** |

### Ablation Study

| Configuration | DINO↑ | CLIP-I↑ | CLIP-T↑ | DC↑ | CC↑ |
|--------------|-------|---------|---------|-----|-----|
| Baseline (no modules) | 0.609 | 0.626 | 0.220 | 1.963 | 1.984 |
| +GAR | 0.618 | 0.626 | 0.231 | 1.993 | 2.002 |
| +LAT | 0.712 | 0.696 | 0.193 | 2.042 | 2.023 |
| +GAR+LAT | 0.746 | 0.696 | 0.202 | 2.099 | 2.040 |
| +GAR+LAT+SINI (Full) | 0.728 | 0.682 | 0.223 | 2.113 | 2.057 |

### Key Findings
- LAT contributes most to content preservation (DINO: 0.609 → 0.712) and is the core module for identity retention.
- GAR primarily enhances style strength (CLIP-T: 0.220 → 0.231) while marginally improving content metrics.
- SINI improves color diversity and style richness; in combination with the other two modules, it achieves the best overall DC/CC scores.
- The hyperparameter $\alpha=0.75$ favors retaining main-branch information; values that are too low cause severe content degradation.

## Highlights & Insights
- Assigning style injection and content protection to different types of attention mechanisms (heterogeneous modulation) is an elegant idea—leveraging cross-attention's inherent cross-modal information processing capability for style injection avoids the structural damage caused by self-attention manipulation.
- The AdaIN + residual noise initialization design elegantly addresses the style–content balance in initial noise, proving more effective than simple noise replacement or blending.
- Compatibility with both SD2.1 and SD3.5 architectures (DDIM vs. DiT) demonstrates the generality of the approach.

## Limitations & Future Work
- The method still has limitations in transferring highly abstract or surrealist styles.
- Running three diffusion model branches simultaneously (content teacher, style teacher, student) incurs substantial computational overhead (approximately 16 seconds per image on SD2.1).
- The three hyperparameters require manual tuning; different styles may require different settings.
- The ArtFID advantage over StyleID is marginal (15.151 vs. 15.161), leaving room for improvement in quantitative evaluation.

## Related Work & Insights
- **vs. StyleID**: StyleID injects style K/V only in self-attention, causing content distortion. HAM shifts style injection to cross-attention, significantly reducing content corruption.
- **vs. DiffArtist**: DiffArtist performs well on content preservation (LPIPS) but achieves weaker stylization than HAM. HAM's LAT module achieves a better balance through its query protection mechanism.
- The heterogeneous attention modulation paradigm is transferable to other domains such as image editing and video stylization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The heterogeneous modulation concept is novel, though individual components such as AdaIN fusion are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are detailed and cover all modules and hyperparameters, though user studies are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with complete formula derivations, though some descriptions are redundant.
- **Value**: ⭐⭐⭐⭐ A practical training-free style transfer solution with strong cross-architecture compatibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](../../AAAI2026/image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[CVPR 2026\] TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)

</div>

<!-- RELATED:END -->
