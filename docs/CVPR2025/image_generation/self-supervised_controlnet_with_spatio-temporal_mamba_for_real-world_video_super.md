---
title: >-
  [Paper Note] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution
description: >-
  [CVPR 2025][Image Generation][Video Super-Resolution] The SCST framework is proposed, which utilizes Spatio-Temporal Continuous Mamba (STCM) for global 3D attention modeling, combines it with a MoCo-based self-supervised ControlNet to extract degradation-agnostic features, and incorporates a three-stage hybrid training strategy to achieve SOTA perceptual quality on real-world video super-resolution benchmarks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Video Super-Resolution"
  - "Mamba"
  - "Self-Supervised Learning"
  - "ControlNet"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: d3b3c916cb670dcc
---

# Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2506.01037](https://arxiv.org/abs/2506.01037)  
**Code**: [https://ssj9596.github.io/scst-project/](https://ssj9596.github.io/scst-project/)  
**Area**: Diffusion Models / Video Super-Resolution  
**Keywords**: Video Super-Resolution, Mamba, Self-Supervised Learning, ControlNet, Contrastive Learning

## TL;DR

The SCST framework is proposed, which utilizes Spatio-Temporal Continuous Mamba (STCM) for global 3D attention modeling, combines it with a MoCo-based self-supervised ControlNet to extract degradation-agnostic features, and incorporates a three-stage hybrid training strategy to achieve SOTA perceptual quality on real-world video super-resolution benchmarks.

## Background & Motivation

**Background**: Video Super-Resolution (VSR) aims to reconstruct high-resolution videos by utilizing temporally complementary information between low-resolution frame sequences. Diffusion-model-based methods (e.g., StableSR, MGLD) have demonstrated powerful generative prior capabilities, but face two core challenges: (1) the randomness of diffusion sampling leads to temporal inconsistency across frames; (2) complex real-world degradations (superimposed blur, noise, compression, etc.) make the condition injection unstable.

**Limitations of Prior Work**: Existing methods employ 3D convolutions or temporal attention (e.g., Upscale-A-Video) to address temporal consistency, but suffer from limited receptive fields. Utilizing optical flow alignment (e.g., MGLD) can enhance temporal coherence, but falls short in spatial recovery. More critically, unknown and complex degradations in real-world LR videos make it difficult for ControlNet to extract clean features, and direct training is prone to generating artifacts.

**Key Challenge**: The simultaneous resolution of two coupled problems is required: modeling inter-frame spatio-temporal dependencies (temporal consistency) and resisting unknown degradations (spatial quality). These two objectives constrain each other, which increases learning complexity.

**Goal**: To design a real-world VSR framework that simultaneously ensures spatio-temporal consistency across frames and resists complex degradations.

**Key Insight**: (1) The linear complexity of Mamba's global attention is suitable for 3D spatio-temporal video modeling; (2) contrastive learning can extract degradation-agnostic feature representations, avoiding the interference of degradation information in LR frames during condition injection.

**Core Idea**: To extend Mamba to 3D spatio-temporal continuous scanning to realize intra-frame + inter-frame global attention, and to train ControlNet using MoCo-style contrastive learning, enabling it to learn to extract clean features aligned with HR from LR.

## Method

Based on the pre-trained Stable Diffusion 2.1, SCST introduces two core modules: Spatio-Temporal Continuous Mamba (STCM) for global 3D attention, and self-supervised ControlNet (MoCoCtrl) for extracting clean conditional features from degraded LR. These two modules are progressively introduced through a carefully designed three-stage training strategy.

### Overall Architecture

The input is a low-resolution video sequence $x^l \in \mathbb{R}^{T \times H \times W \times 3}$ of $T$ frames, and the output is the corresponding high-resolution sequence $x^h \in \mathbb{R}^{T \times sH \times sW \times 3}$. The framework is based on LDM, where a ControlNet encoder $E$ extracts multi-scale features of the LR to be injected into the denoising U-Net $D$. The STCM module is embedded within the U-Net to realize 3D attention, and ControlNet is optimized via MoCo contrastive learning to resist degradations.

### Key Designs

1. **Spatio-Temporal Continuous Mamba (STCM)**:

    - Function: Achieves global 3D attention within and across video frames with linear complexity.
    - Mechanism: The core component is the 3D-Mamba Block, which uses 3D depth-wise convolutions to capture spatio-temporal dependencies, and then processes the feature sequence through 6 scanning paths (3 modes $\times$ 2 directions, original + flipped). The key innovation is the **spatio-temporal continuous scanning strategy**: unlike conventional 3D scanning (which resets the scanning state between frames), this strategy maintains the continuity of pixels both within frames (orange path, scanning sequentially pixel-by-pixel) and across frames (cyan path, tracking the same spatial location across frames). Each patch obtains contextual knowledge along the scanning path through the compressed latent states, processed by SSM with linear complexity.
    - Design Motivation: Full 3D attention (as in some video generation methods) demonstrates outstanding performance but is computationally expensive. Decoupled spatial + temporal attention is computationally efficient but has a limited receptive field. STCM achieves global 3D attention through the linear complexity of Mamba, balancing efficiency and effectiveness. Continuous scanning maintains pixel-level spatio-temporal consistency better than traditional reset scanning.

2. **Self-Supervised ControlNet (MoCoCtrl)**:

    - Function: Extracts clean features aligned with HR from degraded LR videos.
    - Mechanism: Employs a MoCo-like architecture, utilizing a query encoder $E_q$ to process LR frames and a momentum encoder $E_k$ to process HR frames, where the weights of $E_k$ are the Exponential Moving Average (EMA) of $E_q$. A projection head generates $P \times P$ patch-level features, and patch-level contrastive loss is computed as $\mathcal{L}_q = \frac{1}{P^2}\sum_p -\log \frac{\exp(q^p \cdot k_+^p / \tau)}{\exp(q^p \cdot k_+^p / \tau) + \sum_Q \exp(q^p \cdot Q / \tau)}$. Negative samples are sourced from a memory queue containing both HR and LR encodings.
    - Design Motivation: Directly using degraded LR as conditions to train ControlNet leads to unstable optimization and artifacts. Through contrastive learning, the LR encoder is forced to learn to generate features aligned with those of the HR encoder, thereby filtering out degradation noise. Traditional MoCo uses globally pooled features for classification, while here it is adapted to the patch level to capture the spatial details required for super-resolution.

3. **Three-Stage Hybrid Training Strategy**:

    - Function: Progressively introduces various modules to stabilize the training process.
    - Mechanism: Stage 1—Trains only the ControlNet using a mixed HR/LR video dataset (where HR can be regarded as LR with zero degradation), with the mixing ratio gradually decreasing from 1.0 to 0.3, allowing ControlNet to learn reconstruction before adapting to degradation. Reconstruction/SR labels are introduced to distinguish the two tasks. Stage 2—Introduces MoCoCtrl, with the HR/LR ratio fixed at 1:1, leveraging contrastive learning to make full use of the reconstruction priors learned in Stage 1. Stage 3—Introduces STCM, freezing the ControlNet and training only the temporal module using LR.
    - Design Motivation: Directly training all modules end-to-end leads to instability. Allowing ControlNet to first learn to extract clean features (Stage 1+2), and then let the temporal module learn spatio-temporal consistency on top of stable conditional features (Stage 3), decouples the two difficult challenges.

### Loss & Training

The primary loss is the denoising loss $\mathbb{E}_{t,x^h}\|D(x_t^h, t, E(x^l)) - \epsilon_t\|^2$ plus the patch-level contrastive loss of MoCoCtrl. The weights of the original SD 2.1 U-Net are frozen, and only the newly added layers are trained. Training is conducted on 8×A100 GPUs, with Stage 1 taking approximately 12h, Stage 2 about 30h, and Stage 3 about 30h. During inference, the sequence length is 8 frames with 20 sampling steps.

## Key Experimental Results

### Main Results

| Dataset | Metric | RealBasicVSR | MGLD | Upscale-A-Video | **SCST** |
|--------|------|-------------|------|-----------------|----------|
| REDS4 | LPIPS↓ | 0.2545 | 0.2660 | 0.3639 | **0.2518** |
| REDS4 | DISTS↓ | 0.1196 | 0.1171 | 0.1840 | **0.1094** |
| UDM10 | LPIPS↓ | 0.2812 | 0.2551 | 0.2799 | **0.2156** |
| VideoLQ | CLIP-IQA↑ | 0.3881 | 0.3462 | 0.2818 | **0.4859** |
| VideoLQ | MUSIQ↑ | 55.61 | 50.94 | 43.34 | **59.20** |
| VideoLQ | NIQE↓ | 3.698 | 3.727 | 4.876 | **3.566** |

### Ablation Study

| Model | MoCoCtrl | STCM | PSNR↑ | LPIPS↓ | DISTS↓ |
|------|----------|------|-------|--------|--------|
| (a) Baseline | ✗ | ✗ | 21.22 | 0.2824 | 0.1596 |
| (b) + MoCoCtrl | ✓ | ✗ | - | Gain | Gain |
| (c) + STCM | ✗ | ✓ | - | Gain | Gain |
| (d) Full | ✓ | ✓ | 24.31 | **0.2525** | **0.1344** |

### Key Findings

- SCST achieves the best perceptual metrics (LPIPS, DISTS) across all synthetic test sets, and outperforms others on all no-reference metrics on the real-world VideoLQ dataset.
- Both MoCoCtrl and STCM make independent contributions, and their combined use yields superimposed effects.
- The advantage on real-world data is particularly prominent—with CLIP-IQA leading the second-best method by approximately 0.1 (0.4859 vs. 0.3881).
- Spatio-temporal continuous scanning outperforms traditional flattened reset scanning (validated via ablation).
- Qualitative comparison shows that SCST is the only method capable of clearly depicting eagle-eye details and tire textures.

## Highlights & Insights

- Spatio-temporal continuous scanning is introduced to video super-resolution for the first time via Mamba, presenting a compelling design that maintains spatial continuity within frames and pixel-tracking continuity across frames.
- The patch-level contrastive learning of MoCoCtrl cleverly repurposes the MoCo framework but adapts it to pixel-level restoration tasks.
- The three-stage training strategy is reasonably designed—first learning clean features, then learning degradation robustness, and finally learning temporal consistency.
- The significant lead under real-world degradation scenarios validates the importance of degradation-agnostic feature extraction.

## Limitations & Future Work

- During inference, videos need to be processed in segments (due to memory constraints), which may introduce inconsistencies between segments.
- Training all three stages takes more than 72 hours in total (8×A100), representing a relatively high training cost.
- Based on SD 2.1, using newer base models (such as SDXL) might bring further improvements.
- Only 4× super-resolution was evaluated; applicability to other scale factors (such as 2×, 8×) has not been verified.
- The 6 paths of spatio-temporal continuous scanning might contain redundancies, and more economical path designs are worth exploring.

## Related Work & Insights

- **Upscale-A-Video**: Employs 3D convolutions + temporal attention, but suffers from limited receptive fields.
- **MGLD**: Utilizes optical flow alignment, showing good temporal consistency but insufficient spatial recovery.
- **StableSR**: Direct application of image-level SR to videos leads to temporal inconsistency.
- Insight: The application of contrastive learning to low-level vision restoration tasks (rather than high-level semantics), and the feature alignment concept can be generalized to other restoration tasks like deraining and dehazing.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First to introduce Mamba into VSR, with a cleverly designed MoCoCtrl |
| Experimental Thoroughness | 4 | Comprehensive evaluation across multiple datasets, including ablation and qualitative comparisons |
| Writing Quality | 4 | Clear framework and rich illustrations |
| Value | 4 | Significant lead in real-world VSR scenarios |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[CVPR 2026\] STCDiT: Spatio-Temporally Consistent Diffusion Transformer for High-Quality Video Super-Resolution](../../CVPR2026/image_generation/stcdit_spatio-temporally_consistent_diffusion_transformer_for_high-quality_video.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2025\] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)

</div>

<!-- RELATED:END -->
