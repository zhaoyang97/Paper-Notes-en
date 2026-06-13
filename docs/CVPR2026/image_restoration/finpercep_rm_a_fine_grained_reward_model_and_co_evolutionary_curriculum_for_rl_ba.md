---
title: >-
  [Paper Note] FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][Image Super-Resolution] This paper proposes FinPercep-RM, a fine-grained perceptual reward model, and a Co-evolutionary Curriculum Learning (CCL) strategy to address reward hacking and trai…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Image Super-Resolution"
  - "Reward Model"
  - "RLHF"
  - "Fine-grained Quality Assessment"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: f091a2812ad95c35
---

# FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2512.22647](https://arxiv.org/abs/2512.22647)  
**Code**: [https://github.com/lyd-2022/FinPercep-RM](https://github.com/lyd-2022/FinPercep-RM)  
**Area**: Image Restoration / Super-Resolution
**Keywords**: Image Super-Resolution, Reward Model, RLHF, Fine-grained Quality Assessment, Curriculum Learning

## TL;DR

This paper proposes FinPercep-RM, a fine-grained perceptual reward model, and a Co-evolutionary Curriculum Learning (CCL) strategy to address reward hacking and training instability when applying RLHF to real-world image super-resolution. The model simultaneously outputs a global quality score and a spatial degradation heatmap, enabling localized artifact awareness.

## Background & Motivation

Diffusion-based real-world image super-resolution (Real-ISR) has achieved remarkable progress by leveraging generative priors from large-scale T2I models to synthesize rich textures. RLHF, as a successful optimization paradigm in T2I generation, has naturally been transferred to ISR tasks—using image quality assessment (IQA) models as reward signals to guide SR models.

However, existing IQA models (e.g., CLIP-IQA, MANIQA) output only a single global score and are **insensitive to local fine-grained distortions**—an image with prominent local artifacts may receive a high score comparable to a clean reference. This leads to severe **reward hacking**: the generator learns to "please" the imperfect reward signal, converging to results with high global scores but filled with local artifacts and an "over-smoothed" painterly appearance.

On the other hand, directly applying a more sophisticated fine-grained reward model causes **training instability**—high-variance spatial penalty signals induce policy gradient oscillations and convergence failure. This creates a **stability–robustness dilemma**: simple global rewards are stable but susceptible to hacking; complex fine-grained rewards are robust but unstable.

The core insight is: **a good reward model should not only assess "What" the quality is, but also diagnose "Where" the defects are.** Curriculum learning is simultaneously employed to progressively introduce fine-grained rewards from simple to complex.

## Method

### Overall Architecture

The framework comprises three core components: (1) a T2I prior-based SR generator; (2) FinPercep-RM, a diagnostic reward model whose encoder produces a global quality score and whose decoder outputs a fine-grained perceptual degradation map (fg-PDM); and (3) the CCL co-evolutionary curriculum, in which the reward model progressively expands from a simple global IQA to the full fine-grained version, while the generator training simultaneously transitions from simple to complex rewards.

### Key Designs

1. **FinPercep-RM Encoder-Decoder Architecture**:

    - **Function**: Provides both global quality scoring and spatial defect localization simultaneously.
    - **Mechanism**: The encoder (based on a CLIP-IQA backbone) extracts multi-scale features $\{f_i\}_{i=1}^N$, with the deepest feature $f_N$ used for global scoring. The decoder reconstructs a perceptual degradation map $M_{fg-pdm} \in [0,1]$ (Sigmoid-normalized) at the input resolution via upsampling and cross-layer fusion. Crucially, the global score $S_{fgc-global}$ is not directly regressed from $f_N$; instead, $f_N$ is first modulated by the degradation map as $f_N \odot \text{interpolate}(M_{fg-pdm})$, then passed through an MLP for regression—forcing the global score to be intrinsically conditioned on local defects.
    - **Design Motivation**: The decoupling of global scores from local quality in conventional IQA is the root cause of reward hacking. This architectural design enforces that the global score "observes" spatial defects.

2. **FGR-30k Fine-grained Reward Dataset**:

    - **Function**: Provides training data with spatial defect annotations.
    - **Mechanism**: High-quality images $I_{GT}$ are degraded to $I_{LR}$; multiple SR models generate $I_{SR}$; a **region-swapping** strategy then "transplants" artifact regions from $I_{SR}$ into $I_{GT}$ (using random masks and SAM semantic masks) to obtain synthetic samples $I_{syn}$. Ground-truth degradation maps $M_{gt}$ are derived by fusing pixel-level L1 distance and DINOv3 feature-level cosine distance.
    - **Design Motivation**: Existing IQA and preference datasets lack spatial defect annotations. Region swapping provides controllable local artifact injection with precise training supervision.

3. **Co-evolutionary Curriculum Learning (CCL)**:

    - **Function**: Achieves a balance between stability and robustness.
    - **Mechanism**: A dual-path co-evolutionary scheme is employed. (a) Reward model progressive expansion: starting from a simple global IQA ($RM_0$), decoder parameters are introduced in stages through training on FGR-30k, gradually evolving from global scoring to the full model with heatmaps. (b) Generator curriculum co-evolution: the generator first converges with the stable $RM_0$, then progressively transitions to stricter $RM_k$. This easy-to-hard design provides stable initialization in the early phase and robust fine-grained supervision in the later phase.
    - **Design Motivation**: Directly applying the full FinPercep-RM causes policy gradient oscillations and convergence failure (training curves exhibit severe oscillation). Curriculum learning enables a smooth training transition.

### Loss & Training

The FinPercep-RM training loss is: $\mathcal{L}_{total} = \lambda_{map} \mathcal{L}_{map} + \lambda_{rank} \mathcal{L}_{rank} + \lambda_{align} \mathcal{L}_{align}$. $\mathcal{L}_{map}$ is the heatmap L1 loss; $\mathcal{L}_{rank}$ is a triplet ranking loss ($S_{SR} < S_{syn} < S_{GT}$); $\mathcal{L}_{align}$ is an anchor alignment loss that prevents score drift across curriculum stages. RL optimization follows the standard RLHF pipeline.

## Key Experimental Results

### Main Results

**DRealSR Dataset, applied to different SR baselines**

| Base Model | Configuration | LPIPS↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ |
|---------|------|--------|--------|---------|---------|
| SUPIR | Baseline | 0.452 | 65.665 | 0.629 | 0.572 |
| SUPIR | + Standard IQA Reward | 0.465 | 64.892 | 0.612 | 0.589 |
| SUPIR | **+ FinPercep-RM** | **0.428** | **67.234** | **0.648** | **0.586** |
| DreamClear | Baseline | 0.317 | 65.077 | 0.605 | 0.543 |
| DreamClear | + Standard IQA Reward | 0.332 | 64.123 | 0.591 | 0.567 |
| DreamClear | **+ FinPercep-RM** | **0.295** | **67.891** | **0.632** | **0.561** |

### Ablation Study

| Configuration | Training Stability | Reward Hacking | Final Quality | Notes |
|------|----------|---------|---------|------|
| Standard IQA Reward | ✅ Stable | ❌ Severe | Moderate | Visible local artifacts |
| FinPercep-RM (w/o CCL) | ❌ Oscillating | ✅ Mitigated | Poor (no convergence) | Unstable when applied directly |
| **FinPercep-RM + CCL** | **✅ Stable** | **✅ Mitigated** | **Best** | Full method |

### Key Findings

- Standard IQA rewards cause LPIPS to increase (greater distortion) while MUSIQ scores decrease—a classic reward hacking signature: the model learns to please the IQA metric while actual quality degrades.
- FinPercep-RM consistently improves performance across all SR baselines (including ResShift, SUPIR, DreamClear, DiffBIR, SeeSR, and DIT4SR).
- The CCL training curve is smooth and converges to higher final reward values, whereas directly applying FinPercep-RM produces severe oscillations.
- Degradation map visualizations demonstrate that FinPercep-RM accurately localizes texture artifact regions.

## Highlights & Insights

- **Attributing reward hacking to insufficient perceptual granularity** is a precise diagnosis: global scores indeed cannot distinguish between "overall good but locally poor" and "overall good and locally good."
- **The architecture-level coupling of global and local information** is elegant: modulating global features with the degradation map before regression makes the two inseparable at the architectural level.
- **CCL curriculum learning** addresses a general RLHF problem—training instability induced by complex reward signals—and is potentially transferable to other RLHF applications.

## Limitations & Future Work

- The ground-truth degradation maps in FGR-30k rely on a heuristic fusion of pixel-level and feature-level differences, which may not perfectly reflect human perception.
- The stage partitioning and switching timing in CCL require manual tuning; an adaptive curriculum would be preferable.
- Validation is limited to ISR tasks; applicability to other image generation RLHF settings (e.g., T2I) remains unexplored.
- The encoder-decoder structure introduces additional computational overhead, limiting real-time applicability.

## Related Work & Insights

- **vs. CLIP-IQA/MANIQA**: These models output only global scores, lack spatial awareness, and are susceptible to reward hacking.
- **vs. Large-scale IQA (e.g., Q-Align)**: Strong semantic awareness but prohibitively high computational cost for iterative training.
- **vs. NPN (null-space methods)**: A different direction—NPN operates at the null-space level, whereas FinPercep-RM operates at the reward signal level.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic diagnosis of reward hacking in ISR-RLHF, with a dual solution combining architectural design and curriculum learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six SR baselines, multiple datasets, training curve visualizations, and user studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem motivation and rich figures.
- **Value**: ⭐⭐⭐⭐⭐ Significant implications for the application of RLHF in visual generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](../../NeurIPS2025/image_restoration/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)

</div>

<!-- RELATED:END -->
