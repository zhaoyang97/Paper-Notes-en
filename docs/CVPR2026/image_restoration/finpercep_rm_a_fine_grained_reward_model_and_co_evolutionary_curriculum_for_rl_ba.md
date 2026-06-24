---
title: >-
  [Paper Note] FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][Image Super-Resolution] This work proposes a fine-grained perceptual reward model, FinPercep-RM, and a Co-evolutionary Curriculum Learning (CCL) strategy to address reward hacking and training instability in RLHF-based real-world super-resolution. It achieves local defect awareness by simultaneously outputting global quality scores and spatial degradation heatmaps.
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Image Super-Resolution"
  - "Reward Model"
  - "RLHF"
  - "Fine-grained Quality Assessment"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: f70d982e1dc1d366
---

# FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution

**Conference**: CVPR 2026  
**arXiv**: [2512.22647](https://arxiv.org/abs/2512.22647)  
**Code**: [https://github.com/lyd-2022/FinPercep-RM](https://github.com/lyd-2022/FinPercep-RM)  
**Area**: Image Restoration / Super-Resolution  
**Keywords**: Image Super-Resolution, Reward Model, RLHF, Fine-grained Quality Assessment, Curriculum Learning

## TL;DR

This work proposes a fine-grained perceptual reward model, FinPercep-RM, and a Co-evolutionary Curriculum Learning (CCL) strategy to address reward hacking and training instability in RLHF-based real-world super-resolution. It achieves local defect awareness by simultaneously outputting global quality scores and spatial degradation heatmaps.

## Background & Motivation

Real-world image super-resolution (Real-ISR) based on diffusion models has made significant progress by leveraging the generative priors of large-scale T2I models to synthesize rich textures. As a successful optimization paradigm in the T2I domain, RLHF has naturally been transferred to ISR tasks—using Image Quality Assessment (IQA) models as reward signals to guide SR models.

However, existing IQA models (e.g., CLIP-IQA, MANIQA) only output a single global score and are **insensitive to local fine-grained distortions**. An image with obvious local artifacts might receive a high score similar to the original. This leads to severe **Reward Hacking**: the generator learns to "cheat" imperfect reward signals, converging to results with high global scores but filled with local artifacts and an "oil-painting" appearance.

On the other hand, directly employing more complex fine-grained reward models can trigger **training instability**. High-variance spatial penalty signals cause policy gradient oscillations and convergence failure. This creates a **stability-robustness dilemma**: simple global rewards are stable but easily hacked, while complex fine-grained rewards are robust but unstable.

Core Insight: **A good reward model should not only evaluate "What" the quality is but also diagnose "Where" the defects are.** Simultaneously, curriculum learning should be used to gradually introduce fine-grained rewards from simple to complex.

## Method

### Overall Architecture

This paper addresses the issue where, when applying RLHF to Real-ISR, the reward signal granularity is too coarse, allowing the generator to exploit loopholes to boost global scores while leaving numerous local artifacts. The entire system consists of three components: an SR generator that leverages T2I priors for image production; a FinPercep-RM reward model that both provides a global quality score and utilizes a decoder to output a fine-grained perceptual degradation map (fg-PDM) at the same resolution as the input to pinpoint "where it is broken"; and an external CCL co-evolutionary curriculum that progressively upgrades the reward model and generator from "simple global rewards" to "strict rewards with spatial heatmaps." The first two components address "whether the reward sees finely enough," while the third ensures "training stability after seeing finely."

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["FGR-30k Dataset<br/>Defects implanted via region swapping to obtain pixel-level GT degradation maps"] -->|Supervised Training| B
    C["SR Generator<br/>Leverages T2I priors for image generation"] -->|Generated Results| B
    subgraph B["FinPercep-RM Encoder-Decoder Reward Model"]
        direction TB
        B1["CLIP-IQA Encoder<br/>Multi-scale features"] --> B2["Decoder<br/>Fine-grained degradation map (fg-PDM)"]
        B2 --> B3["Degradation map modulates deep features<br/>followed by MLP for global score"]
    end
    B -->|Global score + degradation map as reward| C
    B -.->|Co-evolutionary Curriculum (CCL)| D["Progressive upgrade from RM_0 to RM_k<br/>Generator synchronizes reward switching from easy to hard"]
    D -.->|Stable Optimization| C
```

### Key Designs

**1. Encoder-Decoder Reward Model: Architecturally Coupling Global Scores and Local Defects**

The global scores of traditional IQA (CLIP-IQA, MANIQA) are directly regressed from the deepest features, completely decoupled from where the image is damaged—this is the root of reward hacking. FinPercep-RM ensures that global scores "must have seen" spatial defects. The encoder (CLIP-IQA backbone) extracts multi-scale features $\{f_i\}_{i=1}^N$, and the decoder performs upsampling and cross-layer fusion to reconstruct a perceptual degradation map $M_{fg\_pdm} \in [0, 1]$ (Sigmoid normalized, where higher values represent heavier degradation). Crucially, the global score is not directly regressed from $f_N$, but rather by using the degradation map to modulate the deepest features before passing through an MLP:

$$S_{fgc\text{-}global} = \mathrm{MLP}\big(f_N \odot \mathrm{interpolate}(M_{fg\text{-}pdm})\big)$$

The degradation map acts as a spatial gate multiplied into the global features; defect regions are explicitly dimmed or weighted, forcing the global score to "depend" on local diagnostic results. Consequently, the generator can no longer improve global scores without fixing local artifacts at the architectural level.

**2. FGR-30k Dataset: Controllable "Implanting" of Local Defects via Region Swapping**

Training a decoder capable of locating defects requires supervision with spatial annotations. However, existing IQA and preference datasets only provide total scores without "where it is bad" labels. FGR-30k uses a synthetic pipeline for supervision: high-quality images $I_{GT}$ are degraded into $I_{LR}$, and various $I_{SR}$ images are generated using multiple SR models. Then, region swapping is used to extract real artifact regions from $I_{SR}$ (identified via random and SAM semantic masks) and paste them back into $I_{GT}$, resulting in a synthetic sample $I_{syn}$ that is perfect except for the implanted regions. Since both the clean base and the implanted regions are known, the ground-truth degradation map $M_{gt}$ can be accurately calculated—it merges pixel-level L1 distance and DINOv3 feature-level cosine distance to capture both low-level texture differences and high-level semantic anomalies. Compared to manual annotation of real artifacts, this "controllable implantation" is cost-effective and provides pixel-level precise supervision.

**3. Co-evolutionary Curriculum Learning (CCL): Reconciling "Fine Perception" with "Stable Training"**

Directly using the full FinPercep-RM as a reward signal causes significant issues—high-variance spatial penalty signals lead to severe policy gradient oscillations and non-convergence. CCL circumvents this via a dual-path co-evolution. One path is the progressive expansion of the reward model: starting from a simple model $RM_0$ that only provides global scores, it is trained on FGR-30k in stages, gradually introducing decoder parameters until it becomes the full model $RM_k$ with heatmaps. The other path is the generator's curriculum co-evolution: during early stages, the generator is trained with the smooth $RM_0$ until stable convergence is reached. After obtaining a good initialization, it is switched in stages to increasingly strict $RM_k$. This transition from easy to hard allows the generator to establish a foothold before undergoing fine-grained local scrutiny, avoiding early gradient oscillations while benefiting from robust spatial supervision in later stages.

### Mechanism: How the CCL Curriculum Unfolds

Consider a complete training run to experience the "coordination" of CCL. In Stage 0, the reward model has only the encoder branch $RM_0$, acting as a standard global IQA. The generator trains smoothly under its guidance, with the reward curve rising monotonically and converging to an initial policy. Entering the middle stages, the decoder branch is gradually activated and learns to output degradation maps on FGR-30k. The reward model upgrades to $RM_1, RM_2, \dots$; with each upgrade, the global score is increasingly modulated by the local degradation map, becoming more sensitive to local artifacts. The generator switches to the new reward synchronously. Because a stable initialization already exists, it can digest stricter spatial penalties without collapsing. In the final stage, the full $RM_k$ is online, monitoring both global perception and pixel-wise artifacts. The generator is forced to eliminate both residual texture artifacts and the "oil-painting" look. Compared to a version using the full reward from the start—where curves oscillate violently—the smooth path of CCL from $RM_0$ to $RM_k$ is the source of stability.

### Loss & Training

The training objective for FinPercep-RM consists of three terms:

$$\mathcal{L}_{total} = \lambda_{map}\,\mathcal{L}_{map} + \lambda_{rank}\,\mathcal{L}_{rank} + \lambda_{align}\,\mathcal{L}_{align}$$

where $\mathcal{L}_{map}$ is the L1 loss between the predicted degradation map and the ground truth $M_{gt}$, supervising "where it is bad"; $\mathcal{L}_{rank}$ is the triplet ranking loss, enforcing $S_{SR} < S_{syn} < S_{GT}$ to align global scores with quality order; and $\mathcal{L}_{align}$ is the anchor alignment loss, anchoring scores of different stages to a unified baseline to prevent reward distribution drift and ensure consistent reward scales for the generator. The RL side follows the standard RLHF pipeline, simply replacing the reward with the current $RM_k$ in the co-evolutionary process.

## Key Experimental Results

### Main Results

**DRealSR Dataset, applied to different SR baselines**

| Baseline | Configuration | LPIPS↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ |
|:---|:---|:---|:---|:---|:---|
| SUPIR | Baseline | 0.452 | 65.665 | 0.629 | 0.572 |
| SUPIR | + Standard IQA Reward | 0.465 | 64.892 | 0.612 | 0.589 |
| SUPIR | **+ Ours (FinPercep-RM)** | **0.428** | **67.234** | **0.648** | **0.586** |
| DreamClear | Baseline | 0.317 | 65.077 | 0.605 | 0.543 |
| DreamClear | + Standard IQA Reward | 0.332 | 64.123 | 0.591 | 0.567 |
| DreamClear | **+ Ours (FinPercep-RM)** | **0.295** | **67.891** | **0.632** | **0.561** |

### Ablation Study

| Configuration | Training Stability | Reward Hacking | Final Quality | Description |
|:---|:---|:---|:---|:---|
| Standard IQA Reward | ✅ Stable | ❌ Severe | Medium | Visible local artifacts |
| FinPercep-RM (w/o CCL) | ❌ Oscillating | ✅ Mitigated | Poor (No convergence) | Unstable when used directly |
| **FinPercep-RM + CCL** | **✅ Stable** | **✅ Mitigated** | **Optimal** | Complete method |

### Key Findings

- Standard IQA rewards lead to an increase in LPIPS (distortion) while MUSIQ scores decrease—a typical case of reward hacking: the model learns to please the IQA while actual quality degrades.
- FinPercep-RM consistently improves performance across all SR baselines (including ResShift, SUPIR, DreamClear, DiffBIR, SeeSR, DIT4SR).
- CCL training curves are smooth and converge to higher reward values, whereas the curves without CCL oscillate violently.
- Degradation map visualizations show that FinPercep-RM can accurately locate texture artifact regions.

## Highlights & Insights

- **Attributing reward hacking to insufficient perceptual granularity** is a precise diagnosis: global scores indeed cannot distinguish between "overall good but locally poor" and "overall good and locally good."
- **Architectural coupling of global-local** design is clever: using degradation maps to modulate global features before score regression makes the two inseparable at the architectural level.
- **CCL (Curriculum Learning)** addresses a universal RLHF problem—complex reward signals leading to training instability—and is transferable to other RLHF applications.

## Limitations & Future Work

- The ground-truth degradation maps in FGR-30k rely on a heuristic fusion of pixel and feature differences, which may not perfectly reflect human perception.
- The stage division and switching timing in CCL require manual adjustment; an adaptive curriculum would be more ideal.
- Validated only on ISR tasks; whether it applies to other image generation RLHF (e.g., T2I) remains to be explored.
- The encoder-decoder structure increases computational overhead, limiting real-time performance.

## Related Work & Insights

- **vs CLIP-IQA/MANIQA**: These only provide global scores and lack spatial awareness, making them susceptible to reward hacking.
- **vs Large-scale IQA (e.g., Q-Align)**: While semantic awareness is good, the computational cost is too high for iterative RL training.
- **vs NPN (Null-space project method)**: Different directions—NPN works at the inverse problem space level, while FinPercep-RM works at the reward signal level.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically diagnose reward hacking in ISR-RLHF and propose a dual architecture-curriculum solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 SR baselines, multiple datasets, training curve visualization, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with rich diagrams.
- Value: ⭐⭐⭐⭐⭐ Provides significant insights for the application of RLHF in visual generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DNF-SR: Dual-Input and Negative-Aware Feature Fine-Tuning for Real-World Image Super-Resolution](dnf-sr_dual-input_and_negative-aware_feature_fine-tuning_for_real-world_image_su.md)
- [\[CVPR 2026\] TextOVSR: Text-Guided Real-World Opera Video Super-Resolution](textovsr_text-guided_real-world_opera_video_super-resolution.md)
- [\[CVPR 2026\] One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution](one-step_diffusion_transformer_for_controllable_real-world_image_super-resolutio.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution](time-aware_one_step_diffusion_network_for_real-world_image_super-resolution.md)

</div>

<!-- RELATED:END -->
