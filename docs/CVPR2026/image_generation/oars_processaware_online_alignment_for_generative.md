---
title: >-
  [Paper Note] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution
description: >-
  [CVPR 2026][Image Generation][Real-World Super-Resolution] This paper proposes OARS, a framework that aligns generative real-world image super-resolution models with human visual preferences via COMPASS—an MLLM-based process-aware reward model—and a progressive online reinforcement learning pipeline, achieving adaptive balance between perceptual quality and fidelity.
tags:
  - CVPR 2026
  - Image Generation
  - Real-World Super-Resolution
  - Online RL
  - MLLM Reward
  - Process-Aware
  - Perception-Fidelity Trade-off
date: 2026-05-08
content_hash: 901405937c651f40
---

# OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2603.12811](https://arxiv.org/abs/2603.12811)
**Code**: Unavailable (as of March 2026)
**Area**: Image Generation
**Keywords**: Real-World Super-Resolution, Online RL, MLLM Reward, Process-Aware, Perception-Fidelity Trade-off

## TL;DR
This paper proposes OARS, a framework that aligns generative real-world image super-resolution models with human visual preferences via COMPASS—an MLLM-based process-aware reward model—and a progressive online reinforcement learning pipeline, achieving adaptive balance between perceptual quality and fidelity.

## Background & Motivation
- **State of the Field**: Real-world image super-resolution (Real-ISR) must handle complex and unknown degradations. Early CNN-based methods (L1/L2 losses) produce over-smoothed textures, while diffusion models improve perceptual quality but generalize poorly under standard SFT on unseen degradations, lacking direct optimization mechanisms aligned with human aesthetic preferences.
- **Limitations of Prior Work**: Existing IQA metrics as RL rewards suffer fundamental drawbacks: full-reference (FR) metrics are inapplicable in real-world scenarios without ground truth, while no-reference (NR) metrics lack fine-grained sensitivity to distinguish subtle differences among generative SR outputs. Naively combining FR and NR metrics linearly ignores the varying degrees of input degradation. Offline RL methods (e.g., DPO) face a "pseudo-diversity" problem: SR outputs sampled via different noise seeds collapse into random texture hallucinations rather than genuine structural diversity under strong spatial constraints, limiting preference alignment.
- **Root Cause**: The absence of a process-aware, quality-adaptive reward model and an online exploration strategy capable of overcoming the pseudo-diversity bottleneck.

## Core Problem
How to design a process-aware, quality-adaptive reward model and an online exploration strategy that breaks the pseudo-diversity bottleneck, enabling effective alignment of generative Real-ISR models with human visual preferences.

## Method

### Overall Architecture
OARS consists of two core components: (1) the **COMPASS** reward model—an MLLM-based process-aware scorer that evaluates fidelity preservation and perceptual gain during LR→SR conversion; and (2) a **progressive online RL framework**—comprising cold-start SFT, full-reference RL, and no-reference RL stages, with shallow LoRA optimization on the base model to enable on-policy exploration.

### Key Designs

1. **COMPASS-20K Dataset and Three-Stage Annotation Pipeline**:

    - Constructs 2,400 input images (800 synthetic DIV2K LR + 1,600 real LQ images), processed by 12 SR algorithms to yield 28,800 LR-SR pairs.
    - **Fidelity annotation**: On the synthetic subset, DISTS(SR, GT) distances are normalized to $[0, 1]$.
    - **Perceptual quality gain annotation (three stages)**:
        - Stage 1 — Global anchor scoring: Q-Insight independently scores LR and SR images to obtain globally comparable quality scores.
        - Stage 2 — Intra-group ranking: A dedicated pairwise comparison model (trained on DiffIQA) exhaustively compares all SR outputs for the same LR input and aggregates results into a continuous ranking score $r \in [0, 1]$.
        - Stage 3 — Ranking-guided calibration: Linear calibration (via least-squares estimation of $\alpha^*$ and $\beta^*$) aligns intra-group rankings with the global scale.
    - Qwen3-VL-32B generates explanatory text descriptions; manual inspection removes conflicting samples in the top/bottom 5%.

2. **COMPASS Reward Function — Input Quality-Adaptive Mechanism**:

    - Full-parameter SFT on QwenVL-8B jointly predicts fidelity $F$, input quality $Q_{LR}$, and output quality $Q_{SR}$.
    - Reward formula: $R = F \cdot Q_{LR} + F^{Q_{LR}/\gamma} \cdot \Delta Q$, where $\Delta Q = Q_{SR} - Q_{LR}$ and $\gamma = 7$.
    - The first term $F \cdot Q_{LR}$ measures preservation of original quality; the second term uses the exponent $Q_{LR}/\gamma$ to realize adaptive control.
    - High-quality input → large exponent → high sensitivity to fidelity degradation → conservative enhancement encouraged; low-quality input → small exponent → relaxed fidelity constraint → greater perceptual improvement permitted.

3. **Progressive Online RL (Three Stages)**:

    - **Cold-start stage**: Trains on LR-HR paired data with a Flow Matching objective to acquire basic SR capability.
    - **Full-reference RL stage**: On data with available ground truth, consistency rewards are computed directly via DISTS (rather than via reward model prediction) to avoid reward hacking. A key design choice is applying LoRA to the **base model** (rather than the SFT model)—the base model's higher sampling stochasticity facilitates better exploration.
    - **No-reference RL stage**: On real LQ data without ground truth, COMPASS provides the sole reward signal; LoRA fine-tuning continues on the base model.
    - At inference time, the final LoRA parameters $\Delta_{NR}$ are merged into the cold-start SFT model.

4. **Negative Perceptual Objective and Group Filtering**:

    - For each LR input, $K = 24$ candidates are sampled; groups with high mean and low variance (thresholds 0.9 / 0.05) are filtered out after reward computation.
    - Implicit positive/negative policies are defined as linear combinations of the old and current policies.
    - The loss simultaneously trains the model to learn "what to do" from high-reward samples and "what to avoid" from low-reward samples.

### Loss & Training
- Cold-start: Flow Matching loss $\mathcal{L}_{SFT} = \mathbb{E}[\|v - v_\theta(x_t, t | x_{LR}, c)\|^2]$
- RL stages: Negative perceptual objective $\mathcal{L}_{RL} = \mathbb{E}[r\|v_\theta^+ - v\|^2 + (1-r)\|v_\theta^- - v\|^2]$, where $r$ is the normalized and clipped optimal probability.
- LoRA configuration: rank=32, alpha=64; 6-step sampling for training, 40-step for inference; 8×H20 GPUs.

## Key Experimental Results

| Dataset | Metric | OARS | Qwen-SFT | Best Baseline | Gain (vs. Qwen-SFT) |
|---------|--------|------|----------|---------------|----------------------|
| RealSR | LIQE↑ | **4.3045** | 3.8146 | UARE: 4.0658 | +0.49 |
| RealSR | MUSIQ↑ | **71.41** | 68.57 | UARE: 69.67 | +2.84 |
| DIV2K | LIQE↑ | **4.6668** | 4.3404 | UARE: 4.2627 | +0.33 |
| DIV2K | MUSIQ↑ | **74.07** | 72.35 | UARE: 70.45 | +1.72 |
| RealSet80 | LIQE↑ | **4.5465** | 4.1602 | SeeSR: 4.3317 | +0.39 |
| SRIQA-Bench | All-Acc | **83.1%** | — | A-FINE: 82.4% | Best GT-Free |

- OARS achieves consistent top performance on all NR metrics without notable degradation in FR metrics (PSNR/SSIM) relative to Qwen-SFT.
- User study: OARS receives **47.62%** of votes, the highest among all methods (DP2O-SR: 27.68%).

### Ablation Study
- The three-stage annotation calibration improves accuracy from 78.8% to 81.5%; adding explicit fidelity modeling raises it to 82.3%; the quality-adaptive mechanism at $\gamma = 7$ achieves the best 83.1%.
- Applying RL on the base model (vs. the SFT model) is critical: RL on the SFT model causes severe FR degradation (PSNR: 22.71→21.31), while the base model remains stable (22.71→22.36).
- Using only perceptual gain $\Delta Q$ as reward leads to severe reward hacking (PSNR drops to 21.38, producing artifacts with spuriously high NR scores).
- General-purpose rewards (HPSv2, Qwen25-VL) fail to provide effective feedback for SR; RALI improves perceptual gain but causes severe fidelity degradation.
- The NFT-based OARS converges 5–10× faster than Flow-GRPO and achieves superior NR metric performance.

## Highlights & Insights
- **Paradigm shift to process-aware evaluation**: Rather than scoring SR outputs as static results, OARS evaluates the LR→SR transformation process, decoupling fidelity preservation from perceptual gain—a fundamental innovation in the evaluation framework.
- **Input quality-adaptive reward**: The exponential gating mechanism $F^{Q_{LR}/\gamma}$ dynamically modulates the perception-fidelity trade-off based on input degradation level, yielding an elegant and effective design.
- **Shallow LoRA on the base model**: Performing shallow LoRA optimization on the base model (rather than the SFT model) leverages the base model's higher stochasticity for better on-policy exploration while avoiding reward hacking.
- **Three-stage annotation pipeline**: Unifying global comparability with intra-group fine-grained discriminability addresses the key limitation of NR-IQA insensitivity to generative SR outputs.

## Limitations & Future Work
- The MLLM reward model incurs high computational overhead, constraining online RL training efficiency—distillation into a lightweight scorer is a natural next step.
- The framework is limited to image SR and has not been extended to video SR, where temporal consistency poses additional challenges.
- Validation is restricted to 4× SR; generalization to other scale factors and tasks (denoising, deblurring) remains unexplored.
- The 12 SR methods in COMPASS-20K may not cover the full diversity of all generative paradigms.

## Related Work & Insights
- **vs. DP2O-SR (offline DPO)**: OARS overcomes the pseudo-diversity problem of offline sampling via online RL; applying OARS's no-reference RL stage on top of DP2O-SR yields further consistent improvements across all metrics.
- **vs. Flow-GRPO**: Both perform online RL alignment, but OARS employs forward-process RL (DiffusionNFT-style) rather than trajectory-level RL. As a strongly constrained generation task, SR does not require trajectory-level exploration; OARS is 5–10× more training-efficient.
- **vs. Q-Insight / CLIP-IQA+ and other NR-IQA metrics**: These metrics assess the output in isolation and cannot perceive the enhancement process from LR to SR. COMPASS surpasses all FR and NR baselines on SRIQA-Bench through process-aware evaluation combined with the adaptive mechanism.

## Highlights & Insights (General)
- The concept of "process-aware" evaluation generalizes to other image enhancement and editing tasks: rather than asking only whether the result is good, one should ask what was improved and what was preserved relative to the input.
- The strategy of shallow LoRA on the base model for RL warrants validation in other conditional generation tasks.
- The three-stage annotation pipeline (global + intra-group + calibration) constitutes a general methodology for fine-grained preference annotation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Process-aware reward and adaptive gating are core innovations; the progressive RL framework is well-designed, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, nine baselines, comprehensive ablations (reward formula / RL stages / backbone / RL method), and a user study; extremely thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear with well-motivated methodology and concise formulations, though the high information density requires careful re-reading.
- **Value**: ⭐⭐⭐⭐ — Provides a complete RLHF pipeline for generative SR post-training; COMPASS can be independently reused for other low-level vision tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](../../AAAI2026/image_generation/continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[CVPR 2026\] AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution](alignvar_towards_globally_consistent_visual_autoregression_for_image_super-resol.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)

<!-- RELATED:END -->
