---
title: >-
  [Paper Note] Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning
description: >-
  [ACL 2026][Image Restoration][Test-Time Reinforcement Learning] This paper systematically analyzes sources and amplification mechanisms of spurious signals in test-time RL (TTRL) — mid-frequency answers constitute the ambiguous zone as the primary noise source, while GRPO's within-group normalization amplifies these spurious signals — and proposes DDRL with balanced sampling, fixed advantage values, and consensus offline refinement, achieving 15.3% relative improvement on Qwen2.5-Math-1.5B.
tags:
  - ACL 2026
  - Image Restoration
  - Test-Time Reinforcement Learning
  - Pseudo-Label Noise
  - GRPO Bias
  - Denoising Debiasing
  - Math Reasoning
content_hash: 8dd256707db895b9
---

# Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning

**Conference**: ACL 2026
**arXiv**: [2604.21327](https://arxiv.org/abs/2604.21327)
**Code**: [https://github.com/yuyongcan/DDRL](https://github.com/yuyongcan/DDRL)
**Area**: Image Restoration
**Keywords**: Test-Time Reinforcement Learning, Pseudo-Label Noise, GRPO Bias, Denoising Debiasing, Math Reasoning

## TL;DR
This paper systematically analyzes sources and amplification mechanisms of spurious signals in test-time RL (TTRL) — mid-frequency answers constitute the ambiguous zone as the primary noise source, while GRPO's within-group normalization amplifies these spurious signals — and proposes DDRL with balanced sampling, fixed advantage values, and consensus offline refinement, achieving 15.3% relative improvement on Qwen2.5-Math-1.5B.

## Background & Motivation

**Key Challenge**: (1) Source level — the relationship between answer frequency and reliability is nonlinear: high-frequency answers are mostly correct, low-frequency mostly wrong, mid-frequency highly ambiguous; (2) Amplification level — GRPO's within-group normalization assigns extreme advantage values when positive samples are scarce. In supervised RL this is reasonable, but in TTRL, few positive samples indicate low consensus/high uncertainty.

## Method

### Key Designs

1. **Balanced Confidence Sampling**: Positive samples select top-$K^+$ highest frequency pseudo-label matches (capped at $\lfloor K/2 \rfloor$); negative samples select $K^-$ lowest frequency samples; mid-frequency ambiguous zone is entirely discarded.

2. **Debiased Advantage Estimation**: Fixes advantage values at $A_i = +1$ (positive) or $A_i = -1$ (negative), eliminating the "fewer positives → larger advantage → most unreliable samples get maximum weight" vicious cycle.

3. **Consensus Offline Refinement**: Post-RL SFT refinement using rejection-sampled datasets with high-consensus answers.

## Key Experimental Results

| Model/Method | AIME2024 | MATH-500 | Relative Gain |
|-------------|---------|---------|---------------|
| Qwen2.5-Math-1.5B + TTRL | 15.8 | 73.0 | - |
| Qwen2.5-Math-1.5B + DDRL | **18.2** | **84.2** | **+15.3%** |

### Key Findings
- Removing GRPO normalization alone raises AIME2024 from 15.8% to 20.6%
- All three DDRL components contribute independent gains and are stackable
- Consistent improvement across three different LLM scales

## Highlights & Insights
- The "frequency-reliability" analysis is thorough, clearly locating spurious signal sources in the mid-frequency zone
- Reveals that "reasonable assumptions in supervised RL are violated in unsupervised TTRL" — insightful for all GRPO-based unsupervised methods
- Fixed $+1/-1$ advantage values outperform complex normalization, embodying "simplicity is more robust in noisy environments"

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)
- [\[CVPR 2026\] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](../../CVPR2026/image_restoration/blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](../../ICCV2025/image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/image_restoration/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)

<!-- RELATED:END -->
