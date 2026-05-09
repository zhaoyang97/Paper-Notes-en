---
title: >-
  [Paper Note] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
description: >-
  [ACL 2026][Image Restoration][Retrieval-Augmented Generation] EviOmni learns to extract rational evidence from retrieved documents through a "reason-then-extract" paradigm: integrating evidence reasoning and extraction into a unified trajectory, using knowledge token masking to prevent information leakage, and optimizing via GRPO with verifiable rewards, achieving accuracy surpassing full-text retrieval at ~38x compression across 5 benchmarks.
tags:
  - ACL 2026
  - Image Restoration
  - Retrieval-Augmented Generation
  - Evidence Extraction
  - Reinforcement Learning
  - Reason-Guided Extraction
  - GRPO
content_hash: 9f16aa415f52dbad
---

# Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation

**Conference**: ACL 2026
**arXiv**: [2507.15586](https://arxiv.org/abs/2507.15586)
**Code**: [GitHub](https://github.com/HITsz-TMG/EviOmni)
**Area**: Image Restoration
**Keywords**: Retrieval-Augmented Generation, Evidence Extraction, Reinforcement Learning, Reason-Guided Extraction, GRPO

## TL;DR
EviOmni learns to extract rational evidence from retrieved documents through a "reason-then-extract" paradigm: integrating evidence reasoning and extraction into a unified trajectory, using knowledge token masking to prevent information leakage, and optimizing via GRPO with verifiable rewards, achieving accuracy surpassing full-text retrieval at ~38x compression across 5 benchmarks.

## Background & Motivation

**State of the Field**: RAG enhances LLM accuracy by retrieving external passages, but retrieved passages often contain noise and irrelevant content requiring evidence extraction/denoising.

**Root Cause**: Standard evidence extraction follows a "see-and-extract" approach without deep reasoning over retrieved content — when key clues require cross-passage reasoning, direct extraction misses them.

**Core Idea**: Unify evidence reasoning `<reason>` and evidence extraction `<extract>` into a single generation trajectory, with knowledge token masking for separate evaluation, optimized end-to-end via GRPO with three verifiable rewards (answer, length, format).

## Method

### Key Designs

1. **Rational Evidence Extraction Paradigm**: The model first generates `<reason>` (analyzing each passage's relevance and clues), then generates `<extract>` (concise evidence summary). Reasoning identifies scattered clues, excludes misleading information, and correlates cross-passage information.

2. **Knowledge Token Masking**: Separates evaluation of reasoning and evidence quality during training. Hard masking (replacing input tokens) prevents information leakage from causal attention.

3. **Three Verifiable Rewards**: Answer reward $R^{ans}$ = unigram F1; length reward $R^{len}$ encourages comprehensive reasoning and concise evidence; format reward $R^{fmt}$ ensures correct tag formatting.

## Key Experimental Results

### Main Results

| Method | NQ EM | NQ CR | TQA EM | HotpotQA EM |
|--------|-------|-------|--------|-------------|
| Full (no compression) | 41.97 | 1.0x | 57.02 | 19.20 |
| FilCo | 36.62 | 16.3x | 54.06 | 18.18 |
| **EviOmni** | **41.14** | **38.1x** | **56.84** | **20.46** |

### Key Findings
- Rational evidence answer recall exceeds vanilla evidence by 4-7 percentage points
- Near full-text performance at 38x compression, indicating highly refined extraction
- Generalizes to OOD datasets (HotpotQA)

## Highlights & Insights
- The "reason-then-extract" paradigm shift has broad applicability beyond RAG
- Knowledge token masking elegantly solves the training information leakage challenge
- 38x compression without performance loss has major practical significance for inference efficiency

## Limitations & Future Work
- Reasoning increases generation length despite shorter evidence
- Validated only on QA tasks
- Reasoning quality limited by base model capability

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning](understanding_and_mitigating_spurious_signal_amplification_in_test-time_reinforc.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](../../CVPR2026/image_restoration/diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](../../CVPR2026/image_restoration/learning_to_translate_noise_for_robust_image_denoising.md)
- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](../../ICLR2026/image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)

<!-- RELATED:END -->
