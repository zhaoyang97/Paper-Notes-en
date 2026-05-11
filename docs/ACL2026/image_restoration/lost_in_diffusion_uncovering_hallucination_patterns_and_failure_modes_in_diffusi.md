---
title: >-
  [Paper Note] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models
description: >-
  [ACL 2026][Image Restoration][Diffusion Language Model] This work presents the first systematic comparison of hallucination patterns between diffusion LLMs (dLLMs) and their autoregressive (AR) counterparts…
tags:
  - "ACL 2026"
  - "Image Restoration"
  - "Diffusion Language Model"
  - "Hallucination"
  - "Non-Autoregressive Generation"
  - "Failure Modes"
  - "Test-Time Compute"
content_hash: 1f12ec24a474522b
---

# Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.10556](https://arxiv.org/abs/2604.10556)
**Code**: [github.com/ZeroLoss-Lab/Lost-in-Diffusion](https://github.com/ZeroLoss-Lab/Lost-in-Diffusion)
**Area**: Image Restoration
**Keywords**: Diffusion Language Model, Hallucination, Non-Autoregressive Generation, Failure Modes, Test-Time Compute

## TL;DR
This work presents the first systematic comparison of hallucination patterns between diffusion LLMs (dLLMs) and their autoregressive (AR) counterparts, revealing that current dLLMs exhibit higher hallucination tendency and identifying three diffusion-specific failure modes: premature termination, incomplete denoising, and context intrusion.

## Background & Motivation

**Background**: dLLMs as a non-autoregressive generation paradigm are rapidly emerging, with LLaDA, Dream, and SDAR achieving comparable performance to AR-LLMs on general benchmarks. Theoretically, dLLMs' global planning and bidirectional visibility could mitigate the "snowball effect" and "reversal curse" found in AR models.

**Key Challenge**: dLLMs' global context planning theoretically should reduce hallucination (enabling backtracking), but diffusion's inherent noise may exacerbate it — which effect dominates lacks empirical evidence.

**Key Insight**: Design two carefully controlled paired comparisons — (I) architecture-aligned (LLaDA-8B vs LLaMA-3-8B) and (II) parameter-aligned (Dream-7B vs Qwen2.5-7B, Dream initialized directly from Qwen weights) — to maximally isolate the generation mechanism's impact.

## Method

### Key Designs

1. **Paired Comparison Framework**: Group I for architecture alignment; Group II for parameter alignment (Dream directly initialized from Qwen weights, so any hallucination differences are primarily attributable to the diffusion generation process). Preferentially uses pretrained (non-instruction-tuned) checkpoints.

2. **Canonical Diffusion Inference Settings**: Denoising steps $T$ set equal to sequence length $L$ ($T=L$), temperature 0 for reproducibility.

3. **Inference-Time Compute Dynamic Analysis**: Evaluates different denoising steps $T \in \{128, 256, 512, 1024\}$, revealing divergent behaviors: LLaDA shows early saturation while Dream demonstrates positive scaling.

## Key Experimental Results

### Main Results

| Model | PreciseWikiQA HR ↓ | PreciseWikiQA CR ↑ | LongWiki F1@32 ↑ | NonExistRefusal FA ↓ |
|-------|-------------------|-------------------|------------------|---------------------|
| LLaMA-3-8B (AR) | 85.94 | 10.30 | 0.306 | 73.35 |
| LLaDA-8B (dLLM) | **95.13** | **3.92** | 0.272 | **87.10** |
| Qwen2.5-7B (AR) | 89.06 | 9.06 | **0.387** | 94.05 |
| Dream-7B (dLLM) | 92.54 | 6.04 | 0.340 | **98.50** |

### Key Findings
- dLLMs consistently underperform AR counterparts across all three tasks
- Inference-time compute dynamics diverge: LLaDA shows early saturation (~0.27 F1 across all steps) while Dream monotonically increases from 128 to 1024 steps
- Three diffusion-specific failure modes: premature termination (18%), incomplete denoising (60%), and context intrusion (38-58%)

## Highlights & Insights
- First systematic quantification of dLLM hallucination, filling a critical research gap
- The paired comparison framework design is rigorous, especially the Dream-Qwen pairing providing near-ideal controlled experiments
- The failure mode taxonomy provides valuable vocabulary for understanding dLLM generation behavior

## Limitations & Future Work
- Cannot fully isolate generation paradigm effects — even minimal diffusion adaptation requires weight updates
- Uses canonical diffusion settings ($T=L$); practical acceleration methods may alter hallucination characteristics
- Future direction: dynamic sequence editing capabilities (insertion, deletion, re-masking) are key to dLLMs realizing their full potential

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/image_restoration/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[ACL 2026\] Diffusion-CAM: Faithful Visual Explanations for dMLLMs](diffusion-cam_faithful_visual_explanations_for_dmllms.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](../../AAAI2026/image_restoration/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)

</div>

<!-- RELATED:END -->
