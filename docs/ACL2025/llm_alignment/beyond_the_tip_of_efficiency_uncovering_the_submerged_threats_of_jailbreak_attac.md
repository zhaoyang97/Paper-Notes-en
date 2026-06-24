---
title: >-
  [Paper Note] Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models
description: >-
  [ACL 2025][LLM Alignment][Small Language Models] This paper systematically evaluates the safety of 13 SOTA small language models (<4B parameters) under 5 jailbreak attacks, finding that while SLMs can resist direct attacks, they are significantly more vulnerable under jailbreak attacks than large language models. It further analyzes the impact of SLM practices such as structural compression, quantization, and knowledge distillation on safety.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Small Language Models"
  - "Jailbreak Attacks"
  - "SLM Safety"
  - "Model Compression"
  - "Safety Evaluation"
date: 2026-05-08
content_hash: c27cb44a3322bd53
---

# Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.19883](https://arxiv.org/abs/2502.19883)  
**Code**: None  
**Area**: LLM Alignment / AI Safety  
**Keywords**: Small Language Models, Jailbreak Attacks, SLM Safety, Model Compression, Safety Evaluation

## TL;DR
This paper systematically evaluates the safety of 13 SOTA small language models (<4B parameters) under 5 jailbreak attacks, finding that while SLMs can resist direct attacks, they are significantly more vulnerable under jailbreak attacks than large language models. It further analyzes the impact of SLM practices such as structural compression, quantization, and knowledge distillation on safety.

## Background & Motivation
**Background**: SLMs (<4B parameters) are increasingly popular due to low costs and suitability for edge deployment (e.g., Phi-3-mini, Gemma-2B, etc.), but safety research is heavily concentrated on LLMs of 7B+ parameters.

**Limitations of Prior Work**: The safety of SLMs has barely been systematically studied—does having fewer parameters imply poorer safety alignment? Do SLM-specific techniques (architectural compression, quantization, distillation) introduce safety vulnerabilities?

**Key Challenge**: The pursuit of efficiency in SLMs may sacrifice safety—alignment training on parameter-constrained models might be weaker than on large models.

**Goal**: Comprehensively evaluate the safety of SLMs under jailbreak attacks, revealing the hidden trade-offs between efficiency and safety.

**Key Insight**: A comparison across 13 SLMs + 3 LLMs, using 5 attack methods, 5 attack datasets, and 3 defense methods.

**Core Idea**: SLMs face much more severe jailbreak safety threats than LLMs; beneath the "iceberg" of efficiency lurk structural safety risks.

## Method

### Overall Architecture
Evaluation Matrix: 13 SLMs $\times$ 5 Attacks $\times$ 5 Datasets $\times$ 3 Defenses $\rightarrow$ ASR Analysis

### Key Designs

1. **Model Selection**:

    - SLMs (<4B): 13 models including Phi-3.5-mini, Gemma-2-2B-it, Qwen2-1.5B, TinyLlama, MobileLlama, StableLM, LiteLlama, etc.
    - LLMs (>7B): Phi-3.5-MoE, Gemma-2-9B-it, Qwen2-7B as baselines.

2. **Attack Methods**: GCG (gradient-based suffix), ArtPrompt (ASCII art encoding), DeepInception (nested scenarios), AutoDAN (automated jailbreak), Multilingual Attack (multilingual translation).

3. **Defense Methods**: Llama-Guard-3 (input filtering), SmoothLLM (perturbation smoothing), Backtranslation (backtranslation detection).

4. **Safety Analysis of SLM Techniques**: Architectural compression (parameter pruning), quantization (INT4/INT8), knowledge distillation.

## Key Experimental Results

### Main Results

| Model Type | Direct Attack ASR | Jailbreak Attack ASR | Gap |
|---------|------------|------------|------|
| LLM (>7B) | ~10% | ~20-40% | Moderate |
| SLM (Strong) | ~10% | ~40-60% | Large |
| SLM (Weak, e.g. TinyLlama) | ~72% | ~72-80% | Extremely High |

### Ablation Study: Effectiveness of Defenses on SLMs

| Defense Method | SLM ASR Reduction | Effective? |
|---------|---------------|---------|
| Llama-Guard-3 | Down to ~0% | Highly effective (but requires an extra 8B model) |
| SmoothLLM | Moderate reduction | Partially effective |
| Backtranslation | Low reduction | Limited effect |

### Key Findings
- **SLMs perform well under direct attacks (~10% ASR), but are extremely vulnerable under jailbreak attacks**—GCG achieves 72-80% ASR on some SLMs.
- Parameter size is **weakly correlated** with safety—different models with the same parameter scale vary greatly in safety.
- Quantization has little impact on safety—safety knowledge does not seem to concentrate in the quantized precision parts.
- Knowledge distillation may weaken safety alignment—distillation mainly transfers capabilities rather than safety.
- Llama-Guard-3 is the most effective as an external filter, but adds the overhead of an 8B model—canceling out the efficiency benefits of SLMs.

## Highlights & Insights
- **Precise "Efficiency Iceberg" metaphor**: The superficial efficiency advantages of SLMs hide severe safety vulnerabilities underneath.
- **First systematic safety evaluation of SLMs**: 13 models $\times$ 5 attacks $\times$ 5 datasets, offering comprehensive coverage.
- **Analysis of the safety impact of SLM-specific practices** is a unique contribution—revealing the safety costs of compression/quantization/distillation.

## Limitations & Future Work
- Only text attacks are evaluated, safety of multimodal SLMs is not considered.
- No new SLM-specific defense method is proposed—only existing defenses are evaluated.
- The best practice for safety alignment training on SLMs is not analyzed.
- Some SLMs (e.g., TinyLlama) are insecure even against direct attacks (72% ASR), likely due to insufficient alignment training rather than the limitation of SLM itself.
- Future work can explore how to efficiently inject safety alignment during SLM training.

## Related Work & Insights
- **vs LLM safety research**: Extensive jailbreak research targets 7B+ models, while this work is the first to systematically focus on <4B models.
- **vs Llama-Guard**: Llama-Guard is highly effective as an external filter for SLMs but introduces the overhead of a larger model—signaling the need for lightweight safety filters.
- Insight: Safety evaluation is mandatory before deploying SLMs; one cannot assume small models are naturally safe.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic safety evaluation of SLMs, filling an important research gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13 models $\times$ 5 attacks $\times$ 5 datasets $\times$ 3 defenses, extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and a vivid "iceberg" metaphor.
- Value: ⭐⭐⭐⭐ Provides direct warning signals for the safe deployment of SLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[ACL 2025\] Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space](breaking_the_ceiling_exploring_the_potential_of_jailbreak_attacks_through_expand.md)

</div>

<!-- RELATED:END -->
