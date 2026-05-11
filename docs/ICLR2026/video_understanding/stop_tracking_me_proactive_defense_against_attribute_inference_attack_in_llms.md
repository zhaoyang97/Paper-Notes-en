---
title: >-
  [Paper Note] Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs
description: >-
  [ICLR 2026][Video Understanding][Attribute Inference Attack] TRACE-RPS proposes a unified defense framework against attribute inference attacks in LLMs: TRACE leverages attention mechanisms and reasoning chains to precis…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "Attribute Inference Attack"
  - "Privacy Protection"
  - "LLM Safety"
  - "Attention-based Anonymization"
  - "Optimization-based Defense"
date: 2026-05-08
content_hash: aab55f81d3c1b3dd
---

# Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs

**Conference**: ICLR 2026
**arXiv**: [2602.11528](https://arxiv.org/abs/2602.11528)
**Code**: [https://github.com/Jasper-Yan/TRACE-RPS](https://github.com/Jasper-Yan/TRACE-RPS)
**Area**: Video Understanding
**Keywords**: Attribute Inference Attack, Privacy Protection, LLM Safety, Attention-based Anonymization, Optimization-based Defense

## TL;DR
TRACE-RPS proposes a unified defense framework against attribute inference attacks in LLMs: TRACE leverages attention mechanisms and reasoning chains to precisely locate privacy-leaking text elements for fine-grained anonymization, while RPS employs lightweight suffix optimization to induce model refusal of inference, reducing attribute inference accuracy from ~50% to below 5%.

## Background & Motivation

**Background**: LLMs can infer private attributes (age, location, gender, etc.) from innocuous text shared by users online, enabling large-scale automated privacy violations. Such attacks bypass safety filters since the prompts themselves are entirely benign.

**Limitations of Prior Work**:
   - Existing anonymization methods operate at too coarse a granularity (text-level rather than token-level), failing to precisely identify specific text elements responsible for privacy leakage.
   - A fundamental limitation of anonymization: even after modifying text to conceal sensitive cues, the model's reasoning capability can still infer attributes from the revised text.
   - For attributes with limited categories (e.g., gender or income level), anonymized text still provides interpretable data points.

**Key Challenge**: Attribute inference in LLMs stems from **reasoning capability**, not memorization — weakening reasoning ability would compromise general utility, while anonymization alone cannot prevent inference from bypassing it.

**Key Insight**: A two-stage defense — (1) precise anonymization to reduce information leakage + (2) optimized suffix to induce model refusal, fundamentally blocking inference.

**Core Idea**: Anonymization reduces information exposure + refusal optimization blocks inference behavior = a dual-layer defense.

## Method

### Overall Architecture
A unified defense combining TRACE (fine-grained anonymization) and RPS (refusal-inducing optimization). Before sharing text, users apply TRACE to replace privacy-leaking tokens, then append a suffix via RPS to cause the inference model to decline answering.

### Key Designs

1. **TRACE (Text Revision via Attention and Chain-of-thought Explanation)**:

    - **Function**: Precisely locate and replace text elements that leak private information.
    - **Mechanism**: (1) Extract "privacy tokens" using attention weights — tokens the model focuses on during attribute inference; (2) Generate reasoning chains to reveal the model's inference pathway; (3) Iterative adversarial revision — replace the most privacy-leaking tokens each round until inference fails.
    - **Design Motivation**: More precise than rule-based methods such as Azure PII detection; capable of identifying implicit privacy leakage (e.g., dialectal expressions implying geographic location).

2. **RPS (Refusal-oriented Perturbation Search)**:

    - **Function**: Optimize a suffix to induce the LLM to refuse attribute inference tasks.
    - **Mechanism**: Two-stage lightweight optimization — (1) Initialization: identify the token sequence most likely to elicit "I cannot answer" in logit space; (2) Refinement: local search to maximize refusal probability. Requires white-box logit access.
    - **Design Motivation**: Anonymization only reduces information without blocking inference; RPS fundamentally prevents the model from answering — the two approaches are complementary.

3. **MPS (Misattribution Perturbation Search, alternative strategy)**:

    - **Function**: For highly instruction-following models that are difficult to induce into refusal, guide the model to predict an incorrect attribute value.
    - **Mechanism**: Optimize a suffix to cause the model to predict a wrong attribute rather than refuse.
    - **Design Motivation**: Highly aligned models such as GPT-4o rarely produce refusals; MPS provides an alternative strategy.

### Loss & Training
- RPS optimization objective: $\max_{suffix} \log P_{model}(\text{"I cannot answer"} | P(t \oplus suffix))$
- Two stages: greedy initialization (selecting per-token candidates that maximize refusal probability) + local optimization (token substitution search).
- Requires logit access to open-source models; only TRACE is applied to closed-source models.

## Key Experimental Results

### Main Results (Attribute Inference Accuracy ↓)

| Method | Llama3 | Qwen2.5 | DeepSeek-R1 | GPT-4o |
|--------|--------|---------|-------------|--------|
| No Defense | ~50% | ~50% | ~50% | ~50% |
| Azure PII | ~40% | ~40% | ~40% | ~40% |
| Staab et al. (Anonymization) | ~25% | ~25% | ~25% | ~25% |
| TRACE | ~15% | ~15% | ~15% | ~20% |
| **TRACE-RPS** | **<5%** | **<5%** | **<5%** | N/A (closed-source) |

### Ablation Study

| Configuration | Inference Accuracy ↓ |
|---------------|----------------------|
| TRACE only | ~15% |
| RPS only | ~10% |
| **TRACE + RPS** | **<5%** |

### Key Findings
- **Inference accuracy reduced from 50% to <5%**: TRACE-RPS nearly completely blocks attribute inference on open-source models.
- **Cross-model transferability**: Suffixes optimized on one model remain effective against other models.
- **Robustness to prompt variations**: The defense remains effective even when adversaries alter the inference prompt format.
- **Reasonable utility-privacy trade-off**: Text revised by TRACE preserves semantic integrity and readability.
- **Effective against DeepSeek-R1**: Even models with strong reasoning capabilities can be effectively defended.

## Highlights & Insights
- The **dual-layer design of anonymization + refusal induction** is highly practical — anonymization reduces the information exposure surface while refusal optimization blocks inference behavior. Both defenses are independently effective and stronger in combination.
- **Repurposing jailbreaking optimization techniques for privacy defense** is an elegant inversion — methods such as GCG are designed for attacks, whereas RPS applies the same technical paradigm for defense.
- **Attention-guided privacy token extraction** is substantially more sophisticated than rule-based approaches — it can uncover implicit privacy leakage pathways that are difficult for humans to anticipate.

## Limitations & Future Work
- RPS requires white-box logit access — only TRACE is applicable to closed-source models (e.g., GPT-4o).
- Optimized suffixes may be detectable as anomalous text (though the paper reports minimal impact).
- Evaluation is limited to text-based attribute inference — multimodal inference combining image and text is not considered.
- The MPS (misattribution) strategy may introduce new ethical concerns in certain scenarios.
- Suffix optimization incurs computational cost (lightweight but still requiring multiple forward passes).

## Related Work & Insights
- **vs. Azure PII Detection**: Relies solely on rule-based matching of explicit PII; unable to detect implicit leakage. TRACE identifies implicit leakage via attention weights and reasoning chains.
- **vs. Staab et al. (2025) Anonymization**: Coarse-grained text-level anonymization; TRACE operates with precision at the token level.
- **vs. GCG/Jailbreaking**: Shares the same optimization paradigm, but RPS inversely applies it to induce refusal rather than bypass it.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified framework of anonymization + refusal optimization is creative; the inversion of jailbreaking techniques is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 7 LLMs, cross-model transfer, prompt robustness, and utility-privacy trade-off.
- Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear; the attack-defense relationship is accurately characterized.
- Value: ⭐⭐⭐⭐⭐ Attribute inference is a realistic privacy threat; TRACE-RPS provides a deployable defense solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](../../ICCV2025/video_understanding/aim_adaptive_inference_of_multi_modal_llms_via_token_merging_and_pruning.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](../../CVPR2026/video_understanding/streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)
- [\[ICLR 2026\] AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)

</div>

<!-- RELATED:END -->
