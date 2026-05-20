---
title: >-
  [Paper Note] Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM] This paper proposes REVERSE, the first framework to unify generation adjustment and post-hoc verification within a single VLM. Through hallucination-aware training on 1.3M semi-synthet…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "VLM"
  - "Visual Hallucination"
  - "Self-Correction"
  - "Retrospective Resampling"
  - "Confidence Token"
date: 2026-05-08
content_hash: 3db07b13fc5317c4
---

# Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling

**Conference**: NeurIPS 2025
**arXiv**: [2504.13169](https://arxiv.org/abs/2504.13169)  
**Code**: [GitHub](https://github.com/tsunghan-wu/REVERSE)  
**Area**: Multimodal Large Language Models / Hallucination Mitigation
**Keywords**: VLM, Visual Hallucination, Self-Correction, Retrospective Resampling, Confidence Token

## TL;DR

This paper proposes REVERSE, the first framework to unify generation adjustment and post-hoc verification within a single VLM. Through hallucination-aware training on 1.3M semi-synthetic samples combined with inference-time retrospective resampling, REVERSE enables a VLM to automatically detect and correct hallucinations during generation, achieving a 12% reduction on CHAIR-MSCOCO and a 34% improvement on HaloQuest.

## Background & Motivation

- **Core Problem**: VLMs frequently produce hallucinations in visual understanding (e.g., describing non-existent objects or actions), posing significant risks in safety-critical scenarios such as autonomous driving and assistive technologies.
- **Limitations of Prior Work**: Generation adjustment methods (VCD, OPERA, DoLA, etc.) modify decoding behavior but cannot correct erroneously generated tokens once produced; post-hoc verification methods (Woodpecker, LURE) rely on external models, involve complex pipelines, and tend to reject outputs rather than correct them.
- **Key Challenge**: No existing method can simultaneously generate, verify, and correct within a single model.
- **Key Insight**: Introduce explicit confidence tokens to enable the VLM to self-annotate phrase-level hallucinations, combined with retrospective resampling for runtime self-correction.

## Method

### Confidence Token Design

Three special tokens are added to the VLM vocabulary:
- `<SPAN>`: marks the beginning of a key phrase
- `</CN>`: marks the end of a confident, grounded phrase
- `</UN>`: marks the end of an unconfident, hallucinated phrase

### 1.3M Semi-Synthetic Hallucination-Aware Training Data

Expanded from LLaVA-v1.5-665k, comprising 6.8M QA pairs (3.8M correct answers + 2.9M hallucinated answers):
- Positive phrases are enclosed with `<SPAN>` and `</CN>`
- Negative phrases are enclosed with `<SPAN>` and `</UN>`, with truncation immediately after `</UN>`
- Binary Yes/No and counting questions use rule-based negative generation; open-ended answers use GPT-4o-mini
- 20% of data incorporates query rewriting prompts to support retrospective correction

### Hallucination-Aware Training Loss

A modified cross-entropy loss that applies target masking (weight set to 0) on tokens between `<SPAN>` and `</UN>`, preventing reinforcement of language priors on hallucinated content:

$$L(S) = -\sum_{y_i \in Y} \mathbb{1}_{Hall(i)} \cdot \log P(y_i | X, y_1, ..., y_{i-1}; \theta)$$

where $\mathbb{1}_{Hall(i)}=0$ only when the token lies between `<SPAN>` and `</UN>`.

### Retrospective Resampling

During inference, the generation probability $P(\text{</UN>})$ is continuously monitored. When it exceeds a threshold $\tau$, a hierarchical rollback strategy is triggered:

1. **Local Rollback**: Roll back to the most recent `</CN>` (confidence checkpoint) and attempt local correction.
2. **Sentence-Level Rollback**: If local correction fails $K$ times ($K=10$), roll back to the previous sentence boundary.
3. **Query Rewriting with Hint**: Append a hint to the input — "Hint: potential incorrect phrases → \<placeholder\>"
4. **Termination**: If correction still fails after $N$ attempts ($N=50$), return the current output flagged as potentially hallucinated.

Temperature is gradually increased during rejection sampling (step $\Delta T=0.1$, upper bound $T_0+0.5$) to encourage exploration of alternative expressions.

## Key Experimental Results

### CHAIR-MSCOCO Image Captioning (Lower is Better)

| Method | CHAIRi↓ | CHAIRs↓ |
|--------|---------|---------|
| LLaVA-v1.5 7B | 15.4 | 50.0 |
| HA-DPO | 11.0 | 38.2 |
| HALVA | 11.7 | 41.4 |
| **REVERSE (τ=0.003)** | **10.3** | **37.0** |
| **REVERSE (τ=0.0003)** | **6.1** | **13.6** |

### HaloQuest Open-Ended QA (Accuracy↑)

| Method | Avg Acc↑ | FP | VC | IC |
|--------|----------|----|----|-----|
| LLaVA-v1.5 | 22.6 | 17.1 | 39.5 | 10.7 |
| HALVA | 23.9 | 21.1 | 37.4 | 10.7 |
| **REVERSE (τ=0.003)** | **30.7** | 31.8 | 31.5 | 26.9 |
| **REVERSE (τ=0.0003)** | **32.3** | 29.4 | 18.7 | **58.8** |

### MMHal-Bench (Score↑ / Hall Rate↓)

| Method | Score↑ | Hall. Rate↓ |
|--------|--------|------------|
| LLaVA-v1.5 | 2.11 | 0.54 |
| HALVA | 2.25 | 0.54 |
| **REVERSE (τ=0.003)** | **2.56** | **0.47** |
| **REVERSE (τ=0.0003)** | **3.28** | **0.30** |

### Ablation Study (AMBER-G)

| Component | CHAIR↓ | Cover↑ | Hall↓ | Cog↓ |
|-----------|--------|--------|-------|------|
| LLaVA-v1.5 Baseline | 7.8 | 51.0 | 36.4 | 4.2 |
| + Hallucination-Aware Training | 7.2 | 53.2 | 36.3 | 3.4 |
| + Rejection Sampling | 6.0 | 51.0 | 30.5 | 3.0 |
| + Query Rewriting | 6.0 | 52.2 | 30.4 | 3.0 |

### Efficiency

- 37% of samples require no rollback; among the remainder, more than half require only one correction round.
- At $N=50$, total token generation is approximately 3.05× that of the baseline.
- Verification overhead is negligible (inline token-level confidence estimation), far lower than the external model overhead of Woodpecker.

## Highlights & Insights

1. **First Unified Generate + Verify + Correct Framework**: A single VLM serves as both generator and verifier without external models, performing retrospective correction rather than simple rejection.
2. **Tunable Threshold τ for Expressiveness–Trustworthiness Trade-off**: τ can be continuously adjusted from 0.01 to 0.0001; at τ=0.0001, hallucination control even surpasses GPT-4V, making REVERSE the first method to offer such a user-controllable parameter.
3. **Hallucination-Aware Training Alone Yields Gains**: Even without inference-time rollback, the contrastive learning effect from the training stage already outperforms existing VLMs (analogous to a DPO effect).
4. **Robust to Temperature Variation**: While other methods suffer from simultaneous degradation in hallucination rate and coverage at high temperatures, REVERSE reduces hallucinations while improving coverage even under high temperature.

## Limitations & Future Work

1. **Increased Inference Overhead**: Worst-case token generation reaches 3×; KV-cache reuse could mitigate this but is not yet implemented.
2. **Ineffective for Discriminative VQA**: Rollback provides no additional reasoning benefit in binary Yes/No tasks.
3. **Training Data Dependence on GPT-4o-mini**: May introduce biases and limited coverage.
4. **Threshold τ Requires Per-Model Tuning**: LLaVA uses 0.003, Qwen uses 0.01; confidence scores are not calibrated across models.
5. **Accuracy Drop on VC (Visually Challenging) Subset**: The more conservative generation strategy causes the model to refuse some answerable questions.

## Related Work & Insights

- **Generation Adjustment vs. Post-Hoc Verification**: This paper is the first to demonstrate that the two paradigms can be unified; confidence tokens serve simultaneously as classifiers and rollback triggers.
- **Implicit Connection to DPO**: The contrastive training on positive and negative phrase pairs in hallucination-aware training may produce a DPO-like effect, warranting further investigation.
- **Broader Inspiration**: The retrospective resampling idea is generalizable to LLM factuality checking (e.g., having LLMs self-annotate key claims requiring citation during generation and verifying them on the fly).

## Rating

⭐⭐⭐⭐ — The unified framework is elegantly designed, experiments are comprehensive (3 VLM backbones × multiple benchmarks), and the tunable threshold offers practical value. Limitations include inference overhead and reliance on an external model for training data quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[NeurIPS 2025\] Sherlock: Self-Correcting Reasoning in Vision-Language Models](sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](../../ACL2026/multimodal_vlm/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[CVPR 2026\] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](../../CVPR2026/multimodal_vlm/capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)

</div>

<!-- RELATED:END -->
