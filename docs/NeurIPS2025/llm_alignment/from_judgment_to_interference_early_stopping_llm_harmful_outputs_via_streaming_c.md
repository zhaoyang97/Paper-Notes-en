---
title: >-
  [Paper Note] From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring
description: >-
  [NeurIPS 2025][LLM Alignment][streaming moderation] This paper proposes the Streaming Content Monitor (SCM)—the first harmful content monitor natively designed for partial detection. Built upon the FineHarm dataset (29K…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "streaming moderation"
  - "harmful content detection"
  - "early stopping"
  - "token-level annotation"
  - "partial detection"
date: 2026-05-08
content_hash: 3dfcde6260c30ffe
---

# From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring

**Conference**: NeurIPS 2025
**arXiv**: [2506.09996](https://arxiv.org/abs/2506.09996)
**Code**: [Project](https://liesy.github.io/SCM)
**Area**: LLM Alignment
**Keywords**: streaming moderation, harmful content detection, early stopping, token-level annotation, partial detection

## TL;DR
This paper proposes the Streaming Content Monitor (SCM)—the first harmful content monitor natively designed for partial detection. Built upon the FineHarm dataset (29K samples with token-level annotations) and hierarchical consistency-aware learning, SCM achieves a macro F1 of 0.95+ after observing on average only 18% of response tokens, enabling real-time early stopping of harmful LLM outputs.

## Background & Motivation

**Background**: LLM service providers typically deploy content moderators as external safety guardrails alongside safety alignment (e.g., OpenAI Moderation API, LlamaGuard). However, existing moderators adopt a "full-response detection" paradigm—they must wait until the LLM has generated a complete response before making a harmfulness judgment.

**Limitations of Prior Work**: (a) Full-response detection introduces high latency—harmful content has already been fully generated and exposed to users before being intercepted; (b) Existing partial detection solutions (e.g., ProtectAI, GuardrailsAI) directly apply full-detection models to incomplete outputs, resulting in a train-inference gap; (c) These approaches require re-encoding all previously generated tokens at every decoding step, incurring significant redundant computation.

**Key Challenge**: Earlier detection is better for user experience, but incomplete information makes accurate judgment harder—creating a fundamental tension between latency and detection accuracy.

**Key Insight**: Design data and model architectures natively tailored for streaming detection—FineHarm provides token-level supervision, and SCM is trained from the outset to make judgments based on incomplete semantic content.

## Method

### Overall Architecture
(1) **FineHarm Dataset**: 29K samples with token-level harmfulness annotations; (2) **SCM Model**: a feature extractor + token scorer (used at inference) + holistic scorer (used during training, discarded at inference); (3) **Hierarchical Consistency-Aware Learning**: response-level knowledge guides token-level prediction.

### Key Designs

1. **FineHarm Dataset Construction**:

    - Collected 43K+ annotated responses (from WildGuard's 9 LLMs and uncensored LLM generations).
    - Token-level annotation: a heuristic approach—function words are filtered using POS tags; only content words (nouns, verbs, adjectives) in harmful sentences are labeled as harmful.
    - Key statistics: only 46% of sentences and 32.8% of words in harmful responses are harmful—confirming the necessity of fine-grained annotation.
    - Harmful tokens are uniformly distributed across responses, preventing the model from exploiting positional shortcuts.

2. **SCM Training Strategy**:

    - **Dual-level supervision**: simultaneously trains a token scorer (predicting harmfulness probability $c_{t_i}$ for each token) and a response-level holistic scorer ($c_{hol}$).
    - **Logical consistency constraint**: $F = f_{hol}(h_n) \Rightarrow g(\{f_{tok}(h_i)\})$—if a response is harmful, at least one token must be judged harmful; if benign, all tokens should be judged benign.
    - Loss: $L = \alpha L_{tok} + (1-\alpha) L_{hol} + \beta L_{logic}$
    - During training, the holistic scorer injects global knowledge into token representations; at inference, it is discarded and only the token scorer is used for streaming judgment.

3. **Delay-$k$ Partial Detection**:

    - Rather than triggering on a single token, the system accumulates $k$ harmful tokens before classifying a response as harmful and terminating generation.
    - Formula: $\text{Harm}_{@k} = 1$ if and only if $\sum_{j=1}^{i} \mathbb{1}(c_{t_j}) \geq k$
    - Larger $k$ improves robustness (fewer false positives) but delays stopping (more harmful content leaks).

## Key Experimental Results

### Partial Detection Performance

| Method | Macro F1 | Avg. Detection Position | Notes |
|------|---------|------------|------|
| LlamaGuard (full) | 0.958 | 100% | Full-detection baseline |
| OpenAI Mod (full) | 0.889 | 100% | Full-detection baseline |
| LlamaGuard (partial, k=3) | 0.776 | ~40% | Suffers from train-inference gap |
| **SCM (k=3)** | **0.955** | **18%** | Native partial detection |

### Comparison with Full Detection

| SCM Variant | F1 (harmful) | F1 (benign) | Macro F1 |
|---------|-------------|-------------|---------|
| SCM-full | 0.972 | 0.968 | 0.970 |
| SCM-partial (k=3) | 0.962 | 0.949 | 0.955 |

### As a Safety Alignment Tool

| Method | Harmlessness Score | Helpfulness Score | Notes |
|------|---------|---------|------|
| DPO | 73.2 | 67.5 | Standard alignment |
| **DPO + SCM pseudo-labels** | **78.9** | **68.1** | SCM provides token-level feedback |

### Key Findings
- **0.95+ F1 at 18% tokens**: SCM accurately detects harmfulness after observing less than one-fifth of the response on average.
- **Negligible gap from full detection**: partial detection F1 is only 0.015 below full detection.
- **Outperforms direct application of full-detection models**: LlamaGuard's F1 drops to 0.776 under partial detection, while SCM maintains 0.955.
- **Applicable to safety alignment**: when used as a pseudo-labeler for DPO, harmlessness scores improve by 5.7 points.

## Highlights & Insights
- **Native streaming design**: rather than forcing a full-detection model onto partial outputs, both data annotation and model training are optimized for incomplete semantic inputs.
- **Elegant logical consistency constraint**: propositional logical implication naturally connects token-level and response-level predictions, offering a more principled formulation than hard multi-task losses.
- **Practical FineHarm annotation**: the POS-based heuristic is simple yet experimentally shown to be more reliable than LLM-based annotation and likelihood-difference methods.
- **Dual role**: SCM serves both as a real-time safety guardrail at inference and as a pseudo-labeler to improve alignment during training.

## Limitations & Future Work
- **Noisy POS-heuristic annotations**: not all content words in harmful sentences are actually harmful, potentially introducing false positives.
- **Model capacity constraints**: SCM is built on a small encoder model, limiting its understanding of complex or implicit harmful content.
- **English-only evaluation**: multilingual settings remain untested.
- **Future directions**: (1) optimize token-level annotations via RL or active learning; (2) incorporate prompt intent analysis to improve early detection; (3) extend to multimodal content moderation.

## Related Work & Insights
- **vs. LlamaGuard**: LlamaGuard is designed for full-response detection; directly applying it to partial outputs drops F1 by 18 points. SCM natively supports partial detection.
- **vs. Constitutional Classifier (Sharma 2025)**: trains a classifier based on constitutional rules but still requires complete outputs; SCM supports streaming.
- **Insight**: In LLM serving, the latency of safety moderation directly affects both user experience and security—"detecting one step earlier" may matter more than "detecting one point more accurately."

## Rating
- Novelty: ⭐⭐⭐⭐ The native streaming detection paradigm is novel, and the logical consistency constraint is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared against multiple baselines with thorough ablations; alignment application is also demonstrated.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; the paradigm comparison in Figure 1 is highly intuitive.
- Value: ⭐⭐⭐⭐ Directly applicable to LLM safety deployment; detecting at 18% tokens represents a significant practical advance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](../../ICLR2026/llm_alignment/antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)

</div>

<!-- RELATED:END -->
