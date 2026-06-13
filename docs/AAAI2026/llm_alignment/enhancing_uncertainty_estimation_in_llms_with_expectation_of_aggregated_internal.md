---
title: >-
  [Paper Note] Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief
description: >-
  [AAAI 2026][LLM Alignment][uncertainty estimation] This paper proposes EAGLE, a method that estimates uncertainty by aggregating logits from multiple intermediate hidden layers of an LLM and computing the expectation of…
tags:
  - "AAAI 2026"
  - "LLM Alignment"
  - "uncertainty estimation"
  - "internal state aggregation"
  - "confidence calibration"
  - "EAGLE"
  - "hidden layers"
date: 2026-05-08
content_hash: 1384b0078a8a0844
---

# Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief

**Conference**: AAAI 2026
**arXiv**: [2509.01564](https://arxiv.org/abs/2509.01564)  
**Code**: None  
**Area**: LLM Alignment
**Keywords**: uncertainty estimation, internal state aggregation, confidence calibration, EAGLE, hidden layers


## TL;DR

This paper proposes EAGLE, a method that estimates uncertainty by aggregating logits from multiple intermediate hidden layers of an LLM and computing the expectation of the resulting confidence distribution. EAGLE requires no additional trainable parameters and reduces ECE from 12.6% to 3.2% while improving AUROC from 59.0% to 61.6% across multiple datasets and models.

## Background & Motivation

- **Background**: Uncertainty estimation in LLMs is critical for safe deployment. Existing approaches fall into two main categories: (1) multi-sampling methods (semantic entropy / self-consistency), which are computationally expensive, and (2) self-evaluation (verbalized confidence) methods, which are direct but rely solely on surface-level confidence scores from the final layer.
- **Limitations of Prior Work**: LLMs trained with RLHF tend to be overconfident—the softmax probabilities from the final layer have been "calibrated" to produce high-confidence outputs that please human evaluators, no longer reflecting the model's true uncertainty. Point estimates derived from the final layer alone discard rich internal information.
- **Key Challenge**: The model's internal representations (hidden states across layers) contain fine-grained signals about confidence, yet conventional self-evaluation methods decode a single confidence score only from the final layer—analogous to reading only the conclusion of a book while ignoring all supporting arguments.
- **Key Insight**: Intermediate hidden states are found to naturally separate high-confidence from low-confidence predictions; this property is exploited to aggregate more robust uncertainty signals across multiple layers.

## Method

### Overall Architecture

The EAGLE (**E**xpectation of **AG**gregated internaL b**E**lief) pipeline proceeds as follows: (1) the LLM generates an answer; (2) the LLM is prompted to verbalize its confidence; (3) hidden states at the positions of the confidence tokens are extracted from multiple layers; (4) the hidden states are projected into the vocabulary space and the resulting logits are aggregated; (5) a softmax over the aggregated logits yields a confidence distribution; (6) the expectation of this distribution is computed as the final confidence score.

### Key Designs

1. **Cross-Layer Hidden State Extraction and Projection**

    - Hidden states $H_n^{(l)}$ at the confidence-token positions are extracted from the last $k$ layers.
    - The model's unembedding matrix projects each layer's hidden state into the vocabulary space: $z_n^{(l)} = f_{\text{unembed}}(H_n^{(l)})$.
    - Each layer thus yields a set of logits reflecting that layer's "preference" over tokens at the target position.

2. **Logits Aggregation**

    - Logits from the $k$ layers are combined via a weighted average: $z_n = \sum(w_l \cdot z_n^{(l)}) / k$, with uniform weights by default.
    - Aggregation reduces per-layer noise and overfitting, capturing a more stable internal belief signal.
    - Different layers encode different levels of linguistic and semantic information; aggregation effectively integrates the model's judgments across all processing stages.

3. **Expectation Instead of Argmax**

    - A softmax is applied to the subset of aggregated logits corresponding to the confidence-score tokens (0–9), yielding a probability distribution.
    - The final confidence score is the expectation $\sum(w_s \cdot s)$ rather than the argmax.
    - The expectation captures the full uncertainty of the distribution—if the model is uncertain between "7" and "8," an expected value of approximately 7.5 is more faithful than simply selecting "8."

4. **Fully Training-Free**

    - No probe networks or additional parameters are required.
    - Only a single forward pass is needed (identical to standard self-evaluation); the only additional cost is extracting intermediate hidden states and performing matrix multiplications.
    - Applicable to any open-source decoder-only Transformer.

### Loss & Training

- Completely training-free; no fine-tuning or additional training of any kind is required.
- Self-evaluation prompt design is explored: different prompt variants are tested and found to have a moderate effect on performance.
- Confidence score range: 0–9 (10 discrete values) by default; ablation studies examine different ranges.

## Key Experimental Results

### Main Results (Llama3 8B; lower ECE / higher AUROC is better)

| Method | TriviaQA ECE | GSM8K ECE | MMLU ECE | Avg. ECE | Avg. AUROC |
|---|---|---|---|---|---|
| Self-Eval (SE) | 15.5 | 17.1 | 5.1 | 12.6 | 59.0 |
| Self-Consistency | 27.7 | 25.4 | 7.3 | 20.1 | 54.0 |
| P(true) | 23.3 | 25.1 | 37.9 | 28.8 | 60.5 |
| CSL | 28.7 | 6.3 | 39.2 | 24.7 | 56.9 |
| **EAGLE** | **1.7** | **7.6** | **0.4** | **3.2** | **61.6** |

### Ablation Study (Qwen2.5 72B; lower ECE is better)

| Method | TriviaQA ECE | GSM8K ECE | MMLU ECE | Avg. AUROC |
|---|---|---|---|---|
| Self-Eval | 18.6 | — | — | — |
| EAGLE | **1.7** | — | — | Substantial improvement |

### Key Findings

- **ECE reduced from 12.6% to 3.2%** (Llama3 8B average), a 75% relative reduction, indicating a substantial improvement in calibration quality.
- **ECE of only 0.4% on MMLU**—near-perfect calibration.
- **Intermediate layers (approximately 60%–80% depth) contribute most to uncertainty estimation**, suggesting that critical "decision-making" occurs at these layers.
- **Expectation vs. argmax**: using the distributional expectation is more stable and accurate than selecting the highest-probability score.
- **Cross-model consistency**: strong performance is observed on Llama3 8B/70B and Qwen2.5 7B/72B.
- **Multi-sampling methods (Self-Consistency) yield worse calibration**: sampling consistency does not always reflect true uncertainty.

## Highlights & Insights

- **"Look inside, not at the surface" paradigm**: extracting uncertainty signals from internal model states rather than surface outputs circumvents the overconfidence induced by RLHF.
- **Major advantage of being training-free**: no labeled data or additional training is needed; the method is plug-and-play with near-zero deployment overhead.
- **Discovery of a "decision window" in intermediate layers**: layers at 60%–80% depth contribute most to uncertainty signals, offering insights into the internal decision-making mechanisms of LLMs.

## Limitations & Future Work

- Access to internal hidden states is required, limiting applicability to open-source models; the method cannot be applied to closed-source APIs.
- Uniform-weight aggregation may be suboptimal; adaptive learning of layer weights could further improve performance.
- Discretization of the confidence score (0–9) limits precision; continuous alternatives are worth exploring.
- Validation is currently restricted to QA and mathematical reasoning tasks; effectiveness on open-ended generation tasks remains to be examined.

## Related Work & Insights

- **vs. Semantic Entropy (Kuhn et al. 2023)**: semantic entropy requires multiple samples (high cost), whereas EAGLE requires only a single forward pass; moreover, semantic entropy can yield worse calibration in certain settings.
- **vs. Conventional Self-Evaluation (Verbalized Confidence)**: conventional methods use only the final layer's output and are susceptible to RLHF-induced overconfidence; EAGLE bypasses this issue through multi-layer aggregation.

## Rating

| Dimension | Score | Rationale |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | Aggregating multi-layer internal states for uncertainty estimation is a novel and intuitively clear perspective. |
| Technical Depth | ⭐⭐⭐⭐ | The cross-layer aggregation and expectation-based design is concise yet effective; layer-wise analysis is thorough. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers 4 models × 3 datasets × 5 baselines, with comprehensive layer and prompt analyses. |
| Value | ⭐⭐⭐⭐⭐ | Training-free and plug-and-play; directly applicable to safe LLM deployment. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](../../NeurIPS2025/llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[ACL 2026\] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](../../ACL2026/llm_alignment/maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)
- [\[NeurIPS 2025\] Multi-Environment POMDPs: Discrete Model Uncertainty Under Partial Observability](../../NeurIPS2025/llm_alignment/multi-environment_pomdps_discrete_model_uncertainty_under_partial_observability.md)
- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](../../ACL2026/llm_alignment/wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[ICML 2026\] The Realignment Problem: When Right becomes Wrong in LLMs](../../ICML2026/llm_alignment/the_realignment_problem_when_right_becomes_wrong_in_llms.md)

</div>

<!-- RELATED:END -->
