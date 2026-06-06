---
title: >-
  [Paper Note] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal
description: >-
  [ACL 2026][LLM/NLP][surprisal theory] By fine-tuning neural language models on garden-path sentences, this paper demonstrates the existence of a neural LM that can simultaneously explain garden-path effects and naturalis…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "surprisal theory"
  - "garden-path effects"
  - "human reading times"
  - "language model fine-tuning"
  - "psycholinguistics"
date: 2026-05-08
content_hash: f7deeabb282a630e
---

# An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal

**Conference**: ACL 2026
**arXiv**: [2604.18293](https://arxiv.org/abs/2604.18293)  
**Code**: [github](https://github.com/osekilab/RE-GPE)  
**Area**: LLM/NLP
**Keywords**: surprisal theory, garden-path effects, human reading times, language model fine-tuning, psycholinguistics

## TL;DR

By fine-tuning neural language models on garden-path sentences, this paper demonstrates the existence of a neural LM that can simultaneously explain garden-path effects and naturalistic reading times via surprisal, providing an existence proof for surprisal theory.

## Background & Motivation

**Background**: Surprisal Theory posits that the difficulty of human sentence processing is linearly related to the negative log-probability (surprisal) of each word. Researchers have used language models as proxies for human predictions to validate this hypothesis.

**Limitations of Prior Work**: Although neural LM surprisal captures human reading times on naturalistic corpora reasonably well, it severely underestimates processing difficulty on sentences requiring syntactic disambiguation (e.g., garden-path sentences such as "the horse raced past the barn fell"), predicting only 1/10 to 1/30 of the observed human slowdown.

**Key Challenge**: This failure has sparked debate between two possible explanations — either neural LM probability estimates differ from those of humans, or garden-path effects are fundamentally irreducible to surprisal. Recent work has tended toward the latter, concluding that surprisal theory is insufficient to account for such phenomena.

**Goal**: To investigate the first possibility — whether it is truly impossible to construct a neural language model capable of explaining garden-path effects via surprisal.

**Key Insight**: Rather than evaluating off-the-shelf LMs, the paper fine-tunes LMs to align their surprisal estimates with actual human reading times.

**Core Idea**: By fine-tuning GPT-2 on garden-path sentences so that its surprisal better matches human reading times, the paper provides an existence proof — demonstrating that a neural LM can simultaneously account for garden-path effects and naturalistic reading times.

## Method

### Overall Architecture

Following the fine-tuning methodology of Kiegeland et al. (2024), the approach aligns surprisal-driven reading time estimates with actual human reading times. GPT-2 (S/M/L) is fine-tuned on garden-path sentences and evaluated along three dimensions: (i) whether it generalizes to unseen garden-path items; (ii) whether it retains predictive power on naturalistic reading times; and (iii) whether it preserves general LM capabilities.

### Key Designs

1. **Ridge Regression-Based Reading Time Estimation**:

    - Function: Maps LM surprisal to reading time estimates.
    - Mechanism: The feature vector includes surprisal at the current position and the two preceding positions (capturing spillover effects), along with control variables (word length, position, etc.); regression coefficients are estimated via ridge regression.
    - Design Motivation: Coefficients are estimated on "ordinary" reading times outside the region of interest (ROI), and these same coefficients should also explain reading times at the ROI affected by syntactic disambiguation.

2. **Fine-Tuning Loss with Regularization**:

    - Function: Guides the LM toward a surprisal distribution that more closely matches human reading patterns.
    - Mechanism: The loss function comprises two terms — minimizing the squared residuals between actual and estimated reading times, and penalizing deviation of regression coefficients from their initial values.
    - Design Motivation: The regularization term prevents the LM from artificially inflating estimated reading times at the ROI by reducing surprisal elsewhere, ensuring that the model learns meaningful changes to its probability distribution.

3. **Leave-One-Out Cross-Validation Framework**:

    - Function: Rigorously evaluates generalization on small-scale data.
    - Mechanism: Each fold holds out one sentence pair per garden-path construction type for testing; training and test sets share no overlap in ambiguous verbs or ROI words.
    - Design Motivation: Ensures that evaluation results reflect genuine generalization rather than overfitting.

### Loss & Training

The loss function $\mathcal{L}_B(\theta)$ comprises: (1) a mean squared residual term measuring the discrepancy between predicted and actual reading times; and (2) a coefficient drift penalty term preventing regression coefficients from deviating too far from their initial values. A balanced batch sampling strategy is employed, with each batch containing equal numbers of sentence pairs from each garden-path construction type.

## Key Experimental Results

### Main Results

| Model | Construction | ROI 1 Pre-FT Coverage | ROI 1 Post-FT Coverage |
|------|------|-------------------|-------------------|
| GPT-2 Small | MVRR | 7% | 73% |
| GPT-2 Small | NPS | 19% | 83% |
| GPT-2 Small | NPZ | 15% | 73% |

### Ablation Study

| Configuration | Key Findings | Notes |
|------|---------|------|
| Single-construction FT → cross-construction transfer | MVRR FT → NPS 51.5ms (baseline 9.6ms) | Cross-construction transfer is effective |
| SRC/ORC asymmetry FT | Captures only 22% of human effect | Limited effectiveness where surprisal theory does not apply |
| Naturalistic corpus prediction | Improved across the board after FT | Garden-path FT unexpectedly improves naturalistic text prediction |

### Key Findings
- Fine-tuned GPT-2 Small performs best, capturing approximately 73%–83% of the human reading slowdown at ROI 1, far exceeding the pre-fine-tuning baseline of 7%–19%.
- The fine-tuned model correctly reproduces the cross-construction ordering of human reading slowdown magnitudes (MVRR > NPZ > NPS).
- On naturalistic corpora, the fine-tuned LM shows improved predictive power for human reading times.
- Single-construction fine-tuning also transfers to other constructions, suggesting the model has learned a general mechanism underlying garden-path effects.
- However, fine-tuning shows limited effectiveness on SRC/ORC asymmetries — a phenomenon where surprisal theory is considered inapplicable — indicating that the method is not universally effective.

## Highlights & Insights
- The paper features an elegant research design that transforms a theoretical debate into an operationalizable empirical question — not proving surprisal theory "correct," but providing a constructive existence proof.
- Improvements from fine-tuning on garden-path sentences simultaneously enhance prediction on naturalistic corpora, suggesting that pre-trained LMs exhibit systematic deviations from human predictions.
- The negative results on SRC/ORC are equally informative, showing that the method's limitations align with the known boundaries of surprisal theory's applicability.
- The paper raises a profound theoretical question: if the LM space is unbounded, surprisal theory may be practically unfalsifiable.

## Limitations & Future Work
- The dataset is small (24 pairs × 3 construction types); validation on larger-scale data is needed.
- The paper does not investigate which internal mechanisms of the LM are altered by fine-tuning.
- An existence proof, while important, does not demonstrate that existing LMs naturally acquire the correct human probability distributions.
- The authors propose two directions for improving the falsifiability of surprisal theory: constraining the probability distribution space, and requiring psychological plausibility of the parsing distribution.

## Related Work & Insights
- **vs. van Schijndel & Linzen (2021)**: They treated the failure of neural LM surprisal to explain garden-path effects as evidence against surprisal theory; this paper challenges that conclusion.
- **vs. Kiegeland et al. (2024)**: They fine-tuned LMs on naturalistic corpora; this paper extends their method to garden-path sentences.
- **vs. Huang et al. (2024)**: They systematically demonstrated the extent to which LM surprisal underestimates garden-path effects; this paper provides a counterexample.

## Rating
- Novelty: ⭐⭐⭐⭐ Transforms a theoretical debate into a constructive existence proof — a novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation with both positive and negative results, though the dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical background is clearly articulated; argumentation is logically rigorous.
- Value: ⭐⭐⭐⭐ Makes a significant contribution to theoretical discussions in computational psycholinguistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)

</div>

<!-- RELATED:END -->
