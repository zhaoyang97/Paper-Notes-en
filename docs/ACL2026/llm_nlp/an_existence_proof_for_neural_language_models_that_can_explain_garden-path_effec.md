---
title: >-
  [Paper Note] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal
description: >-
  [ACL 2026][LLM/NLP][Surprisal Theory] By fine-tuning neural language models on garden-path sentences, this study demonstrates the existence of a neural LM that can simultaneously explain garden-path effects and natural r…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Surprisal Theory"
  - "Garden-path effects"
  - "human reading times"
  - "language model fine-tuning"
  - "psycholinguistics"
date: 2026-05-08
content_hash: 6e5eac14d8e0ecb3
---

# An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal

**Conference**: ACL 2026  
**arXiv**: [2604.18293](https://arxiv.org/abs/2604.18293)  
**Code**: [github](https://github.com/osekilab/RE-GPE)  
**Area**: LLM/NLP  
**Keywords**: Surprisal Theory, Garden-path effects, human reading times, language model fine-tuning, psycholinguistics

## TL;DR

By fine-tuning neural language models on garden-path sentences, this study demonstrates the existence of a neural LM that can simultaneously explain garden-path effects and natural reading times through surprisal, providing an existence proof for surprisal theory.

## Background & Motivation

**Background**: Surprisal Theory posits that the difficulty of human sentence processing is linearly related to the negative log-probability (surprisal) of a word. In recent years, researchers have used language models as proxies for human prediction to validate this hypothesis.

**Limitations of Prior Work**: Although the surprisal of neural LMs captures human reading times on natural corpora relatively well, it severely underestimates processing difficulty on sentences requiring syntactic disambiguation (e.g., garden-path sentences like "the horse raced past the barn fell")—predicting only 1/10 to 1/30 of the human reading slowdown.

**Key Challenge**: This failure has sparked a debate between two possible explanations: either the probability distributions estimated by neural LMs differ from those of humans, or garden-path effects inherently cannot be reduced to surprisal. Several recent studies favor the latter, suggesting that surprisal theory is insufficient to explain such phenomena.

**Goal**: To investigate the first possibility—whether it is truly impossible to construct a neural language model that can explain garden-path effects via surprisal.

**Key Insight**: Rather than evaluating off-the-shelf LMs, this work fine-tunes LMs to align their surprisal estimates with actual human reading times.

**Core Idea**: By fine-tuning GPT-2 on garden-path sentences to better match human reading times, the authors provide an "existence proof"—showing that a neural LM can indeed explain both garden-path effects and natural reading times.

## Method

### Overall Architecture

The study adopts the fine-tuning method from Kiegeland et al. (2024), aligning surprisal-driven reading time estimates with actual human reading times. GPT-2 (S/M/L) is fine-tuned on garden-path sentences and then evaluated across three dimensions: (i) generalization to unseen garden-path items; (ii) maintenance of predictive power for natural reading times; and (iii) preservation of general LM capabilities.

### Key Designs

1.  **Ridge Regression-based Reading Time Estimation**:
    - **Function**: Maps LM surprisal to reading time estimates.
    - **Mechanism**: The feature vector includes surprisal at the current and two preceding positions (to capture spillover effects) along with control variables (word length, position, etc.). Regression coefficients are estimated via ridge regression.
    - **Design Motivation**: Coefficients estimated on "ordinary" reading times outside the Region of Interest (ROI) should also be able to explain reading times within the ROI affected by syntactic disambiguation.

2.  **Regularized Fine-tuning Loss Function**:
    - **Function**: Guides the LM to produce surprisal distributions that more closely resemble human reading patterns.
    - **Mechanism**: The loss function consists of two terms—minimizing the squared residual between actual and estimated reading times, and penalizing the deviation of regression coefficients from their initial values.
    - **Design Motivation**: The second regularization term prevents the LM from artificially inflating estimated reading times at the ROI by lowering surprisal outside the ROI, ensuring the model learns plausible probability distribution shifts.

3.  **Leave-One-Out Cross-Validation Framework**:
    - **Function**: Rigorously evaluates generalization capabilities on small-scale data.
    - **Mechanism**: One pair of sentences from each garden-path construction is held out for testing in each fold. There is no overlap between the training and test sets regarding ambiguous verbs or ROI words.
    - **Design Motivation**: To ensure that the evaluation results reflect genuine generalization rather than overfitting.

### Loss & Training

The loss function $\mathcal{L}_B(\theta)$ includes: (1) a mean squared residual term measuring the discrepancy between predicted and actual reading times; (2) a coefficient drift penalty to prevent regression coefficients from moving too far from their initial values. A balanced batch sampling strategy is employed, where each batch contains an equal number of sentence pairs from each garden-path construction.

## Key Experimental Results

### Main Results

| Model | Construction | Pre-tuning ROI 1 Coverage | Post-tuning ROI 1 Coverage |
|-------|--------------|---------------------------|----------------------------|
| GPT-2 Small | MVRR | 7% | 73% |
| GPT-2 Small | NPS | 19% | 83% |
| GPT-2 Small | NPZ | 15% | 73% |

### Ablation Study

| Configuration | Key Findings | Description |
|---------------|--------------|-------------|
| Single-construction tuning → Cross-construction transfer | MVRR tuning → NPS 51.5ms (Baseline 9.6ms) | Cross-construction transfer is effective |
| SRC/ORC asymmetric tuning | Captured only 22% of human effect | Limited effectiveness where surprisal theory is considered inapplicable |
| Natural corpora predictive power | Improved across the board | Garden-path tuning unexpectedly improved natural text prediction |

### Key Findings
- The fine-tuned GPT-2 Small performed best, capturing approximately 73%-83% of the human reading slowdown at ROI 1, a significant improvement over the 7%-19% before fine-tuning.
- The fine-tuned model correctly reproduced the ranking of slowdown magnitudes across constructions (MVRR > NPZ > NPS).
- On natural corpora, the predictive power of the fine-tuned LM for human reading times actually increased.
- Fine-tuning on a single construction could transfer to other constructions, suggesting the model learned a general mechanism for garden-path effects.
- However, the effect of fine-tuning was limited on SRC/ORC asymmetry (a phenomenon where surprisal theory is often deemed inapplicable), indicating the method is not a universal solution.

## Highlights & Insights
- **Clever Research Design**: The study transforms a theoretical debate into an actionable empirical question—instead of trying to prove surprisal theory is "correct," it provides an existence proof.
- Improvement on garden-path sentences also enhanced natural corpus prediction, implying that original LMs have systematic biases relative to human expectations.
- The negative results on SRC/ORC are equally valuable, as they show that the limitations of the method align with the boundaries of surprisal theory.
- The paper raises a profound theoretical challenge: if the space of LMs is unbounded, surprisal theory might become unfalsifiable in practice.

## Limitations & Future Work
- The data scale is small (24 pairs × 3 constructions) and requires validation on larger datasets.
- The internal mechanisms through which fine-tuning changes the LM were not explored.
- An existence proof, while important, does not imply that existing LMs naturally learn the correct human-like probability distributions.
- The authors suggest two directions to improve the falsifiability of surprisal theory: constraining probability distributions and requiring the psychological reality of parsing distributions.

## Related Work & Insights
- **vs van Schijndel & Linzen (2021)**: They argued that the inability of neural LM surprisal to explain garden-path effects serves as evidence against surprisal theory; this paper challenges that conclusion.
- **vs Kiegeland et al. (2024)**: While they fine-tuned LMs on natural corpora, this work extends the methodology to garden-path sentences.
- **vs Huang et al. (2024)**: They systematically demonstrated how much LM surprisal underestimates garden-path effects; this work providing a counterexample.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Converts a theoretical debate into a constructive existence proof; highly original approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multidimensional evaluation with both positive and negative results, though constrained by data scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear exposition of theoretical background and rigorous logical flow.
- **Value**: ⭐⭐⭐⭐ Significant contribution to the theoretical discourse in computational psycholinguistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)
- [\[ACL 2026\] Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ACL 2026\] Generative Interfaces for Language Models](generative_interfaces_for_language_models.md)
- [\[ACL 2026\] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)

</div>

<!-- RELATED:END -->
