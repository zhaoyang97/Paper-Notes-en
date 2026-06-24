---
title: >-
  [Paper Note] Causal Estimation of Tokenisation Bias
description: >-
  [ACL 2025][tokenisation bias] This paper formalizes the impact of tokenizer choice on language model outputs as "tokenisation bias" for the first time. It utilizes Regression Discontinuity Design (RDD) from causal inference to quantify this effect, revealing that when a subword is included in the vocabulary, the probability of its corresponding string can increase by up to 17 times (for smaller models). This indicates that tokenization is an underestimated…
tags:
  - "ACL 2025"
  - "tokenisation bias"
  - "causal inference"
  - "regression discontinuity"
  - "BPE"
  - "WordPiece"
  - "vocabulary"
  - "language model"
date: 2026-05-08
content_hash: ce7208c1997f9832
---

# Causal Estimation of Tokenisation Bias

**Conference**: ACL 2025  
**arXiv**: [2506.03149](https://arxiv.org/abs/2506.03149)  
**Code**: [GitHub](https://github.com/pietrolesci/tokenisation-bias)  
**Institution**: University of Cambridge & ETH Zürich  
**Area**: Others  
**Keywords**: tokenisation bias, causal inference, regression discontinuity, BPE, WordPiece, vocabulary, language model

## TL;DR

This paper formalizes the impact of tokenizer choice on language model outputs as "tokenisation bias" for the first time. It utilizes Regression Discontinuity Design (RDD) from causal inference to quantify this effect, revealing that when a subword is included in the vocabulary, the probability of its corresponding string can increase by up to 17 times (for smaller models). This indicates that tokenization is an underestimated, critical design choice in language modeling.

## Background & Motivation

### Background

- Modern language models are trained on subword sequences but ultimately define probability distributions over strings.
- Different tokenizers (with different vocabularies) map the same string to different subword sequences (e.g., "hello" $\rightarrow$ ⟨he,llo⟩ or ⟨hello⟩).
- Ideally, the choice of tokenizer should not influence the probability assigned to a string by the model; however, in practice, it does.

### Limitations of Prior Work

- Estimating tokenisation bias faces a fundamental causal inference challenge: each model is trained on only one tokenizer, making direct comparison impossible.
- One cannot simply compare the probabilities of in-vocabulary and out-of-vocabulary words because the vocabulary is constructed based on criteria like frequency, causing systematic differences between the two groups.
- Brute-force approaches (training separate models for every possible tokenizer) are computationally infeasible.
- Existing studies recognize that tokenizers affect model performance but lack a quantitative understanding of "how" they do so.

### Key Insight

- Tokenizer vocabularies are typically built via incremental algorithms (BPE by frequency, WordPiece by likelihood gain), which yield a subword ranking.
- The vocabulary size $K$ acts as an arbitrary cutoff: the top $K$ subwords are included, while those below are excluded.
- This cutoff constitutes a natural "discontinuity," allowing the application of a Regression Discontinuity Design (RDD) to estimate the causal effect.
- Subwords near the cutoff share highly similar characteristics, but their inclusion in the vocabulary is quasi-randomly assigned.

## Method

### Overall Architecture

The tokenisation bias is formalized as a causal effect: comparing the log-probability of a string $c_v$ when its corresponding subword $v$ is in the vocabulary (observed value) versus when it is not (counterfactual value). Utilizing the ranking mechanism of BPE/WordPiece and the vocabulary size cutoff as a discontinuity, RDD is applied to conduct causal estimation.

### Key Designs

#### 1. Causal Framework Definition

- Treatment variable $W_v = 1\{v \in T\}$: whether subword $v$ is in the tokenizer vocabulary.
- Potential outcome $Y_v(w)$: the log-probability assigned by the model to the character sequence $c_v$ under treatment condition $w$.
- Causal effect $\tau_v = Y_v(1) - Y_v(0)$: the impact of vocabulary inclusion versus exclusion on the log-probability.
- Average Treatment Effect $\text{ATE} = \mathbb{E}[Y(1) - Y(0)]$ (averaged over subwords near the cutoff).

#### 2. Regression Discontinuity Design (RDD)

- Running variable $R_v$: the ranking of subword $v$ (BPE by merge frequency, WordPiece by likelihood gain).
- Cutoff $c = K$ (vocabulary size): $v$ is included in the vocabulary if $R_v \le K$, and excluded if $R_v > K$.
- Near the cutoff, subwords with a rank difference of $1$ share almost identical characteristics (frequency, length, etc.), but their treatment assignment is quasi-random.
- Local linear regression is utilized to fit the outcomes on both sides of the cutoff. The jump at the discontinuity represents the causal effect.
- Bandwidth $h$ is automatically selected using the Imbens-Kalyanaraman method.

#### 3. Experimental Control and Validation

- Train models of various scales (100M and 850M parameters).
- Use both BPE and WordPiece tokenizers.
- Conduct experiments across multiple vocabulary sizes (1K, 2K, 4K, 8K, 16K, 32K).
- Use McCrary density tests to verify that the running variable is not manipulated near the cutoff.
- Covariate balance tests confirm that subword characteristics are balanced on both sides of the cutoff.
- Placebo tests (estimating effects at non-cutoff positions) confirm that the observed discontinuity effect does not exist elsewhere.

### Loss & Training

- Train Transformer language models from scratch on a fixed dataset.
- Train independent models for each vocabulary size $\times$ tokenizer combination.
- Use character-level marginalization to convert subword probabilities into string probabilities.
- Track the trend of bias during the training process.

## Key Experimental Results

### Main Results: Tokenisation Bias Across Different Model Scales

| Model Scale | Tokeniser | Average Bias (nats) | Probability Multiplier |
|:---|:---|:---|:---|
| 100M Parameters | BPE | 2.88 | ~17.8x |
| 100M Parameters | WordPiece | 2.51 | ~12.3x |
| 850M Parameters | BPE | ~1.0 | ~2.7x |
| 850M Parameters | WordPiece | ~1.0 | ~2.7x |

### Impact of Vocabulary Size

- Bias consistently exists across all vocabulary sizes (1K to 32K).
- Smaller vocabularies generally result in larger bias.
- Bias continually increases during training, showing no signs of convergence.

### Key Findings

1. **Tokenisation bias is ubiquitous**: Significant bias was observed across all tested model scales, vocabulary sizes, and tokenizer types.
2. **Smaller models exhibit larger bias**: The bias for the 100M-parameter model reaches up to 2.88 nats (rendering a 17-fold increase in probability), which is significantly larger than the ~1 nat bias observed in the 850M-parameter model.
3. **Bias grows through training**: Bias continually increases during training, showing that models become increasingly "reliant" on the integrity of in-vocabulary subwords.
4. **BPE displays slightly larger bias than WordPiece**: Biases caused by different tokenizers vary in magnitude but share the same direction.
5. **Causal validation**: McCrary test, covariate balance, and placebo tests are all passed, confirming that the observed effects are indeed causal rather than confounding.

## Highlights & Insights

- **Methodological Innovation**: It is highly elegant to apply the RDD method from causal inference to the tokenization problem in language modeling for the first time.
- **Quantifying an Overlooked Issue**: Although the intuition that "tokenization affects models" is widely accepted, this work provides the first precise causal quantification.
- **17x Probability Gap**: This number is striking—simply because a subword is in the vocabulary, its string probability can differ by a factor of 17.
- **Implications for Practice**: In multilingual or specialized domain scenarios, tokenizer design may play a more vital role than model architecture choices.

## Limitations & Future Work

- RDD can only estimate local causal effects near the cutoff (marginal subwords) and cannot generalize to the core regions of the vocabulary.
- The experiments are constrained by model scale (maximum 850M parameters), leaving it unverified whether the bias further diminishes in 7B+ models.
- This study only considers the bias dimension of whether a subword is included in the vocabulary, leaving internal tokenizer sorting and its effects unanalyzed.
- No specific mitigation methods (such as improved training strategies or regularization) were proposed to address tokenisation bias.
- Analysis of tokenisation bias in multilingual settings presents an important direction for future research.

## Related Work & Insights

- Pimentel & Meister (2024) proposed the theoretical foundation for subword-to-character probability marginalization.
- Empirical studies such as Rust et al. (2021) and Ali et al. (2024) show that tokenizers affect model performance, but do not provide causal evidence.
- Insight: Tokenizer design should be treated as a key design choice of equal importance to model architecture and training data.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Very unique cross-domain innovation combining causal inference and tokenization.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous and complete application of RDD, with thorough validation.
- **Utility**: ⭐⭐⭐ — Primarily a diagnostic tool; practical solutions are yet to be proposed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple scales, multiple tokenizers, and comprehensive causal validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tokenisation is NP-Complete](tokenisation_is_np-complete.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[ACL 2025\] Statistical Deficiency for Task Inclusion Estimation](statistical_deficiency_task_inclusion.md)

</div>

<!-- RELATED:END -->
