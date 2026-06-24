---
title: >-
  [Paper Note] RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals
description: >-
  [ICML2025][Causal Inference][Reward Model Interpretability] Proposes RATE (Rewrite-based Attribute Treatment Estimator), which uses a "double rewriting" strategy to eliminate bias introduced by imperfect LLM counterfactual rewrites, enabling accurate estimation of the causal effects of high-level attributes on reward model scores.
tags:
  - "ICML2025"
  - "Causal Inference"
  - "Reward Model Interpretability"
  - "Average Treatment Effect"
  - "Counterfactual Rewriting"
  - "RLHF"
date: 2026-05-08
content_hash: cb442c30f0519f38
---

# RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals

**Conference**: ICML2025  
**arXiv**: [2410.11348](https://arxiv.org/abs/2410.11348)  
**Code**: [toddnief/RATE](https://github.com/toddnief/RATE)  
**Area**: Causal Inference  
**Keywords**: Causal Inference, Reward Model Interpretability, Average Treatment Effect, Counterfactual Rewriting, RLHF

## TL;DR

Proposes RATE (Rewrite-based Attribute Treatment Estimator), which uses a "double rewriting" strategy to eliminate bias introduced by imperfect LLM counterfactual rewrites, enabling accurate estimation of the causal effects of high-level attributes on reward model scores.

## Background & Motivation

Reward Models (RMs) play a central role in LLM alignment, but they remain black boxes—it is highly challenging to determine what exactly RMs are rewarding. A naive approach compares the average reward difference between samples with and without a specific attribute (naive estimator), but this introduces **confounding factors** into the estimation. For instance, when measuring RM sensitivity to "sentiment", if the negative samples in the evaluation data happen to contain more spelling errors, the naive estimator will erroneously include the effect of spelling errors.

To obtain a reliable measure of attribute sensitivity, the problem needs to be formalized as **causal effect estimation**: measuring how the reward changes when only the target attribute is modified while keeping everything else constant. A natural approach is to use an LLM to generate counterfactual pairs (rewriting to alter only the target attribute). However, LLM-generated rewrites are imperfect and introduce "off-target modifications" (such as correcting grammatical errors or adjusting formatting), leading to significant bias.

## Method

### Causal Framework

Formulating RM explainability as the **Average Treatment Effect** (ATE) of attribute $W$ on the reward:

$$\text{ATE} = \mathbb{E}[R(X, Y(1)) - R(X, Y(0))]$$

Where $Y(1)$ and $Y(0)$ are potential outcome pairs differing only on attribute $W$. ATT (Average Treatment Effect on the Treated) and ATU (Average Treatment Effect on the Untreated) are also defined, as they may differ significantly (human preference itself is asymmetric).

### Issues with Imperfect Rewriting

Approximating counterfactuals using LLM rewrites $\text{Re}(y^i, w)$, but rewriting introduces error:

$$\epsilon_w^i = R(x^i, \text{Re}(y^i, w)) - R(x^i, y^i(w))$$

For example, GPT-4o almost always corrects spelling errors when rewriting sentiment attributes, causing the single-rewrite estimator to have systematic bias.

### RATE: Double-Rewriting Debiasing

Core Idea: **Instead of comparing "original vs rewrite", compare "rewrite vs rewritten-rewrite"**. This way, off-target modifications (such as spelling corrections) occur on both sides, canceling each other out in expectation.

For samples with $w^i = 1$:

$$\delta^i = R(x^i, \text{Re}(\text{Re}(y^i, 0), 1)) - R(x^i, \text{Re}(y^i, 0))$$

For samples with $w^i = 0$:

$$\delta^i = R(x^i, \text{Re}(y^i, 1)) - R(x^i, \text{Re}(\text{Re}(y^i, 1), 0))$$

The final ATE is estimated as the weighted average:

$$\widehat{\text{ATE}}_{\text{RATE}} = \frac{n_1}{n_0 + n_1} \widehat{\text{ATT}}_{\text{RATE}} + \frac{n_0}{n_0 + n_1} \widehat{\text{ATU}}_{\text{RATE}}$$

### Theoretical Guarantees

The unbiasedness and $\sqrt{n}$-consistency of RATE are proved under two mild assumptions:

1. **Assumption 1** (Direction-independent rewriting error): The LLM's off-target modification distribution $P_{\text{Re}}$ does not depend on the value of the target attribute $W$—e.g., the tendency of GPT-4o to correct spelling does not depend on the sentiment direction.
2. **Assumption 2** (Additivity of reward to rewriting error): $R(X, Y(W,Z,\xi)) = R_{W,Z}(X,W,Z) + R_\xi(X,\xi)$, meaning the component of the reward affected by the rewriting error is additive with respect to the components of the target and invariant attributes.

## Key Experimental Results

### Semi-Synthetic Experiment: Spelling Errors × "Starts with a Vowel"

Spelling errors are artificially correlated with "starts with a vowel" (creating a spurious correlation), evaluated using FsfairX-LLaMA3-RM.

| Estimation Method | ATE at 0% Spelling Errors | ATE at 30% Spelling Errors |
|---------|---------------------|---------------------|
| Naive   | ≈ 0                 | Significant negative bias (approx. -0.15)  |
| Single Rewrite | ≈ 0                 | Significant negative bias (approx. -0.10)  |
| **RATE** | **≈ 0**            | **≈ 0 (Correct)**       |

$\rightarrow$ As the spurious correlation strengthens, the biases of the naive and single-rewrite estimators continue to increase, while RATE consistently estimates correctly to be $\approx 0$.

### Sentiment Classifier Validation

Using a DistilBERT sentiment classifier as a "reward model" to measure the treatment effect of "length" (which should be close to zero):

- The naive estimator is highly sensitive to distribution shifts, with bias increasing as the length-sentiment correlation increases.
- RATE remains stable and close to zero across various degrees of correlation.

### Real RM Evaluation (Top Models on RewardBench)

Evaluating multiple RMs (ArmoRM, FsfairX, NCSOFT, etc.) on IMDB / ELI5 / HelpSteer:

- **Length**: The naive estimator reports a large effect, whereas RATE shows a minimal effect $\rightarrow$ "length bias" is largely an artifact of the naive evaluation method.
- **Complexity/Helpfulness**: The naive estimator systematically overestimates the effect.
- **Sentiment**: The naive estimator underestimates the effect.
- NCSOFT claims to have fixed the length bias of FsfairX, but RATE shows that the improvement is not as significant as it appears on the surface, potentially due to unintended penalties on other attributes like complexity.

### Experimental Cost

Using the GPT-4o BatchAPI, the cost of double rewriting 25K IMDB samples is approximately \$60.

## Highlights & Insights

- The **double-rewriting debiasing** concept is extremely clever—it eliminates bias by introducing more noise, which is analogous to the method of differences.
- The theoretical analysis is clean: both assumptions have clear intuitive explanations and are verifiable.
- Reveals an important finding: the "length bias" of RMs is largely an artifact introduced by the naive evaluation method rather than a flaw inherent to the RMs themselves.
- The method is highly generalizable and can be applied to any text attribute that can be manipulated via LLM rewriting.
- Distinguishing between ATT, ATU, and ATE provides finer-grained interpretability.

## Limitations & Future Work

1. **No objective metric for rewriting quality**: The quality of counterfactual rewriting ultimately relies on subjective judgment (checking if the generation is reasonable) and lacks formal verification methods.
2. **Limitations of the additivity assumption**: Assumption 2 requires the reward to be additive with respect to rewriting errors, but real-world RMs may exhibit interaction effects between attributes.
3. **Analyzes the RM in isolation**: It does not study how the causal sensitivity of the RM propagates to the downstream behavior of aligned LLMs.
4. **Only supports binary attributes**: The current framework is restricted to $W \in \{0, 1\}$; continuous attributes must be binarized first.
5. **Reliance on a powerful rewriter LLM**: Rewriting quality is bound by the capabilities of the used LLM, and rewriting prompts require manual iterative tuning.

## Related Work & Insights

- **CausaLM** (Feder et al., 2021): Trains text classifiers to "forget" concepts to estimate treatment effects, using rule-based rewriting.
- **Polyjuice** (Wu et al., 2021): Trains specialized models to generate diverse counterfactuals.
- **RewardBench** (Lambert et al., 2024): A non-causal evaluation benchmark for RMs, which is complementary to the causal framework of RATE.
- RATE can be seen as a paradigm shift in the field of causal explainability, moving from "rule-based rewriting" to "LLM-based rewriting + debiasing".

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The idea of double-rewriting debiasing is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Thorough validation across semi-synthetic and real RMs, but lacks downstream alignment experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, intuitive diagrams, and tightly integrated theory and experiments.
- Value: ⭐⭐⭐⭐ — Substantial contribution to the field of RM interpretability; the finding that "length bias is an artifact" has practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[CVPR 2026\] Retrieving Counterfactuals Improves Visual In-Context Learning](../../CVPR2026/causal_inference/retrieving_counterfactuals_improves_visual_in-context_learning.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ICML 2025\] E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time](e-lda_toward_interpretable_lda_topic_models_with_strong_guarantees_in_logarithmi.md)
- [\[ICLR 2026\] Conformalized Survival Counterfactuals Prediction for General Right-Censored Data](../../ICLR2026/causal_inference/conformalized_survival_counterfactuals_prediction_for_general_right-censored_dat.md)

</div>

<!-- RELATED:END -->
