---
title: >-
  [Paper Note] Scaling Up Active Testing to Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][active testing] By introducing three key simplifications—constructing a fixed surrogate model via in-context learning, using a small surrogate model to evaluate a large target model…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "active testing"
  - "LLM evaluation"
  - "risk estimation"
  - "surrogate model"
  - "label efficiency"
date: 2026-05-08
content_hash: 45136d23ec8130bf
---

# Scaling Up Active Testing to Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2508.09093](https://arxiv.org/abs/2508.09093)  
**Code**: [GitHub](https://github.com/gabrielleberrada/scaling-up-active-testing)  
**Area**: LLM/NLP
**Keywords**: active testing, LLM evaluation, risk estimation, surrogate model, label efficiency

## TL;DR
By introducing three key simplifications—constructing a fixed surrogate model via in-context learning, using a small surrogate model to evaluate a large target model, and eliminating the need for target model predictions during data acquisition—this work scales active testing to LLMs, reducing risk estimation error by 25%–80% relative to random sampling.

## Background & Motivation

**Background**: Frontier models are increasingly complex; annotation costs are high and evaluation data may leak into training sets, necessitating continuous and dynamic collection of new evaluation data.

**Limitations of Prior Work**: Existing active testing methods require iterative gradient-based retraining of surrogate models and inference over the entire pool with both surrogate and target models, making computational costs prohibitive for LLM-scale applications.

**Core Problem**: How to substantially reduce computational cost while preserving the effectiveness of active testing, enabling it to scale to 70B-parameter LLMs.

## Method

### Overall Architecture

Active testing improves risk estimation $R = \mathbb{E}[\ell(f(x), y)]$ by intelligently selecting which test inputs to annotate. This paper addresses three computational bottlenecks.

### Key Design 1: Fixed Surrogate Model (Resolving the Training Bottleneck)

Conventional methods retrain the surrogate model after each newly acquired label. Instead, this work:
- Constructs the surrogate model **once** using a small set of initial labeled data via **in-context learning**
- Keeps the surrogate model **fixed** thereafter
- Completely eliminates the in-loop gradient training overhead

### Key Design 2: Small Surrogate Model (Resolving the Inference Cost Bottleneck)

The surrogate model can be substantially smaller than the target model:
- A 7B model serves as the surrogate to evaluate a 70B target model
- Models as small as Gemma3-4B or Phi-2 can evaluate Llama-2-70B
- Older models (Llama 2) can effectively serve as surrogates for newer models (Gemma 3)

### Key Design 3: No Target Model Predictions Required (Resolving the $N$-Inference Bottleneck)

The surrogate model is used to approximate target model predictions:
- The acquisition function is simplified from the cross-entropy $H[\pi_m(y|x) \| p_f(y|x)]$ to the surrogate model's predictive entropy $H[\pi_m(y|x)]$
- The number of target model inference calls is reduced from $N$ to $M$ ($M \ll N$)

### Risk Estimation

Unbiased risk estimation is performed using LURE (Levelled Unbiased Risk Estimator):

$$\hat{R}_{\text{LURE}} = \frac{1}{M}\sum_{m=1}^M v_m \ell(f(x_{i_m}), y_{i_m})$$

where $v_m$ are importance weights that correct for non-uniform sampling.

### Bootstrap Error Estimation

A bootstrap method is proposed to estimate risk estimation error within a single active testing run, providing confidence intervals for practical deployment.

## Key Experimental Results

### Main Results: Reduction in Risk Estimation Error

| Dataset | Target Model | Surrogate Model | Relative Error Reduction |
|---------|-------------|-----------------|--------------------------|
| SST-2 | 70B-few | 7B-few | ~50% |
| FPB | 70B-few | 7B-few | ~40% |
| HS | 70B-few | 7B-few | ~60% |
| Subj | 70B-few | 7B-few | ~30% |
| Average | — | — | **25%–80%** |

### Cross-Model Surrogate Evaluation

| Surrogate Model | Target Model | Effectiveness |
|----------------|-------------|---------------|
| Llama-2-7B | Llama-2-70B | Effective |
| Gemma3-4B | Llama-2-70B | Effective |
| Phi-2 | Llama-2-70B | Effective |
| Llama-2-7B | Gemma3-4B | Effective |

### Sampling vs. Interpolation Methods

| Method | Robustness | Notes |
|--------|-----------|-------|
| LURE (sampling) | High | Insensitive to surrogate model quality |
| ASE (interpolation) | Low | Sensitive to surrogate quality; degrades with a fixed surrogate |

This confirms the theoretical prediction that sampling-based methods outperform interpolation-based methods when the surrogate is fixed.

### Bootstrap Error Estimation

The 95% confidence interval achieves a true error coverage rate of **88%** (reaching ~94% when $K \geq 100$).

### Key Findings

1. Forgoing surrogate model updates incurs negligible performance loss while saving substantial computational cost.
2. The surrogate model can be 10× smaller than the target model, or even more.
3. The acquisition function that eliminates target model predictions (replacing cross-entropy with predictive entropy) performs surprisingly well.
4. Label noise can severely impair active testing—a stronger surrogate model may paradoxically perform worse in such cases.

## Highlights & Insights

1. **Elegant resolution of three bottlenecks**: Each simplification is theoretically motivated and empirically validated.
2. **Counter-intuitive finding**: Using a smaller surrogate to evaluate a larger target model can yield equal or better performance.
3. **Bootstrap diagnostic tool**: Provides a practical criterion for determining whether active testing is functioning effectively in deployment.
4. **Dataset curation as a by-product**: Active testing can be applied to dataset curation, selecting subsets for model evaluation.

## Limitations & Future Work

1. Experiments are limited to text classification; generative tasks present greater complexity.
2. Active testing may fail under label noise, as demonstrated by the SST-2 case.
3. Improvements on the more challenging MMLU dataset are modest.
4. Theoretical convergence guarantees for the bootstrap estimator are lacking.

## Related Work & Insights

- **Kossen et al. (2021, 2022)**: Proposed sampling- and interpolation-based active testing methods; this work extends them to the LLM setting.
- **TinyBenchmarks/DELE**: Dataset compression methods focused on cross-model generalizability; this work performs model-specific acquisition instead.
- **Insight**: Active testing can be integrated with continuous benchmark updates to mitigate data contamination issues.

## Rating

- Novelty: ⭐⭐⭐⭐ Extends existing methods to LLMs; each simplification is well-motivated but not fundamentally novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, datasets, and settings, including analysis of failure cases.
- Writing Quality: ⭐⭐⭐⭐⭐ The three-bottleneck analytical framework is clearly articulated; experimental design is systematic.
- Value: ⭐⭐⭐⭐ Offers practical utility for improving LLM evaluation efficiency, particularly in annotation-expensive scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](solving_inequality_proofs_with_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[ICML 2026\] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](../../ICML2026/llm_nlp/resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[NeurIPS 2025\] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)

</div>

<!-- RELATED:END -->
