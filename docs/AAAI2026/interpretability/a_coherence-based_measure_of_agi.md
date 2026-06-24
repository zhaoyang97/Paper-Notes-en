---
title: >-
  [Paper Note] A Coherence-Based Measure of AGI
description: >-
  [AAAI 2026][Interpretability][AGI evaluation] This paper identifies that existing AGI scores rely on arithmetic averaging, which implicitly encodes a "compensatory" assumption (strengths offsetting weaknesses), and proposes $\text{AGI}_{\text{AUC}}$—a coherence measure based on the continuous spectrum of generalized means. By integrating over the compensability parameter $p \in [-1, 1]$, the metric penalizes uneven capability profiles and exposes bottlenecks concealed by arit…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "AGI evaluation"
  - "generalized mean"
  - "coherence measure"
  - "cognitive capability balance"
  - "non-compensatory aggregation"
date: 2026-05-08
content_hash: 655633db308d0be5
---

# A Coherence-Based Measure of AGI

**Conference**: AAAI 2026
**arXiv**: [2510.20784](https://arxiv.org/abs/2510.20784)  
**Code**: Not available  
**Area**: Interpretability
**Keywords**: AGI evaluation, generalized mean, coherence measure, cognitive capability balance, non-compensatory aggregation

## TL;DR

This paper identifies that existing AGI scores rely on arithmetic averaging, which implicitly encodes a "compensatory" assumption (strengths offsetting weaknesses), and proposes $\text{AGI}_{\text{AUC}}$—a coherence measure based on the continuous spectrum of generalized means. By integrating over the compensability parameter $p \in [-1, 1]$, the metric penalizes uneven capability profiles and exposes bottlenecks concealed by arithmetic averaging.

## Background & Motivation

**Background**: Hendrycks et al. define an AGI score as the arithmetic mean of scores across 10 cognitive domains based on the CHC (Cattell-Horn-Carroll) theory of cognitive abilities. GPT-4 scores 27% and GPT-5 scores 58% under this scheme. However, arithmetic averaging implicitly encodes a **compensatory assumption**—strong reasoning can compensate for weak memory.

**Limitations of Prior Work**: Psychometric evidence from CHC theory argues against compensability: cognitive abilities are interdependent (reasoning relies on working memory; perception constrains abstraction), and extreme imbalance typically indicates dysfunction rather than high intelligence.

**Key Challenge**: Systems theory supports a bottleneck effect—the overall capability of a complex system is constrained by its weakest component (limiting-factor dynamics), which simple summation cannot capture.

**Goal**: General intelligence should exhibit **coherent sufficiency**—all critical capabilities meeting a balanced threshold—rather than excelling in isolated domains.

## Method

### Overall Architecture

The degree of compensability is parameterized via the generalized power mean family, and a robust coherence measure is derived by integrating over the AUC:

$$\text{AGI}_p = \begin{cases} \left(\frac{1}{n}\sum_{i=1}^n \max(s_i, \varepsilon)^p\right)^{1/p}, & p \neq 0 \\ \left(\prod_{i=1}^n \max(s_i, \varepsilon)\right)^{1/n}, & p = 0 \end{cases}$$

$$\text{AGI}_{\text{AUC}} = \frac{1}{p_{\max} - p_{\min}} \int_{p_{\min}}^{p_{\max}} \text{AGI}_p \, dp$$

### Key Designs

- **Semantics of the compensability index $p$**: $p=1$ (arithmetic mean, strong compensation) → $p=0$ (geometric mean, moderate non-compensation) → $p=-1$ (harmonic mean, strong non-compensation) → $p \to -\infty$ (minimum, strict bottleneck).
- **$\text{AGI}_p$ curve**: The horizontal axis represents $p$ and the vertical axis represents the score. A flatter and higher curve indicates more balanced capabilities. GPT-5's curve drops sharply for $p < 0$, exposing bottlenecks in memory and perception.
- **AUC aggregation**: Integration over $p \in [-1, 1]$ provides a comprehensive measure of model robustness across different compensability assumptions.
- **Stability constant**: $\varepsilon = 10^{-6}$ prevents collapse of the generalized mean when any dimension equals zero.

### Loss & Training

This paper presents a purely evaluative framework with no model training. Numerical integration is performed via the composite trapezoidal rule on a uniform grid over $p$.

## Key Experimental Results

### CHC Domain Score Analysis (GPT-4 / GPT-5 / Ideal AGI)

| Model | $\text{AGI}_1$ (Arithmetic) | $\text{AGI}_{0.5}$ | $\text{AGI}_0$ (Geometric) | $\text{AGI}_{-0.5}$ | $\text{AGI}_{-1}$ (Harmonic) | $\text{AGI}_{\text{AUC}}$ (Ours) |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT-4 | 27% | 16% | ≈0% | ≈0% | ≈0% | **7%** |
| GPT-5 | 58% | 50% | 16% | ≈0% | ≈0% | **24%** |
| Ideal AGI | 100% | 100% | 100% | 100% | 100% | **100%** |

### GPT-4/5 Scores Across Ten Domains

| Domain | Knowledge | Literacy | Math | Reasoning | Working Memory | LT Memory Storage | LT Memory Retrieval | Visual | Auditory | Speed |
|----|:----:|:----:|:----:|:----:|:--------:|:------------:|:------------:|:----:|:----:|:----:|
| GPT-4 | 80 | 60 | 40 | 0 | 20 | 0 | 40 | 0 | 0 | 30 |
| GPT-5 | 90 | 100 | 100 | 70 | 50 | **0** | 40 | 40 | 60 | 30 |

### Key Findings

- **Arithmetic averaging substantially overstates AGI progress**: GPT-5's $\text{AGI}_1 = 58\%$ creates the illusion that the field is "more than halfway there," whereas $\text{AGI}_{\text{AUC}} = 24\%$ more faithfully reflects actual capability.
- **Zero-score domains cause collapse of geometric/harmonic means**: GPT-5's long-term memory storage score of 0% drives $\text{AGI}_0$ down to 16% and $\text{AGI}_{-1} \approx 0\%$.
- **Consistency with external benchmarks**: $\text{AGI}_{\text{AUC}} = 24\%$ for GPT-5 aligns more closely with ARC-AGI-2's 18% than with the arithmetic mean of 58%.
- **"GPT-6" simulation**: Raising only GPT-5's weakest domain (long-term memory storage) from 0% to 30% produces a substantial improvement in $\text{AGI}_{\text{AUC}}$, demonstrating that patching bottlenecks yields far greater returns than strengthening existing capabilities.
- **17-benchmark extended validation**: Repeating the analysis on Gemini 3 Pro, GPT-5.1, Claude Sonnet 4.5, and other models using 17 heterogeneous benchmarks yields coherence patterns fully consistent with the CHC domain analysis.

## Highlights & Insights

1. **Primary conceptual contribution**: The paper frames "compensability" as the central unexamined assumption in AGI evaluation—a foundational issue previously overlooked by the community.
2. **The "GPT-6" thought experiment is particularly illuminating**: Patching only the weakest domain yields disproportionately large coherence gains, intuitively demonstrating the leverage effect of bottleneck remediation.
3. **The generalized mean spectrum is an elegant mathematical tool**: The continuous transition from full compensation ($p=1$) to strict bottleneck ($p \to -\infty$) means the $\text{AGI}_p$ curve itself serves as a diagnostic instrument.
4. **Practical implications**: If the AGI community were to adopt $\text{AGI}_{\text{AUC}}$, model development would shift toward addressing weaknesses rather than amplifying existing strengths.
5. **Framework agnosticism**: The approach is independent of any particular benchmark suite; any collection of domain scores can be aggregated under this framework.
6. **Alignment with ARC-AGI-2 and BIG-Bench Hard** validates that AUC more faithfully reflects functional coherence than arithmetic averaging.

## Limitations & Future Work

- **Dependence on domain score quality**: The normalization and estimation of CHC domain scores are themselves subject to bias (the paper discusses sub-domain inflation in an appendix).
- **Subjective choice of $p$ range**: The interval $[-1, 1]$ is an empirical choice; alternative ranges such as $[-2, 1]$ or $[-0.5, 1]$ would yield different results.
- **Handling of zero values via $\varepsilon$**: Replacing zero-score domains with $10^{-6}$ is mathematically defensible, but semantically a zero score on any capability should arguably render the AGI score zero.
- **Purely evaluative**: The framework diagnoses problems but provides no technical prescription for improving weak domains.
- **Equal domain weighting**: The 10 cognitive domains are treated with uniform weights, without accounting for differential contributions to general intelligence.
- **Absence of temporal dimension**: The framework provides a static snapshot and does not capture coherence dynamics over continual learning or forgetting.
- **Limited multi-model comparison**: The CHC domain analysis covers only GPT-4 and GPT-5; broader comparisons (e.g., Claude, Gemini) would strengthen the conclusions.

## Related Work & Insights

- **Hendrycks et al. (2025)**: The first psychometric AGI definition using a 10-domain arithmetic mean; this paper directly extends and improves upon that framework.
- **Chollet (ARC-AGI)**: Emphasizes out-of-distribution reasoning and abstraction, consistent with the non-compensatory philosophy of this work.
- **Multi-criteria decision theory** (Keeney & Raiffa): Provides the theoretical foundation for non-compensatory aggregation.
- **Bottleneck effects in systems theory** (Kitano): System performance is constrained by the weakest component.
- **BIG-Bench Hard**: GPT-4 scores approximately 6% on this benchmark, closely matching $\text{AGI}_{\text{AUC}} = 7\%$, as opposed to the arithmetic mean of 27%.
- **Gemini 3 Pro Model Evaluation Report**: The paper uses 17 benchmarks from this report for extended validation, demonstrating the generality of the framework.
- Core insight: **The design of an evaluation metric itself encodes assumptions about the nature of capability—choosing an aggregation function is choosing a theory of intelligence.**

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic introduction of the compensability problem into AGI evaluation
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation via CHC domains and 17 benchmarks, though dependent on external data with no original experiments
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically elegant, rigorously argued, and thoroughly discussed
- Value: ⭐⭐⭐⭐ The evaluation framework design approach is broadly applicable; generalized mean aggregation transfers well to multi-task evaluation settings

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder](sparserm_a_lightweight_preference_modeling_with_sparse_autoencoder.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](partially_shared_concept_bottleneck_models.md)
- [\[AAAI 2026\] Flexible Concept Bottleneck Model](flexible_concept_bottleneck_model.md)

</div>

<!-- RELATED:END -->
