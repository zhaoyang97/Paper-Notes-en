---
title: >-
  [Paper Note] How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs
description: >-
  [NeurIPS 2025][Multilingual & Machine Translation][In-context learning] Under a high-dimensional asymptotic framework, this paper proves that Transformers with nonlinear MLP heads are asymptotically equivalent to structu…
tags:
  - "NeurIPS 2025"
  - "Multilingual & Machine Translation"
  - "In-context learning"
  - "data mixing"
  - "high-dimensional asymptotics"
  - "polynomial equivalence"
  - "feature learning"
date: 2026-05-08
content_hash: 77913215ae7ce548
---

# How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs

**Conference**: NeurIPS 2025
**arXiv**: [2510.25753](https://arxiv.org/abs/2510.25753)  
**Code**: [GitHub](https://github.com/KU-MLIP/Data-Mixing-Shapes-ICL-by-Transformers)  
**Area**: Multilingual Translation
**Keywords**: In-context learning, data mixing, high-dimensional asymptotics, polynomial equivalence, feature learning

## TL;DR
Under a high-dimensional asymptotic framework, this paper proves that Transformers with nonlinear MLP heads are asymptotically equivalent to structured polynomial predictors in terms of ICL error, revealing the gain mechanism of nonlinear MLPs on nonlinear tasks and establishing that low noise and structured covariance are key characteristics of high-quality data sources in multi-source data mixing.

## Background & Motivation

**Background**: Theoretical studies on ICL (in-context learning) have been extensive, but most rely on simplifying assumptions: (a) omitting MLPs and using attention only, (b) considering only linear regression tasks with isotropic inputs, and (c) single-source training. These limitations create a substantial gap between theoretical results and actual Transformer behavior.

**Limitations of Prior Work**:
   - The role of MLPs in Transformers has not been adequately characterized theoretically. The few works that study MLPs either adopt non-standard architectures (MLP before attention) or are restricted to specific activation functions or classification tasks.
   - In practice, Transformers are pretrained on multiple heterogeneous data sources, and the quality and mixing ratios of these sources significantly affect ICL performance, yet a theoretical analysis framework is lacking.
   - When feature learning emerges in ICL and how it relates to data structure remain theoretically unclear.

**Key Challenge**: Empirical evidence shows that nonlinear MLPs are critical to Transformer performance, yet theoretical analysis is difficult due to MLP nonlinearity. The mixing effects of multi-source training further complicate theoretical analysis.

**Goal**: (1) How does a nonlinear MLP improve ICL in the high-dimensional asymptotic limit? (2) How do the mixing ratios and data quality of multiple sources affect ICL performance and feature learning?

**Key Insight**: By leveraging Gaussian Universality theory and orthogonal polynomial analysis, the ICL error of Transformer+MLP is precisely equated to that of a finite-degree polynomial model, making theoretical analysis tractable.

**Core Idea**: Under a high-dimensional proportional limit, a Transformer with a two-layer MLP head is equivalent to a structured polynomial predictor for ICL, and this equivalence reveals the gain mechanism of the MLP and the role of data mixing.

## Method

### Overall Architecture
A standard Transformer architecture is considered: linear attention with a two-layer nonlinear MLP head. The input consists of $\ell$ in-context samples $(x_i, y_i)$ and a query $x_{\ell+1}$, with the goal of predicting $y_{\ell+1}$. Training data come from $\mathcal{S}$ heterogeneous sources, each with distinct input covariance $\Sigma_{x,s}$, task covariance $\Sigma_{\xi,s}$, and noise variance $\Delta_s^2$.

### Key Designs

1. **High-Dimensional Proportional Limit Setting**:

    - Function: Establishes the asymptotic analysis framework.
    - Mechanism: The input dimension $d$, context length $\ell$, number of training samples $n$, and hidden dimension $k$ all tend to infinity simultaneously, with the ratios $\ell/d$, $n/d^2$, and $k/n$ held constant. This scaling is carefully chosen: $\ell/d$ and $n/d^2$ are critical for ICL performance of linear Transformers, while $k/n$ ensures model capacity matches data volume.
    - Design Motivation: Only under this proportional limit does the Gaussian Universality theorem guarantee the precise equivalence between Transformers and polynomial models.

2. **Two-Stage Training (Feature Learning + Ridge Regression)**:

    - Function: Introduces genuine feature learning while maintaining analytical tractability.
    - Mechanism: Stage 1 — a single gradient descent step is performed on the MLP first layer $F$ (which fuses attention parameters and the MLP first layer); Stage 2 — ridge regression is applied to the MLP second layer $w$ on a new independent dataset. The step size scaling $\eta = o(d^2)$ controls the strength of feature learning.
    - Design Motivation: Two-stage training makes $\hat{F}$ independent of the Stage 2 training set, simplifying theoretical analysis; meanwhile, a single gradient update suffices to introduce nontrivial feature learning.

3. **Asymptotic Equivalence Theorem**:

    - Function: Proves that Transformer+MLP is equivalent to a finite-degree polynomial model in ICL error.
    - Mechanism: Applying Gaussian Universality, the nonlinear activation $\sigma$ is expanded as a finite Hermite polynomial series: $\sigma(x) \approx \sum_{j=0}^{P} c_j H_j(x)$. Under the proportional limit, contributions from higher-order terms vanish, and the ICL error of the Transformer exactly equals that of a linear model using the same polynomial basis functions.
    - Mathematical Guarantee: The difference in ICL error converges to zero in probability.

4. **Decisive Role of Task Covariance Structure on Feature Learning**:

    - Finding: The gain from feature learning depends on the structure of the task covariance $\Sigma_{\xi,s}$. When $\Sigma_{\xi,s}$ is isotropic, increasing the step size $\eta$ does not improve performance; when $\Sigma_{\xi,s}$ has a low-rank structure, increasing $\eta$ significantly reduces ICL error.
    - Intuition: The first layer of the MLP learns to align with task vectors through gradient updates — this alignment is meaningful only when the task distribution has structured directions.

### Loss & Training
- Stage 1: Single gradient descent step with MSE loss.
- Stage 2: Ridge regression (MSE with $\ell_2$ regularization), with a closed-form solution.
- ICL error is defined as the uniformly averaged MSE over all data sources.

## Key Experimental Results

### Main Results

**Nonlinear MLP vs. Linear Transformer (Synthetic Data)**:

| Model | ICL Error (Darcy Task) | Improvement |
|------|----------|------|
| Linear Transformer (no MLP) | ~0.85 | baseline |
| Transformer + MLP (degree 1) | ~0.75 | 11.8% |
| Transformer + MLP (degree 3) | ~0.55 | 35.3% |
| Transformer + MLP (degree 5) | ~0.45 | **47.1%** |
| Equivalent Polynomial Model (degree 5) | ~0.45 | Exact match |

### Ablation Study

| Change in Data Source Properties | Change in ICL Error | Notes |
|--------------|-----------|------|
| Increase proportion of structured input covariance | Significant decrease | Structured input = high-quality data |
| Increase proportion of structured task covariance | Significant decrease | Task structure is the most critical factor |
| Decrease noise variance of second source | Significant decrease | Low noise = high quality |
| Increase step size η (isotropic task) | No improvement | Feature learning ineffective without structure |
| Increase step size η (structured task) | Significant decrease | Feature learning effective with structure |

### Key Findings
- **Asymptotic equivalence is highly accurate**: The ICL error curves of Transformer+MLP and the equivalent polynomial model nearly perfectly coincide across all experimental settings (different activation functions ReLU/GeLU, different model sizes, different data distributions).
- **Three factors characterizing high-quality data sources**: (1) low target noise, (2) structured input covariance, and (3) structured task covariance.
- **Feature learning requires task structure**: Structured inputs alone are insufficient; the task vector distribution must have a low-rank structure for the MLP to learn useful features.
- **Real-world validation (multilingual sentiment analysis)**: English and German are treated as two separate data sources; validation on Amazon Reviews confirms theoretical predictions — the equivalent model accurately fits actual Transformer behavior, with performance improving as the proportion of English increases (as the embedding model is stronger for English).

## Highlights & Insights
- **First application of Gaussian Universality to ICL theory**: This work successfully incorporates the Transformer architecture into a high-dimensional asymptotic analysis framework, establishing precise equivalence with polynomial models and providing a novel theoretical tool for understanding Transformer ICL.
- **Interaction analysis of feature learning and data mixing**: The paper reveals a counterintuitive finding — structured inputs (e.g., low-rank covariance of natural images) are insufficient to trigger feature learning; the task distribution itself must be structured. This has practical implications for pretraining data selection.
- **Theory-practice bridge**: Asymptotic equivalence reduces complex Transformer behavior to analytically tractable polynomial models, enabling precise computation of data mixing effects.

## Limitations & Future Work
- **Single-layer attention + single-layer MLP**: Theoretical analysis is limited to the simplest architecture; multi-layer attention and multi-layer MLPs remain important open problems.
- **Linear attention**: Linear attention is used instead of softmax attention to ensure tractability, but the nonlinearity of softmax may introduce additional effects.
- **Single-step gradient training**: Actual training involves multi-step end-to-end optimization; the single-step gradient + ridge regression setting is a theoretical simplification.
- **Gaussian input assumption**: The theory relies on Gaussian inputs; equivalence under non-Gaussian inputs such as natural language requires further verification.
- **Applicability of the proportional limit**: The quality of asymptotic approximation under finite dimensions depends on the problem scale.

## Related Work & Insights
- **vs. Zhang et al. (linear Transformer ICL theory)**: The prior work addresses linear tasks with linear attention and no MLP. This paper extends the analysis to nonlinear tasks with MLPs.
- **vs. Kim & Suzuki, Oko et al.**: These works study MLPs but use non-standard architectures (MLP before attention) or are restricted to specific activation functions. This paper uses a standard architecture with general activation functions.
- **vs. Ba et al. (feature learning theory)**: Ba et al. analyze single-step gradient feature learning in standard MLPs; this paper extends their framework to the Transformer ICL setting.

## Rating
- Novelty: ⭐⭐⭐⭐ First to establish high-dimensional asymptotic ICL equivalence for Transformer+MLP; significant theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic and real data, across multiple activation functions and configurations.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous and clearly structured, though the barrier for non-theoretical readers is high.
- Value: ⭐⭐⭐⭐ Deepens theoretical understanding of ICL and offers practical guidance for data mixing strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[NeurIPS 2025\] Quantifying Climate Policy Action and Its Links to Development Outcomes: A Cross-National Data-Driven Analysis](quantifying_climate_policy_action_and_its_links_to_development_outcomes_a_cross-.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](../../ACL2026/multilingual_mt/clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation](../../ACL2026/multilingual_mt/syntax_as_a_rosetta_stone_universal_dependencies_for_in-context_coptic_translati.md)
- [\[AAAI 2026\] How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](../../AAAI2026/multilingual_mt/how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)

</div>

<!-- RELATED:END -->
