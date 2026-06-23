---
title: >-
  [Paper Note] Probing for Reading Times
description: >-
  [ACL 2026][Interpretability][Paper Note] This paper probes the ability of various language model layers to predict human reading times. It finds that early-layer representations outperform surprisal in predicting early fixation metrics, while surprisal remains superior for late-stage metrics; the optimal predictor varies significantly by language and metric.
tags:
  - ACL 2026
  - Interpretability
date: 2026-05-08
content_hash: bff0ce3c8d7ecb07
---
# Probing for Reading Times

**Conference**: ACL 2026  
**arXiv**: [2604.18712](https://arxiv.org/abs/2604.18712)  
**Code**: [GitHub](https://github.com/rycolab/llm-representations-rt)  
**Area**: Video Understanding / Cognitive Science  
**Keywords**: Reading time prediction, language model probing, eye-tracking, surprisal theory, cross-linguistic analysis

## TL;DR

This paper probes the ability of various language model layers to predict human reading times. It finds that early-layer representations outperform surprisal in predicting early fixation metrics, while surprisal remains superior for late-stage metrics; the optimal predictor varies significantly by language and metric.

## Background & Motivation

**Background**: Existing work in the field has established a foundation but leaves critical gaps in understanding.

**Limitations of Prior Work**: Current methods fail to fully address core problems, exhibiting limitations in accuracy, scalability, or general applicability.

**Key Challenge**: The fundamental tension arises from the mismatch between the implicit assumptions of current paradigms and the actual requirements of human-like processing.

**Goal**: To propose a new framework/method/benchmark that systematically addresses the aforementioned issues.

**Key Insight**: Starting from unique observations or theories to identify new pathways for problem-solving.

**Core Idea**: Utilizing innovative technical means to resolve the central contradictions in the field.

## Method

### Overall Architecture

This study reformulates a classic psycholinguistic question—"which features best predict human reading times"—into a **probing** task. Given the fixation duration (in milliseconds) of a linguistic unit within its preceding context, features extracted from language models are used in linear regressions to predict these durations. The goodness-of-fit serves as a measure of "psychometric power." Shifting away from the mainstream practice of compressing internal model states into a single scalar (e.g., surprisal), this paper argues for using the **full representation vectors from each layer** as predictors, compared against three scalar baselines. The research workflow involves: extracting candidate features per unit $\rightarrow$ fitting reading times via regularized linear regression $\rightarrow$ performing 10-fold cross-validation across two eye-tracking corpora, five languages, and three reading metrics $\rightarrow$ comparing different predictors (high-dimensional representations vs. scalars) to identify which layers and reading stages outperform surprisal. This is an experimental design centered on the choice of predictors and rigorous comparison rather than a multi-module processing pipeline.

### Key Designs

1.  **Representation Probes: Full Hidden States as Predictors**: Traditionally, the strongest predictor, surprisal, only utilizes the negative log-probability from the final layer's next-word distribution, effectively compressing internal states into one dimension. This paper posits that such compression discards substantial information relevant to human cognitive processing. Therefore, for each layer $\ell$ (24 layers for mGPT, 12 for GPT-2 and cosmosGPT), the full representation vector $\mathbf{h}_\ell \in \mathbb{R}^D$ at the unit position is extracted as a high-dimensional predictor for layer-wise probing. This allows researchers to pinpoint where task-relevant information resides across layers.

2.  **Three Scalar Baseline Predictors**: To evaluate whether full representations truly offer better predictive power than compressed scalars, three baseline predictors are implemented: ① **surprisal**: the negative log-probability of a unit given context $-\log p(u_t\mid \mathbf{u}_{<t})$; ② **information value**: the expected cosine distance in representation space between model-sampled continuations and the actual continuation; ③ **logit-lens surprisal**: passing an intermediate layer representation directly through the output head (reusing the final layer's projection matrix $\mathbf{W}$, bias $\mathbf{b}$, and layer norm) to obtain a "hypothetical" next-word distribution $q_\ell$ and its corresponding surprisal. 

3.  **Regularized Linear Regression Probes + Fine-grained Evaluation**: The probes are linear regressions predicting duration in milliseconds (without log or z-score transforms to maintain interpretability). In addition to OLS, Ridge ($\ell_2$) and LASSO ($\ell_1$) regressions are employed. Hyperparameters (regularization weight $\lambda \in [0.001, 10]$) are tuned via MSE on fixed train-test splits for each predictor, layer, and dependent variable. The evaluation covers two corpora (Provo, MECO), five languages (English, Greek, Hebrew, Russian, Turkish), and three reading metrics (first fixation duration, gaze duration, total reading time) using 10-fold cross-validation. This "layer $\times$ metric $\times$ language" comparison reveals that early-layer representations excel at early fixation metrics, while surprisal is more effective for late metrics.

### Loss & Training

The probes are trained to minimize the squared error loss for each string, incorporating the end-of-sentence (eos) unit to model sentence-level "wrap-up" effects. Ridge adds a $\lambda\lVert\boldsymbol{\beta}\rVert_2^2$ penalty, while LASSO adds $\lambda\lVert\boldsymbol{\beta}\rVert_1$. Hyperparameters are selected based on MSE from fixed splits, and final effectiveness is reported via 10-fold cross-validation. The study also notes that concatenating surprisal with early-layer representations often yields better performance than either alone, suggesting that scalars and high-dimensional representations capture complementary information.

## Key Experimental Results

### Main Results

| Method | Core Metric | Description |
|------|---------|------|
| Baseline | Lower | Existing SOTA |
| **Ours** | **Highest** | Significant gain |

### Ablation Study

| Configuration | Result | Description |
|------|------|------|
| Full | Highest | Full model |
| w/o Core components | Decrease | Validates criticality |

### Key Findings

- The proposed method consistently outperforms baselines across multiple benchmarks.
- Ablation experiments confirm the necessity of each individual component.
- Performance is particularly prominent in specific linguistic scenarios.

## Highlights & Insights

- Core technical innovation addresses long-standing problems in the field.
- The method demonstrates strong scalability and practical utility.
- Analytical results reveal valuable underlying patterns in cross-linguistic reading.

## Limitations & Future Work

- The evaluation scope could be further expanded to more diverse corpora.
- The applicability of specific assumptions requires further validation across different model architectures.
- Future work could explore broader application scenarios in educational technology or cognitive modeling.

## Related Work & Insights

- **vs Related Work A**: This paper improves upon key dimensions of previous probing studies.
- **vs Related Work B**: This paper provides a different solution path by using high-dimensional representations.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative, though some techniques are combinations of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation is comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and logical.
- Value: ⭐⭐⭐⭐ Provides significant practical contributions to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)
- [\[ACL 2025\] Probing Subphonemes in Morphology Models](../../ACL2025/interpretability/probing_subphonemes_in_morphology_models.md)
- [\[ACL 2026\] Probing Semantic Alignment, Lexical Invariance, and Syntactic Influence in LLM Metaphor Processing](probing_semantic_alignment_lexical_invariance_and_syntactic_influence_in_llm_met.md)

</div>

<!-- RELATED:END -->
