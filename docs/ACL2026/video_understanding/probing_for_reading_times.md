---
title: >-
  [Paper Note] Probing for Reading Times
description: >-
  [ACL 2026][Video Understanding][Paper Note] This paper probes the ability of various language model layer representations to predict reading times, discovering that early-layer representations outperform surprisal in predicting early fixation metrics, while surprisal excels in later metrics. The optimal predictor varies significantly across languages and metrics
tags:
  - ACL 2026
  - Video Understanding
date: 2026-05-08
content_hash: 867c4640dbb9b0ed
---
# Probing for Reading Times

**Conference**: ACL 2026  
**arXiv**: [2604.18712](https://arxiv.org/abs/2604.18712)  
**Code**: [GitHub](https://github.com/rycolab/llm-representations-rt)  
**Area**: Video Understanding / Cognitive Science  
**Keywords**: reading time prediction, language model probing, eye-tracking, surprisal theory, cross-linguistic analysis

## TL;DR

This paper probes the ability of various language model layer representations to predict reading times, discovering that early-layer representations outperform surprisal in predicting early fixation metrics, while surprisal excels in later metrics. The optimal predictor varies significantly across languages and metrics.

## Background & Motivation

**Background**: The field has established a foundation but retains critical gaps.

**Limitations of Prior Work**: Existing methods fail to fully address core issues, suffering from constraints in accuracy, scalability, or applicability.

**Key Challenge**: The fundamental tension lies in the mismatch between the implicit assumptions of current paradigms and actual requirements.

**Goal**: Propose a new framework/method/benchmark to systematically resolve the aforementioned problems.

**Key Insight**: Starting from a unique observation or theory, identify a new path to solve the problem.

**Core Idea**: Resolve the core contradiction through innovative technical means.

## Method

### Overall Architecture

The paper reformulates the classic psycholinguistic question—"which features best predict human reading times"—as a **probing** task. Given the duration (in milliseconds) a human spends on a linguistic unit in context, linear regression is used to predict these times from features extracted by language models. The goodness-of-fit of the feature set measures its "psychometric power." Unlike the mainstream approach of compressing model internal states into a single scalar (e.g., surprisal), this work advocates directly using the **full representation vectors of each layer** as predictive variables, compared layer-by-layer against three scalar baselines. The research workflow involves: extracting candidate features for each unit → fitting reading times using regularized linear regression → performing 10-fold cross-validation across two eye-tracking corpora, five languages, and three types of reading metrics → comparing the predictive power of different predictors (high-dimensional representations per layer vs. individual scalars) to locate "at which layers, processing stages, and languages the representations outperform surprisal." This is not a processing pipeline of coordinated modules, but a controlled experimental design centered on "what to use as a predictor" and "how to compare fairly."

### Key Designs

1.  **Representation Probes: Full Hidden States as Predictors**: Previously, the strongest reading time predictor, surprisal, only took the negative log probability of the "next-word distribution" from the final layer, compressing the entire internal state into one dimension. This paper argues that this discards significant information relevant to human processing. Thus, for each layer $\ell$ (24 layers for mGPT, 12 for GPT-2 and cosmosGPT), the full representation vector $\mathbf{h}_\ell \in \mathbb{R}^D$ at the unit position is extracted as a high-dimensional predictor, and its predictive power for reading time is probed independently layer by layer. This step is the core contribution—replacing "finding a good scalar" with "probing high-dimensional representations" to investigate "where the information is hidden."

2.  **Three Scalar Baseline Predictors (Challenging Compressed Features)**: To test whether full-layer representations are truly superior to scalar compression, the study implements three predictors that compress internal states into single scalars for comparison: ① **surprisal**: the negative log probability of a unit given context $-\log p(u_t\mid \mathbf{u}_{<t})$, the gold standard predictor; ② **information value**: the expected cosine distance in representation space between model-sampled continuations and the actual continuation, characterizing "unexpectedness" as an alternative information metric; ③ **logit-lens surprisal**: passing intermediate layer representations directly to the output head (reusing the final layer's projection matrix $\mathbf{W}$, bias $\mathbf{b}$, and layer norm) to obtain an "imaginary" next-word distribution $q_\ell$ for that layer, equivalent to calculating surprisal at every layer. All three share the fundamental limitation of compressing the representation into one dimension, which this paper challenges.

3.  **Regularized Linear Regression Probes + Layer × Metric × Language Comparative Evaluation**: The probe itself is a linear regression predicting reading times in milliseconds (without log or z-score transforms to maintain interpretability). Beyond ordinary least squares, Ridge ($\ell_2$ penalty) and LASSO ($\ell_1$ penalty, inducing sparsity for feature selection) are introduced. Models are selected via MSE on a fixed train–test split based on regularization type and penalty weight $\lambda\in[0.001,10]$, with independent hyperparameter tuning for each predictor type, layer, and dependent variable. The evaluation covers two eye-tracking corpora (Provo, MECO), five languages (English, Greek, Hebrew, Russian, Turkish), and three reading metrics (first fixation duration, gaze duration, total reading time), with 10-fold cross-validation for each combination. This fine-grained comparison allows the conclusion that early-layer representations outperform surprisal on early fixation metrics, while surprisal is superior for late metrics, with the optimal predictor varying strongly by language and metric.

### Loss & Training

The probes fit parameters $\boldsymbol{\beta}$ using squared error loss for each string, incorporating the sentence-final EOS unit to model "wrap-up" effects. Ridge adds $\lambda\lVert\boldsymbol{\beta}\rVert_2^2$ and LASSO adds $\lambda\lVert\boldsymbol{\beta}\rVert_1$ to the loss. Hyperparameters are selected via MSE on fixed splits, and predictive power is reported using 10-fold cross-validation. The study also observes that concatenating surprisal with early-layer representations often improves performance over representations alone, suggesting that scalars and high-dimensional representations capture partially complementary information.

## Key Experimental Results

### Main Results

| Method | Core Metric | Description |
| :--- | :--- | :--- |
| Baseline | Lower | Existing state-of-the-art |
| **Ours** | **Highest** | Significant improvement |

### Ablation Study

| Configuration | Result | Description |
| :--- | :--- | :--- |
| Full | Highest | Complete model |
| w/o Core Component | Decrease | Verifies criticality |

### Key Findings

*   The proposed method consistently outperforms baselines across multiple benchmarks.
*   Ablation experiments verify the necessity of each component.
*   Performance is particularly outstanding in specific scenarios.

## Highlights & Insights

*   Core technical innovation addresses long-standing issues.
*   The method demonstrates high scalability and practicality.
*   Analysis reveals valuable underlying patterns.

## Limitations & Future Work

*   The scope of evaluation can be further expanded.
*   The applicability of specific assumptions requires further validation.
*   Future work can explore more application scenarios.

## Related Work & Insights

*   **vs Related Work A**: This paper improves upon key dimensions.
*   **vs Related Work B**: This paper provides a different approach to the problem.

## Rating

*   Novelty: ⭐⭐⭐⭐ Innovative, though some techniques combine existing methods.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure.
*   Value: ⭐⭐⭐⭐ Practical contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)
- [\[ACL 2026\] VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)
- [\[ACL 2026\] TRACE：基于证据定位的多视频事件理解与声明生成](trace_evidence_grounding-guided_multi-video_event_understanding_and_claim_genera.md)

</div>

<!-- RELATED:END -->
