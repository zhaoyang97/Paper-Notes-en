---
title: >-
  [Paper Note] AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference
description: >-
  [ICLR 2026 (Oral)][Video Understanding][LLM value assessment] This paper proposes AdAEM, an adaptive and self-extensible evaluation framework for LLM values. By leveraging information-theoretic optimization…
tags:
  - "ICLR 2026 (Oral)"
  - "Video Understanding"
  - "LLM value assessment"
  - "dynamic benchmark"
  - "information-theoretic optimization"
  - "Schwartz value theory"
  - "cultural differences"
date: 2026-05-08
content_hash: cf0a0f1b7819c037
---

# AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference

**Conference**: ICLR 2026 (Oral)  
**arXiv**: [2505.13531](https://arxiv.org/abs/2505.13531)  
**Code**: [https://github.com/ValueCompass/AdAEM](https://github.com/ValueCompass/AdAEM)  
**Area**: Video Understanding  
**Keywords**: LLM value assessment, dynamic benchmark, information-theoretic optimization, Schwartz value theory, cultural differences

## TL;DR
This paper proposes AdAEM, an adaptive and self-extensible evaluation framework for LLM values. By leveraging information-theoretic optimization, AdAEM automatically generates test questions that maximally reveal value differences across LLMs, addressing the "insufficient informativeness" limitation of existing static benchmarks that fail to distinguish models' value orientations.

## Background & Motivation

Although LLMs have achieved remarkable progress in knowledge and instruction following, they may still generate harmful, biased, or illegal content. Evaluating the intrinsic value orientations of LLMs has become an important approach for comprehensively diagnosing model misalignment, cultural adaptability, and bias.

**Limitations of Prior Work**: Existing value evaluation benchmarks suffer from an "insufficient informativeness" challenge — test questions are either outdated, contaminated, or overly generic, capable of capturing only the safety-oriented values shared across LLMs (e.g., HHH), leading to convergent and indistinguishable evaluation results. For example, on existing benchmarks SVS and ValueBench, GPT-4 and GLM-4 (developed in the US and China, respectively) exhibit nearly identical preferences on the hedonism dimension, which is clearly unreasonable.

**Key Challenge**: Static benchmarks cannot evolve alongside LLM development and are unable to explore contentious topics arising from cultural differences.

**Core Idea**: Design a self-extensible dynamic evaluation framework that automatically generates test questions eliciting value differences by probing the internal value boundaries of multiple LLMs from diverse cultures and time periods, theoretically maximizing an information-theoretic objective.

## Method

### Overall Architecture
**Input**: A collection of LLMs from different cultures/time periods + initial general social topics. **Output**: A value evaluation benchmark containing highly discriminative test questions. The pipeline consists of two core components: informativeness optimization (exploitation) and an exploration algorithm, forming a Multi-Armed Bandit-style iterative process.

### Key Designs

1. **Informativeness Optimization**:

    - The objective function is based on the generalized Jensen-Shannon divergence, maximizing the divergence in value distributions exhibited by different LLMs on a given question.
    - A "decoupling" regularization term is incorporated to prevent the value evaluation results from being dominated by the value orientation of the question itself.
    - An EM-style iterative optimization is adopted: the E-step (Response Generation) samples LLM opinions and selects the highest-scoring ones; the M-step (Question Refinement) fixes the opinions and optimizes the question to increase its informativeness.
    - Each evaluation step covers four dimensions: value conformity, value difference, semantic coherence, and semantic difference.

2. **Exploration Algorithm**:

    - Based on a Multi-Armed Bandit (MAB) variant, it adaptively decides whether to continue optimizing the current topic or explore new ones.
    - A UCB strategy selects the most promising topics for expansion and optimization.
    - A smaller, faster LLM set ($P_1$) is used for low-cost exploration, while a stronger LLM set ($P_2$) is used for final scoring.
    - A budget $B$ controls the total number of exploration steps, balancing question quality and computational cost.

3. **Evaluation Metric Design**:

    - Opinion-based value assessment: multiple opinions are extracted from LLM responses, and Schwartz 10-dimensional value labels are identified for each opinion and merged via logical OR.
    - Relative ranking-based aggregation: the TrueSkill system (Bayesian skill assessment) is used to perform multi-dimensional comparative ranking across all LLMs, with win rates serving as the final value scores — more reliable than absolute scoring.

### Loss & Training
No training is required. All optimization is performed in-context via LLM API calls. The core optimization objective is:
$$x^* = \arg\max_x \sum_{i=1}^K \left\{ \alpha_i \text{KL}[p_{\theta_i}(v|x) \| p_M(v|x)] + \frac{\beta}{2} \sum_v |p̂(v|x) - p_{\theta_i}(v|x)| \right\}$$

## Key Experimental Results

### Main Results
The AdAEM Bench is constructed based on the 10 dimensions of Schwartz value theory, comprising 12,310 test questions covering 106 countries.

| Benchmark | # Questions | Avg. Length | Self-BLEU | Similarity |
|-----------|-------------|-------------|-----------|------------|
| SVS | 57 | 13.00 | 52.68 | 0.61 |
| ValueBench | 40 | 15.00 | 26.27 | 0.60 |
| ValueDCG | 4,561 | 11.21 | 13.93 | 0.36 |
| **AdAEM** | **12,310** | **15.11** | **13.42** | **0.44** |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Value priming experiment | Target value +31%, opposing value −58% | $p < 0.01$, validates assessment validity |
| Within-group value change | +17% | Consistent with Schwartz value structure predictions |
| Reliability analysis | Cronbach's $\alpha = 0.8991$ | "Good" reliability |
| Human evaluation improvement | Reasonableness +6.7%, value discriminability +31.6% | Cohen's $\kappa = 0.93$ |

### Key Findings
- Benchmarking 16 LLMs reveals four notable findings: (1) more advanced LLMs exhibit stronger preferences for safety-related dimensions (e.g., universalism); (2) LLMs within the same series share similar value orientations regardless of model size; (3) reasoning-oriented and chat-oriented LLMs differ significantly in value profiles; (4) larger LLMs amplify preferences along specific dimensions.
- AdAEM surpasses baseline benchmarks in informativeness score after only a few iterations.
- Under different topic categories (technological innovation vs. philosophical belief), all LLMs exhibit markedly different value orientations.
- GLM-4 (developed in China) and GPT-4-Turbo (developed in the US) exhibit significant regional divergence on culturally relevant topics.

## Highlights & Insights
- This is the first work to introduce dynamic evaluation into the LLM value assessment domain; the theoretically driven self-extensible mechanism is particularly elegant.
- The information-theoretic objective function is well-designed, simultaneously accounting for discriminability and decoupling.
- The exploitation–exploration strategy based on Multi-Armed Bandits is natural and efficient.
- The adoption of the TrueSkill scoring system is more reliable than conventional absolute scoring.
- Acceptance as ICLR 2026 Oral reflects strong recognition from reviewers.
- Cross-cultural analysis reveals cultural biases embedded in LLM training data and alignment strategies.

## Limitations & Future Work
- The framework relies solely on Schwartz value theory and does not cover Moral Foundations Theory (MFT), Kohlberg's stages of moral development, or other frameworks.
- The work primarily focuses on English-language contexts without sufficiently exploring multilingual and multicultural settings.
- Only a limited set of representative LLMs is included due to budget constraints.
- Automatically generated contentious content may be subject to malicious exploitation.
- The value classifier (GPT-4o) may itself introduce biases.

## Related Work & Insights
- AdAEM complements static benchmarks such as ValueBench and ValueDCG by introducing a dynamic evaluation paradigm.
- Inspired by dynamic evaluation works such as DyVal, this paper is the first to apply the paradigm to value assessment.
- Related to black-box optimization works such as PromptAgent, but with an objective function oriented toward value discriminability.
- Insight: the proposed approach is transferable to other evaluation scenarios requiring dynamic benchmarks, such as safety evaluation and cultural adaptability testing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [\[ACL 2026\] Distorted or Fabricated? A Survey on Hallucination in Video LLMs](../../ACL2026/video_understanding/distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](../../ACL2026/video_understanding/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](../../CVPR2026/video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](../../CVPR2026/video_understanding/how_should_video_llms_output_time.md)

</div>

<!-- RELATED:END -->
