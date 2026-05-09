---
title: >-
  [Paper Note] Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
description: >-
  [NeurIPS 2025 (D&B Oral)][Signal & Communication][mode collapse] This work introduces the Infinity-Chat dataset (26K open-ended real-world user queries with 31,250 human annotations) to expose the "Artificial Hivemind" phenomenon in language models — severe intra-model repetition and inter-model homogeneity in open-ended generation — and demonstrates that Reward Models and LM Judges fail to calibrate on samples with high inter-annotator preference divergence.
tags:
  - "NeurIPS 2025 (D&B Oral)"
  - "Signal & Communication"
  - mode collapse
  - LM diversity
  - open-ended generation
  - Infinity-Chat
  - artificial hivemind
  - human preferences
date: 2026-05-08
content_hash: 31f7e46e8c5ae341
---

# Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

**Conference**: NeurIPS 2025 (D&B Oral)  
**arXiv**: [2510.22954](https://arxiv.org/abs/2510.22954)  
**Code**: To be confirmed  
**Area**: AI Safety / LLM Diversity  
**Keywords**: mode collapse, LM diversity, open-ended generation, Infinity-Chat, artificial hivemind, human preferences

## TL;DR
This work introduces the Infinity-Chat dataset (26K open-ended real-world user queries with 31,250 human annotations) to expose the "Artificial Hivemind" phenomenon in language models — severe intra-model repetition and inter-model homogeneity in open-ended generation — and demonstrates that Reward Models and LM Judges fail to calibrate on samples with high inter-annotator preference divergence.

## Background & Motivation

**Background**: LLMs exhibit strong capabilities in creative content generation (e.g., story writing, brainstorming, advice-giving), yet repeated sampling from the same model tends to yield highly similar outputs. Such homogeneity, through prolonged exposure, may gradually homogenize human thought.

**Limitations of Prior Work**: Methods for evaluating the diversity of LM outputs are extremely limited — existing work focuses only on narrow tasks (e.g., random number generation, name generation) or intra-model repeated sampling, and lacks systematic evaluation of multi-model output diversity under real-world open-ended queries.

**Key Challenge**: LLM training — particularly the RLHF/DPO alignment stage — incentivizes models to converge toward "standard answers." However, open-ended questions have no single correct answer; excessive convergence implies the loss of diversity across the answer space.

**Goal**: (1) Construct the first large-scale open-ended query dataset for diversity evaluation; (2) systematically quantify intra-model and inter-model homogeneity in LMs; (3) investigate individual human preference variation on open-ended content and its impact on automatic evaluation.

**Key Insight**: Real open-ended queries from user–LM interaction logs are collected, an open-ended question taxonomy is established, and large-scale cross-annotation reveals the idiosyncratic nature of human preferences.

**Core Idea**: Language models are not only repetitive within a single model (intra-model), but also strikingly similar across different models (inter-model), forming an "Artificial Hivemind" — a phenomenon that existing automatic evaluators fail to detect.

## Method

### Overall Architecture
The work comprises three components: (1) construction of the Infinity-Chat dataset — 26K diverse open-ended queries with the first open-ended question taxonomy (6 major categories, 17 subcategories); (2) a large-scale LM diversity study comparing intra-model and inter-model homogeneity across multiple LMs in open-ended generation; and (3) a human preference study — 31,250 annotations (25 independent annotators per sample) revealing the divergence between collective and individual preferences.

### Key Designs

1. **Infinity-Chat Dataset Construction**:

    - Function: Collect 26K real-world open-ended user queries.
    - Mechanism: Open-ended queries with no unique correct answer are filtered from actual user–LM interaction logs, and the first comprehensive taxonomy is established: 6 major categories (brainstorm & ideation, creative writing, recommendation, opinion & advice, planning, open-ended QA) and 17 subcategories.
    - Design Motivation: Existing datasets are either small-scale or restricted to closed-ended tasks, failing to reflect LM diversity issues in real-world scenarios.

2. **Mode Collapse Quantification**:

    - Function: Systematically measure the degree of homogeneity in LM outputs.
    - Mechanism: The same query is sampled multiple times from the same model (intra-model) and once each from different models (inter-model); homogeneity is quantified via semantic similarity and related metrics.
    - Design Motivation: This is the first work to distinguish and quantify intra-model repetition and inter-model homogeneity at scale on real open-ended tasks.

3. **Analysis of Individual Specificity in Human Preferences**:

    - Function: Investigate preference variation across different annotators for the same open-ended response.
    - Mechanism: 25 independent annotations per sample (absolute ratings + pairwise preferences) are collected; samples with high vs. low annotator agreement are characterized.
    - Design Motivation: To expose systematic blind spots in automatic evaluation (RM, LM Judge) — their calibration degrades significantly on samples with high annotator preference divergence.

### Loss & Training
This paper involves no model training and is an analytical work. The core contributions are the dataset and empirical findings.

In the experimental setup, the same query is sampled multiple times from a single model (intra-model) and once each from different models (inter-model), with homogeneity quantified via semantic similarity metrics. For each sample, absolute ratings and pairwise preferences from 25 independent annotators are collected to analyze divergence between collective and individual preferences.

## Key Experimental Results

### Infinity-Chat Dataset Statistics

| Metric | Value |
|--------|-------|
| Open-ended queries | 26K |
| Taxonomy | 6 major categories + 17 subcategories |
| Human annotations | 31,250 |
| Annotators per sample | 25 |

### Mode Collapse Findings

| Phenomenon | Description |
|------------|-------------|
| Intra-model repetition | A single model repeatedly generates highly similar responses across multiple samples of the same query. |
| Inter-model homogeneity | Different models (e.g., GPT-4, Claude, LLaMA) generate strikingly similar responses to the same open-ended query. |
| Relative severity | Inter-model homogeneity is more pronounced than intra-model repetition. |

### Automatic Evaluation Calibration

| Evaluator | High-agreement samples | Low-agreement samples | Note |
|-----------|----------------------|-----------------------|------|
| Reward Model | Well-calibrated | Significantly miscalibrated | RM fails to discriminate when human preferences diverge. |
| LM Judge | Well-calibrated | Significantly miscalibrated | Same as above. |

### Key Findings
- Inter-model homogeneity is more concerning than intra-model repetition: the convergence of "ideas" across different LLMs is likely a consequence of overlapping training data and/or RLHF alignment.
- On "easy" samples with high annotator agreement, RM and LM Judge performance aligns with human judgment; on "subjective" samples with high individual preference divergence, automatic evaluators are systematically miscalibrated.
- This miscalibration implies that the RM signal used in RLHF may exacerbate homogeneity on subjective open-ended tasks.

## Highlights & Insights
- **The "Artificial Hivemind" concept**: Naming inter-model homogeneity as the Artificial Hivemind vividly captures the phenomenon in which independently trained LMs produce remarkably similar outputs; the concept carries significant communicative and conceptual impact.
- **First open-ended question taxonomy**: The 6-category, 17-subcategory taxonomy fills a critical gap and provides a standardized classification framework for subsequent open-ended generation evaluation.
- **Exposing systematic blind spots in automatic evaluation**: The failure of RM and LM Judge on samples with high subjective preference divergence directly challenges the current alignment training paradigm.

## Limitations & Future Work
- **English-only scope**: The dataset and analyses are limited to English LMs; homogeneity patterns in other languages and multilingual settings may differ substantially.
- **No proposed solutions**: While the problem is identified, no concrete methods for mitigating the Artificial Hivemind effect are proposed; future work should develop actionable diversity-enhancement strategies.
- **High annotation cost**: The design of 25 independent annotators per sample is methodologically rigorous but difficult to scale to other datasets or domains.
- **Sources of homogeneity unattributed**: Inter-model homogeneity may stem from overlapping training data, architectural convergence, or similar RLHF alignment objectives, but no causal analysis is conducted.
- **Focus limited to open-ended generation**: Diversity issues in closed-ended tasks (e.g., coding, mathematical reasoning) are not addressed and may exhibit distinct homogeneity patterns.
- **Future directions**: (1) Develop decoding/training methods that promote output diversity (e.g., personalized alignment, diversity regularization); (2) investigate how training data overlap drives inter-model homogeneity; (3) design RMs sensitive to individual preferences; (4) establish standardized metrics for quantifying inter-model diversity.

## Related Work & Insights
- **vs. traditional diversity evaluation**: Prior work relies on n-gram diversity or self-BLEU; this paper introduces semantic-level diversity evaluation combined with inter-model comparison, providing a more comprehensive perspective.
- **vs. RLHF alignment research**: Alignment research asks "does the model conform to human preferences?"; this paper raises the further question of "whose preferences?" — a unified RM may systematically erase individual preference variation.
- **Insight**: This work offers a novel angle on the alignment tax — alignment may not only reduce model capability but also systematically diminish output diversity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic exposure of inter-model homogeneity; novel and influential concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale dataset + multi-model comparison + human annotation; limited by the absence of ablation details.
- Writing Quality: ⭐⭐⭐⭐ Problem framing is clear; the Hivemind metaphor is apt.
- Value: ⭐⭐⭐⭐⭐ NeurIPS Oral; dataset and findings carry broad implications for AI safety and alignment research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models](../../ACL2026/signal_comm/policyllm_towards_excellent_comprehension_of_public_policy_for_large_language_mo.md)
- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](../../CVPR2026/signal_comm/clay_conditional_visual_similarity.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)

<!-- RELATED:END -->
