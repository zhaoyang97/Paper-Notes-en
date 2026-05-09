---
title: >-
  [Paper Note] Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
description: >-
  [NeurIPS 2025 (D&B Oral)][AI Safety][mode collapse] This work constructs the Infinity-Chat dataset (26K open-ended real user queries + 31,250 human annotations) to reveal the "Artificial Hivemind" phenomenon in open-ended language model generation—characterized by severe intra-model repetition and inter-model homogeneity—and demonstrates that Reward Models and LM Judges fail to calibrate on samples with high inter-annotator preference disagreement.
tags:
  - "NeurIPS 2025 (D&B Oral)"
  - AI Safety
  - mode collapse
  - LM diversity
  - open-ended generation
  - Infinity-Chat
  - artificial hivemind
  - human preferences
date: 2026-05-08
content_hash: 31899073f9eb07e9
---

# Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

**Conference**: NeurIPS 2025 (D&B Oral)  
**arXiv**: [2510.22954](https://arxiv.org/abs/2510.22954)  
**Code**: To be confirmed  
**Area**: AI Safety / LLM Diversity  
**Keywords**: mode collapse, LM diversity, open-ended generation, Infinity-Chat, artificial hivemind, human preferences

## TL;DR
This work constructs the Infinity-Chat dataset (26K open-ended real user queries + 31,250 human annotations) to reveal the "Artificial Hivemind" phenomenon in open-ended language model generation—characterized by severe intra-model repetition and inter-model homogeneity—and demonstrates that Reward Models and LM Judges fail to calibrate on samples with high inter-annotator preference disagreement.

## Background & Motivation

**Background**: LLMs demonstrate strong capabilities in creative content generation (e.g., story writing, brainstorming, advice-giving), yet outputs obtained through repeated sampling tend to be highly similar. Such homogenization may gradually "homogenize" human thought through prolonged exposure.

**Limitations of Prior Work**: Methods for evaluating LM output diversity are extremely limited—existing work focuses only on narrow tasks (e.g., random number generation, name generation) or repeated sampling from a single model, lacking systematic evaluation of multi-model output diversity under real-world open-ended queries.

**Key Challenge**: LLM training—especially the RLHF/DPO alignment phase—encourages models to converge toward "canonical answers." However, open-ended questions have no single correct answer, and excessive convergence implies a loss of diversity in the answer space.

**Goal**: (1) Construct the first large-scale open-ended query dataset for diversity evaluation; (2) systematically quantify intra-model and inter-model homogenization in LMs; (3) study individual preference variation among humans on open-ended content and its impact on automatic evaluation.

**Key Insight**: Real user–LM interaction logs are collected to source open-ended queries, a taxonomic framework for open-ended questions is established, and large-scale cross-annotation is used to uncover the idiosyncratic nature of human preferences.

**Core Idea**: LMs are not only repetitive within a single model (intra-model), but are also highly similar across different models (inter-model), forming an "Artificial Hivemind"—a phenomenon that existing automatic evaluators fail to detect.

## Method

### Overall Architecture
The work comprises three components: (1) construction of the Infinity-Chat dataset—26K diverse open-ended queries and the first open-ended question taxonomy (6 major categories, 17 subcategories); (2) a large-scale LM diversity study comparing intra-model and inter-model homogenization across multiple LMs in open-ended generation; (3) a human preference study—31,250 annotations (25 independent annotators per sample) revealing the divergence between collective and individual preferences.

### Key Designs

1. **Infinity-Chat Dataset Construction**:

    - **Function**: Collect 26K real-world open-ended user queries.
    - **Mechanism**: Open-ended queries with no unique correct answer are filtered from actual user–LM interaction logs, and the first comprehensive taxonomy is established: 6 major categories (brainstorm & ideation, creative writing, recommendation, opinion & advice, planning, open-ended QA) and 17 subcategories.
    - **Design Motivation**: Existing datasets are either small-scale or limited to closed-ended tasks, failing to capture LM diversity issues in realistic settings.

2. **Mode Collapse Quantification**:

    - **Function**: Systematically measure the degree of homogenization in LM outputs.
    - **Mechanism**: The same query is sampled multiple times from a single model (intra-model) and once each from different models (inter-model); semantic similarity metrics are used to quantify homogenization.
    - **Design Motivation**: This is the first work to distinguish and quantify intra-model repetition and inter-model homogeneity at scale on real open-ended tasks.

3. **Analysis of Individual Specificity in Human Preferences**:

    - **Function**: Examine preference variation across annotators for the same open-ended response.
    - **Mechanism**: 25 independent annotations per sample (absolute ratings + pairwise preferences) are collected, and sample characteristics with high vs. low annotator agreement are analyzed.
    - **Design Motivation**: To expose the systematic blind spots of automatic evaluators (RM, LM Judge)—their calibration degrades significantly on samples with high inter-annotator preference disagreement.

### Loss & Training
This paper involves no model training and is an analytical study. The core contributions are the dataset and empirical findings.

In the experimental setup, the same query is sampled multiple times from a single model (intra-model) and once each from different models (inter-model), with semantic similarity metrics used to quantify homogenization. Each sample receives absolute ratings and pairwise preferences from 25 independent annotators, enabling analysis of collective vs. individual preference divergence.

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
| Intra-model repetition | A single model produces highly similar responses across multiple samples of the same query |
| Inter-model homogeneity | Different models (e.g., GPT-4, Claude, LLaMA) generate surprisingly similar responses to the same open-ended query |
| Relative severity | Inter-model homogeneity is more severe than intra-model repetition |

### Automatic Evaluator Calibration

| Evaluator | High-agreement samples | Low-agreement samples | Note |
|-----------|----------------------|----------------------|------|
| Reward Model | Well calibrated | Significantly miscalibrated | RM fails to discriminate when human preferences diverge |
| LM Judge | Well calibrated | Significantly miscalibrated | Same as above |

### Key Findings
- Inter-model homogeneity is of greater concern than intra-model repetition: the convergence of "ideas" across different LLMs may be a consequence of RLHF and overlapping training data.
- On "easy" samples with high annotator agreement, both RM and LM Judge align well with human judgments; on "subjective" samples with high individual preference variation, automatic evaluators are systematically miscalibrated.
- This miscalibration implies that RM signals used in RLHF may exacerbate homogenization on highly subjective open-ended tasks.

## Highlights & Insights
- **The "Artificial Hivemind" concept**: Naming inter-model homogeneity the Artificial Hivemind vividly captures the phenomenon in which independently trained LMs produce strikingly similar outputs—a concept with significant communicative and conceptual impact.
- **First open-ended question taxonomy**: The 6-category, 17-subcategory taxonomy fills a critical gap and provides a standardized classification framework for future open-ended generation evaluation.
- **Exposing systematic blind spots in automatic evaluation**: The failure of RM and LM Judge on samples with high subjective preference disagreement directly challenges current alignment training paradigms.

## Limitations & Future Work
- **English only**: The dataset and analyses are limited to English LMs; homogenization in other languages and multilingual settings may differ substantially.
- **No remediation proposed**: The work identifies the problem but does not propose concrete methods to mitigate the Artificial Hivemind effect; subsequent work should develop actionable diversity-enhancement strategies.
- **High annotation cost**: The 25-annotator-per-sample design is rigorous but difficult to replicate at scale across other datasets or domains.
- **Sources of homogenization not disentangled**: Inter-model homogeneity may stem from overlapping training data, converging model architectures, or similar RLHF alignment objectives, but no causal analysis is conducted.
- **Focus limited to open-ended generation**: Diversity issues in closed-ended tasks (e.g., coding, mathematical reasoning) are not addressed and may exhibit different homogenization patterns.
- **Future directions**: (1) Develop decoding/training methods that promote output diversity (e.g., personalized alignment, diversity regularization); (2) investigate how training data overlap contributes to inter-model homogeneity; (3) design RMs sensitive to individual preferences; (4) establish standardized metrics for quantifying inter-model diversity.

## Related Work & Insights
- **vs. traditional diversity evaluation**: Prior work evaluates diversity via n-gram metrics or self-BLEU; this paper introduces semantic-level diversity evaluation with inter-model comparison, covering broader dimensions.
- **vs. RLHF alignment research**: Alignment research asks "does the model conform to human preferences?"; this paper raises the question "whose preferences?"—a unified RM may erase individual preference variation.
- **Insights**: This work offers a new perspective on the alignment tax—alignment may not only reduce model capability but may also systematically diminish output diversity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic revelation of inter-model homogeneity; concept is novel and impactful
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale dataset + multi-model comparison + human annotation; limited by the absence of ablation details
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; the Hivemind metaphor is apt
- Value: ⭐⭐⭐⭐⭐ NeurIPS Oral; dataset and findings carry far-reaching implications for AI safety and alignment research

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)
- [\[NeurIPS 2025\] Locally Optimal Private Sampling: Beyond the Global Minimax](locally_optimal_private_sampling_beyond_the_global_minimax.md)
- [\[NeurIPS 2025\] Beyond Last-Click: An Optimal Mechanism for Ad Attribution](beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [\[NeurIPS 2025\] Causally Reliable Concept Bottleneck Models](causally_reliable_concept_bottleneck_models.md)
- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)

<!-- RELATED:END -->
