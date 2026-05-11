---
title: >-
  [Paper Note] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy
description: >-
  [ICLR 2026][LLM Alignment][Reward model] This paper proposes a two-stage preference data curation pipeline based on Human-AI synergy. Stage 1 accumulates approximately 1M preference pairs over 8 iterative rounds via huma…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Reward model"
  - "preference data curation"
  - "Human-AI synergy"
  - "data quality"
  - "scalable curation"
date: 2026-05-08
content_hash: cb4398a50f4999e4
---

# Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy

**Conference**: ICLR 2026
**arXiv**: [2507.01352](https://arxiv.org/abs/2507.01352)
**Code**: SynPref-40M dataset publicly available
**Area**: Alignment RLHF / Reward Modeling
**Keywords**: Reward model, preference data curation, Human-AI synergy, data quality, scalable curation

## TL;DR

This paper proposes a two-stage preference data curation pipeline based on Human-AI synergy. Stage 1 accumulates approximately 1M preference pairs over 8 iterative rounds via human verification, error-driven adaptive retrieval, and preference-guided LLM annotation. Stage 2 scales the dataset to 26M pairs using dual-RM consistency filtering. The resulting Skywork-Reward-V2 8B model achieves 97.8% on RewardBench and an average of 88.6% across 7 mainstream benchmarks, comprehensively surpassing all open-source 70B reward models.

## Background & Motivation

Reward models (RMs) are a core component of the RLHF pipeline, responsible for converting human preference signals into optimizable scalar rewards. However, as of September 2024, the development of open-source RMs had effectively stagnated: 16 of the top 20 models on the RewardBench leaderboard directly or indirectly used the same base model or highly similar training data. A more critical issue is that improvements in RewardBench scores from approximately 80 to 90+ do not consistently translate to gains on other benchmarks or downstream tasks. The authors conducted a cross-benchmark correlation analysis of 31 top open-source RMs across 7 benchmarks, finding weak Pearson correlations between RewardBench and other benchmarks, with some dimensions even exhibiting negative correlations.

**The fundamental bottleneck lies not in model architecture or loss functions, but in the preference data itself.** Existing preference datasets suffer from three systemic flaws: (1) narrow coverage, concentrated on a limited number of task types; (2) insufficient quality of synthetic annotations, as biases introduced by pure LLM annotation cannot be self-corrected; and (3) lack of rigorous quality control, since human annotation is high-quality but not scalable. The paper also conducts comparative experiments on multiple loss function variants for the Gemma-2-27B family (including improved ranking losses, contrastive losses, etc.), finding that the original version still achieves the best overall performance, indicating that improving training algorithms alone cannot compensate for deficiencies in data quality.

Core Idea: use human verification to guide LLM annotation (rather than replace it), then achieve simultaneous expansion of quality and scale through error-driven retrieval and consistency filtering.

## Method

### Overall Architecture

The paper constructs SynPref-40M (40 million preference pairs, of which 26 million pass curation) using a two-stage pipeline:

- **Stage 1 (Small-scale human-driven iterative curation)**: 8 iterative rounds, each consisting of three steps—RM training and evaluation → error-driven retrieval → preference-aware LLM annotation—accumulating approximately 1M preference pairs.
- **Stage 2 (Large-scale automatic consistency curation)**: The best RM from Stage 1 and an independently trained gold RM are used to apply dual consistency filtering to in-the-wild data, scaling to approximately 26M pairs without additional human effort.

### Key Designs

**1. Rigorous Human Verification Protocol**

Annotators do not simply inspect conversation histories and two responses. Each preference pair is accompanied by a 5-tuple of attributes: task category, preference objectivity, controversiality, desired attributes, and instance-level annotation guidelines. Annotators are permitted to use search engines, frontier LLM assistants, and domain-specific LLMs (e.g., for math or code) as auxiliary tools, but are prohibited from relying entirely on LLM outputs; the final judgment must be made by a human. Factual verification tasks require the use of a search engine; code correctness tasks require executing the code and verifying its output. This "tool-augmented human annotation" yields significantly higher annotation quality than bare human annotation (+3.2 vs. +0.4).

**2. Error-Driven Adaptive Retrieval**

In each iterative round, the current RM is first evaluated on a gold validation set to identify incorrectly predicted samples. These error samples' $(x, a)$ (conversation + attribute) embeddings are then used as queries to retrieve semantically similar new samples from an unverified pool. The retrieval count is dynamically adjusted based on RM confidence:

$$k = \begin{cases} k_{\max}, & \text{if } p \le 0.5 \text{ (incorrect prediction)} \\ \lceil k_{\max} \cdot (1 - p) \rceil, & \text{if } p > 0.5 \text{ (correct prediction)} \end{cases}$$

where $k_{\max} = 8$. Intuitively, regions where the RM performs poorly are allocated more new samples for subsequent annotation, analogous to uncertainty sampling in active learning.

**3. Preference-Aware LLM Annotation**

When annotating newly retrieved samples with LLMs, rather than directly asking the LLM to judge, the method first retrieves semantically similar human-annotated samples from the gold set as few-shot examples inserted into the prompt, anchoring LLM judgments to human-verified preferences. Multiple strong LLMs are then used for annotation, with intra-model self-consistency aggregation performed first, followed by cross-model merging to mitigate single-model bias. Response order in the prompt is randomized to eliminate position bias.

**4. Stage 2 Dual-RM Consistency Filtering and Recovery Mechanism**

For in-the-wild data, samples where the current best RM has confidence >0.5 are directly retained. Inconsistent samples undergo LLM re-annotation (reusing the retrieval + few-shot scheme from Stage 1, but without human involvement). An additional gold RM trained solely on human-verified data performs a secondary check: only samples that pass both the gold RM and the best RM / LLM consistency check are retained. Samples rejected by both RMs are not discarded outright—their chosen/rejected labels are flipped and the samples are "recycled," obtaining additional training data at zero additional annotation cost. Experiments confirm consistent performance gains from this mechanism across all stages and iterations.

### Training Details

- Loss function: standard Bradley-Terry pairwise objective, $p = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$
- 8 model scales: Qwen3 0.6B/1.7B/4B/8B + Llama-3.2 1B/3B + Llama-3.1 8B (standard version + 40M version)
- Maximum context length 16K tokens, large batch size 10240, constant learning rate, 1 epoch
- Large-batch training saves approximately 35% of total training compute

## Key Experimental Results

### Main Results: Comprehensive Evaluation across 7 Benchmarks

| Model | Params | RB | RB-v2 | PPE-Pref | PPE-Corr | RMB | RM-Bench | JudgeBench | Avg |
|-------|--------|------|-------|----------|----------|------|----------|------------|-----|
| OffsetBias-8B | 8B | 89.0 | 64.8 | 59.2 | 64.1 | 57.8 | 71.3 | 63.5 | 67.1 |
| ArmoRM-8B | 8B | 90.4 | 66.5 | 60.6 | 60.6 | 64.6 | 69.2 | 59.7 | 67.4 |
| Skywork-V1-27B | 27B | 94.3 | 75.3 | 63.6 | 61.9 | 69.4 | 67.6 | 66.5 | 71.2 |
| Nemotron-70B | 70B | 93.9 | 76.7 | 64.2 | 63.2 | 64.9 | 72.2 | 65.8 | 71.6 |
| INF-ORM-70B | 70B | 95.1 | 76.5 | 64.2 | 64.4 | 70.5 | 75.4 | 70.2 | 73.8 |
| **Skywork-V2-Qwen3-1.7B** | **1.7B** | 90.3 | 68.3 | 67.6 | 70.5 | 78.1 | 78.7 | 72.9 | **75.2** |
| **Skywork-V2-Llama-8B** | **8B** | 96.4 | 84.1 | 77.3 | 83.4 | 86.4 | 92.8 | 80.0 | **85.8** |
| **Skywork-V2-Llama-8B-40M** | **8B** | **97.8** | **86.5** | **79.8** | **87.2** | **89.3** | **96.0** | **83.4** | **88.6** |

Several key comparisons: (1) The 1.7B Skywork-V2 surpasses the previously strongest 70B model INF-ORM on all benchmarks except RewardBench and RB-v2; (2) the 8B version ranks first across all 7 benchmarks; (3) the 40M version gains an additional average improvement of +2.8 points through recycled flipped data.

### Ablation Study: Comparison of Data Curation Methods

| Curation Method | Gain Relative to Seed RM |
|-----------------|--------------------------|
| Direct addition of uncurated data (no curation) | ≈0 (12M data fails to surpass seed model) |
| Pure LLM curation (self-consistency aggregation) | +0.1 pt (likely within optimization noise) |
| Human curation (bare annotation) | +0.4 pt |
| Human curation + preference attributes | +1.1 pt |
| Human curation + LLM curation | +2.3 pt |
| Full protocol (tool-augmented human + adaptive retrieval + LLM) | **+3.2 pt** |
| Only 290K curated data (1.8% of full set) | Already surpasses prev. SOTA 70B model |

### Other Key Experimental Results

- **RM-Bench style bias resistance**: Most baseline models show large performance gaps across Easy/Normal/Hard style conditions (e.g., INF-ORM-70B: Normal 80.0 vs. Hard 54.0, a gap of 26 points). Skywork-V2-8B-40M achieves 93.5 under Hard conditions (gap of only 4.1 points), indicating that SynPref-40M training yields more debiased preference representations.
- **Best-of-N scaling**: In RMB Best-of-N evaluation, all 8 Skywork-V2 variants outperform GPT-4o (maximum gap +20 points), and exhibit positive scaling curves across 5 tasks in PPE Correctness.
- **RewardBench v2 precise instruction following**: All existing RMs score below 50 on this dimension, while Skywork-V2-8B-40M reaches 67.8, surpassing Claude-3.7-Sonnet (54.4) and Gemini-2.5-Flash (55.3).
- **JudgeBench mathematical reasoning**: Skywork-V2-Llama-3B achieves 87.5 on the math subtask, matching o3-mini (high); the 8B-40M version reaches 89.3, exceeding it.

## Highlights & Insights

- **Data quality overwhelmingly outweighs quantity**: An RM trained on 12M uncurated data fails to match the seed model, while only 290K (1.8%) curated data already surpasses the previous 70B SOTA. This directly challenges the naive assumption that more preference data is always better.
- **Pure LLM curation is nearly ineffective**: It yields only a +0.1 point gain. This explains why open-source preference datasets relying heavily on LLM-synthesized annotations have failed to advance RM progress—without human-calibrated anchors, biases in LLM annotations self-reinforce.
- **Error-driven retrieval is a critical bridge**: It maximizes the value of limited human annotations—rather than randomly labeling more data, it precisely identifies the RM's blind spots and supplements them in a targeted manner.
- **Elegance of the recovery mechanism**: Preference pairs rejected by both RMs suggest the original labels may be incorrect. Flipping chosen/rejected and reusing them as "correction data" provides additional training data at zero cost, with experiments confirming consistent performance improvements across all stages and iterations.
- **Tool-augmented human annotation >> bare human annotation**: Allowing annotators to use search engines and LLM tools (while retaining the final judgment for humans) elevates annotation quality from +0.4 to +3.2, providing an important reference for future data annotation protocol design.

## Limitations & Future Work

- Subjective preferences (e.g., writing style) do not exhibit data scaling behavior; curation is primarily effective for objective preferences.
- Stage 1 still depends on human annotation resources, requiring the labor investment of 8 iterative rounds.
- Only the pairwise Bradley-Terry objective is employed; pointwise scoring and listwise ranking methods are not explored.
- Base models at 70B+ scale are not attempted due to training cost and deployment considerations, and the marginal benefit of data quality advantages on larger models remains unknown.

## Related Work & Insights

- **vs. ArmoRM / Nemotron / INF-ORM (70B scale)**: These models may perform strongly on individual benchmarks, but all fall short of Skywork-V2 8B when evaluated comprehensively across 7 benchmarks, demonstrating that data quality can compensate for a 9× model scale gap.
- **vs. generative reward models (DeepSeek-GRM, RM-R1)**: Such approaches enhance judgment capability through reasoning chains or meta-evaluation, yet Skywork-V2 surpasses them comprehensively using only the Bradley-Terry objective, indicating that data-level and model-level improvements are orthogonal.
- **vs. active learning**: Error-driven retrieval is essentially an active learning strategy for preference annotation. The key distinction is that rather than directly having humans annotate retrieved samples, it uses human gold data to guide LLM annotation, achieving a balance between quality and efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The two-stage Human-AI synergy pipeline is systematic and elegant, with error-driven retrieval, preference-aware annotation, and the recovery mechanism forming a tightly integrated framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 benchmarks × 8 model scales × thorough ablations across both data and method dimensions yield a highly complete chain of evidence.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is described clearly; Section 2 establishes motivation thoroughly before presenting the method, with strong logical coherence.
- Value: ⭐⭐⭐⭐⭐ The paper provides a complete solution from data curation to model training for reward model development. SynPref-40M and the full model series are open-sourced, enabling direct reproduction and application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[NeurIPS 2025\] Can DPO Learn Diverse Human Values? A Theoretical Scaling Law](../../NeurIPS2025/llm_alignment/can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](../../NeurIPS2025/llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[ICLR 2026\] Learning Ordinal Probabilistic Reward from Preferences (OPRM)](learning_ordinal_probabilistic_reward_from_preferences.md)
- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)

</div>

<!-- RELATED:END -->
