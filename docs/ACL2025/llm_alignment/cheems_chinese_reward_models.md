---
title: >-
  [Paper Note] Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch
description: >-
  [ACL 2025][LLM Alignment][reward model] To fill the gap in Chinese Reward Model resources, this paper constructs CheemsBench (the first large-scale Chinese RM evaluation benchmark) and CheemsPreference (the first large-scale Chinese preference dataset). Trained via human-machine collaborative annotation and a distant supervision filtering strategy, CheemsRM significantly outperforms all existing open-source RMs in Chinese scenarios.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "reward model"
  - "Chinese preference"
  - "benchmark"
  - "distant supervision"
  - "RLHF"
date: 2026-05-08
content_hash: 887026a1bf3f2c76
---

# Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch

**Conference**: ACL 2025  
**arXiv**: [2502.17173](https://arxiv.org/abs/2502.17173)  
**Code**: [https://github.com/AlignRM/CheemsRM](https://github.com/AlignRM/CheemsRM)  
**Area**: RLHF Alignment  
**Keywords**: reward model, Chinese preference, benchmark, distant supervision, RLHF

## TL;DR
To fill the gap in Chinese Reward Model resources, this paper constructs CheemsBench (the first large-scale Chinese RM evaluation benchmark) and CheemsPreference (the first large-scale Chinese preference dataset). Trained via human-machine collaborative annotation and a distant supervision filtering strategy, CheemsRM significantly outperforms all existing open-source RMs in Chinese scenarios.

## Background & Motivation
**Background**: The Reward Model is a core component of RLHF, but current RM research is highly concentrated on English scenarios (e.g., RewardBench, UltraRM, Skywork-Reward), and the development of Chinese RMs lags significantly behind.

**Limitations of Prior Work**:
   - Existing Chinese preference datasets are small in scale (Huozi has only a few thousand samples) and restricted in domain (specific scenarios like Zhihu Q&A).
   - Existing RM benchmarks are all in English (e.g., RewardBench) and cannot evaluate the performance of RMs in Chinese scenarios.
   - Heavy reliance on GPT-synthesized annotation data makes it difficult to accurately reflect the true preferences of Chinese users.

**Key Challenge**: The lack of high-quality Chinese preference data and evaluation benchmarks makes it impossible for Chinese RMs to effectively learn and capture the preferences of Chinese users.

**Goal**: To build a Chinese RM resource system from scratch, including an evaluation benchmark, a preference dataset, and training methodologies.

**Key Insight**: To center on human annotation while leveraging distant supervision strategies to scale up.

**Core Idea**: First build high-quality small datasets and a benchmark using purely human annotations, and then utilize the RM trained on this human data to filter large-scale GPT-annotated data, achieving a balance between quality and scale.

## Method

### Overall Architecture
It consists of three parts: (1) CheemsBench evaluation benchmark: 2,492 prompts × 5 responses, employing five rounds of human triple-wise comparisons and a conflict resolution algorithm to generate reliable partial order rankings; (2) CheemsPreference dataset: 27K human instructions + multi-model sampling + human-machine collaborative annotation (small human dataset + large GPT dataset + RM filtering); (3) CheemsRM: trained on CheemsPreference based on Qwen2.5-72B-Instruct.

### Key Designs

1. **CheemsBench Construction — Multi-Response Triple-Wise Comparison + Conflict Resolution**:

    - **Function**: Samples 5 responses for each prompt and conducts 5 rounds of triple-wise comparisons to generate reliable partial order rankings.
    - **Mechanism**: Transforms annotation results into a directed preference graph, using DFS to detect cycles (conflicts) $\rightarrow$ merging nodes in cycles into a single super-node $\rightarrow$ repeating until the graph is acyclic $\rightarrow$ performing topological sorting to obtain the partial order. Two evaluation metrics, Accuracy and Exact Match, are used.
    - **Design Motivation**: Traditional pairwise comparison has limitations in reflecting downstream task performance (Wen et al., 2024). Multi-response evaluation aligns better with practical usage scenarios (such as best-of-N sampling). Triple-wise comparison provides higher information density than pairwise comparison while avoiding the high annotation cost of full ranking.
    - **Data Source**: 1,146 open-source prompts + 1,346 real human instructions, covering categories such as reasoning, comprehension, generation, and complex instructions.

2. **CheemsPreference Construction — Distant Supervision Strategy**:

    - **Function**: Uses an RM trained on a small, human-annotated dataset to filter a large, GPT-annotated dataset.
    - **Mechanism**: (a) Human annotators label 3,260 prompts (37K comparisons); (b) GPT-4o annotates 27,861 prompts (332K comparisons), performing pairwise comparisons on $C_N^2$ pairs; (c) the RM trained on human data filters out conflicts and errors in GPT annotations, retaining consistent preference chains.
    - **Design Motivation**: Pure human annotation is prohibitively expensive (3K is the limit), while pure GPT annotation has unreliable quality (due to position bias and inconsistency). Distant supervision achieves a balance between the two.
    - **Length Debiasing**: Phrases/pairs are divided into two groups based on whether chosen is longer/shorter than rejected, and the larger group is downsampled to balance length bias.

3. **CheemsRM Training — Multi-Response Bradley-Terry Loss**:

    - **Function**: Trains a discriminative RM on multi-response partial order data.
    - **Mechanism**: The Bradley-Terry loss is formulated as $\mathcal{L}' = -\mathbb{E}[\log\sigma(r(x, y_w) - r(x, y_l))]$, with an added Gaussian regularization term $\mathcal{L} = \mathcal{L}' + \mathbb{E}[r^2(x, y)]$ to stabilize training. A greedy sample-based batch strategy is used to group all responses of the same prompt into a single batch as much as possible.
    - **Design Motivation**: Compared to standard pairwise training, multi-response provides richer comparison signals; Gaussian regularization prevents the reward scores from exploding.

## Key Experimental Results

### Main Results

**Performance of various RMs on CheemsBench**:

| Model | RewardBench | Open Prompt Acc. | Human Instr. Acc. | Overall |
|------|:-----------:|:----------------:|:-----------------:|:-------:|
| Skywork-Reward-Gemma-27B | **0.938** | 0.754 | 0.748 | 0.535 |
| Nemotron-70B-Reward | 0.941 | 0.750 | 0.722 | 0.515 |
| Skywork-Critic-70B (gen) | 0.933 | 0.755 | 0.731 | 0.516 |
| GPT-4o (gen) | 0.846 | 0.640 | 0.727 | 0.457 |
| **CheemsRM (Ours)** | 0.919 | **0.857** | **0.832** | **0.657** |

CheemsRM leads significantly with an Overall score of **0.657**, outperforming the runner-up's 0.535 by a large margin (+12.2%). Its Exact Match scores reach 0.508 and 0.431, respectively, far exceeding other models (best <0.33).

### Ablation Study

**Ablation on Preference Data Sources**:

| Data Source | Open Acc. | Human Acc. | Overall |
|---------|:---------:|:----------:|:-------:|
| GPT-only Annotation | 0.815 | 0.789 | 0.590 |
| Human-only Annotation | 0.829 | 0.811 | 0.614 |
| GPT + Human | 0.839 | 0.820 | 0.633 |
| GPT + Human + Distant Supervision Filtering | **0.857** | **0.832** | **0.657** |

**Comparison of Preference Datasets (Backbone: Qwen2.5-72B)**:

| Dataset | Open Acc. | Human Acc. |
|-------|:---------:|:----------:|
| Huozi (Best Existing Chinese) | 0.728 | 0.682 |
| HH-RLHF (English) | 0.753 | 0.740 |
| Ultrafeedback (Best English) | 0.769 | 0.749 |
| **CheemsPreference** | **0.857** | **0.832** |

### Key Findings
- The strongest existing English RM (Skywork-Reward-Gemma-27B, with RewardBench of 0.938) degrades significantly in Chinese scenarios, with an Overall score of only 0.535.
- Although there are only 3K human-labeled samples, they yield better training results than 28K GPT-labeled samples, demonstrating that data quality is far more crucial than data scale.
- Distant supervision filtering further improves the Overall score by 2.4% on top of GPT + Human, validating the effectiveness of the filtering strategy.
- RMs perform worst on "comprehension" tasks and best on "reasoning" tasks, suggesting that current RMs are better at judging objective correctness rather than subjective quality.

## Highlights & Insights
- **Exquisite Design of the Distant Supervision Strategy**: Utilizing an RM trained on a small amount of human-annotated data to filter massive GPT-annotated data. This leverages a small-scale asset to achieve large-scale filtration, which is highly practical and generalizable to building preference datasets for other languages and domains.
- **Conflict Resolution Algorithm**: Formalizing annotation disagreement as a cycle detection problem within a graph and resolving it using DFS + node merging + topological sorting. This approach is elegant, scalable, and reusable in any multi-annotator scenario.
- **First to Systematically Reveal the Gap Between Chinese and English RMs**: Even top-tier models on RewardBench perform poorly in Chinese scenarios, presenting highly significant findings.

## Limitations & Future Work
- **Backbone Dependency on Qwen2.5-72B**: CheemsRM has high computational costs; future work could explore its performance on smaller models (e.g., 7B-14B).
- **Preference Classification Taxonomies Rely heavily on Manual Design**: The classification of 8 major categories and dozens of subcategories might overlook certain unique Chinese scenarios (e.g., classical Chinese comprehension, dialect processing).
- **Evaluation Limited to Discriminative Use Cases**: The downstream performance of using CheemsPreference for DPO/PPO training remains unverified.
- **High Cost of GPT-4o Annotations**: Calling GPT-4o for 28K prompts × $C_5^2$ pairs incurs non-negligible costs; cheaper alternatives could be explored.

## Related Work & Insights
- **vs RewardBench**: RewardBench is the standard English RM benchmark, and CheemsBench fills the gap for Chinese. However, CheemsBench additionally introduces multi-response evaluation and Exact Match metrics.
- **vs Skywork-Reward**: Skywork-Reward achieves SOTA on RewardBench (0.938) but only 0.535 on CheemsBench, indicating that English RM capabilities cannot easily transfer to Chinese.
- **vs UltraFeedback**: UltraFeedback is one of the strongest general English preference datasets, whereas CheemsPreference vastly outperforms its training efficacy in Chinese scenarios.
- The distant supervision + conflict resolution strategy can be directly applied to construct RM resources for other languages, such as Japanese and Korean.

## Rating
- Novelty: ⭐⭐⭐ Primarily resource contribution; technical novelty is moderate (with the distant supervision strategy being innovative).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation (comparing 16+ RMs, ablations across multiple datasets, and downstream task correlation analysis).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, solid data, and abundant figures/tables.
- Value: ⭐⭐⭐⭐ Fills the gap in Chinese RM resources, offering significant reference value to the Chinese LLM alignment community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment](../../ICLR2026/llm_alignment/evaluating_and_improving_cultural_awareness_of_reward_models_for_llm_alignment.md)
- [\[ICLR 2026\] StoryAlign: Evaluating and Training Reward Models for Story Generation](../../ICLR2026/llm_alignment/storyalign_evaluating_and_training_reward_models_for_story_generation.md)
- [\[ICLR 2026\] Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance](../../ICLR2026/llm_alignment/eliminating_inductive_bias_in_reward_models_with_information-theoretic_guidance.md)
- [\[ACL 2025\] TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)
- [\[ACL 2025\] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](prmbench_a_fine-grained_and_challenging_benchmark_for_process-level_reward_model.md)

</div>

<!-- RELATED:END -->
