---
title: >-
  [Paper Note] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion
description: >-
  [ACL 2026][Recommender Systems][Cold-Start] Cold-EQS is proposed as a query suggestion framework for cold-start e-commerce scenarios. It utilizes answerability, factual accuracy…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Cold-Start"
  - "Query Suggestion"
  - "Quality-Driven Reward"
  - "Uncertainty Sampling"
  - "E-commerce Dialogue"
date: 2026-05-08
content_hash: b675a0465ee8c4a9
---

# Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion

**Conference**: ACL 2026  
**arXiv**: [2603.22922](https://arxiv.org/abs/2603.22922)  
**Code**: [GitHub](https://github.com/QiSun123/Cold-EQS)  
**Area**: E-Commerce / Reinforcement Learning  
**Keywords**: Cold-Start, Query Suggestion, Quality-Driven Reward, Uncertainty Sampling, E-commerce Dialogue

## TL;DR

Cold-EQS is proposed as a query suggestion framework for cold-start e-commerce scenarios. It utilizes answerability, factual accuracy, and information gain as intrinsic quality rewards and continuously optimizes query suggestion quality through iterative reinforcement learning, achieving a 6.81% improvement in online chatUV.

## Background & Motivation

**Background**: Query Suggestion (QS) is a core component of e-commerce dialogue systems. In multi-turn interactions, AI assistants proactively provide clickable suggested queries to help users refine their needs with minimal effort. Existing generative methods typically use LLMs for query generation and align them using Click-Through Rate (CTR) models.

**Limitations of Prior Work**: (1) Generative methods rely heavily on large-scale online click data to train effective CTR models (e.g., 20M+ interaction records), which is unavailable during the cold-start phase; (2) Existing methods primarily focus on the **relevance and diversity** of queries while neglecting the **intrinsic quality** of the queries themselves. Generated queries may be unanswerable (exceeding downstream agent capabilities), contain hallucinated facts ("Buy an iPhone for one dollar"), or merely repeat the user's original question without providing additional information.

**Key Challenge**: In the cold-start phase, CTR models cannot be trained due to the lack of click data, yet query suggestion quality must still be continuously optimized. Furthermore, click signals themselves can be noisy—high-click queries do not necessarily lead to high-quality multi-turn interactions.

**Goal**: The goal is to continuously optimize the quality of query suggestions through intrinsic quality signals during the cold-start phase without relying on large amounts of click data.

**Core Idea**: Instead of using CTR as a reward, three intrinsic quality dimensions—answerability, factual accuracy, and information gain—are used as RL reward signals. This is combined with uncertainty sampling to select difficult samples from online data without click signals for iterative training.

## Method

### Overall Architecture

Cold-EQS is a four-stage iterative framework: (1) Warm-start SFT using a small amount of click data; (2) GRPO reinforcement learning using quality-driven rewards; (3) Selection of difficult samples from non-click online data through uncertainty sampling; (4) Continuous optimization via multi-round iterative RL training. The base model is Qwen3-4B.

### Key Designs

1. **Quality-Driven Reward Mechanism**:

    - **Function**: Replaces CTR signals and provides intrinsic quality assessment independent of click data.
    - **Mechanism**: For each rollout, $k$ query suggestions are generated. Each suggestion $s_i$ is evaluated across three dimensions: answerability $r_{\mathrm{ans}}(s_i)$ (whether the downstream agent can answer), factual accuracy $r_{\mathrm{fact}}(s_i)$ (whether it contains hallucinations), and information gain $r_{\mathrm{info}}(s_i)$ (whether it brings new information). The total reward is defined as $r_q = r_f \cdot \frac{1}{k}\sum_{i=1}^k r_{\mathrm{ans}}(s_i) \cdot r_{\mathrm{fact}}(s_i) \cdot r_{\mathrm{info}}(s_i)$, with Qwen-30B-A3B serving as the reward evaluator.
    - **Design Motivation**: Click signals are unavailable and inherently noisy in the cold-start phase (e.g., "iPhone for one dollar" may have high clicks but low quality), making intrinsic quality assessment more reliable.

2. **Uncertainty Sampling**:

    - **Function**: Selects the most valuable difficult samples from online data lacking click signals.
    - **Mechanism**: For each online query $q$, the model generates $k$ suggestions and scores them to calculate the reward variance $u_q = \frac{1}{k}\sum_{i=1}^k (R(s_i) - \frac{1}{k}\sum_j R(s_j))^2$. Queries with high uncertainty are selected as difficult samples for the next round of RL training.
    - **Design Motivation**: Training exclusively on click data introduces bias; uncertainty sampling increases data diversity and focuses on the model's weaknesses.

3. **Iterative RL Training**:

    - **Function**: Continuously optimizes query suggestion quality.
    - **Mechanism**: Multi-round iteration: In each round, the current model generates suggestions on non-click data $\rightarrow$ uncertainty sampling selects difficult samples $\rightarrow$ RL training updates the model $\rightarrow$ repeat. Each round uses a mixture of click data and difficult sample data.
    - **Design Motivation**: A single RL training session is insufficient to cover all scenarios; iterative training allows for the continuous discovery and resolution of model weaknesses.

### Loss & Training

RL training is conducted using the GRPO algorithm. The reward is a multiplicative combination of three dimensions (answerability $\times$ factuality $\times$ information gain $\times$ format) to ensure all three dimensions are satisfied simultaneously. The SFT phase utilizes positive click samples, while the RL phase employs quality-driven rewards. The base model is Qwen3-4B, and the reward evaluator is Qwen-30B-A3B.

## Key Experimental Results

### Main Results

| Model | Strict Accuracy | Valid Rate | Notes |
|------|----------------|------------|------|
| GPT-4.1-mini | 70.5 | 79.2 | One of the best closed-source models |
| Qwen-flash | 75.4 | 81.3 | Best closed-source model |
| Qwen3-4B(base) | 36.7 | 53.2 | Open-source baseline |
| Cold-EQS(Qwen3-4B) | Significant Gain | Significant Gain | Approaches or exceeds closed-source models |
| **Online Metrics** | **+6.81% chatUV** | | Performance verified in actual deployment |

### Key Findings
- Offline metrics show a strong positive correlation with online performance, validating the effectiveness of intrinsic quality assessment.
- In cold-start scenarios, quality-driven rewards are more reliable than click-driven rewards.
- Uncertainty sampling effectively mitigates the bias resulting from exclusive reliance on click data.
- Iterative RL training leads to continuous performance improvements.
- After training with Cold-EQS, the 4B model can approach or even surpass large closed-source models.

## Highlights & Insights
- **Paradigm Shift from Clicks to Quality**: Replaces CTR signals with intrinsic quality rewards, making it suitable for cold-start and more robust.
- **Rationality of Three-Dimensional Quality Assessment**: Answerability, factuality, and information gain precisely cover the core quality dimensions of query suggestions.
- **Offline-Online Consistency**: The strong positive correlation between offline quality metrics and online chatUV makes offline iteration reliable.
- **Actual Industrial Deployment**: Beyond academic experiments, the framework's practical value was verified on Alibaba's international e-commerce system.
- **Contribution of EQS-Benchmark**: Provides a standardized evaluation benchmark for e-commerce query suggestions.

## Limitations & Future Work
- The reward evaluator (Qwen-30B-A3B) itself may introduce bias, and the evaluator's quality directly impacts training effectiveness.
- Experiments were primarily validated on Alibaba's international e-commerce platform; cross-platform/cross-language generalization remains unknown.
- Uncertainty sampling uses a simple variance metric; more complex uncertainty estimation methods (e.g., Bayesian methods) might be more effective.
- The EQS-Benchmark is relatively small (16,949 entries); a larger-scale benchmark needs to be constructed.

## Related Work & Insights
- **vs CTR-based QS (Min et al., 2025)**: CTR methods require 20M+ online records, whereas Cold-EQS works effectively in the cold-start phase.
- **vs Retrieval-based Methods**: Retrieval methods are limited by historical query pools and cannot generate novel suggestions; Cold-EQS can generate diverse new queries.
- **vs Standard SFT**: Pure SFT amplifies click noise (e.g., "iPhone for one dollar"), whereas RL + quality rewards avoid this issue.

## Rating
- Novelty: ⭐⭐⭐⭐ The intrinsic quality-driven cold-start RL framework is novel in the e-commerce QS field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes offline and online experiments and multi-model comparisons, though ablation details could be more extensive.
- Writing Quality: ⭐⭐⭐ Well-structured, but author anonymization was incomplete, and some descriptions could be clearer.
- Value: ⭐⭐⭐⭐ Solves a practical industrial problem; the 6.81% online improvement validates its practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SenseJudge: Human-Centric Preference-Driven Judgment Framework](sensejudge_human-centric_preference-driven_judgment_framework.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[ICML 2026\] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](../../ICML2026/recommender/position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve.md)
- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[ACL 2026\] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)

</div>

<!-- RELATED:END -->
