---
title: >-
  [Paper Note] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning
description: >-
  [NeurIPS 2025][LLM Alignment][reward model] This paper proposes ResponseRank, a method that robustly learns utility differences by exploiting local relative differences in proxy signals of preference strength (e.g.…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "reward model"
  - "preference strength"
  - "response ranking"
  - "RLHF"
  - "sample efficiency"
date: 2026-05-08
content_hash: a29996e783a5179a
---

# ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning

**Conference**: NeurIPS 2025

**arXiv**: [2512.25023](https://arxiv.org/abs/2512.25023)

**Code**: None

**Area**: LLM Alignment

**Keywords**: reward model, preference strength, response ranking, RLHF, sample efficiency

## TL;DR

This paper proposes ResponseRank, a method that robustly learns utility differences by exploiting local relative differences in proxy signals of preference strength (e.g., response time and annotator agreement), significantly improving the sample efficiency of reward models.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Binary preference labels in RLHF (A is preferred over B) convey only the direction of preference, not its intensity. However, preference strength is critical for decision-making under uncertainty and for the generalization of preference models:

**Information Loss**: "Strongly preferring apple" and "slightly preferring apple" are indistinguishable under binary annotation.

**Unreliable Proxy Signals**: Proxy signals such as response time and annotator agreement can reflect preference strength but are noisy and subject to confounding factors.

**Incomparable Absolute Values**: Absolute values of response time are not comparable across different annotators or different questions.

**Low Sample Efficiency**: Failing to utilize preference strength information necessitates larger amounts of annotated data.

## Method

### Overall Architecture

ResponseRank converts preference strength information into ranking constraints by comparing the relative differences of proxy signals within carefully constructed strata, and uses these constraints to train more accurate reward models.

### Key Designs

**1. Local Stratification Strategy**

- Samples are stratified by feature (e.g., same prompt, same annotator).
- Proxy signals are compared **only within the same stratum**, controlling for systematic bias.
- Cross-stratum comparisons are avoided to eliminate confounding effects (e.g., differences in prompt difficulty).

**2. Relative Strength Ranking**

- Within each stratum, samples are sorted by proxy signal value in descending order.
- Higher proxy signal values (e.g., shorter response time, higher annotator agreement) indicate stronger preference.
- Ranking constraints are generated: if the preference of $(x_i, y_i^w, y_i^l)$ is stronger than that of $(x_j, y_j^w, y_j^l)$, then $|r(y_i^w) - r(y_i^l)| > |r(y_j^w) - r(y_j^l)|$.

**3. Ranking Constraint Training**

A ranking loss is added on top of the standard BT loss:
$$\mathcal{L}_{\text{rank}} = \sum_{(i,j) \in \text{rank-pairs}} \max(0, \Delta_j - \Delta_i + \text{margin})$$

where $\Delta_i = r(y_i^w) - r(y_i^l)$ is the utility difference.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{BT}} + \lambda \mathcal{L}_{\text{rank}}$$

- $\mathcal{L}_{\text{BT}}$: Standard Bradley-Terry preference loss.
- $\mathcal{L}_{\text{rank}}$: Ranking constraint based on preference strength.
- $\lambda$: Trade-off coefficient.

## Key Experimental Results

### Main Results

Synthetic preference learning (with simulated response time):

| Method | 20% Data | 50% Data | 100% Data |
|--------|----------|----------|-----------|
| BT (Standard) | 0.62 | 0.71 | 0.78 |
| Weighted BT | 0.65 | 0.73 | 0.79 |
| ResponseRank | **0.72** | **0.78** | **0.82** |

Language model reward learning (annotator agreement as proxy, RewardBench Accuracy):

| Method | Chat | Safety | Reasoning | Avg. |
|--------|------|--------|-----------|------|
| BT Standard | 72.5 | 80.3 | 68.2 | 73.7 |
| Margin-BT | 73.8 | 81.2 | 69.5 | 74.8 |
| ResponseRank | **76.2** | **83.5** | **72.1** | **77.3** |

### Ablation Study

Robustness under varying proxy signal quality:

| Noise Level | BT | Weighted BT | ResponseRank |
|-------------|-----|-------------|--------------|
| No noise | 0.78 | 0.85 | **0.86** |
| Low noise | 0.78 | 0.82 | **0.84** |
| Medium noise | 0.78 | 0.78 | **0.82** |
| High noise | 0.78 | 0.72 | **0.80** |

### Key Findings

1. ResponseRank achieves with only 20% of the data the same performance level as standard BT trained on 100%, representing a 5× improvement in sample efficiency.
2. Local stratification is the key to robustness—Weighted BT under high noise actually performs worse than standard BT.
3. The Pearson Distance Correlation (PDC) metric effectively distinguishes ordinal accuracy from cardinal utility learning.
4. Using simulated returns as proxy signals in RL control tasks is equally effective.

## Highlights & Insights

- **Minimal Assumptions**: Only assumes that proxy signals are locally valid, without making assumptions about their global distribution.
- **Robust Design**: The advantage over Weighted BT becomes more pronounced as noise increases.
- **New Metric PDC**: Pearson Distance Correlation disentangles ordinal and cardinal learning.

## Limitations & Future Work

1. The stratification strategy requires a sufficient number of within-stratum samples and may be unsuitable for small-scale datasets.
2. A systematic comparison of proxy signal choices (response time vs. annotator agreement vs. others) is lacking.
3. Ranking constraints only consider pairwise relationships and do not exploit total-order information across multiple samples.
4. Integration with direct preference optimization methods such as DPO remains unexplored.

## Related Work & Insights

- **RLHF**: Standard reinforcement learning from human feedback framework.
- **Preference Learning with RT**: Related work on response-time-assisted preference learning.
- **Learning to Rank**: Ranking learning methods from information retrieval.

## Rating

- ⭐ Novelty: 8/10 — The design combining local stratification and relative ranking is concise and effective.
- ⭐ Value: 8/10 — A 5× improvement in sample efficiency is highly valuable for practical annotation workflows.
- ⭐ Writing Quality: 8/10 — Experiments cover three settings: synthetic, language, and RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[NeurIPS 2025\] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay](improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)

</div>

<!-- RELATED:END -->
