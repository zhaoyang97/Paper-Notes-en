---
title: >-
  [Paper Note] From Lists to Emojis: How Format Bias Affects Model Alignment
description: >-
  [ACL 2025][LLM Alignment][RLHF] This paper systematically investigates how preference models (including human annotators, GPT-4, and open-source models) in RLHF exhibit formatting biases towards features such as bold text, lists, and emojis. It demonstrates that injecting less than 1% biased data can significantly introduce bias and proposes a debiasing method utilizing a dual-head reward model.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "RLHF"
  - "format bias"
  - "reward model"
  - "preference learning"
  - "alignment"
date: 2026-05-08
content_hash: 0e03011c3e0ba29b
---

# From Lists to Emojis: How Format Bias Affects Model Alignment

**Conference**: ACL 2025  
**arXiv**: [2409.11704](https://arxiv.org/abs/2409.11704)  
**Code**: None  
**Area**: Others  
**Keywords**: RLHF, format bias, reward model, preference learning, alignment

## TL;DR

This paper systematically investigates how preference models (including human annotators, GPT-4, and open-source models) in RLHF exhibit formatting biases towards features such as bold text, lists, and emojis. It demonstrates that injecting less than 1% biased data can significantly introduce bias and proposes a debiasing method utilizing a dual-head reward model.

## Background & Motivation

Reinforcement Learning from Human Feedback (RLHF) has become a key technology in LLM training pipelines for aligning model outputs with human preferences. However, a core challenge facing RLHF is "reward hacking," in which policy models achieve high scores by exploiting weaknesses in the reward model without actually improving response quality.

Verbosity bias is a well-known form of reward hacking where preference models tend to favor longer responses. However, this paper points out that besides verbosity bias, there is also a substantial amount of **format bias**, including preferences for bold text, lists, emojis, exclamation marks, links, and affirmative tones. These biases have been heavily neglected in existing research.

The core motivation of this paper is to investigate whether scores on current popular Instruction Following (IF) benchmarks truly reflect model capabilities. The authors find that top-ranking models on the AlpacaEval leaderboard exhibit a strong preference for generating responses containing bold text, lists, and exclamation marks, suggesting that rankings might reflect format manipulation capabilities rather than content quality.

## Method

### Overall Architecture

The research framework of this paper comprises three tiers: (1) identifying and quantifying format bias in preference models; (2) investigating how bias propagates from preference data to reward models and then to downstream alignment algorithms; and (3) proposing a debiasing method.

### Key Designs

1. **Format Bias Identification and Quantification**: The authors select seven formatting patterns (length, emoji, bold text, exclamation mark, list, link, and affirmative tone) and analyze the occurrence ratio of each pattern in preferred vs. unpreferred responses across four mainstream preference datasets (RLHFlow-700K, LMSYS-Arena-55K, AlpacaEval, and UltraFeedback). They find that the preference discrepancy for bold text is most pronounced in GPT-4 annotated datasets (preferred 42.76% vs unpreferred 16.78%).

2. **Bias Evaluation Dataset Construction**: For each formatting pattern, the authors construct an evaluation set by generating response pairs with and without specific formats. Specifically, responses containing specific formats are selected from model generations, and the formatting is removed to form paired samples. 200 pairs are generated for each pattern for evaluation. The win rate of an unbiased reward model should ideally be close to 50%.

3. **Bias Injection Experiments**: Based on the UltraFeedback dataset (71.6K pairs after filtering), a small amount (less than 1%) of biased data is injected. In the biased data, responses containing the target format are labeled as preferred. Experiments show that injecting merely 0.7% of list-biased data can elevate the reward model's win rate for lists from 51% to 77.5%.

4. **Dual-Head Reward Model Debiasing**: A dual-head RM is trained to separately predict the true reward $r^A$ and the format-related decoupled reward $r^D$. The total reward is the sum of both. A constraint loss is applied to minimize the correlation between the true reward head and format patterns while encouraging the decoupled head to capture the format information.

### Loss & Training

The overall training objective consists of two components:

- **Ranking Loss**: The standard Bradley-Terry preference ranking loss, using the sum of the two reward heads.
- **Constraint Loss**: $\mathcal{L}^C = |\rho(r^A, \#_p)| - \rho(r^D, \#_p)$, where $\rho$ is the Pearson correlation coefficient and $\#_p$ is the heuristic format counting function. This loss encourages the true reward head to be independent of format while ensuring the decoupled head is highly correlated with format.

To address the sparseness of formatting signals (e.g., bold text appears in less than 2% of responses), the authors propose a **reordering technique**: preference pairs with and without formatting are grouped and aligned, and the constraint loss calculation is skipped in batches where all format counts are zero.

## Key Experimental Results

### Main Results

| Preference Model | Bold Win Rate | List Win Rate | Emoji Win Rate | Link Win Rate |
|---------|----------|----------|-----------|----------|
| GPT-4 Turbo | 89.5% | 75.8% | 86.8% | 87.3% |
| ArmoRM-8B | 98.0% | 50.5% | 55.0% | 27.0% |
| Pairwise-PM-8B | 97.0% | 93.5% | 70.5% | 84.8% |
| Skywork-Critic-8B | 99.3% | 88.8% | 97.3% | 75.0% |
| FsfairX-8B | 95.5% | 68.5% | 15.0% | 64.5% |

Bias Injection Experiments:

| Training Data | Bold Win Rate | List Win Rate |
|---------|----------|----------|
| Baseline (Deformatted) | 57.5% | 51.0% |
| Baseline + 0.70% Bold | 88.0% | - |
| Baseline + 0.70% List | - | 77.5% |
| Baseline + 0.70% Bold + 1.40% List | 83.0% | 80.0% |

### Ablation Study

| Debiasing Config | Bold Win Rate | List Win Rate | Chat | Chat-Hard | Safety | Reasoning |
|---------|----------|----------|------|-----------|--------|-----------|
| No Debiasing | 89.0% | 92.5% | 98.3% | 71.4% | 83.1% | 85.1% |
| Bold Debiasing + Reordering (λ=0.2) | 52.5% | - | 97.5% | 71.1% | 83.0% | 87.3% |
| List Debiasing + Reordering (λ=0.1) | - | 54.0% | 98.4% | 72.9% | 83.6% | 89.4% |
| Bold&List Debiasing + Reordering (λ=0.2) | 50.5% | 53.0% | 97.2% | 72.8% | 82.9% | 89.7% |

### Key Findings

1. **Format bias is ubiquitous**: GPT-4, human annotators, and all open-source preference models exhibit preferences for specific formats, with GPT-4 generally showing stronger bias than humans.
2. **Minimal biased data can exploit reward models**: Less than 1% of biased data can significantly alter the behavior of a reward model.
3. **Online algorithms amplify bias**: Compared to offline DPO, online iterative DPO and PPO are more likely to exploit and amplify format bias, as online methods continuously explore and adapt to new data.
4. **Passive data filtering is insufficient**: Simply removing training data containing formatting results in a loss of 57% of the dataset, leading to degraded model capabilities.
5. **Reordering technique solves the sparsity issue**: For sparse formatting patterns, the reordering technique significantly improves the debiasing effect while preserving the performance of the reward model.

## Highlights & Insights

- This work is the first to systematically extend format bias research from verbosity to multiple patterns such as bold text, lists, and emojis, filling an important research gap.
- The "preference reversal" phenomenon is striking: GPT-4 sometimes prefers lower-quality responses simply because they have better formatting.
- The 1% attack threshold in bias injection experiments highlights the critical importance of preference data quality control in real-world systems.
- Re-evaluating the AlpacaEval ranking with the debiased reward model reveals that Llama-series models drop in ranking while closed-source models rise, exposing potential issues with current leaderboards.

## Limitations & Future Work

- Online DPO and PPO experiments are only conducted on relatively small models (8B), lacking verification on large-scale models.
- The hyperparameter $\lambda_C$ in the debiasing method requires manual tuning, presenting a trade-off between bias reduction and overall model capability.
- Debiasing experiments focus only on bold and list formats; the effectiveness on other formats (emojis, links, etc.) has not been verified.
- The impact of format bias in multi-turn conversation scenarios is not investigated.

## Related Work & Insights

- **ODIN** (Chen et al., 2024) proposed the concept of decoupled rewards to mitigate reward hacking in RLHF; this paper extends this idea to format bias.
- **OffsetBias** (Park et al., 2024) attempted debiasing but neglected format bias, which inadvertently aggravated list bias after running their debiasing procedure.
- The findings of this paper offer key insights for RLHF system design: there is a critical need to explicitly distinguish between format and content in both training and evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ Extends format bias from verbosity to multiple patterns with a novel perspective, though the core idea (decoupled reward) draws on existing work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple datasets, models, and algorithms (DPO/PPO/BoN), presenting a complete experimental pipeline.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich tables, though somewhat notation-heavy.
- Value: ⭐⭐⭐⭐⭐ Offers direct guidance for RLHF practices, exposing systemic issues in current alignment training and evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Model Extrapolation Expedites Alignment](expo_model_extrapolation.md)
- [\[ACL 2025\] HAF-RM: A Hybrid Alignment Framework for Reward Model Training](haf-rm_a_hybrid_alignment_framework_for_reward_model_training.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](../../ICML2025/llm_alignment/on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ICLR 2026\] Why DPO is a Misspecified Estimator and How to Fix It](../../ICLR2026/llm_alignment/why_dpo_is_misspecified_estimator.md)
- [\[ICLR 2026\] Reward Model Routing in Alignment](../../ICLR2026/llm_alignment/reward_model_routing_in_alignment.md)

</div>

<!-- RELATED:END -->
