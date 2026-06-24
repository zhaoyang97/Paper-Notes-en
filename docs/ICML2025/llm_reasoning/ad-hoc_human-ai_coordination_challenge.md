---
title: >-
  [Paper Note] Ad-Hoc Human-AI Coordination Challenge (AH2AC2)
description: >-
  [ICML2025][Reasoning][Human-AI Coordination] This work proposes the AH2AC2 challenge—based on the cooperative card game Hanabi—which constructs human proxy agents via behavioral cloning and regularized reinforcement learning, and open-sources a limited human dataset to provide a standardized, reproducible evaluation framework for human-AI ad-hoc coordination research.
tags:
  - "ICML2025"
  - "Reasoning"
  - "Human-AI Coordination"
  - "Ad-Hoc Teamplay"
  - "Hanabi"
  - "Behavioural Cloning"
  - "Regularised RL"
  - "Human Proxy Agents"
  - "benchmark"
date: 2026-05-08
content_hash: 3c2974eb6d6186d4
---

# Ad-Hoc Human-AI Coordination Challenge (AH2AC2)

**Conference**: ICML2025  
**arXiv**: [2506.21490](https://arxiv.org/abs/2506.21490)  
**Code**: [FLAIROx/ah2ac2](https://github.com/FLAIROx/ah2ac2)  
**Area**: LLM Reasoning  
**Keywords**: Human-AI Coordination, Ad-Hoc Teamplay, Hanabi, Behavioural Cloning, Regularised RL, Human Proxy Agents, benchmark

## TL;DR

This work proposes the AH2AC2 challenge—based on the cooperative card game Hanabi—which constructs human proxy agents via behavioral cloning and regularized reinforcement learning, and open-sources a limited human dataset to provide a standardized, reproducible evaluation framework for human-AI ad-hoc coordination research.

## Background & Motivation

### Core Problem
AI agents in the real world must collaborate with humans in an ad-hoc manner (ad-hoc teamplay), but existing methods face two major dilemmas:

**Self-play overfitting**: Agents trained via traditional self-play develop exclusive, idiosyncratic communication protocols, making them unable to coordinate with unseen partners (especially humans).

**Unreproducible evaluation**: Prior assessments of human-AI coordination relied on closed-source datasets and private proxy agents, making research results difficult to replicate and compare.

### Why Hanabi?
Hanabi is an ideal testbed because it simultaneously features:

- **Imperfect information**: Players cannot see their own cards.
- **Restricted communication**: Information can only be transmitted through limited hints.
- **Theory of Mind**: Players must reason about their partner's intentions and knowledge.
- **Cooperative actions**: All players share a single goal (achieving a maximum score of 25 points).

### Research Motivation
Existing literature lacks standardized evaluation methods for human-AI coordination. Various studies employ different proxy agents and datasets, preventing fair comparison between methods. This work aims to bridge this gap by providing a unified evaluation protocol and public baselines.

## Method

### Overall Architecture: AH2AC2 Challenge Design

AH2AC2 comprises two parts of evaluation tasks:

1. **Part 1 — Coordination with Human Proxies**: Participants develop agents that play 1,000 Hanabi games with the human proxy agents provided by the paper.
2. **Part 2 — Human Action Prediction**: Predicting human player actions on a closed-source human game dataset (evaluated using cross-entropy loss).

Data sources are from the hanab.live platform: a total of 101,096 two-player games and 46,525 three-player games were collected. Among these, 3,079 games (1,858 for two-player + 1,221 for three-player) are open-sourced for participants to use, intentionally limiting the data volume to encourage research into data-efficient methods.

### Key Designs: HDR-IPPO (Human-Data-Regularised IPPO)

Training of the human proxy agents consists of two steps:

**Step 1: Behavioural Cloning (BC)**

- Use an LSTM architecture to model the policy $\pi_\theta^{BC}$, taking the action-observation history (AOH) as input.
- Train via supervised learning using standard cross-entropy loss.
- Evaluate self-play scores at the end of each training epoch and save the optimal parameters $\theta'$.

**Step 2: Regularised Reinforcement Learning**

- Initialize the human proxy policy $\pi_\theta^{HP}$ from the optimal BC parameters $\theta'$.
- Train via self-play using IPPO (Independent PPO).
- Add a KL regularization term to constrain the final policy from deviating too far from human behavior:

$$\mathcal{L}_t^{\text{HDR-IPPO}}(\theta) = (1 - \lambda) \cdot \mathcal{L}_t^{\text{IPPO}}(\theta) + \lambda \cdot D_{\text{KL}}(\pi_{\theta'}^{BC} \| \pi_\theta^{HP})$$

where $\lambda \in [0, 1]$ controls the regularization strength. This allows the final policy to enhance its game-playing capability while retaining human style.

### Loss & Training

- Use LSTM architecture to capture sequential features of actions and observations.
- Although model parameters are fixed, the agents' behaviors are dynamic—they make decisions based on the complete game history (including partners' actions).
- Train 4 human proxies in total (2 for the two-player setting + 2 for the three-player setting), ensuring diversity via different hyperparameters and random seeds.
- The evaluation API hides agent weights to prevent overfitting; pre-registration is required to submit experiments.

### Overfitting-Proof Design of Evaluation Protocol

- Human proxies are not released publicly and are only interacted with via the API.
- Each evaluation is limited to 1,000 games.
- Experiments must be pre-registered to obtain API access.
- Evaluation results are published on the leaderboard.

## Key Experimental Results

### Open-Source Dataset Statistics

| Setting | Metric | Min | Max | Mean | Median | Std |
|------|------|--------|--------|--------|--------|--------|
| 2P (1858 games) | Score | 13 | 25 | 23.37 | 24 | 1.86 |
| 2P (1858 games) | Game Length | 52 | 76 | 65.45 | 66 | 3.35 |
| 3P (1221 games) | Score | 14 | 25 | 23.25 | 24 | 1.91 |
| 3P (1221 games) | Game Length | 45 | 67 | 57.86 | 58 | 3.38 |

### Human Proxy Self-Play Performance (5,000 Games Evaluation)

| Metric | HP1 (2P) | HP2 (2P) | HP3 (3P) | HP4 (3P) |
|------|----------|----------|----------|----------|
| Avg Self-Play Score | 22.55±0.03 | 22.97±0.03 | 20.88±0.03 | 21.21±0.03 |
| Gain Relative to BC | +3.0 | +4.0 | +15.7 | +13.9 |
| Perfect Score % | 23.86% | 29.66% | 2.76% | 3.88% |
| BC Perfect Score % | 16.12% | 19.88% | 1.34% | 1.80% |
| Zero Score % | 0.10% | 0.04% | 0.34% | 0.20% |
| BC Zero Score % | 11.42% | 17.70% | 75.82% | 66.02% |

The improvement in the three-player (3P) setting is particularly significant: while the BC policy's zero-score rate was as high as 66-76%, HDR-IPPO reduced it to less than 0.5%.

### AH2AC2 Leaderboard Main Results

| Players | Method | Mean | Median | Cross-Entropy |
|--------|------|------|--------|--------|
| 2P | OBL (L4) | **21.04** | **22** | 1.33 |
| 2P | BR-BC | 19.41 | 20 | 10.82 |
| 2P | FCP | 14.01 | 16 | 3.52 |
| 2P | OP | 13.91 | 19 | 7.81 |
| 2P | HDR-IPPO | 12.76 | 15 | 0.96 |
| 2P | IPPO | 10.16 | 14 | 12.60 |
| 2P | DeepSeek-R1 (H-Group) | 9.91 | 0 | - |
| 2P | DeepSeek-R1 | 5.43 | 0 | - |
| 2P | BC | 2.12 | 0 | 0.86 |
| 3P | DeepSeek-R1 (H-Group) | **14.62** | **18** | - |
| 3P | HDR-IPPO | 14.03 | 16 | 0.80 |
| 3P | OP | 12.87 | 18 | 6.40 |
| 3P | BR-BC | 11.89 | 12 | 29.89 |
| 3P | FCP | 11.55 | 6 | 5.97 |
| 3P | IPPO | 6.34 | 0 | 8.60 |
| 3P | BC | 3.31 | 0 | 0.70 |

### Ablation Study: Human Proxy Behavior Analysis

| Setting | Source | IPP (Information/Play) | Communicativeness |
|------|------|-------------------|------------------------------|
| 2P | Human Dataset | 0.44 | 0.47 |
| 2P | HP1 | 0.43 | 0.45 |
| 2P | HP2 | 0.44 | 0.48 |
| 3P | Human Dataset | 0.42 | 0.49 |
| 3P | HP3 | 0.44 | 0.47 |
| 3P | HP4 | 0.44 | 0.46 |

The human proxy agents align almost identically with real human data across both behavioral metrics—IPP and Communicativeness—validating their human likeness.

### Action Prediction Performance

| Metric | HP1 (2P) | HP2 (2P) | HP3 (3P) | HP4 (3P) |
|------|----------|----------|----------|----------|
| Accuracy | 0.63 | 0.63 | 0.43 | 0.44 |
| Accuracy Drop vs. BC | -0.03 | -0.08 | -0.08 | -0.07 |
| Cross-Entropy Loss | 0.53 | 0.54 | 0.63 | 0.60 |
| Top-10% Accuracy | 0.82 | 0.82 | 0.71 | 0.73 |
| Top-20% Accuracy | 0.95 | 0.95 | 0.87 | 0.88 |

While maintaining high self-play performance, the human proxy agents' action prediction accuracy drops by only 3-8 percentage points compared to pure BC.

## Highlights & Insights

1. **The power of BC + Regularised RL**: A strategy relying solely on behavioral cloning suffers from a 76% zero-score rate in the three-player setting. Adding KL-regularized IPPO reduces this rate to 0.3%, proving that regularized RL is an effective means to remedy the generalization flaws of BC.
2. **The unexpected advantage of OBL**: OBL, which uses no human data, surprisingly achieves the highest score (21.04) in the two-player setting, revealing a critical research gap—existing methods cannot effectively leverage small amounts of human data to enhance coordination capability.
3. **Preliminary power of LLMs**: DeepSeek-R1 outperforms all other baselines in the three-player setting, demonstrating the intrinsic cooperative potential of LLMs. However, its poor performance in the two-player setting and median score of 0 indicate highly unstable behavior.
4. **Exquisitely designed evaluation protocol**: The triple-mechanism of API + pre-registration + limited game count effectively prevents overfitting, reflecting a rigorous approach to experimental science.
5. **Urgent need for data efficiency**: Open-source datasets are intentionally limited (only 3,079 games vs. 147,000 games for training), explicitly pointing to data-efficient learning as a critical research direction.

## Limitations & Future Work

1. **Limited to 2/3 player settings**: Hanabi supports 2-5 players, but due to data availability constraints, the challenge only covers two-player and three-player scenarios.
2. **Lack of direct human evaluation**: Although the human proxy agents align closely with humans on behavioral metrics, they have not been validated through direct play against real human players.
3. **Single source of data**: All data originates from the H-Group convention player community on hanab.live, which may not represent the broader diversity of human playstyles.
4. **Insufficient LLM evaluation**: DeepSeek-R1 was only evaluated over 100 games (compared to 1,000 games for other methods), casting doubt on its statistical significance.
5. **Lack of theoretical analysis**: The effectiveness of HDR-IPPO is primarily empirically validated, lacking theoretical explanations for why KL regularization preserves human compatibility.
6. **Exclusion of game variants**: Hanabi variants such as rainbow cards were not tested, making it impossible to assess the generalization capability of the methods.
7. **Poor performance of FCP in complex partially observable environments**, indicating that population-based methods might not be suitable for such scenarios.

## Related Work & Insights

- **Hanabi benchmark**: Bard et al. (2019) introduced the Hanabi Challenge, and SPARTA approached a perfect score (24.61/25) in self-play, but ad-hoc coordination remains an open problem.
- **Regularised RL paradigm**: Bakhtin et al. (2022) used human data to regularize RL in Diplomacy; Cornelisse & Vinitsky (2024) extended this approach to autonomous driving; this paper applies it to Hanabi.
- **Zero-Shot Coordination (ZSC)**: OBL (Hu et al., 2021) and OP (Hu et al., 2020) explore coordination methods that bypass human data.
- **FCP**: Strouse et al. (2021) proposed population-based methods, and this paper presents the first evaluation of their performance in Hanabi.
- **LLMs as game-playing agents**: The evaluation of DeepSeek-R1 opens up new avenues for examining Theory of Mind capabilities of LLMs in cooperative games.

## Rating
- Novelty: ⭐⭐⭐⭐ (4/5) — The first standardized and open-source evaluation framework for Hanabi Human-AI coordination, featuring a rigorously designed evaluation protocol.
- Experimental Thoroughness: ⭐⭐⭐⭐ (4/5) — Rich baselines (including LLMs) and multidimensional validation of human-likeness, though the LLM evaluation is limited in scale.
- Writing Quality: ⭐⭐⭐⭐ (4/5) — Clearly structured, with well-defined problems and motivations.
- Value: ⭐⭐⭐⭐⭐ (5/5) — Provides a much-needed standardized benchmark for Human-AI coordination research. The open-source data and API-based evaluation represent important contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoAct: Co-Active LLM Preference Learning with Human-AI Synergy](../../ACL2026/llm_reasoning/coact_co-active_llm_preference_learning_with_human-ai_synergy.md)
- [\[ACL 2025\] Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation](../../ACL2025/llm_reasoning/towards_safety_reasoning_in_llms_ai-agentic_deliberation_for_policy-embedded_cot.md)
- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](../../ICML2026/llm_reasoning/cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[AAAI 2026\] BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](../../AAAI2026/llm_reasoning/blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](../../ICML2026/llm_reasoning/reward_modeling_from_natural_language_human_feedback.md)

</div>

<!-- RELATED:END -->
