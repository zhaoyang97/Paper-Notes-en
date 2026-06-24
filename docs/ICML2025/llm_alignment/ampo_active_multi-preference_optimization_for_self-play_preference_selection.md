---
title: >-
  [Paper Note] AMPO: Active Multi-Preference Optimization for Self-play Preference Selection
description: >-
  [ICML 2025][LLM Alignment][Multi-preference optimization] The AMPO framework is proposed, combining online policy generation, multi-preference group contrastive loss, and active subset selection. By intelligently choosing small but highly informative subsets from a large pool of candidate responses for preference optimization, it achieves state-of-the-art results on AlpacaEval.
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "Multi-preference optimization"
  - "Active learning"
  - "Subset selection"
  - "Contrastive learning"
  - "Self-play alignment"
  - "k-medoids"
date: 2026-05-08
content_hash: 919284fcf27d3834
---

# AMPO: Active Multi-Preference Optimization for Self-play Preference Selection

**Conference**: ICML 2025

**arXiv**: [2502.18293](https://arxiv.org/abs/2502.18293)

**Code**: [HuggingFace Datasets](https://huggingface.co/Multi-preference-Optimization)

**Area**: LLM Alignment

**Keywords**: Multi-preference optimization, Active learning, Subset selection, Contrastive learning, Self-play alignment, k-medoids

## TL;DR

The AMPO framework is proposed, combining online policy generation, multi-preference group contrastive loss, and active subset selection. By intelligently choosing small but highly informative subsets from a large pool of candidate responses for preference optimization, it achieves state-of-the-art results on AlpacaEval.

## Background & Motivation

Traditional preference optimization methods (such as DPO) rely on pairwise comparisons and fail to fully capture the nuances of human judgment. Multi-preference approaches provide richer alignment signals by simultaneously considering an entire group of responses, but they face a critical bottleneck: modern LLMs can easily generate dozens of candidate responses per query, making it computationally infeasible to incorporate all responses into the training objective.

Specific issues include:

**Redundancy**: A large number of sampled responses are highly similar or near-duplicates, providing limited incremental information for gradient updates.

**Computational Bottleneck**: Processing all generated responses leads to memory explosion and diminishing training returns.

**Under-Coverage**: Focusing only on the best and worst answers can overlook crucial intermediate patterns—the "islands" of subtle failure modes.

The paper illustrates this with an "island metaphor": the response space of each prompt can be viewed as a group of semantic islands, and an ideal subset selection strategy should cover all islands rather than focusing solely on the highest peak or the lowest valley.

## Method

### Overall Architecture

AMPO unifies three core components:

1. **Online Policy Data Generation**: The model samples responses from its current policy distribution.
2. **Group Contrastive Preference Learning**: Uses a reference-free SWEPO/REFA objective.
3. **Active Subset Selection**: Selects a small and efficient training subset from a large candidate pool.

For each prompt $x$, $N$ responses are sampled from the policy $P_\theta(\cdot|x)$ (temperature 0.8). After being scored by a reward model, a subset of $K < N$ responses is selected for training.

### Key Designs: Active Selection Strategies

#### AMPO-BottomK (Baseline)

The simplest approach: directly select the $k$ lowest-reward responses as negative samples:

$$S^- = \text{argtopk}_i(-r_i, k)$$

Disadvantage: It might miss problematic patterns whose rewards are slightly higher than the bottom-k but are crucial for learning.

#### AMPO-Coreset (Clustering Selection)

Clusters the $N$ candidate responses into $k$ clusters in the embedding space and selects the response with the lowest reward from each cluster:

$$i_j^- = \arg\min_{i \in C_j} r_i, \quad j = 1, \ldots, k$$

This ensures that each semantic "pattern" contributes at least one negative sample, achieving broad coverage across different semantic regions.

#### AMPO-OptSelect (Theoretically Optimal Selection)

Maximizes expected rewards based on the assumption of Lipschitz continuity. Let the weight be defined as $w_i = \exp(\bar{r} - r_i)$ (giving higher weight to lower rewards). The coverage cost is defined as:

$$\text{Cost}(S) = \sum_{i=1}^n w_i \min_{j \in S} A_{i,j}$$

where $A_{i,j} = \|\mathbf{e}_i - \mathbf{e}_j\|_2$. Minimizing this cost is equivalent to a weighted k-medoids problem, which can be solved approximately via Mixed-Integer Programming (MIP) or local search.

### Loss & Training

A reference-free group contrastive objective (SWEPO/REFA) is adopted:

$$L_{\text{swepo}}(\theta) = -\log\left(\frac{\sum_{i \in S^+} \exp[s'_\theta(y_i|x)]}{\sum_{i \in (S^+ \cup S^-)} \exp[s'_\theta(y_i|x)]}\right)$$

where $s'_\theta(y_i|x) = \log P_\theta(y_i|x) + \alpha(r_i - \bar{r})$, and $\alpha$ is a hyperparameter. This loss encourages the model to increase the log-probability of positive samples while decreasing that of negative samples.

## Key Experimental Results

### Main Results

| Method | AlpacaEval LC (%) | AlpacaEval WR (%) | Arena-Hard WR (%) | MT-Bench |
|------|-------------------|--------------------|--------------------|----------|
| GPT-4 Base | 28.4 | 28.4 | 26.9 | 7.93 |
| Best-vs-worst (SimPO) | 47.6 | 44.7 | 34.6 | 7.51 |
| AMPO-BottomK | 50.8 | 50.5 | 35.3 | 8.11 |
| **AMPO-Coreset** | **52.4** | **52.1** | **39.4** | **8.12** |
| AMPO-OptSelect | 51.6 | 51.2 | 37.9 | 7.96 |

The base model is Llama-3-Instruct 8B. Different variants of AMPO outperform strong baselines such as SimPO across all metrics.

### Ablation Study: Key Hyperparameters

| Analysis Dimension | Key Findings |
|----------|----------|
| Sampling Temperature | Performance generally decreases as the temperature increases; Coreset and OptSelect are more robust to temperature changes. |
| Gamma Parameter | LC-WR and WR scores continuously improve as gamma increases from 1 to 3. |
| Beta Parameter | Consistently strong performance is produced within the 5.0-10.0 range. |
| Embedding Space Diversity | t-SNE visualization shows that the responses selected by Coreset/OptSelect are more scattered and achieve broader coverage. |

### Key Findings

1. **Limitations of BottomK**: The selected negative samples are highly concentrated in dense regions of the embedding space, leading to redundant feedback.
2. **Advantages of Coreset**: It covers more diverse semantic regions, achieving the most significant improvement on Arena-Hard (+4.8% vs. SimPO).
3. **Consistency between Theory and Practice**: The Lipschitz theoretical guarantees of OptSelect are consistent with the experimental outcomes.

## Highlights & Insights

1. **Solid Theoretical Foundation**: It is proved that minimizing the coverage cost is equivalent to maximizing the expected reward under Lipschitz constraints (Theorem 6.1), and local search can achieve a 5x approximation guarantee in polynomial time (Theorem 6.2).
2. **Intuitive Island Metaphor**: Analogizing the subset selection problem to covering semantic islands helps explain why diversity-based selection outperforms selection based purely on scores.
3. **High Practicality**: It avoids exhausting all responses; a small, well-selected subset is sufficient to achieve superior alignment.
4. **Open-Source Contribution**: The AMPO-Coreset-Selection and AMPO-Opt-Selection datasets are publicly released.

## Limitations & Future Work

1. The validation of the experiments is primarily on Llama-3 8B, lacking evaluation on larger model scales.
2. The Lipschitz constant $L$ is difficult to estimate precisely in practice.
3. The quality of the reward model directly impacts the effectiveness of subset selection.
4. The single positive sample assumption (selecting only the highest-reward response as the positive sample) may lack flexibility.

## Related Work & Insights

- **Preference Optimization (DPO/SimPO/ORPO)**: AMPO unifies these methods under a multi-preference framework, replacing pairwise comparisons with group contrastive objectives.
- **Active Learning**: Treats response selection as an active learning problem, selecting the most valuable samples from an information-theoretic perspective.
- **Coreset Construction**: Leverages core-set theory from computational geometry to achieve efficient subset coverage.
- **Insights**: The principles of active selection can be extended to other alignment scenarios (e.g., query selection in RLHF).

## Rating

- Novelty: ⭐⭐⭐⭐ — The unified framework of group contrastive learning and active selection is novel, effectively combining theoretical analysis with practice.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation on AlpacaEval, Arena-Hard, and MT-Bench with thorough ablation studies.
- Value: ⭐⭐⭐⭐ — Directly applicable to improving LLM alignment training workflows.
- Recommendation Index: ⭐⭐⭐⭐ — Highly recommended for researchers focusing on LLM alignment and preference optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](../../ICLR2026/llm_alignment/activedpo_active_direct_preference_optimization_for_sample-efficient_alignment.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](../../ACL2025/llm_alignment/automixalign_adaptive_data_mixing.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](../../ACL2025/llm_alignment/expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)
- [\[ACL 2026\] Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs](../../ACL2026/llm_alignment/team-based_self-play_with_dual_adaptive_weighting_for_fine-tuning_llms.md)

</div>

<!-- RELATED:END -->
