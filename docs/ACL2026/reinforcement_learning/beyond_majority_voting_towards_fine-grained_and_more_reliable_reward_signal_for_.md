---
title: >-
  [Paper Note] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning
description: >-
  [ACL 2026][Reinforcement Learning][TTRL] Addressing the two major pain points of "confirmation bias + sparse rewards" caused by using majority voting for pseudo-labels in TTRL…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "TTRL"
  - "majority voting"
  - "step-wise confidence"
  - "subgroup partition"
  - "Pareto optimization"
date: 2026-05-08
content_hash: 3472f607a2bb8234
---

# Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2512.15146](https://arxiv.org/abs/2512.15146)  
**Code**: <https://github.com/szu-tera/SCOPE>  
**Area**: Reinforcement Learning / LLM Reasoning / Test-Time Training / Reward Model  
**Keywords**: TTRL, majority voting, step-wise confidence, subgroup partition, Pareto optimization

## TL;DR
Addressing the two major pain points of "confirmation bias + sparse rewards" caused by using majority voting for pseudo-labels in TTRL, SCOPE proposes step-wise confidence-weighted voting (moving beyond simple frequency) and dynamic Pareto-optimal subgroup partitioning (independently bootstrapping local consensus for each subgroup). On Qwen3-8B, it improves AIME 2024 performance from 47.13 to 52.70 and AIME 2025 from 27.40 to 31.00.

## Background & Motivation

**Background**: RLVR (RL with Verifiable Rewards) is a core training paradigm for top reasoning models like R1, Qwen3, and o1, but it depends on extensive human annotation. TTRL proposes training without labels at test time by using majority voting on multiple sampled responses to obtain a "consensus answer" as a pseudo-label for GRPO training.

**Limitations of Prior Work**: (1) **Confirmation Bias**: Majority voting treats all votes equally; if the model's most confident but incorrect answer happens to have the most votes, it is repeatedly reinforced, exacerbating errors. (2) **Sparse Rewards**: The entire group of $|\mathcal{G}|$ samples shares a single global consensus label, resulting in rewards that are either all correct or all incorrect, lacking fine-grained signals to help the model distinguish the "degree of error."

**Key Challenge**: Majority voting represents a brute-force discard of token-level confidence information—LLMs actually possess confidence levels at each step during reasoning. A correct solution might be a "minority vote that is confident at every step," while an incorrect solution might be a "majority vote that falters at an intermediate step." Furthermore, a single consensus flattens within-group diversity, making it difficult to balance quality and exploration.

**Goal**: To inject two types of signals into TTRL without human labels: (a) fine-grained step-wise confidence to allow correct but minority solutions to emerge, and (b) subdivision into subgroups to simultaneously obtain dense rewards and diverse supervision.

**Key Insight**: It is observed that step-wise confidence (the mean negative log-probability of top-$k$ tokens within each reasoning step) reflects the LLM's certainty in each segment of its reasoning chain. Simultaneously, partitioning the candidate pool into subgroups of size $m$, where each group votes independently, generates $n=|\mathcal{G}|/m$ different local consensuses, automatically providing diverse supervision targets.

**Core Idea**: Replace "equal-weighted voting" with "confidence-weighted voting" and "global consensus" with "automatically selected subgroup sizes" to make TTRL reward signals more accurate (de-biasing) and denser (de-sparsifying).

## Method

### Overall Architecture
The training iteration of SCOPE consists of four steps: (1) Sample $|\mathcal{G}|$ responses, split reasoning steps by `\n\n`, and calculate the average step confidence $\mathcal{C}_{AvgStep}^{(i)}$ for each response; (2) Evaluate several candidate subgroup sizes $m \in \mathcal{M}$, calculate the quality rate $q$ and exploration rate $e$ for each $m$, construct a Pareto frontier, and select the optimal $m^*$; (3) Partition responses into $n$ subgroups according to $m^*$, obtaining a local consensus $o_j^*$ for each group via bootstrap sampling and confidence-weighted voting; (4) Calculate rewards using $r(o, o_j^*) = \mathds{1}[\text{Ans}(o) = o_j^*]$ and update the strategy using GRPO.

### Key Designs

1.  **Step-wise Confidence-Weighted Voting**:
    - **Function**: Uses the "average step-level confidence" of each response as the voting weight, allowing minority solutions with stable and confident reasoning to prevail over majority solutions that are frequent but fail in the middle stages.
    - **Mechanism**: Token-level confidence is $\mathcal{C}_t = - \frac{1}{k} \sum_{j=1}^{k} \log P_t(j)$ (top-$k$ average negative log-probability). Reasoning steps are split by `\n\n`, with each step's confidence being $\mathcal{C}_{s_k} = \frac{1}{N_k} \sum_{t=1}^{N_k} \mathcal{C}_t$. The average for the entire response is $\mathcal{C}_{AvgStep}^{(i)} = \frac{1}{|\mathcal{L}|} \sum_{k=1}^{|\mathcal{L}|} \mathcal{C}_{s_k}$. The consensus label is determined by $o^* = \operatorname{argmax}_y \sum_{i=1}^{|\mathcal{G}|} \mathcal{C}_{AvgStep}^{(i)} \cdot \mathds{1}[\text{Ans}(o_i) = y]$.
    - **Design Motivation**: Token-level signals are too jittery (high-frequency function words push noise to extremes), and trace-level signals are too flat (one error is buried by dozens of correct segments). Step-level acts as a natural structural unit of reasoning, preserving structural precision while avoiding jitter. Alternatives like bottom-10% or tail-10% were disproven as focusing only on the weakest step penalizes "difficult but correct" reasoning.

2.  **Independent Bootstrap Voting within Subgroups**:
    - **Function**: Partitions $|\mathcal{G}|$ responses into $n$ subgroups of size $m$. Each subgroup independently derives a local consensus $o_j^*$, and the reward is calculated based on the subgroup goal: $r(o, o_j^*) = \mathds{1}[\text{Ans}(o) = o_j^*]$.
    - **Mechanism**: Subgroups are defined as $\mathcal{S} = \{S_j = \{o_{(j-1)m+1}, \dots, o_{jm}\}\}_{j=1}^{n}$. For each subgroup, a candidate set is obtained via bootstrap sampling from the global pool, and $o_j^*$ is determined by confidence-weighted voting. This creates $n$ independent estimates and potentially $n$ different supervision targets.
    - **Design Motivation**: A single global consensus forces all samples in a group to share a 0/1 reward—either all correct or all wrong—making the reward density extremely sparse. Subgrouping allows $n$ subgroups to hold different "local truths," encouraging GRPO to explore more reasoning paths. Subgroups that are too small ($m=1$) rely on single samples and are noisy; subgroups that are too large ($m=|\mathcal{G}|$) degenerate into global consensus.

3.  **Pareto-optimal Automatic Selection of $m^*$**:
    - **Function**: Eliminates manual tuning of $m$ by dynamically selecting the optimal subgroup size at each training step to balance "reasoning quality" and "exploration diversity."
    - **Mechanism**: Defines two metrics—quality $q = \frac{1}{|\mathcal{G}|} \sum_{j=1}^{n}\sum_{l=1}^{m} \mathds{1}[\text{Ans}(o_{(j-1)m+l}) = o_j^*]$ (within-group consistency) and exploration $e = \frac{|\{o_1^*, \dots, o_n^*\}|}{n}$ (ratio of unique consensuses). Enumerating candidate $m \in \{1, 2, 4, \dots\}$ yields a point set $(q_k, e_k)$ to form a Pareto frontier. For each point, the z-norm weighted distance $d_k = \sqrt{\lambda(1-\hat{q}_k)^2 + (1-\lambda)(1-\hat{e}_k)^2}$ is calculated, and $m^* = \operatorname{argmin}_{m_k} d_k$ is selected. $\lambda = 0.7$ is empirically optimal.
    - **Design Motivation**: A fixed $m$ is inevitably suboptimal—early stages require small $m$ for diversity to explore, while later stages require large $m$ for stable consensus. The Pareto framework naturally adjusts based on the current quality-exploration distribution.

### Loss & Training
The standard GRPO objective is used: $\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{|\mathcal{G}|}\sum_i \frac{1}{|o_i|}\sum_t \min[\rho_{i,t}\mathcal{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\mathcal{A}_i] - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]\right]$, where advantage $\mathcal{A}_i = (r(o_i) - \mu_g)/(\sigma_g + \epsilon)$. SCOPE's only modification is the source of $r(\cdot)$—shifting from "global majority vote" to "subgroup-specific confidence-weighted vote." Pareto evaluation adds approximately 10% additional computation.

## Key Experimental Results

### Main Results

| Model | Method | AIME 2024 | AIME 2025 | AMC | MATH-500 | Avg |
|------|------|-----------|-----------|-----|----------|-----|
| Qwen2.5-Math-1.5B | TTRL | 16.48 | 9.86 | 48.87 | 72.58 | 36.95 |
| Qwen2.5-Math-1.5B | **SCOPE** | **22.50** | **14.90** | **51.20** | **76.85** | **41.36** |
| Qwen3-1.7B | TTRL | 19.37 | 19.23 | 50.45 | 78.18 | 41.91 |
| Qwen3-1.7B | **SCOPE** | **21.66** | **19.71** | **53.46** | **81.27** | **44.02** |
| LLaMA3.1-8B-Inst | TTRL | 9.56 | 0.96 | 32.08 | 62.93 | 26.38 |
| LLaMA3.1-8B-Inst | **SCOPE** | **14.37** | **1.44** | **35.24** | 61.67 | **28.18** |
| Qwen3-8B | TTRL | 47.13 | 27.40 | 68.55 | 89.74 | 58.21 |
| Qwen3-8B | **SCOPE** | **52.70** | **31.00** | **74.09** | 91.01 | **62.20** |

On Qwen3-8B, AIME 2025 relative performance increased by 13.1%, and AMC increased by 8.1%. On LLaMA3.1-8B, AIME 2024 relative performance increased by 50.3%.

### Ablation Study

| Configuration | Qwen3-8B AIME 2024 | Qwen3-8B AIME 2025 |
|------|---------------------|---------------------|
| TTRL | 47.13 | 27.40 |
| **SCOPE Full** | **52.70** | **31.00** |
| w/o Conf (Pure Majority Voting) | 47.70 (-5.00) | 28.36 (-2.64) |
| w/o Subgroup (Global Consensus) | 47.91 (-4.79) | 26.92 (-4.08) |

Pseudo-Label Accuracy (PLA) analysis: TTRL 25.42 → SCOPE 30.71 (+20.81%), showing that subgroup partitioning reduces rather than exacerbates pseudo-label drift.

### Key Findings
- Both modules are essential: removing confidence drops performance by 5 points, and removing subgrouping drops it by 4-6 points, indicating that "precise signals" and "dense signals" are independent dimensions.
- Step-wise confidence is superior to other granularities: trace-level average masks critical errors, and bottom-10% focus penalizes "difficult but correct" solutions.
- Automatic $m^*$ selection outperforms fixed $m$: dynamic selection balances fast convergence with high peak values.
- $\lambda$ controls the quality-exploration trade-off: $\lambda=0.5$-$0.7$ is the sweet spot.
- Causal chain confirmed: SCOPE's 20% improvement in PLA directly leads to downstream gains, verifying that confirmation bias is a real bottleneck for TTRL.

## Highlights & Insights
- Weighting "voting weights" with the LLM's own step-wise confidence is a natural yet novel idea in TTRL—effectively externalizing internal reasoning signals as reward signals.
- Subgroup partitioning transforms the "single consensus" into "multiple local consensuses," effectively changing the group-relative advantage in GRPO from a binary 0/1 to a finer distribution.
- Pareto dynamic selection of $m^*$ is an elegant engineering solution that adapts to each step without manual scheduling.
- Step definition using `\n\n` is sufficient, suggesting that modern reasoning LLMs already have consistent step分割 habits.

## Limitations & Future Work
- Step partitioning relies on newline heuristics and assumes structured reasoning segments.
- Pareto evaluation introduces ~10% compute overhead.
- Verification is limited to mathematical benchmarks; transferability to open generation or multi-modal scenarios remains to be seen.
- Slight performance drop on MATH-500 with LLaMA3.1-8B, possibly due to noise in confidence signals from a weaker model.

## Related Work & Insights
- **vs TTRL**: Shares the base paradigm but replaces frequency-only voting with step-wise confidence and single global consensus with subgroups.
- **vs INTUITOR**: While INTUITOR uses self-certainty as an intrinsic reward, SCOPE uses confidence as a voting weight to combine group consensus and individual certainty.
- **vs Co-rewarding / EVOL-RL**: Those methods enhance diversity via semantic consistency or novelty, whereas SCOPE approaches diversity from the "spatial structure of reward signals."

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ACL 2026\] SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time](what_if_consensus_lies_selective-complementary_reinforcement_learning_at_test_ti.md)
- [\[CVPR 2026\] Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](../../CVPR2026/reinforcement_learning/specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/reinforcement_learning/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)

</div>

<!-- RELATED:END -->
