---
title: >-
  [Paper Note] LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification
description: >-
  [ACL 2026][Reinforcement Learning][RLVR] LENS discovers that many exploration failures in RLVR are caused not by problem difficulty…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "distraction tokens"
  - "instruction purification"
  - "rollout efficiency"
  - "reasoning enhancement"
date: 2026-05-08
content_hash: 2b69d504d13fb8c1
---

# LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification

**Conference**: ACL 2026  
**arXiv**: [2601.21244](https://arxiv.org/abs/2601.21244)  
**Code**: [https://github.com/RUCBM/LENS](https://github.com/RUCBM/LENS)  
**Area**: Reinforcement Learning / LLM Reasoning  
**Keywords**: RLVR, distraction tokens, instruction purification, rollout efficiency, reasoning enhancement

## TL;DR

LENS discovers that many exploration failures in RLVR are caused not by problem difficulty, but by a small amount (<5%) of distraction tokens in the prompt. By identifying and removing these tokens to improve rollout success rates and transferring the learning signals to policy optimization on the original noisy prompts, it achieves an average improvement of 3.88% and a 1.6x speedup.

## Background & Motivation

**Background**: RLVR (e.g., GRPO) has significantly enhanced LLM reasoning capabilities. However, a core challenge in complex tasks is that correct rollouts are extremely rare, leading to a lack of positive samples and low training efficiency or collapse.

**Limitations of Prior Work**: Existing strategies follow two paths: (1) scaling exploration (increasing the number of rollouts), which remains computationally expensive without improving efficiency; (2) filtering zero-variance prompts (skipping entirely failed prompts), which sacrifices the exploration of difficult samples. Neither addresses the root cause of exploration failure.

**Key Challenge**: Low-success-rate prompts contain valuable training signals, but current methods either ignore them (filtering) or inefficiently attempt to overcome them via brute-force exploration scaling.

**Goal**: Identify the root cause of exploration failure and design a targeted solution to improve rollout efficiency without increasing computational costs.

**Key Insight**: Fine-grained token-level analysis reveals that failures are often caused by a few tokens introducing excessive distraction—these tokens cause the policy to deviate too far from the reference model in token space. Simply deleting these tokens can increase the rollout accuracy of failed prompts by over 20%.

**Core Idea**: A few distraction tokens in the prompt are the critical cause of exploration failure. By first "purifying" the prompt to obtain successful rollouts and then "transferring" the learning signals back to the original prompt, the model learns to ignore distractions rather than relying on purification.

## Method

### Overall Architecture

LENS consists of two stages: (1) Distraction token identification and purification—calculating a distraction score for each token in the prompt and removing a small number of high-scoring tokens; (2) Calibrated Rollout Policy Optimization (CRPO)—using successful rollouts from purified prompts to replace failed rollouts of original prompts, followed by policy optimization on the original prompts using importance correction and sample reweighting.

### Key Designs

1.  **Distraction Token Identification**:
    - **Function**: Precisely locate the few tokens in the prompt that cause exploration failure.
    - **Mechanism**: The distraction score is defined as the absolute log-probability deviation between the current policy and the reference model at each token position: $S_I(s,a) = |\log \pi_\theta(a|s) - \log \pi_{\text{ref}}(a|s)|$. A high distraction score indicates that the token causes the policy to deviate excessively from the reference distribution. Tokens are ranked by score in descending order, and the top $k = \lceil \gamma \cdot |x_i| \rceil$ tokens (where $\gamma$ is 1%-5%) are removed to obtain the purified prompt.
    - **Design Motivation**: The reference model provides a stable distribution baseline learned from data; large deviations typically originate from reward over-optimization or noise/misleading signals.

2.  **Calibrated Rollout Policy Optimization (CRPO)**:
    - **Function**: Transfer learning signals from purified prompts to the policy optimization of the original prompts.
    - **Mechanism**: For prompts with low success rates (success rate < $\tau$), rollouts are sampled on the purified prompt. If the success rate improves after purification, successful samples from the purified rollout replace an equivalent number of failed samples in the original rollout. Crucially, all policy optimization is performed on the original prompts (using importance ratio $\rho(y;\theta) = \frac{\pi_\theta(y|x_i)}{\tilde{w}(y) \pi_{\text{old}}(y|x^{\text{roll}}(y))}$ to correct distribution mismatch), forcing the model to learn correct reasoning even under noisy conditions.
    - **Design Motivation**: Training directly on purified prompts only results in "learning in a clean environment," which fails to transfer to noisy environments. CRPO uses a transfer mechanism to teach the model to identify and ignore distractions.

3.  **Sample Reweighting**:
    - **Function**: Balance the weights of original successful samples and purified successful samples.
    - **Mechanism**: The original success rate $\bar{a}_i$ is used as a scaling factor: original successful samples are weighted by $\bar{a}_i$, while purified successful samples and unreplaced failed samples are weighted by $1-\bar{a}_i$. Optimization is performed using a PPO-style clipped objective with KL regularization.
    - **Design Motivation**: When the original success rate is low, the signal from purified samples should be trusted more; when it is high, the original samples should maintain their dominance.

### Loss & Training

A PPO-style clipped objective is used: $$\mathcal{L}(\theta) = -\sum_{y} \min(\rho(y;\theta)\hat{A}(y), \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat{A}(y)) + \beta D_{\text{KL}}$$. Advantages are calculated in a group-relative manner across the reconstructed rollout set.

## Key Experimental Results

### Main Results

**Mathematical Reasoning Benchmark Pass@1 (Llama3.2-3B-Instruct)**

| Method | MATH | Olympiad | AIME24 | Avg (7 benchmarks) |
| :--- | :--- | :--- | :--- | :--- |
| + GRPO | 51.60 | 44.68 | 6.25 | 23.98 |
| + DAPO | 53.00 | 47.01 | 9.79 | 25.32 |
| + GRPO_extended | 51.20 | 44.68 | 6.25 | 24.33 |
| + **LENS_GRPO** | **55.80** | **48.83** | **10.62** | **27.03** |

### Ablation Study

| Config | Key Metric | Description |
| :--- | :--- | :--- |
| Full LENS | Optimal | Identification + Purification + CRPO |
| Purification Only (No CRPO) | Suboptimal | Model only learns in a clean environment |
| Random Deletion | Decrease | Validates the effectiveness of distraction scores |
| ~20% prompts benefit | — | Necessitates the conditional activation design of CRPO |

### Key Findings

- **Significant reduction in zero-reward prompts**: LENS reduces the proportion of zero-reward prompts on DeepMath from ~80% (GRPO) to ~40%.
- Deleting <5% of tokens can improve rollout accuracy for failed prompts by over 20%, validating the hypothesis that "a few tokens cause most failures."
- LENS reaches GRPO's performance with only 60% of the training steps, achieving a 1.6x speedup.
- Improvements of 1.83% were observed on four out-of-distribution general reasoning benchmarks, indicating transferable robustness.
- Compared to scaling exploration (increasing rollouts) and prompt filtering, LENS achieves superior performance with fewer computational resources.

## Highlights & Insights

- The discovery that "a few distraction tokens cause exploration failure" is counter-intuitive and compelling, opening a new perspective for RLVR research.
- The philosophy of CRPO is ingenious: instead of training in a clean environment and expecting transfer, it uses signals from a clean environment to calibrate learning in a noisy one, essentially teaching the model to "ignore distractions."
- The definition of the distraction score (as log-probability deviation between the policy and reference model) is simple and efficient, requiring no auxiliary models.

## Limitations & Future Work

- Validated only on 3B-4B scale models; effectiveness on larger models (7B+) is unknown.
- The distraction score depends on the reference model; a low-quality reference model may lead to misjudgment.
- The deletion ratio $\gamma$ requires manual tuning and may vary across datasets.
- Only about 20% of prompts exhibit improved rollout accuracy after deletion, which limits the scope of CRPO's conditional activation.

## Related Work & Insights

- **vs GRPO/DAPO**: LENS is a plug-and-play improvement that enhances rollout quality without altering the underlying RL algorithm.
- **vs Scaling Exploration**: Scaling increases computational costs without improving efficiency, whereas LENS improves efficiency without increasing costs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "distraction token" discovery and "purification + transfer" solution are entirely fresh perspectives.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete baseline comparisons and ablations, though model scale coverage is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, precise methodology, and clear algorithm pseudocode.
- Value: ⭐⭐⭐⭐⭐ Provides a new framework and practical solution for exploration efficiency in RLVR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Less is More: Clustered Cross-Covariance Control for Offline RL](../../ICLR2026/reinforcement_learning/less_is_more_clustered_cross-covariance_control_for_offline_rl.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[NeurIPS 2025\] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners](../../NeurIPS2025/reinforcement_learning/when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)
- [\[ACL 2026\] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning](beyond_majority_voting_towards_fine-grained_and_more_reliable_reward_signal_for_.md)

</div>

<!-- RELATED:END -->
