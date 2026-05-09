---
title: >-
  [Paper Note] ExGRPO: Learning to Reason from Experience
description: >-
  [ICLR 2026][Reinforcement Learning][Experience Replay] This paper presents the first systematic study of what types of reasoning experiences are most valuable for RLVR, finding that medium-difficulty problems paired with low-entropy trajectories are most effective. Based on these findings, it proposes the ExGRPO framework for experience management and mixed-policy optimization, achieving an average gain of +3.5 points on mathematical reasoning and +7.6 points on general reasoning.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Experience Replay
  - RLVR
  - Reasoning RL
  - Experience Management
  - GRPO
date: 2026-05-08
content_hash: c4798edbd2c0e5ab
---

# ExGRPO: Learning to Reason from Experience

**Conference**: ICLR 2026
**arXiv**: [2510.02245](https://arxiv.org/abs/2510.02245)
**Code**: [GitHub](https://github.com/RanranZhang/ExGRPO)
**Area**: LLM Reasoning / Reinforcement Learning
**Keywords**: Experience Replay, RLVR, Reasoning RL, Experience Management, GRPO

## TL;DR
This paper presents the first systematic study of what types of reasoning experiences are most valuable for RLVR, finding that medium-difficulty problems paired with low-entropy trajectories are most effective. Based on these findings, it proposes the ExGRPO framework for experience management and mixed-policy optimization, achieving an average gain of +3.5 points on mathematical reasoning and +7.6 points on general reasoning.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become a core paradigm for enhancing LLM reasoning capabilities, with on-policy methods such as GRPO being the dominant approach. During training, models generate large volumes of reasoning trajectories (experiences).

**Limitations of Prior Work**: Standard on-policy training discards rollout experiences after a single gradient update, leading to wasted computational resources and training instability. Although experience replay has been extensively studied in traditional RL, the fundamental question of what experiences are most valuable in large-model RLVR settings remains underexplored.

**Key Challenge**: The vast amount of collected experience is not uniformly valuable — some problems are too easy (providing no learning signal), others too hard (introducing excessive noise); some trajectories reason correctly, while others arrive at correct answers through flawed reasoning. Identifying and exploiting high-value experiences is therefore critical.

**Goal**: (1) What constitutes valuable reasoning experience? (2) How can such experiences be systematically managed and reused?

**Key Insight**: A systematic analysis of experience value along two dimensions — problem difficulty and trajectory entropy. The study finds that medium-difficulty problems (accuracy 25%–75%) provide the strongest optimization signal, and low-entropy trajectories correspond to higher-quality reasoning chains.

**Core Idea**: Manage experiences via difficulty-based bucketing, and prioritize sampling of medium-difficulty, low-entropy trajectories for mixed on-policy/off-policy optimization.

## Method

### Overall Architecture
ExGRPO augments GRPO with a three-stage experience management pipeline (collection → bucketing → selection) and mixed-policy optimization. A replay buffer maintains historically successful trajectories, and each training batch mixes newly sampled on-policy data with off-policy experience samples.

### Key Designs

1. **Experience Collection & Partition**:

    - Function: Collect successful trajectories into the buffer and partition them into Easy/Medium/Hard buckets based on the latest per-question accuracy.
    - Mechanism: Accuracy is defined as $\text{Acc}(q^*) = k/K$, with buckets defined as Easy $[75\%, 100\%)$, Medium $(25\%, 75\%]$, and Hard $(0, 25\%]$. A Retired Set is introduced: questions for which all rollouts are correct are removed from the buffer to prevent overfitting to easy problems.
    - Design Motivation: Problems of different difficulty levels provide learning signals of varying strength and therefore require differentiated treatment.

2. **Experience Selection**:

    - Function: A two-step selection process — first sample questions according to difficulty distribution, then select low-entropy trajectories.
    - Mechanism: Question sampling probability is $p \propto \mathcal{N}(\text{Acc}(q^*); \mu=0.5, \sigma=1)$, prioritizing medium-difficulty questions; for each question, the trajectory with the lowest entropy under the current policy is selected: $o^* \leftarrow \arg\min_{o_i} H(o_i; \pi_\theta)$.
    - Design Motivation: Medium-difficulty problems provide the strongest optimization signal (empirically verified); low-entropy trajectories correspond to higher-quality reasoning chains (empirically, high-entropy trajectories tend to reach correct answers through flawed reasoning, and their repeated sampling causes a "snowball effect" that contaminates training).

3. **Mixed-Policy Optimization**:

    - Function: Jointly optimize over on-policy new samples and off-policy historical experiences, with importance weighting to correct for distributional shift.
    - Mechanism: $\mathcal{J}_{\text{ExGRPO}} = (1-\rho)\cdot\mathcal{J}_{\text{on}} + \rho\cdot\mathcal{J}_{\text{exp}}$, where the off-policy term uses importance weights $w_t^*(\theta) = \frac{\pi_\theta(o_t^*|q^*)}{\pi_{\theta_{\text{past}}}(o_t^*|q^*)}$.
    - Design Motivation: Replaying exclusively low-entropy off-policy trajectories may impair exploration; mixing with on-policy data preserves exploratory capacity. Importance weighting ensures unbiased gradient estimation.

### Loss & Training
- Built on Dr.GRPO: length normalization and standard deviation normalization are removed.
- The mixing ratio $\rho$ controls the proportion of experience samples.
- Off-policy samples are incorporated into mixed advantage estimation groups: 1 historical trajectory + $K{-}1$ new rollouts.

## Key Experimental Results

### Main Results
Gains across five backbone models (1.5B–8B) on mathematical and general reasoning:

| Model | Math Avg. Gain | General Reasoning Gain | Notes |
|-------|---------------|----------------------|-------|
| Qwen2.5-Math-1.5B | +3–4 pts | +7–8 pts | Per benchmark |
| Qwen2.5-Math-7B | +3–4 pts | +7–8 pts | AIME/AMC, etc. |
| Llama-3.1-8B | Stable training | Significant gain | On-policy collapses |
| LUFFY model | Consistent gain | Consistent gain | On-policy collapses |

### Ablation Study

| Configuration | Math Metric | Notes |
|---------------|------------|-------|
| Full ExGRPO | Best | Complete framework |
| w/o difficulty bucketing (random sampling) | Degraded | Medium-difficulty prioritization is critical |
| w/o low-entropy selection | Degraded | Low-entropy trajectories are higher quality |
| w/o importance weighting | Degraded | Distributional shift must be corrected |
| w/o Retired Set | Degraded | Overfitting to easy problems |

### Key Findings
- ExGRPO maintains stable training on both weak models (Llama-3.1-8B) and strong models (LUFFY), whereas on-policy GRPO collapses.
- Medium-difficulty problems contribute the most; Hard-bucket problems contribute the least, but should not be discarded entirely as they provide complementary signals.
- High-entropy correct trajectories — where the model arrives at a correct answer through flawed reasoning ("lucky guessing") — are amplified under replay, producing a snowball effect; low-entropy selection effectively mitigates this.
- Experience replay reduces average training overhead rather than increasing it, as reusing historical rollouts decreases the number of new generations required.

## Highlights & Insights
- **Systematic Analysis of Experience Value**: This is the first work to analyze the value of experiences in RLVR along two dimensions — problem difficulty and trajectory entropy — yielding a concise and compelling finding: medium difficulty + low entropy. This insight has broad implications for the RLVR community.
- **Discovery of the Snowball Effect**: High-entropy trajectories, despite being answer-correct, involve flawed reasoning; their repeated sampling contaminates training. The paper identifies a concrete degeneration case in which the model learns to solve math problems via code blocks, directly attributing this to high-entropy experience.
- **Retired Set Design**: Removing fully solved problems from the buffer is a simple yet effective mechanism — it prevents overfitting to easy problems and concentrates resources on medium-difficulty problems that still carry learning value.

## Limitations & Future Work
- The difficulty bucketing thresholds (25%/75%) are fixed; they should be adjusted dynamically as model capability evolves during training.
- Entropy is an imperfect proxy for trajectory quality — high-entropy trajectories may still be valuable in certain contexts, such as when exploring novel solution strategies.
- Validation is limited to mathematical reasoning; the optimal experience characteristics for other domains such as code reasoning may differ.
- The "staleness" of experience — historical trajectories may no longer be optimal after policy updates — remains an open issue.

## Related Work & Insights
- **vs. GRPO**: ExGRPO augments GRPO with experience management, yielding +3.5 pts on math and +7.6 pts on general reasoning.
- **vs. ReMix/RePO**: These methods also employ experience replay but overlook data quality; ExGRPO's bucketing and low-entropy selection provide finer-grained control.
- **vs. LUFFY**: LUFFY mixes expert data with on-policy data, whereas ExGRPO reuses the model's own historical experiences and requires no additional external data.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel perspective on experience value analysis; the snowball effect discovery is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five backbone models, math + general reasoning benchmarks, detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation analysis; the preliminary study is persuasive.
- Value: ⭐⭐⭐⭐⭐ Directly actionable for RLVR training practice; insights are transferable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] How LLMs Learn to Reason: A Complex Network Perspective](how_llms_learn_to_reason_a_complex_network_perspective.md)
- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](../../NeurIPS2025/reinforcement_learning/training_language_models_to_reason_efficiently.md)
- [\[CVPR 2026\] Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning](../../CVPR2026/reinforcement_learning/reinforce_to_learn_elect_to_reason_a_dual_paradigm_for_video_reasoning.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)

<!-- RELATED:END -->
