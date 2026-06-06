---
title: >-
  [Paper Note] SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time
description: >-
  [ACL 2026][Reinforcement Learning][Test-Time Reinforcement Learning] This paper proposes SCRL (Selective-Complementary Reinforcement Learning), a robust test-time reinforcement learning framework. It mitigates label nois…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Test-Time Reinforcement Learning"
  - "Pseudo-label Noise"
  - "Negative Labels"
  - "Consensus Reliability"
  - "Unsupervised Reasoning"
date: 2026-05-08
content_hash: 71a075d03ccff338
---

# SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time

**Conference**: ACL 2026  
**arXiv**: [2603.19880](https://arxiv.org/abs/2603.19880)  
**Code**: [https://github.com/Jasper-Yan/SCRL](https://github.com/Jasper-Yan/SCRL)  
**Area**: Reinforcement Learning / LLM Reasoning  
**Keywords**: Test-Time Reinforcement Learning, Pseudo-label Noise, Negative Labels, Consensus Reliability, Unsupervised Reasoning

## TL;DR

This paper proposes SCRL (Selective-Complementary Reinforcement Learning), a robust test-time reinforcement learning framework. It mitigates label noise amplification through selective positive pseudo-labels (filtering unreliable majorities with strict consensus criteria) and entropy-gated negative pseudo-labels (introducing negative supervision signals to prune incorrect trajectories for the first time in TTRL), achieving up to a 10.1 percentage point improvement over TTRL on AIME25.

## Background & Motivation

**Background**: Test-Time Reinforcement Learning (TTRL) has become a key paradigm for unsupervised reasoning, allowing LLMs to self-improve on unlabeled test streams by deriving pseudo-rewards via majority voting consensus.

**Limitations of Prior Work**: Existing TTRL methods rely entirely on a positive pseudo-label strategy—majority voting to select the most frequent answer as the positive label. However, on difficult problems, answer distributions are highly dispersed and consensus is weak. The group normalization in GRPO amplifies noise: when the positive label ratio $f$ is small, the normalized advantage of positive samples $\hat{A}^+ = \sqrt{(1-f)/f}$ becomes large. A few incorrect positive pseudo-labels can disproportionately affect policy updates, leading to premature convergence to spurious solutions.

**Key Challenge**: On difficult problems, identifying the correct answer is hard, but identifying incorrect answers is relatively easier. However, existing methods neglect the potential of negative labels—when the correct answer cannot be reliably identified, the search space can be narrowed by pruning incorrect trajectories.

**Goal**: To utilize both positive and negative signals in TTRL, pruning the search space with negative labels when consensus is unreliable rather than forcing a positive label choice.

**Key Insight**: Distinguish between "low-frequency but possibly correct" and "low-frequency and definitely incorrect" answers using generation uncertainty (token-level entropy). High frequency with low entropy indicates "likely correct," while low frequency with high entropy indicates "highly likely incorrect."

**Core Idea**: Provide positive supervision only when the consensus is sufficiently strong (selectivity) and prune definitely incorrect trajectories via negative labels when consensus is insufficient (complementarity). These cooperate through dynamic reward shaping for robust test-time learning.

## Method

### Overall Architecture

SCRL consists of three components: (1) Selective positive pseudo-labels—assigning positive labels only when the answer distribution is sharply concentrated and clearly separated from the runner-up; (2) Entropy-gated negative pseudo-labels—assigning negative labels to answers that satisfy both low frequency and high uncertainty; (3) Dynamic reward shaping—calibrating reward magnitudes based on consensus strength and integrating positive/negative signals. It is built upon the GRPO algorithm.

### Key Designs

1.  **Selective Positive Pseudo-labels**:
    - **Function**: Prevents reinforcing incorrect answers as positive labels during weak consensus.
    - **Mechanism**: Given an answer distribution $\{p_j\}$ of $N$ responses, declare a positive pseudo-label $y^+ = a_{j^*}$ if and only if the top proportion $p_{j^*} \geq \tau_{\text{pos}}$ (sufficient support) and $(p_{j^*} - p_{(2)}) > \tau_{\text{marg}}$ (sufficient margin from the second-best). Otherwise, $y^+ = \varnothing$, foregoing positive supervision.
    - **Design Motivation**: When the distribution is dispersed, the majority answer might be only slightly more frequent than others and thus unreliable. The dual conditions of a strict threshold and margin ensure positive signals are provided only during high confidence.

2.  **Entropy-gated Negative Pseudo-labels**:
    - **Function**: Introduces negative supervision in TTRL for the first time to prune definitely incorrect trajectories.
    - **Mechanism**: Calculate the average token-level entropy $\bar{H}_j$ for the trajectory corresponding to each answer $a_j$. Answer $a_j$ is marked as a negative pseudo-label if and only if $p_j < \tau_{\text{neg}}$ (low frequency) and $\bar{H}_j \geq \bar{H}$ (above the global average uncertainty). The key constraint $\bar{H}_j \geq \bar{H}$ ensures that low-frequency but low-uncertainty answers (potential rare correct solutions) are not penalized.
    - **Design Motivation**: Low frequency can indicate either "rare but correct" or "definitely incorrect." The entropy condition distinguishes these—high uncertainty suggests the model is "unconfident" in the trajectory, making it likely to be incorrect reasoning.

3.  **Dynamic Reward Shaping**:
    - **Function**: Calibrates the reward magnitude of positive and negative signals based on consensus strength.
    - **Mechanism**: Positive reward = answer proportion $p(a_i)$ (stronger consensus yields larger reward); Negative reward = $(p(a_i) - \tau_{\text{neg}})$ (rarer answers receive heavier penalties); Entropy penalty = $-\lambda_H(\bar{H}(a_i) - \bar{H})$ (biases toward low-uncertainty responses). These are combined in a weighted sum.
    - **Design Motivation**: Fixed rewards amplify noise when consensus strength varies. Dynamic scaling ensures signal intensity is proportional to consensus reliability.

### Loss & Training

Uses GRPO as the base RL algorithm. AdamW optimizer is used with a cosine learning rate scheduler (peak $5 \times 10^{-7}$). 64 (or 32) candidate responses are generated per rollout for label estimation, with 32 (or 16) downsampled for training updates. Thresholds are set at $\tau_{\text{pos}}=0.375, \tau_{\text{marg}}=0.125, \tau_{\text{neg}}=0.125$. Training is conducted on 8×A100 80GB GPUs.

## Key Experimental Results

### Main Results

**pass@1 Accuracy (%) on Qwen2.5-Math-7B**

| Method | AIME25 | AMC | MATH-500 | Minerva | Average |
|------|--------|-----|---------|---------|------|
| Baseline | 4.6 | 34.0 | 46.5 | 10.1 | 23.6 |
| + TTRL | 16.8 | 65.7 | 85.7 | 14.5* | 41.6 |
| **+ Ours (SCRL)** | **26.9** | **66.9** | **85.6** | **41.6** | **49.3** |
| Gain vs TTRL | +10.1 | +1.2 | -0.1 | +27.1 | +7.7 |

*\*Note: TTRL performance on Minerva degraded sharply after reaching a peak of 14.5%*

**pass@1 (%) on Llama-3.1-8B-Instruct**

| Method | AIME24 | AMC | Average |
|------|--------|------|------|
| + TTRL | 10.0 | 32.3 | 21.2 |
| + RESTRAIN | 16.7 | 40.0 | 28.4 |
| **+ Ours (SCRL)** | **21.9** | 36.1 | **29.0** |

### Ablation Study

| Configuration | AIME25 | AMC | Description |
|------|--------|-----|------|
| Full SCRL | 26.9 | 66.9 | Complete model |
| w/o Negative Labels | 19.4 | 65.3 | Negative label contribution: +7.5 (AIME25) |
| w/o Selective Positives | 21.5 | 66.1 | Selectivity contribution: +5.4 |
| w/o Entropy Gating | 23.1 | 65.8 | Entropy condition contribution: +3.8 |
| w/o Dynamic Reward | 24.2 | 66.0 | Dynamic reward contribution: +2.7 |

### Key Findings

- The performance gain is largest (+10.1) on the hardest task (AIME25), which is precisely where the weak consensus problem is most severe.
- TTRL exhibits training instability on Minerva (performance peaks then drops sharply), while SCRL maintains stable training dynamics (41.6% vs 14.5%).
- Negative labels contribute most to difficult tasks (+7.5 on AIME25), validating the strategy of "excluding what is wrong when unsure what is right."
- SCRL efficacy is more significant under low rollout budgets—when budgets are constrained, consensus is less reliable, making SCRL's protection mechanisms more critical.
- Consistently effective across model families (Qwen, Llama) and scales (1B-7B), demonstrating model-agnosticism.

## Highlights & Insights

- The insight "consensus can be wrong" directly addresses the fundamental bottleneck of TTRL, and the negative label mechanism provides a natural and elegant solution.
- Entropy gating is critical for distinguishing "rarely correct" from "definitely incorrect"—relying solely on frequency or uncertainty is insufficient; their intersection provides the necessary reliability.
- Dynamic reward shaping ensures signal intensity adaptively matches consensus reliability, preventing the noise amplification inherent in fixed rewards.

## Limitations & Future Work

- Threshold parameters ($\tau_{\text{pos}}, \tau_{\text{marg}}, \tau_{\text{neg}}$) are fixed across all experiments, which may lack flexibility for certain tasks.
- Evaluation is limited to mathematics and general reasoning; code generation and other tasks remain unexplored.
- Negative labels contribute less to simple tasks and may introduce unnecessary complexity.
- Future work could explore adaptive threshold mechanisms and finer-grained uncertainty estimation.

## Related Work & Insights

- **vs TTRL**: TTRL uses only positive labels and amplifies noise during weak consensus; SCRL adds selectivity and negative labels as protective mechanisms.
- **vs RESTRAIN**: RESTRAIN penalizes overconfidence and low-consistency responses but remains within a positive label framework; SCRL introduces genuine negative supervision signals.
- **vs SPINE**: SPINE limits updates to high-entropy forking tokens; SCRL operates at the answer level, offering strong complementarity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introduces negative supervision signals in TTRL for the first time; the selective-complementary framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across models, scales, and tasks, featuring detailed ablation and label quality analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with complete formal derivations.
- Value: ⭐⭐⭐⭐⭐ Successfully addresses a core bottleneck in TTRL with significant performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning](beyond_majority_voting_towards_fine-grained_and_more_reliable_reward_signal_for_.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](../../AAAI2026/reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/reinforcement_learning/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)

</div>

<!-- RELATED:END -->
