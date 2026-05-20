---
title: >-
  [Paper Note] SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time
description: >-
  [ACL 2026][Reinforcement Learning][Test-time reinforcement learning] This paper proposes SCRL (Selective-Complementary Reinforcement Learning), a robust test-time reinforcement learning framework that mitigates label noi…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Test-time reinforcement learning"
  - "pseudo-label noise"
  - "negative labels"
  - "consensus reliability"
  - "unsupervised reasoning"
date: 2026-05-08
content_hash: d9a52b4af92e00cd
---

# SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time

**Conference**: ACL 2026
**arXiv**: [2603.19880](https://arxiv.org/abs/2603.19880)  
**Code**: [https://github.com/Jasper-Yan/SCRL](https://github.com/Jasper-Yan/SCRL)  
**Area**: Reinforcement Learning / LLM Reasoning
**Keywords**: Test-time reinforcement learning, pseudo-label noise, negative labels, consensus reliability, unsupervised reasoning

## TL;DR

This paper proposes SCRL (Selective-Complementary Reinforcement Learning), a robust test-time reinforcement learning framework that mitigates label noise amplification through selective positive pseudo-labels (filtering unreliable majorities via strict consensus criteria) and entropy-gated negative pseudo-labels (introducing negative supervision signals into TTRL for the first time to prune erroneous trajectories), achieving up to 10.1 percentage points improvement over TTRL on AIME25.

## Background & Motivation

**Background**: Test-time reinforcement learning (TTRL) enables LLMs to self-improve on unlabeled test streams by deriving pseudo-rewards from majority-vote consensus, and has become a key paradigm for unsupervised reasoning.

**Limitations of Prior Work**: Existing TTRL methods rely entirely on positive pseudo-label strategies — majority voting selects the most frequent answer as the positive label. On difficult problems, however, the answer distribution is highly dispersed and consensus is weak. GRPO's group normalization amplifies noise: when the positive label proportion $f$ is small, the normalized advantage of positive samples $\hat{A}^+ = \sqrt{(1-f)/f}$ becomes large, causing a small number of incorrect positive pseudo-labels to disproportionately affect policy updates and leading to premature convergence to spurious solutions.

**Key Challenge**: On difficult problems, identifying the correct answer is hard, whereas identifying incorrect answers is comparatively easier. Yet existing methods overlook the potential of negative labels — when the correct answer cannot be reliably identified, pruning erroneous trajectories can effectively narrow the search space.

**Goal**: To simultaneously exploit positive and negative signals in TTRL, pruning the search space via negative labels when consensus is unreliable rather than forcing the selection of a positive label.

**Key Insight**: Distinguish between "low-frequency but possibly correct" and "low-frequency and genuinely incorrect" answers using generation uncertainty (token-level entropy): high-frequency + low entropy implies likely correct; low-frequency + high entropy implies likely incorrect.

**Core Idea**: Provide positive supervision only when consensus is sufficiently strong (selectivity); when consensus is insufficient, prune demonstrably incorrect trajectories via negative labels (complementarity). The two mechanisms work in concert with dynamic reward shaping to achieve robust test-time learning.

## Method

### Overall Architecture

SCRL comprises three components: (1) selective positive pseudo-labels — positive labels are assigned only when the answer distribution is sharply concentrated with a clear margin over the runner-up; (2) entropy-gated negative pseudo-labels — negative labels are assigned to answers that are simultaneously low-frequency and highly uncertain; (3) dynamic reward shaping — reward magnitudes are calibrated based on consensus strength, integrating both positive and negative signals. The framework builds upon the GRPO algorithm.

### Key Designs

1. **Selective Positive Pseudo-Labels**:

    - **Function**: Prevent erroneous answers from being reinforced as positive labels under weak consensus.
    - **Mechanism**: Given the answer distribution $\{p_j\}$ over $N$ responses, a positive pseudo-label $y^+ = a_{j^*}$ is declared if and only if the highest proportion $p_{j^*} \geq \tau_{\text{pos}}$ (sufficient support) and $(p_{j^*} - p_{(2)}) > \tau_{\text{marg}}$ (sufficient margin over the runner-up). Otherwise, $y^+ = \varnothing$ and positive supervision is withheld.
    - **Design Motivation**: When the answer distribution is dispersed, the majority-voted answer may only marginally outpace alternatives, making it unreliable. The dual condition of threshold and margin ensures positive signals are provided only at high confidence.

2. **Entropy-Gated Negative Pseudo-Labels**:

    - **Function**: Introduce negative supervision into TTRL for the first time to prune demonstrably incorrect trajectories.
    - **Mechanism**: The mean token-level entropy $\bar{H}_j$ is computed for the trajectories corresponding to each answer $a_j$. An answer $a_j$ is designated as a negative pseudo-label if and only if $p_j < \tau_{\text{neg}}$ (low frequency) and $\bar{H}_j \geq \bar{H}$ (above the global mean uncertainty). The constraint $\bar{H}_j \geq \bar{H}$ ensures that low-frequency but low-uncertainty answers — which may be rare yet correct solutions — are not penalized.
    - **Design Motivation**: Low frequency may indicate either "rare but correct" or "genuinely incorrect." The entropy condition distinguishes the two cases: high uncertainty signals that the model is unconfident on that trajectory, suggesting likely erroneous reasoning.

3. **Dynamic Reward Shaping**:

    - **Function**: Calibrate reward magnitudes for positive and negative signals according to consensus strength.
    - **Mechanism**: Positive reward = answer proportion $p(a_i)$ (stronger consensus yields higher reward); negative reward = $(p(a_i) - \tau_{\text{neg}})$ (lower frequency incurs heavier penalty); entropy penalty = $-\lambda_H(\bar{H}(a_i) - \bar{H})$ (favoring low-uncertainty responses). The three terms are combined with weighted summation.
    - **Design Motivation**: Fixed rewards amplify noise as consensus strength varies. Dynamic scaling ensures reward signal intensity is proportional to consensus reliability.

### Loss & Training

GRPO is used as the base RL algorithm. AdamW optimizer with cosine learning rate scheduling (peak $5 \times 10^{-7}$). Rollout generates 64 (or 32) candidate responses for label estimation, with 32 (or 16) downsampled for training updates. Thresholds: $\tau_{\text{pos}}=0.375$, $\tau_{\text{marg}}=0.125$, $\tau_{\text{neg}}=0.125$. Hardware: 8×A100 80GB GPUs.

## Key Experimental Results

### Main Results

**pass@1 accuracy (%) on Qwen2.5-Math-7B**

| Method | AIME25 | AMC | MATH-500 | Minerva | Avg. |
|--------|--------|-----|---------|---------|------|
| Baseline | 4.6 | 34.0 | 46.5 | 10.1 | 23.6 |
| + TTRL | 16.8 | 65.7 | 85.7 | 14.5* | 41.6 |
| **+ SCRL** | **26.9** | **66.9** | **85.6** | **41.6** | **49.3** |
| Δ vs TTRL | +10.1 | +1.2 | -0.1 | +27.1 | +7.7 |

*Note: TTRL reaches a peak of 14.5% on Minerva before suffering severe performance degradation.*

**pass@1 (%) on Llama-3.1-8B-Instruct**

| Method | AIME24 | AMC | Avg. |
|--------|--------|------|------|
| + TTRL | 10.0 | 32.3 | 21.2 |
| + RESTRAIN | 16.7 | 40.0 | 28.4 |
| **+ SCRL** | **21.9** | 36.1 | **29.0** |

### Ablation Study

| Configuration | AIME25 | AMC | Note |
|--------------|--------|-----|------|
| Full SCRL | 26.9 | 66.9 | Complete model |
| w/o negative labels | 19.4 | 65.3 | Negative labels contribute +7.5 (AIME25) |
| w/o selective positive labels | 21.5 | 66.1 | Selectivity contributes +5.4 |
| w/o entropy gating | 23.1 | 65.8 | Entropy condition contributes +3.8 |
| w/o dynamic reward | 24.2 | 66.0 | Dynamic reward contributes +2.7 |

### Key Findings

- The largest gains appear on the most difficult tasks (AIME25, +10.1), precisely the setting where weak consensus is most severe.
- TTRL exhibits training instability on Minerva (performance rises then sharply collapses), while SCRL maintains stable training dynamics (41.6% vs. 14.5%).
- Negative labels contribute most on difficult tasks (AIME25 +7.5), validating the strategy of "eliminating what is wrong when the correct answer is unknown."
- SCRL's advantage is more pronounced under low rollout budgets — when the budget is constrained, consensus is less reliable, and SCRL's protective mechanisms become more critical.
- Consistent effectiveness across model families (Qwen, Llama) and scales (1B–7B) demonstrates model-agnostic generality.

## Highlights & Insights

- The insight that "consensus may be wrong" directly challenges the fundamental assumption of TTRL; the negative label mechanism offers a natural and elegant solution.
- Entropy gating is the key to distinguishing "rarely correct" from "genuinely incorrect" — frequency alone or uncertainty alone is insufficient; only their intersection yields reliable discrimination.
- Dynamic reward shaping adaptively matches signal strength to consensus reliability, avoiding the noise amplification inherent in fixed rewards.

## Limitations & Future Work

- Threshold parameters ($\tau_{\text{pos}}, \tau_{\text{marg}}, \tau_{\text{neg}}$) are fixed across all experiments and may lack flexibility for certain tasks.
- Validation is limited to mathematical and general reasoning tasks; domains such as code generation are not explored.
- Negative labels contribute marginally on simple tasks, potentially introducing unnecessary complexity.
- Future work may explore adaptive threshold mechanisms and finer-grained uncertainty estimation.

## Related Work & Insights

- **vs. TTRL**: TTRL relies solely on positive labels and amplifies noise under weak consensus; SCRL adds two protective mechanisms — selectivity and negative labels.
- **vs. RESTRAIN**: RESTRAIN penalizes overconfident and low-consistency responses but remains within the positive-label framework; SCRL introduces genuinely negative supervision signals.
- **vs. SPINE**: SPINE restricts updates to high-entropy forking tokens; SCRL operates at the answer level, making the two approaches strongly complementary.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First introduction of negative supervision signals into TTRL; the selective-complementary framework is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-model, multi-scale, multi-task evaluation with comprehensive ablations and label quality analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a core bottleneck in TTRL with substantial performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](../../AAAI2026/reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/reinforcement_learning/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](../../ICLR2026/reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)

</div>

<!-- RELATED:END -->
