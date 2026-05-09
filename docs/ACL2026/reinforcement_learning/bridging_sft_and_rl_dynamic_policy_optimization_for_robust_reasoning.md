---
title: >-
  [Paper Note] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning
description: >-
  [ACL 2026][Reinforcement Learning][SFT-RL integration] This paper proposes DYPO (Dynamic Policy Optimization), which dynamically routes samples to different optimization paths based on difficulty grading — Hard samples use multi-teacher distillation to reduce SFT bias, while Mid samples use Group Alignment Loss to reduce RL variance. DYPO achieves an average improvement of 4.8% on mathematical reasoning benchmarks and 13.3% on OOD tasks.
tags:
  - ACL 2026
  - Reinforcement Learning
  - SFT-RL integration
  - bias-variance tradeoff
  - dynamic difficulty grading
  - multi-teacher distillation
  - gradient variance reduction
date: 2026-05-08
content_hash: 683c40a5f1f4d788
---

# Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning

**Conference**: ACL 2026
**arXiv**: [2604.08926](https://arxiv.org/abs/2604.08926)
**Code**: [GitHub](https://github.com/Tocci-Zhu/DYPO)
**Area**: Reinforcement Learning
**Keywords**: SFT-RL integration, bias-variance tradeoff, dynamic difficulty grading, multi-teacher distillation, gradient variance reduction

## TL;DR
This paper proposes DYPO (Dynamic Policy Optimization), which dynamically routes samples to different optimization paths based on difficulty grading — Hard samples use multi-teacher distillation to reduce SFT bias, while Mid samples use Group Alignment Loss to reduce RL variance. DYPO achieves an average improvement of 4.8% on mathematical reasoning benchmarks and 13.3% on OOD tasks.

## Background & Motivation

**Background**: LLM post-training is primarily divided into two lines: SFT and RL. SFT is stable (low variance) but suffers from fitting bias due to static data; RL offers strong exploration (low bias) but incurs high gradient variance from sampling randomness. The standard practice is a sequential "SFT → RL" pipeline.

**Limitations of Prior Work**: (1) In sequential pipelines, SFT bias misleads subsequent RL exploration (bias propagation); (2) existing hybrid strategies (e.g., SuperRL, CHORD) mix SFT and RL via simple loss weighting, ignoring the fundamental statistical conflict between their gradient signals; (3) a uniform optimization strategy is applied across all difficulty levels — easy samples provide marginal signal, hard samples yield extremely sparse rewards, and only medium-difficulty samples carry the most informative signal.

**Key Challenge**: The statistical conflict between the high-bias, low-variance nature of SFT gradients and the low-bias, high-variance nature of RL gradients cannot be resolved by simple weighting, as it constitutes a multi-dimensional mismatch.

**Goal**: To propose a structured solution that simultaneously alleviates SFT's fitting bias and RL's gradient variance, enabling efficient and stable unified post-training.

**Key Insight**: From the theoretical perspective of bias-variance decomposition, different difficulty levels require different optimization strategies — Hard samples need knowledge injection (SFT direction), Mid samples need reinforcement learning (RL direction), and Easy samples can be skipped.

**Core Idea**: Based on group rollout outcomes, samples are dynamically categorized into Easy/Hard/Mid tiers and routed to the corresponding optimization paths: skip / multi-teacher distillation SFT / GAL-augmented RL.

## Method

### Overall Architecture
For each prompt, a group of $k$ trajectories is generated and categorized into Easy (all correct), Hard (all incorrect), and Mid (partially correct) tiers based on accuracy. Easy samples are skipped; Hard samples receive low-bias supervision via multi-teacher distillation; Mid samples are optimized with a hybrid objective combining GRPO and GAL for low-variance RL. The total loss is:
$$\mathcal{L}_{DYPO} = \mathbb{I}_\mathcal{H} \cdot \gamma \mathcal{L}_{SFT} + \mathbb{I}_\mathcal{M} \cdot (\alpha \mathcal{L}_{GRPO} + (1-\alpha) \mathcal{L}_{GAL})$$

### Key Designs

1. **Dynamic Difficulty Grading**:

    - **Function**: Routes samples to the most suitable optimization path based on rollout outcomes.
    - **Mechanism**: For each query, $k$ trajectories are sampled and evaluated with binary reward $R(\tau_i) \in \{0,1\}$. All correct → Easy (skip); all incorrect → Hard (SFT distillation); mixed → Mid (RL optimization). Easy samples contribute no gradient (avoiding saturation), Hard samples avoid futile RL exploration, and Mid samples receive the most informative RL signal.
    - **Design Motivation**: A uniform optimization strategy cannot handle the diversity of sample difficulty. Gradients from easy samples approach zero (wasting computation), while reward signals from hard samples are too sparse, causing RL gradient variance to explode.

2. **Multi-Teacher Distillation**:

    - **Function**: Provides low-bias supervision for Hard samples, replacing the biased guidance of a single teacher.
    - **Mechanism**: A set of reasoning trajectories from $m$ teacher models is maintained; for each Hard query, a trajectory from a randomly selected teacher is used as the SFT target. Theoretically, a single teacher introduces a bias of $\|\mathbf{b}_{sys} + \mathbf{b}_i\|$, while an ensemble of $m$ teachers reduces the idiosyncratic bias to $\sigma_{bias}^2/m$ (assuming teacher bias directions are uncorrelated).
    - **Design Motivation**: The fitting bias of single-teacher SFT is a root cause of constrained RL exploration. Using DeepSeek-R1 and Qwen3-235B as two teachers substantially reduces idiosyncratic bias.

3. **Group Alignment Loss (GAL)**:

    - **Function**: Serves as a variance-control complement to GRPO by stabilizing RL gradients via contrastive learning.
    - **Mechanism**: Successful and failed trajectories from Mid samples are organized into positive-negative pairs, and a DPO-style contrastive loss $\mathcal{L}_{GAL} = -\log\sigma(\beta_{GAL} \cdot d(\tau_s, \tau_f))$ is applied to pull successful paths closer and push failed paths apart. A key distinction from standard DPO is that GAL uses on-policy rollouts rather than static preference data. The gradient weight $w_d = 1 - \sigma(\beta_{GAL} d)$ is strictly bounded within $(0,1)$, whereas GRPO's $\hat{A}_i$ is unbounded.
    - **Design Motivation**: GRPO gradient variance $\approx \Sigma_s/k$ is limited by group size $k$; GAL's variance naturally decays to zero as the model learns to distinguish positive from negative samples ($\sigma \to 1$). Upon mixing, $Var(g_{mix}) < Var(g_{GRPO})$.

### Loss & Training
The total loss is a weighted combination routed by difficulty tier. Hard samples: standard NLL loss from multi-teacher distillation (weight $\gamma$). Mid samples: $\alpha \cdot \mathcal{L}_{GRPO} + (1-\alpha) \cdot \mathcal{L}_{GAL}$. Eight trajectories are sampled per prompt, with a maximum response length of 8192, learning rate $1 \times 10^{-6}$, trained on 2×8 A800 GPUs using the verl framework.

## Key Experimental Results

### Main Results (Qwen2.5-Math-7B)

| Benchmark | DYPO | SFT→RL | CHORD | SRFT | Gain (vs. best) |
|-----------|------|--------|-------|------|-----------------|
| AIME 24 | 36.0 | 25.8 | 31.2 | 30.7 | +4.8 |
| AIME 25 | 28.7 | 23.1 | 24.4 | 26.0 | +2.7 |
| AMC | 67.0 | 62.7 | 66.8 | 69.8 | -2.8 |
| MATH-500 | 89.2 | 87.2 | 89.4 | 88.4 | -0.2 |
| ID Avg | 52.5 | 47.7 | 50.2 | 50.9 | +1.6 |
| ARC-c (OOD) | 81.8 | 72.4 | 81.1 | 81.6 | +0.2 |
| GPQA-D (OOD) | 41.4 | 24.2 | 40.4 | 40.4 | +1.0 |
| OOD Avg | 61.6 | 48.3 | 60.8 | 61.0 | +0.6 |

### Ablation Study

| Configuration | ID Avg | OOD Avg | Notes |
|---------------|--------|---------|-------|
| SFT only | 44.1 | 50.0 | High bias, low variance |
| RL only | 45.2 | 61.4 | Low bias, high variance |
| SFT→RL | 47.7 | 48.3 | Sequential pipeline |
| DYPO | 52.5 | 61.6 | Dynamic grading + dual mitigation |

### Key Findings
- DYPO outperforms the strongest baseline (SRFT) by +5.3 points on AIME 24 and surpasses pure RL by +10.9 points, demonstrating that dynamic grading and GAL effectively mitigate RL instability.
- OOD generalization is particularly strong: DYPO exceeds the SFT baseline by +16.7% on GPQA-D, indicating that the model improves reasoning strategies rather than memorizing templates.
- DYPO is equally effective on Qwen3-4B-Base (ID Avg improves +18.8% vs. SFT), confirming cross-model generalizability.

## Highlights & Insights
- **Bias-variance decomposition perspective**: Analyzing the fundamental conflict in SFT-RL integration through the lens of statistical learning theory offers deeper insight than purely engineering-driven "simple weighting" solutions. This analytical framework is transferable to other multi-objective optimization scenarios.
- **Adaptive variance decay of GAL**: As the model learns to distinguish positive from negative samples, GAL's gradient variance naturally decays to zero — making it an "adaptive regularizer" rather than an auxiliary loss with a fixed weight.
- **Simplicity of dynamic difficulty grading**: The Easy/Hard/Mid classification is accomplished solely via group rollout accuracy, requiring no additional classifier or reward model, resulting in extremely low implementation overhead.

## Limitations & Future Work
- The method relies on binary rewards (correct/incorrect) and is not directly applicable to open-ended generation tasks that admit partially correct responses.
- Multi-teacher distillation requires reasoning trajectories from multiple strong teacher models, increasing data preparation cost.
- The Easy/Hard/Mid boundaries are hard thresholds (all correct / all incorrect), which may discard informative signal from borderline samples.
- Experiments are conducted exclusively on mathematical reasoning; effectiveness on other reasoning tasks such as NLP comprehension and code generation remains to be verified.

## Related Work & Insights
- **vs. SuperRL**: SuperRL switches between SFT and RL at the dataset level, whereas DYPO performs instance-level routing and optimizes bias and variance separately.
- **vs. CHORD**: CHORD mixes objectives via dynamic soft weighting but still applies a unified optimization strategy; DYPO dispatches samples to entirely different loss functions based on difficulty.
- **vs. LUFFY**: LUFFY also integrates SFT and RL but with a fixed mixing ratio; DYPO's dynamic grading makes the mixing strategy sample-adaptive.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The bias-variance analysis perspective is novel and the GAL design is elegant, though the core idea of dynamic difficulty grading is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 ID and 2 OOD benchmarks, two base models, and multiple baselines; however, non-mathematical reasoning tasks are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical analysis is rigorous and clearly presented; the bias-variance decomposition derivation chain is complete and the experimental comparisons are comprehensive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)
- [\[ICLR 2026\] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](../../ICLR2026/reinforcement_learning/fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](../../NeurIPS2025/reinforcement_learning/opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](../../ICLR2026/reinforcement_learning/rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)

</div>

<!-- RELATED:END -->
