---
title: >-
  [Paper Note] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning
description: >-
  [ACL 2026][Reinforcement Learning][SFT and RL integration] This paper proposes DYPO (Dynamic Policy Optimization), which routes samples to different optimization paths based on dynamic difficulty grading—using multi-teac…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "SFT and RL integration"
  - "Bias-variance tradeoff"
  - "Dynamic difficulty grading"
  - "Multi-teacher distillation"
  - "Gradient variance reduction"
date: 2026-05-08
content_hash: a3ada99830c93ba5
---

# Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.08926](https://arxiv.org/abs/2604.08926)  
**Code**: [GitHub](https://github.com/Tocci-Zhu/DYPO)  
**Area**: Reinforcement Learning  
**Keywords**: SFT and RL integration, Bias-variance tradeoff, Dynamic difficulty grading, Multi-teacher distillation, Gradient variance reduction

## TL;DR
This paper proposes DYPO (Dynamic Policy Optimization), which routes samples to different optimization paths based on dynamic difficulty grading—using multi-teacher distillation for Hard samples to reduce SFT bias and Group Alignment Loss for Mid samples to reduce RL variance. It achieves an average improvement of 4.8% on mathematical reasoning benchmarks and a 13.3% gain on OOD tasks.

## Background & Motivation

**Background**: LLM post-training primarily follows two routes: SFT and RL. SFT is stable (low variance) but limited by fitting bias from static data; RL is exploratory (low bias) but suffers from high gradient variance due to sampling randomness. The industry typically adopts a sequential "SFT → RL" pipeline.

**Limitations of Prior Work**: (1) SFT bias in the sequential pipeline misleads subsequent RL exploration (bias propagation); (2) existing integration strategies (e.g., SuperRL, CHORD) only mix SFT and RL through simple loss weighting, ignoring the fundamental statistical conflict between their gradient signals; (3) a uniform optimization strategy is applied to samples of varying difficulty—easy samples provide marginal signals, hard samples have extremely sparse rewards, and only medium-difficulty samples are most informative.

**Key Challenge**: The statistical conflict between the high-bias, low-variance gradients of SFT and the low-bias, high-variance gradients of RL. Simple weighting cannot resolve this multi-dimensional mismatch.

**Goal**: To propose a structured solution that simultaneously mitigates SFT fitting bias and RL gradient variance, achieving efficient and stable unified post-training.

**Key Insight**: From the theoretical perspective of bias-variance decomposition, it is identified that samples of different difficulty levels require different optimization strategies—Hard samples require knowledge injection (SFT direction), Mid samples require reinforcement learning (RL direction), and Easy samples can be skipped.

**Core Idea**: Dynamically categorize samples into Easy, Hard, and Mid tiers based on group rollout results, routing them to skipped, multi-teacher distillation SFT, and GAL-enhanced RL optimization paths, respectively.

## Method

### Overall Architecture
For each prompt, a set of $k$ trajectories is generated and categorized into Easy (all correct), Hard (all incorrect), and Mid (partially correct) based on accuracy. Easy samples are skipped; Hard samples use multi-teacher distillation to provide low-bias supervision; Mid samples employ a hybrid objective of GRPO + GAL to achieve low-variance RL. Total loss: $\mathcal{L}_{DYPO} = \mathbb{I}_\mathcal{H} \cdot \gamma \mathcal{L}_{SFT} + \mathbb{I}_\mathcal{M} \cdot (\alpha \mathcal{L}_{GRPO} + (1-\alpha) \mathcal{L}_{GAL})$.

### Key Designs

1. **Dynamic Difficulty Grading**:

    - **Function**: Routes samples to the most appropriate optimization path based on rollout results.
    - **Mechanism**: Samples $k$ trajectories for each query and evaluates them with binary rewards $R(\tau_i) \in \{0,1\}$. All correct → Easy (skip), all incorrect → Hard (SFT distillation), mixed → Mid (RL optimization). This ensures Easy samples do not contribute gradients (avoiding saturation), Hard samples avoid ineffective RL exploration, and Mid samples receive the most informative RL signals.
    - **Design Motivation**: A uniform optimization strategy cannot handle the diversity of sample difficulty. Gradients for simple samples approach zero (wasting computation), while reward signals for difficult samples are too sparse, leading to RL gradient variance explosion.

2. **Multi-Teacher Distillation**:

    - **Function**: Provides low-bias supervisory signals for Hard samples, replacing the biased guidance of a single teacher.
    - **Mechanism**: Maintains a collection of reasoning trajectories from $m$ teacher models and randomly samples a trajectory from one teacher as the SFT target for Hard queries. Theoretically, a single teacher introduces a bias of $\|\mathbf{b}_{sys} + \mathbf{b}_i\|$, while an ensemble of $m$ teachers reduces specific bias to $\sigma_{bias}^2/m$ (assuming teacher bias directions are uncorrelated).
    - **Design Motivation**: Fitting bias from single-teacher SFT is a root cause of restricted RL exploration. Using two teachers, such as DeepSeek-R1 and Qwen3-235B, significantly reduces specific bias.

3. **Group Alignment Loss (GAL)**:

    - **Function**: Serves as a variance control supplement for GRPO, stabilizing RL gradients through contrastive learning.
    - **Mechanism**: Constructs positive and negative pairs from successful and failed trajectories of Mid samples, using a DPO-style contrastive loss $\mathcal{L}_{GAL} = -\log\sigma(\beta_{GAL} \cdot d(\tau_s, \tau_f))$ to pull successful paths closer and push failed paths away. A key difference from standard DPO is that GAL uses on-policy rollouts rather than static preference data. The gradient weight $w_d = 1 - \sigma(\beta_{GAL} d)$ is strictly bounded in $(0,1)$, whereas $\hat{A}_i$ in GRPO is unbounded.
    - **Design Motivation**: GRPO gradient variance $\approx \Sigma_s/k$ is limited by the group size $k$; the variance of GAL naturally decays to 0 as the model learns to distinguish positive and negative samples ($\sigma \to 1$). After mixing, $Var(g_{mix}) < Var(g_{GRPO})$.

### Loss & Training
The total loss is a weighted combination based on graded routing. Hard samples: standard NLL loss for multi-teacher distillation (weight $\gamma$). Mid samples: $\alpha \cdot \mathcal{L}_{GRPO} + (1-\alpha) \cdot \mathcal{L}_{GAL}$. Each prompt samples 8 trajectories, with a maximum response length of 8192, a learning rate of $1 \times 10^{-6}$, and training conducted on 2×8 A800 GPUs using the verl framework.

## Key Experimental Results

### Main Results (Qwen2.5-Math-7B)

| Benchmark | DYPO | SFT→RL | CHORD | SRFT | Gain (vs strongest) |
|-----------|------|--------|-------|------|-------------|
| AIME 24 | 36.0 | 25.8 | 31.2 | 30.7 | +4.8 |
| AIME 25 | 28.7 | 23.1 | 24.4 | 26.0 | +2.7 |
| AMC | 67.0 | 62.7 | 66.8 | 69.8 | -2.8 |
| MATH-500 | 89.2 | 87.2 | 89.4 | 88.4 | -0.2 |
| ID Avg | 52.5 | 47.7 | 50.2 | 50.9 | +1.6 |
| ARC-c (OOD) | 81.8 | 72.4 | 81.1 | 81.6 | +0.2 |
| GPQA-D (OOD) | 41.4 | 24.2 | 40.4 | 40.4 | +1.0 |
| OOD Avg | 61.6 | 48.3 | 60.8 | 61.0 | +0.6 |

### Ablation Study

| Configuration | ID Avg | OOD Avg | Description |
|------|--------|---------|------|
| SFT only | 44.1 | 50.0 | High bias, low variance |
| RL only | 45.2 | 61.4 | Low bias, high variance |
| SFT→RL | 47.7 | 48.3 | Sequential pipeline |
| DYPO | 52.5 | 61.6 | Dynamic grading + dual mitigation |

### Key Findings
- DYPO scores +5.3 points higher than the strongest baseline (SRFT) on AIME 24 and +10.9 higher than pure RL, indicating that dynamic grading and GAL effectively mitigate RL instability.
- Outstanding OOD generalization: Outperforms the SFT baseline by +16.7% on GPQA-D, proving that DYPO improves reasoning strategies rather than memorizing templates.
- Equally effective on Qwen3-4B-Base (ID Avg increase of +18.8% vs SFT), demonstrating cross-model generalizability.

## Highlights & Insights
- **Bias-Variance Decomposition Perspective**: Analyzing the fundamental conflict of SFT-RL integration using statistical learning theory is more profound than "simple weighting" engineering solutions. This analytical framework is transferable to other multi-objective optimization scenarios.
- **Adaptive Variance Decay of GAL**: As the model learns to distinguish between positive and negative samples, the gradient variance of GAL naturally decays to 0—making it an "adaptive regularizer" rather than an auxiliary loss with fixed weighting.
- **Simplicity of Dynamic Difficulty Grading**: Classification into Easy/Hard/Mid is achieved using only group rollout accuracy, requiring no additional classifiers or reward models, which makes implementation costs extremely low.

## Limitations & Future Work
- Relies on binary rewards (correct/incorrect), which are not directly applicable to partially correct open-ended generation tasks.
- Multi-teacher distillation requires reasoning trajectories from multiple strong teacher models, increasing data preparation costs.
- The boundaries for Easy/Hard/Mid are hard partitions (all correct/all incorrect), which may lose information from boundary samples.
- Verified only on mathematical reasoning; effectiveness on other reasoning tasks like NLP understanding and coding remains to be confirmed.

## Related Work & Insights
- **vs SuperRL**: SuperRL performs binary switching between SFT and RL, while DYPO performs instance-level routing and optimizes bias and variance separately.
- **vs CHORD**: CHORD mixes objectives via dynamic soft weights but remains a uniform optimization; DYPO assigns completely different loss functions based on difficulty.
- **vs LUFFY**: LUFFY also integrates SFT+RL but uses a fixed mixing ratio; DYPO's dynamic grading allows the mixing strategy to adapt per sample.

## Rating
- Novelty: ⭐⭐⭐⭐ The bias-variance analysis perspective is novel and the GAL design is ingenious, though the idea of dynamic difficulty grading itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 ID and 2 OOD benchmarks, two base models, and multiple baselines; however, non-mathematical reasoning tasks are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical analysis is rigorous and clear, with a complete derivation chain for bias-variance decomposition and comprehensive experimental comparisons.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[ACL 2026\] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[ICLR 2026\] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](../../ICLR2026/reinforcement_learning/fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)

</div>

<!-- RELATED:END -->
