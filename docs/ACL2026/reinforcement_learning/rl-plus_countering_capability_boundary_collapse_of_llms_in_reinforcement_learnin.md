---
title: >-
  [Paper Note] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization
description: >-
  [ACL 2026][Reinforcement Learning][Capability Boundary Collapse] RL-PLUS proposes a hybrid-policy optimization approach that addresses external data distribution mismatch via Multiple Importance Sampling (MIS) and guides…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Capability Boundary Collapse"
  - "Hybrid-Policy Optimization"
  - "Multiple Importance Sampling"
  - "Exploration-Based Advantage Function"
  - "RLVR"
date: 2026-05-08
content_hash: f227cc456bd2379d
---

# RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization

**Conference**: ACL 2026
**arXiv**: [2508.00222](https://arxiv.org/abs/2508.00222)
**Code**: [GitHub](https://github.com/YihongDong/RL-PLUS)
**Area**: LLM Reasoning / Reinforcement Learning
**Keywords**: Capability Boundary Collapse, Hybrid-Policy Optimization, Multiple Importance Sampling, Exploration-Based Advantage Function, RLVR

## TL;DR

RL-PLUS proposes a hybrid-policy optimization approach that addresses external data distribution mismatch via Multiple Importance Sampling (MIS) and guides models to learn low-probability but correct reasoning paths via an Exploration-Based Advantage Function (EAF), successfully overcoming the capability boundary collapse induced by RLVR and achieving SOTA on six mathematical reasoning benchmarks (average 53.4), with consistent cross-model improvements of up to 69.2%.

## Background & Motivation

**Background**: Reinforcement learning with verifiable rewards (RLVR, e.g., GRPO/DAPO) has significantly enhanced the complex reasoning capabilities of LLMs by rewarding correct answers to optimize long-chain reasoning.

**Limitations of Prior Work**: RLVR is inherently on-policy — models can only learn from trajectories they generate themselves. This leads to "capability boundary collapse": although pass@1 improves, pass@128 drops below the base model. That is, RLVR makes models better at selecting known correct paths (inward exploitation) but narrows the range of solvable problems (capability boundary). Meanwhile, policy entropy collapses sharply, rendering models excessively deterministic.

**Key Challenge**: Under the vast action space and sparse rewards of LLMs, RLVR cannot effectively guide models to explore new reasoning paths (outward exploration). SFT can introduce external knowledge but is poor at internalizing reasoning principles. Naively combining the two (e.g., GRPO w/ SFT Loss) leads to performance degradation.

**Goal**: To design an RLVR method that effectively integrates external data and internal exploration to break through the capability ceiling of the base model.

**Key Insight**: Inspired by Confucius — "Learning without thought is labor lost; thought without learning is perilous" — current RLVR is "thought without learning" (relying solely on self-generated reasoning). The key challenge is thus: (1) how to handle the distribution mismatch of external data, and (2) how to efficiently extract new knowledge from external data.

**Core Idea**: Perform reinforcement learning with a hybrid policy (internal on-policy trajectories + external data), stabilizing off-policy updates via MIS and amplifying learning signals for low-probability correct paths via EAF.

## Method

### Overall Architecture

The training objective of RL-PLUS integrates two components: $\mathcal{J}_{\text{RL-PLUS}} = \underbrace{\mathbb{E}_{(o_i, A_i) \sim \mathcal{D}_o}[r_{i,t}(\theta) A_i]}_{\text{Inward Exploitation}} + \underbrace{\mathbb{E}_{(e_i, A_{i,t}^c) \sim \mathcal{D}_e}[r_{i,t}^m(\theta) A_{i,t}^c]}_{\text{Outward Exploration}}$. The first term is standard GRPO (optimizing existing reasoning capabilities); the second is the core innovation (learning new knowledge from external data), using MIS to correct distribution shift and EAF to focus on low-probability correct paths.

### Key Designs

1. **Multiple Importance Sampling (MIS)**:

    - Function: Addresses the distribution mismatch between external data and the current policy.
    - Mechanism: Rather than directly estimating the unknown external policy $\pi_\omega$, sampling is treated as originating from a mixture policy $\pi_\omega + \pi_{\theta_{old}}$. The importance weight is defined as $r_{i,t}^m(\theta) = \frac{2\pi_\theta(e_{i,t}|q, e_{i,<t})}{\pi_\omega(e_{i,t}|q, e_{i,<t}) + \pi_{\theta_{old}}(e_{i,t}|q, e_{i,<t})}$. For the unknown $\pi_\omega$, the Bayesian optimal estimator $\hat{\pi}_\omega^* = \frac{1}{2}\pi_{\theta_{old}} + \frac{1}{2}\mathcal{U}$ (equal-weight mixture of the old policy and a uniform distribution) is adopted, minimizing L2 risk under maximum uncertainty.
    - Design Motivation: Standard on-policy IS introduces systematic bias for external data (Lemma A.5), while direct off-policy IS suffers from excessive variance (Lemma A.7). MIS provides a low-bias, low-variance compromise — even when $\pi_\omega$ differs greatly from $\pi_\theta$, the presence of $\pi_{\theta_{old}}$ prevents weight explosion.

2. **Exploration-Based Advantage Function (EAF)**:

    - Function: Guides the model to attend to correct but low-probability reasoning paths, i.e., "new knowledge."
    - Mechanism: $A_{i,t}^c = \frac{R_i - \text{mean}(R)}{\text{std}(R)} \cdot C_{i,t}$, where the weight $C_{i,t} = (1 - \text{detach}(\pi_\theta(e_{i,t}|q, e_{i,<t})))^\gamma$. Inspired by Focal Loss — when the model assigns low probability to a correct token (indicating poor exploration), $C_{i,t}$ increases, amplifying the advantage signal for that path and compelling the model to attend to neglected reasoning strategies.
    - Design Motivation: Models naturally favor high-probability tokens (existing knowledge), while new knowledge is often hidden in low-probability paths. Merely stabilizing the introduction of external data is insufficient; explicit guidance is needed for models to "see" and learn from these new paths.

3. **Removal of Clipping**:

    - Function: Permits larger optimization steps for high-value external data.
    - Mechanism: Standard GRPO applies $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$ to constrain update magnitude. RL-PLUS removes clipping for the external data term, since low-probability events (new knowledge) are precisely where large optimization steps are needed — clipping would suppress learning from these highly informative paths.
    - Design Motivation: Outward exploration requires more "aggressive" policy updates; clipping conflicts with the exploration objective.

### Loss & Training

Training is conducted on Qwen2.5-Math-7B using NuminaMath-1.5 as the training dataset (including external data). The KL regularization term is omitted, as long-chain CoT reasoning requires the policy to deviate substantially from the initial policy.

## Key Experimental Results

### Main Results

**Six Mathematical Reasoning Benchmarks (Qwen2.5-Math-7B)**

| Method | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | Avg. |
|--------|--------|--------|-----|----------|---------|----------|------|
| Base Model | 11.5 | 4.9 | 31.3 | 43.6 | 7.4 | 15.6 | 19.0 |
| GRPO | 25.1 | 15.3 | 62.0 | 84.4 | 39.3 | 46.8 | 45.5 |
| LUFFY | 29.4 | 23.1 | 65.6 | 87.6 | 37.5 | 57.2 | 50.1 |
| SFT+GRPO | 25.8 | 23.1 | 62.7 | 87.2 | 39.7 | 50.4 | 48.2 |
| **RL-PLUS** | **33.4** | **25.9** | **68.1** | **90.2** | **43.8** | **58.8** | **53.4** |

**OOD Tasks (Coding + Scientific QA)**

| Method | HumanEval | LeetCode | LiveCodeBench | ARC-c | GPQA | MMLU-Pro | Avg. |
|--------|-----------|----------|---------------|-------|------|----------|------|
| GRPO | 63.4 | 21.1 | 15.3 | 81.7 | 40.4 | 47.5 | 44.9 |
| SFT+GRPO | 59.8 | 8.3 | 9.7 | 72.4 | 24.2 | 37.7 | 35.4 |
| **RL-PLUS** | **68.3** | **27.8** | **19.2** | **82.3** | **40.4** | **54.7** | **48.8** |

### Ablation Study

| Method | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | Avg. |
|--------|--------|--------|-----|----------|---------|----------|------|
| RL-PLUS (Full) | **33.4** | **25.9** | **68.1** | **90.2** | **43.8** | **58.8** | **53.4** |
| - EAF | 28.3 | 24.1 | 67.8 | 88.8 | 40.4 | 56.0 | 50.9 |
| - MIS | 25.1 | 15.3 | 62.0 | 84.4 | 39.3 | 46.8 | 45.5 |

### Key Findings

- MIS is the more critical component (removing it causes a 7.9-point drop vs. 2.5 points for EAF), indicating that stably incorporating external data is foundational.
- RL-PLUS achieves an absolute improvement of 11.9 points on LLaMA-3.1-8B (where GRPO is nearly ineffective), demonstrating strong generalizability.
- Pass@k curves show RL-PLUS consistently outperforms the base model at all values of $k$, confirming a genuine breakthrough of the capability boundary.
- GRPO w/ SFT Loss underperforms standalone GRPO (40.1 vs. 45.5), demonstrating that naively integrating external knowledge is non-trivial.
- Training dynamics show that RL-PLUS policy entropy does not collapse to zero, preserving exploration capacity.

## Highlights & Insights

- The formalization and experimental validation of "capability boundary collapse" is highly convincing — pass@k curves serve as intuitive and compelling evidence.
- The theoretical analysis is rigorous: variance boundedness proofs for MIS and the Bayesian optimal policy estimation are supported by complete mathematical derivations.
- The work addresses a practically challenging problem: how to effectively leverage external data during RL training without incurring distribution mismatch or training collapse.
- Strong OOD performance suggests that RL-PLUS learns general reasoning capabilities rather than task-specific heuristics.

## Limitations & Future Work

- The quality and coverage of the external data source (NuminaMath-1.5) affects performance; strategies for external data selection are not deeply explored.
- The equal-weight assumption in the Bayesian policy estimator (50% $\pi_{\theta_{old}}$, 50% $\mathcal{U}$) may not be optimal.
- Validation is limited to mathematical reasoning tasks; effectiveness on other reasoning tasks such as code generation warrants further investigation.
- Sensitivity analysis of the $\gamma$ hyperparameter is insufficient.

## Related Work & Insights

- **vs. LUFFY**: LUFFY selectively imitates high-quality external trajectories but handles mixture policies in a coarser manner; RL-PLUS provides a theoretically superior distribution correction via MIS.
- **vs. ReLIFT**: ReLIFT alternates between RL and online fine-tuning; RL-PLUS performs both simultaneously within a unified framework.
- **vs. GRPO w/ SFT Loss**: Directly adding SFT loss is detrimental, underscoring that the manner in which external data is integrated is critical.
- **Inspiration**: The application of the Focal Loss idea to RL advantage functions represents an interesting cross-domain transfer.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The problem formulation, MIS scheme, and EAF design are all original and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, six OOD tasks, four base models, pass@k analysis, and complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations, comprehensive experiments, and clear paper structure.
- Value: ⭐⭐⭐⭐⭐ Addresses a core limitation of RLVR with strong generality; provides important reference value for LLM reasoning training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)

</div>

<!-- RELATED:END -->
