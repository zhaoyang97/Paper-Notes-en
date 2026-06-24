---
title: >-
  [Paper Note] Jackpot: Align Actor-Policy Distribution for Scalable and Stable RL for LLM
description: >-
  [ICLR 2026][Reinforcement Learning][off-policy RL] Jackpot utilizes "Optimal Budgeted Rejection Sampling (OBRS)" to directly align the actor (rollout) distribution with the policy (training) distribution. Combined with Top-K probability estimation and a stabilized Jackpot-PPO loss, it enables stable convergence for LLM reinforcement learning under extreme off-policy settings, including large-batch, asynchronous, and even "disparate model" rollout/training configurations.
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "off-policy RL"
  - "distribution mismatch"
  - "rejection sampling"
  - "OBRS"
  - "importance sampling"
  - "PPO"
  - "large-batch training"
date: 2026-05-08
content_hash: 8c273cd30c047b22
---

# Jackpot: Align Actor-Policy Distribution for Scalable and Stable RL for LLM

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5RATVAQGPx](https://openreview.net/forum?id=5RATVAQGPx)  
**Code**: [https://infini-ai-lab.github.io/jpt_website/](https://infini-ai-lab.github.io/jpt_website/)  
**Area**: Reinforcement Learning / LLM Post-training  
**Keywords**: off-policy RL, distribution mismatch, rejection sampling, OBRS, importance sampling, PPO, large-batch training  

## TL;DR
Jackpot utilizes "Optimal Budgeted Rejection Sampling (OBRS)" to directly align the actor (rollout) distribution with the policy (training) distribution. Combined with Top-K probability estimation and a stabilized Jackpot-PPO loss, it enables stable convergence for LLM reinforcement learning under extreme off-policy settings, including large-batch, asynchronous, and even "disparate model" rollout/training configurations.

## Background & Motivation
**Background**: RL has become a critical paradigm for LLM post-training (alignment, reasoning, coding), but training is extremely expensive—over 70% of the time is spent on rollouts (sampling trajectories to calculate rewards). Allowing the actor and policy distributions to diverge could unlock large-batch training, asynchronous training, and the use of lightweight (quantized/sparse/distilled) models for rollouts, significantly increasing throughput.

**Limitations of Prior Work**: When the actor and policy distributions diverge, existing importance sampling (IS) correction methods struggle. Truncated Importance Sampling (TIS) uses $\min\!\big(\frac{p_{\text{ref}}(x)}{p_{\text{inf}}(x)}, C\big)$ for correction: small truncation thresholds $C$ are too conservative and converge much slower than on-policy; large thresholds lead to collapse before the policy converges. Worse, when the actor drifts too far, tokens sampled with high probability by the actor may have extremely low probability under the policy ($p_{\text{inf}} > p_{\text{target}}$), meaning TIS trains on tokens the policy would never select during inference, creating an increasing training-inference mismatch.

**Key Challenge**: IS-based methods only "re-weight samples post-hoc" and cannot change the trajectories actually sampled by the actor. This creates a trade-off where methods are either stable but slow to learn, or aggressive but prone to collapse.

**Goal**: Can we **directly modify the actor's sampling distribution and trajectories** to narrow the distribution gap from the source?

**Core Idea**: **Transform the actor distribution using rejection sampling**. Classical rejection sampling requires the post-sampling distribution to match the target exactly, resulting in near-zero acceptance rates on a 100k-vocab LLM. Jackpot employs **OBRS (Optimal Budgeted Rejection Sampling)**—given a target acceptance "budget," it derives the rejection rule that minimizes KL divergence. It does not force identical distributions but **theoretically guarantees** that the adjusted actor distribution is strictly closer to the policy than the original, achieving an optimal trade-off between sample efficiency and distribution alignment.

## Method

### Overall Architecture
Jackpot introduces lightweight modifications to the standard PPO/GRPO training loop, consisting of three components: **(1) OBRS Rejection-Reweighting**, masking tokens with probability $\min(1, \frac{p_{\text{target}}}{\lambda p_{\text{inf}}})$ to pull the effective distribution towards the target; **(2) Top-K Probability Estimation + Batch-level Bias Correction**, avoiding full-vocabulary summation for the 100k-vocab normalization constant; **(3) Stabilized Jackpot-PPO Loss**, integrating OBRS importance ratios and PPO trust region constraints. The pipeline requires no extra sampling trajectories, no additional forward passes, and no changes to vLLM.

```mermaid
flowchart TD
    A[Phase 1: Standard rollout<br/>actor p_inf sampling trajectories] --> B[Save Top-K logprobs in same forward]
    B --> C[Phase 2: PPO Update]
    C --> D[OBRS Rejection Mask<br/>min(1, p_target/λp_inf)]
    D --> E[Top-K Z_approx estimation<br/>Batch-level κ correction]
    E --> F[Jackpot Weights<br/>ρ = min(w_OBRS,c1)·min(p_ref/p_new,c2)]
    F --> G[L_final = SG(ρ)·L_PPO<br/>Update policy]
```

### Key Designs

**1. OBRS Rejection-Reweighting: Aligning distributions at the source**. For a token $x$ sampled by the actor, it is accepted with probability $\min(1, \frac{p_{\text{target}}(x)}{\lambda p_{\text{inf}}(x)})$. Rejected tokens are masked and do not contribute to loss or gradients. The effective distribution of preserved tokens becomes $p_{\text{OBRS}}(x) = \frac{\min(p_{\text{inf}}(x), p_{\text{target}}(x)/\lambda)}{\sum_{x'} \min(p_{\text{inf}}(x'), p_{\text{target}}(x')/\lambda)}$. Larger scaling factors $\lambda$ (fixed at $\lambda=1$ in experiments, corresponding to control knob $C$) lead to closer alignment but lower acceptance. Theoretically, OBRS is the unique rule that minimizes KL to the target for a given average acceptance rate, ensuring $p_{\text{kept}}$ is strictly closer to the target than $p_{\text{inf}}$. Numerical simulations show it maintains ~95% acceptance even when initial KL is large, while reducing KL by nearly an order of magnitude.

**2. Jackpot-PPO Loss: Joint stabilization of OBRS ratios and trust regions**. Applying PPO directly to OBRS-transformed distributions can violate trust regions. Jackpot overlays the OBRS correction onto the TIS ratio, rewriting $\min(\frac{p_{\text{ref}}}{p_{\text{inf}}}, C)$ as $\min\!\big(Z\cdot\max(\lambda, \frac{p_{\text{ref}}(x)}{p_{\text{inf}}(x)}), C\big)$, where $Z = \sum_{x'} \min(p_{\text{inf}}(x'), \frac{p_{\text{ref}}(x')}{\lambda})$ is the normalization constant. Furthermore, under high staleness (large batches/async), the reference policy might be too outdated. The authors propose **approximating the latest policy $p_{\text{new}}$**: using the change of variables $\mathbb{E}_{x\sim p_{\text{ref}}}[f(x)] = \mathbb{E}_{x\sim p_{\text{new}}}[\frac{p_{\text{ref}}}{p_{\text{new}}} f(x)]$, the final loss is $L = \mathbb{E}_{x\sim p_{\text{inf}}}\big[\min(Z\cdot\max(\lambda, \frac{p_{\text{new}}(x)}{p_{\text{inf}}(x)}), C_1)\cdot \min(\frac{p_{\text{ref}}}{p_{\text{new}}}, C_2)\cdot f(x)\big]$. Two truncation constants $C_1, C_2$ bound the OBRS ratio and the original PPO trust region, with weights applied via stop-gradient to the standard PPO objective.

**3. Top-K Normalization + Batch-level Bias Correction: Computable normalization constants**. Calculating $Z$ requires summing over the entire vocabulary ($|V|>100\text{k}$), which exceeds memory limits if full logit vectors are stored. Jackpot exploits the sparsity of LM probability distributions, summing over $V_k = \text{top-}k(p_{\text{inf}}) \cup \text{top-}k(p_{\theta_{\text{new}}})$ (the union of Top-K tokens from both distributions) to obtain $Z_{\text{approx}}$. At $k=20$, the additional computation is <3%. Since truncation underestimates the true value ($\mathbb{E}[Z_{\text{approx}}] \le Z$), the correction uses the identity that the normalization constant $Z$ equals the expected acceptance rate $\bar{\alpha} = \sum_a \min(p_{\text{inf}}(a), \frac{p_{\text{target}}(a)}{\lambda})$. The empirical acceptance rate $\hat{\bar{\alpha}}$ can be unbiasedly estimated during the rollout phase as "accepted samples / proposed samples." A batch-level correction factor $\kappa = \frac{\hat{\bar{\alpha}}}{\frac{1}{B}\sum_i Z_{\text{approx}}^{(i)}$ scales the low-variance but biased $Z_{\text{approx}}$ to an unbiased scale.

## Key Experimental Results

### Main Results: Large-batch Training (Qwen3-Base + DeepScaleR)
Rollout with large batches, train with small batches (maximizing staleness). Best scores (subset of benchmarks):

| Setting | Method | MATH-500 | AMC22&23 | AIME24(Mean@16) | AIME25(Mean@16) |
|------|------|----------|----------|------|------|
| Qwen3-4B 64× | On Policy | 81.55 | 58.43 | 23.12 | 22.91 |
| | Off Policy | 71.15 | 39.15 | 13.96 | 11.04 |
| | TIS+Adjust | 79.50 | 57.22 | 18.75 | 17.71 |
| | **Jackpot** | **80.05** | 53.92 | **20.63** | **18.13** |
| Qwen3-4B 128× | Off Policy | 60.20 | 33.00 | 8.00 | 5.00 |
| | TIS+Adjust | 19.10 | 7.80 | 1.00 | 1.00 (Collapse) |
| | **Jackpot** | **80.00** | **51.20** | **19.16** | **18.52** |

In the extreme 128× setting, TIS+Adjustment collapses immediately (MATH-500 only 19.1), while Jackpot stays close to on-policy performance, outperforming the off-policy baseline by +20% on AMC and +8% on AIME.

### Extreme Off-policy: Disparate Rollout/Training Models
Trainer = Qwen2.5-3B-Base, Rollout models = Qwen2.5-1.5B-Instruct / Qwen2.5-Math-1.5B-Instruct, trained on MATH-8K. Mean@k results for AMC22&23:

| Method | rollout model | step40 | step50 | step60 | step70 |
|------|------|------|------|------|------|
| Vanilla GRPO | Q2.5-1.5B-IT | 33.3 | 30.0 | 22.4 | 5.7 |
| TIS | Q2.5-1.5B-IT | 32.5 | 26.8 | 0 (Collapse) | 0 |
| **Jackpot** | Q2.5-1.5B-IT | **33.9** | **31.9** | **28.8** | **13.4** |

Under extreme mismatch where rollout and training models are different, Jackpot continues to improve, achieving a +12% gain on MATH-500 compared to the baseline, whereas TIS and GRPO collapse early.

### Key Findings
- **Delayed Collapse**: The primary value of Jackpot is significantly postponing training collapse under large mismatch, bringing off-policy convergence speed close to on-policy.
- **Standalone Efficacy**: In mild mismatch settings (e.g., KV cache FP8 quantization), OBRS rejection sampling alone (without additional truncation tricks) can recover a collapsing training run.
- **Near-Zero Overhead**: $k=20$ adds <3% computation while supporting 64×+ batch sizes, achieving over 4× end-to-end throughput speedup compared to small-batch on-policy training.

## Highlights & Insights
- **Paradigm Shift**: Moves from "post-hoc sample weighting" (IS/TIS) to "proactive distribution transformation" (rejection sampling), addressing actor-policy mismatch fundamentally rather than via patches.
- **Theory Meets Engineering**: Combines theoretical OBRS optimality proofs with practical Top-K + bias correction for production RL systems.
- **The "$Z = \bar{\alpha}$" Identity**: Equating the hard-to-calculate normalization constant with the easily measurable empirical acceptance rate is an elegant engineering solution.
- **Ortho-stackable**: Jackpot modifies $p_{\text{inf}}$, allowing TIS and other methods to be stacked on top for residual correction. It is plug-and-play and requires no vLLM modifications.

## Limitations & Future Work
- **Acceptance-Alignment Trade-off**: Larger $\lambda/C$ improves alignment but lowers acceptance; extreme settings might waste rollout compute on masked tokens, an effect not fully quantified.
- **Limits of Top-K Approximation**: While $k=20$ suffices for focused mathematical reasoning, its bias in higher-entropy tasks (open-ended generation, agentic tasks) remains unverified.
- **Narrow Task Scope**: Experiments focus on Qwen models and mathematical reasoning (GSM8K/MATH/AMC/AIME), excluding code, general alignment, or multi-turn agent tasks.
- **Extreme Settings as "Early Glimpse"**: While outperforming baselines in disparate model settings, the absolute score decline after step 70 (13.4) suggests true long-term stability is still a challenge.

## Related Work & Insights
- **Evolution of Mismatch Correction**: Progressing from IMPALA's V-trace and TIS to FlashRL, AReal, and LlamaRL’s $p_{\text{ref}}/p_{\text{inf}}$ truncation; Jackpot is orthogonal, focusing on "changing the distribution" rather than "changing weights."
- **Rejection Sampling Lineage**: OBRS (Verine et al. 2024) stems from classical rejection sampling traditions in ML. This work grafts it onto RL using acceptance criteria similar to speculative decoding (Leviathan et al. 2023), but without re-sampling or backtracking trajectories.
- **System-level Complementarity**: Complements numerical solutions like FP32 LM heads or deterministic inference to improve off-policy RL reliability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ —— Introduces OBRS rejection sampling to LLM RL for the first time, addressing mismatch via distribution transformation.
- **Experimental Thoroughness**: ⭐⭐⭐ —— Covers large-batch, quantization, and disparate model settings, though tasks are limited to math and models to the Qwen series.
- **Writing Quality**: ⭐⭐⭐⭐ —— Clear motivation, complete derivations, and intuitive explanations of the $Z=\bar{\alpha}$ correction.
- **Value**: ⭐⭐⭐⭐ —— Directly addresses the 70% cost bottleneck of RL rollouts; plug-and-play and low-overhead for high-throughput RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GEPO: Group Expectation Policy Optimization for Stable Heterogeneous Reinforcement Learning](gepo_group_expectation_policy_optimization_for_stable_heterogeneous_reinforcemen.md)
- [\[ICLR 2026\] Scalable Offline Model-Based RL with Action Chunks](scalable_offline_model-based_rl_with_action_chunks.md)
- [\[ICLR 2026\] Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning](escaping_policy_contraction_contraction-aware_ppo_cappo_for_stable_language_mode.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring](master_skill_learning_with_policy-grounded_synergy_of_llm-based_reward_shaping_a.md)

</div>

<!-- RELATED:END -->
