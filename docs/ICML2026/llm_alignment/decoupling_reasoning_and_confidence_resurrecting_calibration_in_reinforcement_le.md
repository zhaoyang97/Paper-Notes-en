---
title: >-
  [Paper Note] Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards
description: >-
  [ICML 2026][LLM Alignment][RLVR] This paper theoretically proves that in Reinforcement Learning from Verifiable Rewards (RLVR) training (e.g., GRPO)…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "RLVR"
  - "Confidence Calibration"
  - "Gradient Conflict"
  - "Decoupled Optimization"
  - "GRPO"
date: 2026-05-08
content_hash: cbf4c8272eae3dd8
---

# Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards

**Conference**: ICML 2026  
**arXiv**: [2603.09117](https://arxiv.org/abs/2603.09117)  
**Code**: https://github.com/icip-cas/DCPO (Available)  
**Area**: Alignment RLHF / LLM Calibration / RLVR  
**Keywords**: RLVR, Confidence Calibration, Gradient Conflict, Decoupled Optimization, GRPO

## TL;DR
This paper theoretically proves that in Reinforcement Learning from Verifiable Rewards (RLVR) training (e.g., GRPO), the objectives of "improving accuracy" and "reducing calibration error" have negatively correlated gradient directions under the Fisher metric and are irreconcilable. Consequently, the authors propose DCPO: the model is explicitly prompted to output a verbalized confidence segment after the reasoning trajectory. Distinct rewards, advantages, and gradient masks are assigned to reasoning tokens and confidence tokens. While maintaining the same accuracy as GRPO, DCPO reduces the Expected Calibration Error (ECE) from 0.435 to 0.128 (a 71.6% relative reduction).

## Background & Motivation

**Background**: RLVR (Reinforcement Learning from Verifiable Rewards) has become the standard training paradigm for reasoning models such as GRPO and DeepSeek-R1. Using automatically verifiable 0/1 rewards for online policy optimization significantly improves accuracy in mathematical and coding tasks.

**Limitations of Prior Work**: Models trained via RLVR are severely over-confident. Empirical tests on Qwen3-8B under GRPO training show that the average predicted confidence increases from 0.88 to 0.98+, confidence variance drops from 0.006 to 0.001, and the Positive Calibration Error (PCE) rises from 0.312 to 0.362. Even incorrect answers are assigned confidence levels near 1. In high-risk scenarios like medicine, law, and finance, such over-confidence can mislead users.

**Key Challenge**: Prior approaches (e.g., RLCR, CCGSPG) couple calibration objectives (Brier loss, token confidence terms) into the RL reward for joint optimization. This results in an "accuracy-calibration tradeoff" where calibration improvements invariably lead to accuracy drops. The authors diagnose that these two objectives conflict fundamentally in the parameter space, a problem that cannot be solved by simply tuning weights.

**Goal**: (1) Identify the mathematical root cause of over-confidence in RLVR; (2) Design an RL training framework that suppresses over-confidence without sacrificing reasoning accuracy.

**Key Insight**: Starting from the gradient inner product under the Fisher metric, the authors prove that when a model is already over-confident ($\text{Conf}_\theta > \mathbb{E}[R]$), the Fisher inner product of $\nabla J_\text{acc}$ and $\nabla J_\text{cal}$ is strictly less than zero. Therefore, the only solution is to **structurally decouple the optimization of the two objectives into different parameter or token subspaces**, rather than adjusting coefficients within a single loss function.

**Core Idea**: The model first generates a reasoning trajectory $o_r$ and then outputs a verbalized confidence $o_c$. Different rewards and advantages are applied to these two segments, and gradient flow between them is blocked via masks. This completely decouples "solving the problem correctly" from "knowing how certain it is."

## Method

### Overall Architecture
DCPO (Decoupled Calibration Policy Optimization) is built upon the group sampling of GRPO. The pipeline is as follows: Given a prompt $q$, the policy samples $G$ structured responses $o = [o_r\ \texttt{<conf>}\ o_c]$, where $o_r$ contains the reasoning and final answer, and $o_c$ is an explicit confidence value. Two rewards are calculated for each response: a reasoning reward $R(o_r)=\mathbb{I}(y_\text{pred}=y_\text{label})$ and a confidence reward $R_c(o_c)=-|\text{conf}(o_c)-R_{IG}|$ (where $R_{IG}$ is a hybrid group-instance supervision signal). Group-relative normalization is performed to obtain $A_r$ and $A_c$. Finally, a token-level mask ensures $A_r$ is only applied to the $o_r$ segment and $A_c$ to the $o_c$ segment. This channels reasoning and confidence parameter updates through separate gradient paths.

### Key Designs

1.  **Block-wise Verbalized Confidence Rollout (Structural Decoupling)**:
    - **Function**: Forces model output into two distinct blocks: "Reasoning Segment + Confidence Segment," separated by a special token `<conf>`.
    - **Mechanism**: The prompt requires the model to follow a standard Chain-of-Thought (CoT) to provide an answer, then output a scalar confidence value (e.g., 0.85) after the `<conf>` token. This allows rewards and masks to target specific token subsets accurately; non-compliant outputs receive a format penalty.
    - **Design Motivation**: If using logit-based confidence (e.g., $\text{Conf}(y)=\prod \pi_\theta(y_i|y_{<i})$), the reasoning token probabilities contribute to both correctness and confidence, making decoupling impossible. Verbalized confidence occupies independent token positions, which is the physical prerequisite for gradient masking.

2.  **Decoupled Advantage Estimation with Hybrid Calibration Target (Reward Decoupling + Low-variance Supervision)**:
    - **Function**: Designs two independent advantages for reasoning and calibration, using a hybrid of group-level and instance-level accuracy as the regression target for calibration.
    - **Mechanism**: The reasoning reward follows standard 0/1 correctness from GRPO. The calibration reward uses $R_{IG}=\lambda \tilde{R}_G + (1-\lambda) R(o_r)$ as an estimate of the model's true capability on that prompt, where $\tilde{R}_G=\frac{1}{G}\sum R(o_{r,i})$ is the average accuracy within the group. The confidence reward is $R_c(o_c)=-|\text{conf}(o_c)-R_{IG}|$, pulling the verbalized number toward the actual accuracy. Both rewards are normalized via mean/std within the group to obtain $A_{r,i}$ and $A_{c,i}$.
    - **Design Motivation**: Proposition 4.3 proves that instance-level binary supervision $R(y)$ is a single Bernoulli sample with variance $4p(1-p)$, which pushes confidence toward extreme 0/1 values. Conversely, the group average $\tilde{R}_G$ is an unbiased estimate of $\mathbb{E}[R]$ with variance $O(1/G)$, serving as a significantly more stable calibration target. The hybrid $\lambda$ interpolates between stability (favoring group) and instance-level discriminativeness.

3.  **Masked Gradient Optimization (Gradient Channel Isolation)**:
    - **Function**: Applies different advantages based on token type within the PPO/GRPO surrogate objective.
    - **Mechanism**: A token mask is constructed for each response to divide the sequence into $o_r$ and $o_c$ segments. The final optimization objective is defined as:
    $$\frac{1}{G}\sum_i \frac{1}{|o_i|}[\sum_{y_j \in o_r}\hat{\rho}_{i,j}A_{r,i} + \sum_{y_j \in o_c}\hat{\rho}_{i,j}A_{c,i}]$$
    where $\hat\rho$ is the clipped importance ratio. Accuracy gradients only update the conditional distribution of reasoning tokens, while confidence gradients only update the distribution of tokens following `<conf>`.
    - **Design Motivation**: This translates the gradient conflict theorem from Section 4.2 into an actionable solution. Since the two rewards structurally do not act on the same set of token logits, the Fisher negative inner product conflict is physically eliminated. Theorem 5.1 further proves that under this decoupling, the optimal confidence for a proper scoring rule is exactly equal to the true expected accuracy $\mathbb{E}[c|q]=\mathbb{E}_{y\sim\pi_\theta}[R(y)]$, without dragging down the reasoning policy.

### Loss & Training
The base model is Qwen3-8B (non-thinking), trained on the DeepScaler dataset. Group size $G$ follows GRPO defaults. $\lambda$ in the hybrid calibration target is selected via ablation (DCPO-I for $\lambda=0$, DCPO-G for $\lambda=1$, and DCPO for the hybrid). Format penalties are applied to ensure verbalized confidence is parsable.

## Key Experimental Results

### Main Results
Comparing Base / GRPO / RLCR / CCGSPG / DCPO across 5 math benchmarks (MATH-500 / AIME24 / AIME25 / AMC23 / AMC24), with confidence measured in verbalized form.

| Method | Overall Acc ↑ | Overall ECE ↓ | Overall PCE ↓ | Overall AUROC ↑ |
|:---|:---:|:---:|:---:|:---:|
| Base (verbal) | 46.4 | 0.435 | 0.426 | 0.609 |
| GRPO (verbal) | 57.4 | 0.372 | 0.363 | 0.532 |
| RLCR | 56.5 | 0.139 | 0.128 | 0.753 |
| CCGSPG | 57.6 | 0.230 | 0.283 | 0.815 |
| **Ours (DCPO)** | **60.8** | **0.128** | **0.126** | **0.881** |

**Key Comparison**: DCPO's accuracy is on par with or higher than GRPO (60.8 vs 57.4), while ECE is slashed from GRPO's 0.372 to 0.128 (a 71.6% reduction relative to Base). While RLCR nears this calibration, its accuracy drops by 1.1 points. CCGSPG maintains accuracy but suffers from a higher ECE of 0.230.

### Ablation Study
| Configuration | Overall Acc | Overall ECE | Description |
|:---|:---:|:---:|:---|
| DCPO (Hybrid) | 60.8 | 0.128 | Full model |
| DCPO-G (Group only) | 60.5 | 0.209 | Calibration target uses only $\tilde R_G$; accuracy stays high but ECE is higher |
| DCPO-I (Instance only) | 58.7 | 0.138 | Calibration target uses only $R(o_r)$; ECE is similar but Acc drops 2 points |

### Key Findings
- The hybrid group + instance target achieves SOTA in both Acc and ECE, validating the theoretical judgment that group signals provide low variance while instance signals provide discriminativeness.
- AUROC shows the most significant improvement (0.532 → 0.881), indicating that verbalized confidence is not just numerically close to accuracy but also possesses strong "correctness discrimination" ability—a useful byproduct of RLVR + decoupled rewards.
- Additional code generation experiments on LiveCodeBench and HumanEval+ yield consistent conclusions: DCPO suppresses over-confidence while maintaining GRPO-level accuracy across domains.

## Highlights & Insights
- **Theory-to-Architecture Derivation**: The Fisher negative inner product in Proposition 4.2 isn't just an ex-post explanation; it directly informs the structural decision to split objectives into different token subsets. The logical loop from "why coupling fails" to "how to decouple" is highly convincing.
- **Reusing GRPO Group Sampling for Low-variance Supervision**: $\tilde R_G$ is effectively "free." Since GRPO already samples $G$ rollouts to calculate advantages, using their average accuracy as a regression target adds no extra annotation or critic networks. Finding new signals within existing structures is an elegant approach.
- **Transferable "Verbalized Confidence + Masked Gradient" Pattern**: This framework can be applied to any task where an LLM must "know what it doesn't know" (factual QA, tool use, agent decision-making). As long as the output can be segmented into a "task block" and a "meta-cognitive block," it can be trained with this decoupled reward framework.
- **Indicator Design Nuance**: The authors introduce PCE (computing ECE only on bins where confidence > accuracy) to specifically characterize over-confidence. This avoids the illusion of improved calibration that occurs when ECE drops solely due to increased accuracy.

## Limitations & Future Work
- **Dependency on Verbalization Capability**: The base model must be able to output parsable confidence numbers reliably; otherwise, format penalties may dominate early training. Smaller models might require an SFT warm-start.
- **End-of-trajectory Confidence**: Confidence is provided only once at the end of the trajectory, which is coarse-grained. Step-level calibration might require extending masks to intermediate checkpoints.
- **Theoretical Assumption $\text{Cov}(R, \phi) > 0$**: This requires confidence features to be positively correlated with accuracy. This might not hold for a near-random base model early in training, perhaps explaining why calibration loss fluctuates in initial steps.
- **Comparison with Preference-based RMs**: DCPO uses only verifiable rewards and has not been directly compared against routes using learned preference models or confidence heads.
- **$\lambda$ Tuning**: The hybrid coefficient relies on empirical ablation rather than an adaptive scheme based on group size or task difficulty.

## Related Work & Insights
- **vs RLCR (Damani et al., 2025)**: RLCR adds Brier Score loss to RLVR rewards, a typical "coupled optimization." This paper theoretically proves its inevitable gradient conflict; empirically, while RLCR's calibration is close to DCPO, its accuracy drops (56.5 vs 60.8), confirming the tradeoff.
- **vs CCGSPG (Liu et al., 2025)**: CCGSPG reshapes GRPO advantages based on token-level confidence but remains a coupled solution. DCPO outperforms it significantly in ECE (0.128 vs 0.230) and AUROC, proving structural decoupling is superior to signal re-weighting.
- **vs Inference-time Calibration**: Post-hoc methods don't modify model weights, relying instead on external predictors or sampling tricks. DCPO bakes calibration into the weights, incurring no extra deployment overhead despite higher training costs.
- **vs Original GRPO**: DCPO can be seen as a minimally invasive extension of GRPO, adding only a confidence rollout and a set of masks without altering the core PPO logic. It is nearly a "drop-in" for existing GRPO infrastructures.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of theoretical diagnosis (Fisher gradient conflict) and structural decoupling is rare, though verbalized confidence and group rewards have appeared elsewhere.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 math and 3 code benchmarks with DCPO-I/G ablations and multiple baselines, though validation on larger models (70B+) is missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The chain of reasoning from theory to observation to algorithm to experiment is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a critical pain point of the RLVR era (over-confidence) with a simple, GRPO-compatible solution that has high industrial utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards](simultaneous_multi-objective_alignment_across_verifiable_and_non-verifiable_rewa.md)
- [\[ACL 2026\] Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data](../../ACL2026/llm_alignment/too_correct_to_learn_reinforcement_learning_on_saturated_reasoning_data.md)
- [\[AAAI 2026\] DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](../../AAAI2026/llm_alignment/decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](../../ICLR2026/llm_alignment/no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)

</div>

<!-- RELATED:END -->
