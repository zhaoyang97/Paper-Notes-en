---
title: >-
  [Paper Note] Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards
description: >-
  [ICML 2026][LLM Alignment][RLVR] This paper theoretically demonstrates that the objectives of "improving accuracy" and "reducing calibration error" in RLVR (e.g., GRPO) training have negatively correlated gradient directions under the Fisher metric and are irreconcilable. It proposes DCPO: allowing the model to explicitly output a verbalized confidence segment after the reasoning trajectory, assigning independent rewards / advantages / masked gradients to reasoning tokens and…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "RLVR"
  - "Confidence Calibration"
  - "Gradient Conflict"
  - "Decoupled Optimization"
  - "GRPO"
date: 2026-05-08
content_hash: 3aebe66ef97f91f4
---

# Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards

**Conference**: ICML 2026  
**arXiv**: [2603.09117](https://arxiv.org/abs/2603.09117)  
**Code**: https://github.com/icip-cas/DCPO (Available)  
**Area**: Alignment RLHF / LLM Calibration / RLVR  
**Keywords**: RLVR, Confidence Calibration, Gradient Conflict, Decoupled Optimization, GRPO

## TL;DR
This paper theoretically demonstrates that the objectives of "improving accuracy" and "reducing calibration error" in RLVR (e.g., GRPO) training have negatively correlated gradient directions under the Fisher metric and are irreconcilable. It proposes DCPO: allowing the model to explicitly output a verbalized confidence segment after the reasoning trajectory, assigning independent rewards / advantages / masked gradients to reasoning tokens and confidence tokens. While maintaining the same accuracy as GRPO, it reduces the ECE from 0.435 to 0.128 (a 71.6% relative reduction).

## Background & Motivation

**Background**: Reinforcement Learning from Verifiable Rewards (RLVR) has become the standard training paradigm for reasoning models like GRPO and DeepSeek-R1. Using automatically verifiable 0/1 rewards to optimize policies online can significantly improve accuracy in mathematics and coding tasks.

**Limitations of Prior Work**: Models trained via RLVR are severely over-confident. Empirical tests on Qwen3-8B under GRPO training show that the average predicted confidence rises from 0.88 to over 0.98, confidence variance drops from 0.006 to 0.001, and Positive Calibration Error (PCE) increases from 0.312 to 0.362; even incorrect answers are assigned confidence levels near 1. In high-risk scenarios such as medical, legal, and financial fields, such over-confidence can mislead users.

**Key Challenge**: Previous approaches (RLCR, CCGSPG) coupled calibration objectives (Brier loss, token confidence terms) into the RL reward for joint optimization, resulting in an "accuracy-calibration tradeoff"—improving calibration inevitably leads to an accuracy drop. The authors diagnose this as a fundamental directional conflict of these two objectives in the parameter space, which cannot be resolved by merely tuning weights.

**Goal**: (1) Identify the mathematical root cause of over-confidence in RLVR; (2) Design an RL training framework that suppresses over-confidence without sacrificing reasoning accuracy.

**Key Insight**: Starting from the gradient inner product under the Fisher metric, the authors prove that when a model is already over-confident ($\text{Conf}_\theta > \mathbb{E}[R]$), the Fisher inner product of $\nabla J_\text{acc}$ and $\nabla J_\text{cal}$ is strictly less than zero. Therefore, the only solution is to **structurally decouple the two objectives into different parameter subspaces or token subspaces for optimization**, rather than tuning coefficients within the same loss function.

**Core Idea**: The model is tasked to generate a reasoning trajectory $o_r$ followed by a verbalized confidence $o_c$. Different rewards and advantages are assigned to these two token segments, and gradients are prevented from cross-interference through masking, completely decoupling "solving the problem" from "knowing one's level of certainty."

## Method

### Overall Architecture
DCPO (Decoupled Calibration Policy Optimization) aims to resolve the mutually exclusive goals of maintaining accuracy and suppressing over-confidence in standard RLVR. It is built upon the group sampling of GRPO. Given a prompt $q$, the policy samples $G$ structured responses $o = [o_r\ \texttt{<conf>}\ o_c]$, where $o_r$ contains the reasoning chain and final answer, and $o_c$ following the `<conf>` tag is the explicit confidence value. Two sets of rewards are calculated for each response: a reasoning reward $R(o_r)=\mathbb{I}(y_\text{pred}=y_\text{label})$ based on answer correctness, and a confidence reward $R_c(o_c)=-|\text{conf}(o_c)-R_{IG}|$ based on the proximity of the output value to the true accuracy. These are normalized within the group into advantages $A_r$ and $A_c$, respectively. Token-level masks ensure $A_r$ backpropagates only to the $o_r$ segment and $A_c$ only to the $o_c$ segment. Reasoning and confidence thus utilize two non-interfering gradient channels, realizing the "gradient conflict theorem" in an executable architecture.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    Q["prompt q"] --> G["GRPO group sampling<br/>Sample G responses"]
    G --> S["Structured confidence rollout<br/>Split each into o_r reasoning + ⟨conf⟩ + o_c confidence"]
    subgraph REW["Decoupled advantage + Hybrid calibration target"]
        direction TB
        S --> RR["Reasoning reward R(o_r) = correctness 0/1"]
        S --> RC["Confidence reward R_c = −|conf − R_IG|<br/>R_IG = λ·group mean + (1−λ)·instance correctness"]
        RR --> AR["Normalization within group → A_r"]
        RC --> AC["Normalization within group → A_c"]
    end
    REW --> M["Masked gradient optimization<br/>A_r backprops to o_r, A_c backprops to o_c"]
    M --> O["Update policy: Accuracy maintained, ECE significantly reduced"]
```

### Key Designs

**1. Structured confidence rollout: Physically separating reasoning and confidence tokens**

The diagnosis identifies that over-confidence stems from the reasoning token probabilities simultaneously serving both to "calculate correctly" and "express certainty," making independent optimization impossible. If logit-based confidence continues to be used (e.g., $\text{Conf}(y)=\prod \pi_\theta(y_i|y_{<i})$), confidence remains a byproduct of reasoning probabilities, and adjusting one necessarily affects the other. DCPO decouples the two via the generation structure: the prompt requires the model to provide an answer via a Chain of Thought and then output a scalar confidence (e.g., 0.85) separately after a special token `<conf>`. Formatting penalties are applied to non-compliant outputs. This allows confidence to occupy independent token positions, enabling subsequent rewards and masks to precisely target respective token subsets—a physical prerequisite for the decoupling scheme.

**2. Decoupled advantage + Hybrid calibration target: Finding a low-variance, discriminative regression target for confidence**

With tokens separated, the reasoning reward follows the 0/1 correctness of GRPO. The challenge lies in determining the target value for confidence. The simplest target is the instance's own correctness $R(o_r)$, but Lemma 4.3 indicates this is a single Bernoulli sample with a high variance of $4p(1-p)$, which would force confidence toward extremes of 0 or 1 and exacerbate over-confidence. Since GRPO already involves $G$ rollouts, their average accuracy $\tilde{R}_G=\frac{1}{G}\sum R(o_{r,i})$ is an unbiased estimator of the true expectation $\mathbb{E}[R]$ with variance $O(1/G)$, providing a stable supervision source. DCPO interpolates the two into a hybrid target $R_{IG}=\lambda \tilde{R}_G + (1-\lambda) R(o_r)$ and defines the confidence reward as $R_c(o_c)=-|\text{conf}(o_c)-R_{IG}|$, forcing the output to converge toward the model's true capability for that specific prompt. $\lambda$ balances stability (group mean) and instance-level discrimination (correctness). Both rewards are normalized (mean/std) within the group to obtain $A_{r,i}$ and $A_{c,i}$.

**3. Masked gradient optimization: Physically eliminating Fisher negative inner product conflicts via token masks**

With two token segments and two sets of advantages, the final step is to ensure their gradients do not interfere. DCPO constructs a token mask for each response to split the sequence into $o_r$ and $o_c$ segments. The optimization objective is:

$$\frac{1}{G}\sum_i \frac{1}{|o_i|}\Big[\sum_{y_j \in o_r}\hat{\rho}_{i,j}A_{r,i} + \sum_{y_j \in o_c}\hat{\rho}_{i,j}A_{c,i}\Big]$$

where $\hat\rho$ is the clipped importance ratio. The accuracy gradient updates only the conditional distribution of reasoning tokens, while the confidence gradient updates only the distribution of tokens after `<conf>`. The two rewards structurally never land on the same set of logits. Thus, the conflict where the Fisher inner product of $\nabla J_\text{acc}$ and $\nabla J_\text{cal}$ is strictly negative (when over-confident) is physically isolated. Theorem 5.1 further guarantees that under this decoupling, the optimal confidence for a proper scoring rule equals the true expected accuracy $\mathbb{E}[c|q]=\mathbb{E}_{y\sim\pi_\theta}[R(y)]$, ensuring that calibrating does not hinder the reasoning policy.

### Loss & Training
The base model is Qwen3-8B (non-thinking). The training set is DeepScaler, with group size $G$ following the GRPO default. $\lambda$ in the hybrid calibration target is selected via ablation (DCPO-I for $\lambda=0$, DCPO-G for $\lambda=1$, and DCPO for the hybrid). Formatting penalties ensure that verbalized confidence is parsable.

## Key Experimental Results

### Main Results
DCPO is compared against Base / GRPO / RLCR / CCGSPG on 5 math benchmarks (MATH-500 / AIME24 / AIME25 / AMC23 / AMC24), using verbalized confidence.

| Method | Overall Acc ↑ | Overall ECE ↓ | Overall PCE ↓ | Overall AUROC ↑ |
|------|---------------|---------------|---------------|-----------------|
| Base (verbal) | 46.4 | 0.435 | 0.426 | 0.609 |
| GRPO (verbal) | 57.4 | 0.372 | 0.363 | 0.532 |
| RLCR | 56.5 | 0.139 | 0.128 | 0.753 |
| CCGSPG | 57.6 | 0.230 | 0.283 | 0.815 |
| **DCPO** | **60.8** | **0.128** | 0.126 | **0.881** |

Key comparison: DCPO's accuracy is on par with or higher than GRPO (60.8 vs 57.4), while ECE is slashed from GRPO's 0.372 to 0.128, a 71.6% reduction relative to the Base. RLCR reaches similar calibration but loses 1.1 points in accuracy, while CCGSPG maintains accuracy but its ECE remains at 0.230.

### Ablation Study

| Configuration | Overall Acc | Overall ECE | Description |
|------|-------------|-------------|------|
| DCPO (Hybrid) | 60.8 | 0.128 | Full model |
| DCPO-G (Group only) | 60.5 | 0.209 | Calibration target uses only $\tilde R_G$; accuracy stays high but ECE is higher |
| DCPO-I (Instance only) | 58.7 | 0.138 | Calibration target uses only $R(o_r)$; ECE is similar but Acc drops by 2 points |

### Key Findings
- The hybrid group + instance target achieves SOTA in both Acc and ECE, validating the theoretical judgment in Section 4.3 regarding "low-variance group signals" and "discriminative instance signals."
- AUROC is the metric most improved by DCPO (0.532 → 0.881), indicating that verbalized confidence is not just numerically close to accuracy but also possesses strong "correctness discrimination" capabilities—a natural byproduct of RLVR + decoupled rewards.
- Additional code generation experiments on LiveCodeBench and HumanEval+ yield consistent conclusions: DCPO significantly suppresses over-confidence while maintaining GRPO accuracy across domains.

## Highlights & Insights
- **Direct derivation from theory to architecture**: Proposition 4.2 regarding negative Fisher inner products is not a post-hoc explanation but a direct driver for the architectural decision to "separate objectives into different token subsets." The logical loop from "why coupling fails" to "how to decouple" is far more convincing than heuristic reward engineering.
- **Reusing GRPO group sampling as a low-variance supervision source**: $\tilde R_G$ is essentially free. Since GRPO already samples $G$ rollouts to calculate advantages, the authors directly use the average accuracy of these rollouts as the confidence regression target, requiring no additional labeling or critic networks. Finding new signals within existing structures is an elegant approach.
- **Transferable verbalized confidence + masked gradient pattern**: This can be applied to any task requiring an LLM to "know what it doesn't know" (factual QA, tool use, Agent decision-making), provided the output can be segmented into "task block + metacognitive block" for training with the same decoupled reward framework.
- **Detailed metric design**: The authors introduce PCE (ECE calculated only for bins where confidence > accuracy) to specifically characterize over-confidence, avoiding the illusion of reduced ECE caused passively by increased accuracy. This metric is better suited for monitoring RLVR training than ECE.

## Limitations & Future Work
- **Dependency on the model's "verbalized confidence" capability**: The base model must be able to stabley output parsable confidence numbers; otherwise, format penalties will dominate early training. Smaller models might require SFT for a cold start.
- **Confidence is provided only at the end of the trajectory**: This is a coarse-grained whole-sequence confidence and cannot pinpoint where a reasoning chain begins to falter. Fine-grained step-level calibration may require extending the mask to intermediate checkpoints.
- **Theoretical assumption $\text{Cov}(R, \phi) > 0$**: This requires confidence features to be positively correlated with accuracy, which may not hold for a newly trained, nearly random base model, potentially explaining the vibration of calibration loss in early training steps.
- **Implicit comparison with RLHF-style reward models**: DCPO uses only verifiable rewards and has not been directly compared against routes using learned confidence heads + preference data.
- **$\lambda$ hyperparameter tuning**: The hybrid coefficient relies on empirical ablation without an adaptive scheme based on group size or task difficulty.

## Related Work & Insights
- **vs RLCR (Damani et al., 2025)**: RLCR adds Brier Score loss to RLVR rewards, a typical "coupled optimization." This paper theoretically proves its inevitable gradient conflict. Experimentally, while RLCR's calibration is close to DCPO (ECE 0.139 vs 0.128), its accuracy drops (56.5 vs 60.8), confirming the tradeoff.
- **vs CCGSPG (Liu et al., 2025)**: CCGSPG reshapes GRPO advantages based on token-level confidence, which remains a coupled scheme. DCPO is significantly better in ECE (0.128 vs 0.230) and leads in AUROC, suggesting structural decoupling is more effective than signal weight adjustment.
- **vs Inference-time calibration (Chhikara, Ni et al.)**: Post-hoc methods do not alter model weights and rely on external predictors or sampling tricks. DCPO bakes calibration into the weights, incurring no extra overhead during deployment, though training costs are higher.
- **vs Original GRPO**: The "two sets of advantages + token mask" in DCPO can be seen as a minimally intrusive extension of GRPO—adding only a confidence rollout and a set of masks without changing the PPO core. It is almost drop-in compatible with existing GRPO training infrastructure.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of theoretical diagnosis (Fisher gradient conflict) + structural decoupling is rare, though verbalized confidence and group rewards have appeared previously.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 math + 3 code benchmarks, including DCPO-I/G ablations and multiple baselines, though validation on larger models (70B+) is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ The chain of reasoning from theory → empirical observation → algorithm → experiment is very clear, and the introduction of the PCE metric enhances persuasiveness.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a key pain point of LLM deployment in the RLVR era (over-confidence). The solution is simple and compatible with existing GRPO infrastructure, offering high industrial deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards](simultaneous_multi-objective_alignment_across_verifiable_and_non-verifiable_rewa.md)
- [\[ICLR 2026\] Annotation-Efficient Honesty Alignment via Confidence Elicitation and Calibration](../../ICLR2026/llm_alignment/annotation-efficient_honesty_alignment_via_confidence_elicitation_and_calibratio.md)
- [\[ACL 2026\] Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data](../../ACL2026/llm_alignment/too_correct_to_learn_reinforcement_learning_on_saturated_reasoning_data.md)
- [\[ICML 2026\] TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning](truthrl_incentivizing_truthful_llms_via_reinforcement_learning.md)
- [\[AAAI 2026\] DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](../../AAAI2026/llm_alignment/decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)

</div>

<!-- RELATED:END -->
