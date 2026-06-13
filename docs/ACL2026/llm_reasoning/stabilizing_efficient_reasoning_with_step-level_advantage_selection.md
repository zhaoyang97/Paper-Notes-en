---
title: >-
  [Paper Note] Stabilizing Efficient Reasoning with Step-Level Advantage Selection
description: >-
  [ACL2026][LLM Reasoning][Inference Compression] This paper discovers that short-context GRPO itself strongly compresses reasoning length but leads to training instability due to incorrect credit assignment of truncated s…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Inference Compression"
  - "GRPO"
  - "Credit Assignment"
  - "Short-context Training"
  - "step-level advantage"
date: 2026-05-08
content_hash: 3a4741ad43bbe617
---

# Stabilizing Efficient Reasoning with Step-Level Advantage Selection

**Conference**: ACL2026 Findings  
**arXiv**: [2604.24003](https://arxiv.org/abs/2604.24003)  
**Code**: https://github.com/HanNight/SAS  
**Area**: Model Compression / Efficient LLM Inference  
**Keywords**: Inference Compression, GRPO, Credit Assignment, Short-context Training, step-level advantage

## TL;DR
This paper discovers that short-context GRPO itself strongly compresses reasoning length but leads to training instability due to incorrect credit assignment of truncated samples. The authors propose Step-level Advantage Selection (SAS) to selectively zero out advantages at the reasoning step granularity, significantly reducing inference tokens while maintaining or improving Pass@1.

## Background & Motivation
**Background**: Long chain-of-thought (CoT) and test-time scaling enable LLMs to perform better on mathematical, logical, and coding tasks, at the cost of increasingly long reasoning trajectories, significant latency, and higher costs. Recent efficient inference methods typically incorporate length penalties, token budgets, or pruning mechanisms during reinforcement learning (RL) post-training to encourage the model to be concise without losing accuracy.

**Limitations of Prior Work**: Many length control methods simultaneously use an easily overlooked training condition: they transition reasoning models originally trained in 16K to 24K long contexts into 4K short contexts for post-training. Consequently, it has been unclear whether the reduction in output length stems from explicit length rewards or from the short-context constraint itself.

**Key Challenge**: While short contexts do compress reasoning, they cause rollouts that are logically correct but truncated at the final answer to be judged as failures. In standard GRPO, every token in a failed rollout receives a negative advantage, and every token in a correct rollout receives a positive advantage. This simultaneously punishes useful intermediate reasoning and reinforces redundant steps in correct answers.

**Goal**: The authors aim to retain the compression signals provided by short contexts while correcting the coarse-grained rollout-level credit assignment. Specifically, the method needs to identify which reasoning steps are reliable and which are noisy, ensuring policy updates originate only from trustworthy steps.

**Key Insight**: The paper treats reasoning trajectories as comprising discrete reasoning steps rather than monolithic strings. The model's own token log probabilities can serve as an approximation of step confidence, avoiding the need for an additional process reward model (PRM).

**Core Idea**: Instead of using explicit length rewards, SAS selectively zeros out advantages at a step-level granularity. This prevents the reinforcement of low-confidence steps in correct rollouts and protects high-confidence steps in failed rollouts from being punished.

## Method

### Overall Architecture
SAS training is built upon GRPO. Multiple rollouts are sampled for each math problem, assigned 0/1 rewards via a rule-based verifier, and advantages are computed through group normalization. Differing from standard GRPO, SAS does not assign the same rollout advantage to every token. Instead, it segments the output into reasoning steps using double newlines, ranks these steps based on their average token log probability, and selectively zeros out the advantages for specific steps.

This "zeroing" operation has different implications for correct versus failed rollouts. For correct rollouts (positive advantage), zeroing reduces the reinforcement of low-confidence steps. For failed rollouts (negative advantage), zeroing protects high-confidence steps from incorrect punishment. The method requires no changes to model architecture or sampling and avoids additional reward models, performing only lightweight post-processing during training.

### Key Designs
1. **Decoupling experiment for short-context compression**:
	- **Function**: To prove that 4K short-context post-training is already a strong compression signal.
	- **Mechanism**: Starting from DeepScaleR-1.5B-Preview, the authors trained using standard GRPO and task correctness rewards in a 4K maximum context without any length penalties. Results showed that output length dropped rapidly early in training, approaching or even becoming shorter than efficient methods like LAPO and ThinkPrune.
	- **Design Motivation**: Without decoupling context length, it is impossible to determine if gains from existing methods stem from length rewards or context changes. This observation also explains why simple short-context training "seems effective" but suffers from accuracy fluctuations and entropy collapse later.

2. **Masking low-confidence steps in correct rollouts**:
	- **Function**: To prevent the model from reinforcing self-doubt, repeated checks, or irrelevant detours in correct answers as high-quality reasoning.
	- **Mechanism**: For rollouts with reward=1, reasoning steps are segmented and their average log probabilities are computed as confidence scores. A ratio $r$ of low-confidence steps have their token advantages set to 0, while other steps retain positive advantages.
	- **Design Motivation**: A correct answer does not imply every step is worth learning. Long CoTs often contain redundant verification or formulaic repetitions. Rollout-level positive feedback solidifies these redundancies, leading to over-reasoning.

3. **Protecting high-confidence steps in failed rollouts**:
	- **Function**: To mitigate mis-punishment caused by short-context truncation.
	- **Mechanism**: For rollouts with reward=0, step confidence is similarly calculated, but high-confidence steps have their negative advantages changed to 0. Low-confidence or clearly erroneous steps retain negative advantages.
	- **Design Motivation**: The authors found that when 8K correct rollouts were truncated to 4K, approximately 29% were marked as failures by the verifier due to missing final answers. Many intermediate steps in these samples are correct; applying negative signals to the entire trajectory undermines training stability.

### Loss & Training
The training objective is the PPO-style GRPO clipped surrogate, utilizing the SAS-modified token-level advantages. The main experiment uses approximately 40K math problems from the DeepScaleR-Preview-Dataset with DeepScaleR-1.5B-Preview as the base model. The training context is fixed at 4K, learning rate at 1e-6, batch size at 128, with 8 rollouts per prompt over 500 steps. The default selection ratio is $r=0.3$, and checkpoints are selected based on the Accuracy-Efficiency Score (AES) on the AIME24 validation set.

## Key Experimental Results

### Main Results
Main results for mathematical reasoning indicate that SAS simultaneously improves accuracy and compresses output length, outperforming baselines with explicit length rewards or pruning mechanisms.

| Method | Avg Pass@1 | Avg Output Tokens | AES | Observation |
|------|-------------|----------------|-----|------|
| DeepScaleR | 52.37 | 5118 | 0.00 | Long-context base model, long output |
| GRPO-4K | 53.61 | 3775 | 0.33 | Short context compresses significantly, but unstable |
| L1-Max | 51.97 | 2071 | 0.33 | Most aggressive compression, builds on accuracy loss |
| LAPO-I | 53.68 | 4001 | 0.30 | Biased towards accuracy, limited compression |
| ThinkPrune-4k | 53.66 | 3878 | 0.33 | Pruning is effective, but trade-off inferior to SAS |
| SAS | 54.54 | 3407 | 0.46 | Highest accuracy, shorter than strong baselines |

Across mathematics datasets, SAS maintains or exceeds GRPO-4K performance on AIME2024, MATH, AMC, and OlympiadBench while reducing average output from DeepScaleR's 5118 tokens to 3407 tokens.

| Dataset | DeepScaleR Pass@1 / token | GRPO-4K Pass@1 / token | SAS Pass@1 / token |
|--------|----------------------------|--------------------------|--------------------|
| AIME2024 | 33.75 / 6755 | 38.75 / 5282 | 39.79 / 4876 |
| AIME2025 | 26.88 / 6444 | 25.42 / 4812 | 26.67 / 4295 |
| MATH | 86.36 / 2809 | 85.09 / 1976 | 86.58 / 1768 |
| AMC | 67.62 / 4761 | 70.18 / 3636 | 71.46 / 3090 |
| OlympiadBench | 47.23 / 4824 | 47.62 / 3651 | 48.19 / 3008 |

Generalization experiments support the conclusions: on GPQA-Diamond, LSAT, and MMLU, SAS achieves an average Pass@1 of 38.30 with 2729 tokens, compared to DeepScaleR's 37.44 / 4416.

| Method | GPQA-Diamond | LSAT | MMLU | Avg Pass@1 | Avg Token | AES |
|------|--------------|------|------|-------------|------------|-----|
| DeepScaleR | 35.70 | 28.64 | 47.99 | 37.44 | 4416 | 0.00 |
| GRPO-4K | 32.48 | 28.45 | 48.73 | 36.55 | 2496 | 0.32 |
| L1-Max | 36.05 | 26.66 | 48.94 | 37.22 | 2242 | 0.46 |
| LAPO-I | 36.17 | 28.42 | 48.71 | 37.77 | 3331 | 0.27 |
| ThinkPrune-4k | 35.83 | 28.13 | 48.91 | 37.62 | 3127 | 0.31 |
| SAS | 37.18 | 28.32 | 49.39 | 38.30 | 2729 | 0.45 |

### Ablation Study
Ablations show that gains do not stem from simple advantage sparsification, but from "confidence-based, step-level, dual processing of correct and failed rollouts."

| Configuration | Avg Pass@1 | Avg Token | AES | Description |
|------|-------------|------------|-----|------|
| SAS full | 54.54 | 3407 | 0.46 | Complete method |
| Only Correct | 53.90 | ~full | 0.43 | Stability drops without protecting failed rollouts |
| Random Steps | N/A | Longer | 0.38 | Randomly selected steps perform worse than confidence ranking |
| Token Level | < full | Longer | 0.39 | Token granularity is less stable than semantic step granularity |
| selection ratio 0.1 | 53.52 | 3259 | 0.43 | Strong compression, accuracy slightly below 0.3 |
| selection ratio 0.3 | 54.54 | 3407 | 0.46 | Best default configuration |
| selection ratio 0.9 | 53.06 | 3482 | 0.36 | Still better than base model, showing robustness |

### Key Findings
- Short-context post-training is itself a powerful compression signal; output shortening cannot be solely attributed to explicit length rewards.
- Verifier failures caused by truncation are a major source of training noise; ~29% of originally correct 8K outputs are judged as failures when truncated to 4K.
- The correlation (nDCG@k) between step-level confidence and external Qwen2.5-Math-PRM-7B rankings reaches 0.9022, suggesting internal log probabilities are sufficient proxies for step quality.
- SAS increases training time per step from 279.08s (GRPO) to 327.15s (an ~17% overhead) without requiring extra forwards, rollouts, or memory.

## Highlights & Insights
- The most valuable contribution is not a new length penalty, but identifying "short-context training" as a hidden variable that compresses reasoning. This discovery changes how we interpret experimental gains in efficient inference papers.
- The zeroing operation in SAS is concise yet generates distinct semantics for positive and negative rollouts: denoising for the former and protection for the latter. This clever use of the GRPO advantage sign structure is a key highlight.
- Using average log probabilities avoids the risks of PRM reward hacking and additional training, making the method easy to integrate into existing RL pipelines.
- This logic is transferable to code generation, tool use, and long-form planning: as long as output can be segmented into semantic steps, rollout-level feedback can be decomposed into step-level credit assignment.

## Limitations & Future Work
- Experiments primarily utilized a 1.5B base model; stability on larger models or different RL recipes remains to be verified.
- All main experiments were fixed at 4K context; truncation rates and optimal selection ratios may vary as context length changes from 2K to 16K.
- Step segmentation relies on double newlines, consistent with DeepScaleR/Qwen data formats but potentially incompatible with other model families or styles.
- Confidence is derived from the current policy; while correlated with PRM rankings, it may underestimate rare but correct steps in poorly calibrated models or OOD tasks.
- Future work could combine SAS with adaptive contexts, dynamic step segmentation, and task-level verifier confidence to reduce dependence on fixed hyperparameters.

## Related Work & Insights
- **vs L1 / LCPO**: L1 methods penalize length explicitly, often sacrificing accuracy; SAS corrects credit assignment under short-context constraints without direct length rewards.
- **vs LAPO**: LAPO models the length distribution of successful solutions; SAS focuses on which steps within a single trajectory should contribute to updates. They are complementary.
- **vs ThinkPrune**: ThinkPrune prunes redundancy via progressive token limits; SAS reduces the reinforcement of redundant steps via training signals, offering a lighter mechanism.
- **vs Entropy/Confidence RL**: These methods modify rewards directly; SAS does not change the reward but uses confidence to decide if the advantage participates in the update, acting as a credit assignment correction.
- **Insight**: Efficient reasoning does not necessarily require "shortness" in the reward function. The fundamental issue may be determining which steps in long outputs are worth learning and which are noise caused by verifiers and context limits.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Insightful focus on short-context hidden variables and step-level credit assignment over conventional rewards.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive across math, general reasoning, and ablations, though model scale coverage is limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and analysis; formulas are a bit crowded in some cached segments.
- Value: ⭐⭐⭐⭐⭐ Direct practical value for efficient RL inference and CoT compression; serves as a reminder to control for the short-context variable in future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ACL 2026\] SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning](shape_stage-aware_hierarchical_advantage_via_potential_estimation_for_llm_reason.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](../../ICLR2026/llm_reasoning/stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)

</div>

<!-- RELATED:END -->
