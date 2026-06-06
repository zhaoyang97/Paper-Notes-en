---
title: >-
  [Paper Note] Efficient Process Reward Modeling via Contrastive Mutual Information
description: >-
  [ACL 2026][LLM Reasoning][Process Reward Models] The paper proposes CPMI (Contrastive Pointwise Mutual Information), an efficient automated step-level reward annotation method. By contrasting the changes in conditional p…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Process Reward Models"
  - "step-level supervision"
  - "mutual information"
  - "contrastive learning"
  - "mathematical reasoning"
date: 2026-05-08
content_hash: 16d3e8b4f2c89c14
---

# Efficient Process Reward Modeling via Contrastive Mutual Information

**Conference**: ACL 2026  
**arXiv**: [2604.10660](https://arxiv.org/abs/2604.10660)  
**Code**: [GitHub](https://github.com/nakyungLee20/CPMI)  
**Area**: LLM Reasoning  
**Keywords**: Process Reward Models, step-level supervision, mutual information, contrastive learning, mathematical reasoning

## TL;DR
The paper proposes CPMI (Contrastive Pointwise Mutual Information), an efficient automated step-level reward annotation method. By contrasting the changes in conditional probabilities of correct and incorrect answers given a reasoning step, it estimates step-wise contributions. This approach reduces construction time by 84% and token generation by 98% compared to Monte Carlo estimation, while achieving higher accuracy on process-level evaluations and mathematical reasoning benchmarks.

## Background & Motivation

**Background**: Process Reward Models (PRMs) verify Chain-of-Thought (CoT) trajectories by evaluating the correctness of intermediate reasoning steps, which is more reliable than Outcome Reward Models (ORMs) that only evaluate final answers. However, training PRMs requires step-level annotated data, traditionally provided by humans or high-performance LLMs.

**Limitations of Prior Work**: (1) Manual annotation of step-level rewards is extremely costly and time-consuming; (2) Automated methods like Monte Carlo (MC) estimation require a large number of LLM rollouts to obtain low-variance reward signals, which is computationally expensive—requiring dozens of sampled trajectories per step to estimate correctness; (3) MC estimation is particularly unstable in the early steps of a reasoning chain due to short prefixes and high subsequent variance.

**Key Challenge**: There is a significant gap between the acquisition cost of step-level supervision signals and the training requirements of PRMs—high-quality annotations are needed in large quantities but are extremely expensive to obtain.

**Goal**: To design a method that estimates step-level rewards with only a single forward pass, eliminating the dependency on multiple MC rollouts.

**Key Insight**: The relationship between MC estimation ($\lambda \to 1$, long-term return) and the proposed method ($\lambda \to 0$, single-step bootstrap) is understood from a TD($\lambda$) perspective. It is hypothesized that pretrained LLMs already encode sufficient mathematical knowledge to infer a step's contribution by observing the change in the model's probability of producing the correct answer after that step is added.

**Core Idea**: Step Reward = (Incremental log-probability of the correct answer after adding the step) - (Incremental log-probability of incorrect answers after adding the step), representing a contrastive pointwise mutual information.

## Method

### Overall Architecture
For each reasoning step: (1) Calculate the log-probability difference of the model outputting the correct answer with and without the step; (2) Similarly calculate the difference for incorrect answers; (3) The difference between these two values constitutes the CPMI reward. Normalized CPMI is then used as soft labels to train the PRM. During inference, the PRM scores candidate trajectories to select the optimal one.

### Key Designs

1. **CPMI Reward Formula**:

    - Function: Estimates step-level contribution via a single forward pass.
    - Mechanism: $r_{\text{CPMI}}^i = [\log p_\theta(A|q,s_i) - \log p_\theta(A|q)] - \frac{1}{M}\sum_{m=1}^M [\log p_\theta(\tilde{A}|q,s_i) - \log p_\theta(\tilde{A}|q)]$. The first term quantifies the boost to the correct answer's probability, while the second term quantifies the suppression of incorrect answer probabilities. An effective step should simultaneously increase the correct probability and decrease incorrect probabilities.
    - Design Motivation: Pure PMI (focusing only on the correct answer) lacks discriminative power. Introducing a contrastive signal significantly enhances the distinction of the reward signal.

2. **Theoretical Connection: CPMI $\approx$ Jeffreys Divergence**:

    - Function: Provides a theoretical foundation for the CPMI reward.
    - Mechanism: Under the assumption of a single correct answer in mathematical reasoning, CPMI can be interpreted as an approximation of the Jeffreys Divergence (symmetric KL divergence) between the answer distributions conditioned "with the step" and "without the step." This implies that CPMI favors steps that cause large, symmetric shifts in the answer distribution.
    - Design Motivation: The symmetry of Jeffreys Divergence ensures bidirectional penalties—detecting not only the increase in correct answer probability but also the decrease in incorrect answer probability.

3. **CPMI-Merge Hybrid Strategy**:

    - Function: Addresses the issue of high noise in CPMI for early steps of the reasoning chain.
    - Mechanism: Uses MC estimation (global information) for initial steps (e.g., step 1) and CPMI (local bootstrapping) for subsequent steps. It leverages the complementary strengths of both—MC captures global info but is expensive, while CPMI provides dense feedback but is unstable early on.
    - Design Motivation: From a TD-$\lambda$ perspective, this seeks an optimal balance between $\lambda=1$ (MC) and $\lambda=0$ (CPMI).

### Loss & Training
Qwen3-4B-Base is used as the PRM backbone with an additional two-layer linear reward head. Training utilizes BCE loss with CPMI rewards normalized via z-score as soft labels. During inference, candidate trajectories are selected based on weighted scores from the PRM.

## Key Experimental Results

### Main Results (Efficiency + Quality)

| Reward Type | AUC | PB | PRMB | MATH | Time (Ratio) | Token (Ratio) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MC | 0.759 | 27.7 | 38.8 | 45.4 | 1.00 | 1.00 |
| PAV | 0.757 | 36.6 | 49.6 | 47.2 | 1.17 | 2.38 |
| **CPMI** | **0.765** | 34.6 | **58.8** | 48.2 | **0.16 (↓84%)** | **0.02 (↓98%)** |
| **CPMI_Merge** | **0.766** | **36.8** | **60.7** | **49.4** | 0.30 (↓70%) | 0.18 (↓82%) |

### Ablation Study

| Configuration | Description |
| :--- | :--- |
| No Contrastive (PMI only) | AUC decreases, lacks discriminative power |
| No Prompt Averaging | Reward variance increases |
| Different M (Negative samples) | M=4 offers the best balance |
| CPMI-Merge (step 1) | More stable than pure CPMI |

### Key Findings
- **CPMI reduces construction time by 84% and token generation by 98%** while achieving higher quality (AUC 0.765 vs. MC 0.759).
- **Significant outperformance over MC on the process-level benchmark PRMB (58.8 vs. 38.8)**, indicating that step-level signals generated by CPMI are more effective for process-level verification.
- **The contrastive signal is critical**: Removing the contrastive term (using only PMI) leads to a significant drop in performance.
- **CPMI-Merge further improves stability**: It eliminates noise in early steps while retaining most efficiency advantages.
- **RelEff (Relative Efficiency ratio)** is as high as 6-10x, showing that CPMI is far superior to MC in the quality-cost trade-off.

## Highlights & Insights
- **Unified understanding of MC and CPMI from a TD-$\lambda$ perspective** is an elegant theoretical contribution—MC = $\lambda \to 1$ (full return), CPMI = $\lambda \to 0$ (bootstrapping), applying classic reinforcement learning frameworks to PRM training.
- **98% token reduction** means CPMI makes the construction of large-scale PRM datasets practically feasible—moving from "days on a GPU cluster" to "hours on a single machine."
- **Theoretical guarantees for CPMI rewards** (Jeffreys Divergence approximation) ensure it is more than just a heuristic method, providing a solid theoretical basis.

## Limitations & Future Work
- CPMI relies on the quality of the pretrained LLM's internal probability distribution; if the model's mathematical knowledge is insufficient, probability estimates may be unreliable.
- Validated only on mathematical reasoning tasks; effectiveness on other tasks requiring PRMs, such as code generation or logical reasoning, remains to be confirmed.
- The construction strategy for hard negative samples (M=4 + heuristic perturbations) may not be sufficiently systematic.
- The instability of CPMI in early steps requires the CPMI-Merge strategy, which increases design complexity.
- The assumption of a single correct answer may not hold for certain tasks.

## Related Work & Insights
- **vs. MC Estimation (Math-Shepherd)**: MC requires sampling dozens of complete trajectories per step, whereas CPMI requires only one forward pass.
- **vs. PAV (Setlur et al.)**: PAV still relies on MC rollouts, while CPMI completely eliminates the rollout requirement.
- **vs. Contrastive Decoding (Li et al.)**: While contrastive decoding manipulates logits during inference, CPMI uses contrastive signals during training data construction; the purposes differ, but the underlying philosophy is similar.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ CPMI formula is elegant; TD-$\lambda$ theoretical connection is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison of efficiency and quality; theory and experiments validate each other.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation and rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ 98% token reduction makes large-scale PRM training feasible, significantly impacting the reasoning enhancement field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences](c2_scalable_rubric-augmented_reward_modeling_from_binary_preferences.md)
- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)
- [\[ICML 2026\] GRPO is Secretly a Process Reward Model](../../ICML2026/llm_reasoning/grpo_is_secretly_a_process_reward_model.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](../../ICML2026/llm_reasoning/reward_modeling_from_natural_language_human_feedback.md)

</div>

<!-- RELATED:END -->
