---
title: >-
  [Paper Note] Efficient Process Reward Modeling via Contrastive Mutual Information
description: >-
  [ACL 2026][LLM Reasoning][Process Reward Model] This paper proposes CPMI (Contrastive Pointwise Mutual Information), an efficient automatic step-level reward annotation method that estimates step-wise contributions by co…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Process Reward Model"
  - "Step-level Supervision"
  - "Mutual Information"
  - "Contrastive Learning"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 619584b29141e152
---

# Efficient Process Reward Modeling via Contrastive Mutual Information

**Conference**: ACL 2026
**arXiv**: [2604.10660](https://arxiv.org/abs/2604.10660)  
**Code**: [GitHub](https://github.com/nakyungLee20/CPMI)  
**Area**: LLM Reasoning
**Keywords**: Process Reward Model, Step-level Supervision, Mutual Information, Contrastive Learning, Mathematical Reasoning

## TL;DR
This paper proposes CPMI (Contrastive Pointwise Mutual Information), an efficient automatic step-level reward annotation method that estimates step-wise contributions by contrasting the conditional probability shifts a reasoning step induces on correct versus incorrect answers. Compared to Monte Carlo estimation, CPMI reduces construction time by 84% and token generation by 98%, while achieving higher accuracy on both process-level evaluation benchmarks and mathematical reasoning benchmarks.

## Background & Motivation

**State of the Field**: Process Reward Models (PRMs) verify Chain-of-Thought trajectories by evaluating the correctness of intermediate reasoning steps, proving more reliable than Outcome Reward Models (ORMs) that assess only final answers. However, training PRMs requires step-level annotated data—traditionally provided by human annotators or high-capability LLMs.

**Limitations of Prior Work**: (1) Manual step-level reward annotation is prohibitively costly and time-consuming; (2) Automated methods such as Monte Carlo estimation require extensive LLM rollouts to obtain low-variance reward signals, incurring substantial computational overhead—dozens of trajectories must be sampled per step to estimate correctness rates; (3) MC estimation is particularly unstable for early steps in a reasoning chain, where short prefixes lead to high variance in subsequent completions.

**Root Cause**: A fundamental gap exists between the acquisition cost of step-level supervision signals and the training demands of PRMs—high-quality annotations are required in large quantities yet remain extremely expensive to obtain.

**Paper Goals**: Design a method capable of estimating step-level rewards via a single forward pass, eliminating the dependency on repeated MC rollouts.

**Starting Point**: The relationship between MC estimation (λ→1, long-horizon return) and the proposed method (λ→0, single-step bootstrapping) is framed through the lens of TD(λ). The paper hypothesizes that pretrained LLMs already encode sufficient mathematical knowledge, such that the contribution of a reasoning step can be inferred by observing how the model's probability over the correct answer changes upon incorporating that step.

**Core Idea**: Step-level reward = (increase in log-probability of the correct answer after adding the step) − (increase in log-probability of incorrect answers after adding the step), i.e., a contrastive pointwise mutual information score.

## Method

### Overall Architecture
For each reasoning step: (1) compute the difference in the model's log-probability of the correct answer with and without the step; (2) compute the analogous difference for incorrect answers; (3) subtract the latter from the former to obtain the CPMI reward. Normalized CPMI scores are then used as soft labels to train the PRM. At inference time, the PRM scores candidate trajectories to select the optimal one.

### Key Designs

1. **CPMI Reward Formula**:

    - **Function**: Estimates step-level contribution via a single forward pass.
    - **Mechanism**: $r_{\text{CPMI}}^i = [\log p_\theta(A|q,s_i) - \log p_\theta(A|q)] - \frac{1}{M}\sum_{m=1}^M [\log p_\theta(\tilde{A}|q,s_i) - \log p_\theta(\tilde{A}|q)]$. The first term quantifies the step's contribution to increasing the probability of the correct answer; the second term quantifies its suppression of incorrect answer probabilities. Effective steps should simultaneously increase the probability of correct answers and decrease that of incorrect ones.
    - **Design Motivation**: Plain PMI (considering only the correct answer) lacks discriminative power. Introducing a contrastive signal—contrasting probability shifts for correct versus incorrect answers—substantially improves the reward signal's discriminability.

2. **Theoretical Connection: CPMI ≈ Jeffreys Divergence**:

    - **Function**: Provides a theoretical grounding for the CPMI reward.
    - **Mechanism**: Under the single-correct-answer assumption common in mathematical reasoning, CPMI can be interpreted as an approximation of the Jeffreys divergence (symmetric KL divergence) between the conditional answer distributions with and without the reasoning step. This implies that CPMI favors steps that induce large, symmetric shifts in the answer distribution.
    - **Design Motivation**: The symmetry of Jeffreys divergence ensures bidirectional penalization—detecting not only increases in correct-answer probability but also decreases in incorrect-answer probability.

3. **CPMI-Merge Hybrid Strategy**:

    - **Function**: Addresses the high noise of CPMI at early steps in the reasoning chain.
    - **Mechanism**: MC estimation (capturing global information) is applied to initial steps (e.g., step 1), while CPMI (local bootstrapping) is used for subsequent steps. This exploits the complementary strengths of MC and CPMI—MC captures global information but is costly; CPMI provides dense feedback but is unstable at early steps.
    - **Design Motivation**: From the TD-λ perspective, this approach finds the optimal balance between λ=1 (MC) and λ=0 (CPMI).

### Loss & Training
Qwen3-4B-Base serves as the PRM backbone with a two-layer linear reward head appended. The model is trained with BCE loss, using z-score-normalized CPMI rewards as soft labels. At inference time, candidate trajectories are scored and ranked by the PRM to select the optimal output.

## Key Experimental Results

### Main Results (Efficiency + Quality)

| Reward Type | AUC | PB | PRMB | MATH | Time (ratio) | Tokens (ratio) |
|-------------|-----|----|------|------|--------------|----------------|
| MC | 0.759 | 27.7 | 38.8 | 45.4 | 1.00 | 1.00 |
| PAV | 0.757 | 36.6 | 49.6 | 47.2 | 1.17 | 2.38 |
| **CPMI** | **0.765** | 34.6 | **58.8** | 48.2 | **0.16 (↓84%)** | **0.02 (↓98%)** |
| **CPMI_Merge** | **0.766** | **36.8** | **60.7** | **49.4** | 0.30 (↓70%) | 0.18 (↓82%) |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| No contrast (PMI only) | AUC decreases; insufficient discriminative power |
| No prompt averaging | Increased reward variance |
| Varying M (number of negative samples) | M=4 achieves optimal balance |
| CPMI-Merge (step 1) | More stable than pure CPMI |

### Key Findings
- **CPMI reduces construction time by 84% and token generation by 98%** while achieving higher quality (AUC 0.765 vs. MC 0.759).
- **CPMI substantially outperforms MC on the process-level benchmark PRMB (58.8 vs. 38.8)**, indicating that CPMI-derived step-level signals are more effective for process-level verification.
- **The contrastive signal is critical**: removing the contrastive term (using PMI alone) leads to a significant performance drop.
- **CPMI-Merge further improves stability**: it eliminates noise at early steps while preserving most of the efficiency gains.
- **RelEff (relative efficiency ratio)** reaches 6–10×, demonstrating that CPMI substantially dominates MC on the quality–cost trade-off.

## Highlights & Insights
- **Unifying MC and CPMI under the TD-λ framework** constitutes an elegant theoretical contribution—MC corresponds to λ→1 (full return) and CPMI to λ→0 (bootstrapping)—applying a classical reinforcement learning framework to PRM training.
- **The 98% reduction in token generation** renders large-scale PRM dataset construction practically feasible, transforming a task that previously required GPU clusters running for days into one completable on a single machine within hours.
- **Theoretical guarantees for the CPMI reward** (via the Jeffreys divergence approximation) elevate it beyond a heuristic, endowing it with principled justification.

## Limitations & Future Work
- CPMI relies on the quality of the pretrained LLM's internal probability distribution—if the model's mathematical knowledge is insufficient, probability estimates may be unreliable.
- Validation is limited to mathematical reasoning tasks; effectiveness on other domains requiring PRMs—such as code generation and logical reasoning—remains to be confirmed.
- The hard negative construction strategy (M=4 with heuristic perturbations) may lack systematicity.
- The instability of CPMI at early reasoning steps necessitates CPMI-Merge as a remedy, adding design complexity.
- The single-correct-answer assumption may not hold in certain tasks.

## Related Work & Insights
- **vs. MC Estimation (Math-Shepherd)**: MC requires sampling dozens of complete trajectories per step; CPMI requires only a single forward pass.
- **vs. PAV (Setlur et al.)**: PAV still relies on MC rollouts; CPMI eliminates the need for rollouts entirely.
- **vs. Contrastive Decoding (Li et al.)**: Contrastive decoding manipulates logits at inference time, whereas CPMI applies contrastive signals during training data construction; the use cases differ, but the underlying philosophy is shared.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The CPMI formulation is elegant, and the theoretical connection to TD-λ is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison of efficiency and quality; theory and experiments mutually validate each other.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and experimental design is rigorous.
- Value: ⭐⭐⭐⭐⭐ The 98% token reduction makes large-scale PRM training practically viable, with significant implications for reasoning-enhanced systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](../../ICLR2026/llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](../../ICLR2026/llm_reasoning/drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)

</div>

<!-- RELATED:END -->
