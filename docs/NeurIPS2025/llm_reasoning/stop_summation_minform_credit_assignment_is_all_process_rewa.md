---
title: >-
  [Paper Note] Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning
description: >-
  [NeurIPS 2025][LLM Reasoning][Process Reward Model] PURE identifies the root cause of reward hacking induced by PRMs as the standard sum-form credit assignment in RL ($V(s) = \sum \gamma^t r_t$)…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Process Reward Model"
  - "Credit Assignment"
  - "Reward Hacking"
  - "Min-Form"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: ff66c3309ba82568
---

# Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2504.15275](https://arxiv.org/abs/2504.15275)
**Code**: [github.com/CJReinforce/PURE](https://github.com/CJReinforce/PURE)
**Area**: LLM Reasoning
**Keywords**: Process Reward Model, Credit Assignment, Reward Hacking, Min-Form, Reinforcement Learning

## TL;DR
PURE identifies the root cause of reward hacking induced by PRMs as the standard sum-form credit assignment in RL ($V(s) = \sum \gamma^t r_t$), and proposes a min-form alternative ($V(s) = \min_{t' \geq t} r_{t'}$). By constraining the value function to the minimum of future rewards rather than their cumulative sum, PURE significantly mitigates reward hacking—achieving reasoning performance comparable to rule-based reward methods using only 30% of the training steps.

## Background & Motivation

**Background**: PRMs have proven effective for test-time scaling (e.g., Best-of-N), yet applying PRMs to RL fine-tuning frequently leads to reward hacking—the model learns to exploit high-scoring steps as rated by the PRM rather than genuinely improving reasoning quality.

**Limitations of Prior Work**: Standard RL credit assignment defines $V(s_t) = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$. When the PRM assigns inaccurately high scores to certain steps, cumulative summation amplifies these errors, causing the model to favor sequences that trigger high-reward steps regardless of reasoning correctness → training collapse.

**Key Challenge**: Sum-form accumulation allows a single high-reward step to inflate the value of an entire trajectory, yet no PRM can assign perfect rewards at every step → reward hacking is nearly inevitable.

**Goal**: How to design credit assignment so that PRMs can be safely used for RL fine-tuning?

**Key Insight**: Define the value function as the **minimum** of future rewards rather than their **cumulative sum**—requiring the model to ensure no individual step is too poor in order to achieve high value, rather than succeeding on only a few steps.

**Core Idea**: Replace $V(s_t) = \sum \gamma^{t'-t} r_{t'}$ with $V(s_t) = \min_{t' \geq t} r_{t'}$, eliminating the disproportionate inflation of overall value caused by any single high-reward step.

## Method

### Overall Architecture
PURE (Process sUpervised Reinforcement lEarning) retains the standard PPO/GRPO framework. The sole core modification is the credit assignment: advantages are computed via the min-form rather than the standard gamma-weighted sum. PURE is compatible with any PRM.

### Key Designs

1. **Min-Form Credit Assignment**:

    - **Function**: Defines the value of each token as the minimum reward over all subsequent steps.
    - **Mechanism**: $V(s_t) = \min_{t' \geq t} r_{t'}$, with advantage $A_t = V(s_t) - V(s_{t-1})$. The weakest-link effect—value is determined by the worst step.
    - **Design Motivation**: Under sum-form, the model can compensate for poor steps with a few high-reward ones; under min-form, a single poor step degrades overall value, compelling the model to improve quality at every step uniformly.

2. **Optional Rule-Based Reward Supplement (10%)**:

    - **Function**: Augments PRM supervision with a small proportion of verifiable outcome rewards.
    - **Mechanism**: Rule-verified correctness rewards are used as anchors for 10% of training samples to further suppress reward hacking.
    - **Design Motivation**: Since PRMs are inherently imperfect, a small quantity of rule-based rewards provides ground-truth calibration.

### Loss & Training
Standard RL framework with min-form advantage estimation. Validated on three base models including Qwen2.5-Math-7B. Training requires only 30% of the steps needed by standard methods.

## Key Experimental Results

### Main Results

| Method | AMC23 | Avg. (5 Benchmarks) | Training Steps | Notes |
|--------|-------|---------------------|----------------|-------|
| Sum-form + PRM | Collapse | Collapse | — | Collapses at the start of training |
| **Min-form + PRM (PURE)** | ~75% | ~50% | **30%** | Only 30% of steps required |
| Rule-based reward | ~75% | ~50% | 100% | Standard method |
| **PURE + 10% rule-based reward** | **82.5%** | **53.3%** | 100% | Best model |

### Key Findings
- **Sum-form + PRM causes immediate training collapse**: Standard RL credit assignment combined with a PRM collapses completely at the early stage of training, confirming the severity of reward hacking.
- **Min-form achieves comparable performance with only 30% of training steps**: The dense reward signal from the PRM accelerates learning; combined with min-form, hacking no longer occurs.
- **PURE + 10% rule-based reward yields the strongest model**: AMC23 82.5%, average 53.3% across 5 benchmarks—the best result reported on Qwen2.5-Math-7B.

## Highlights & Insights
- **First clear diagnosis of the root cause of PRM + RL = reward hacking**: The culprit is not PRM inaccuracy per se, but the amplification of minor PRM errors by sum-form credit assignment. This insight carries broad implications for the field.
- **The "weakest-link" intuition behind min-form is elegantly simple**: Value is determined by the bottleneck step, compelling the model to improve every step uniformly.

## Limitations & Future Work
- Min-form may be overly conservative: a single abnormally low-scoring step penalizes the value of the entire trajectory.
- Validation is currently limited to mathematical reasoning tasks.
- PRM quality remains important—min-form mitigates but does not fully eliminate reward hacking.

## Related Work & Insights
- **vs. PRIME**: PRIME addresses reward hacking by co-training the PRM via implicit reward; PURE resolves the issue fundamentally from the credit assignment perspective. The two approaches are orthogonal and can be combined.
- **vs. Rule-based reward RL**: PURE demonstrates that PRMs can be safely used in RL and that 30% of training steps suffice—the dense signal from PRMs is genuinely beneficial.
- **vs. GPO/VinePPO**: These methods refine the granularity of credit assignment (step-level vs. token-level); PURE modifies the mathematical form of credit assignment (min vs. sum).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Min-form credit assignment is an entirely novel paradigm; the analysis diagnosing the root cause of reward hacking is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three base models + five benchmarks + reward hacking case studies.
- Writing Quality: ⭐⭐⭐⭐ Problem framing is clear, though notation could be made more concise.
- Value: ⭐⭐⭐⭐⭐ Resolves the central obstacle to using PRMs in RL, enabling dense process rewards to finally be applied safely.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[NeurIPS 2025\] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)
- [\[NeurIPS 2025\] Know What You Don't Know: Uncertainty Calibration of Process Reward Models](know_what_you_dont_know_uncertainty_calibration_of_process_reward_models.md)
- [\[NeurIPS 2025\] ARM: Adaptive Reasoning Model](arm_adaptive_reasoning_model.md)

</div>

<!-- RELATED:END -->
