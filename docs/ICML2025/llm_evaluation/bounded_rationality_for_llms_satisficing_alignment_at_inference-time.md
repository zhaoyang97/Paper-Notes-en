---
title: >-
  [Paper Note] Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time
description: >-
  [ICML 2025][LLM Evaluation][Inference-time alignment] This paper proposes SITAlign, a satisficing alignment framework based on bounded rationality, which maximizes the primary objective (e.g., helpfulness) at inference time while ensuring secondary objectives (e.g., harmlessness) satisfy threshold constraints. Solved through duality theory, it achieves a 22.3% win rate improvement over state-of-the-art multi-objective decoding on GPT-4 evaluation.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "Inference-time alignment"
  - "Bounded rationality"
  - "Satisficing decision-making"
  - "Multi-objective alignment"
  - "Controlled decoding"
date: 2026-05-08
content_hash: 7327c283afbfa1ad
---

# Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time

**Conference**: ICML 2025  
**arXiv**: [2505.23729](https://arxiv.org/abs/2505.23729)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Inference-time alignment, Bounded rationality, Satisficing decision-making, Multi-objective alignment, Controlled decoding

## TL;DR
This paper proposes SITAlign, a satisficing alignment framework based on bounded rationality, which maximizes the primary objective (e.g., helpfulness) at inference time while ensuring secondary objectives (e.g., harmlessness) satisfy threshold constraints. Solved through duality theory, it achieves a 22.3% win rate improvement over state-of-the-art multi-objective decoding on GPT-4 evaluation.

## Background & Motivation

**Background**: LLM alignment requires simultaneously satisfying multiple attributes (safety, helpfulness, truthfulness, conciseness). Existing methods typically model this as a multi-objective weighted optimization.

**Limitations of Prior Work**: Weighted combinations assume that all objective sub-dimensions should be maximized simultaneously. However, determining weights is difficult, and this approach ignores actual human decision-making. Herbert Simon's theory of bounded rationality indicates that humans use a "satisficing" strategy: optimizing the primary objective while ensuring other objectives simply meet a satisficing threshold.

**Key Challenge**: Multi-objective weighted optimization struggles to handle conflicting objectives; furthermore, fine-tuning methods are computationally expensive and cannot adapt to personalized thresholds of different users or scenarios.

**Goal**: How to dynamically achieve satisficing alignment at inference time to "maximize the primary objective while keeping secondary objectives within satisficing thresholds"?

**Key Insight**: Model alignment as a constrained optimization problem, where the primary objective is the objective function and the secondary objectives act as inequality constraints.

**Core Idea**: Convert the constrained problem into an unconstrained Lagrangian problem using duality theory, solving it token-by-token within a controlled decoding framework.

## Method

### Overall Architecture
SITAlign generates tokens sequentially at inference time:
1. Given a primary reward model $r_1$ and secondary reward models $r_2, \dots, r_m$, along with thresholds $\beta_2, \dots, \beta_m$.
2. Objective: $\max_\pi \mathbb{E}[Q_1^*(s_t, z)] - \beta_1 D_{KL}[\pi || \pi_{sft}]$ s.t. $\mathbb{E}[Q_i^*(s_t, z)] \geq \beta_i$.
3. Absorb constraints through Lagrangian duality and dynamically adjust the token distribution.

### Key Designs

1. **Satisficing Alignment Modeling**:

    - Function: Decomposes multi-dimensional preferences into "maximizing the primary objective + meeting secondary objective thresholds".
    - Mechanism: Inspired by bounded rationality—once harmlessness exceeds a certain threshold, the marginal utility of further improvement diminishes; thus, it is better to concentrate resources on helpfulness.
    - Design Motivation: Experiments confirm that 90% of harmless responses have reward scores $\geq -12$ (PKU-SafeRLHF); thus, setting a threshold suffices.

2. **Controlled Decoding via Duality**:

    - Function: Transforms the constrained optimization into a Lagrangian dual problem.
    - Mechanism: Introduces Lagrangian multipliers $\lambda_i$, combining constraints into a weighted decoding objective of $r_1 + \sum \lambda_i r_i$, where $\lambda_i$ is dynamically updated during decoding.
    - Design Motivation: Duality methods convert constraint satisfaction into adaptive weight adjustment, avoiding manual weight tuning.

### Loss & Training
- A fully inference-time method, requiring no fine-tuning.
- Multiplier updates utilize subgradient ascent.
- The theory provides suboptimality bounds.

## Key Experimental Results

### Main Results

| Setup | Method | GPT-4 Win Rate (Primary Obj) | Constraint Satisfaction |
|------|------|-------------------|---------|
| Helpfulness↑ + Harmlessness≥Threshold | Multi-obj decoding | 35.2% | ✓ |
| | **SITAlign** | **57.5%** (+22.3%) | ✓ |
| Helpfulness↑ + Humor≥Threshold | Multi-obj decoding | 41.1% | ✓ |
| | **SITAlign** | **51.3%** (+10.2%) | ✓ |

### Ablation Study

| Configuration | Primary Obj Win Rate | Constraint Satisfaction Rate | Description |
|------|-----------|-----------|------|
| Primary Obj Only (No Constraints) | High | Low (~60%) | Safety ignored |
| Equal-Weight Multi-Objective | Medium | Medium | Hard to tune weights |
| SITAlign | High | High (>95%) | Optimal trade-off |

### Key Findings
- SITAlign consistently outperforms multi-objective weighted methods on the primary objective while keeping the constraint satisfaction rate above 95%.
- The satisficing decision paradigm aligns better with actual requirements than weighted optimization.
- The inference-time method allows dynamic adjustment of thresholds without retraining.

## Highlights & Insights
- The integration of the **bounded rationality perspective** with LLM alignment is highly natural—not all dimensions need to be optimal, meeting a threshold is sufficient.
- Flexibility of the inference-time method: different users can set different thresholds without retraining.
- The duality method automatically learns the "tightness" of constraints, avoiding manual weight tuning.

## Limitations & Future Work
- Predefined thresholds are required; although the paper provides guidelines, domain knowledge is still needed.
- The quality of the Q-function estimation affects performance (requires reasonable value function approximation).
- Token-by-token optimization at inference time increases latency.
- Evaluation was conducted only on 7B models.

## Related Work & Insights
- **vs Shi et al. (Multi-Objective Decoding)**: Weighted combinations of all objectives without distinguishing priority.
- **vs ARGS/DeAL**: Single-objective controlled decoding, fails to handle multi-dimensional constraints.
- **vs ConfPO**: ConfPO is a training-time method, whereas SITAlign is an inference-time method, making them complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ An innovative perspective combining bounded rationality and LLM alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three evaluation setups, GPT-4 evaluation.
- Writing Quality: ⭐⭐⭐⭐ Compelling motivation, backed by both theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Proposes a new paradigm for alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](../../ICLR2026/llm_evaluation/guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](../../AAAI2026/llm_evaluation/optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](../../ECCV2024/llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ICML 2025\] Unlocking Post-hoc Dataset Inference with Synthetic Data](unlocking_post-hoc_dataset_inference_with_synthetic_data.md)

</div>

<!-- RELATED:END -->
