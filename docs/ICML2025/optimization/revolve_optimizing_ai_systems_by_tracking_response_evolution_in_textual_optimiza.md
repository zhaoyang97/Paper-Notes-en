---
title: >-
  [Paper Note] REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization
description: >-
  [ICML 2025][Optimization][Textual Optimization] REVOLVE guides optimization by tracking the "evolutionary" trends of responses across iterations in LLM systems. It is more stable and efficient than immediate-feedback-based methods like TextGrad, improving prompt optimization, solution refinement, and code optimization by 7.8%, 20.72%, and 29.17% respectively.
tags:
  - "ICML 2025"
  - "Optimization"
  - "Textual Optimization"
  - "LLM System Optimization"
  - "Response Evolution"
  - "TextGrad"
  - "Prompt Engineering"
date: 2026-05-08
content_hash: b06c95237fbe2153
---

# REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization

**Conference**: ICML 2025  
**arXiv**: [2412.03092](https://arxiv.org/abs/2412.03092)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Textual Optimization, LLM System Optimization, Response Evolution, TextGrad, Prompt Engineering

## TL;DR
REVOLVE guides optimization by tracking the "evolutionary" trends of responses across iterations in LLM systems. It is more stable and efficient than immediate-feedback-based methods like TextGrad, improving prompt optimization, solution refinement, and code optimization by 7.8%, 20.72%, and 29.17% respectively.

## Background & Motivation

### Background

**Background**: Background: LLM systems (comprising prompt + tool interaction) perform exceptionally well in complex tasks, but optimization for specific tasks still relies on manual prompt engineering and hyperparameter tuning.

**Limitations of Prior Work**: Automatic optimization methods such as TextGrad leverage LLM textual feedback as "gradients" for iterative refinement. However, they only focus on immediate feedback (analogous to considering only the current derivative in numerical gradient descent), which easily leads to stagnation or oscillation when adjustments are too small or unstable.

**Key Challenge**: Immediate feedback vs. trend awareness—only looking at the improvement direction of the current step is insufficient for making globally optimal adjustment decisions.

**Key Insight**: Analogous to using historical information like momentum/accelerated gradients in numerical optimization, this work introduces cross-iteration response evolution history to guide optimization.

**Core Idea**: During each optimization step, rather than providing only current feedback, information on how responses evolve over time is also supplied, enabling the LLM to make more forward-looking, progressive adjustments.

### Goal

**Goal**: ### Overall Architecture
Input: LLM system to be optimized (prompt / solution / code) → multiple iterations → record response + evaluation at each step → aggregate evolutionary trend → LLM generates refinement suggestions based on evolutionary history → update system → until convergence.


## Method

### Overall Architecture
Input: LLM system to be optimized (prompt / solution / code) → multiple iterations → record response + evaluation at each step → aggregate evolutionary trend → LLM generates refinement suggestions based on evolutionary history → update system → until convergence.

### Key Designs

1. **Response Evolution Tracking**:

    - Record the full response and score of each iteration
    - Construct evolutionary trajectory: $(r_1, s_1) \to (r_2, s_2) \to ... \to (r_t, s_t)$
    - Provide the optimizer LLM with not only the current refinement direction but also the complete evolutionary trend
    - Design Motivation: Analogous to the momentum method, using historical information to avoid short-sighted optimization

2. **Adaptive Adjustment Strategy**:

    - When improvement slows down: increase adjustment step size (analogous to accelerated gradient)
    - When oscillation occurs: stabilize the optimization direction
    - When continuous improvement is made: maintain the current direction
    - Design Motivation: Apply different strategies for different optimization phases

3. **General Framework Application**:

    - Prompt Optimization: Refine the system prompt to improve task accuracy
    - Solution Refinement: Iteratively optimize natural language or structured solutions
    - Code Optimization: Progressively debug and improve code implementations
    - Design Motivation: A unified textual optimization paradigm applicable to various LLM system scenarios

### Loss & Training
- No traditional loss function; instead, task metrics are used as evaluation signals.
- The LLM acts as the optimizer, generating textual "gradients" based on the evolutionary history.

## Key Experimental Results

### Main Results

| Task Type | Metric | REVOLVE | TextGrad | Gain |
|----------|------|---------|----------|------|
| Prompt Optimization | Accuracy | Best | Baseline | +7.8% |
| Solution Refinement | Quality Score | Best | Baseline | +20.72% |
| Code Optimization | Pass Rate | Best | Baseline | +29.17% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full REVOLVE | Best | Evolution Tracking + Adaptive Adjustment |
| W/o Evolution Tracking | Decreased | Degrades to a TextGrad-like approach |
| Only last 2 steps | Partial Decrease | History is too short to determine the trend |
| Fixed Adjustment Strategy | Decreased | Lacks adaptivity |

### Key Findings
- REVOLVE converges in fewer iterations, saving computational cost.
- Evolution trend information is crucial for preventing optimization stagnation and oscillations.
- The gain is most significant (+29.17%) on code optimization tasks, where code modification requires a more stable direction.

## Highlights & Insights
- **Interdisciplinary Analogy**: Migrating momentum and accelerated gradient concepts from numerical optimization to text-based optimization.
- **Simple Yet Effective**: No modifications to the underlying LLMs are required—only the information structuring of the optimizer prompt is updated.
- **High Generality**: The same framework is seamlessly applicable to various optimization scenarios (prompt/solution/code).

## Limitations & Future Work
- The growth of historical information may eventually exceed the LLM context window.
- The quality of evaluation signals directly impacts optimization performance.
- Requires more LLM calls (for recording and analyzing the evolutionary history).

## Related Work & Insights
- TextGrad (Yuksekgonul et al. 2024) is the direct predecessor.
- OPRO (Yang et al. 2024) utilizes LLMs for prompt optimization.
- Insight: Many strategies in classical optimization theory can be "translated" into LLM textual optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear paradigm in migration of optimization theory insights into textual optimization
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive verification across three scenarios
- Writing Quality: ⭐⭐⭐⭐ Easy-to-understand narrative that draws analogies to traditional optimization
- Value: ⭐⭐⭐⭐ A highly practical optimization tool for LLM systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](../../NeurIPS2025/optimization/preference_learning_with_response_time_robust_losses_and_guarantees.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](../../CVPR2026/optimization/enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](../../NeurIPS2025/optimization/deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](../../NeurIPS2025/optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](../../ICML2026/optimization/distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)

</div>

<!-- RELATED:END -->
