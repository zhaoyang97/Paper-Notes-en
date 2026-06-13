---
title: >-
  [Paper Note] REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation
description: >-
  [ICML 2026][LLM Evaluation][LLM-as-a-Judge] Addressing the fundamental flaw where standard RL uses binary 0/1 rewards for numerical scoring tasks, effectively ignoring the ordinal structure of scores…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "LLM-as-a-Judge"
  - "Regression-Aware Reward"
  - "Generalized Policy Gradient"
  - "Policy-Dependent Reward"
  - "Correlation Optimization"
date: 2026-05-08
content_hash: e8260b0739a4b27f
---

# REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation

**Conference**: ICML 2026  
**arXiv**: [2603.17145](https://arxiv.org/abs/2603.17145)  
**Code**: https://github.com/YasminZhang/REAL (Available)  
**Area**: Reinforcement Learning / LLM Post-training / LLM Evaluation  
**Keywords**: LLM-as-a-Judge, Regression-Aware Reward, Generalized Policy Gradient, Policy-Dependent Reward, Correlation Optimization  

## TL;DR
Addressing the fundamental flaw where standard RL uses binary 0/1 rewards for numerical scoring tasks, effectively ignoring the ordinal structure of scores, the authors integrate the "expected value prediction + squared error" objective from RAFT into the RL framework. Because the reward explicitly depends on policy parameters, a Generalized Policy Gradient is derived, which cleanly decomposes into a "CoT Exploration term" and a "Prediction Refinement term." Evaluated across 8B–32B base models, REAL consistently outperforms SFT and standard RL. On Qwen3-32B, it achieves a Gain of 8.4 Pearson and 7.2 Spearman points over SFT.

## Background & Motivation
**Background**: LLM-as-a-Judge has become a central component of evaluation, alignment, and preference modeling, where models output numerical scores representing "quality/correctness/preference strength." Mainstream training approaches include: (1) SFT (e.g., Prometheus), which treats scores as discrete tokens using cross-entropy; (2) Regression-aware SFT (e.g., RAFT/TRACT), which combines "expected value prediction $\hat y_\theta(x, c) = \sum_{k \in \mathcal{K}} k \cdot \pi_\theta(k|x,c)$" with squared error to recover ordinal structures.

**Limitations of Prior Work**: Extending regression-aware objectives to RL post-training is the logical next step, as RL enables the model to **actively explore** its own CoT trajectories, whereas SFT is limited to imitating fixed ground-truth reasoning chains. However, current RL frameworks (PPO/GRPO/DPO) rely on rule-based verifiers providing binary rewards $r = \mathbf{1}(y = y^*)$. This is disastrous for regression: if the ground truth is 5, a prediction of 4 and a prediction of 1 are treated as equally incorrect by standard RL, despite 4 being significantly closer. Fig. 2 empirically demonstrates that standard RL from a TRACT checkpoint causes correlation metrics to collapse.

**Key Challenge**: To ensure RL utilizes the "exploration of CoT space" while acknowledging that "distance between scores matters," regression rewards must be used. However, the regression reward $r = -(\hat y_\theta - y^*)^2$ makes the reward $\hat y_\theta$ explicitly dependent on policy parameters $\theta$. This violates the $\nabla_\theta r = 0$ premise of the standard REINFORCE derivation, rendering standard policy gradient formulas incorrect.

**Goal**: (1) Provide a formalized framework for integrating regression rewards into RL; (2) Theoretically link this to correlation metrics (Pearson/Spearman), as these are the primary downstream evaluation metrics for LLM-as-a-Judge; (3) Verify improvements in OOD generalization across model scales.

**Key Insight**: Utilize the **Generalized Policy Gradient Estimator** (Schulman 2015) to explicitly handle the unconventional setting where rewards depend on parameters.

**Core Idea**: By leveraging the mathematical fact that the Generalized Policy Gradient decomposes into exploration and refinement terms, regression-awareness is elegantly embedded in RL. Furthermore, it is theoretically proven that minimizing squared error is equivalent to optimizing Pearson correlation.

## Method
### Overall Architecture
The input consists of $(x, y^*)$ evaluation pairs, where $x$ is the "prompt + response" and $y^* \in \mathcal{K} = \{0, 1, \dots, 9\}$ is the label. The policy $\pi_\theta$ autoregressively generates a CoT $c$ followed by a score $y$. Instead of sampling $y$, REAL uses the **RAIL expected value predictor** $\hat y_\theta(x, c) = \sum_{k \in \mathcal{K}} k \cdot \pi_\theta(k | x, c)$ to collapse the distribution into a continuous expectation for squared error calculation. During training, $K$ CoT trajectories $\{c_i\}$ are sampled for each $x$ to estimate the advantage via RLOO, updating the policy with a two-term gradient.

### Key Designs
1.  **REAL Objective and Implicit Policy-Dependent Reward**:
    -   **Function**: Port the regression loss from SFT (RAFT) to RL, changing the CoT source from ground-truth to policy-sampled.
    -   **Core Idea**: The objective function is defined as $\mathcal{L}_{\text{REAL}}(\theta) = \mathbb{E}_{(x, y^*) \sim \mathcal{D}, c \sim \pi_\theta(\cdot | x)}[(\hat y_\theta(x, c) - y^*)^2 - \lambda \log \pi_\theta(y^* | x, c)]$. The first term is the squared error, and the second is an auxiliary NTP loss on the final answer token. The implicit reward $r_{\text{REAL}}(\theta, x, c) = -(\hat y_\theta(x, c) - y^*)^2 + \lambda \log \pi_\theta(y^* | c, x)$ **explicitly depends on $\theta$**.
    -   **Design Motivation**: Unlike TRACT, which uses a fixed $\pi_{\text{temp}}$ to sample CoTs, REAL uses the current policy $\pi_\theta$, allowing the CoT and reward to co-evolve. The RAIL predictor is crucial because it incorporates the shape of the 0–9 distribution into the gradient, which is far denser than single-token probability info.

2.  **Natural Decomposition of Generalized Policy Gradient**:
    -   **Function**: Provide a mathematically sound gradient estimator for parameter-dependent rewards.
    -   **Core Idea**: Using the generalized policy gradient lemma (Schulman 2015) to expand the chain rule on $\mathcal{L}(\theta) = \mathbb{E}_{x, c \sim \pi_\theta}[r(\theta, x, c)]$ yields $\nabla_\theta \mathcal{L} = \mathbb{E}[\underbrace{r(\theta, x, c) \nabla_\theta \log \pi_\theta(c | x)}_{\text{Term 1: CoT Update}} + \underbrace{\nabla_\theta r(\theta, x, c)}_{\text{Term 2: Prediction Refinement}}]$. Term 2 expands into $-2(\hat y_\theta - y^*) \nabla_\theta \hat y_\theta + \lambda \nabla_\theta \log \pi_\theta(y^* | x, c)$, where $\nabla_\theta \hat y_\theta = \sum_k k \cdot \nabla_\theta \pi_\theta(k | x, c)$.
    -   **Design Motivation**: This decomposition is elegant: Term 1 treats CoT $c$ as an "action" for exploration (policy gradient style), while Term 2 treats the score $y$ as "known ground truth" for backpropagation correction (backprop style). Unlike GRPO, which treats $c$ and $y$ as homogeneous tokens, REAL **explicitly acknowledges their structural differences**: CoTs are high-dimensional sequences requiring exploration, while final answers are low-cardinality discrete variables suitable for direct regression.

3.  **RLOO Stabilization and $\beta$ Weighting**:
    -   **Function**: Transform the theoretical gradient into a stable engineering objective.
    -   **Core Idea**: For each $x$, sample $K$ CoT trajectories and calculate advantage $A^{(i)} = r^{(i)} - \frac{1}{K-1}\sum_{j \ne i} r^{(j)}$. The stabilized gradient is $\nabla \mathcal{L} \approx \frac{1}{K} \sum_i [\tilde A^{(i)} \nabla_\theta \log \pi_\theta(c_i | x) + \beta \nabla_\theta r_{\text{REAL}}(\theta, x, c_i)]$, where $\beta$ controls the strength of the refinement term.
    -   **Design Motivation**: $\beta$ is a hyperparameter where $\beta = 1.0$ is the theoretically rigorous value. Experiments show $\beta = 1.0$ is sufficient; it serves as an interface for future refinement/exploration tuning.

### Loss & Training
$\mathcal{L}_{\text{REAL}}(\theta) = \mathbb{E}_{(x, y^*), c \sim \pi_\theta}[(\hat y_\theta(x, c) - y^*)^2 - \lambda \log \pi_\theta(y^* | x, c)]$, utilized with an RLOO estimator and $\beta = 1.0$. $K$ follows medium-scale GRPO conventions.

## Key Experimental Results

### Main Results (Selected from Table 2, Mistral2-7B and Qwen3-32B; Metrics ×100)

| Model | Method | Paradigm | Inference | FB Bench (r/ρ) | FLASK (r/ρ) | Vic. Bench (r/ρ) | MT Bench (r/ρ) | Avg r | Avg ρ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Mistral2-7B | RAFT | SFT | RAIL | 87.9 / 88.0 | 41.8 / 41.9 | 52.8 / 51.3 | 39.9 / 41.8 | 55.6 | 55.8 |
| Mistral2-7B | Standard RL | RL | RAIL | 93.7 / 93.7 | 51.6 / 50.5 | 58.0 / 56.0 | 52.9 / 50.7 | 64.1 | 62.7 |
| Mistral2-7B | **Ours (REAL)** | RL | RAIL | 93.2 / 93.4 | **56.0 / 54.1** | **63.3 / 60.2** | **59.3 / 56.9** | **67.9** | **66.2** |
| Qwen3-32B | RAFT | SFT | RAIL | 85.4 / 86.5 | 52.1 / 52.9 | 51.9 / 52.0 | 61.1 / 59.6 | 62.6 | 62.8 |
| Qwen3-32B | **Ours (REAL)** | RL | RAIL | **91.1 / 91.7** | **58.9 / 58.6** | **65.1 / 60.7** | **68.9 / 69.1** | **71.0** | **70.0** |

On the in-domain FB Bench, REAL is competitive with standard RL, but on OOD benchmarks (FLASK, Vicuna, MT), REAL wins by 4–8 points. This indicates that regression rewards enhance generalization rather than just memorizing training patterns.

### Ablation Study

| Config | Key Change | Observation |
| :--- | :--- | :--- |
| Full REAL | RL + Regression Reward + Dual Gradient | Consistent OOD SOTA |
| w/o Term 1 (≈ TRACT) | Degrades to SFT refinement | Loses CoT exploration, OOD drops 3–5 points |
| w/o Term 2 (Standard RL w/ MSE) | No prediction gradient | Correlation metrics collapse during training |
| $\lambda = 0$ | No NTP auxiliary | Performance stable, verifying regression as main driver |

### Key Findings
-   **OOD > In-domain**: REAL’s primary strength is OOD generalization. Binary rewards "memorize" correctness within the training distribution but fail to learn the underlying "score distance" structure.
-   **Metric Collapse in Standard RL**: Standard RL actively degrades Pearson/Spearman coefficients when starting from a pre-trained regression checkpoint (Fig. 2), highlighting the negative effect of binary rewards on regression tasks.
-   **Term 2 Dominance**: Term 2 (Prediction Refinement) handles the bulk of the optimization. Removing it makes the task unlearnable, whereas removing Term 1 retains some performance (comparable to TRACT) but sacrifices OOD robustness.
-   **Lemma 3.1**: Minimizing squared error at the sample level is mathematically equivalent to optimizing Pearson correlation at the group level—bridging the gap between the training objective and the evaluation metric.

## Highlights & Insights
-   **Generalized Policy Gradient** elegantly resolves the "forbidden zone" of parameter-dependent rewards. This opens doors for optimizing any differentiable metric (entropy, calibration, confidence) directly in RL.
-   **Structure-Aware Optimization**: By treating final numerical scores as regression targets while treating CoTs as actions, REAL respects the task's structural constraints—CoTs require high-variance exploration, while answers allow low-variance backprop.
-   **Theoretical Alignment**: The finding that $\beta=1.0$ works optimally without tuning suggests that the formalization correctly captures the underlying mathematics of the problem.

## Limitations & Future Work
-   **Output Scope**: Limited to single scalar numerical outputs $\{0, ..., 9\}$. Multi-dimensional rubrics (e.g., Prometheus) or free-form text justifications require extensions.
-   **Semantic Calibration**: REAL does not supervise CoT content quality. Whether CoTs could degenerate into placeholders under a pure regression objective remains an open question.
-   **Computational Overhead**: Calculating $\nabla_\theta \hat y_\theta$ for $K$ trajectories requires backpropagating through all digit tokens, potentially increasing training costs relative to standard RL.

## Related Work & Insights
-   **vs TRACT**: TRACT uses SFT on self-generated CoTs. REAL is the generalization of TRACT into RL; TRACT is effectively the "Term 2-only" version of REAL.
-   **vs Standard RL**: While standard RL collapses variance by treating all "close misses" as 0, REAL uses RAIL expectations to preserve ordinal density—a paradigm shift from classification to regression in RL feedback.
-   **vs JEPO**: JEPO tackles non-verifiable $y^*$ using marginal log-likelihood. REAL focuses on ordered numerical scores and outperforms JEPO in regression-specific metrics (Table 14).

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ 
-   Experimental Thoroughness: ⭐⭐⭐⭐ 
-   Writing Quality: ⭐⭐⭐⭐⭐ 
-   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IF-Critic: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation](../../ACL2026/llm_evaluation/if-critic_towards_a_fine-grained_llm_critic_for_instruction-following_evaluation.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](../../ICLR2026/llm_evaluation/unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [\[ICML 2026\] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)
- [\[ICML 2026\] Resolution Diagnostics for Paired LLM Evaluation](resolution_diagnostics_for_paired_llm_evaluation.md)
- [\[ACL 2026\] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](../../ACL2026/llm_evaluation/rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)

</div>

<!-- RELATED:END -->
