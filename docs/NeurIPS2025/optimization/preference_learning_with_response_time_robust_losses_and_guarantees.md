---
title: >-
  [Paper Note] Preference Learning with Response Time: Robust Losses and Guarantees
description: >-
  [NeurIPS 2025][Optimization][Preference learning] This paper incorporates response time information from user decision-making into the preference learning framework, reducing the estimation error of reward model learning from exponential to polynomial order via a Neyman orthogonal loss function.
tags:
  - NeurIPS 2025
  - Optimization
  - Preference learning
  - response time
  - reward model
  - Neyman orthogonality
  - robust loss
date: 2026-05-08
content_hash: 10d5fc57309855b0
---

# Preference Learning with Response Time: Robust Losses and Guarantees

**Conference**: NeurIPS 2025

**arXiv**: [2505.22820](https://arxiv.org/abs/2505.22820)

**Code**: None

**Area**: Optimization / Preference Learning

**Keywords**: Preference learning, response time, reward model, Neyman orthogonality, robust loss

## TL;DR

This paper incorporates response time information from user decision-making into the preference learning framework, reducing the estimation error of reward model learning from exponential to polynomial order via a Neyman orthogonal loss function.

## Background & Motivation

### Root Cause

**State of the Field**: Binary preference data has become a central signal for fine-tuning foundation models (e.g., RLHF), yet temporal information embedded in the user decision process—such as response/reaction time—is typically discarded. Intuitively, a rapid choice (e.g., apple vs. orange) suggests stronger preference, while hesitation implies that the two options differ only marginally.

**Limitations of Prior Work**:

- **Information waste**: The standard Bradley-Terry model uses only binary choice outcomes, discarding metadata such as response time.
- **Exponential error growth**: For linear reward functions, the estimation error of conventional preference learning grows exponentially with reward magnitude.
- **Low sample efficiency**: Treating all samples equally when preference strength varies significantly leads to poor utilization of available data.

This paper builds on the Evidence Accumulation Drift Diffusion (EZ) model to infer preference strength from response time, and designs robust loss functions with theoretical guarantees.

## Method

### Overall Architecture

Preference learning is formulated as the problem of learning a reward function $r(x)$ from joint observations of binary choices and response times. The EZ drift-diffusion model is used to connect response time with preference strength.

### Key Designs

**1. EZ Drift-Diffusion Model**

- Assumes the decision-maker accumulates evidence between two options until a decision boundary is reached.
- The drift rate $v$ is proportional to the reward difference $r(x_1) - r(x_2)$.
- The expected response time satisfies $\mathbb{E}[T] = f(v)$, where $f$ is a known decreasing function.
- Response time thus encodes information about preference strength.

**2. Neyman Orthogonal Loss Function**

- A doubly robust loss function is designed such that the estimator is insensitive to estimation errors in nuisance parameters (e.g., the conditional expectation of response time).
- Key formulation: $\mathcal{L}(\theta) = \ell_{\text{BT}}(\theta) + \lambda \cdot \psi(T, \theta)$
- Here $\ell_{\text{BT}}$ is the standard Bradley-Terry loss and $\psi$ is a correction term based on response time.
- The Neyman orthogonality condition is satisfied, achieving oracle convergence rates.

**3. Improved Theoretical Error Bounds**

- Conventional methods: error $\sim \exp(O(\|r\|))$, exponential dependence on reward magnitude.
- Proposed method: error $\sim \text{poly}(\|r\|)$, polynomial dependence — a significant improvement.

### Loss & Training

Training proceeds in two stages:
1. **Stage 1**: Estimate nuisance parameters (conditional expectation of response time) using nonparametric methods.
2. **Stage 2**: Optimize reward model parameters using the Neyman orthogonal loss.

## Key Experimental Results

### Main Results

Estimation error comparison (MSE) under the linear reward function setting:

| Method | $\|r\|=1$ | $\|r\|=2$ | $\|r\|=5$ | $\|r\|=10$ |
|--------|-----------|-----------|-----------|------------|
| BT-MLE | 0.08 | 0.35 | 4.72 | 89.3 |
| Weighted-BT | 0.06 | 0.22 | 2.15 | 41.6 |
| Ours (Neyman Orthogonal) | **0.05** | **0.12** | **0.48** | **2.31** |

Image preference learning experiment (Accuracy):

| Method | Validation Acc | Test Acc | Training Size |
|--------|---------------|----------|---------------|
| Standard BT | 72.3% | 70.1% | 10K |
| Weighted BT | 74.1% | 72.5% | 10K |
| Neyman Orthogonal (Ours) | **76.8%** | **75.2%** | 10K |
| Standard BT | 76.5% | 74.8% | 50K |

### Ablation Study

Effect of nuisance estimation quality on final performance:

| Nuisance Estimation Quality | BT + RT (Weighted) | Neyman Orthogonal (Ours) |
|-----------------------------|--------------------|--------------------------|
| Exact (oracle) | 0.10 | 0.05 |
| Good | 0.18 | 0.06 |
| Moderate | 0.32 | 0.08 |
| Poor | 0.65 | 0.15 |

### Key Findings

1. As reward magnitude increases, the error of conventional methods grows exponentially, whereas the proposed method exhibits only polynomial growth.
2. Neyman orthogonality endows the method with robustness to errors in nuisance parameter estimation.
3. Convergence guarantees are established in nonparametric reward spaces as well.
4. Response time information is especially valuable for samples where preference differences are subtle.

## Highlights & Insights

- **Theoretical elegance**: The paper introduces Neyman orthogonality from semiparametric statistics into preference learning, providing a solid theoretical foundation.
- **Exponential to polynomial**: The improvement in error bounds represents a qualitative leap rather than a constant-factor gain.
- **Practical relevance**: Response time is easy to collect in deployed systems, making the method immediately applicable.

## Limitations & Future Work

1. The EZ model assumptions may not hold in certain real-world scenarios (e.g., multitasking users).
2. Experiments are primarily conducted on image preference tasks and synthetic data; validation on LLM alignment settings is lacking.
3. Response time may be confounded by external factors such as UI design and device heterogeneity.
4. Computational efficiency in the nonparametric setting warrants further investigation.

## Related Work & Insights

- **Bradley-Terry Model**: A classical approach to modeling preference strength.
- **EZ Drift Diffusion**: The standard cognitive science model for decision-making processes.
- **Semiparametric Statistics**: Inspiration from the Neyman orthogonality and Double Machine Learning (DML) frameworks.

## Rating

- ⭐ Novelty: 9/10 — Elegantly integrates response time with preference learning theory.
- ⭐ Value: 7/10 — Strong theoretical contributions, but insufficient experiments in LLM settings.
- ⭐ Writing Quality: 8/10 — Rigorous theoretical derivations and well-designed experiments.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees](mess_dynamically_learned_inference-time_llm_routing_in_model_zoos_with_service_l.md)
- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](verbalized_algorithms.md)

<!-- RELATED:END -->
