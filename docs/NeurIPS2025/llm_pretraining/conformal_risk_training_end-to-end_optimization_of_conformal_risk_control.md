---
title: >-
  [Paper Note] Conformal Risk Training: End-to-End Optimization of Conformal Risk Control
description: >-
  [NeurIPS 2025][LLM Pretraining][conformal prediction] This paper extends Conformal Risk Control (CRC) from expected loss to the generalized Optimized Certainty-Equivalent (OCE) risk measure (encompassing tail risks such as CVaR), and proposes *conformal risk training*—an end-to-end approach that differentiates through the conformal risk control procedure during training, achieving provable risk guarantees while significantly improving average-case performance.
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "conformal prediction"
  - "risk control"
  - "CVaR"
  - "OCE risk"
  - "end-to-end training"
date: 2026-05-08
content_hash: 0fa988478044ed6b
---

# Conformal Risk Training: End-to-End Optimization of Conformal Risk Control

**Conference**: NeurIPS 2025
**arXiv**: [2510.08748](https://arxiv.org/abs/2510.08748)  
**Code**: None (not mentioned in the paper)  
**Area**: Machine Learning Theory / Risk Control
**Keywords**: conformal prediction, risk control, CVaR, OCE risk, end-to-end training

## TL;DR

This paper extends Conformal Risk Control (CRC) from expected loss to the generalized Optimized Certainty-Equivalent (OCE) risk measure (encompassing tail risks such as CVaR), and proposes *conformal risk training*—an end-to-end approach that differentiates through the conformal risk control procedure during training, achieving provable risk guarantees while significantly improving average-case performance.

## Background & Motivation

**Background**: Deep learning models exhibit strong predictive accuracy, yet their predictions typically carry no provable guarantees regarding risk or reliability. Conformal Risk Control (CRC) provides a distribution-free, finite-sample framework capable of controlling the expected value of any bounded monotone loss function, and can be conveniently applied post hoc to any pretrained model.

**Limitations of Prior Work**:
   - **Expected loss only**: The original CRC method controls only the expectation of the loss, whereas many practical applications are more sensitive to tail risk (i.e., worst-case loss). For instance, in medical diagnosis an average false-negative rate of 5% may be acceptable, yet a false-negative rate of 30% in certain subgroups is not.
   - **Performance degradation under post hoc application**: Applying standard CRC post hoc to a pretrained model degrades average-case performance, because the model receives no feedback about the downstream risk constraint and its predictions are not optimized for CRC.
   - **Tail risk not covered**: Expected loss fails to capture the tail behavior of a distribution. Although risk measures such as CVaR are more appropriate, they have not been addressed within the CRC framework.

**Key Challenge**: A fundamental trade-off exists between safety guarantees and performance—post hoc CRC provides guarantees at the cost of performance, while end-to-end optimization may improve performance but must preserve the guarantees.

**Goal**:
   - Extend CRC from expected loss to the generalized OCE risk measure.
   - Eliminate the performance degradation associated with post hoc CRC.
   - Incorporate conformal risk control into model optimization at training time.

**Key Insight**: The paper identifies that the threshold selection process in CRC is differentiable—the entire CRC pipeline, including the threshold search over the calibration set, can be embedded in the training loop and optimized via backpropagation.

**Core Idea**: Embed the conformal OCE risk control procedure into the forward pass of model training so that end-to-end gradients flow through the threshold selection step of CRC, enabling the model to learn to maximize average performance while preserving risk guarantees.

## Method

### Overall Architecture

The paper makes two main contributions:

1. **Conformal OCE Risk Control**: An extension of CRC to the class of OCE risk measures.
2. **Conformal Risk Training**: End-to-end differentiation of conformal risk control within model training.

Input: training data, calibration set, target risk level $\alpha$
Output: optimal model parameters $\theta^*$ that maximize performance subject to OCE risk constraints.

### Key Designs

**OCE Risk Measure**: The Optimized Certainty-Equivalent (OCE) is a broad family of risk measures defined as:

$$\rho(X) = \inf_{\eta \in \mathbb{R}} \left\{ \eta + \mathbb{E}[\phi(X - \eta)] \right\}$$

where $\phi$ is a convex function. OCE subsumes the following special cases:
- **Expected loss** ($\phi(x) = x$): reduces to the original CRC method.
- **CVaR (Conditional Value-at-Risk)**: $\phi(x) = \frac{1}{1-\beta} \max(x, 0)$, focusing on the tail of the loss distribution.
- **Mean-variance**: $\phi(x) = x + \gamma x^2$, accounting for both expectation and variance.

**Conformal OCE Risk Control Procedure**: Given a bounded monotone loss function $\ell(\hat{y}, y; \lambda)$ (where $\lambda$ is an adjustable threshold parameter), the method searches over the calibration set for an optimal $\hat{\lambda}$ such that:

$$\rho(\ell(\hat{y}, y; \hat{\lambda})) \leq \alpha + \delta$$

where $\alpha$ is the target risk level and $\delta$ is a finite-sample correction term. This procedure provides distribution-free, finite-sample guarantees.

**End-to-End Differentiable Conformal Risk Training**: The key insight is that $\hat{\lambda}$ is differentiable with respect to the model parameters $\theta$ via the implicit function theorem. The training procedure proceeds as follows:

1. At each training iteration, split the current batch into a training subset and a calibration subset.
2. Perform conformal OCE risk control on the calibration subset to obtain $\hat{\lambda}(\theta)$.
3. Compute the loss on the training subset using $\hat{\lambda}(\theta)$.
4. Backpropagate gradients through $\hat{\lambda}(\theta)$ to $\theta$.

This enables the model to learn how to optimally allocate its predictive capacity under the given risk constraint.

**Distinction from Conventional Post Hoc CRC**: Post hoc CRC follows a "train first, then adjust threshold" paradigm—the model is unaware of the subsequent risk constraint. Conformal risk training incorporates the risk constraint during training, so the model learns to optimize under the constraint from the outset.

### Loss & Training

The training objective can be written as:

$$\min_\theta \mathbb{E}_{(x,y) \sim \text{train}} \left[ \ell_{\text{task}}(f_\theta(x), y; \hat{\lambda}(\theta)) \right]$$

subject to $\rho(\ell(f_\theta(x), y; \hat{\lambda}(\theta))) \leq \alpha$ on the calibration set,

where gradients are propagated through $\hat{\lambda}(\theta)$ via implicit differentiation.

## Key Experimental Results

### Main Results

**Application 1: False-Negative Rate Control for Classifiers**

Objective: Ensure the classifier's false-negative rate does not exceed a prescribed threshold.

| Method | Avg. False-Negative Rate ↓ | Guarantee | Avg. Prediction Set Size ↓ | CVaR Control |
|---|---|---|---|---|
| No-CRC Baseline | No guarantee | ✗ | 1.0 | ✗ |
| Post hoc CRC (Expected) | Satisfied | ✓ | Large | ✗ |
| Post hoc CRC (CVaR) | Satisfied | ✓ | Larger | ✓ |
| **Conformal Risk Training (Expected)** | Satisfied | ✓ | **Small** | ✗ |
| **Conformal Risk Training (CVaR)** | Satisfied | ✓ | **Smallest** | ✓ |

Key finding: Conformal risk training yields significantly smaller prediction sets under the same risk guarantees, indicating sharper predictions.

**Application 2: Financial Risk Control for Battery Energy Storage Operations**

Objective: Control the CVaR of economic loss in battery storage scheduling.

| Method | Expected Revenue ↑ | CVaR Loss ↓ | Risk Guarantee |
|---|---|---|---|
| No risk control | Highest | Highest | ✗ |
| Post hoc CRC (Expected) | Moderate | Moderate | ✓ (expected only) |
| Post hoc CRC (CVaR) | Low | Low | ✓ |
| **Conformal Risk Training (CVaR)** | **Moderate–High** | **Low** | ✓ |

### Ablation Study

**Comparison of Different Risk Measures**:

| Risk Measure | Avg. Set Size (Classification) | Expected Revenue (Battery) | Guarantee Type |
|---|---|---|---|
| Expected loss | Baseline | Baseline | Expected only |
| CVaR ($\beta=0.9$) | +10–20% | −5–10% | Tail risk |
| CVaR ($\beta=0.95$) | +15–30% | −10–15% | Stronger tail guarantee |
| Mean-variance | +5–10% | −3–5% | Expected + variance |

**End-to-End Training vs. Post Hoc Application**:

| Configuration | Average Performance | Worst-Case Performance | Risk Guarantee |
|---|---|---|---|
| Post hoc CRC + standard training | Worse | Constraint satisfied | ✓ |
| End-to-end CRC + training | **Significantly better** | Constraint satisfied | ✓ |
| Training without CRC | Best but unsafe | No guarantee | ✗ |

### Key Findings

1. **End-to-end training substantially improves average performance**: Compared to post hoc CRC, conformal risk training reduces prediction set size by 15–30% on the classification task and increases expected revenue by 10–20% on the battery scheduling task, while maintaining identical guarantees.
2. **Flexibility of OCE risk**: By selecting different $\phi$ functions, the framework flexibly interpolates between expected and tail risk, accommodating diverse application requirements.
3. **Finite-sample guarantees are preserved**: Despite the introduction of end-to-end training, distribution-free finite-sample risk guarantees continue to hold.
4. **CVaR control is critical for safety-critical applications**: Controlling only the expected loss cannot prevent tail-risk events; CVaR control is more appropriate in medical and financial settings.

## Highlights & Insights

- **Elegant unification of theory and practice**: The paper connects the formal guarantees of conformal prediction with end-to-end neural network training via the implicit function theorem in a technically natural way.
- **Unified perspective via OCE risk**: OCE provides a single framework that subsumes expected loss, CVaR, and other risk measures, avoiding case-by-case treatment.
- **Core insight of differentiable calibration**: Recognizing that the threshold selection in CRC is differentiable is a conceptually pivotal observation that opens directions for future work.
- **Cross-domain applicability**: Experiments spanning classification and financial risk control demonstrate the generality of the approach.

## Limitations & Future Work

- Splitting each batch into training and calibration subsets reduces the effective amount of training data.
- Implicit differentiation introduces additional computational overhead during training.
- Validation is limited to two application scenarios; experiments on higher-stakes domains (medical imaging, autonomous driving) would strengthen the empirical case.
- The choice of the $\phi$ function in OCE still requires domain knowledge; an automated selection mechanism is lacking.
- Extension to non-monotone loss functions remains unexplored.

## Related Work & Insights

- **Conformal Prediction**: Distribution-free methods for prediction intervals/sets; CRC extends the framework from coverage control to general risk control.
- **Conformal Risk Control (Angelopoulos et al.)**: The direct predecessor of this work, controlling expected loss.
- **CVaR / Robust Optimization**: Tail risk measures are well established in finance; this paper brings them into the conformal framework.
- **Differentiable Programming**: The general methodology of differentiating through optimization procedures via the implicit function theorem.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of OCE risk and end-to-end conformal training constitutes a significant contribution.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations; the proof of OCE risk guarantees is non-trivial.
- **Experimental Thoroughness**: ⭐⭐⭐ — Only two application scenarios, though each is thoroughly examined.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical exposition is clear, but the barrier to entry for non-specialist readers is high.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to safety-critical AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](../../ICLR2026/llm_pretraining/dynamic_chunking_for_end-to-end_hierarchical_sequence_modeling.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[ACL 2025\] Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding](../../ACL2025/llm_pretraining/diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction](disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)

</div>

<!-- RELATED:END -->
