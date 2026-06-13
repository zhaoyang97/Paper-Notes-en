---
title: >-
  [Paper Note] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks
description: >-
  [NeurIPS 2025][LLM Pretraining][Feature Learning] This paper proposes the Alternating Gradient Flow (AGF) theoretical framework to explain the stepwise "saddle-to-saddle" feature learning dynamics in neural networks. Tra…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Feature Learning"
  - "Saddle-point Dynamics"
  - "Alternating Gradient Flows"
  - "Dormant Neurons"
  - "Two-layer Networks"
date: 2026-05-08
content_hash: c32004279d62193a
---

# Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2506.06489](https://arxiv.org/abs/2506.06489)  
**Code**: [https://github.com/danielkunin/alternating-gradient-flows](https://github.com/danielkunin/alternating-gradient-flows)  
**Area**: Deep Learning Theory
**Keywords**: Feature Learning, Saddle-point Dynamics, Alternating Gradient Flows, Dormant Neurons, Two-layer Networks

## TL;DR
This paper proposes the Alternating Gradient Flow (AGF) theoretical framework to explain the stepwise "saddle-to-saddle" feature learning dynamics in neural networks. Training is modeled as an alternating process between utility maximization for dormant neurons and cost minimization for active neurons, unifying feature selection analysis across diagonal linear networks, attention models, and modular addition. Predictions from AGF exhibit high agreement with actual gradient flow behavior.

## Background & Motivation

### State of the Field

**Background**: "Saddle-to-saddle" dynamics have been empirically observed in neural network training—the loss stagnates near saddle points and then drops sharply, repeating in alternation. This suggests that networks learn distinct features incrementally.

**Limitations of Prior Work**: (a) NTK theory neglects feature learning—linear approximations do not explain saddle escape; (b) mean-field theory is too complex to yield clear intuition; (c) a unified mathematical framework for feature learning across diverse architectures (linear networks, Transformers, MLPs) remains lacking.

**Key Challenge**: The incremental nature of feature learning (one feature at a time) appears to contradict the continuity of gradient descent (all parameters updated simultaneously)—why does continuous optimization produce discrete learning steps?

**Goal**: To provide a unified mathematical framework explaining stepwise feature learning across different architectures.

**Key Insight**: Neurons are categorized as either "dormant" (weights near zero) or "active" (weights large). Training proceeds as alternating optimization between these two groups—dormant neurons search for the most valuable new features (utility maximization), while active neurons adjust weights to minimize residuals (cost minimization).

**Core Idea**: Utility maximization for dormant neurons (directional dynamics, finding feature directions) + cost minimization for active neurons (radial dynamics, adjusting weight magnitudes) = stepwise saddle-to-saddle feature learning.

## Method

### Overall Architecture
The utility function $\mathcal{U}_i(\theta; r) = E_x[\langle f_i(x;\theta), r(x) \rangle]$ measures neuron $i$'s geometric contribution to residual $r(x)$. **Directional dynamics**: $\frac{d\theta_i}{dt} = \eta\|\theta_i\|^{\kappa-2} P^\perp_{\theta_i} \nabla \mathcal{U}_i$—searches for the highest-utility direction on the hypersphere. **Radial dynamics**: $\frac{d\|\theta_i\|}{dt} = \eta\kappa\|\theta_i\|^{\kappa-1} \mathcal{U}_i$—weight magnitude grows when utility is positive and shrinks when negative. **Jump time**: $\tau_i = \inf\{t > 0 | \mathcal{S}_i(t) > c_i/\eta\}$—a dormant neuron activates when its accumulated utility exceeds a threshold.

### Key Designs

1. **Alternating Optimization Framework**:

    - Function: Explains stepwise feature learning.
    - Mechanism: At a saddle point, active neurons have converged while dormant neurons slowly accumulate utility. When a dormant neuron's accumulated utility exceeds a threshold, it abruptly "awakens" to learn a new feature → a new saddle point → the process repeats.
    - Design Motivation: Explains why continuous gradient descent produces discrete learning steps—the threshold mechanism creates a phase transition from dormant to active states.

2. **Universality of the Utility Function**:

    - Function: Covers diverse architectures within a single framework.
    - Mechanism: Diagonal linear networks, quadratic networks, and attention layers all admit a utility function—the contribution of a new neuron to the current residual. The functional form differs across architectures, but the framework is the same.
    - Design Motivation: Unified treatment: linear networks → greedy low-rank learning; modular addition → Fourier frequencies learned in descending order of amplitude.

3. **Convergence Proof (Theorem 3.1)**:

    - Function: Proves that AGF converges exactly to gradient flow as $\alpha \to 0$.
    - Mechanism: AGF is an alternating approximation of gradient flow; it converges as the alternating step size tends to zero.
    - Design Motivation: Provides theoretical guarantees that AGF is not a heuristic but a rigorous limiting case of gradient flow.

### Loss & Training
- Purely theoretical work with experimental validation.
- Experimental verification on diagonal linear networks and modular addition (GSM8K).

## Key Experimental Results

### Main Results

| Setting | AGF Prediction | Actual Gradient Flow | Agreement |
|---------|---------------|---------------------|-----------|
| Diagonal linear network | Greedy low-rank learning | Exact match | Excellent |
| Modular addition (quadratic network) | Fourier frequencies learned in descending order | Match | Excellent |
| GSM8K (Llama 3.1%) | Staircase loss descent | Match | Good |

### Key Findings
- AGF predictions and gradient flow agree exactly on diagonal linear networks (Figure 3).
- In modular addition, networks indeed learn Fourier frequencies in descending order of amplitude—AGF accurately predicts the learning sequence.
- The framework unifies several previously independent analyses of linear networks and attention layers.

## Highlights & Insights
- **Dormant/active alternation as a unified learning paradigm**: Saddle-to-saddle dynamics across different architectures can all be explained within this framework.
- **Utility maximization explains feature selection**: New neurons do not activate randomly; they select the direction that maximally reduces the current residual.
- **From theory to practice**: Loss curves from Llama 3.1 on GSM8K also conform to the predictions—the framework is not limited to toy models.

## Limitations & Future Work
- Restricted to two-layer networks—analysis for deeper networks remains to be developed.
- On natural data (CIFAR-10), the staircase structure smooths out as network width increases rather than remaining discrete.
- Many proof details for complex settings are deferred to the appendix.

## Related Work & Insights
- **vs. NTK theory**: NTK ignores feature learning; AGF specifically analyzes *how* features are learned.
- **vs. Mean-field theory**: Mean-field theory is too complex; AGF provides a more intuitive alternating optimization perspective.
- **vs. Incremental learning**: The stepwise feature learning in AGF has a natural connection to incremental and curriculum learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Alternating gradient flow is an entirely new theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across diagonal networks, quadratic networks, attention layers, and Llama.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical intuition is clear and figures are excellent.
- Value: ⭐⭐⭐⭐⭐ Unifies feature learning theory across multiple architectures; a significant advance in deep learning theory.

### Supplementary Notes on the Method
- **Intuitive interpretation of the utility function**: $\mathcal{U}_i(\theta; r) = E_x[\langle f_i(x;\theta), r(x) \rangle]$ measures "how much the current residual would be reduced by activating neuron $i$ in direction $\theta$"—analogous to greedy feature selection.
- **Physical interpretation of jump time $\tau_i$**: A dormant neuron must accumulate sufficient "evidence" (utility integral exceeding threshold $c_i/\eta$) before activating—analogous to the action potential threshold in neuroscience.
- **Separation of directional vs. radial dynamics**: Directional dynamics search for the optimal direction on the hypersphere ($P^\perp_\theta \nabla \mathcal{U}$—projected onto the orthogonal complement of $\theta$); radial dynamics govern weight magnitude (positive utility → growth, negative utility → decay).
- **Connection to the Lottery Ticket Hypothesis**: LTH posits the existence of good subnetworks; AGF explains what "good" means—high-utility neurons are automatically selected.
- **Implications for Transformer training**: AGF predicts that attention layers also learn features incrementally (in descending frequency order), consistent with the staircase loss observed for Llama 3.1 on GSM8K—theory aligns with practice.
- **Refined limitations**: On natural data (CIFAR-10), the staircase structure smooths out as width increases—because the feature space is more continuous, discrete jumps are smoothed away. This suggests AGF is an exact description for narrow networks and an approximation for wide networks.
- **Connection to information theory**: Utility maximization is analogous to information gain—the "usefulness" of a new feature can be measured by mutual information.
- **Implications for regularization**: The automatic compression in AGF can be viewed as implicit regularization.
- **Natural connection to curriculum learning**: AGF predicts that networks learn simple features before complex ones—consistent with the easy-before-hard principle.
- **Potential applications**: Neural architecture search for automatically determining optimal depth; transfer learning with more compressible representations; federated learning with fewer active layers to reduce communication.
- **Future directions**: Extending AGF to residual networks beyond the two-layer case requires handling cascading effects among intermediate saddle points.
- **Potential connection to scaling laws**: The stepwise feature learning in AGF may be the microscopic mechanism underlying the sudden emergence of capabilities observed in scaling laws.
- **Implications for Transformer training**: Attention layers also learn features incrementally, consistent with the staircase loss of Llama 3.1 on GSM8K.
- **Clarification of theoretical limits**: AGF is an exact description for narrow networks and an approximation for wide networks—because wider networks have a more continuous feature space in which discrete jumps are smoothed out.
- **Unification with the information-theoretic perspective**: Utility maximization is analogous to information gain; the usefulness of new features can be quantified via mutual information.
- **Implications for deep learning curriculum design**: Understanding saddle-to-saddle dynamics can inform better learning rate scheduling strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)

</div>

<!-- RELATED:END -->
