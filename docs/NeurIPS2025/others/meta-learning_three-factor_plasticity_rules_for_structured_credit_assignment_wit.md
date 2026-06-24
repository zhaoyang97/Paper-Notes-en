---
title: >-
  [Paper Note] Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback
description: >-
  [NeurIPS 2025 (Workshop: Symmetry and Geometry in Neural Representations)][Synaptic plasticity] This paper proposes a meta-learning framework that automatically discovers local neo-Hebbian synaptic plasticity rules via outer-loop gradient optimization, enabling recurrent neural networks to perform structured credit assignment using only sparse, delayed reward signals, thereby providing new insights into the learning mechanisms of biological neural networks.
tags:
  - "NeurIPS 2025 (Workshop: Symmetry and Geometry in Neural Representations)"
  - "Synaptic plasticity"
  - "three-factor learning rules"
  - "meta-learning"
  - "recurrent neural networks"
  - "credit assignment"
date: 2026-05-08
content_hash: 53bcb1ecb0dae7d1
---

# Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback

**Conference**: NeurIPS 2025 (Workshop: Symmetry and Geometry in Neural Representations)  
**arXiv**: [2512.09366](https://arxiv.org/abs/2512.09366)  
**Code**: None  
**Area**: Computational Neuroscience / Meta-Learning / Biologically Plausible Learning Rules  
**Keywords**: Synaptic plasticity, three-factor learning rules, meta-learning, recurrent neural networks, credit assignment

## TL;DR

This paper proposes a meta-learning framework that automatically discovers local neo-Hebbian synaptic plasticity rules via outer-loop gradient optimization, enabling recurrent neural networks to perform structured credit assignment using only sparse, delayed reward signals, thereby providing new insights into the learning mechanisms of biological neural networks.

## Background & Motivation

Biological brains can learn complex behaviors from sparse, delayed feedback signals, yet the underlying synaptic plasticity mechanisms remain poorly understood. Experimental evidence suggests that synaptic changes depend on the co-activation of pre- and postsynaptic neurons as well as potentially other local variables. However, most training methods for artificial recurrent networks (e.g., BPTT) are biologically implausible, requiring symmetric forward and backward connections and non-local information.

Two key problems exist:

**Limitations of hand-crafted rules**: Prior work relies predominantly on manually designed synaptic update rules, leaving the design space largely unexplored.

**Biological implausibility of BPTT**: Standard training methods require continuous error signals to iteratively optimize connection weights, whereas biological systems typically receive only sparse reward feedback at the end of a task.

The authors ask: can meta-optimization automatically discover local synaptic plasticity rules capable of supporting structured credit assignment under sparse feedback?

## Method

### Overall Architecture

The framework adopts a two-level nested training structure:
- **Inner loop**: A recurrent neural network is trained across multiple episodes using local plasticity rules (three-factor rules), receiving sparse rewards only at the end of each episode.
- **Outer loop**: Plasticity parameters are optimized via gradient descent using forward-mode differentiation (tangent-propagation).

### Key Designs

1. **Network dynamics**: A firing-rate neuron model is employed, coupled via synaptic matrix $\mathbf{W} \in \mathbb{R}^{N \times N}$, together with input matrix $\mathbf{W}_{in}$ and output matrix $\mathbf{W}_{out}$. The dynamics are governed by:
    $\frac{d\mathbf{x}^t}{dt} = -\mathbf{x}^t + \mathbf{W}\phi(\mathbf{x}^t) + \mathbf{W}_{in}\mathbf{u}^t$
   where $\phi(\cdot) = \tanh(\cdot)$ is the single-neuron transfer function.

2. **Parameterized eligibility traces**: Each synapse maintains an eligibility trace $e_{ij}$ whose evolution is governed by a polynomial function:
    $\frac{de_{ij}^t}{dt} = \sum_{0 \leq k,\ell \leq d} \theta_{k,\ell} (r_j^t)^k (\bar{x}_i - x_i^t)^\ell - \frac{e_{ij}^t}{\tau_e}$
   Here $\theta_{k,\ell}$ are learnable coefficients and $d=5$ is the polynomial degree. Unlike conventional eligibility traces based solely on first-order correlations, the polynomial parameterization captures richer pre- and postsynaptic activity interactions.

3. **Three-factor synaptic update rule**: The weight matrix $\mathbf{W}$ is updated at the end of each episode according to a reward-modulated rule:
    $[\boldsymbol{\mu}_\Theta^{(h)}]_{ij} = \eta \cdot e_{ij}^{T_h} \cdot (R^{(h)} - \bar{R}^{(h)})$
   The three factors are: presynaptic activity, postsynaptic activity (encoded via the eligibility trace), and the reward prediction error (the third factor).

### Loss & Training

- **Outer-loop optimization**: A REINFORCE estimator is used to approximate meta-gradients, avoiding expensive backpropagation through the learning dynamics.
- **Tangent-propagation**: Forward-mode differentiation propagates sensitivities across trials to compute gradients of plasticity parameters with respect to the mean weight update.
- Three sensitivity variables (state tangent vector, trace tangent vector, eligibility trace tangent vector) are defined, propagated forward within each trial and accumulated across trials via the weight matrix tangent vector $\mathbf{U}_{k,\ell}^{(h)}$.
- Gradient validation: finite differences and forward-mode differentiation are compared over 500 trials, yielding highly consistent results.

## Key Experimental Results

### Gradient Validation

| Validation Method | Comparison | Result |
|---------|---------|------|
| Per-trial gradient | FM vs. FD (trials 1, 250, 500) | Highly consistent |
| Cumulative gradient | Accumulated over 500 trials | FM provides accurate estimates |
| Relative error | Per-trial relative gradient error | Negligible (~$10^{-5}$ order) |

### Dynamical Analysis Tools

| Analysis Dimension | Method | Purpose |
|---------|------|------|
| Fixed-point localization | Damped Newton method (200 random initializations) | Identify network steady states |
| Stability analysis | Jacobian eigenvalue analysis | Determine fixed-point stability |
| Non-normality | Henrici index $\|\mathbf{J}\|_F^2 - \sum|\lambda_i|^2$ | Quantify transient amplification |
| Readout alignment | Overlap between output vectors and eigenvectors | Identify which modes influence output |
| Input sensitivity | Linear response $\mathbf{p} = (-\mathbf{J})^{-1}\mathbf{W}_{in}$ | Quantify per-neuron input sensitivity |

### Key Findings

1. **Forward-mode differentiation is efficient and accurate**: Gradient computations over 500 trials match numerical finite differences closely, validating the correctness of tangent-propagation.
2. **Different plasticity rules yield qualitatively distinct representations and dynamics**: Rules discovered by meta-learning naturally give rise to qualitatively different learning trajectories and internal representations.
3. **Advantages of polynomial eligibility traces**: Richer pre- and postsynaptic interaction patterns are captured compared to conventional first-order eligibility traces; the signs of the coefficients encode Hebbian or anti-Hebbian directions.

## Highlights & Insights

- **Bottom-up methodology**: Rather than hand-crafting rules, the framework allows rules to emerge automatically, exploring a broad design space of local plasticity rules.
- **Biological plausibility**: The entire framework relies only on local information (pre- and postsynaptic activity) and delayed reward signals, requiring no biologically implausible backpropagation.
- **Elegant use of forward-mode differentiation**: The computational burden of backpropagating through hundreds of trials is avoided, making meta-learning tractable for long-horizon credit assignment.
- **Polynomial parameterization design**: A flexible function family is provided for eligibility traces, with each coefficient independently controlling the Hebbian or anti-Hebbian direction.

## Limitations & Future Work

- As a workshop paper, experimental validation is relatively limited, lacking systematic evaluation on complex cognitive tasks.
- Only the recurrent weight matrix $\mathbf{W}$ is optimized; input and output weights are not updated by plasticity rules.
- No direct performance comparison with other biologically plausible learning algorithms (e.g., e-prop, RFLO) is provided.
- The choice of polynomial degree $d=5$ lacks systematic ablation.
- Scalability to larger networks and more complex tasks has not been verified.
- Whether the meta-learned rules genuinely reflect plasticity mechanisms in biological brains requires further validation.

## Related Work & Insights

- This work extends the meta-learning plasticity framework of Confavreux et al. (2023) to the sparse feedback setting.
- It contrasts with the hand-crafted reward-modulated learning rules of Miconi (2017), replacing manual design with meta-learning.
- Dynamical systems tools such as fixed-point analysis and Jacobian eigendecomposition provide a rich perspective for understanding networks after learning.
- Insight: meta-learning can be applied not only to model parameters but also to the learning rules themselves, which is meaningful both for understanding biological systems and for designing novel learning algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐ (The idea of meta-learning plasticity rules is creative, though the framework is not entirely novel)
- Experimental Thoroughness: ⭐⭐⭐ (Limited experiments as a workshop paper; task-level performance benchmarks are absent)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical derivations are clear and rigorous)
- Value: ⭐⭐⭐⭐ (Provides a valuable methodological contribution to the computational neuroscience and meta-learning communities)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Plasticity as the Mirror of Empowerment](plasticity_as_the_mirror_of_empowerment.md)
- [\[ACL 2025\] Detecting Sockpuppetry on Wikipedia Using Meta-Learning](../../ACL2025/others/detecting_sockpuppetry_on_wikipedia_using_meta-learning.md)
- [\[ACL 2025\] Meta-Learning Neural Mechanisms rather than Bayesian Priors](../../ACL2025/others/meta-learning_neural_mechanisms_rather_than_bayesian_priors.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](../../ACL2025/others/learning_to_reason_from_feedback_at_test-time.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](../../ICCV2025/others/is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)

</div>

<!-- RELATED:END -->
