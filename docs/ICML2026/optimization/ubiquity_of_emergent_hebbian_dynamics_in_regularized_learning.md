---
title: >-
  [Paper Note] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] This paper demonstrates that near the steady state of L2 weight decay, the learning signals of **nearly any** learning rule (including SGD, Adam, DFA, and even Random Networks) **spontaneously** align toward the Hebbian direction. Conversely, sufficiently strong noise flips this alignment toward an anti-Hebbian directi
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: c0f2013825d12271
---
# Ubiquity of Emergent Hebbian Dynamics in Regularized Learning

**Conference**: ICML 2026  
**arXiv**: [2505.18069](https://arxiv.org/abs/2505.18069)  
**Code**: None  
**Area**: Optimization Theory / Biological Plasticity Modeling  
**Keywords**: Hebbian Learning, Weight Decay, Learning Signal Alignment, Noise-Regularization Phase Diagram, Neural Plasticity

## TL;DR
This paper demonstrates that near the steady state of L2 weight decay, the learning signals of **nearly any** learning rule (including SGD, Adam, DFA, and even Random Networks) **spontaneously** align toward the Hebbian direction. Conversely, sufficiently strong noise flips this alignment toward an anti-Hebbian direction, with a clear phase transition boundary emerging at $\gamma \propto \sigma^2$.

## Background & Motivation

**Background**: Classical neuroscience regards Hebbian and anti-Hebbian plasticity as core mechanisms of brain learning—"fire together, wire together" is achieved via local homosynaptic rules (e.g., STDP), regulated by homeostatic constraints (e.g., Oja's rule) to prevent weight divergence. In machine learning, deep networks are almost exclusively trained using gradient descent with weight decay and stochastic perturbations, which appears fundamentally different from biological mechanisms.

**Limitations of Prior Work**: Historically, the observation of Hebbian/anti-Hebbian structures in synaptic updates has often been used as "counter-evidence" to prove that the brain cannot be performing global error-driven optimization. This has created a rigid boundary between the two learning paradigms.

**Key Challenge**: Can the mere observation of "Hebbian-form updates" conclude that the underlying computation is Hebbian? In other words, is the Hebbian signature *identifiable*? Prior Work by Ziyin et al. (2025b) suggested that weight decay relates to Hebbian alignment but offered only scattered observations without a unified mechanism or an explanation for when anti-Hebbian dynamics emerge.

**Goal**: (1) Provide a *generic* near-steady-state mechanism explaining why L2 weight decay forces learning signals to project in the Hebbian direction; (2) Provide a dual mechanism explaining why noise flips it to anti-Hebbian; (3) Perform empirical validation across various architectures, optimizers, and tasks.

**Key Insight**: The authors note that in biology, Hebbian forces are **expansive**, requiring homeostatic mechanisms for **contraction**. In deep learning, learning signals must also be expansive to resist the contraction of weight decay. **Once a system enters a near-steady state, these opposing forces must balance**, imposing geometric constraints on the direction of learning signals.

**Core Idea**: By viewing weight decay as a "universal anti-Hebbian term," the steady-state condition $\mathbb{E}_x[g(x,\theta)] \approx \gamma W$ automatically forces the expected learning signal to have the same sign as the Hebbian direction $\bar H(W)=\mathbb{E}_x[h_b h_a^\top]$; noise introduces an additional quadratic term that reverses this inequality.

## Method

This paper is a theoretical and simulation-based study rather than a new algorithm. The "Method" section details the mathematical derivations of the two mechanisms and the experimental paradigms.

### Overall Architecture

Consider a hidden layer $h_b = W h_a(x)$, where $h_a$ is the post-activation of the previous layer and $h_b$ is the pre-activation of the current layer. Updates with weight decay are written as:

$$\Delta W \propto \underbrace{-(\nabla_{h_b}\ell)\, h_a^\top}_{\text{Learning Signal}} - \gamma W$$

Define the Hebbian direction as $\Delta_{\rm Hebb} W = h_b h_a^\top$ and the anti-Hebbian direction as $-h_b h_a^\top$. The core metric is the **alignment** between the "learning signal" and the Hebbian update, characterized by the Frobenius inner product or cosine similarity. The argument consists of two parts: Section 3 provides two theorems for positive alignment, and Section 4 provides the phase diagram where noise flips alignment to negative.

### Key Designs

**1. Hebbian Alignment Theorem at Steady State (Strong Form): Alignment as a Monotonic Function of Weight Decay**

The first step proves that standard gradient learning signals indeed align with the Hebbian direction near steady state, with alignment increasing monotonically with weight decay. Starting from the steady-state constraint $\mathbb{E}_x[(\nabla_{h_b}\ell) h_a^\top] + \gamma W \approx 0$, right-multiplying both sides by $W^\top$ and substituting $h_b = W h_a$ yields $\mathbb{E}_x[(\nabla_{h_b}\ell) h_b^\top] = -\gamma W W^\top$. Taking the trace gives $\mathrm{Tr}\,\mathbb{E}_x[(\nabla_{h_b}\ell)^\top h_b] = -\gamma\,\mathrm{Tr}[WW^\top] < 0$. Assuming the "presynaptic norm is approximately constant" $\|h_a\|^2 \approx \mathbb{E}\|h_a\|^2$ (holding under neural collapse or normalization), the Frobenius inner product of the learning signal and Hebbian update simplifies to $\gamma\,\mathbb{E}\|h_a\|^2\,\mathrm{Tr}[WW^\top] > 0$. This conclusion is stronger than a simple sign agreement—it explicitly describes statistical correlation and the monotonic dependence of alignment on $\gamma$.

**2. Universal Hebbian Projection Theorem for Arbitrary Learning Rules (Weak Form): Removing the "True Gradient" Assumption**

A counter-intuitive step is removing the assumption that learning signals must be true negative gradients. For a general update $\Delta W = g(x,\theta) - \gamma W$, the steady state implies $\mathbb{E}_x[g(x,\theta)] \approx \gamma W$. Calculating the inner product with the Hebbian direction $\bar H(W) := \mathbb{E}_x[h_b h_a^\top]$ gives:

$$\langle \mathbb{E}_x[g],\bar H\rangle_F = \gamma\langle W,\bar H\rangle_F = \gamma\,\mathbb{E}_x \|h_b\|^2 > 0$$

This derivation does not depend on $g$ being a gradient, so Adam, DFA, and even Random NN (using a random network as a teacher) satisfy this. This elevates Hebbian alignment to a "universal projection" in regularized learning, leading to the crucial conclusion: Hebbian signatures are non-identifiable.

**3. Noise-Induced anti-Hebbian Phase Transition: Explaining Sign Flips and Phase Boundaries**

In linear regression $\ell(w)=\tfrac12 (w^\top x - y)^2$ with parameter noise injection $w = v + \epsilon$, $\epsilon \sim \mathcal N(0,\sigma I)$, the alignment between the SGD learning signal $\Delta_{\rm SGD} w = -x(w^\top x - y)$ and the Hebbian update $\Delta_{\rm Hebb} w = x w^\top x$ expands to:

$$\mathbb{E}_\epsilon[(\Delta_{\rm SGD}w)^\top \Delta_{\rm Hebb}w] = -\|x\|^2\big[(v^\top x)^2 + \sigma^2\|x\|^2 - v^\top x\,y\big]$$

The additional $-\sigma^2\|x\|^4$ term from noise pushes alignment toward negative values. Combined with weight decay, it approximates $\approx -\sigma^2 c_0 + \gamma c_1$, placing the phase boundary on a parabola $\gamma \propto \sigma^2$, consistent with the "white phase transition band" in Figure 4.

### Loss & Training

Standard CE/MSE are used. Settings include:
- **SCE** (Standard Classification Experiment): 2-layer 128-d tanh MLP on CIFAR-10, cross-entropy, $\eta=0.01$, batch=256, 50 epochs.
- **SRE** (Standard Regression Experiment): Student-teacher regression with 32-dim Gaussian input/output; transformer variants use 2 layers, 4 heads, and 32-d embeddings.
Large **batches (256)** are emphasized for alignment measurement to capture steady-state expectations.

## Key Experimental Results

### Main Results

**Table 1 (Abridged)**: Mean $\pm$ std cosine alignment between 2nd-layer learning signal and Hebbian update across weight decay $\gamma$ in SRE (10 seeds).

| Model | Learning Rule | $\gamma=0$ | $\gamma=5\!\times\!10^{-5}$ | $\gamma=5\!\times\!10^{-4}$ | $\gamma=5\!\times\!10^{-3}$ |
|------|---------|-----------|------------------------------|------------------------------|------------------------------|
| Regression MLP | Adam | $-0.02\pm0.00$ | $0.10\pm0.00$ | $\mathbf{0.66\pm0.01}$ | — |
| Regression MLP | SGD | $-0.10\pm0.01$ | $-0.06\pm0.01$ | $0.17\pm0.01$ | $\mathbf{0.59\pm0.01}$ |
| Regression MLP | DFA | $0.45\pm0.05$ | $0.45\pm0.04$ | $0.68\pm0.05$ | $\mathbf{0.87\pm0.00}$ |
| Regression MLP | Random NN | $0.00\pm0.00$ | $0.00\pm0.00$ | $0.05\pm0.00$ | $\mathbf{0.50\pm0.00}$ |
| Transformer | Adam | $-0.02\pm0.02$ | $0.50\pm0.24$ | $\mathbf{0.99\pm0.02}$ | — |
| Transformer | SGD | $0.00\pm0.01$ | $0.04\pm0.01$ | $0.47\pm0.06$ | $\mathbf{0.88\pm0.03}$ |

Note that Random NN aligns significantly at higher $\gamma$ despite not "learning" anything useful.

### Ablation Study

**Noise-Decay Phase Diagram (Figure 4 Summary)**: Steady-state alignment signs across parameter noise $\sigma$ and weight decay $\gamma$ in SRE.

| Configuration | Hebbian Alignment Sign | Mechanism |
|------|------------------|------|
| Low Noise + High $\gamma$ | Strongly Positive ($>0.5$) | Contraction dominates (Hebbian) |
| High Noise + Low $\gamma$ | Strongly Negative ($<-0.3$) | Expansion dominates (Anti-Hebbian) |
| Phase Band $\gamma \approx \sigma^2$ | $\approx 0$ | Matches parabolic prediction |
| Very Low Noise + Very Low $\gamma$ | Near 0 | Non-steady state |

### Key Findings
- **Regularization Dictates Sign**: Increasing $\gamma$ to $5\!\times\!10^{-3}$ jumps alignment from $\approx 0$ to $0.5$-$0.99$ across almost all combinations.
- **"Non-learning Rules Align"**: Random NN and DFA also align, proving that Hebbian signatures are non-identifiable.
- **Optimal Generalization is non-Hebbian**: The best validation loss occurs near the phase transition band where alignment is near 0.
- **Early "Alignment Bump"**: An early peak in Hebbian alignment occurs during training (especially with ReLU), driven by **feature alignment** rather than norm expansion.
- **Steady-State Oscillations**: Individual neuron updates oscillate between directions; stronger oscillations often correlate with better generalization.

## Highlights & Insights

- **Reframing Hebbian Identity**: Previous neuroscience treated Hebbian signatures as evidence against gradient descent. This paper provides a clean counter-example (Random NN), lifting the identification problem to the **signal-dynamics layer**.
- **Universal Projection Perspective**: $\bar H(W)$ is a directional cone forced by steady-state regularization. Any algorithm following $\mathbb{E}[g] \approx \gamma W$ is projected by this cone, providing an "optimizer-agnostic" alignment theorem.
- **Observable Phase Transitions**: The $\gamma \propto \sigma^2$ boundary provides a verifiable prediction for neuroscience—brain regions with high noise and slow weight decay should systematically skew anti-Hebbian.

## Limitations & Future Work

- **Steady-State Assumption**: The theorem relies on expected updates reaching zero, which may not happen in large models or long-tail training.
- **Presynaptic Norm Assumption**: The dependency on neural collapse or normalization layers might not hold for all high-dimensional representations.
- **Model Scale**: Experiments are limited to small MLPs and Transformers on CIFAR-10.
- **Future Work**: Extending analysis to non-L2 regularization (e.g., sparsity), dual-source noise models, and architectures with explicit normalization layers (BN/LN).

## Related Work & Insights

- **vs Xie & Seung 2003 (CHA)**: CHA is equivalent to BP in the equilibrium limit but is not homosynaptic. Ours shows "phenomenological equivalence" without requiring homosynaptic rules.
- **vs Ziyin et al. 2025b**: This paper formalizes their scattered observations into a rigorous statistical correlation with a dual anti-Hebbian theory.
- **vs DFA/Feedback Alignment**: Those works design biologically plausible surrogates; ours argues that "looking Hebbian" is not a valid criterion for plausibility.
- **vs Oja's rule**: Classical Hebbian rules require explicit normalization; this paper shows that L2 weight decay provides an implicit normalization path.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
