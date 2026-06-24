---
title: >-
  [Paper Note] Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations
description: >-
  [NeurIPS 2025][Optimization][representational drift] This work theoretically demonstrates that the statistical properties (variance and dimensionality) of task-irrelevant stimuli are key drivers of representational drift in online learning. Across Oja's rule, Similarity Matching, autoencoders, and supervised two-layer networks, a drift rate of $D \propto \lambda_\perp^2 (n-m)$ is consistently observed. Furthermore, learning-noise-induced drift exhibits anisotropic geometric s…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "representational drift"
  - "task-irrelevant noise"
  - "online learning"
  - "Hebbian learning"
  - "SGD noise"
date: 2026-05-08
content_hash: 5d6cde89a0b2b976
---

# Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations

**Conference**: NeurIPS 2025
**arXiv**: [2510.21588](https://arxiv.org/abs/2510.21588)  
**Code**: None  
**Area**: Neuroscience / Representation Learning / Optimization Theory
**Keywords**: representational drift, task-irrelevant noise, online learning, Hebbian learning, SGD noise

## TL;DR
This work theoretically demonstrates that the statistical properties (variance and dimensionality) of task-irrelevant stimuli are key drivers of representational drift in online learning. Across Oja's rule, Similarity Matching, autoencoders, and supervised two-layer networks, a drift rate of $D \propto \lambda_\perp^2 (n-m)$ is consistently observed. Furthermore, learning-noise-induced drift exhibits anisotropic geometric structure, qualitatively distinct from the isotropic drift induced by Gaussian synaptic noise.

## Background & Motivation

**Background**: Neuroscience experiments have shown that neural representations at the single-neuron level continuously change even when behavior and task performance remain stable — a phenomenon known as representational drift. Computational models have reproduced this phenomenon from multiple perspectives.

**Limitations of Prior Work**: The noise sources underlying drift remain unclear — they may stem from biological factors such as synaptic turnover, or from the sampling stochasticity inherent in online learning. Prior work has focused primarily on how SGD noise drives parameters toward flatter loss landscape regions, without systematically studying different architectures and learning rules or clarifying how data statistics relate to drift.

**Key Challenge**: Task-irrelevant stimuli are learned to be suppressed by the network (zero output), and intuitively should not affect representations. However, the nature of online learning means the network cannot fully ignore any part of the data distribution — even suppressed stimuli continue to update weights.

**Core Idea**: Task-irrelevant stimuli continuously perturb weights via a multiplicative coupling term in the learning update rule ($\Delta W^* = \eta \tilde{W} x_{||} x_\perp^T$), causing task-relevant representations to diffuse (drift) along the tangent space of the solution manifold.

## Method

### Overall Architecture
Under the online learning setting, once the network has converged to an optimal solution, continued online learning causes parameters to diffuse along the solution manifold. The dynamics are approximated via a stochastic differential equation (SDE), decomposed into a normal component (attracted back to the manifold) and a tangential component (pure diffusion/drift). The tangential diffusion coefficient is then derived as a function of the data distribution.

### Key Designs

1. **SDE Decomposition Framework**

    - **Function**: Approximates post-convergence parameter dynamics as a continuous-time SDE, decomposed into normal and tangential components relative to the solution manifold.
    - **Mechanism**: Near the solution point $\tilde{\theta}$, the normal component $d\theta_N = -H(\theta_N - \tilde{\theta})dt + \sqrt{\eta}C_N d B_t$ has a Hessian-driven restoring force (mean reversion), while the tangential component $d\theta_T = \sqrt{\eta}C_T dB_t'$ is pure Brownian motion. The drift rate is determined by the tangential diffusion coefficient $C_T$.
    - **Design Motivation**: Reduces complex high-dimensional stochastic dynamics to a diffusion problem on a manifold, enabling closed-form solutions.

2. **Unified Analysis Across Four Architectures**

    - **Oja's Rule** (unsupervised): Single-layer network learning the $m$-dimensional principal subspace. The solution manifold has $O(m)$ rotational symmetry. Drift rate: $D_y \approx \frac{\eta^3 \lambda_\perp^2}{8}(m-1)(n-m)$
    - **Similarity Matching** (unsupervised): Same principal subspace learning objective as Oja's rule but using a different biologically plausible learning rule. Yields the same drift rate formula — indicating drift characteristics are determined by task structure rather than learning rule details.
    - **Linear Autoencoder** (SGD): Bottleneck hidden layer learning a $p$-dimensional principal subspace. $D_h \approx \frac{\eta^3 \lambda_\perp^2}{32}(p-1)(n-p)$
    - **Supervised Two-Layer Network** (SGD): Task-irrelevant subspace determined by the null space of the input-output mapping $P$. $D_h \approx \frac{\eta^3 \gamma^4}{16}(k-1)(k+2+(n-k)\lambda_\perp/2)$

3. **Geometric Distinction: Learning Noise vs. Synaptic Noise**

    - **Function**: Contrasts drift characteristics induced by learning noise (from online sampling stochasticity) and Gaussian synaptic noise ($\varepsilon_{ij} \sim \mathcal{N}(0, \eta\sigma^2_{syn})$).
    - **Core Finding**: Learning-noise-induced drift is **anisotropic** (drift rates differ across directions), whereas synaptic noise induces **isotropic** drift. The two also differ qualitatively in how drift rate depends on output dimensionality — non-monotonic (first increasing then decreasing) under learning noise, and monotonically increasing under synaptic noise.
    - **Design Motivation**: If these two geometric patterns can be distinguished experimentally, they can reveal the dominant mechanism driving drift in the brain.

### Loss & Training
- Oja / SM: No explicit loss; online Hebbian updates.
- Autoencoder / two-layer network: MSE loss + weight decay + SGD.
- All experiments use small learning rates in an online learning setting; drift is observed by continuing training after convergence.

## Key Experimental Results

### Main Results (Gaussian Data)

| Architecture | Drift Rate Formula | $\lambda_\perp$ Dependence | Dimensionality Dependence |
|---|---|---|---|
| Oja | $D \approx \frac{\eta^3 \lambda_\perp^2}{8}(m-1)(n-m)$ | $\propto \lambda_\perp^2$ | $\propto (n-m)$ |
| Similarity Matching | Same as Oja | $\propto \lambda_\perp^2$ | $\propto (n-m)$ |
| Linear Autoencoder | $D \approx \frac{\eta^3 \lambda_\perp^2}{32}(p-1)(n-p)$ | $\propto \lambda_\perp^2$ | $\propto (n-p)$ |
| Supervised Two-Layer | $D \propto (k-1)(k+2+(n-k)\lambda_\perp/2)$ | Linear | $\propto (n-k)$ |

All theoretical predictions closely match simulation results (Figure 3).

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| MNIST, varying output dimension $m$ | Drift rate is non-monotonic | Increasing $m$ enlarges the representation space (increases drift) but shrinks the task-irrelevant space (reduces noise), producing a trade-off |
| $m = n$ | Drift rate → 0 | No task-irrelevant subspace; learning noise vanishes |
| Nonlinear network (ReLU) | Drift rate still increases with $\lambda_\perp$ | Qualitative conclusions of linear theory hold in nonlinear networks |
| Increasing synaptic noise $\sigma_{syn}$ | Drift rate–dimensionality relation becomes monotonically increasing | Qualitatively different from the non-monotonic relation under learning noise |

### Key Findings
- Task-irrelevant stimuli drive drift via the $x_{||} x_\perp^T$ coupling term — both components must be simultaneously present to produce an update.
- Drift rate is jointly controlled by representation space size (proportional to $m-1$) and noise source magnitude (proportional to $n-m$), yielding non-monotonic dependence.
- Learning noise vs. synaptic noise yields distinguishable experimental predictions: anisotropic vs. isotropic geometry, and non-monotonic vs. monotonic dimensionality dependence.
- Cross-architecture consistency: despite differences in specific formulas, all architectures exhibit drift dependence on task-irrelevant stimuli.

## Highlights & Insights
- **Multiplicative Noise Structure**: The key insight is the form of the learning update at the solution point, $\Delta W^* \propto x_{||} x_\perp^T$ — a product of task-relevant and task-irrelevant components. This more precisely reveals the data-dependent structure of drift than a generic "SGD noise" framing.
- **Experimentally Testable Predictions**: The anisotropic vs. isotropic geometric distinction can be verified through long-term recording experiments, providing actionable hypotheses for whether drift in the brain is primarily driven by learning or synaptic noise.
- **Cross-Learning-Rule Universality**: Oja's rule and Similarity Matching yield identical drift rates despite different learning rules, indicating that drift is determined by task structure rather than learning rule details.

## Limitations & Future Work
- The theory is primarily based on linear networks and small learning rate assumptions; generalization to deep nonlinear networks requires further work.
- Only stationary data distributions are considered; non-stationary environments (e.g., continual learning / catastrophic forgetting) are not addressed.
- Experimental validation is limited in scale (toy Gaussian data + MNIST + simple two-layer networks), with no verification on large-scale deep networks.
- No direct comparison with neuroscience experimental data is made.

## Related Work & Insights
- **vs. Kunin et al. [25]**: That work studies drift in two-layer autoencoders with expanded hidden layers under SGD + weight decay; this paper extends the analysis to multiple architectures and supervised learning, highlighting the role of task-irrelevant stimuli.
- **vs. Qin et al. [9]**: That work drives drift via synaptic noise (additive Gaussian noise model); this paper shows that learning noise alone is sufficient, and yields different geometric predictions.
- **Implications for Deep Learning**: The "drift" produced by SGD after convergence may be influenced by irrelevant dimensions in the training data, which has implications for understanding implicit regularization and parameter space exploration in overparameterized networks.

## Rating
- Novelty: ⭐⭐⭐⭐ — Systematically links task-irrelevant stimuli to representational drift; the multiplicative noise structure insight is elegant.
- Experimental Thoroughness: ⭐⭐⭐ — Primarily theoretical; experimental validation is confined to simple models with no large-scale experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Progresses clearly from intuitive examples to theoretical derivations to comparative analysis.
- Value: ⭐⭐⭐⭐ — Informative for both neuroscience and deep learning theory, with experimentally verifiable predictions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[ICML 2026\] Probing Neural TSP Representations for Prescriptive Decision Support](../../ICML2026/optimization/probing_neural_tsp_representations_for_prescriptive_decision_support.md)
- [\[NeurIPS 2025\] FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts](flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)
- [\[ICML 2025\] Adjustment for Confounding using Pre-Trained Representations](../../ICML2025/optimization/adjustment_for_confounding_using_pre-trained_representations.md)
- [\[NeurIPS 2025\] OrthoGrad Improves Neural Calibration](orthograd_improves_neural_calibration.md)

</div>

<!-- RELATED:END -->
