---
title: >-
  [Paper Note] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?
description: >-
  [ICLR 2026][Optimization][SignSGD] Under the Power-Law Random Features model, this paper systematically analyzes the scaling laws of SignSGD…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "SignSGD"
  - "scaling laws"
  - "linear regression"
  - "random features"
  - "learning rate schedule"
date: 2026-05-08
content_hash: 0ba17b1b8a0e7c3f
---

# Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?

**Conference**: ICLR 2026
**arXiv**: [2603.02069](https://arxiv.org/abs/2603.02069)  
**Code**: None  
**Area**: Optimization Theory
**Keywords**: SignSGD, scaling laws, linear regression, random features, learning rate schedule

## TL;DR

Under the Power-Law Random Features model, this paper systematically analyzes the scaling laws of SignSGD, reveals two distinctive effects of SignSGD relative to SGD—drift normalization and noise reshaping—and demonstrates that the compute-optimal scaling exponent of SignSGD can surpass that of SGD in noise-dominated regimes.

## Background & Motivation

SignSGD is a core component of adaptive optimizers such as Adam—it updates parameters using the sign of the gradient rather than the gradient itself. In large-scale language model training, Adam/AdamW is the de facto standard, yet theoretical understanding of why and when SignSGD outperforms SGD remains limited.

**Limitations of Prior Work**:
1. Traditional analyses of SignSGD mostly focus on convex optimization or simple settings, without accounting for the scaling law phenomena observed in modern deep learning.
2. Paquette et al. (2024) analyzed the scaling laws of SGD under the power-law random features model but did not address SignSGD.
3. A precise characterization of the conditions under which SignSGD outperforms SGD is lacking.

**Core Problem**:
- How does the population risk of SignSGD scale with model size, number of training steps, and learning rate?
- Under compute-optimal configurations, when do the scaling behaviors of SignSGD and SGD differ?
- What unique effects does the WSD (warmup-stable-decay) learning rate schedule have on SignSGD?

The authors adopt the power-law random features model as the analytical framework, as it simultaneously captures feature decay and target decay—two key dimensions central to understanding scaling laws.

## Method

### Overall Architecture

Analytical framework:
- **Model**: Linear model $f(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x})$
- **Features**: Gaussian-sketched random features with power-law decaying spectrum $\lambda_k \propto k^{-\alpha}$ (feature decay parameter $\alpha$)
- **Target**: Power-law decaying target $\beta_k \propto k^{-\gamma}$ (target decay parameter $\gamma$)
- **Optimizer**: One-pass SignSGD
- **Metric**: Population risk (generalization error)

### Key Designs

1. **Closed-Form Expression for Population Risk**: The paper derives a closed-form expression for the population risk of a linear model trained with SignSGD as a function of model size $d$, number of training steps $T$, learning rate $\eta$, feature decay $\alpha$, and target decay $\gamma$.

   → Mechanism: Linearized analysis of SignSGD's nonlinear update rule under the random features model.
   → Design Motivation: Obtaining an interpretable closed form enables direct comparison with known results for SGD.

2. **Drift-Normalization Effect**: Taking the sign of the gradient normalizes the update step size along each coordinate. At the expectation level, this changes the scaling behavior of the effective drift (bias) term.

   → Mechanism: The sign operation removes gradient magnitude information, equalizing the update step size across all directions.
   → Design Motivation: This explains why SignSGD/Adam converges faster in certain settings—it automatically balances the learning speed across different feature directions.

3. **Noise-Reshaping Effect**: SignSGD modifies not only the drift term but also the structure of the noise term. Specifically, the sign operation reshapes the stochastic gradient noise from a non-uniform distribution that depends on feature scaling into a more uniform one.

   → Mechanism: After taking the sign, the noise contributions of large and small gradients are equalized.
   → Design Motivation: Noise reshaping is the key mechanism by which SignSGD can outperform SGD—in noise-dominated regimes, the reshaped noise can exhibit better decay behavior.

4. **Compute-Optimal Scaling Laws**: Under optimal learning rate selection, the paper derives the compute-optimal configuration—i.e., how to allocate model size and training steps given a fixed total compute budget $C = d \times T$.

   → Mechanism: Analysis analogous to the Chinchilla law, but tailored to SignSGD.
   → Design Motivation: In practice, the most important question is "given a compute budget, what is the best configuration?"

5. **Analysis of WSD Schedule**: The paper analyzes the effect of the warmup-stable-decay learning rate schedule under SignSGD. When feature decay is fast (large $\alpha$) but target decay is slow (small $\gamma$), WSD further reduces the noise term and sharpens the compute-optimal exponent.

   → Mechanism: Piecewise analysis of the contributions from different learning rate phases.
   → Design Motivation: WSD is the standard scheduling strategy in modern LLM training; understanding its interaction with SignSGD is crucial.

### Loss & Training

The theoretical framework primarily employs:
- Random Matrix Theory
- Deterministic Equivalents
- Power-law asymptotic expansions

All results are rigorously proved (89-page paper with 25 figures).

## Key Experimental Results

### Main Results

**Compute-Optimal Exponent Comparison: SignSGD vs. SGD**

| Parameter Regime | SignSGD Exponent | SGD Exponent | Winner | Notes |
|---|---|---|---|---|
| Noise-dominated (high $T$ relative to $d$) | Steeper | Shallower | SignSGD | Noise-reshaping effect benefits SignSGD |
| Bias-dominated (low $T$ relative to $d$) | Similar or shallower | Similar | SGD or tie | Drift normalization may be unfavorable |
| Balanced | Transitional | Transitional | Depends on specific $\alpha, \gamma$ | Two effects compete |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Fixed $\alpha$, varying $\gamma$ | Exponent change | Slower target decay → larger advantage for SignSGD |
| Fixed $\gamma$, varying $\alpha$ | Exponent change | Feature decay affects both similarly |
| With/without WSD schedule | Noise term reduction | WSD is effective when feature decay is fast + target decay is slow |
| Different learning rates | Optimal configuration | Optimal learning rate selection differs from SGD |

### Key Findings

1. **Clear Conditions for SignSGD to Outperform SGD**: When training is in the noise-dominated regime (high step-to-model-size ratio), SignSGD achieves better compute-optimal scaling. This is consistent with the empirically observed advantage of Adam over SGD—large-scale training typically operates in this regime.
2. **Two Effects Can Be Isolated**: Drift normalization and noise reshaping have precise mathematical expressions and can be analyzed independently for their contributions to scaling laws.
3. **Theoretical Justification for WSD Schedule**: This work provides the first theoretical explanation for the effectiveness of WSD scheduling with SignSGD—it sharpens the scaling exponent by reducing the noise term.
4. **Power-Law Parameters Are Decisive**: The two parameters $\alpha$ (feature decay) and $\gamma$ (target decay) fully determine the relative performance of SignSGD and SGD.

## Highlights & Insights

1. **Filling an Important Theoretical Gap**: This is the first work to establish a complete scaling law theory for SignSGD, providing a counterpart to existing analyses of SGD.
2. **Discovery of Two Distinctive Effects**: Drift normalization and noise reshaping are key concepts for understanding why Adam-type optimizers work, and are likely to generalize to broader settings.
3. **Practical Guidance**: The conditions under which SignSGD/Adam surpasses SGD are made explicit—namely, the noise-dominated regime—which aligns with the practical scenario of large-scale LLM training.
4. **Theoretical Foundation for WSD Scheduling**: The paper provides an explanation for the widely used but theoretically understudied WSD schedule.
5. **Thoroughness of Analysis**: The 89-page paper with 25 figures covers a wide range of parameter space regimes, representing an exceptionally meticulous theoretical contribution.

## Limitations & Future Work

1. **Simplifying Model Assumptions**: Linear regression with random features differs substantially from real deep network training. Although the power-law feature model captures some core phenomena, nonlinear effects and inter-parameter coupling are ignored.
2. **One-Pass Training Assumption**: One-pass SGD/SignSGD assumes each sample is seen only once, whereas practical training typically involves multiple epochs.
3. **Momentum Not Considered**: Practical Adam incorporates first- and second-order momentum, while SignSGD is a simplification. The inclusion of momentum may alter scaling behavior.
4. **Gaussian-Sketched Features**: The Gaussian assumption on features may fail to capture structure present in real data.
5. **Extension to Nonlinear Settings**: Extending the analysis to two-layer or multi-layer networks is an important but challenging direction.
6. **Empirical Validation**: Whether the theoretical predictions align with scaling behavior observed in actual deep learning training remains to be verified empirically.

## Related Work & Insights

- **Paquette et al. (2024)**: Scaling law analysis of SGD under the power-law random features model; the most direct point of comparison for this paper.
- **Chinchilla Scaling Laws** (Hoffmann et al., 2022): The original work on compute-optimal configurations; this paper derives analogous results for SignSGD.
- **Neural Scaling Laws** (Kaplan et al., 2020): Scaling relations between model size and data volume; this paper complements that work from an optimizer perspective.
- Insights: Understanding the effect of the optimizer on scaling laws is as important as understanding model architecture and data; simple nonlinearities such as the sign operation can produce complex and beneficial effects.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic analysis of SignSGD scaling laws; drift normalization and noise reshaping are novel concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Extremely thorough theoretical analysis and numerical validation (89 pages), but lacks deep learning experiments.
- Writing Quality: ⭐⭐⭐⭐ — Theoretically rigorous, though the exceptional length may hinder readability.
- Value: ⭐⭐⭐⭐ — Provides important insights into the theory of adaptive optimizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](../../ICML2026/optimization/muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](when_to_restart_exploring_escalating_restarts_on_convergence.md)

</div>

<!-- RELATED:END -->
