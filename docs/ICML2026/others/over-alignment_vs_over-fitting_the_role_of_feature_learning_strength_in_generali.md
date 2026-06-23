---
title: >-
  [Paper Note] Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization
description: >-
  [ICML 2026][Others][Paper Note] This work provides the first empirical discovery in standard classification tasks that an "optimal value for Feature Learning Strength (FLS)" exists—it is neither "the larger the better" nor "the smaller the better." Through finite-time gradient flow analysis of two-layer ReLU networks under logistic loss, the authors
tags:
  - ICML 2026
  - Others
date: 2026-05-08
content_hash: 56a1392210a92fbe
---
# Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization

**Conference**: ICML2026  
**arXiv**: [2602.00827](https://arxiv.org/abs/2602.00827)  
**Code**: To be confirmed  
**Area**: Deep Learning Theory / Generalization & Implicit Bias  
**Keywords**: Feature Learning Strength, Implicit Bias, Neuron Alignment, Gradient Flow, Over-Alignment

## TL;DR
This work provides the first empirical discovery in standard classification tasks that an "optimal value for Feature Learning Strength (FLS)" exists—it is neither "the larger the better" nor "the smaller the better." Through finite-time gradient flow analysis of two-layer ReLU networks under logistic loss, the authors decompose the error into two quantifiable opposing terms: over-fitting caused by excessive FLS and "over-alignment" caused by insufficient FLS, rigorously characterizing the existence of an optimal FLS.

## Background & Motivation

**Background**: Understanding why over-parameterized neural networks generalize is a core mystery in deep learning. A mainstream explanation is **implicit bias**—gradient descent prefers specific solutions, selecting "good" minimizers without explicit regularization. **Feature learning strength (FLS)**—defined as the reciprocal of the effective scaling of model output, controlled via initialization scale $\alpha$ or output multiplier $c$—is widely regarded as the key knob determining whether learning dynamics enter the "feature learning regime" or the "NTK / kernel regime."

**Limitations of Prior Work**: Existing theories almost consistently suggest that "stronger feature learning always leads to better generalization." Evidence for this conclusion mostly stems from **asymptotic analysis**: either the mean-field limit where $\alpha \to 0$, or the implicit bias limit where training time $t \to \infty$. However, real-world training involve finite time and finite samples—typically stopping early when training loss reaches a threshold (or budget is exhausted). In this practical context of "stopping once the target training risk $\eta$ is reached," the theoretical "larger is better" contradicts the engineering observation that a "moderate temperature" is optimal.

**Key Challenge**: FLS determines two competing factors: (i) Larger FLS (smaller $\alpha$) allows weights to align more precisely with the empirical class mean direction $\mathbf{x}_+/\|\mathbf{x}_+\|$ during Phase 1; (ii) but the empirical class mean does not equal the Bayesian optimal direction $\mathbf{s}_+$. With finite samples, the angle $\phi > 0$ between them exists, and excessive alignment nails the predictor to a direction deviating from the Bayes optimum. This is the essence of "over-alignment."

**Goal**: This work addresses two research questions—Q1: Is the relationship between FLS and generalization monotonic empirically? Q2: If an optimal FLS exists, what is its mathematical origin?

**Key Insight**: The authors define the "moment the target training risk $\eta$ is achieved" as the stopping time $t_{\eta, \alpha}$, and study the gradient flow of two-layer ReLU networks with logistic loss on Gaussian mixture data. Leveraging ODE results from min2024early and boursier2025early regarding Phase 1 neuron alignment, they rigorously characterize the angular deviation of weights as a function of $\alpha$. They decompose the excess error into an **over-alignment term $\mathsf{OA}(\alpha)$** and an **over-fitting term $\mathsf{OF}(\alpha)$**, finding that both are inversely monotonic with respect to $\alpha$, necessitating an optimal FLS in the interior.

**Core Idea**: Under a finite-time training paradigm, generalization error = over-alignment + over-fitting. These terms vary inversely with FLS; the optimal FLS arises from this tradeoff, consistent across theoretical analysis and empirical results on architectures like VGG and ResNet.

## Method

### Overall Architecture

The paper begins with an empirical study (Section 3) followed by theoretical analysis (Section 5). The empirical part uses a unified "output multiplier + learning rate" reparameterization—$f \mapsto cf$ while setting $\eta \mapsto \eta / c$—where a smaller $c$ is equivalent to a larger FLS. Heatmaps of test accuracy on the $(c, \eta/c)$ plane for CIFAR-10/100 and BigGAN synthetic data reveal the universal existence of an "optimal FLS." The theoretical part focuses on two-layer ReLU + logistic loss + binary Gaussian mixture, splitting training into two stages: Phase 1 neuron alignment and Phase 2 margin maximization. Lower bounds for the weight direction $\psi_j(t)$ and effective predictor direction $\Psi(t)$ relative to $\alpha$ are provided for both stages, leading to an upper bound decomposition of the excess error.

### Key Designs

**1. Unified Parameterization of FLS and Stopping Time at $\eta$: Realigning Theory with Early Stopping Reality**

Previous FLS theories mostly focused on asymptotic limits—either mean-field $\alpha\to0$ or implicit bias $t\to\infty$—which are disconnected from practical training where loss is reduced to a threshold. This led to the conclusion that "FLS should be as large as possible," contradicting engineering intuition. This work abstracts FLS into a scalar: controlled either by initialization scale $\mathbf{W}(0)=\alpha\mathsf{W}$ or by an output multiplier $c$, proving that $f\mapsto cf$ with $\eta\mapsto\eta/c$ is analytically equivalent. A key step is fixing the training endpoint at the moment the training loss first drops to $\eta$:

$$t_{\eta,\alpha}:=\inf\{t\ge t_\alpha:\hat{L}_+(\theta_t)\le\eta\}$$

Instead of $t\to\infty$. This allows a fair comparison across different $\alpha$ at the same $\eta$, isolating the interference of "FLS changing convergence speed" and enabling the explicit modeling of the "over-alignment vs over-fitting" tradeoff.

**2. Two-Stage Neuron Alignment Analysis and Angular Bounds: Quantifying the $\sqrt{\alpha}$ Scaling**

Training is divided into Phase 1 (neuron alignment, length $t_\alpha=\Theta(\log(1/\alpha)/n)$) and Phase 2 (margin maximization). At the end of Phase 1, the inner product of weight direction and the empirical class mean $\mathbf{x}_+/\|\mathbf{x}_+\|$ has a lower bound:

$$\psi_j(t_\alpha)\ge\sqrt{\zeta(\alpha)}\tanh\big((t_\alpha-t_1)\|\mathbf{x}_+\|\sqrt{\zeta(\alpha)}\big),\quad \zeta(\alpha)=1-\frac{4\alpha n\sqrt{h}\,\mathbf{x}_{max}^2\mathsf{W}_{max}^2}{\|\mathbf{x}_+\|}$$

This implies the angle between the weight direction and $\mathbf{x}_+/\|\mathbf{x}_+\|$ is proportional to $\sqrt{\alpha}$. In Phase 2, the alignment of individual neurons is transferred to the effective predictor $\hat{\mathbf{w}}_\alpha(t)$ using conic-hull properties, proving $\Psi(t_{\eta, \alpha}) \approx \Psi(t_\alpha)$. This $\sqrt{\alpha}$ angular bound serves as the backbone for expressing excess error as a differentiable function of $\alpha$.

**3. Excess Error Decomposition: Inverse Monotonicity of Over-Alignment and Over-Fitting**

This is the core conceptual innovation. The excess error is written as the sum of two terms: $\mathcal{E}(\hat{\mathbf{w}}_\alpha)-\mathcal{E}^*=\mathsf{OA}(\alpha)+\mathsf{OF}(\alpha)$, where the effective predictor is constrained within a cone $H(\alpha)=\{\mathbf{v}\in\mathbb{S}^{d-1}:\langle\mathbf{x}_+/\|\mathbf{x}_+\|,\mathbf{v}\rangle\ge\Psi(t_{\eta,\alpha})\}$. The over-alignment term $\mathsf{OA}(\alpha)=\inf_{\mathbf{v}\in H(\alpha)}\mathcal{E}(\mathbf{v})-\mathcal{E}^*$ measures how far the best direction in the cone still deviates from the Bayes optimum $\mathbf{s}_+$—as $\alpha$ decreases, the cone shrinks, and the optimal direction in the cone deviates more from $\mathbf{s}_+$, causing this term to increase. The over-fitting term $\mathsf{OF}(\alpha)=\mathcal{E}(\hat{\mathbf{w}}_\alpha)-\inf_{\mathbf{v}\in H(\alpha)}\mathcal{E}(\mathbf{v})$ measures the extra error from randomness within the cone—as $\alpha$ increases, the cone widens, increasing the candidate solution space and this term. The opposing monotonicity of these two terms guarantees the existence of an interior optimal FLS.

### Loss & Training

Theoretical assumptions: (i) Data $\mathbf{x}_i = \kappa y_i \mathbf{s}_i + \sigma \mathbf{z}_i$, $\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, symmetric binary Gaussian mixture; (ii) Training set satisfies orthogonal separability $y\tilde{y}\langle \mathbf{x}, \tilde{\mathbf{x}}\rangle / (\|\mathbf{x}\|\|\tilde{\mathbf{x}}\|) \geq \lambda$; (iii) Logistic loss with gradient flow optimization; (iv) Second-layer weights initialized as $v_j(0) \sim \text{Unif}(\{\|\mathbf{w}_j(0)\|, -\|\mathbf{w}_j(0)\|\})$. Empirical training uses SGD (no momentum, augmentation, weight decay, or LR scheduler), training until train acc $\geq 99\%$ and comparing peak test acc.

## Key Experimental Results

### Main Results

| Architecture | Dataset | Default FLS ($c=2^0$) | Optimal FLS | Gain |
|------|--------|-------------------|----------|------|
| ResNet-50 | CIFAR-100 | 53.57% | 59.76% ($c=2^{-4}$) | +6.19% |
| ResNet-18 | BigGAN edim=128 | 59.95% | 76.62% ($c=2^{-6}$) | +16.67% |
| VGG-19 / ResNet-18/34 | CIFAR-100 | Medium | Optimal internal c | Universal U-shape |
| 5-layer CNN | BigGAN edim=128 | — | $c^* \propto n^{-2} h^{-1}$ | Matches theory |

### Ablation Study

| Setting | Result | Significance |
|------|------|------|
| Training risk as stopping criterion | Optimal FLS exists | Confirms main conclusion |
| Validation risk as stopping criterion (Table 1) | Optimal $c$ remains unchanged | Generalizable across stopping methods |
| Data difficulty (edim 32 → 64 → 128) | FLS gain increases with difficulty | Harder tasks benefit more from FLS tuning |
| Across widths / dataset sizes | $c^* \propto n^{-2}h^{-1}$ | Optimal FLS scales predictably |

### Key Findings
- **U-shaped generalization curves appear across all architectures**: VGG-19 and ResNet-18/34/50 on CIFAR-100 exhibit heatmaps where "medium $c$ is best," proving this is not an architecture-specific phenomenon.
- **Harder tasks yield higher gains from FLS tuning**: As the effective dimension of BigGAN increases from 32 to 128, the gap between optimal FLS and default FLS grows from a few points to over 16 points.
- **Theoretical scaling laws are transferable**: On a 5-layer CNN, the measured optimal output multiplier $c^*$ as a function of width $h$ and data size $n$ aligns with the theoretical $O(n^{-2}h^{-1})$, suggesting FLS can be tuned via scaling rules like $\mu$P.
- **Numerical simulation validates the decomposition**: Direct numerical evaluation of $\mathsf{OA}(\alpha)$ and $\mathsf{OF}(\alpha)$ (Fig. 5) shows inverse monotonicity, where their sum recovers the actual excess error curve.

## Highlights & Insights
- **Conceptual Nomenclature**: Naming the failure of small $\alpha$ as "over-alignment" creates a counterpart to traditional over-fitting. This provides a geometric narrative for why FLS cannot be infinitely increased.
- **From Asymptotic to Finite Time**: Shifting FLS analysis from limits to the "stop at $\eta$" regime is a vital step in bridging theory and engineering. The stopping time $t_{\eta, \alpha}$ allows Phase 2 to simply inherit results from Phase 1, simplifying proofs while staying realistic.
- **Actionable Takeaway**: The work suggests listing FLS as a formal hyperparameter axis (alongside LR and weight decay) and provides a predictive scaling law $c^* \propto n^{-2} h^{-1}$ for practical use.

## Limitations & Future Work
- The theory relies strictly on the orthogonal separability assumption (Assumption 4.1); whether inverse monotonicity of OA/OF holds for general distributions remains unproven.
- The work covers a minimal model (two-layer ReLU + gradient flow + Gaussian mixture) without components like BN, dropout, Adam, or momentum, which might alter Phase 1 ODE fixed points.
- Experiments are limited to small vision models. Verifying if optimal FLS and the same scaling laws apply to Transformers or LLMs with complex data is the most valuable next step.

## Related Work & Insights
- **vs woodworth2020kernel / atanasov2025the**: These works argue that "stronger feature learning is always better" but operate in asymptotic or online settings. This paper refutes this using finite-time analysis.
- **vs petrini2022learning**: Also studies FLS and generalization but focuses on spherical regression or binary choices between regimes. This work proves the optimal FLS lies between regimes in standard classification.
- **vs masarczyk2025unpacking / agarwala2023temperature**: Previous works empirically observed optimal values for temperature scaling; this work provides the first rigorous OA/OF decomposition framework to explain the mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The OA vs OF conceptual decomposition is a genuine new insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid validation across architectures, datasets, and scales.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from empirical to theoretical to scaling law.
- Value: ⭐⭐⭐⭐⭐ Bridges the gap between implicit bias theory and practical tuning rules.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](../../ACL2025/others/revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)
- [\[ICML 2026\] Return-to-Go is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](return-to-go_is_more_than_a_number_q-guided_alignment_for_return-conditioned_sup.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](../../ICLR2026/others/noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)
- [\[CVPR 2026\] Data-Centric Meta-Learning for Robust Few-Shot Generalization](../../CVPR2026/others/data-centric_meta-learning_for_robust_few-shot_generalization.md)
- [\[CVPR 2025\] TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](../../CVPR2025/others/traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)

</div>

<!-- RELATED:END -->
