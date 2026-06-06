---
title: >-
  [Paper Note] Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization
description: >-
  [ICML2026][Feature Learning Strength] This work empirically discovers that an "optimal Feature Learning Strength (FLS)" exists in standard classification tasks—finding that neither maximizing nor minimizing FLS is ideal.…
tags:
  - "ICML2026"
  - "Feature Learning Strength"
  - "Implicit Bias"
  - "Neuron Alignment"
  - "Gradient Flow"
  - "Over-alignment"
date: 2026-05-08
content_hash: 42fb2217c6d047cc
---

# Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization

**Conference**: ICML2026  
**arXiv**: [2602.00827](https://arxiv.org/abs/2602.00827)  
**Code**: To be confirmed  
**Area**: Deep Learning Theory / Generalization and Implicit Bias  
**Keywords**: Feature Learning Strength, Implicit Bias, Neuron Alignment, Gradient Flow, Over-alignment

## TL;DR
This work empirically discovers that an "optimal Feature Learning Strength (FLS)" exists in standard classification tasks—finding that neither maximizing nor minimizing FLS is ideal. Through limited-time gradient flow analysis of two-layer ReLU networks under logistic loss, it decomposes generalization error into two quantifiable opposing terms: over-fitting caused by excessive FLS and "over-alignment" caused by insufficient FLS, rigorously characterizing the existence of an optimal FLS.

## Background & Motivation

**Background**: Understanding why over-parameterized neural networks generalize is a central puzzle in deep learning. A mainstream explanation is **implicit bias**—gradient descent favors specific solutions, selecting "good" minimizers even without explicit regularization. Among these, **feature learning strength (FLS)**—defined as the inverse of the effective scaling of the model output, controlled via initialization scale $\alpha$ or an output multiplier $c$—is widely regarded as a key knob determining whether learning dynamics enter the "feature learning regime" or the "NTK / kernel regime."

**Limitations of Prior Work**: Existing theories almost consistently suggest that "stronger feature learning always leads to better generalization." Evidence for this stems largely from **asymptotic analysis**: either the mean-field limit where $\alpha \to 0$ or the implicit bias limit where training time $t \to \infty$. However, real-world training involve finite time and finite samples—typically stopping early when training loss reaches a threshold (or budget is exhausted). In this practical context of "stopping immediately after achieving target training risk $\eta$," the theoretical "bigger is better" contradicts the "moderate temperature is best" observed in engineering.

**Key Challenge**: FLS governs two competing factors: (i) higher FLS (smaller $\alpha$) allows weights in Phase 1 to align precisely with the empirical class mean direction $\mathbf{x}_+/\|\mathbf{x}_+\|$; (ii) however, the empirical mean does not equal the Bayes optimal direction $\mathbf{s}_+$. With finite samples, the angle $\phi > 0$ between them exists, and excessive alignment fixes the predictor in a direction deviating from the Bayes optimum. This is the essence of "over-alignment."

**Goal**: To decompose the problem into two research questions—Q1: Is the empirical relationship between FLS and generalization non-monotonic? Q2: If an optimal FLS exists, what is its mathematical origin?

**Key Insight**: The authors define the stopping time $t_{\eta, \alpha}$ as the moment the "target training risk $\eta$ is achieved," studying gradient flow for a two-layer ReLU network under logistic loss on Gaussian mixture data. Leveraging ODE results on Phase 1 neuron alignment (min2024early, boursier2025early), they rigorously characterize the weight angular deviation as a function of $\alpha$. They decompose excess error into an **over-alignment term $\mathsf{OA}(\alpha)$** and an **over-fitting term $\mathsf{OF}(\alpha)$**, finding they are inversely monotonic with respect to $\alpha$, thus necessitating an internal optimal FLS.

**Core Idea**: Under the finite-time training paradigm, generalization error = over-alignment + over-fitting. These two components vary inversely with FLS, and the optimal FLS arises from this trade-off. This theory is consistent with empirical results across different architectures (VGG/ResNet).

## Method

### Overall Architecture

The paper first presents empirical results (Section 3), followed by theory (Section 5). The empirical section uses a unified "output multiplier + learning rate" re-parameterization—$f \mapsto cf$ while $\eta \mapsto \eta / c$—where a smaller $c$ is equivalent to a larger FLS. Test accuracy heatmaps on the $(c, \eta/c)$ plane for CIFAR-10/100 and BigGAN synthetic data reveal the universal existence of an "optimal FLS." The theoretical section focuses on two-layer ReLU + logistic loss + binary Gaussian mixtures, splitting training into two stages: Phase 1 neuron alignment and Phase 2 margin maximization. Lower bounds for weight directions $\psi_j(t)$ and effective predictor directions $\Psi(t)$ are derived as functions of $\alpha$, leading to an upper bound decomposition of excess error.

### Key Designs

1. **Unified Parameterization of FLS and Stopping Time at $\eta$**:
    - **Function**: Abstracts FLS into a scalar $\alpha$ (initialization scale $\mathbf{W}(0) = \alpha \mathsf{W}$) or $c$ (output multiplier), proving their analytical equivalence; defines stopping time $t_{\eta, \alpha} := \inf\{t \geq t_\alpha : \hat{L}_+(\theta_t) \leq \eta\}$.
    - **Mechanism**: By fixing the training endpoint to when the training loss first hits $\eta$ rather than $t \to \infty$, the theoretical analysis aligns with early-stopping practices. Comparing different $\alpha$ at the same $\eta$ fairly isolates the interference of FLS on convergence speed.
    - **Design Motivation**: Addresses the limitation of prior FLS theories that only cover asymptotic limits and are disconnected from early stopping; the introduction of stopping time is a prerequisite for making the "over-alignment vs over-fitting" trade-off explicit.

2. **Two-Phase Neuron Alignment Analysis and Angular Bounds**:
    - **Function**: Provides a lower bound for the inner product between weight directions and empirical class means in Phase 1 (alignment phase, length $t_\alpha = \Theta(\log(1/\alpha)/n)$): $\psi_j(t_\alpha) \geq \sqrt{\zeta(\alpha)} \tanh((t_\alpha - t_1)\|\mathbf{x}_+\|\sqrt{\zeta(\alpha)})$, where $\zeta(\alpha) = 1 - 4\alpha n \sqrt{h} \mathbf{x}_{max}^2 \mathsf{W}_{max}^2 / \|\mathbf{x}_+\|$.
    - **Mechanism**: The angle between weights and $\mathbf{x}_+ / \|\mathbf{x}_+\|$ at the end of Phase 1 is proportional to $\sqrt{\alpha}$ (Corollary 5.3). In Phase 2, conic-hull properties propagate single-neuron alignment to the effective predictor $\hat{\mathbf{w}}_\alpha(t)$, proving $\Psi(t_{\eta, \alpha}) \approx \Psi(t_\alpha)$, meaning Phase 2 primarily inherits the alignment results of Phase 1.
    - **Design Motivation**: Quantifies the intuition that "smaller $\alpha$ leads to stronger alignment" into a differentiable angular bound, serving as the backbone for decomposing excess error as a function of $\alpha$.

3. **Excess Error Decomposition: Over-Alignment + Over-Fitting**:
    - **Function**: Decomposes $\mathcal{E}(\hat{\mathbf{w}}_\alpha) - \mathcal{E}^* = \mathsf{OA}(\alpha) + \mathsf{OF}(\alpha)$, where $H(\alpha) = \{\mathbf{v} \in \mathbb{S}^{d-1} : \langle \mathbf{x}_+/\|\mathbf{x}_+\|, \mathbf{v}\rangle \geq \Psi(t_{\eta, \alpha})\}$ is the cone containing the effective predictor.
    - **Mechanism**: $\mathsf{OA}(\alpha) = \inf_{\mathbf{v} \in H(\alpha)} \mathcal{E}(\mathbf{v}) - \mathcal{E}^*$ measures the gap from the Bayes optimal $\mathbf{s}_+$ even if the best direction is chosen within the cone. As $\alpha$ decreases and the cone narrows, the optimal direction within the cone deviates more from $\mathbf{s}_+$, making this term **monotonically increasing**. $\mathsf{OF}(\alpha) = \mathcal{E}(\hat{\mathbf{w}}_\alpha) - \inf_{\mathbf{v} \in H(\alpha)} \mathcal{E}(\mathbf{v})$ measures additional error from randomness within the cone. As $\alpha$ increases and the cone widens, the candidate space grows, and over-fitting risk **monotonically increases**.
    - **Design Motivation**: This is the core conceptual innovation—mapping the failure of FLS being "too large or too small" to two geometric mechanisms: high FLS pins the predictor into a narrow cone deviating from Bayes (over-alignment), while low FLS allows the cone to become wide enough to include poor candidates (over-fitting).

### Loss & Training

Theoretical assumptions: (i) Data $\mathbf{x}_i = \kappa y_i \mathbf{s}_i + \sigma \mathbf{z}_i$, $\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, symmetric binary Gaussian mixture; (ii) Training set satisfies orthogonal separability $y\tilde{y}\langle \mathbf{x}, \tilde{\mathbf{x}}\rangle / (\|\mathbf{x}\|\|\tilde{\mathbf{x}}\|) \geq \lambda$; (iii) Logistic loss, optimized via gradient flow; (iv) Second-layer weight initialization $v_j(0) \sim \text{Unif}(\{\|\mathbf{w}_j(0)\|, -\|\mathbf{w}_j(0)\|\})$ to utilize balance properties. Empirical training uses direct SGD (no momentum, augmentation, weight decay, or lr scheduler), training until train acc $\geq 99\%$ and comparing peak test acc.

## Key Experimental Results

### Main Results

| Architecture | Dataset | Default FLS ($c=2^0$) | Optimal FLS | Gain |
|--------------|---------|-----------------------|-------------|------|
| ResNet-50 | CIFAR-100 | 53.57% | 59.76% ($c=2^{-4}$) | +6.19% |
| ResNet-18 | BigGAN edim=128 | 59.95% | 76.62% ($c=2^{-6}$) | +16.67% |
| VGG-19 / ResNet-18/34 | CIFAR-100 | Moderate | Internal optimal c | Universal U-shape |
| 5-layer CNN | BigGAN edim=128 | — | $c^* \propto n^{-2} h^{-1}$ | Matches theory |

### Ablation Study

| Setting | Result | Meaning |
|---------|--------|---------|
| Training risk as stopping criterion | Optimal FLS exists | Main conclusion holds |
| Validation risk as early stopping (Table 1) | Optimal $c$ remains unchanged | Not dependent on specific stopping method |
| Data difficulty edim 32 → 64 → 128 | Gain from optimal FLS grows | Harder tasks benefit more from FLS tuning |
| Across widths / dataset sizes | $c^* \propto n^{-2}h^{-1}$ | Optimal FLS transfers across scales |

### Key Findings
- **U-shaped generalization curves appear on all architectures**: VGG-19 and ResNet-18/34/50 on CIFAR-100 all exhibit heatmaps where "moderate $c$ is best and extremes are worse," indicating this is not a single-architecture phenomenon.
- **Harder tasks yield higher gains from FLS tuning**: As the effective dimension of BigGAN increases from 32 to 128, the accuracy gap between optimal and default FLS expands from a few points to over 16 points, suggesting that for difficult tasks, FLS tuning determines viability.
- **Theoretical scaling laws are transferable**: On a 5-layer CNN, sweeping width $h$ and sample size $n$ shows that the empirical optimal multiplier $c^*$ matches the theoretical prediction $O(n^{-2}h^{-1})$, implying FLS can be tuned via scaling rules similar to $\mu$P.
- **Numerical simulation validates the decomposition**: Direct numerical evaluation of $\mathsf{OA}(\alpha)$ and $\mathsf{OF}(\alpha)$ (Fig. 5) confirms the inverse monotonicity of the two curves, and their sum recovers the actual excess error curve.

## Highlights & Insights
- **Naming theoretical concepts**: Coining the failure of small $\alpha$ as "over-alignment" to contrast with traditional "over-fitting" creates a clear geometric narrative. This terminology provides a precise way to communicate why FLS cannot be infinitely increased.
- **From limits to finite time**: Moving FLS analysis from asymptotic limits to "stopping at $\eta$" is a major step in bridging theory and practice. The introduction of the stopping time $t_{\eta, \alpha}$ allows Phase 2 to inherit Phase 1 results, simplifying proofs while remaining realistic.
- **Actionable takeaway**: The paper explicitly suggests treating FLS as a formal hyperparameter axis (equivalent to lr or weight decay) and provides a predictive scaling law $c^* \propto n^{-2} h^{-1}$, which is directly applicable to tuning.

## Limitations & Future Work
- The theory strictly relies on orthogonal separability (Assumption 4.1); whether inverse monotonicity of OA/OF holds for general data distributions remains unproven.
- The work covers only a minimal model (two-layer ReLU + gradient flow + Gaussian mixtures) and does not account for modern components like BN, dropout, Adam, or momentum, which might alter the Phase 1 ODE fixed points.
- Experiments are restricted to small vision models; verifying if optimal FLS and the same scaling laws apply to Large Language Models (Transformers) with complex data is the most critical next step.

## Related Work & Insights
- **vs woodworth2020kernel / atanasov2025the**: These works argue "stronger feature learning is always better," but they use asymptotic or online settings. Ours contradicts this using finite-time analysis and identifies the geometric mechanism for optimal FLS.
- **vs petrini2022learning**: Also studies FLS and generalization but only selects between spherical regression tasks and the infinite-width limit. Ours shows the optimal FLS lies between regimes and holds broadly across standard classification tasks.
- **vs masarczyk2025unpacking / agarwala2023temperature**: Previous works empirically observed optimal values for temperature scaling but lacked theoretical explanation. Ours provides the first rigorous framework via OA/OF decomposition, elevating empirical observations to predictable scaling laws.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The OA vs OF conceptual decomposition is a genuine new insight and the first to rigorously prove optimal FLS in standard classification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid evidence across multiple architectures, datasets, and stopping criteria, with cross-scale scaling law validation.
- Writing Quality: ⭐⭐⭐⭐ Clear flow from empirical to theoretical to scaling laws; Fig. 4 effectively visualizes abstract proofs.
- Value: ⭐⭐⭐⭐⭐ Reshapes common knowledge in implicit bias literature and provides actionable tuning rules; benefits both theory and engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection](../../NeurIPS2025/others/harnessing_feature_resonance_under_arbitrary_target_alignment_for_out-of-distrib.md)
- [\[ICLR 2026\] AnyUp: Universal Feature Upsampling](../../ICLR2026/others/anyup_universal_feature_upsampling.md)
- [\[ICML 2026\] Return-to-Go is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](return-to-go_is_more_than_a_number_q-guided_alignment_for_return-conditioned_sup.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](../../ICLR2026/others/noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)
- [\[AAAI 2026\] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation](../../AAAI2026/others/forest_vs_tree_the_n_k_trade-off_in_reproducible_ml_evaluation.md)

</div>

<!-- RELATED:END -->
