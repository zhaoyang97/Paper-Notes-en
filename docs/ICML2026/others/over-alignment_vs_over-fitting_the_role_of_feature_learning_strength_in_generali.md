---
title: >-
  [Paper Note] Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization
description: >-
  [ICML 2026][Others][Paper Note] This paper provides the first empirical evidence in standard classification tasks that an "optimal Feature Learning Strength (FLS)" exists—it is neither the larger the better nor the smaller the better. Through a finite-time gradient flow analysis of two-layer ReLU networks under logistic loss, the authors decompose th
tags:
  - ICML 2026
  - Others
date: 2026-05-08
content_hash: 1de45c848db142ed
---
# Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization

**Conference**: ICML2026  
**arXiv**: [2602.00827](https://arxiv.org/abs/2602.00827)  
**Code**: TBD  
**Area**: Deep Learning Theory / Generalization and Implicit Bias  
**Keywords**: Feature Learning Strength, Implicit Bias, Neuron Alignment, Gradient Flow, Over-Alignment

## TL;DR
This paper provides the first empirical evidence in standard classification tasks that an "optimal Feature Learning Strength (FLS)" exists—it is neither the larger the better nor the smaller the better. Through a finite-time gradient flow analysis of two-layer ReLU networks under logistic loss, the authors decompose the generalization error into two quantifiable opposing terms: over-fitting caused by excessive FLS and "over-alignment" caused by insufficient FLS, rigorously characterizing the existence of an optimal FLS.

## Background & Motivation

**Background**: Understanding why over-parameterized neural networks generalize is a core mystery in deep learning. A mainstream explanation is **implicit bias**—gradient descent prefers specific solutions, selecting "good" minimizers even without explicit regularization. Among these factors, **feature learning strength (FLS)**—defined as the inverse of the effective scaling of the model output, controlled by the initialization scale $\alpha$ or the output multiplier $c$—is widely regarded as the key knob determining whether the learning dynamics follow the "feature learning regime" or the "NTK / kernel regime."

**Limitations of Prior Work**: Existing theories almost consistently suggest that "stronger feature learning always leads to better generalization." Evidence for this conclusion mostly comes from **asymptotic analysis**: either the mean-field limit where $\alpha \to 0$, or the implicit bias limit as training time $t \to \infty$. However, real-world training involves finite time and finite samples—usually early-stopping when the training loss reaches a certain threshold (or budget is exhausted). In the practical context of "stopping immediately after the target training risk $\eta$ is achieved," the theoretical "the larger, the better" contradicts the engineering observation that "moderate temperatures perform best."

**Key Challenge**: FLS determines two competing effects: (i) higher FLS (smaller $\alpha$) allows weights to align more precisely with the empirical class mean direction $\mathbf{x}_+/\|\mathbf{x}_+\|$ during Phase 1; (ii) however, the empirical class mean direction is not equal to the Bayes optimal direction $\mathbf{s}_+$. With finite samples, the angle $\phi > 0$ between them exists, and over-strong alignment effectively nails the predictor to a direction deviating from the Bayes optimum. This is the essence of "over-alignment."

**Goal**: To answer two research questions—Q1: Is the empirical relationship between FLS and generalization monotonic? Q2: If an optimal FLS exists, what is its mathematical origin?

**Key Insight**: The authors use the "moment when the target training risk $\eta$ is achieved" as the stopping time $t_{\eta, \alpha}$, studying the gradient flow of two-layer ReLU networks with logistic loss on Gaussian mixture data. Leveraging ODE results on neuron alignment from min2024early and boursier2025early, they rigorously characterize the angular deviation of weights as a function of $\alpha$. They then decompose the excess error into an **over-alignment term $\mathsf{OA}(\alpha)$** and an **over-fitting term $\mathsf{OF}(\alpha)$**, finding that both are inversely monotonic with respect to $\alpha$, thus necessitating the existence of an optimal FLS in the interior.

**Core Idea**: Under the finite-time training paradigm, Generalization Error = Over-Alignment + Over-Fitting. These two terms vary in opposite directions with FLS; the optimal FLS arises from this trade-off, consistent across theoretical results and empirical architectures (VGG/ResNet).

## Method

### Overall Architecture

The paper first conducts empirical studies (Section 3) followed by theoretical analysis (Section 5). The empirical part uses a unified re-parameterization of "output multiplier + learning rate"—mapping $f \mapsto cf$ while setting $\eta \mapsto \eta / c$, such that a smaller $c$ is equivalent to a larger FLS. Test accuracy heatmaps are generated across the $(c, \eta/c)$ plane on CIFAR-10/100 and BigGAN synthetic data, revealing the ubiquity of "optimal FLS." The theoretical part focuses on two-layer ReLU + logistic loss + binary Gaussian mixture classification, splitting training into two phases: Phase 1 (neuron alignment) and Phase 2 (margin maximization). Lower bounds for the weight directions $\psi_j(t)$ and the effective predictor direction $\Psi(t)$ with respect to $\alpha$ are provided, culminating in an upper bound decomposition of the excess error.

### Key Designs

**1. Unified Parameterization of FLS and the "Stop at $\eta$" Criterion: Bringing Theory Back to Early-Stopping Reality**

Previous FLS theories mostly discussed asymptotic limits—either the mean-field limit ($\alpha\to0$) or the implicit bias limit ($t\to\infty$)—which are severely disconnected from actual training where losses are reduced to a threshold. This led to conclusions that FLS is always better when larger, contradicting engineering intuition. This work abstracts FLS into a scalar: it can be controlled via initialization scale $\mathbf{W}(0)=\alpha\mathsf{W}$ or the output multiplier $c$, proving that $f\mapsto cf$ with $\eta\mapsto\eta/c$ are analytically equivalent. The critical step is fixing the training endpoint at the moment the training loss first drops to $\eta$:

$$t_{\eta,\alpha}:=\inf\{t\ge t_\alpha:\hat{L}_+(\theta_t)\le\eta\}$$

instead of $t\to\infty$. This allows for a fair comparison of different $\alpha$ values at the same $\eta$, isolating the interference of "FLS changing convergence speed," which is the prerequisite for making the "over-alignment vs over-fitting" trade-off explicit. Without this stopping time, the two terms could not be written as differentiable functions of $\alpha$.

**2. Two-Stage Neuron Alignment Analysis and Angular Lower Bound: Quantifying "Smaller $\alpha$ Means Stronger Alignment" at $\sqrt{\alpha}$ Scale**

Training is divided into Phase 1 (neuron alignment, length $t_\alpha=\Theta(\log(1/\alpha)/n)$) and Phase 2 (margin maximization). At the end of Phase 1, the inner product between the weight direction and the empirical class mean direction $\mathbf{x}_+/\|\mathbf{x}_+\|$ is lower-bounded by:

$$\psi_j(t_\alpha)\ge\sqrt{\zeta(\alpha)}\tanh\big((t_\alpha-t_1)\|\mathbf{x}\|\|\sqrt{\zeta(\alpha)}\big),\quad \zeta(\alpha)=1-\frac{4\alpha n\sqrt{h}\,\mathbf{x}_{max}^2\mathsf{W}_{max}^2}{\|\mathbf{x}_+\|}$$

This implies that the angle between the weight direction and $\mathbf{x}_+/\|\mathbf{x}_+\|$ is proportional to $\sqrt{\alpha}$ (Corollary 5.3). In Phase 2, the conic-hull property is used to transfer single-neuron alignment to the effective predictor $\hat{\mathbf{w}}_\alpha(t)$, proving $\Psi(t_{\eta, \alpha})\approx\Psi(t_\alpha)$—meaning Phase 2 almost entirely inherits the alignment result from Phase 1. This $\sqrt{\alpha}$ angular lower bound is the backbone for expressing excess error as a differentiable function of $\alpha$ and directly yields the scaling laws for optimal FLS.

**3. Excess Error Decomposition: Counter-Monotonicity of Over-Alignment and Over-Fitting**

This is the most core conceptual innovation. The excess error is written as the sum of two terms: $\mathcal{E}(\hat{\mathbf{w}}_\alpha)-\mathcal{E}^*=\mathsf{OA}(\alpha)+\mathsf{OF}(\alpha)$, where the effective predictor is constrained within the cone $H(\alpha)=\{\mathbf{v}\in\mathbb{S}^{d-1}:\langle\mathbf{x}_+/\|\mathbf{x}_+\|,\mathbf{v}\rangle\ge\Psi(t_{\eta,\alpha})\}$. The over-alignment term $\mathsf{OA}(\alpha)=\inf_{\mathbf{v}\in H(\alpha)}\mathcal{E}(\mathbf{v})-\mathcal{E}^*$ measures "even when picking the best within the cone, how far it still deviates from the Bayes optimal $\mathbf{s}_+$"—as $\alpha$ decreases and the cone shrinks, the best direction within the cone deviates more from $\mathbf{s}_+$, making this term monotonically increasing. The over-fitting term $\mathsf{OF}(\alpha)=\mathcal{E}(\hat{\mathbf{w}}_\alpha)-\inf_{\mathbf{v}\in H(\alpha)}\mathcal{E}(\mathbf{v})$ measures "additional error brought by randomness within the cone"—as $\alpha$ increases and the cone widens, the candidate solution space grows, making this term also monotonically increasing. Since the two terms are inversely monotonic with respect to $\alpha$, too small an FLS nails the predictor to a narrow cone deviating from the Bayes direction (over-alignment), while too large an FLS makes the cone wide enough to accommodate too many candidates (over-fitting). Their opposing monotonicity mathematically guarantees the existence of an optimal FLS in the interior.

### Loss & Training

Theoretical Assumptions: (i) Data $\mathbf{x}_i = \kappa y_i \mathbf{s}_i + \sigma \mathbf{z}_i$, where $\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, a symmetric binary Gaussian mixture; (ii) The training set satisfies orthogonal separability $y\tilde{y}\langle \mathbf{x}, \tilde{\mathbf{x}}\rangle / (\|\mathbf{x}\|\|\tilde{\mathbf{x}}\|) \geq \lambda$; (iii) Logistic loss with gradient flow optimization; (iv) Second-layer weights initialized as $v_j(0) \sim \text{Unif}(\{\|\mathbf{w}_j(0)\|, -\|\mathbf{w}_j(0)\|\})$ to utilize balancedness properties. Empirical training uses direct SGD (no momentum, no augmentation, no weight decay, no lr scheduler), training until train acc $\geq 99\%$ and comparing peak test acc.

## Key Experimental Results

### Main Results

| Architecture | Dataset | Default FLS ($c=2^0$) | Optimal FLS | Gain |
|--------------|---------|-----------------------|-------------|------|
| ResNet-50 | CIFAR-100 | 53.57% | 59.76% ($c=2^{-4}$) | +6.19% |
| ResNet-18 | BigGAN edim=128 | 59.95% | 76.62% ($c=2^{-6}$) | +16.67% |
| VGG-19 / ResNet-18/34 | CIFAR-100 | Moderate | Internal optimal c | Universal U-shape |
| 5-layer CNN | BigGAN edim=128 | — | $c^* \propto n^{-2} h^{-1}$ | Matches theoretical scaling |

### Ablation Study

| Setting | Result | Meaning |
|---------|--------|---------|
| Training risk as stop criterion | Optimal FLS exists | Main conclusion holds |
| Validation risk as stop criterion (Table 1) | Optimal $c$ remains unchanged | Independent of specific stopping method |
| Data difficulty edim 32 → 64 → 128 | Optimal FLS gain increases | Harder tasks make FLS tuning more rewarding |
| Across widths / across dataset sizes | $c^* \propto n^{-2}h^{-1}$ | Optimal FLS transferable across scales |

### Key Findings
- **U-shaped generalization curves appear on all architectures**: VGG-19 and ResNet-18/34/50 on CIFAR-100 all show heatmaps where "moderate $c$ is best, extremes are worse," indicating this is not an architecture-specific phenomenon.
- **The harder the task, the greater the benefit of tuning FLS**: As the effective dimension of BigGAN increases from 32 to 128, the accuracy gap between optimal and default FLS expands from a few points to over 16 points, implying that for difficult tasks, tuning FLS determines usability.
- **Theoretical scaling laws are transferable**: Sweeping width $h$ and data volume $n$ on a 5-layer CNN shows that the empirical optimal output multiplier $c^*$ matches the theoretical prediction $O(n^{-2}h^{-1})$ closely, suggesting that FLS tuning can follow rules like $\mu$P rather than requiring re-sweeping.
- **Numerical simulation validates the decomposition**: Direct numerical evaluation of $\mathsf{OA}(\alpha)$ and $\mathsf{OF}(\alpha)$ (Fig. 5) shows that the two curves are indeed inversely monotonic, and their sum recovers the actual excess error curve, elevating the theoretical decomposition from a formal abstraction to an empirically observable dual-component model.

## Highlights & Insights
- **Naming of Theoretical Concepts**: Naming the "failure of small $\alpha$" as **over-alignment**, paired against traditional **over-fitting**, turns the question of "why FLS cannot be infinitely increased" into a geometric story that can be told in one sentence—this naming convention itself is a contribution likely to be cited.
- **From Asymptotic to Finite**: Shifting FLS analysis from asymptotic limits to a "stop at $\eta$" finite-time setting is a significant step in connecting theory with engineering. The surgical use of the stopping time $t_{\eta, \alpha}$ ensures Phase 2 mostly inherits Phase 1, simplifying proofs while remaining realistic.
- **Actionable Takeaway**: It clearly recommends treating FLS as a formal hyperparameter axis (on par with lr and weight decay) and provides the predictive scaling law $c^* \propto n^{-2} h^{-1}$, which is directly applicable to practical tuning.

## Limitations & Future Work
- The theory strictly relies on the orthogonal separability assumption (Assumption 4.1). Whether the inverse monotonicity of OA/OF is guaranteed for general data distributions remains unproven; relaxing this to real-world data is a key open problem.
- It only covers a minimal model (two-layer ReLU + gradient flow + Gaussian mixture) and does not address real-world training components like BN, dropout, Adam, or momentum, which might alter the fixed points of the Phase 1 ODE.
- Experiments were restricted to small-scale vision classification. Verifying whether optimal FLS exists in large-scale models like Transformers/LLMs and whether the same scaling laws apply to complex data is the most promising future direction.

## Related Work & Insights
- **vs woodworth2020kernel / atanasov2025the**: These works argue that "stronger feature learning is always better," but utilize asymptotic/online settings. This paper directly counters this with finite-time analysis for "stop at $\eta$" and provides the geometric mechanism for optimal FLS.
- **vs petrini2022learning**: Also studies FLS and generalization but only chooses between two extreme regimes (spherical regression and infinite width). This paper proves the optimal FLS lies between these regimes and holds universally in standard classification.
- **vs masarczyk2025unpacking / agarwala2023temperature**: Previous works empirically observed that temperature scaling has an optimal value but lacked theoretical explanation. This paper provides the first rigorous framework through OA/OF decomposition, elevating empirical observations to predictable scaling laws.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The conceptual decomposition of over-alignment vs over-fitting is a genuine insight, providing the first rigorous proof of optimal FLS in standard classification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across multiple architectures, datasets, stopping criteria, and cross-scale scaling law validations.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from empirical to theoretical to scaling laws. Figure 4 visualizes the abstract proofs well.
- Value: ⭐⭐⭐⭐⭐ Rewrites common knowledge in implicit bias literature while providing actionable tuning rules, serving both theory and engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](../../ACL2025/others/revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)
- [\[CVPR 2026\] Align Once to Explain: Feature Alignment for Scalable B-cosification of Foundational Vision Transformers](../../CVPR2026/others/align_once_to_explain_feature_alignment_for_scalable_b-cosification_of_foundatio.md)
- [\[ICML 2026\] Return-to-Go is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](return-to-go_is_more_than_a_number_q-guided_alignment_for_return-conditioned_sup.md)
- [\[CVPR 2026\] Data-Centric Meta-Learning for Robust Few-Shot Generalization](../../CVPR2026/others/data-centric_meta-learning_for_robust_few-shot_generalization.md)
- [\[CVPR 2026\] On the Role of Temporal Granularity in the Robustness of Spiking Neural Networks](../../CVPR2026/others/on_the_role_of_temporal_granularity_in_the_robustness_of_spiking_neural_networks.md)

</div>

<!-- RELATED:END -->
