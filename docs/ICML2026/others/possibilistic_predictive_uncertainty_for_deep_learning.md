---
title: >-
  [Paper Note] Possibilistic Predictive Uncertainty for Deep Learning
description: >-
  [ICML 2026][Epistemic uncertainty] This paper replaces the Bayesian probabilistic framework with possibility theory to propose DAPPr—projecting the possibilistic posterior in the parameter space onto the predictive space…
tags:
  - "ICML 2026"
  - "Epistemic uncertainty"
  - "possibility theory"
  - "Dirichlet"
  - "second-order predictor"
  - "EDL"
date: 2026-05-08
content_hash: f57a873d7319030d
---

# Possibilistic Predictive Uncertainty for Deep Learning

**Conference**: ICML 2026  
**arXiv**: [2605.00600](https://arxiv.org/abs/2605.00600)  
**Code**: https://github.com/MaxwellYaoNi/DAPPr  
**Keywords**: Epistemic uncertainty, possibility theory, Dirichlet, second-order predictor, EDL

## TL;DR
This paper replaces the Bayesian probabilistic framework with possibility theory to propose DAPPr—projecting the possibilistic posterior in the parameter space onto the predictive space via supremum, fitting it with a learnable Dirichlet possibility function, and ultimately obtaining an epistemic uncertainty modeling method that requires only 10 lines of code, directly replaces cross-entropy, and outperforms the EDL family in OOD detection.

## Background & Motivation
**Background**: It is a well-known pain point that deep networks are overconfident on out-of-distribution samples. Currently, mainstream epistemic uncertainty modeling follows two paths: Bayesian deep learning (BNN / MC Dropout / Deep Ensemble) and second-order predictors (EDL / PostNet / Prior Networks).

**Limitations of Prior Work**: The Bayesian route is theoretically rigorous but requires posterior marginalization in high-dimensional parameter space, which is computationally expensive and difficult to scale. Second-order predictors are efficient, but their objectives are mostly heuristic, lacking rigorous derivation from probabilistic axioms. EDL has even been noted for pathological behavior where "more data leads to higher uncertainty."

**Key Challenge**: A trade-off exists between theoretical rigor and computational feasibility—Bayesian is rigorous but expensive, while second-order is cheap but ad hoc. The authors argue the root cause is treating epistemic uncertainty as probability, whereas the "sum to 1" constraint of probability distributions is naturally more suited for characterizing aleatoric randomness rather than "ignorance."

**Goal**: (1) Find a rigorous uncertainty representation framework that does not require parameter space integration; (2) Derive a training objective with a closed-form solution; (3) Perform a head-to-head comparison with the EDL family on standard benchmarks.

**Key Insight**: The authors start from possibility theory, proposed by Zadeh in 1978 but largely ignored in deep learning, which replaces integration with supremum and sum-to-1 with max-normalization, making it naturally suitable for expressing epistemic information such as "which hypotheses cannot be excluded."

**Core Idea**: The possibilistic posterior of model parameters is projected onto the simplex via supremum, then parametrically approximated on the simplex using a Dirichlet possibility function; the entire pipeline yields a closed-form solution under cross-entropy.

## Method
The elegance of DAPPr lies in how a "projected posterior," which originally requires constrained optimization in high-dimensional parameter space, is compressed into 10 lines of PyTorch code through the triplet of over-parameterized assumption + Dirichlet parameterization + Danskin's Theorem.

### Overall Architecture
The input is a standard classification sample $(\bm{x}, \bm{y})$; the model $\Phi'_{\bm{\psi}}$ outputs Dirichlet parameters $\bm{\alpha} = \mathrm{softplus}(\mathrm{logits}) + 1$; a Dirichlet possibility function $g_{\bm{\psi}}(\bm{p}|\bm{x})$ is defined; during inference, $1 - \max_k \alpha_k / \alpha_0$ is used to calculate aleatoric uncertainty and $K / \alpha_0$ to calculate epistemic uncertainty (where $\alpha_0 = \sum_k \alpha_k$ is the total evidence). A "projection + approximation" two-step pipeline is constructed during training, ultimately resulting in a closed-form surrogate loss under cross-entropy.

### Key Designs

1. **Possibilistic Posterior + Supremum Projection**:
    - **Function**: Defines a possibilistic posterior in the parameter space under a uniform prior: $\pi(\bm{\theta}|\mathcal{D}) = \exp(-L(\bm{\theta};\mathcal{D})) / \sup_{\bm{\theta}'}\exp(-L(\bm{\theta}';\mathcal{D}))$, where smaller loss indicates higher plausibility. This is then projected onto the simplex using a possibilistic change-of-variable: $g^*_{\bm{x}}(\bm{p}|\mathcal{D}) = \sup\{\pi(\bm{\theta}|\mathcal{D}) : \Phi_{\bm{\theta}}(\bm{x}) = \bm{p}\}$.
    - **Mechanism**: The marginalization requiring integration in Bayesian frameworks is replaced with constrained optimization under supremum—the fundamental difference between possibility theory and probability theory. Using the over-parameterized assumption (a sufficiently large network can fit any single point without affecting other samples), it is proved that $\inf_{\Phi_{\bm{\theta}}(\bm{x})=\bm{p}} L(\bm{\theta}; \mathcal{D} \setminus \{(\bm{x},\bm{y})\}) \approx c_{\bm{x}}$ is nearly independent of $\bm{p}$, simplifying the projected posterior to $g^*_{\bm{x}}(\bm{p}|\mathcal{D}) \propto \exp(-\ell(\bm{p}, \bm{y}))$.
    - **Design Motivation**: Parameter space integration is the source of high costs in Bayesian methods. Replacing it with sample-wise leave-one-out infimum via supremum and over-parameterization, then approximating it as a constant via capacity assumptions, is a clever two-stage simplification.

2. **Maxitive Pseudo-divergence Training Objective**:
    - **Function**: Uses $D_{\mathrm{max}}(f\|g) = \max_{\theta} \log(f(\theta)/g(\theta))$ to measure the deviation between two possibility functions. The training objective is defined as $\mathcal{L}(\bm{\psi}; \mathcal{D}) = \mathbb{E}_{\bm{x}}[\max_{\bm{p}}(\log g_{\bm{\psi}}(\bm{p}|\bm{x}) - \log g^*_{\bm{x}}(\bm{p}|\mathcal{D}))]$, which essentially penalizes the learned function for maximum pointwise overestimation of the projected posterior.
    - **Mechanism**: This is a min-max problem where the inner maximizer $\bm{p}^*$ depends on $\bm{\psi}$. Danskin's Theorem is used to equate the outer gradient to the derivative with respect to $\bm{\psi}$ at the inner maximizer. Under Dirichlet parameterization, the inner max of the cross-entropy loss has a closed-form solution $\tilde{\bm{p}}^* = (\bm{\alpha} - \bm{y}) / (\alpha_0 - 1)$, requiring $\alpha_k > 1$ (enforced by softplus + 1).
    - **Design Motivation**: Combining "maxitive divergence instead of KL," "Danskin for min-max," and "Dirichlet parameterization for closed-form" turns an abstract possibility theory framework into a differentiable and trainable simple loss, representing the paper's key engineering contribution.

3. **Spurious Evidence Regularization**:
    - **Function**: Adds a regularization term $\mathcal{R}(\bm{x}) = \|(\bm{1} - \bm{y}) \odot \bm{\alpha}\|_2^2$ alongside the cross-entropy surrogate to penalize evidence assigned to incorrect categories.
    - **Mechanism**: The surrogate objective encourages each sample to be fitted with arbitrary precision, which may cause $\alpha_0$ to grow unboundedly, corresponding to unrealistically high evidence. This regularization only penalizes $\alpha$ on wrong classes, keeping total evidence controlled without hindering the growth of evidence for the correct class.
    - **Design Motivation**: A common pain point in the EDL series is the difficulty in controlling evidence. Using a concise mask + L2 to directly limit overconfidence in incorrect categories avoids the complex Fisher regularization used in EDL.

### Loss & Training
The final training objective is as follows (implementable in 10 lines of PyTorch):

$$\ell_{\bm{\psi}}(\bm{x}) = \alpha_0 \log \alpha_0 + \sum_k \alpha_k \log(\tilde{p}^*_k / \alpha_k) + \lambda \|(\bm{1} - \bm{y}) \odot \bm{\alpha}\|_2^2$$

Where $\tilde{\bm{p}}^* = (\bm{\alpha} - \bm{y} + \epsilon) / (\alpha_0 - 1)$ is detached to prevent gradient backpropagation. $\lambda$ controls the regularization strength and is the only explicit hyperparameter.

## Key Experimental Results

### Main Results
Comparison with SOTA EDL family ($\mathcal{I}$-EDL / R-EDL / $\mathcal{F}$-EDL) + Bayesian baselines (MC Dropout / DUQ / PostNet) on MNIST / CIFAR-10 / CIFAR-100:

| Dataset | Metric | DAPPr | $\mathcal{F}$-EDL | R-EDL | $\mathcal{I}$-EDL | EDL |
|------|------|------|------|------|------|------|
| MNIST Test Acc | ↑ | 99.26 | 99.30 | 99.33 | 99.21 | 98.22 |
| MNIST Conf AUPR | ↑ | 99.99 | 99.93 | 99.99 | 99.98 | 99.99 |
| MNIST→KMNIST OOD | ↑ | **98.81** | 98.74 | 98.69 | 98.33 | 96.31 |
| MNIST→FMNIST OOD | ↑ | **99.55** | 99.31 | 99.29 | 98.86 | 98.08 |

DAPPr consistently outperforms the strongest variants of the EDL family in OOD detection, while remaining competitive in accuracy and confidence calibration.

### Ablation Study
The paper performs empirical validation of the over-parameterization assumption, scans the spurious evidence regularization strength $\lambda$, and compares results on more complex benchmarks such as long-tailed distributions, distribution shift detection, and fine-grained classification:

| Configuration | Key Effect | Description |
|------|------|------|
| No regularization ($\lambda = 0$) | Unbounded $\alpha_0$ growth | Fits each sample with arbitrary precision, destroying uncertainty representation. |
| Large $\lambda$ | Suppressed evidence | Overall higher uncertainty, slight drop in accuracy. |
| Moderate $\lambda$ | Optimal trade-off | Highest OOD AUPR. |
| Eq. (11) Approximation Check | Leave-one-out loss nearly independent of $\bm{p}$ | Empirical support for the over-param assumption. |

### Key Findings
- In OOD detection tasks where epistemic uncertainty is crucial, DAPPr consistently surpasses all EDL variants, indicating that objectives derived from possibility theory are more sensitive in OOD scenarios than heuristic EDL.
- Spurious evidence regularization is not just an engineering trick but theoretically caps unbounded behavior from over-fitting single samples, significantly impacting final calibration.
- The closed-form $\tilde{\bm{p}}^*$ allows training costs to match standard cross-entropy without ensemble or sampling overhead, enabling direct replacement in existing pipelines.

## Highlights & Insights
- Introducing possibility theory to deep uncertainty is the largest conceptual contribution—while the academic community has focused almost exclusively on the probability theory framework for decades, the max operator of possibility theory aligns naturally with the epistemic semantics of "cannot be excluded."
- Danskin's Theorem is used elegantly to collapse a min-max problem into a single-layer gradient at the inner-maximizer, avoiding the instability of GAN-style adversarial training.
- The ability to replace cross-entropy with 10 lines of PyTorch code is an engineering-friendly design with near-zero migration cost, which could significantly promote adoption.
- The over-parameterized assumption is a powerful simplification trick—approximating a leave-one-out optimization problem as a constant. This approach can be transferred to other methods involving parameter space integration (e.g., influence functions, data attribution).

## Limitations & Future Work
- The over-parameterization assumption might fail in underparameterized scenarios or when samples are highly sensitive (e.g., few-shot / multi-task conflicts). While empirical validation is provided, theoretical boundary characterization is lacking.
- The spurious evidence regularization strength $\lambda$ is the only explicit hyperparameter and still requires tuning for new datasets; an adaptive version could be considered in the future.
- Currently, Dirichlet approximation is only performed on the simplex for classification tasks; extending this to more complex output spaces like regression or structured prediction requires finding new families of possibility functions.
- Comparison with calibration methods like conformal prediction is missing; it is currently unclear if DAPPr's uncertainty can be directly converted into guaranteed coverage intervals.

## Related Work & Insights
- **Ours vs. EDL Family**: EDL is based on subjective logic / Dempster-Shafer theory with heuristic objectives; DAPPr strictly derives the objective from possibility theory and consistently outperforms the strongest EDL variants in OOD.
- **Ours vs. Bayesian Deep Learning (BNN/MC Dropout/Deep Ensemble)**: Bayesian routes require ensembles or sampling; DAPPr uses a single model for inference with costs identical to standard classification while still expressing epistemic uncertainty.
- **Ours vs. PostNet / Natural Posterior Networks**: Those methods use normalizing flows to fit the posterior, which is complex and requires extra components; DAPPr is much simpler, using Dirichlet parameterization and a closed-form maximizer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introduces possibility theory to deep epistemic uncertainty for the first time, with a novel theoretical foundation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across MNIST / CIFAR / long-tail / distribution shift / fine-grained benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Rigorous and clear derivation, building step-by-step from possibility concepts to closed-form solutions.
- Value: ⭐⭐⭐⭐⭐ Extremely high engineering value, achieving SOTA OOD by replacing cross-entropy with 10 lines of code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](../../NeurIPS2025/others/uncertainty_estimation_by_flexible_evidential_deep_learning.md)
- [\[ICML 2026\] Sequential Group Composition: A Window into the Mechanics of Deep Learning](sequential_group_composition_a_window_into_the_mechanics_of_deep_learning.md)
- [\[ICML 2026\] Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations](rectified_lpjepa_joint-embedding_predictive_architectures_with_sparse_and_maximu.md)
- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)

</div>

<!-- RELATED:END -->
