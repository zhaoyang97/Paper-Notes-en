---
title: >-
  [Paper Note] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data
description: >-
  [NeurIPS 2025][Optimization][implicit bias] This paper provides the first complete characterization of the implicit bias of Normalized Steepest Descent (NSD) and Normalized Momentum Descent (NMD) on multiclass linearly s…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "implicit bias"
  - "spectral descent"
  - "Muon"
  - "margin maximization"
  - "multiclass classification"
  - "Schatten norm"
date: 2026-05-08
content_hash: 7067bcd709da8902
---

# Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data

**Conference**: NeurIPS 2025
**arXiv**: [2502.04664](https://arxiv.org/abs/2502.04664)
**Code**: None
**Area**: Optimization
**Keywords**: implicit bias, spectral descent, Muon, margin maximization, multiclass classification, Schatten norm

## TL;DR
This paper provides the first complete characterization of the implicit bias of Normalized Steepest Descent (NSD) and Normalized Momentum Descent (NMD) on multiclass linearly separable data: these algorithms converge to the maximum-margin solution under the corresponding $p$-norm at a rate of $\mathcal{O}(1/\sqrt{t})$, with Spectral Descent (spectral norm) and Muon as special cases, and further extended to Adam (max-norm margin).

## Background & Motivation
**Background**: Adam/AdamW are the de facto standard optimizers for large language model training. Muon orthogonalizes updates via Newton-Schulz iterations (i.e., spectral descent) and has demonstrated strong performance on NanoGPT, with recent extensions to large-scale LLM training.

**Theoretical Gap**: Existing theoretical work on Spectral Descent / Muon has focused primarily on convergence rates (e.g., gradient norm decay in non-convex settings), leaving their implicit bias—i.e., which solutions are preferred in overparameterized models—largely unanalyzed.

**Core Problem**: Under multiclass linearly separable data with cross-entropy loss, what is the implicit bias of Spectral Descent and its momentum variants?

**Richness of the Multiclass Setting**: In multiclass classification, the parameter is a matrix rather than a vector, which naturally accommodates the Schatten norm family (spectral norm, nuclear norm, Frobenius norm), making the analysis richer than the binary case and more suitable for studying spectral descent-type algorithms.

## Method

### Problem Setup
- Multiclass linear model: $W \in \mathbb{R}^{k \times d}$, input $h_i \in \mathbb{R}^d$, label $y_i \in [k]$
- Cross-entropy loss: $\mathcal{L}(W) = -\frac{1}{n} \sum_{i} \log \mathbb{S}_{y_i}(W h_i)$
- Maximum margin: $\gamma := \max_{\|W\| \leq 1} \min_{i, c \neq y_i} (e_{y_i} - e_c)^\top W h_i$

### Normalized Steepest Descent (NSD)
The update direction is:

$$\Delta_t := \arg\max_{\|\Delta\| \leq 1} \langle \nabla_t, \Delta \rangle$$

- **Max-norm** → SignGD: $\Delta_t = \text{sign}(\nabla_t)$
- **Frobenius norm** → NGD: $\Delta_t = \nabla_t / \|\nabla_t\|_2$
- **Spectral norm** → Spectral-GD: $\Delta_t = U_t V_t^\top$ (where $\nabla_t = U_t \Sigma_t V_t^\top$)

### Normalized Momentum Descent (NMD)
Same as NSD but the steepest direction is taken with respect to the momentum $M_t$ rather than the gradient:

$$M_t = \beta_1 M_{t-1} + (1-\beta_1) \nabla_t, \quad \Delta_t := \arg\max_{\|\Delta\| \leq 1} \langle M_t, \Delta \rangle$$

- **Spectral norm** → **Muon**: $\Delta_t = \tilde{U}_t \tilde{V}_t^\top$ (SVD applied to $M_t$)

### Unified Analysis Framework: Surrogate Function $\mathcal{G}(W)$

**Core Idea**: A surrogate function is constructed to unify all NSD/NMD variants:

$$\mathcal{G}(W) := \frac{1}{n} \sum_{i \in [n]} (1 - \mathbb{S}_{y_i}(W h_i))$$

**Key Properties**:
1. **Dual-norm lower bound on gradient** (Lemma 1): $\|\nabla \mathcal{L}(W)\|_* \geq \gamma \cdot \mathcal{G}(W)$
2. **Upper bound on second-order terms** (Lemma 2): Hessian terms are controlled by $\mathcal{G}(W) \cdot \|\Delta\|^2$
3. **Approximate equivalence with loss** (Lemma 3): $\mathcal{L}(W) \leq 2\mathcal{G}(W)$ when the loss is sufficiently small

Unification follows from norm ordering relations: $\|A\|_{\max} \leq \|||A|||_p \leq \|A\|_{\text{sum}}$ (for all entry-wise and Schatten $p$-norms).

### Main Theorems

**Theorem 1 (NSD Margin Convergence)**: With learning rate $\eta_t = \Theta(1/\sqrt{t})$, the margin gap of NSD satisfies:

$$\gamma - \frac{\min_{i,c \neq y_i} (e_{y_i} - e_c)^\top W_t h_i}{\|W_t\|} \leq \mathcal{O}\left(\frac{\log t + n}{\sqrt{t}}\right)$$

**Theorem 2 (NMD Margin Convergence)**: The margin gap of NMD (including Muon) is $\mathcal{O}\left(\frac{d\log t + dn}{\sqrt{t}}\right)$.

**Adam Extension**: Adam (without the $\epsilon$ constant) converges to the max-norm maximum-margin solution at rate $\mathcal{O}\left(\frac{d\log t + nd}{t^{1/3}}\right)$.

### Key Technique for NMD: Class-wise Surrogate Decomposition
Class-wise surrogate functions $\mathcal{G}_c(W)$ and $\mathcal{Q}_c(W)$ are defined to exploit favorable properties of the softmax, enabling control of the sum-norm of the momentum-gradient discrepancy $\Omega_t = M_t - \nabla_t$:

$$\|\Omega_t\|_{\text{sum}} \leq 2B\beta_1^{t/2} \mathcal{G}(W_t) + 2\alpha_M d \eta_t \mathcal{G}(W_t)$$

This avoids an extra factor of $k$ that would arise in a naive analysis.

## Key Experimental Results

### Synthetic Data Experiments ($k=10$, $d=25$, 50 samples per class)

| Algorithm | Preferred Margin Norm | Correlation with $V_\infty$ | Correlation with $V_2$ | Correlation with $V_{\text{spec}}$ |
|------|-------------------|---------------------|----------------|--------------------------|
| SignGD | Max-norm ✅ | High | Low | Low |
| NGD | 2-norm ✅ | Low | High | Low |
| Spectral-GD | Spectral norm ✅ | Low | Low | High |
| Muon | Spectral norm ✅ | Low | Low | High |
| Signum | Max-norm ✅ | High | Low | Low |
| NMD-GD | 2-norm ✅ | Low | High | Low |

### Two-Layer Neural Network Experiments (MNIST, hidden dimension 100)

| Setting | Observation |
|------|---------|
| Train first layer only | Spectral-GD and Muon exhibit the fastest growth in spectral-norm margin $\gamma_a^V$ |
| Joint training of both layers | Spectral-GD and Muon still exhibit the fastest growth in $\gamma_b^{V,W}$ |
| Comparison with SGD variants | SignGD/NGD show notably slower growth in spectral-norm margin than Spectral-GD/Muon |

**Key Finding**: The norm preference observed in the linear setting persists in the nonlinear setting.

## Highlights & Insights
- **First non-asymptotic implicit bias result for Spectral-GD/Muon**: A concrete $\mathcal{O}(1/\sqrt{t})$ margin convergence rate is established on multiclass separable data.
- **Elegance of the unified framework**: Through norm ordering and the surrogate function $\mathcal{G}(W)$, a single analysis covers all entry-wise and Schatten norm families.
- **Class-wise decomposition technique**: Eliminates an extra factor of $k$ in the NMD analysis, simplifying the proof and improving the result.
- **Practical significance**: Provides theoretical grounding for understanding why emerging optimizers such as Muon and Shampoo perform well in LLM training—they implicitly bias toward spectral-norm margin maximization.

## Limitations & Future Work
- **Extra $d$ factor in NMD rate**: Theorem 2 incurs an additional $d$ compared to Theorem 1; whether this can be removed is an open problem.
- **Adam rate is only $\mathcal{O}(t^{-1/3})$**: Slower than the $\mathcal{O}(t^{-1/2})$ rate of NSD/NMD, leaving room for improvement.
- **Restricted to linear multiclass models**: Extension to homogeneous and non-homogeneous neural networks is an important direction.
- **Linearly separable assumption**: Real LLM training data does not satisfy linear separability.
- **Muon implementation discrepancy**: Practical Muon uses Newton-Schulz iterations to approximate the SVD, whereas the theoretical analysis assumes exact SVD.

## Rating
- Novelty: ⭐⭐⭐⭐ Incorporates Muon/Spectral Descent into a unified implicit bias framework, filling an important gap.
- Theoretical Depth: ⭐⭐⭐⭐⭐ The surrogate function construction and norm ordering techniques are highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic data experiments plus nonlinear extensions adequately validate the theoretical predictions.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and effective use of summary tables, though the dense notation requires careful reading.
- Value: ⭐⭐⭐⭐ Makes an important contribution to understanding the implicit bias of modern optimizers (Muon, Shampoo, Adam).

## Related Work & Insights

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)

</div>

<!-- RELATED:END -->
