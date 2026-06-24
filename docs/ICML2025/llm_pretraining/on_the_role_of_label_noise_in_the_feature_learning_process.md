---
title: >-
  [Paper Note] On the Role of Label Noise in the Feature Learning Process
description: >-
  [ICML 2025][LLM Pretraining][label noise] The training dynamics of a two-layer ReLU CNN under label noise are rigorously analyzed from the perspective of feature learning theory. This analysis reveals a clear two-stage behavior: in Stage I, the model learns the signal to fit clean samples (achieving good generalization); in Stage II, after loss convergence, the model memorizes noise to overfit noisy samples (degrading generalization). This provides rigorous theoretical guaran…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "label noise"
  - "feature learning"
  - "training dynamics"
  - "early stopping"
  - "sample selection"
  - "CNN"
date: 2026-05-08
content_hash: d606951747cac6d5
---

# On the Role of Label Noise in the Feature Learning Process

**Conference**: ICML 2025  
**arXiv**: [2505.18909](https://arxiv.org/abs/2505.18909)  
**Code**: None  
**Area**: Learning Theory  
**Keywords**: label noise, feature learning, training dynamics, early stopping, sample selection, CNN

## TL;DR

The training dynamics of a two-layer ReLU CNN under label noise are rigorously analyzed from the perspective of feature learning theory. This analysis reveals a clear two-stage behavior: in Stage I, the model learns the signal to fit clean samples (achieving good generalization); in Stage II, after loss convergence, the model memorizes noise to overfit noisy samples (degrading generalization). This provides rigorous theoretical guarantees for early stopping and small-loss sample selection.

## Background & Motivation

**Background**: When facing label noise, overparameterized deep networks are prone to overfitting, leading to degraded generalization. Although many practical methods (such as early stopping, sample selection, and label correction) have been developed, the theoretical understanding of why these methods are effective remains insufficient.

**Limitations of Prior Work**: Existing theoretical analyses are mostly limited to the lazy training regime (the NTK framework), which requires the weights to not deviate far from initialization or the network to be infinitely wide. This essentially represents the linear dynamics of a static kernel and cannot capture genuine feature learning behavior. Although Frei et al. (2021) analyzed the early training stage, that phase is indistinguishable from a linear classifier.

**Key Challenge**: Empirical observations clearly show that networks "learn simple patterns first, and memorize noise later." However, a complete mathematical characterization under feature learning theory is missing. The fundamental difficulty is that when $n \cdot \text{SNR}^2 = \Theta(1)$, the signal and noise are of the same order of magnitude, making their dynamics closely intertwined and difficult to decouple.

**Goal**: This paper aims to fully characterize the impact of label noise on training dynamics within the theoretical framework of feature learning, revealing the two-stage mechanism and providing correctness proofs for early stopping and sample selection.

**Key Insight**: A signal-noise data distribution is leveraged (where each sample consists of a label-dependent signal patch and a label-independent noise patch), under the critical condition that $n \cdot \text{SNR}^2 = \Theta(1)$, which puts signal and noise in a competitive relationship.

**Core Idea**: Label noise creates a "race" during training between signal learning and noise memorization. The former wins first, but the latter eventually catches up. The transition point marks the optimal position for early stopping.

## Method

### Overall Architecture

The analysis is based on three pillars: 
(1) Data: $\mathbf{x} = [y\boldsymbol{\mu}, \boldsymbol{\xi}]$, where the signal $\boldsymbol{\mu}$ is fixed, noise $\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma_\xi^2 \mathbf{I}_d)$, and labels are flipped with probability $\tau$.
(2) A two-layer ReLU CNN: $f = F_{+1} - F_{-1}$.
(3) Signal-noise decomposition: $\mathbf{w}_{j,r}^{(t)} = \mathbf{w}_{j,r}^{(0)} + j\gamma_{j,r}^{(t)}\|\boldsymbol{\mu}\|^{-2}\boldsymbol{\mu} + \sum_i \rho_{j,r,i}^{(t)}\|\boldsymbol{\xi}_i\|^{-2}\boldsymbol{\xi}_i$, where $\gamma$ tracks signal learning and $\rho$ tracks noise memorization.

### Key Designs

1. **Stage I Analysis (Theorem 4.1)**:

    - **Function**: Characterizes the early training phase, where the model fits clean samples and ignores noisy samples.
    - **Mechanism**: At $T_1 = \Theta(\eta^{-1}nm\sigma_\xi^{-2}d^{-1})$, both the signal coefficients and noise coefficients reach $\Theta(1)$, but $\gamma_{j,r}^{(T_1)} > \bar{\rho}_{\tilde{y}_i,r,i}^{(T_1)}$ strictly holds—meaning signal learning strictly dominates noise memorization. In this stage, the loss derivatives $|\ell_i'|$ of all samples have a constant lower-bound, keeping gradient contributions balanced. In clean samples, signal and noise synergize ($\tilde{y}=y$), while in noisy samples, they conflict ($\tilde{y} \neq y$). Consequently, clean samples are classified correctly, while noisy samples are classified towards their **true labels** (meaning the model "rejects" the incorrect label).
    - **Design Motivation**: The condition $n \cdot \text{SNR}^2 = \Theta(1)$ ensures that the signal only "weakly" dominates the noise, which is a necessary condition for the two-stage behavior.

2. **Stage II Analysis (Theorem 4.2)**:

    - **Function**: Characterizes the late training phase, where the loss converges and the network overfits noisy samples.
    - **Mechanism**: For the training loss to converge, **all** samples (including those with noisy labels) must be classified correctly. For noisy samples, since the signal direction opposes the noisy label, this can only be achieved by increasing the noise coefficients $\bar{\rho}$. Eventually, the noise coefficients of at least $\tau'n$ noisy samples exceed their signal coefficients, resulting in an unavoidable test error lower-bounded by $\geq 0.5\min\{\tau_+, \tau_-\}$. The proof uses proof by contradiction: if not enough noisy samples were overfitted, the training loss would fail to converge.
    - **Design Motivation**: Proves that overfitting is not optional—as long as training continues until the loss converges, the model will **inevitably** memorize the noise.

3. **Guarantees for Early Stopping and Sample Selection (Proposition 4.3)**:

    - **Function**: Provides rigorous theoretical support for two widely used practical techniques.
    - **Mechanism**: **Early stopping**: Stopping at $T_1$ yields a test error of $\leq \exp(-dn^{-1}/C')$. **Sample selection**: At $T_1$, the loss of clean samples is $\leq \log 2$ while that of noisy samples is $\geq \log 2$. Thus, the threshold of $\log 2$ achieves perfect separation.
    - **Design Motivation**: Although $T_1$ cannot be calculated precisely in practice, the theory guarantees the existence of an optimal stopping point, for which validation accuracy can serve as a practical proxy.

### Loss & Training

The model uses logistic loss $\ell(f, \tilde{y}) = \log(1 + \exp(-f \cdot \tilde{y}))$, trained via full-batch gradient descent (GD) with a constant learning rate.

## Key Experimental Results

### Main Theorem Overview

| Theorem | Phase | Conclusion |
|------|------|------|
| Thm 4.1 | Stage I ($t=T_1$) | $\gamma > \rho$; clean samples are all correct; noisy samples classified by true labels |
| Thm 4.2 | Stage II (Loss convergence) | Clean samples remain correct; $\tau'n$ noisy samples have $\rho>\gamma$; test error $\geq 0.5\min\{\tau_+,\tau_-\}$ |
| Thm 4.4 | Noiseless baseline | All samples are always correct, test error is exponentially small |
| Prop 4.3 | Early stopping / Selection | Stopping at $T_1$ yields test error $\leq \exp(-d/nC')$; $\log(2)$ threshold achieves perfect separation |

### Noise vs. Noiseless Comparison

| Setting | Stage I Test Error | Test Error after Loss Convergence |
|------|---------------|-----------------|
| Label Noise-Free | Low | Remains low ($\leq \exp(-n\|\mu\|^4 / C_D\sigma_\xi^4 d)$) |
| With Label Noise ($\tau>0$) | Low (near 0) | **Inevitably high** ($\geq 0.5\min\{\tau_+,\tau_-\}$) |

### Key Findings

- The mechanism for the transition between the two stages is clear: in Stage I, the loss derivative $|\ell_i'|$ is uniform across all samples, balancing gradient contributions. In Stage II, the loss derivative of clean samples $|\ell_i'| \to 0$ (as they are already fitted), leaving the $|\ell_i'|$ of noisy samples to dominate and drive noise memorization.
- The discovery of $\log 2$ as the threshold for loss separation is determined solely by the value of the logistic function at the decision boundary, independent of the data distribution, indicating its universality.
- The key distinction from Kou et al. (2023) is that under their condition of $n \cdot \text{SNR}^2 \gg 1$, the signal always dominates, preventing the two-stage behavior from occurring.
- Although the signal coefficient $\gamma$ might temporarily decrease during Stage II, it remains strictly positive. The model does not "forget" the signal; rather, the noise is superimposed on top of it.

## Highlights & Insights

- This is the first work to fully characterize the two-stage feature learning behavior under label noise, moving beyond the linear analysis of the lazy regime.
- The elegant and practical discovery of the $\log 2$ threshold provides the first rigorous theoretical justification for small-loss selection methods like Co-teaching.
- The proof by contradiction used for Stage II is sophisticated: assuming that not enough noise is overfitted leads to the contradiction that the loss cannot converge.
- The comparison with the noiseless setting clearly demonstrates the "cost" of label noise: the test error has a positive lower bound with noise, whereas it can be exponentially small without noise.

## Limitations & Future Work

- The theory is limited to two-layer CNNs, binary classification, and a signal-noise data distribution, which still leaves a gap compared to practical deep networks and natural data.
- The condition $n \cdot \text{SNR}^2 = \Theta(1)$ requires the signal strength and sample size to match precisely, whereas real-world SNR varies widely.
- Instance-dependent label noise (a more realistic noise model) is not considered.
- Full-batch GD is analyzed instead of SGD, leaving the effects of mini-batch stochasticity unaddressed.
- No practical method for calculating $T_1$ is provided.

## Related Work & Insights

- **vs. Kou et al. (2023)**: Both study feature learning theory, but Kou et al. require $n \cdot \text{SNR}^2 \gg 1$, where the two-stage phenomenon does not occur. The $\Theta(1)$ condition in this work captures the correct regime.
- **vs. Frei et al. (2021)**: This prior work focuses only on the early training phase and does not capture the degradation caused by noise. The current work covers the complete training process.
- **vs. Li et al. (2020)**: This prior work relies on a lazy regime analysis and cannot describe feature learning. This work operates in the rich regime.
- **Insights**: The theory implies that "free training followed by constraint/stopping" is more reasonable than "regularization from the very beginning."

## Rating

- **Novelty**: ⭐⭐⭐⭐ Fully characterizes the two-stage feature learning behavior under label noise for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐ Primarily theoretical, with validation on synthetic data and small-scale CIFAR.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression from theorems to intuitions and then to proof ideas.
- **Value**: ⭐⭐⭐⭐ Provides a solid theoretical foundation for early stopping and sample selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[AAAI 2026\] Rectified Noise: A Generative Model Using Positive-incentive Noise](../../AAAI2026/llm_pretraining/rectified_noise_a_generative_model_using_positive-incentive_noise.md)
- [\[ICML 2026\] Focus and Dilution: The Multi-stage Learning Process of Attention](../../ICML2026/llm_pretraining/focus_and_dilution_the_multi-stage_learning_process_of_attention.md)
- [\[ICML 2025\] Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)
- [\[ICML 2025\] Machine Learning from Explanations](machine_learning_from_explanations.md)

</div>

<!-- RELATED:END -->
