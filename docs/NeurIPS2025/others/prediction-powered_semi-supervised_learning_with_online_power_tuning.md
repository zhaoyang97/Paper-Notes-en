---
title: >-
  [Paper Note] Prediction-Powered Semi-Supervised Learning with Online Power Tuning
description: >-
  [NeurIPS 2025][Semi-supervised learning] This paper extends the Prediction-Powered Inference (PPI) framework to the training phase of semi-supervised learning. It proposes an unbiased gradient estimator and designs an on…
tags:
  - "NeurIPS 2025"
  - "Semi-supervised learning"
  - "prediction-powered inference"
  - "online learning"
  - "pseudo-label debiasing"
  - "AdaGrad"
date: 2026-05-08
content_hash: bc6b7eeebfe46197
---

# Prediction-Powered Semi-Supervised Learning with Online Power Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2510.22586](https://arxiv.org/abs/2510.22586)
**Code**: [GitHub](https://github.com/noashoham/PP-SSL)
**Area**: Semi-Supervised Learning / Statistical Inference
**Keywords**: Semi-supervised learning, prediction-powered inference, online learning, pseudo-label debiasing, AdaGrad

## TL;DR

This paper extends the Prediction-Powered Inference (PPI) framework to the training phase of semi-supervised learning. It proposes an unbiased gradient estimator and designs an online AdaGrad algorithm to dynamically tune the interpolation parameter $\lambda$ between pseudo-labels and true labels, achieving convergence rates matching the optimal fixed $\lambda$ while maintaining unbiasedness.

## Background & Motivation

Semi-supervised learning (SSL) leverages large amounts of unlabeled data to augment training with limited labeled data. Pseudo-labeling is one of its core strategies: a teacher model generates artificial labels for unlabeled data to expand the training set. However, when the teacher model performs poorly on certain subgroups, pseudo-labels introduce bias (confirmation bias), causing the student model to perform worse on those subgroups than if trained on labeled data alone.

The PPI framework debiases by computing the loss difference between true labels and pseudo-labels on labeled data: $L_{\text{PPI}} = L_n + \tilde{L}_N^f - L_n^f$, so that pseudo-label bias is canceled in expectation. PPI++ further introduces an interpolation parameter $\lambda \in [0,1]$ to balance the contributions of labeled and pseudo-labeled data. However, two critical limitations remain:

**Analysis holds only at the asymptotic optimum**, providing no finite-time guarantees on how pseudo-labels affect convergence during training.

**$\lambda$ selection relies on offline estimation**, requiring prior knowledge of the labeled data variance $\sigma^2$ and teacher model accuracy $\mathcal{E}^f$, which are unknown in practice, and fixed estimates may be severely suboptimal.

The core motivation of this paper is to extend PPI from offline statistical inference to online SSL training, while employing online learning to adaptively tune $\lambda$.

## Method

### Overall Architecture

PP-SSL is an unbiased semi-supervised learning algorithm within a teacher-student framework. At each iteration:

1. Obtain $n$ labeled samples and $N$ unlabeled samples ($N \gg n$)
2. Generate pseudo-labels for unlabeled data using a fixed teacher model $f$
3. Construct a prediction-powered gradient estimator
4. Jointly update model parameters $w$ and the interpolation parameter $\lambda$

### Key Designs

1. **Prediction-Powered (PP) Gradient Estimator**: The core formula is $g_{\text{PP}}^{\lambda} = g^n + \lambda(\tilde{g}^{N,f} - g^{n,f})$, where $g^n$ is the standard mini-batch gradient on labeled data, $\tilde{g}^{N,f}$ is the pseudo-label gradient on unlabeled data, and $g^{n,f}$ is the pseudo-label gradient on labeled data. **Unbiasedness guarantee**: Since labeled and unlabeled data are drawn from the same marginal distribution $\mathbb{P}_X$, we have $\mathbb{E}[\tilde{g}^{N,f} - g^{n,f}] = 0$, hence $\mathbb{E}[g_{\text{PP}}^{\lambda}] = \nabla \mathcal{L}(w)$. **Design motivation**: This corrects the bias introduced by pseudo-labels, ensuring unbiased gradient estimates even when the teacher model is inaccurate.

2. **Variance Analysis and Optimal $\lambda$**: The gradient variance upper bound is $(1-\lambda)^2 \sigma^2 + \lambda^2(r\sigma^2 + (1+r)\sigma_e^2)$, where $r = n/N$ and $\sigma_e^2$ relates to the teacher prediction error $\mathcal{E}^f$ via $\sigma_e^2 \leq L_Y^2 \cdot \mathcal{E}^f$. Minimizing yields $\lambda^* = \frac{1}{1+r} \cdot \frac{\sigma^2}{\sigma^2 + \sigma_e^2}$. When the teacher is accurate ($\sigma_e^2 \ll \sigma^2$), $\lambda^* \to 1/(1+r)$ and variance is substantially reduced; when the teacher is poor, $\lambda^* \to 0$, gracefully degenerating to labeled-only training. **Design motivation**: Establishes a quantitative relationship among teacher quality, data ratio, and variance reduction, providing a theoretical target for online tuning.

3. **Online AdaGrad Tuning of $\lambda$**: Cumulative variance minimization is cast as an online learning problem. Define $h_t(\lambda) = \|g_t^n + \lambda(\tilde{g}_t^{N,f} - g_t^{n,f})\|^2$, which is convex in $\lambda$. A one-dimensional AdaGrad update is applied: $\lambda_{t+1} = \text{clamp}(\lambda_t - \gamma_t \nabla h_t(\lambda_t); 0, 1)$, with adaptive learning rate $\gamma_t = (2\sum_{s=1}^{t}\|\nabla h_s(\lambda_s)\|^2)^{-1/2}$. **Design motivation**: AdaGrad requires no knowledge of $\sigma^2$ or $\sigma_e^2$ and automatically adapts to changes in teacher quality; the online regret bound ensures that dynamic tuning incurs only a lower-order additive cost.

### Loss & Training

Model parameters $w$ are updated with an AdaGrad step: $w_{t+1} = w_t - \eta_t g_t^{\lambda_t}$, with $\eta_t = \eta_0 (\sum_{s=1}^{t}\|g_s^{\lambda_s}\|^2)^{-1/2}$. The convergence rate is $\mathcal{O}(\sqrt{M\beta V^*/T} + M\beta/T + \sqrt{M\beta}G/T)$, where the third term is the additive cost from online learning, which decays much faster than the dominant term $\mathcal{O}(\sqrt{V^*/T})$. Thus the overall convergence rate matches that of using the optimal fixed $\lambda^*$.

## Key Experimental Results

### Main Results

| Dataset | Metric | PP-SSL | PPI++ | SSL | Only Labeled | Notes |
|--------|------|--------|-------|-----|-------------|------|
| Synthetic linear regression (high bias $\mu=7$) | MSE (full) | **Lowest** | 2nd lowest | High | Moderate | PP-SSL shows clear advantage under high bias |
| Synthetic linear regression (Group B) | MSE | **Lowest** | 2nd lowest | Highest | Moderate | SSL performs worse than labeled-only |
| California Housing | MSE | Close to PPI++ | Close to PP-SSL | Higher | Higher | Both debiasing methods outperform baselines |
| UTKFace age estimation | MAE | **Lowest** | 2nd lowest | Highest | Moderate | PP-SSL also benefits deep models |
| CIFAR-10 (with corruption) | Accuracy | **Highest** | 2nd highest | Low | Moderate | Largest gain on Group B accuracy |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Fixed $\lambda$ (sweep 0–1) | MSE | PP-SSL matches the best fixed $\lambda$ without manual tuning |
| Varying bias level $\mu$ | Group B MSE | Larger bias leads to greater advantage of PP-SSL over SSL |
| Varying $N_B/N_A$ ratio | MSE | Debiasing is most effective when teacher quality is poor |
| With/without group indicator | MSE | PP-SSL remains effective even without group membership information |

### Key Findings

- Conventional SSL can **perform worse than labeled-only training** when the teacher model performs poorly on certain subgroups; the proposed method effectively resolves this via debiasing.
- Online tuning of $\lambda$ automatically adapts to teacher quality without requiring prior variance or error estimation.
- This work provides the first finite-time convergence guarantees for PPI-type methods, going beyond purely asymptotic analysis.
- Consistent improvements are demonstrated on deep neural networks (ResNet50).

## Highlights & Insights

- **Elegant unification of theory and practice**: The derivation starts from variance analysis to obtain the optimal $\lambda$, then naturally transitions to online learning upon recognizing the dependence on unknown quantities — the logical chain is remarkably tight.
- **Clever construction of AdaGrad for $\lambda$**: $h_t(\lambda)$ is exactly a convex quadratic in $\lambda$, enabling theoretical guarantees for AdaGrad; the AdaGrad step sizes for both $w$ and $\lambda$ implicitly adapt to $\sigma^2$ and $\sigma_e^2$.
- **High practical value**: In the current LLM era where pseudo-labels and synthetic data are extensively used, this method provides a principled framework for balancing the contributions of real and synthetic data.

## Limitations & Future Work

- The framework assumes labeled and unlabeled data share the same distribution; in practice, unlabeled data may come from a different domain or be generated by generative models.
- The theoretical analysis assumes a fixed teacher model; self-training scenarios with continuously updated teachers require more complex dynamic regret analysis.
- The convergence analysis targets approximate stationary points in non-convex optimization, with no guarantees for global optima.
- The tightness of the variance upper bound depends on the gradient Lipschitz constant $L_Y$, an assumption that may be strong for certain loss functions.

## Related Work & Insights

- **Comparison with Doubly-Robust Self-Training**: The latter also employs a PPI-like loss but controls $\lambda$ via a predefined step function and provides only asymptotic guarantees.
- **Comparison with PPI++**: PPI++ tunes $\lambda$ offline (minimizing asymptotic variance), whereas PP-SSL tunes it online (minimizing cumulative second moment), making the latter more robust in practice.
- **Insight**: The paradigm of "debiasing + online parameter tuning" may be equally applicable to settings such as knowledge distillation and learning from synthetic data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The extension from PPI to online SSL is clearly motivated and makes a substantial contribution; the design of online $\lambda$ tuning is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Coverage spans synthetic, real-world, and deep learning settings with thorough ablations, though large-scale benchmarks (e.g., ImageNet) are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous exposition, complete theoretical derivations, consistent notation, and easy to follow.
- **Value**: ⭐⭐⭐⭐ — Highly practical given the widespread use of pseudo-labels and synthetic data today, supported by solid theoretical foundations.

## Additional Notes

- Lemma 3.1 establishes a bridge between the Lipschitz continuity of the loss gradient with respect to labels and the teacher error $\mathcal{E}^f$, applicable to squared loss and logistic regression loss.
- The method naturally fits the teacher-student framework where the teacher is fixed and the student is trained independently, offering better stability than self-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[NeurIPS 2025\] Computable Universal Online Learning](computable_universal_online_learning.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)

</div>

<!-- RELATED:END -->
