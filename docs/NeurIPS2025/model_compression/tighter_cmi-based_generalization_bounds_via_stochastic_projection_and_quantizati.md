---
title: >-
  [Paper Note] Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization
description: >-
  [NeurIPS 2025][Model Compression][generalization bounds] By incorporating **stochastic projection** and **lossy compression** into the CMI (conditional mutual information) framework…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "generalization bounds"
  - "conditional mutual information"
  - "stochastic projection"
  - "lossy compression"
  - "memorization"
date: 2026-05-08
content_hash: 49b084862dd091be
---

# Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2510.23485](https://arxiv.org/abs/2510.23485)  
**Code**: None  
**Area**: Model Compression
**Keywords**: generalization bounds, conditional mutual information, stochastic projection, lossy compression, memorization

## TL;DR

By incorporating **stochastic projection** and **lossy compression** into the CMI (conditional mutual information) framework, this paper derives tighter generalization bounds, resolves the failure of classical CMI bounds on SCO counterexamples, and proves that memorization is not necessary for good generalization.

## Background & Motivation

**Background**: Information-theoretic methods are important tools for analyzing the generalization error of learning algorithms. Mutual information (MI) bounds [Xu & Raginsky 2017] and conditional mutual information (CMI) bounds [Steinke & Zakynthinou 2020] provide intuitive generalization guarantees—the less information the algorithm's output reveals about the training data, the better the generalization. CMI avoids the divergence of MI bounds in continuous/deterministic settings via a "super-sample" construction.

**Limitations of Prior Work**: Recent work by Attias et al. [2024] and Livni [2023] constructs elegant counterexamples showing that classical CMI bounds become vacuous on certain stochastic convex optimization (SCO) problems—the generalization bound does not decay with training set size $n$, remaining at $\Theta(LR)$. This finding has even raised fundamental doubts about the practical utility of information-theoretic generalization bounds.

**Key Challenge**: The key insight of the counterexamples is that the hypothesis space dimension $D$ grows as $\Omega(n^4 \log n)$ with $n$ (overparameterization), causing the CMI term to explode. In practice, however, learning algorithms may only be meaningful in low-dimensional subspaces. Classical CMI bounds fail to exploit the structural property that the effective dimension of the model is far lower than its nominal dimension.

**Goal**: (1) Fix the failure of CMI bounds on SCO counterexamples; (2) show that the new CMI bounds incorporating projection and compression still yield the correct $\mathcal{O}(1/\sqrt{n})$ generalization behavior; (3) prove that memorization is not necessary for good generalization.

**Key Insight**: Since the counterexamples rely on high-dimensional hypothesis spaces, compressing the model via random projection to a low-dimensional subspace plus quantization—while controlling the distortion introduced by the projection—can keep the CMI term bounded.

**Core Idea**: Use a random projection $\Theta^\top W$ to compress the $D$-dimensional model to $d$ dimensions (where $d$ can be chosen as 1), then quantize to $\hat{W}$, so that the CMI bound depends on the low-dimensional $d$ rather than the potentially explosive $D$.

## Method

### Overall Architecture

Given a learning algorithm $\mathcal{A}: \mathcal{Z}^n \to \mathcal{W} \subseteq \mathbb{R}^D$, an auxiliary algorithm is constructed as follows:
1. **Stochastic Projection**: $\Theta \in \mathbb{R}^{D \times d}$ projects $W$ to $d$ dimensions ($d \ll D$)
2. **Lossy Compression**: The projected $\Theta^\top W$ is quantized to obtain $\hat{W}$
3. **Back-Projection**: The auxiliary model $\Theta \hat{W}$ replaces the original model $W$

### Key Designs

#### 1. Main Theorem (Theorem 1)

**Core Formula**:
$$\text{gen}(\mu, \mathcal{A}) \leq \inf_{P_{\hat{W}|\Theta^\top W}} \inf_{P_\Theta} \mathbb{E}_{P_{\tilde{\mathbf{S}}} P_\Theta} \left[\sqrt{\frac{2\Delta\ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{n} \mathsf{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}})}\right] + \epsilon$$

where $\epsilon$ measures the difference in generalization error between the auxiliary and original models, and $\Delta\ell_{\hat{w}}$ is the loss fluctuation term.

**Design Motivation**: The original CMI bound depends on the CMI of the $D$-dimensional model, which may be large; the new bound depends on the CMI of the $d$-dimensional compressed model, where $d$ can be freely chosen to ensure the bound is non-vacuous. The key is that $\epsilon$ (the distortion term) can be controlled via concentration inequalities.

#### 2. Resolution of the CLB Counterexample (Theorem 3)

**Problem Setting**: $\ell_c(z, w) = -L\langle w, z \rangle$, $\mathcal{W} = \mathcal{B}_D(R)$, $D = \Omega(n^4 \log n)$.

**Core Result**:
$$\text{gen}(\mu, \mathcal{A}) \leq \frac{8LR}{\sqrt{n}}$$

For optimal sample complexity $N(\varepsilon, \delta) = \Theta(L^2 R^2 / \varepsilon^2)$, this yields $\text{gen}(\mu, \mathcal{A}) = \mathcal{O}(\varepsilon)$.

**Key Technique**: Projection to $d = 1$ dimension suffices. The approach uses Johnson-Lindenstrauss projection plus noise injection in the low-dimensional space, controlling the information flow via the Markov chain $(S_n, \Theta, W) - \Theta^\top W - \hat{W}$.

#### 3. Non-Necessity of Memorization (Theorem 6)

**Function**: Proves that for any learning algorithm $\mathcal{A}$, there exists an auxiliary algorithm $\mathcal{A}^* = \Theta \tilde{\mathcal{A}}(\Theta^\top \mathcal{A}(S_n))$ satisfying:
- Generalization error differs from the original algorithm by at most $\mathcal{O}(n^{-r})$ (for arbitrary $r > 0$)
- No adversary can track the training data

**Projection Dimension**: $d = 500r \log(n)$.

### Loss & Training

This is a purely theoretical work with no training procedure. The core construction consists of:
- Projection matrix $\Theta$: Johnson-Lindenstrauss random projection (i.i.d. Gaussian rows)
- Quantizer $P_{\hat{W}|\Theta^\top W}$: independent Gaussian noise added in the projected space
- Distortion control: convex-Lipschitz properties combined with concentration inequalities

## Key Experimental Results

### Main Results

This is a purely theoretical work with no numerical experiments. Main theoretical results compared:

| Problem Instance | Classical CMI Bound | New Bound (Ours) | Gain |
|---|---|---|---|
| CLB ($\mathcal{P}_{cvx}^{(D)}$) | $\Theta(LR)$ (no decay) | $8LR/\sqrt{n}$ | No decay → $\mathcal{O}(1/\sqrt{n})$ |
| CSL ($\mathcal{P}_{scvx}^{(D)}$) | No decay | $8L_cR/\sqrt{n}$ | No decay → $\mathcal{O}(1/\sqrt{n})$ |
| Livni SCO | MI bound vacuous | $\mathcal{O}(1/\sqrt{n})$ | Vacuous → meaningful |
| Generalized linear SCO | — | $\mathcal{O}(1/n^{1/4})$ | Non-convex extension |

### Ablation Study

**Effect of Projection Dimension $d$**:

| Problem | Required $d$ | Remarks |
|---|---|---|
| CLB counterexample | $d = 1$ | Extreme compression suffices |
| CSL counterexample | $d = 1$ | Strong convexity does not change projection requirement |
| Generalized linear SCO | $d = \Theta(\sqrt{n})$ | Larger problem class requires higher-dimensional projection |
| Memorization analysis | $d = \Theta(\log n)$ | Logarithmic growth is sufficient |

### Key Findings

1. The failure of classical CMI bounds is **not** an inherent defect of the CMI framework, but rather a failure to exploit the low-dimensional structure of the model.
2. Only $d = 1$ dimensional projection is needed to fix the CLB and CSL counterexamples.
3. Memorization is not necessary for generalization: the auxiliary algorithm can avoid memorizing training data while maintaining comparable generalization performance.
4. However, the norm of the auxiliary algorithm may grow as $\Omega(n^3)$, which is the key factor preventing the applicability of Attias et al.'s memorization necessity theorem.

## Highlights & Insights

1. **Resolving the "Crisis" in the CMI Framework**: This work directly addresses the negative results raised by Attias et al. and Livni, demonstrating the power of CMI augmented with projection and compression.
2. **A New Understanding of Memorization**: A constructive proof is provided showing that any memorizing algorithm admits an equivalent non-memorizing variant, revealing the deeper relationship between memorization and generalization.
3. **Connection to Rate-Distortion Theory**: The projection-plus-quantization construction is essentially an application of classical rate-distortion coding ideas from information theory to learning theory.
4. **Deep Insight**: The CMI explosion caused by overparameterization can be eliminated by projecting onto the effective subspace, offering a new perspective for understanding generalization in deep learning.

## Limitations & Future Work

1. **Purely Theoretical**: No experimental validation; the numerical tightness of the theoretical bounds remains unclear.
2. **Choice of Projection**: The selection of the optimal projection matrix is an open problem; JL projection may not be optimal.
3. **Convexity Assumption**: The optimal $\mathcal{O}(1/\sqrt{n})$ rate is only proven for SCO problems; the generalized non-convex version decays more slowly at $\mathcal{O}(1/n^{1/4})$.
4. **Practical Algorithm Design**: How to translate the projection-plus-compression idea into practical regularization or training strategies remains an open problem.
5. **Norm Growth of Auxiliary Model**: The norm of the auxiliary algorithm may be large, potentially causing numerical instability in practice.

## Related Work & Insights

- **Steinke & Zakynthinou [2020]**: The foundational work establishing the CMI framework; the direct target of improvement in this paper.
- **Attias et al. [2024]**: Counterexample construction for CMI bounds; directly addressed and resolved in this work.
- **Sefidgaran et al. [2022]**: Pioneer of rate-distortion methods for generalization bounds; this paper develops analogous ideas within the CMI framework.
- **Johnson-Lindenstrauss**: A classical dimensionality reduction tool whose novel role in learning theory is cleverly exploited here.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contributions are significant, elegantly resolving the "crisis" in the CMI framework and deepening the understanding of memorization. However, the work is purely theoretical with no experiments, and the gap to practical algorithm design remains substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Perturbation Bounds for Low-Rank Inverse Approximations under Noise](perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[NeurIPS 2025\] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems](efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)
- [\[NeurIPS 2025\] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[NeurIPS 2025\] Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression](binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)

</div>

<!-- RELATED:END -->
