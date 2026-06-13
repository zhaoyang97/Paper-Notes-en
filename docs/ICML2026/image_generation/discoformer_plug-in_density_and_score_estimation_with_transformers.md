---
title: >-
  [Paper Note] DiScoFormer: Plug-In Density and Score Estimation with Transformers
description: >-
  [ICML 2026][Image Generation][Density estimation] Ours proposes DiScoFormer, a Transformer that is equivariant to sample permutation and coordinate affine transformations. It maps any i.i.d. sample set to its correspondi…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Density estimation"
  - "score estimation"
  - "Transformer"
  - "Kernel Density Estimation"
  - "Equivariant networks"
date: 2026-05-08
content_hash: 93dd3f6a52420269
---

# DiScoFormer: Plug-In Density and Score Estimation with Transformers

**Conference**: ICML 2026 Oral  
**arXiv**: [2511.05924](https://arxiv.org/abs/2511.05924)  
**Code**: To be confirmed  
**Area**: Scientific Computing / Non-parametric Statistics / Density and Score Estimation  
**Keywords**: Density estimation, score estimation, Transformer, Kernel Density Estimation, Equivariant networks

## TL;DR
Ours proposes DiScoFormer, a Transformer that is equivariant to sample permutation and coordinate affine transformations. It maps any i.i.d. sample set to its corresponding density $f$ and score $\nabla\log f$ in a single forward pass. It is theoretically proven that self-attention, under appropriate parametrization, can exactly reproduce normalized Gaussian KDE. Experimentally, it outperforms classical KDE across various distributions (GMM, Laplace, Student-$t$), sample sizes, and dimensions, and can serve as a plug-and-play score oracle for Fisher information, entropy estimation, and Fokker–Planck-type PDE solving.

## Background & Motivation

**Background**: Estimating density $f$ and score $\nabla\log f$ from samples is a fundamental primitive in generative models, Bayesian inference, and kinetic equation solving. Currently, there are two main approaches: Kernel Density Estimation (KDE, e.g., Parzen window/Silverman's rule), which provides closed-form, interpretable, distribution-agnostic estimation; and neural score learning (e.g., denoising score matching/diffusion models), which achieves high accuracy in high dimensions.

**Limitations of Prior Work**: Both camps have critical weaknesses. KDE suffers from the "curse of dimensionality"—the bias-variance tradeoff of the bandwidth is rigid, causing errors to explode as dimensions increase, and score estimation naturally carries $O(h^2)$ bias. Neural score matching, though accurate, is **transductive**: it must be retrained for every target distribution, making it unusable as a "plug-and-play" statistical primitive.

**Key Challenge**: There is a structural contradiction between generalization capability (KDE's universality across distributions) and high accuracy (neural networks' precision in high dimensions)—one leaves inductive bias entirely to a fixed kernel function, while the other is tied exclusively to a single target distribution.

**Goal**: Decompose this into two sub-problems: (i) find an **architectural** network that respects permutation/affine symmetries to "internalize" KDE's symmetric inductive biases; (ii) make the network learn an **operator** $X \mapsto (\log f, \nabla\log f)$ instead of a function for a specific distribution, enabling generalization across distributions and sample sizes.

**Key Insight**: Reformulate density/score estimation as sequence-to-operator learning—the input is the sample sequence $X=\{x_i\}_{i=1}^n$, and the output includes two sequences corresponding to the samples: $\log f(x_i)$ and $\nabla\log f(x_i)$. Transformer's self-attention is naturally equivariant to token permutation, matching the structure of i.i.d. "unordered sets". By adding affine equivariance, it inherits the symmetries of KDE. The authors further observe that the exponential kernel form of softmax attention is structurally similar to Gaussian KDE, suggesting that the Transformer is a **data-adaptive generalization of KDE**.

**Core Idea**: Utilize a permutation-equivariant and affine-equivariant Transformer to learn a universal score/density operator across distributions. Theoretically prove that attention can strictly reproduce normalized KDE, merging KDE's "universality" with neural networks' "accuracy + multi-scale adaptability" within the same architecture.

## Method

### Overall Architecture
DiScoFormer receives two sets of points: context $X \in \mathbb{R}^{n_x \times d}$ (i.i.d. samples defining empirical density) and queries $Y \in \mathbb{R}^{n_y \times d}$ (locations to calculate $\log f, \nabla\log f$). It outputs a scalar $\log f(y_i)$ and vector $\nabla\log f(y_i)$ for each query point. The pipeline is: Whitening layer (ensuring affine equivariance via $S^{-1/2}$ transform + log-determinant correction) $\rightarrow$ Standard Transformer encoder (no positional encoding + cross-attention for $X, Y$) $\rightarrow$ Two shared-backbone output heads for $T$ (log-density) and $S$ (score). Finally, outputs in whitened coordinates are mapped back via $A^\top$. Training data consists of real-time sampled GMM flows (1–10 components, variable dimensions) with joint $\alpha\,\mathcal{L}_T + (1-\alpha)\mathcal{L}_S$ supervision. Test-time training (TTT) can be enabled to align score outputs with the autograd of log-density as a label-free consistency loss for OOD adaptation.

### Key Designs

1. **Whitening-based Affine Equivariant Layer**:
    - **Function**: Makes the model strictly (or approximately) equivariant to translation, isotropic/anisotropic scaling, and rotation, such that $T(PXA+\mathbf{1}\mu^\top) = PT(X) - \log|\det A|\,\mathbf{1}$ and $S(PXA+\mathbf{1}\mu^\top) = PS(X)A^{-\top}$.
    - **Mechanism**: Subtract mean $m$ from both $X$ and $Y$; compute matrix inverse square root $A = S^{-1/2}$ of the regularized scatter matrix $S = X_c^\top X_c + \varepsilon I$; transform $X, Y$ to whitened coordinates $X_w = X_c A, Y_w = Y_c A$. Transformer computes $\log f_w, s_w$ in whitened space, followed by variable substitution correction—$\log f = \log f_w + \log\det A$ and $s = s_w A^\top$. Whitening reduces arbitrary linear transforms to $O(d)$ rotation/reflection residuals; the remaining $O(d)$ equivariance is learned via random orthogonal rotation augmentation during training.
    - **Design Motivation**: Pure Transformer + position-independence only ensures permutation equivariance. To incorporate KDE's "coordinate-scale agnosticism," explicit affine normalization is required. Whitening is closed-form, differentiable, and naturally incorporates geometric transform formulas for score/density.

2. **Attention = Data-adaptive KDE (Constructive Equivalence Theorem)**:
    - **Function**: Theoretically proves that softmax (cross-)attention is a strict generalization of normalized Gaussian KDE. With appropriate $Q, K$ projections and $\|z\|^2$ scalar features, a single cross-attention layer can exactly reproduce KDE density and score.
    - **Mechanism**: For any semi-definite $B$, cross-attention weights $A_{ij} = \frac{\exp(y_i^\top B x_j)}{\sum_k \exp(y_i^\top B x_k)}$ can be rewritten as $A_{ij} = \frac{w_j \exp(-\tfrac{1}{2}\|y_i - x_j\|_B^2)}{\sum_k w_k \exp(-\tfrac{1}{2}\|y_i - x_k\|_B^2)}$ via polarization identity, where $w_j = \exp(\tfrac{1}{2}\|x_j\|_B^2)$. Appending $\|z\|^2$ to tokens allows canceling the $w_j$ term. Thus, a single residual cross-attention block (width $d_\text{model} \geq 2d+1$, no FFN/LN) plus an affine readout and per-query log-normalizer can output the normalized KDE score $\nabla\log\hat{f}_{h,X}(y_i)$ and log-density $\log\hat{f}_{h,X}(y_i)$ (Prop. 3.5, Cor. 3.6).
    - **Design Motivation**: This theorem upgrades "Transformers can do density estimation" from an empirical observation to a structural necessity. Multi-layer/multi-head configurations allow learning more flexible, data-adaptive multi-scale kernels than fixed rules.

3. **Cross-attention + Joint (density, score) Heads + Test-time Training**:
    - **Function**: Extends estimation from sample points $X$ to arbitrary query points $Y$, and achieves label-free OOD adaptation via differential consistency.
    - **Mechanism**: Uses $X$ as context and $Y$ as query for cross-attention. Jointly optimizes $\log f$ and score heads with shared backbone via $\mathcal{L} = \alpha \mathcal{L}_T + (1-\alpha)\mathcal{L}_S$. During inference, minimizing consistency loss $\mathcal{L}_\text{con} = \tfrac{1}{n}\sum_i \|S(C,Q)_i - \nabla_{q_i} T(C,Q)_i\|_2^2$ enables fine-tuning without ground-truth labels.
    - **Design Motivation**: Cross-attention turns the model into a true non-parametric smoothing operator. Joint training utilizes shared geometric features, and TTT provides a path for label-free OOD adaptation.

### Loss & Training
Joint optimization of $\mathcal{L} = \alpha\,\mathcal{L}_T + (1-\alpha)\mathcal{L}_S$, where $\mathcal{L}_T, \mathcal{L}_S$ are the MSE for $\log f$ and score respectively. Training data is generated on-the-fly (Algorithm 1): each batch randomly draws $k \in [k_\text{min}, k_\text{max}]$ GMM components to generate signals. Standard config: 4 encoder layers, dim 128, 8 heads, GELU, pre-norm, no PE, ~800k parameters. Large-$n$ experiments use $d_\text{model}=256, 6$ layers, and random sampling of context sizes $n \in [2^8, 2^{14}]$ to enable sequence length extrapolation.

## Key Experimental Results

### Main Results
All experiments on 48GB L40S; baselines include Scott's rule KDE and Score-Debiased KDE (SD-KDE).

| Dataset / Setting | Metric | Ours | KDE | Gain |
|--------|------|------|------|------|
| GMM 2D, $n=2^{14}$ | Rel. score MSE (%) | 6.80 | 17.2 | $\approx 2.5\times$ |
| GMM 10D, $n=2^{14}$ | Rel. score MSE (%) | 2.83 | 52.9 | $\approx 18\times$ |
| GMM 10D, $n=2^{17}$ (Extrapolation) | Rel. score MSE (%) | 2.74 | OOM | KDE OOM |
| 2D Laplace, $n=2048$ | Score MSE | 0.2990 | 0.2990 | OOD Generalization |

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| Full Model | Affine Rel. MSE $\sim 10^{-4}$ | Whitening + Augmentation ensure approximate equivariance |
| GMM 1-10 modes $\rightarrow$ 1-19 modes | MSE increases slightly | Stable OOD mode generalization |
| Large $n$ Transformer, test $2^{17}$ | MSE continues to drop | Successful sequence length extrapolation |
| TTT 0 $\rightarrow$ 4 steps (Laplace/Student-$t$) | Score MSE further drops | Consistency loss provides OOD adaptation |

### Key Findings
- Higher dimensions lead to larger gains for Transformer vs. KDE: from $2$-$3\times$ at $d=2$ to $\sim 18\times$ at $d=10$. DiScoFormer avoids OOM issues at large $n$, mitigating the "curse of dimensionality."
- Automatic head specialization (far-range / mid-range / directional kernels) aligns with "multi-scale KDE" theory, proving Transformer learns adaptive kernels.
- Training only on GMM transfers well to Laplace and Student-$t$; 4-step TTT further reduces error.
- Functioning as a plug-and-play oracle, it provides high-fidelity scores for Fisher info and Fokker–Planck PDE solvers without additional training.

## Highlights & Insights
- **Strict proof of "attention = data-adaptive KDE generalization"**: Prop 3.3-3.6 provide constructive equivalence—appending $\|z\|^2$ allows single-head cross-attention to exactly equal normalized Gaussian KDE.
- **Whitening + Rotation Augmentation**: Whitening handles translation/scaling strictly; remaining rotation equivariance is learned via augmentation, avoiding expensive equivariant layers.
- **Joint heads + Consistency $\rightarrow$ TTT**: Using the mathematical relationship between score and grad log-density for self-supervision is a reusable trick for OOD adaptation.

## Limitations & Future Work
- Current training data is exclusively GMM; although GMMs are dense, extreme distributions (heavy-tailed/discrete) still rely on TTT.
- Max dimension tested is $d=10$; whitening requires non-singular covariance, needing low-rank/spectral strategies for higher dimensions like ImageNet.
- Attention complexity is $O(n_x n_y)$, similar to KDE; linear attention or Nyström approximations could address large-sample bottlenecks.
- No explicit normalization ensures $\int \exp(\log f) = 1$.

## Related Work & Insights
- **vs. Score-Debiased KDE (Epstein et al., 2025)**: SD-KDE needs an external score oracle; DiScoFormer acts as this missing component.
- **vs. Score Neural Operator (Liao et al., 2024)**: SNO uses RKHS embedding; DiScoFormer consumes raw i.i.d. samples directly, offering better cross-density generalization.
- **vs. Score-based Generative Models**: Generative models are transductive (one net per distribution); DiScoFormer is a distribution-agnostic operator—"train once, use anywhere."
- **vs. Set Transformer/DeepSets**: Shares permutation-equivariance but adds affine-equivariance and theoretical links to non-parametric statistics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First constructive proof of attention as KDE generalization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive scanning of dimensions/samples and OOD tests; lacks real-world high-dimensional data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from theory to visualization.
- Value: ⭐⭐⭐⭐⭐ Provides a long-missing plug-in score oracle for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](../../CVPR2026/image_generation/taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](../../ICLR2026/image_generation/sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[ICML 2026\] Krause Synchronization Transformers](krause_synchronization_transformers.md)
- [\[ICML 2026\] Rao-Blackwellized Score Matching on Manifolds](rao-blackwellized_score_matching_on_manifolds.md)
- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)

</div>

<!-- RELATED:END -->
