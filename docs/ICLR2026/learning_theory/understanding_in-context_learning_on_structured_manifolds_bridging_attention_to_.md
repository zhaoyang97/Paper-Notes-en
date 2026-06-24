---
title: >-
  [Paper Note] Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods
description: >-
  [ICLR 2026][Learning Theory][In-Context Learning] This paper establishes the first theoretical framework for In-Context Learning (ICL) of Hölder function regression on manifolds. It proves that the attention mechanism of transformers essentially performs Gaussian kernel regression (Nadaraya–Watson estimation) and derives generalization error bounds revealing that the error decay rate depends solely on the **intrinsic dimension** $d$ of the data rather than the ambient dimensi…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "In-Context Learning Theory"
  - "In-Context Learning"
  - "Kernel Methods"
  - "Attention Mechanism"
  - "Manifold Regression"
  - "Generalization Error Bound"
date: 2026-05-08
content_hash: 46690104f2576154
---

# Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=WbRULwqsIy](https://openreview.net/forum?id=WbRULwqsIy)  
**Area**: Learning Theory / In-Context Learning Theory  
**Keywords**: In-Context Learning, Kernel Methods, Attention Mechanism, Manifold Regression, Generalization Error Bound

## TL;DR
This paper establishes the first theoretical framework for In-Context Learning (ICL) of Hölder function regression on manifolds. It proves that the attention mechanism of transformers essentially performs Gaussian kernel regression (Nadaraya–Watson estimation) and derives generalization error bounds revealing that the error decay rate depends solely on the **intrinsic dimension** $d$ of the data rather than the ambient dimension $D$.

## Background & Motivation

**Background**: ICL enables transformers to complete new tasks during inference using only a few examples $\{(x_i, f(x_i))\}$ provided in the prompt, without updating any parameters. Existing theoretical work mostly focuses on ICL for **linear models** (least squares, ridge, Lasso, generalized linear models), proving that transformers can implement these linear algorithms in-context.

**Limitations of Prior Work**: Theoretical understanding remains weak for non-linear models. Existing non-linear studies (such as sequence-to-sequence universal approximation or hierarchical Hölder classification) focus on **learning a single fixed function**. However, the essence of ICL is a **single model adapting instantly to multiple different tasks** via a prompt—this "task generalization" is rarely characterized in existing frameworks. More importantly, while real-world image or language data typically exhibit low-dimensional geometric structures (the manifold hypothesis), whether transformer ICL can exploit such geometric priors and how the error scales is entirely unknown.

**Key Challenge**: The attention mechanism appears as a "black box"—each query token aggregates prompt information through learned relevance scores. What is the relationship between this mechanism and classical regression algorithms in statistical learning? Without this bridge, it is impossible to rigorously analyze ICL generalization performance or characterize the role of geometric structure.

**Goal**: (1) Provide a provable algorithmic interpretation for the attention mechanism; (2) Derive generalization error bounds for ICL manifold regression based on this interpretation; (3) Decompose the error scaling law into "learning as an algorithm" and "predicting as a single-task regressor," clarifying the role of geometry (intrinsic dimension).

**Key Insight**: The authors observe that the structure of the attention score $\sigma((KH)^\top QH)$ is highly similar to the distance-weighted "importance weight" in classical kernel methods—the weighting of prompt tokens by softmax attention corresponds to the weights given by a Gaussian kernel $e^{-\|x_{n+1}-x_i\|^2/h^2}$.

**Core Idea**: Explain ICL through the unified perspective that "transformers implicitly execute kernel regression." Explicitly construct a transformer that **exactly implements** the Nadaraya–Watson kernel estimator with zero error, then leverage mature statistical theories of kernel regression to derive generalization bounds dependent on the intrinsic dimension.

## Method

### Overall Architecture

This is a **purely theoretical** work rather than a new model or architecture. The logical chain is: strictly establish the correspondence "Attention = Kernel Method" (construct a specific transformer to compute Gaussian kernel regression results exactly), then use this correspondence as an analytical tool to prove ICL generalization error bounds.

**Problem Setup**: Data $x$ is sampled from a $d$-dimensional compact Riemannian manifold $\mathcal{M}$ ($d \ll D$) embedded in $\mathbb{R}^D$. The target function $f$ is an $\alpha$-Hölder function on $\mathcal{M}$ ($0 < \alpha \le 1$). A prompt is defined as
$$s = \{x_1, y_1, \dots, x_n, y_n;\ x_{n+1}\},\quad y_i = f(x_i),$$
where the goal is to predict $f(x_{n+1})$. During training, $\Gamma$ different tasks/functions $\{f^\gamma\}$ are observed. Each task provides a prompt of length $n$. The empirical risk $R_{n,\Gamma}(T_\theta)$ is minimized to obtain $\hat T$. During inference, a prompt from a new task is provided to evaluate the squared generalization error $R_n(\hat T)$.

The theory rests on three pillars: **Attention as Kernel Method** (interpreting attention as kernel weights) → **Explicit Transformer Construction** (a 5-layer network where the first 4 compute geometric quantities and the last performs softmax normalization) → **Generalization Error Bound** (bias-variance decomposition + kernel regression statistical theory to obtain a scaling rate depending solely on $d$).

### Key Designs

**1. Attention as Kernel Method: Interpreting attention scores as Gaussian kernel weights and proving exact implementation**

To address the "black box" nature of attention, the authors use the classical Nadaraya–Watson kernel estimator as a bridge:
$$\mathcal{K}_h(s) = \frac{\sum_{i=1}^n e^{-\|x_{n+1}-x_i\|^2/h^2}\, y_i}{\sum_{i=1}^n e^{-\|x_{n+1}-x_i\|^2/h^2}},$$
which is a weighted average of prompt labels $y_i$ using an (unnormalized) Gaussian kernel with bandwidth $h$. The core observation is that the weighting of prompt tokens by softmax attention, with its denominator normalization and "higher weight for shorter distance," is **structurally identical** to this kernel estimator. By setting the query–key inner product to compute $-\|x_{n+1}-x_i\|^2/h^2$, the softmax scores become Gaussian kernel weights. Lemma 1 formalizes this as an **exact equality**: there exists a transformer $T_h^*$ (with parameters depending only on $D, n, b, R, h$ and independent of $f$ or data points, termed "universal") such that $T_h^*(s) = \mathcal{K}_h(s)$ for any input $s$, with **zero approximation error**. This shows transformers do not just mimic kernel regression but can precisely "compile" the algorithm into their weights.

**2. Explicit Transformer Construction: Using ReLU for geometry in $L-1$ layers and softmax for kernel normalization in the final layer**

This provides a constructive implementation for a 5-layer network. The input is arranged as a matrix $H \in \mathbb{R}^{(D+5)\times \ell}$ via linear embedding and sinusoidal position encoding. Layers 1 to 4 (using **ReLU** activation) transform $H$ into an intermediate matrix that computes difference vectors $x_{n+1}-x_i$, negative squared distances $-\|x_{n+1}-x_i\|^2/h^2$, and copies labels $y_i$ into position. The final layer uses a **softmax** single-head attention with sparse query/key matrices $Q_{\text{data}}, K_{\text{data}}$ and value matrix $V=e_{D+1}e_{D+2}^\top$. Masking the $(n+2)$-th to $(2n+1)$-th tokens, the $(n+1)$-th output token becomes:
$$[A(H)]_{n+1} = \sum_{j=1}^n \frac{e^{-\|x_{n+1}-x_j\|^2/h^2}}{\sum_{k=1}^n e^{-\|x_{n+1}-x_k\|^2/h^2}} y_j \cdot e_{D+1} = \mathcal{K}_h(s)\, e_{D+1},$$
The decoding layer reads row $D+1$, column $n+1$ to obtain $\mathcal{K}_h(s)$. The network scale is small: $L_T = 5$ layers, embedding dimension $D+5$, number of heads $m_T = nD$, and weight magnitude $\kappa = O(D^8 n^2 b^8 R^4 / h^8)$. This division between "ReLU for geometry" and "softmax for normalization" provides mechanistic evidence for attention carrying kernel regression.

**3. Generalization Error Bound: Decomposing error into "algorithm learning" and "minimax regression" terms**

Using $T_h^*$ as an anchor, the generalization error of $\hat T$ is compared via bias-variance decomposition: Approximation error III ($T_h^*$ vs. ground truth) + Statistical errors I & II ($\hat T$ vs. $T_h^*$ due to finite tasks). Approximation error is bounded via kernel regression analysis:
$$\text{III} \le O\!\Big(\tfrac{\log(h^{-1})^{1+3d/2}}{n h^d} + h^{2\alpha}[\log(h^{-1})]^2\Big),$$
Statistical errors are controlled via covering numbers: $\text{I+II} \le O\big(\tfrac{nD^3\sqrt{\log(nD\Gamma/h)}}{\sqrt{\Gamma}} + h^2\big)$. Setting the optimal bandwidth $h = n^{-1/(2\alpha+d)}$ yields the result in Theorem 1:
$$R_n(\hat T) \le C_1\Big(nD^3 \Gamma^{-1/2}\sqrt{\log(nD\Gamma)}\Big) + C_2\Big(n^{-\frac{2\alpha}{2\alpha+d}}\,\log^{1+\frac{3d}{2}} n\Big).$$
The terms have distinct meanings: **the first represents the scaling law**—the transformer as an "in-context kernel algorithm learner" learns the algorithm from $\Gamma$ tasks and generalizes to new ones, with error decaying at $\Gamma^{-1/2}$; **the second is the minimax regression rate**—given prompt length $n$, the rate $n^{-2\alpha/(2\alpha+d)}$ matches the lower bound for Hölder regression (up to a log factor), suggesting transformers are near-optimal as $\Gamma \to \infty$. Crucially, $d$ is the **intrinsic dimension**, meaning ICL effectively exploits low-dimensional geometric structures to avoid the curse of dimensionality.

### Loss & Training
The training objective is Empirical Risk Minimization: minimizing the Mean Squared Error (MSE) across $\Gamma$ tasks:
$$R_{n,\Gamma}(T_\theta) = \frac{1}{\Gamma}\sum_{\gamma=1}^{\Gamma}\Big(T_\theta(\{x_i^\gamma, y_i^\gamma\}_{i=1}^n;\, x_{n+1}^\gamma) - y_{n+1}^\gamma\Big)^2.$$
Transformer blocks use a residual structure $B(\theta;H)=\text{FFN}(\text{MHA}(H)+H)+\text{MHA}(H)+H$. Attention layers use ReLU from the first to penultimate layers, and softmax for the final layer.

## Key Experimental Results

Experiments used simulations to verify the theory. Setup: Manifold is a 2D sphere $\mathcal{M}=S^2$; target functions are random linear combinations of the first 10 spherical harmonics; $\Gamma=50000$ tasks; prompt length $n \in \{4,8,16,32\}$.

### Main Results: Correlation between Attention Scores and Gaussian Kernels

| Context Length $n$ | Mean Pearson Correlation | p-value |
|------|------|------|
| 4 | $0.86 \pm 0.21$ | $0.14 \pm 0.21$ |
| 8 | $0.75 \pm 0.22$ | $0.09 \pm 0.19$ |
| 16 | $0.69 \pm 0.22$ | $0.06 \pm 0.17$ |
| 32 | $0.67 \pm 0.19$ | $0.03 \pm 0.12$ |

The final layer attention scores of a trained transformer were compared to Gaussian kernels $e^{-\|x_{n+1}-x_i\|^2}$. The Mean Pearson Correlation across 5000 samples ranged from 0.67 to 0.86, with strong positive correlations. This validates that transformers implicitly perform Gaussian kernel regression. Similar "kernel shapes" were observed in specific heads of a pre-trained GPT-2 on natural language.

### Generalization Error Bound Validation

| Dimension | Setup | Observed Phenomenon | Consistency |
|------|------|------|------|
| Error vs. Task Count $\Gamma$ | Fixed $n$, log-log plot | Slope initially follows theory $-0.5$, then plateaus | Matches $\Gamma^{-1/2}$ in (15); second term dominates as $\Gamma$ grows |
| Error vs. Prompt Length $n$ | Fixed $\Gamma$ | Test MSE decreases as $n$ increases | Consistent with minimax rate; faster convergence for larger $\Gamma$ |

### Key Findings
- **Attention $\approx$ Gaussian Kernel is measurable**: Correlation remains significantly positive while p-values decrease with larger $n$, suggesting the kernel structure becomes statistically more robust in longer contexts.
- **Theoretical Slopes Reproduced**: The $\Gamma^{-1/2}$ slope of $-0.5$ on log-log plots provides direct evidence for the scaling law term.
- **Term Balancing Verified**: The overall trend of MSE depends on the balance between the two error terms; as $\Gamma$ increases, the second term dominates, leading to faster convergence with respect to $n$.

## Highlights & Insights
- **Exact Equality vs. Approximation**: Lemma 1 provides a construction with zero approximation error and universal weights, turning an intuitive mapping into a provable fact.
- **Intrinsic Dimension $d$ replaces Ambient Dimension $D$**: The error rate $n^{-2\alpha/(2\alpha+d)}$ shows ICL avoids the curse of dimensionality by exploiting low-dimensional geometry, explaining how LLMs handle high-dimensional data effectively.
- **Dual-role Error Decomposition**: Decomposing error into "algorithm learner" (learning kernel regression) and "single-task regressor" (predicting via prompt) provides a clear template for future non-linear ICL analysis.
- **Transferable Tools**: The approach of proving "exact implementation of a classical algorithm" followed by "leveraging classical statistical theory" can be extended to other non-linear ICL settings like operator learning.

## Limitations & Future Work
- **Specific Assumptions**: Analysis is restricted to Gaussian kernels, $\alpha$-Hölder smoothness, and uniform distributions on manifolds. General kernels or non-uniform samplings are not covered.
- **Non-optimal Constants**: Focus is on scaling laws; constants $C_1, C_2$ might contain factors like $d^{d/2}$, which could explode with intrinsic dimension.
- **Existence vs. Trainability**: Lemma 1 proves such a transformer exists, but does not guarantee that standard gradient descent will converge to this specific construction.
- **Task Distribution**: Assumes training and testing tasks are sampled from the same distribution $\rho_f$; behaviors under distribution shift are not analyzed.

## Related Work & Insights
- **vs. Linear ICL Theory (Bai et al. 2023; Von Oswald et al. 2023)**: Prior work proved ICL for linear algorithms; this paper extends this to **non-linear kernel regression** and introduces the role of manifold geometry.
- **vs. Single-task Non-linear Approximation**: While prior studies focused on approximating single functions where network size scales exponentially with dimension, this paper focuses on **multi-task generalization** via prompts.
- **vs. Geometric Deep Learning (Chen et al. 2022)**: Sample complexity for standard networks depends on intrinsic dimension; this paper extends this conclusion to the **transformer ICL setting** for the first time.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] Critical Attention Scaling in Long-Context Transformers](critical_attention_scaling_in_long-context_transformers.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)

</div>

<!-- RELATED:END -->
