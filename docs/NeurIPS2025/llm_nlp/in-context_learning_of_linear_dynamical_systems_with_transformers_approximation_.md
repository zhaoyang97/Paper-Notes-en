---
title: >-
  [Paper Note] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation
description: >-
  [NeurIPS 2025][LLM/NLP][in-context learning] This paper analyzes the ICL approximation capability of linear Transformers on noisy linear dynamical systems: $O(\log T)$ depth suffices to achieve $O(\log T / T)$ test error…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "in-context learning"
  - "linear dynamical systems"
  - "approximation theory"
  - "depth separation"
  - "transformer expressivity"
date: 2026-05-08
content_hash: a1bde3f3c2d1743a
---

# In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation

**Conference**: NeurIPS 2025
**arXiv**: [2502.08136](https://arxiv.org/abs/2502.08136)
**Code**: None
**Area**: LLM Theory / ICL
**Keywords**: in-context learning, linear dynamical systems, approximation theory, depth separation, transformer expressivity

## TL;DR
This paper analyzes the ICL approximation capability of linear Transformers on noisy linear dynamical systems: $O(\log T)$ depth suffices to achieve $O(\log T / T)$ test error (approaching the least-squares estimator), while single-layer linear Transformers admit an irreducible lower bound — revealing a depth-separation phenomenon under non-IID data.

## Background & Motivation
**Background**: ICL theory has primarily studied linear regression under IID data, where a single-layer linear Transformer can implement one-step gradient descent, achieving test error $O(1/T)$. In practice, however, language data is sequentially correlated (non-IID).

**Limitations of Prior Work**: ICL theory for non-IID settings (e.g., dynamical systems) remains underdeveloped. Can Transformers learn dynamical systems via ICL? How many layers are required? What is the essential difference from the IID case?

**Key Challenge**: In the IID setting, a single-layer Transformer suffices. Whether this holds under non-IID data is unclear. How does the temporal correlation between tokens in dynamical systems affect ICL?

**Goal**: (1) Quantify upper bounds on the ICL approximation capability of deep Transformers; (2) Prove approximation lower bounds for single-layer Transformers; (3) Reveal the essential distinction between IID and non-IID settings.

**Key Insight**: Construct a Transformer that approximately implements Richardson iteration to solve the least-squares problem, then leverage statistical properties of the least-squares estimator to derive test error bounds.

**Core Idea**: ICL under non-IID data requires depth — an $O(\log T)$-layer Transformer can approximate the least-squares estimator by unrolling Richardson iteration, whereas a single layer cannot.

## Method

### Overall Architecture
Consider a noisy linear dynamical system $x_t = Wx_{t-1} + \xi_t$, where the Transformer must predict $x_{T+1}$ from the sequence $(x_0, \ldots, x_T)$. The test loss is defined as $\sup_{W \in \mathcal{W}} \mathbb{E}[\|\hat{y}_T - Wx_T\|^2]$ — a worst-case measure over all possible system matrices $W$.

### Key Designs

1. **Upper Bound for Deep Transformers (Theorem 1)**:

    - **Function**: Construct an $O(\log T)$-depth linear Transformer that implements approximate least-squares prediction.
    - **Mechanism**: Least-squares estimation requires computing $X_t^{-1}x_t$ (the inverse of the empirical covariance). This is unrolled into multi-step linear operations via Richardson iteration (each step corresponding to one Transformer layer), converging in $O(\log T)$ steps.
    - **Key Result**: $L_T(\theta) \lesssim \log(T)/T$, approaching the parametric optimal rate $O(1/T)$.
    - **Design Motivation**: Direct matrix inversion is not feasible within a Transformer, but iterative methods are.

2. **Lower Bound for Single-Layer Transformers (Theorem 2)**:

    - **Function**: Prove that single-layer linear attention has an irreducible positive lower bound on test loss for dynamical systems.
    - **Mechanism**: The prediction of single-layer attention is a linear function of empirical second-order statistics multiplied by $x_T$. Since the data is non-IID, these statistics relate to $W$ nonlinearly (via $\sigma^2/(1-w^2)$), and no fixed linear function can uniformly fit all $W$.
    - **Key Result**: $\inf_{\mathbf{p},\mathbf{q}} L_T(\mathbf{p}, \mathbf{q}) = \Omega(1)$, i.e., the error does not vanish as $T \to \infty$.

3. **Root Cause of the IID vs. Non-IID Distinction**:

    - Under IID data, the empirical covariance $X_t$ converges to a constant $\Sigma$ (identical across all tasks $W$), so a single-layer Transformer can approximate $\Sigma^{-1}$ with fixed parameters.
    - Under non-IID data (dynamical systems), the limit of $X_t$ depends on $W$ (via $\sigma^2/(1-W^2)$), and fixed single-layer parameters cannot adapt to all $W$ — multiple layers are required to implement adaptive matrix inversion.

### Loss & Training
- Training loss: autoregressive prediction $\frac{1}{T-1}\sum_{t=1}^{T-1}\|\hat{y}_t - x_{t+1}\|^2$
- Test loss: worst-case $\sup_W \mathbb{E}[\|\hat{y}_T - Wx_T\|^2]$ (uniform-in-task)

## Key Experimental Results

### Main Results
(This paper is primarily theoretical; experiments serve as validation.)

| Configuration | Test Loss Behavior w.r.t. $T$ | Theoretical Prediction |
|---|---|---|
| Deep ($L = O(\log T)$) | $\to 0$ at rate $\sim \log T / T$ | ✅ Consistent with Theorem 1 |
| Single-layer | Converges to a positive constant (does not vanish) | ✅ Consistent with Theorem 2 |
| IID data + single-layer | $\to 0$ at rate $\sim 1/T$ | ✅ Consistent with prior results |

### Ablation Study

| Configuration | Result | Remarks |
|---|---|---|
| Increasing depth | Faster convergence for deeper models | Validates that logarithmic depth suffices |
| Varying noise $\sigma$ | Scaling law unchanged | Affects constants but not the rate |

### Key Findings
- **Depth separation**: Under non-IID data, single-layer linear Transformers suffer an irreducible $\Omega(1)$ error, while $O(\log T)$ layers achieve $O(\log T/T)$ — a qualitative gap.
- **Root cause of IID vs. non-IID distinction**: Single-layer Transformers suffice for ICL under IID data, but multiple layers are necessary under non-IID data — because whether the limit of the empirical covariance depends on the task is the decisive factor.
- **Richardson iteration is more efficient than gradient descent**: The former requires only $O(\log T)$ steps, whereas the latter requires $O(T)$ steps.

## Highlights & Insights
- **First depth-separation result proven in ICL theory**: All prior work analyzed the IID setting where a single layer suffices. Under non-IID data, depth becomes a necessary condition — an important contribution to ICL theory.
- **A new answer to "what algorithm does the mesa-optimizer implement?"**: For dynamical systems, the corresponding algorithm is Richardson iteration (rather than gradient descent), since it must solve a data-dependent linear system.
- **Uniform-in-task loss is a more principled metric**: It ensures that ICL provides guarantees for any downstream task.

## Limitations & Future Work
- **Analysis restricted to linear Transformers and linear dynamical systems**: Softmax attention and nonlinear systems are not addressed.
- **Single-layer lower bound relies on a parameter restriction**: The MLP weights are fixed to the identity; the lower bound for a fully unconstrained single layer remains an open problem.
- **Training dynamics are not analyzed**: The results are purely existential (approximation-theoretic); it is not proven that gradient descent can find the constructed good parameters.
- **Future directions**: Extension to softmax attention, nonlinear systems, and analysis of training convergence.

## Related Work & Insights
- **vs. Zhang et al. (2023, IID ICL)**: A single layer suffices for ICL under IID data; this paper proves a single layer is insufficient under non-IID data.
- **vs. Von Oswald et al. (2023b)**: They construct Transformer-like models but without approximation error guarantees; this paper provides quantitative bounds.
- **vs. Zheng et al. (2024)**: They analyze noiseless dynamical systems; this paper analyzes the noisy case — noise is the key factor necessitating multiple layers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First approximation theory + depth separation for non-IID ICL, with clear contributions.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; empirical validation is adequate but not in-depth.
- Writing Quality: ⭐⭐⭐⭐ Theorems are clearly stated; proof sketches aid understanding.
- Value: ⭐⭐⭐⭐⭐ Reveals a fundamental connection between ICL depth requirements and data correlation structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](linear_transformers_implicitly_discover_unified_numerical_algorithms.md)
- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[NeurIPS 2025\] Composing Linear Layers from Irreducibles](composing_linear_layers_from_irreducibles.md)
- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](system_prompt_optimization_with_meta-learning.md)

</div>

<!-- RELATED:END -->
