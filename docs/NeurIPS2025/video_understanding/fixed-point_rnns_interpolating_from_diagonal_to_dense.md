---
title: >-
  [Paper Note] Fixed-Point RNNs: Interpolating from Diagonal to Dense
description: >-
  [NeurIPS 2025][Video Understanding][Fixed-Point Iteration] This paper proposes the Fixed-Point RNN framework, which parameterizes dense linear RNNs as fixed points of diagonal linear RNNs. By varying the number of iterat…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Fixed-Point Iteration"
  - "Dense Linear RNN"
  - "State-Space Model"
  - "Mamba"
  - "State Tracking"
date: 2026-05-08
content_hash: 31052b89f3fbc23e
---

# Fixed-Point RNNs: Interpolating from Diagonal to Dense

**Conference**: NeurIPS 2025
**arXiv**: [2503.10799](https://arxiv.org/abs/2503.10799)
**Code**: None (not yet released)
**Area**: Sequence Modeling / State Space Models
**Keywords**: Fixed-Point Iteration, Dense Linear RNN, State-Space Model, Mamba, State Tracking

## TL;DR

This paper proposes the Fixed-Point RNN framework, which parameterizes dense linear RNNs as fixed points of diagonal linear RNNs. By varying the number of iterations, the model dynamically interpolates between diagonal (efficient) and dense (expressive) regimes, achieving state-of-the-art results simultaneously on state-tracking ($A_5$/$S_5$) and copying tasks for the first time.

## Background & Motivation

**Background**: Linear RNNs (e.g., Mamba), enabled by diagonal state transition matrices for efficient parallel training, have emerged as a mainstream alternative to Transformers. Theoretical work shows that diagonal structure severely limits state-tracking capacity, while dense transition matrices theoretically match the expressiveness of full nonlinear RNNs.

**Limitations of Prior Work**:
- **Insufficient expressiveness of diagonal RNNs**: Mamba fails to learn state-tracking tasks such as $A_5$/$S_5$.
- **Efficiency issues of dense RNNs**: Parameterizing $A_t \in \mathbb{R}^{d \times d}$ requires $O(d^3)$ time and $O(d^2)$ parameters.
- **Limitations of structured approaches**: Methods such as DeltaProduct enhance expressiveness via Householder structure, but the choice of structure remains ad hoc.

**Key Challenge**: A fundamental trade-off between parallelism and expressiveness—diagonal RNNs are parallelizable but limited in expressiveness, while dense RNNs are expressive but not parallelizable. A systematic approach to navigate between them is lacking.

**Goal**: Design a general framework that allows models to **adaptively** trade off efficiency against expressiveness, rather than fixing the structure at design time.

**Key Insight**: Drawing on ideas from Deep Equilibrium Models, dense RNNs are implicitly represented as fixed points of diagonal RNNs.

**Core Idea**: Dense RNNs are realized via fixed-point iteration of $h = f_\theta(x, h)$, alternating between channel mixing and sequence mixing, with the number of iterations adapting to task difficulty.

## Method

### Overall Architecture

Target dense RNN: $h_t^* = Q_t^{-1} \Lambda_t h_{t-1}^* + B_t x_t$

Rewritten as a fixed-point equation:

$$h_t^* = \Lambda_t h_{t-1}^* + Q_t B_t x_t + (I - Q_t) h_t^*$$

Corresponding diagonal RNN:

$$h_t^\ell = \Lambda_t h_{t-1}^\ell + Q_t B_t x_t + (I - Q_t) h_t^{\ell-1}$$

After $\ell$ iterations, the hidden state converges to that of the dense RNN, $h^*$.

### Key Designs

#### 1. Stability Guarantee (Banach Fixed-Point Theorem)

Two conditions must be satisfied:
- **Temporal dimension**: $\|\Lambda\|_2 < 1$ (contractiveness of the diagonal transition) + input normalization $(I - \Lambda_t)$
- **Depth dimension**: $\|I - Q_t\|_2 < 1$ (contractiveness of the channel mixing)

When both conditions hold, iteration from any initialization $h^0 = 0$ converges to a unique fixed point, and all intermediate states remain bounded.

#### 2. Structural Options for the Channel Mixer $Q_t$

| Structure | Parameterization | Parameter Count |
|-----------|-----------------|----------------|
| DPLR | $Q_t = I - \sum_{i=1}^r \alpha_{it} \bar{u}_{it} \bar{u}_{it}^\top$ | $O(d^2 r)$ |
| Householder (H) | $Q_t = \prod_{i=1}^r (I - \alpha_{it} \bar{u}_{it} \bar{u}_{it}^\top)$ | $O(d^2 r)$ |
| Kronecker (K) | $Q_t = I - (\bar{K}_t^1 \otimes \bar{K}_t^2)$ | $O(d^2)$ |

Experiments show that the Kronecker structure is most suitable for state tracking, while the Householder structure also performs well on memory tasks.

#### 3. Hidden-State-Dependent Channel Mixing

$Q_t = \mathcal{M}(x_t + h_{t-1}^{\ell-1})$: the channel mixer depends not only on the input but also on the hidden state from the previous iteration. This is critical for length generalization.

#### 4. FP-Mamba: Matrix-Valued Hidden States

The framework is extended to Mamba's matrix-valued hidden state $H_t \in \mathbb{R}^{d_{\text{state}} \times d_{\text{inner}}}$:

$$H_t^\ell = \lambda_t \odot H_{t-1}^\ell + \bar{b}_t^\ell (\Delta_t \tilde{x}_t^\ell)^\top$$

where $\tilde{x}_t^\ell = Q_t^\ell (x_t - y_t^{\ell-1}) + y_t^{\ell-1}$.

Each fixed-point iteration consists of one channel mixing step ($Q_t$) followed by one Mamba sequence mixing step.

#### 5. Adaptive Number of Iterations

Convergence criterion: $\frac{\|h^\ell - h^{\ell-1}\|_\infty}{\|h^\ell\|_\infty} < 0.1$

The model automatically adapts to task difficulty:
- State tracking $A_5$ (hard): $\ell^* \approx$ sequence length
- Language modeling (low channel mixing demand): $\ell^* \ll T$
- Copying task: $\ell^*$ is much lower than the maximum sequence length

### Loss & Training

- **Truncated backpropagation**: Gradients are computed only at the fixed point ($k=0$), eliminating the need to store the iterative computation graph.
- **No additional memory overhead**: Compared to a single-layer diagonal RNN, only sequential overhead in the forward pass is added.
- **Randomized training**: $\ell_{\max} \sim \Gamma(4,1)$ can be sampled to accelerate training, requiring on average only 4 iterations.

## Key Experimental Results

### Main Results: State Tracking $A_5$/$S_5$

- A single-layer FP-Mamba-K (Kronecker) trained on length 16 generalizes to lengths **80+** ($A_5$).
- A single-layer FP-Mamba-H also achieves significant generalization on $S_5$.
- In comparison, 2-layer Mamba/Mamba-2 fail to learn even on training lengths; 2-layer Gated DeltaNet learns at training length but exhibits limited generalization.
- LSTM serves as an upper bound with perfect generalization.

### Main Results: Copying Task

| Model | $2\times$ Length Generalization Accuracy |
|-------|------------------------------------------|
| Mamba (2-layer) | Low |
| Mamba-2 (2-layer) | Moderate |
| Gated DeltaNet (2-layer) | **High** |
| **FP-Mamba-H (1-layer)** | **High (matches GDN)** |

FP-Mamba is the **only model that achieves state-of-the-art results on both state-tracking and copying tasks simultaneously**.

### Ablation Study

**Contribution of hidden-state dependence** (Table 1, copying task):

| $\lambda_t$ dep. | $Q_t$ dep. | $b_t$ dep. | $c_t$ dep. | $2\times$ Gen. Accuracy |
|:-:|:-:|:-:|:-:|---:|
| | | | | 0.11 |
| | ✓ | | | 0.53 |
| | | ✓ | ✓ | 0.81 |
| ✓ | ✓ | ✓ | ✓ | **0.94** |

Hidden-state dependence in $b_t$ and $c_t$ is the key to unlocking copying capability.

### Training Time vs. Performance

Relationship between wall-clock training time and the longest generalizable sequence length:
- **Baseline models** (increasing depth): Regardless of training time, generalization length does not exceed training length 16.
- **FP-Mamba** (increasing iterations): Generalization length continues to grow with training time.

### Key Findings

1. **Fixed-point iteration count adapts to task difficulty**: Easy tasks (language modeling) yield small $\ell^*$; hard tasks ($A_5$/$S_5$) yield $\ell^*$ close to sequence length.
2. **Depth vs. width**: Stacking more layers cannot overcome the fundamental expressiveness limitation of diagonal RNNs, whereas fixed-point iteration with the same parameter count can.
3. **Convergence in language modeling**: During language pre-training, the effective iteration count $\ell^*$ per layer stabilizes quickly at a small value.
4. **Simplified backpropagation**: Computing gradients only at the fixed point—without backpropagating through the entire iterative process—is sufficient for stable training.

## Highlights & Insights

1. **Generality of the framework**: Not restricted to a specific channel mixing structure (DPLR/Householder/Kronecker); various structures can be used as plug-and-play components.
2. **Elegant efficiency–expressiveness trade-off**: The number of iterations provides a continuous knob from 0 (purely diagonal) to $\infty$ (dense RNN).
3. **Deep connection to classical theory**: The Banach fixed-point theorem guarantees convergence, and implicit differentiation from Deep Equilibrium Models eliminates memory overhead.
4. **Simultaneous solution to two task types**: Breaks the dichotomy between "state tracking requires dense structure" and "copying requires linear-attention memory."
5. **Adaptive computation**: The model automatically identifies task difficulty and adjusts computation accordingly, analogous to the Adaptive Computation Time idea of Graves (2016).

## Limitations & Future Work

1. **Worst-case complexity**: $O(\ell^* \cdot \log T)$; if $\ell^* \sim T$, this degrades to $O(T^2)$.
2. **Lack of efficient GPU implementation**: Repeated computation in fixed-point iteration has the potential to be fused into a single kernel, but this has not yet been realized.
3. **Insufficient large-scale language modeling validation**: Language pre-training experiments are conducted at a small scale (context length 2048) without comparison to DeltaProduct or similar methods at the 1.3B scale.
4. **Non-unique fixed points**: Convergence guarantees for the hidden-state-dependent variant are weaker, and the fixed point may not be unique.
5. **Training time overhead**: Although backpropagation incurs no additional cost, iterative forward passes still increase training time.

## Related Work & Insights

- **Deep Equilibrium Models (Bai et al.)**: The source of inspiration for fixed-point iteration, applied here in the context of linear RNNs.
- **DeltaNet/DeltaProduct**: Enhances expressiveness through Householder structure via an "explicit" structured approach; FP-RNN achieves density through "implicit" iteration.
- **Mamba/Mamba-2**: Representative diagonal RNNs serving as the base layer for FP-Mamba.
- **LSTM**: An expressive upper bound among nonlinear RNNs; FP-RNN aims to approximate its expressiveness within a linear framework.
- **Adaptive Computation Time (Graves)**: A pioneering work on adaptive computation.

## Rating

⭐⭐⭐⭐⭐

The theoretical framework is elegant (fixed-point iteration + Banach theorem + implicit differentiation), the experiments are comprehensive (simultaneously addressing state tracking and copying), and the insights are profound (adaptive computation reveals the relationship between task difficulty and sequential computation demands). This work represents a milestone in addressing the expressiveness limitations of linear RNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](../../ICCV2025/video_understanding/online_dense_point_tracking_with_streaming_memory.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/video_understanding/alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)
- [\[NeurIPS 2025\] Dense SAE Latents Are Features, Not Bugs](dense_sae_latents_are_features_not_bugs.md)
- [\[NeurIPS 2025\] TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video](tapvid-360_tracking_any_point_in_360_from_narrow_field_of_view_video.md)

</div>

<!-- RELATED:END -->
