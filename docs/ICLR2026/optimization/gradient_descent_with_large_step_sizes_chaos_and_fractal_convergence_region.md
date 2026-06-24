---
title: >-
  [Paper Note] Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region
description: >-
  [ICLR 2026][Optimization][large learning rate] This paper provides a rigorous proof for matrix factorization problems: when gradient descent uses large step sizes close to the critical threshold, fractal convergence boundaries and chaotic dynamics emerge in the parameter space. The final minimum reached (or even whether convergence occurs) becomes extremely sensitive to initialization, causing commonly assumed implicit biases—such as "flatness/minimum norm/balance"—to fail co…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "large learning rate"
  - "matrix factorization"
  - "chaotic dynamics"
  - "fractal convergence region"
  - "implicit bias"
  - "topological entropy"
date: 2026-05-08
content_hash: b90bf0aefce396ff
---

# Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wsxGCaBjWC](https://openreview.net/forum?id=wsxGCaBjWC)  
**Code**: To be confirmed  
**Area**: optimization  
**Keywords**: large learning rate, matrix factorization, chaotic dynamics, fractal convergence region, implicit bias, topological entropy  

## TL;DR
This paper provides a rigorous proof for matrix factorization problems: when gradient descent uses large step sizes close to the critical threshold, fractal convergence boundaries and chaotic dynamics emerge in the parameter space. The final minimum reached (or even whether convergence occurs) becomes extremely sensitive to initialization, causing commonly assumed implicit biases—such as "flatness/minimum norm/balance"—to fail completely.

## Background & Motivation
- **Background**: The step size (learning rate) is a key hyperparameter determining gradient descent dynamics and generalization performance. Recent research generally suggests that large step sizes introduce "good" implicit biases—favoring flat minima, balanced solutions, sparse features, and minimum-norm solutions while improving generalization (phenomena like Edge of Stability and catapulting belong to this line of work).
- **Limitations of Prior Work**: Most existing conclusions are restricted to small or bounded step size ranges, lacking rigorous theory for step sizes "approaching the critical maximum." Existing chaos analyses (Kong & Tao 2020; Chen et al. 2024b) only handle 1D scalar losses and fail to characterize the "selection of minima" central to high-dimensional optimization.
- **Key Challenge**: While empirical observations have suggested that the set of convergent step sizes or initializations can form fractals (Sohl-Dickstein 2024; Zhu et al. 2023), no study has strictly proven the origin of these fractals or their implications for implicit bias. The optimistic narrative of "large step size = good bias" conflicts with the empirical phenomenon of "fractal/chaos = unpredictability."
- **Goal**: Using analytically tractable non-convex overparameterized models (matrix factorization), this paper answers two fundamental questions: given an initial point, what is the critical (maximum) step size that allows convergence? And what implicit bias does gradient descent induce when approaching this critical step size?
- **Core Idea**: **[Strict Characterization of Chaos]** By treating gradient descent as a discrete dynamical system in parameter space, the authors prove that at near-critical step sizes, it acts as a "stretch + multiple fold" covering map. This leads to the emergence of self-similar (fractal) structures at the convergence boundaries, sensitivity to initialization, and positive topological entropy—marking the first rigorous characterization of chaos and fractal convergence regions in machine learning optimization.

## Method

### Overall Architecture
Rather than proposing a new algorithm, this paper provides a precise dynamical analysis of "constant step size gradient descent on shallow matrix factorization with $\ell_2$ regularization: $\min_{U,V}\tfrac12\|U^\top V-Y\|_F^2+\tfrac{\lambda}{2}(\|U\|_F^2+\|V\|_F^2)$." The logical progression moves from simplicity to complexity: first deriving the closed-form critical step size and proving chaos/topological entropy for scalar factorization $L=\tfrac12(u^\top v-y)^2$ (unregularized), then adding regularization to prove fractal convergence boundaries, followed by using symmetric reduction to project high-dimensional boundaries onto a plane for self-similarity analysis, generalizing these results to general matrix factorization under orthogonal initialization, and finally revealing the unified mechanism—the "folding-covering" behavior of the gradient descent update map.

```mermaid
flowchart TD
    A["Scalar Factorization L=½(uᵀv−y)²<br/>Unregularized"] -->|Closed-form Critical Step Size + Topological Entropy ≥ log 3| B["Chaos: Sensitivity to Initialization<br/>Convergence Region ≈ Smooth Ellipsoid"]
    A2["Scalar Factorization + ℓ₂ Regularization"] -->|Symmetric Reduction T(u,v)=(uᵀv,‖u‖²+‖v‖²)| C["Planar Map F<br/>Projected Boundary T(∂D'') Self-similar (Degree 3)"]
    C -->|Reattaching Fibers| D["Fractal Convergence Boundary<br/>Box Dimension ≈ 1.249"]
    B --> E["General Matrix Factorization (Orthogonal Init)<br/>Dynamics Decouple Column-wise → Conclusions Hold"]
    D --> E
    E --> F["Unified Mechanism: GDη is a 3-Covering Map<br/>Stretch + Fold → Self-similarity + Mixing → Sensitivity"]
    F --> G["Polynomial Networks: GDη is a Covering Map Almost Everywhere<br/>Exp: Deep ReLU Nets Also Exhibit Fractals/Chaos"]
```

### Key Designs

**1. Closed-form Critical Step Size and Chaos without Regularization (Theorem 1–2)**: For unregularized scalar factorization $L=\tfrac12(u^\top v-y)^2$, the authors provide an exact expression for the critical step size valid for almost all initializations: $\eta^*(\bar u,\bar v)=\min\big\{\tfrac{1}{|y|},\ \tfrac{8}{\|\bar u\|^2+\|\bar v\|^2+\sqrt{(\|\bar u\|^2+\|\bar v\|^2)^2-16y(\bar u^\top\bar v-y)}}\big\}$, where convergence occurs for $\eta<\eta^*$ and divergence occurs almost surely for $\eta>\eta^*$. The two terms correspond to the instability of all minima when $\eta|y|>1$ and a convergence region $D'_\eta$ characterized by an ellipsoidal domain. Regarding sensitivity: for fixed $\eta|y|<1$, any small neighborhood around a point on the boundary $\partial D'_\eta$ contains initializations converging to arbitrarily large norm minima, arbitrarily imbalanced minima, or the saddle point $(0,0)$—meaning the final state is unpredictable regardless of initialization precision. Theorem 1 also proves topological entropy $h(\mathrm{GD}_\eta)\ge\log 3$ and the existence of periodic orbits of any period. Theorem 2 shows that even after excluding the zero-measure attraction basins of unstable minima, the support of the "sharpness/norm" distribution of sampled minima still covers the entire interval $(\gamma_{\min},2/\eta)$.

**2. Regularization-Induced Fractal Boundaries (Theorem 4)**: Adding $\ell_2$ regularization bounds the set of global minima. While intuition might suggest this makes behavior more predictable, the opposite occurs—not only is the convergence endpoint unpredictable, but "whether convergence happens at all" becomes unpredictable. The authors prove that for $0<\lambda\le\min\{1/\eta-|y|,1/(2\eta)\}$, the projection $T(\partial D''_\eta)$ of the convergence boundary is a degree-3 self-similar set with a box-counting dimension estimated at $1.249$. When $y=0$, the convergence region is even unbounded, forming "spikes" that are infinitely replicated at all scales by self-similarity. Sensitivity is manifested by the algorithm choosing either the nearest minimum $p_-$ or the furthest $p_+$, but this choice is unpredictable near the boundary, contrasting with the stable "nearest-distance bias" seen at small step sizes.

**3. Symmetric Reduction of High-Dimensional Boundaries (Proposition 3)**: Analyzing $\partial D''_\eta\subset\mathbb R^{2d}$ directly is difficult due to saddle point basins creating internal boundaries and the high dimensionality itself. The authors introduce the map $T(u,v)=(u^\top v,\ \|u\|_2^2+\|v\|_2^2)$ to reduce parameters to $\mathbb R^2$, proving a planar map $F:\mathbb R^2\to\mathbb R^2$ exists such that $(z_{t+1},w_{t+1})=F(z_t,w_t)$. Since the fibers of $T$ are almost everywhere smooth manifolds like $S^{d-1}\times S^{d-1}$, the geometric complexity of the high-dimensional convergence boundary is preserved in the 2D projection $T(\partial D''_\eta)$, enabling complete study of fractal properties on a plane.

**4. Folding-Covering Mechanism and Generalization to Real Networks (Proposition 6)**: All phenomena are attributed to a unified mechanism: there exists a set $C$ on which $\mathrm{GD}_\eta$ is a 3-covering map, stretching and folding $C$ three times to cover $\mathrm{GD}_\eta(C)\supset C$. When $C$ contains an invariant boundary $\mathrm{GD}_\eta(\partial D_\eta)=\partial D_\eta$, the boundary is repeatedly stretched and folded over itself, creating self-similarity and "mixing" that generates sensitivity. This is generalized to any polynomial-activation network with polynomial loss: except for finite step size values, there exists a zero-measure set $K_\eta$ such that $\mathrm{GD}_\eta$ is a covering map on every connected component of $\mathbb R^p\setminus K_\eta$, explaining the fractals observed in deep ReLU networks.

## Key Experimental Results

Experiments verify that the theoretical chaos/fractal phenomena exist in real training scenarios beyond the toy matrix factorization model.

### Main Results: Two Step-Size Intervals in Deep ReLU Networks
A depth-3 ReLU network was trained on a CIFAR-10 binary subset (MSE loss, 5000 steps, fixed init) while scanning step sizes:

| Step Size Interval | Final Norm/Sharpness Behavior | Theoretical Correspondence |
|---|---|---|
| EoS Interval (Smaller) | Follows smooth curves, sharpness ≈ $2/\eta$, predictable | Consistent with Cohen et al. (2021) |
| Chaos Interval (Near Critical) | Highly sensitive to step size, sharpness spans all values below $2/\eta$ | Verifies Theorem 2 |

Adding Polyak momentum maintains the two-interval phenomenon, though sharpness in the small-step interval forms clusters rather than the $(2+2\beta)/\eta$ predicted elsewhere.

### Key Findings: Sensitivity to Initialization under Weight Decay
Training the same network with weight decay for 3000 steps and visualizing random 2D slices of parameter space colored by final loss/sharpness:

| Observed Feature | Result |
|---|---|
| Final Loss Slice | Fractal basins of attraction appear: global minima (dark blue), sub-optimal (white), and divergence (dark red) are intertwined. |
| Final Sharpness Slice | Extremely sensitive to initialization, spanning a wide range below $2/\eta$; similar patterns exist without weight decay (see Appendix). |

- **In the chaos regime, the "best prediction" for final sharpness is the entire interval $(\gamma_{\min},2/\eta)$**—no amount of initialization precision can reduce this error.
- **Fractals are not exclusive to toy models**: Deep ReLU networks with real data exhibit fractal basin boundaries.
- **Regularization/Weight Decay amplifies unpredictability**: It upgrades the problem from "unpredictable endpoint" to "unpredictable convergence."
- **EoS and Chaos are distinct intervals**: They appear sequentially as step size increases; the former is predictable, while the latter is not.

## Highlights & Insights
- **First strict proof of fractal convergence regions and chaos in ML optimization**: Moving beyond 1D analysis to high-dimensional problems with closed-form critical step sizes and box-dimension estimates.
- **Dismantling the optimistic "Large Step Size = Good Bias" narrative**: Near the critical step size, implicit biases like flatness, minimum norm, or balance completely vanish.
- **Symmetric reduction $T$ "visualizes" high-dimensional chaos**: A key technical contribution that preserves boundary geometry in a 2D plane.
- **The Folding-Covering mechanism as a root cause**: A unified geometric explanation for both self-similarity (folding over boundaries) and sensitivity (mixing via folding).
- **Questioning the predictability of Edge of Stability**: Identifying an overlooked "chaos interval" beyond EoS where sharpness no longer tracks $2/\eta$.

## Limitations & Future Work
- **Theoretical rigor limited to matrix factorization + orthogonal init**: Chaotic behavior for general initializations or deeper matrix models currently relies on experimental evidence or "transient chaos" intuition.
- **Incomplete covering map mechanism**: Proposition 6 proves $\mathrm{GD}_\eta$ is a covering map but does not strictly prove that the necessary conditions for chaos (mapping $C$ back to itself) always hold in general networks.
- **Invariants fail at large step sizes**: While $d=1$ dynamics show any continuous invariant must be constant (breaking imbalance conservation $u^2-v^2$), this isn't strictly generalized for $d \ge 2$.
- **Lack of practical countermeasures**: The conclusion is a warning about the danger of near-critical step sizes; how to leverage large steps without entering the chaotic zone remains open.

## Related Work & Insights
- **Large Step Size Dynamics**: Extends studies like Edge of Stability (Cohen et al. 2021) and catapult (Lewkowycz et al. 2020) by exploring the "even larger" chaotic interval.
- **Implicit Bias**: Challenges findings that large step sizes induce balance (Wang et al. 2022) or nearest-distance bias (Gunasekar et al. 2018), showing these fail near the critical threshold.
- **Chaos in Optimization**: Advances beyond the 1D Li-Yorke chaos proofs of Kong & Tao (2020) and Chen et al. (2024b) to characterize high-dimensional endpoint sensitivity.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (First strict characterization of fractal convergence and chaos in ML.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Verification on real networks, though primarily focused on supporting theory.)
- **Writing Quality**: ⭐⭐⭐⭐ (Logical progression from simple to complex; high theoretical density.)
- **Value**: ⭐⭐⭐⭐⭐ (Profound implications for understanding large learning rates and implicit bias.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Flatland: The Adventures of Gradient Descent with Large Step Sizes](../../ICML2026/optimization/flatland_the_adventures_of_gradient_descent_with_large_step_sizes.md)
- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](on_the_convergence_direction_of_gradient_descent.md)
- [\[ICLR 2026\] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime](on_the_convergence_behavior_of_preconditioned_gradient_descent_toward_the_rich_l.md)
- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)
- [\[ICML 2026\] Gradient Descent with Large Step Size Restores Symmetry in Deep Linear Networks with Multi-Pathway](../../ICML2026/optimization/gradient_descent_with_large_step_size_restores_symmetry_in_deep_linear_networks_.md)

</div>

<!-- RELATED:END -->
