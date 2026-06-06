---
title: >-
  [Paper Note] Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation
description: >-
  [ICML 2026][3D Vision][Optimal Transport] This paper introduces CDOT (Convex Distance Operator Transport), which "operatorizes" the distance matrices and coupling of each metric space. By replacing the non-convex squared…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Optimal Transport"
  - "Gromov–Wasserstein"
  - "Distance Operator"
  - "Convex Optimization"
  - "Frank–Wolfe"
date: 2026-05-08
content_hash: 5aab7c6f10c88732
---

# Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation

**Conference**: ICML 2026  
**arXiv**: [2606.02047](https://arxiv.org/abs/2606.02047)  
**Code**: None  
**Area**: Optimal Transport / Convex Optimization / Metric Measure Spaces  
**Keywords**: Optimal Transport, Gromov–Wasserstein, Distance Operator, Convex Optimization, Frank–Wolfe

## TL;DR
This paper introduces CDOT (Convex Distance Operator Transport), which "operatorizes" the distance matrices and coupling of each metric space. By replacing the non-convex squared pairwise distance difference in FGW with $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$, the authors obtain a heterogeneous space alignment framework that is **strictly convex with respect to the coupling $\pi$**, while remaining a valid pseudo-metric and possessing finite-sample risk bounds.

## Background & Motivation

**Background**: The de facto standard for comparing probability distributions across heterogeneous domains is Gromov–Wasserstein (GW) and its node-featured version, Fused GW (FGW). These methods measure structural misalignment using squared pairwise distance differences like $|d_\mathcal{X}(X,X') - d_\mathcal{Y}(Y,Y')|^2$, overlaid with a feature alignment term $\mathbb{E}_\pi[\|f_\mathcal{X}(X)-f_\mathcal{Y}(Y)\|^2]$. This framework is widely used in tasks requiring cross-domain comparisons such as graph classification, brain connectome alignment, and shape matching.

**Limitations of Prior Work**: The structural term in FGW/GW involves a **tensor product form** $\pi \otimes \pi$. While quadratic in $\pi$, its Hessian is indefinite, making the **objective non-convex**. Consequently: (i) Frank–Wolfe or projected gradient methods can only guarantee convergence to local stationary points; (ii) the methods are highly sensitive to graphs with different node cardinalities or local geometric jitter—slight perturbations in node counts or edge weights can cause pairwise distance matching to fail; (iii) almost no non-convex GW variant provides statistical consistency or finite-sample bounds for the "actual coupling output by the algorithm," usually only proving the distance between empirical and population risks. Table 1 compares GW, FGW, Entropic GW, Sliced GW, Low-rank GW, GW-SDP, IsoRank, and COPT, showing that the column for "simultaneously satisfying pseudo-metric, convexity, consistency, and finite-sample bounds on mm spaces" is almost entirely marked with ✗.

**Key Challenge**: Retaining the **pairwise distance** comparison of GW yields elegant pseudo-metric properties, but it also embeds non-convexity directly into the objective. Attempting to convexify (e.g., via entropy regularization, SDP relaxation, or slicing) often breaks the metric properties or loses the explicit transport plan. The authors aim to achieve three things simultaneously: **convexity + pseudo-metric + explicit coupling**.

**Goal**: (1) Propose an alignment objective that is strictly convex in $\pi$ and remain geometry-aware; (2) prove that it induces a valid pseudo-metric on attributed compact mm spaces; (3) provide an algorithm with global convergence in polynomial time, accompanied by a finite-sample risk decomposition.

**Key Insight**: This work elevates "distance" from a matrix to an operator—defining the distance operator as $(D_{\mathbb{P}_X} f)(x) = \int d_\mathcal{X}(x,x') f(x') \mathbb{P}_X(dx')$ and the "coupling" as a conditional expectation operator $(T_\pi g)(x) = \int g(y) \pi(dy|x)$. At the operator level, "aligning structures" naturally translates to whether $D_X T_\pi$ and $T_\pi D_Y$ **intertwine**, measured by the difference in Hilbert–Schmidt norms. This "operator intertwining error" is a linear function of $T_\pi$ → the squared HS norm is quadratic and positive semi-definite with respect to $\pi$ → directly convex.

**Core Idea**: By using "the conditional average distance from each $x$ to $Y$ under $\pi$" minus "the conditional average distance from each $y$ to $X$ under $\pi$" as the structural misalignment measure, the method **upgrades GW's edge-by-edge comparison to a comparison of aggregated distance profiles**. This eliminates the non-convex $\pi\otimes\pi$ tensor product and makes the model more robust to node count differences and local noise (e.g., Figure 1 shows two cycle graphs with different node counts are structurally equivalent under CDOT but not under FGW).

## Method

### Overall Architecture

The inputs are two attributed compact mm spaces $\mathfrak{X} = (\mathcal{X}, d_\mathcal{X}, \mathbb{P}_X, f_\mathcal{X})$ and $\mathfrak{Y} = (\mathcal{Y}, d_\mathcal{Y}, \mathbb{P}_Y, f_\mathcal{Y})$, and the output is a transport plan $\pi \in \Pi(\mathbb{P}_X, \mathbb{P}_Y)$ (optionally projected to a permutation matrix $\hat P$). The pipeline involves:

1.  **Operatorizing Geometry**: Representing the geometry of both spaces as distance operators $D_{\mathbb{P}_X}$ and $D_{\mathbb{P}_Y}$, and the coupling to be solved as a conditional expectation operator $T_\pi$.
2.  **Convex CDOT Objective**: $\mathcal{L}_\alpha(\pi) = (1-\alpha)\,\mathbb{E}_\pi[c_f(X,Y)] + \tfrac{\alpha}{2}\|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$, where $c_f(x,y)=\|f_\mathcal{X}(x)-f_\mathcal{Y}(y)\|_2^2$.
3.  **Discretization**: $D_{\hat{\mathbb{P}}_X}$ uses the normalized distance matrix $d_\mathcal{X}(X_i,X_j)/n_X$, and $T_\pi$ is represented by $n\pi$. The structural term becomes $\|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}}_Y}\|_F^2$, resulting in a **convex QP** over the transport polytope.
4.  **Solver**: The Frank–Wolfe (FW) algorithm is used (projection-free), solving a linear minimization (standard LP, solvable by OT solvers) at each step. The paper also introduces "lazy gradient FW" to update gradients incrementally using the quadratic structure, reducing the constant factors.
5.  **Optional Hard Matching**: The Hungarian algorithm is used to solve the LAP $\hat P = \arg\max_{P\in\mathcal{P}_n} \mathrm{Tr}(P^\top \hat\pi)$, projecting the soft coupling into a permutation.

### Key Designs

1.  **Operator-Layer Structure Regularization $\mathcal{R}(\pi) = \|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$**:
    *   Function: Replaces the bilinear, non-convex pairwise distance difference in FGW with a convex operator intertwining error in $\pi$.
    *   Mechanism: It can be proven via integral transforms that $\mathcal{R}(\pi) = \iint \Gamma_\pi(x,y)^2\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$, where $\Gamma_\pi(x,y) = \mathbb{E}_\pi[d_\mathcal{X}(x,X)|Y=y] - \mathbb{E}_\pi[d_\mathcal{Y}(y,Y)|X=x]$. That is, CDOT compares the difference between "the conditional average distance profile of $x$ in $\mathcal{X}$" and "the conditional average distance profile of $y$ in $\mathcal{Y}$" rather than pairwise distances. The convexity follows from the HS norm squared with respect to $T_\pi$ (and thus $\pi$), combined with the linearity of the feature term, making the overall $\mathcal{L}_\alpha$ strictly convex (Theorem 3.4). Furthermore, the square root of this objective $d_{\mathrm{CT}}^{(\alpha)}$ constitutes a pseudo-metric on attributed compact mm spaces (Theorem 3.5).
    *   Design Motivation: The "pairwise distance" $|d-d'|^2$ form of GW, which inherently includes $\pi\otimes\pi$, is the root of non-convexity. Replacing "pairwise" with "aggregated expectations" preserves geometric meaning (relationships between a node and its surrounding structure) while reducing the objective from bilinear to linear plus quadratic positive semi-definite forms, which is the key to convexification. This also makes the model naturally robust to node cardinality differences.

2.  **Dispersion Gap Decomposition: $\mathcal{R}_{\mathrm{GW},2}(\pi) - \mathcal{R}(\pi) = \mathcal{V}(\pi)$**:
    *   Function: Quantifies exactly what CDOT omits compared to GW within the same coordinate system, theoretically explaining why GW is non-convex.
    *   Mechanism: Dispersion is defined as $\mathcal{V}(\pi) = \iint (\mathrm{Var}_\pi[d_\mathcal{X}(x,X)|Y=y] + \mathrm{Var}_\pi[d_\mathcal{Y}(y,Y)|X=x])\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$, capturing the "conditional variance of distance" induced by the coupling. Theorem 3.7 strictly proves that GW structural cost = CDOT structural cost + dispersion, which is equivalent to decomposing GW into a "convex structural term + concave dispersion penalty."
    *   Design Motivation: This decomposition provides a clean geometric explanation for non-convexity. GW contour lines in the $(\mathcal{R},\mathcal{V})$ plane are $x+y=c$ lines, which can be tangent to non-convex "dart-shaped" feasible regions at local optima. CDOT contour lines are vertical $x=c$ lines, heading straight toward the global optimum horizontally (Figure 2). This also explains why GW prefers nearly deterministic couplings (minimizing dispersion), while CDOT tends toward more diffused soft couplings (requiring LAP post-processing for hard matching).

3.  **Frank–Wolfe + Glued Measure Risk Decomposition**:
    *   Function: Decomposes the gap between "the coupling $\hat\pi$ output by the algorithm on finite samples" and "the population optimum $\pi^*$" into two controllable terms.
    *   Mechanism: (i) Since discrete CDOT is a convex QP on the transport polytope, standard FW (with step size $\gamma_t=2/(t+2)$) achieves global convergence at $O(1/T)$. (ii) A glued measure $\Phi_n(\hat\pi)(dx,dy) = \int Q_X(dx|\hat x) Q_Y(dy|\hat y) \hat\pi(d\hat x, d\hat y)$ ($Q_X, Q_Y$ are optimal couplings for $\mathbb{P}_X$↔$\hat{\mathbb{P}}_X$ and $\mathbb{P}_Y$↔$\hat{\mathbb{P}}_Y$) is used to lift the discrete solution back to the population space. Theorem 5.6 provides $|\mathcal{L}_\alpha(\Phi_n(\hat\pi)) - \min \mathcal{L}_\alpha| \le \tfrac{32\alpha n_{\min}}{T+3} + C\,(W_1^{d_\mathcal{X}}(\mathbb{P}_X,\hat{\mathbb{P}}_X) + W_1^{d_\mathcal{Y}}(\mathbb{P}_Y,\hat{\mathbb{P}}_Y))$. Corollary 5.7 achieves almost sure risk consistency as $n_{\min}/T_n\to 0$.
    *   Design Motivation: Previous GW-style methods could only prove that empirical and population risks were close, but not that "the algorithm's returned $\hat\pi$ itself" approximated the population optimum. CDOT's convexity allows the "global convergence" and "empirical-population connection" steps to be controlled separately, providing the first finite-sample bound for an algorithmic output on mm spaces.

### Loss & Training

The empirical objective $\hat{\mathcal{L}}_\alpha(\pi) = (1-\alpha)\langle C_f, \pi\rangle_F + \tfrac{\alpha}{2}\, n_X n_Y \|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}}_Y}\|_F^2$. A fusion weight of $\alpha=0.5$ is the default for synthetic and real-world data. Iteration counts $T \in \{50, 100, 200\}$ are sufficient to reduce optimization error to negligible levels. The per-step complexity is $\mathcal{O}(n^3)$ (same order as FGW), and the lazy gradient variant significantly reduces the constant factor. Overall, the complexity is three orders of magnitude lower than the $\mathcal{O}(n^6)$ of GW-SDP. Applications requiring hard matching follow with the Hungarian algorithm.

## Key Experimental Results

### Main Results

On synthetic 2D clustering point clouds ($N=4n$, repeated 100 times, reporting MSE), CDOT is compared with FGW, Entropic FGW, IsoRank, Spectral, and COPT:

| $n$ | CDOT | FGW | EFGW | IsoRank | Spectral | COPT |
|----|------|------|------|---------|----------|------|
| 100 | **0.0077** | 0.0146 | 0.0098 | 0.0141 | 1.3447 | 0.6664 |
| 300 | **0.0027** | 0.0055 | 0.0038 | 0.0053 | 1.3276 | 0.6670 |
| 500 | **0.0016** | 0.0034 | 0.0025 | 0.0033 | 1.3373 | 0.6670 |

CDOT achieves the lowest MSE across all sample sizes, with MSE monotonically decreasing as $n$ increases—directly validating the statistical consistency of Theorem 5.6. Spectral and COPT, which do not use feature information, consistently fail to learn the correct alignment.

Node alignment accuracy for OASIS-3 brain connectomes (pairwise matching of 100 subjects):

| Method | Diffusion Distance | Geodesic Distance | Topology |
|------|----------------|---------------|----------|
| CDOT | **0.6136** | 0.4640 | – |
| FGW  | 0.1853 | **0.5375** | – |
| EFGW | 0.4097 | 0.4583 | – |
| IsoRank | – | – | **0.4055** |
| Spectral / COPT | – | – | 0.0737 / 0.0253 |

CDOT significantly leads when using diffusion distance (0.61 vs FGW 0.19). FGW performs slightly better on geodesic distance because FGW directly penalizes pairwise distance distortion, favoring geometric information sensitive to single shortest paths. Conversely, diffusion averages all paths, reducing the "pairwise difference contrast" required by GW; CDOT's operator aggregation successfully leverages this global signal.

Graph classification benchmarks (node counts 17–40):

| Dataset | CDOT | FGW | GW | COPT |
|--------|------|-----|----|----- |
| MUTAG  | **0.862** | 0.825 | 0.718 | 0.633 |
| IMDB-B | **0.642** | 0.602 | – | 0.637 |
| PROTEINS | **0.755** | 0.736 | 0.661 | 0.695 |
| NCI1   | **0.748** | 0.730 | 0.571 | 0.599 |
| ENZYMES| **0.513** | 0.445 | 0.238 | 0.235 |

CDOT wins on 5/5 datasets, with gains of 1.7–6.8 percentage points over FGW and even larger gaps compared to GW/COPT.

### Ablation Study

| Configuration | Key Finding | Description |
|------|----------|------|
| $\alpha$ sweep | Most stable near $\alpha=0.5$ | Feature and structural terms need balance; excessive structural weight biases toward pure geometry. |
| Iteration count $T$ | $T=200$ is sufficient | Consistent with the $O(1/T)$ rate in Theorem 5.6. |
| Distance normalization | Stable with max-normalization | Operator scale is sensitive to the HS norm. |
| Soft vs. LAP | Soft couplings are naturally diffuse | Dispersion gap explanation: CDOT does not penalize dispersion, requiring LAP for hard matching. |
| Distance (Diff. vs Geo.) | CDOT prefers diffusion | Confirms CDOT uses aggregated profiles while FGW uses pairwise differences. |

### Key Findings

- **Convexity matters**: CDOT systematically outperforms FGW on synthetic data. The fact that EFGW (which smooths GW to be "near-convex") performs similarly to CDOT reinforces that non-convexity is FGW's primary bottleneck.
- **Dispersion gap is more than theory**: The inverse performance on brain connectomes (diffusion vs. geodesic) corresponds exactly to geometries with high/low distance variance. CDOT performs better in high-variance geometries as it does not penalize dispersion.
- **Consistency of algorithm outputs**: CDOT wins consistently on real data like brain connectomes and graph classification, often with lower variance (std 0.04 vs FGW 0.07 on IMDB-B), suggesting FW is much more stable on a convex landscape than FGW in local optima traps.

## Highlights & Insights

- **"Operatorizing Geometry" is the master key to convex GW**: By lifting distance matrices and couplings to operators, "structural alignment" naturally becomes the intertwining error of two operators, eliminating the bilinear $\pi\otimes\pi$. This paradigm of "lifting to operator layers then using linear spectral theory" has precedents in kernel mean embedding and HSIC, but its application to convexify GW is novel. The key is that $T_\pi$ is a linear operator of $\pi$.
- **Dispersion gap is a rare "exact decomposition of non-convexity"**: Being able to write $\mathcal{R}_{\mathrm{GW}} = \mathcal{R}_{\mathrm{CDOT}} + \mathcal{V}$ (convex + concave), alongside the geometric contour analysis in the $(\mathcal{R},\mathcal{V})$ plane, upgrades the "GW local optima trap" from an empirical observation to a structural explanation. This analysis can be extended to other $\pi\otimes\pi$ based objectives like sliced GW or low-rank GW.
- **Glued measure is the standard bridge between discrete and population optima**: Using $\Phi_n$ to lift $\hat\pi$ back to the population space and $W_1$ to control statistical error is a powerful trick that can be reused in other consistency proofs for algorithmic outputs, such as entropic OT estimation or neural OT.

## Limitations & Future Work

- The authors acknowledge: (i) Zero-discriminativity is still a step away—$d_{\mathrm{CT}}^{(\alpha)} = 0$ does not imply mm space equivalence, but rather a form of "fractional structural equivalence." A complete characterization of the zero-set remains for future work. (ii) The statistical rate on $\mathbb{R}^d$ is $O(n^{-1/d})$, and no polynomial rate is guaranteed in infinite-dimensional settings without additional structure.
- Observations: (i) The produced couplings are highly diffuse; almost all hard-matching applications require the Hungarian algorithm, yet the approximation gap between soft and hard remains unquantified. (ii) The per-step $\mathcal{O}(n^3)$ complexity is still a bottleneck for graphs with thousands of nodes; lazy FW only provides a constant speedup without offering sub-cubic variants like sliced or low-rank methods. (iii) Assumption 5.5 (Lipschitzness of the optimal coupling's conditional distribution under $W_1$) is quite strong and requires case-by-case verification in discrete graph tasks.
- Improvement ideas: Use the dispersion decomposition as a "diagnostic tool" to design "semi-convex CDOT" with controllable concave terms, aiming to partially restore GW's hard-matching preference while retaining CDOT's global optimality; combine with entropic regularization or slicing for $\mathcal{O}(n^2)$ approximations; extend the operator framework to non-Euclidean metrics (tree metrics, Wasserstein-on-Wasserstein).

## Related Work & Insights

- **vs FGW (Vayer et al., 2020)**: FGW aligns pairwise distances via $\mathbb{E}_{\pi\otimes\pi}|d_\mathcal{X}-d_\mathcal{Y}|^2$, which is non-convex in $\pi$. CDOT uses operator intertwining, resulting in a convex objective. Both share $\mathcal{O}(n^3)$ complexity, but CDOT adds pseudo-metric properties, consistency, and finite-sample bounds.
- **vs Entropic GW (Peyré et al., 2016)**: Entropy smoothes GW to be "near-convex" but breaks the original pseudo-metric properties and depends on $\varepsilon$ tuning. CDOT is convex in its **original unregularized** form without needing an entropy bridge.
- **vs GW-SDP (Chen et al., 2024)**: Achieves convexity via SDP relaxation at the cost of $\mathcal{O}(n^6)$ complexity and loss of pseudo-metric properties. CDOT maintains $\mathcal{O}(n^3)$ and remains a pseudo-metric.
- **vs Sliced GW / Sliced FGW**: Slicing avoids non-convexity by projecting to 1D but often fails to output explicit couplings or sacrifices metric properties. CDOT retains the "explicit transport plan + metricity + convexity" triad.
- **Inspiration**: The "lift geometry to operators, operate in RKHS / $L^2$ layers" paradigm is highly suitable for graph kernels, scene-graph alignment, and cross-modal OT where both "convexity" and "geometry-awareness" are required. The dispersion gap decomposition serves as a general tool for future research on the convexification of non-convex OT objectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first truly convex geometry-aware OT objective; the dispersion gap provides a precise decomposition of GW's non-convexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers typical heterogeneous alignment scenarios across synthetic, brain connectome, and 5 graph datasets, though checks on larger scales (>1k nodes) or high-frequency applications like 3D shapes are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure; Table 1 provides high information density; the flow from convexity and pseudo-metrics to dispersion gaps and risk bounds is seamless.
- Value: ⭐⭐⭐⭐ Fills a critical gap in the GW family regarding "convexity + consistency + finite-sample bounds," offering significant theoretical value and immediate practical utility for graph alignment and brain connectome analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AvAtar: Learning to Align via Active Optimal Transport](avatar_learning_to_align_via_active_optimal_transport.md)
- [\[ICML 2026\] Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)
- [\[NeurIPS 2025\] Fully Dynamic Algorithms for Chamfer Distance](../../NeurIPS2025/3d_vision/fully_dynamic_algorithms_for_chamfer_distance.md)
- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](../../ICCV2025/3d_vision/identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[CVPR 2026\] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](../../CVPR2026/3d_vision/glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)

</div>

<!-- RELATED:END -->
