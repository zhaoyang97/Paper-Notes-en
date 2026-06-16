---
title: >-
  [Paper Note] Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation
description: >-
  [ICML 2026][3D Vision][Gromov–Wasserstein] This paper proposes CDOT (Convex Distance Operator Transport), which "operatorizes" the distance matrices and coupling of each metric space. By replacing the non-convex squared pairwise distance difference in FGW with $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$, the authors obtain a framework for heterogeneous space al
tags:
  - ICML 2026
  - 3D Vision
  - Gromov–Wasserstein
  - Frank–Wolfe
date: 2026-05-08
content_hash: 4c155625842257d3
---
# Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation

**Conference**: ICML 2026  
**arXiv**: [2606.02047](https://arxiv.org/abs/2606.02047)  
**Code**: N/A  
**Area**: Optimal Transport / Convex Optimization / Metric Measure Spaces  
**Keywords**: Optimal Transport, Gromov–Wasserstein, Distance Operator, Convex Optimization, Frank–Wolfe

## TL;DR
This paper proposes CDOT (Convex Distance Operator Transport), which "operatorizes" the distance matrices and coupling of each metric space. By replacing the non-convex squared pairwise distance difference in FGW with $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$, the authors obtain a framework for heterogeneous space alignment that is **strictly convex with respect to the coupling $\pi$**, while remaining a valid pseudo-metric and possessing finite-sample risk bounds.

## Background & Motivation

**Background**: The de facto standard for comparing probability distributions across heterogeneous domains is Gromov–Wasserstein (GW) and its feature-augmented version, Fused GW (FGW). They measure structural misalignment using squared pairwise distance differences like $|d_\mathcal{X}(X,X') - d_\mathcal{Y}(Y,Y')|^2$, overlaid with a feature alignment term $\mathbb{E}_\pi[\|f_\mathcal{X}(X)-f_\mathcal{Y}(Y)\|^2]$. This framework is widely used in tasks requiring cross-domain comparisons, such as graph classification, brain connectome alignment, and shape matching.

**Limitations of Prior Work**: The structural term in FGW/GW involves a **tensor product form** $\pi \otimes \pi$, which is quadratic in $\pi$ but with an indefinite Hessian, making the **objective non-convex**. Consequently: (i) Algorithms like Frank–Wolfe or projected gradient only guarantee convergence to local stationary points; (ii) The models are highly sensitive to differences in node cardinality or local geometric jitters—slight perturbations in nodes or edge weights can cause pairwise distance comparisons to fail; (iii) Almost all non-convex GW variants lack statistical consistency and finite-sample bounds for the "actual coupling output by the algorithm," only proving the distance between empirical and population risks. Table 1 compares GW, FGW, Entropic GW, Sliced GW, Low-rank GW, GW-SDP, IsoRank, and COPT, showing that the column "Simultaneously possessing pseudo-metric, convexity, consistency, and finite-sample bounds on mm spaces" is almost entirely marked with ✗.

**Key Challenge**: Retaining the **pairwise distance** comparison of GW yields elegant pseudo-metric properties but hardcodes non-convexity into the objective. Attempts to convexify (e.g., entropic regularization, SDP relaxation, slicing) often break the metric property or lose an explicit transport plan. The authors aim to achieve three things simultaneously: **convexity + pseudo-metric + explicit coupling**.

**Goal**: (1) Provide an alignment objective that is strictly convex in $\pi$ and geometry-aware; (2) Prove that it induces a valid pseudo-metric on attributed compact mm spaces; (3) Develop an algorithm with global convergence in polynomial time, accompanied by a finite-sample risk decomposition.

**Key Insight**: Elevate "distances" from matrices to operators—defining the distance operator as $(D_{\mathbb{P}_X} f)(x) = \int d_\mathcal{X}(x,x') f(x') \mathbb{P}_X(dx')$ and the "coupling" as a conditional expectation operator $(T_\pi g)(x) = \int g(y) \pi(dy|x)$. At the operator level, "structural alignment" naturally becomes the question of whether $D_X T_\pi$ and $T_\pi D_Y$ **intertwine**, measured by the Hilbert–Schmidt norm difference. This "operator commutation error" is a linear function of $T_\pi \rightarrow$ the squared HS norm is quadratic and positive semi-definite in $\pi \rightarrow$ directly convex.

**Core Idea**: Use the difference between "the conditional average distance from each $x$ to $Y$ viewed by $\pi$" and "the conditional average distance from each $y$ to $X$ viewed by $\pi$" as the structural misalignment measure. **This upgrades the edge-by-edge comparison of GW to an aggregated distance profile comparison**, eliminating the non-convex $\pi\otimes\pi$ tensor product while remaining robust to cardinality differences and local noise (as shown in Figure 1, two cycles with different node counts are structurally equivalent under CDOT but not under FGW).

## Method

### Overall Architecture

The core problem CDOT addresses is how to measure structural misalignment between two heterogeneous metric spaces without sacrificing convexity. While FGW compares pairwise distance differences $|d_\mathcal{X}-d_\mathcal{Y}|^2$—which is geometrically intuitive but forces the non-convex $\pi\otimes\pi$ into the objective—CDOT adopts a different perspective. It first promotes both "distance" and "coupling" to operators, then formulates "structural alignment" as the commutation of these two operators.

Specifically, the input consists of two attributed compact mm spaces $\mathfrak{X} = (\mathcal{X}, d_\mathcal{X}, \mathbb{P}_X, f_\mathcal{X})$ and $\mathfrak{Y} = (\mathcal{Y}, d_\mathcal{Y}, \mathbb{P}_Y, f_\mathcal{Y})$, and the output is a transport plan $\pi \in \Pi(\mathbb{P}_X, \mathbb{P}_Y)$. The first step expresses the distances on both sides as distance operators $D_{\mathbb{P}_X}, D_{\mathbb{P}_Y}$ and the target coupling as a conditional expectation operator $T_\pi$. The CDOT objective is then defined as the feature alignment term plus a "commutation error" structural term: $\mathcal{L}_\alpha(\pi) = (1-\alpha)\,\mathbb{E}_\pi[c_f(X,Y)] + \tfrac{\alpha}{2}\|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$, where $c_f(x,y)=\|f_\mathcal{X}(x)-f_\mathcal{Y}(y)\|_2^2$. For finite samples, the distance operator is discretized using a normalized distance matrix $d_\mathcal{X}(X_i,X_j)/n_X$, the coupling is represented by $n\pi$, and the structural term becomes $\|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}_Y}}\|_F^2$. The entire problem is a convex quadratic program (QP) over the transport polytope. It is solved using the projection-free Frank–Wolfe algorithm, where each step involves solving a linear minimization problem (a standard LP, solvable via OT solvers). If a hard matching is required downstream, the Hungarian algorithm projects the soft coupling onto a permutation matrix $\hat P = \arg\max_{P\in\mathcal{P}_n} \mathrm{Tr}(P^\top \hat\pi)$.

### Key Designs

**1. Operator-level Structural Regularization: Replacing FGW's Non-convex Pairwise Differences with a Convex "Commutation Error"**

The bilinear form of $\pi\otimes\pi$ in the FGW structural term is the source of non-convexity. CDOT replaces it with the regularization term $\mathcal{R}(\pi) = \|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$. A crucial observation is that via integral transformation, $\mathcal{R}(\pi) = \iint \Gamma_\pi(x,y)^2\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$, where $\Gamma_\pi(x,y) = \mathbb{E}_\pi[d_\mathcal{X}(x,X)\mid Y=y] - \mathbb{E}_\pi[d_\mathcal{Y}(y,Y)\mid X=x]$. In other words, CDOT compares the "conditional average distance profile of $x$ in $\mathcal{X}$" against the "conditional average distance profile of $y$ in $\mathcal{Y}$"—aggregating "pairwise distances" into "expected distances."

This reformulation is effective because $T_\pi$ is a linear operator with respect to $\pi$, so the squared HS norm is quadratic and positive semi-definite in $T_\pi$ (and thus $\pi$). Combined with the linear feature term, the overall $\mathcal{L}_\alpha$ is strictly convex (Theorem 3.4). Meanwhile, it retains geometric meaning by characterizing the relationship between a node and its surrounding structure; thus, the square root of the objective $d_{\mathrm{CT}}^{(\alpha)}$ remains a valid pseudo-metric on attributed compact mm spaces (Theorem 3.5). Aggregated expectations are also naturally robust to cardinality differences: two spaces with similar geometry but different node counts can be identified as equivalent under CDOT (Figure 1).

**2. Dispersion Gap Decomposition: Quantifying What CDOT Omits Relative to GW**

To explain why GW is non-convex and where CDOT gains convexity, the paper provides a clean decomposition within the same coordinate system. Defining dispersion $\mathcal{V}(\pi) = \iint \big(\mathrm{Var}_\pi[d_\mathcal{X}(x,X)\mid Y=y] + \mathrm{Var}_\pi[d_\mathcal{Y}(y,Y)\mid X=x]\big)\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$, which captures the "conditional variance of distances" induced by the coupling, Theorem 3.7 proves that the GW structural cost is exactly the CDOT structural cost plus this dispersion:

$$\mathcal{R}_{\mathrm{GW},2}(\pi) - \mathcal{R}(\pi) = \mathcal{V}(\pi).$$

Thus, GW equals "Convex structural term + Concave dispersion penalty." This gives a geometric explanation for non-convexity: in the $(\mathcal{R},\mathcal{V})$ plane, GW contour lines are diagonal $x+y=c$, which can be tangent to non-convex "dart-shaped" feasible regions at local optima. CDOT contour lines are vertical $x=c$, heading straight toward the global optimum along the horizontal axis (Figure 2). This also explains why GW prefers nearly deterministic couplings (to minimize dispersion), while CDOT does not penalize dispersion and tends towards more diffuse soft couplings, requiring a subsequent LAP for hard matching.

**3. Frank–Wolfe + Glued Measure: Connecting Algorithmic Output to Population Optimality**

Prior GW-style methods could prove empirical risk approximates population risk but could not guarantee that "the specific $\hat\pi$ returned by the algorithm" approximates the population optimum. CDOT's convexity allows this to be controlled in two steps. First, discrete CDOT is a convex QP, and standard FW (step size $\gamma_t=2/(t+2)$) achieves global convergence at $O(1/T)$, bounding the optimization error. Second, a glued measure $\Phi_n(\hat\pi)(dx,dy) = \int Q_X(dx\mid \hat x) Q_Y(dy\mid \hat y) \hat\pi(d\hat x, d\hat y)$ (where $Q_X, Q_Y$ are optimal couplings for $\mathbb{P}_X\leftrightarrow\hat{\mathbb{P}}_X$ and $\mathbb{P}_Y\leftrightarrow\hat{\mathbb{P}}_Y$) is used to lift the discrete solution back to the population space, bounding the statistical error via $W_1$. Together, Theorem 5.6 provides a finite-sample bound for the algorithmic output:

$$\big|\mathcal{L}_\alpha(\Phi_n(\hat\pi)) - \min \mathcal{L}_\alpha\big| \le \tfrac{32\alpha n_{\min}}{T+3} + C\,\big(W_1^{d_\mathcal{X}}(\mathbb{P}_X,\hat{\mathbb{P}}_X) + W_1^{d_\mathcal{Y}}(\mathbb{P}_Y,\hat{\mathbb{P}}_Y)\big),$$

where the first term is optimization error and the second is statistical error. Corollary 5.7 further provides almost sure risk consistency as $n_{\min}/T_n\to 0$—a guarantee for algorithmic output previously unavailable for the GW family on mm spaces.

### Loss & Training

The empirical objective is $\hat{\mathcal{L}}_\alpha(\pi) = (1-\alpha)\langle C_f, \pi\rangle_F + \tfrac{\alpha}{2}\, n_X n_Y \|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}}_Y}\|_F^2$. The fusion weight defaults to $\alpha=0.5$ (applicable to both synthetic and real data), and iterations $T \in \{50,100,200\}$ are sufficient to make optimization error negligible. Per-step complexity is $\mathcal{O}(n^3)$ (same as FGW). The paper's lazy gradient FW variant uses the quadratic structure for incremental gradient updates to reduce constant factors, making it three orders of magnitude faster than the $\mathcal{O}(n^6)$ GW-SDP. Applications requiring hard matching apply a Hungarian step after convergence.

## Key Experimental Results

### Main Results

Synthetic 2D clustering point clouds ($N=4n$, 100 repetitions, reporting MSE). CDOT is compared against FGW, Entropic FGW, IsoRank, Spectral, and COPT:

| $n$ | CDOT | FGW | EFGW | IsoRank | Spectral | COPT |
|----|------|------|------|---------|----------|------|
| 100 | **0.0077** | 0.0146 | 0.0098 | 0.0141 | 1.3447 | 0.6664 |
| 300 | **0.0027** | 0.0055 | 0.0038 | 0.0053 | 1.3276 | 0.6670 |
| 500 | **0.0016** | 0.0034 | 0.0025 | 0.0033 | 1.3373 | 0.6670 |

CDOT achieves the lowest MSE across all sample sizes, and the MSE decreases monotonically as $n$ increases—empirically validating the statistical consistency of Theorem 5.6. Spectral and COPT, which ignore feature information, fail to learn the correct alignment.

OASIS-3 Brain Connectome Node Alignment Accuracy (pairwise matching of 100 subjects):

| Method | Diffusion Distance | Geodesic Distance | Topology |
|------|----------------|---------------|----------|
| CDOT | **0.6136** | 0.4640 | – |
| FGW  | 0.1853 | **0.5375** | – |
| EFGW | 0.4097 | 0.4583 | – |
| IsoRank | – | – | **0.4055** |
| Spectral / COPT | – | – | 0.0737 / 0.0253 |

CDOT leads significantly under diffusion distance (0.61 vs. FGW 0.19). FGW performs slightly better on geodesic distance because FGW directly penalizes pairwise distance distortion, favoring geometric information sensitive to single shortest paths. Diffusion distance, which averages all paths, lowers the "pairwise difference contrast" needed by GW, whereas CDOT's operator aggregation effectively utilizes such global signals.

Graph Classification Benchmarks (Node counts 17–40):

| Dataset | CDOT | FGW | GW | COPT |
|--------|------|-----|----|----- |
| MUTAG  | **0.862** | 0.825 | 0.718 | 0.633 |
| IMDB-B | **0.642** | 0.602 | – | 0.637 |
| PROTEINS | **0.755** | 0.736 | 0.661 | 0.695 |
| NCI1   | **0.748** | 0.730 | 0.571 | 0.599 |
| ENZYMES| **0.513** | 0.445 | 0.238 | 0.235 |

CDOT outperforms in 5/5 datasets, with improvements over FGW ranging from 1.7–6.8 percentage points, and larger gaps over GW/COPT.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|----------|------|
| $\alpha$ sweep | Most stable around $\alpha=0.5$ | Balance needed; excessive structural weight biases toward pure geometry. |
| Iterations $T=50/100/200$ | $T=200$ is sufficient | Consistent with the $O(1/T)$ rate in Theorem 5.6. |
| Distance normalization | Stable with max-value normalization | Operator scale is sensitive to the HS norm. |
| Soft vs. LAP Post-proc | Soft coupling has inherent diffusion | Dispersion gap explanation: CDOT doesn't penalize dispersion, requiring LAP for hard matching. |
| Distance Choice | CDOT prefers diffusion; FGW prefers geodesic | Confirms CDOT uses aggregated profiles while FGW uses pairwise differences. |

### Key Findings

- **Convexity is genuinely effective**: CDOT systematically outperforms FGW on synthetic data. EFGW (which smooths GW to be "near-convex") performs closer to CDOT, suggesting non-convexity is the primary bottleneck for FGW.
- **Dispersion gap is not just a theoretical toy**: The performance swap in brain connectomes (diffusion vs. geodesic) corresponds to high/low variance geometries. CDOT performs better in high-variance geometries because it does not penalize dispersion.
- **Algorithmic output itself is consistent**: CDOT consistently wins on real data (brain connectomes, graph classification) with generally lower variance (std 0.04 on IMDB-B vs. FGW 0.07), suggesting FW on a convex landscape is far more stable than FGW in local optima traps.

## Highlights & Insights

- **"Operatorized Geometry" is the key to convexifying GW**: Lifting distance matrices and couplings to operators turns "structural alignment" into an operator commutation error, eliminating the bilinear $\pi\otimes\pi$. This paradigm of "lifting to operator space and using spectral theory" has precedents in kernel mean embedding and HSIC, but applying it to convexify GW is novel.
- **The dispersion gap provides a precise decomposition of non-convexity**: Formulating $\mathcal{R}_{\mathrm{GW}} = \mathcal{R}_{\mathrm{CDOT}} + \mathcal{V}$ (Convex + Concave) allows for a geometric interpretation of GW's local optima traps. This analysis can be extended to other $\pi\otimes\pi$ objectives like sliced or low-rank GW.
- **Glued measure is a standard bridge for consistency proofs**: Using $\Phi_n$ to lift $\hat\pi$ back to population space and bounding statistical error via $W_1$ is a reusable trick for providing guarantees on algorithmic outputs in other OT contexts.

## Limitations & Future Work

- The authors acknowledge: (i) Discriminability—$d_{\mathrm{CT}}^{(\alpha)} = 0$ does not imply mm space identity, only a "fractional structural equivalence." Characterizing the zero set is future work; (ii) The statistical rate in $\mathbb{R}^d$ is $O(n^{-1/d})$, and no polynomial rate exists in the infinite-dimensional setting without further structural assumptions.
- Self-identified limitations: (i) The resulting couplings are highly diffuse, requiring Hungarian matching for almost all hard-matching tasks; the approximation gap between soft and hard remains unquantified; (ii) Per-step $\mathcal{O}(n^3)$ complexity remains a bottleneck for graphs with thousands of nodes; lazy FW offers only constant speedup; (iii) Assumption 5.5 (Lipschitz optimal coupling conditionals) is quite strong and may require case-by-case verification in discrete graph tasks.
- Improvement ideas: Use the dispersion decomposition as a "diagnostic" to design "semi-convex CDOT" with controllable concave terms; combine with entropic or sliced methods for $\mathcal{O}(n^2)$ approximations; extend operator frameworks to non-Euclidean metrics (tree metrics, Wasserstein-on-Wasserstein).

## Related Work & Insights

- **vs. FGW (Vayer et al., 2020)**: FGW uses $\mathbb{E}_{\pi\otimes\pi}|d_\mathcal{X}-d_\mathcal{Y}|^2$ for pairwise alignment (non-convex). CDOT uses operator commutation error (convex). Both are $\mathcal{O}(n^3)$, but CDOT adds pseudo-metric + consistency + finite-sample bounds.
- **vs. Entropic GW (Peyré et al., 2016)**: Entropic regularization smooths GW to be "near-convex" but breaks the pseudo-metric property and depends on $\varepsilon$ tuning. CDOT is convex in its **original, unregularized** form.
- **vs. GW-SDP (Chen et al., 2024)**: Achieves convexity via SDP relaxation at a cost of $\mathcal{O}(n^6)$ complexity and loss of the pseudo-metric. CDOT maintains $\mathcal{O}(n^3)$ and the metric property.
- **vs. Sliced GW / FGW**: Slicing avoids non-convexity via 1D projections but either does not output explicit couplings or sacrifices the metric property. CDOT retains the "explicit transport plan + metricity + convexity" triad.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First truly convex geometry-aware OT objective; the dispersion gap provides an insightful decomposition of GW's non-convexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic, brain connectome, and 5 graph datasets. Lacks larger-scale graphs (>1k nodes) and high-frequency applications like 3D shapes.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure; Table 1 provides high information density; consistent logical flow from convexity to risk bounds.
- Value: ⭐⭐⭐⭐ Fills a critical gap in the GW family regarding "convexity + consistency + sample bounds." Highly significant theoretical value with direct practical applicability to graph and connectome analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AvAtar: Learning to Align via Active Optimal Transport](avatar_learning_to_align_via_active_optimal_transport.md)
- [\[ECCV 2024\] Differentiable Convex Polyhedra Optimization from Multi-view Images](../../ECCV2024/3d_vision/differentiable_convex_polyhedra_optimization_from_multi-view_images.md)
- [\[ICML 2026\] Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)
- [\[CVPR 2025\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](../../CVPR2025/3d_vision/3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)
- [\[ICML 2026\] Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning](geometry-guided_modeling_of_foundation_features_enables_generalizable_object_sha.md)

</div>

<!-- RELATED:END -->
