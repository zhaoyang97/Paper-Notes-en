---
title: >-
  [Paper Note] Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation
description: >-
  [ICML 2026][3D Vision][Gromov–Wasserstein] This paper proposes CDOT (Convex Distance Operator Transport). By "operatorizing" the distance matrices and coupling of each metric space and replacing the non-convex squared pairwise distance difference in FGW with $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$, it achieves a framework for heterogeneous space alignment t
tags:
  - ICML 2026
  - 3D Vision
  - Gromov–Wasserstein
  - Frank–Wolfe
date: 2026-05-08
content_hash: 5a4f56a48565b631
---
# Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation

**Conference**: ICML 2026  
**arXiv**: [2606.02047](https://arxiv.org/abs/2606.02047)  
**Code**: N/A  
**Area**: Optimal Transport / Convex Optimization / Metric Measure Spaces  
**Keywords**: Optimal Transport, Gromov–Wasserstein, Distance Operator, Convex Optimization, Frank–Wolfe

## TL;DR
This paper proposes CDOT (Convex Distance Operator Transport). By "operatorizing" the distance matrices and coupling of each metric space and replacing the non-convex squared pairwise distance difference in FGW with $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$, it achieves a framework for heterogeneous space alignment that is **strictly convex with respect to the coupling $\pi$**, while remaining a valid pseudo-metric and possessing finite-sample risk bounds.

## Background & Motivation

**Background**: The de facto standard for comparing probability distributions across heterogeneous domains is Gromov–Wasserstein (GW) and its variant with node features, Fused GW (FGW). They use squared pairwise distance differences, such as $|d_\mathcal{X}(X,X') - d_\mathcal{Y}(Y,Y')|^2$, to measure structural misalignment, combined with a feature alignment term $\mathbb{E}_\pi[\|f_\mathcal{X}(X)-f_\mathcal{Y}(Y)\|^2]$. This framework is widely used in tasks requiring cross-domain comparisons, such as graph classification, brain connectome alignment, and shape matching.

**Limitations of Prior Work**: FGW/GW structural terms exhibit a **tensor product form** $\pi \otimes \pi$, which is quadratic in $\pi$ but has an indefinite Hessian, making the **objective non-convex**. Consequently: (i) Frank–Wolfe or projected gradient methods can only guarantee convergence to local stationary points; (ii) they are highly sensitive to differing node cardinalities and local geometric jitter—slight perturbations in nodes or edge weights can cause pairwise distance comparisons to fail; (iii) almost all non-convex GW variants lack statistical consistency and finite-sample bounds for the "actual coupling output by the algorithm," only proving the distance between empirical and population risks. Table 1 compares GW, FGW, Entropic GW, Sliced GW, Low-rank GW, GW-SDP, IsoRank, and COPT, showing that the column "Simultaneously possessing pseudo-metric, convexity, consistency, and finite-sample bounds on mm-spaces" is almost entirely marked with ✗.

**Key Challenge**: Retaining the **pairwise distance** comparison of GW yields elegant pseudo-metric properties but inherently locks non-convexity into the objective. Attempting to convexify (e.g., via entropy regularization, SDP relaxation, or slicing) often breaks the metric properties or loses the explicit transport plan. The authors aim to achieve three things simultaneously: **Convexity + Pseudo-metric + Explicit Coupling**.

**Goal**: (1) Provide an alignment objective that is strictly convex in $\pi$ and preserves geometric awareness; (2) prove it induces a valid pseudo-metric on attributed compact mm-spaces; (3) provide an algorithm with global convergence in polynomial time, accompanied by a finite-sample risk decomposition.

**Key Insight**: Elevate "distance" from a matrix to an operator—defining the distance operator $(D_{\mathbb{P}_X} f)(x) = \int d_\mathcal{X}(x,x') f(x') \mathbb{P}_X(dx')$ and the coupling as a conditional expectation operator $(T_\pi g)(x) = \int g(y) \pi(dy|x)$. At the operator level, "aligning structures" naturally becomes a question of whether $D_X T_\pi$ and $T_\pi D_Y$ **intertwine**. This is measured using the Hilbert–Schmidt norm difference. Since this "operator intertwining error" is a linear function of $T_\pi$, its squared HS norm is quadratic and positive semi-definite with respect to $\pi$, thus directly convex.

**Core Idea**: Use the difference between "the conditional average distance from each $x$ to $Y$ seen by $\pi$" and "the conditional average distance from each $y$ to $X$ seen by $\pi$" as the structural misalignment measure. This **upgrades GW's edge-to-edge comparison to an aggregated distance profile comparison**. This eliminates the non-convex $\pi\otimes\pi$ tensor product and makes the model more robust to node cardinality differences and local noise (e.g., in Figure 1, two cycle graphs with different node counts are structurally equivalent under CDOT but not under FGW).

## Method

### Overall Architecture

The core problem CDOT addresses is how to measure structural misalignment between two heterogeneous metric spaces without sacrificing convexity. While FGW compares pairwise distance differences $|d_\mathcal{X}-d_\mathcal{Y}|^2$, which is geometrically intuitive but results in a non-convex $\pi\otimes\pi$ objective, CDOT takes a different perspective. It elevates both "distance" and "coupling" to operators, transforming "structural alignment" into a question of operator intertwining.

Specifically, given two attributed compact mm-spaces $\mathfrak{X} = (\mathcal{X}, d_\mathcal{X}, \mathbb{P}_X, f_\mathcal{X})$ and $\mathfrak{Y} = (\mathcal{Y}, d_\mathcal{Y}, \mathbb{P}_Y, f_\mathcal{Y})$, the goal is to output a transport plan $\pi \in \Pi(\mathbb{P}_X, \mathbb{P}_Y)$. The first step involves expressing the distances as operators $D_{\mathbb{P}_X}$ and $D_{\mathbb{P}_Y}$, and representing the coupling as a conditional expectation operator $T_\pi$. The CDOT objective is then formulated as a feature alignment term plus a structural term representing the "operator intertwining error": $\mathcal{L}_\alpha(\pi) = (1-\alpha)\,\mathbb{E}_\pi[c_f(X,Y)] + \tfrac{\alpha}{2}\|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$, where $c_f(x,y)=\|f_\mathcal{X}(x)-f_\mathcal{Y}(y)\|_2^2$. For finite samples, the distance operator is discretized using a normalized distance matrix $d_\mathcal{X}(X_i,X_j)/n_X$, and the coupling is scaled as $n\pi$. The structural term becomes $\|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}}_Y}\|_F^2$, making the entire problem a convex Quadratic Program (QP) over the transportation polytope. This is solved using the projection-free Frank–Wolfe algorithm, where each step requires solving a linear minimization problem (standard LP, solvable via OT solvers). If a hard matching is required for downstream tasks, the Hungarian algorithm projects the soft coupling onto a permutation matrix $\hat P = \arg\max_{P\in\mathcal{P}_n} \mathrm{Tr}(P^\top \hat\pi)$.

### Key Designs

**1. Operator-Level Structural Regularizer: Replacing non-convex pairwise differences with convex "Operator Intertwining Error"**

The bilinear form of $\pi\otimes\pi$ in the FGW structural term is the source of non-convexity. CDOT replaces this with a regularizer $\mathcal{R}(\pi) = \|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$. A key observation is that via integral transformation, $\mathcal{R}(\pi) = \iint \Gamma_\pi(x,y)^2\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$, where $\Gamma_\pi(x,y) = \mathbb{E}_\pi[d_\mathcal{X}(x,X)\mid Y=y] - \mathbb{E}_\pi[d_\mathcal{Y}(y,Y)\mid X=x]$. In other words, CDOT compares the difference between the "conditional average distance profile of $x$ in $\mathcal{X}$" and the "conditional average distance profile of $y$ in $\mathcal{Y}$"—aggregating "pairwise distances" into "expected distances."

This modification is effective because $T_\pi$ is a linear operator relative to $\pi$, so the squared HS norm is quadratic and positive semi-definite with respect to $T_\pi$ (and thus $\pi$). Combined with the linear feature term, the total objective $\mathcal{L}_\alpha$ is strictly convex (Theorem 3.4). Meanwhile, it retains geometric meaning by characterizing the relationship between a node and its surrounding structure. Consequently, the square root of the objective $d_{\mathrm{CT}}^{(\alpha)}$ remains a valid pseudo-metric on attributed compact mm-spaces (Theorem 3.5). Aggregated expectations are also naturally robust to node cardinality differences: two spaces with similar geometry but different node counts can be identified as equivalent under CDOT (Figure 1).

**2. Dispersion Gap Decomposition: Quantifying exactly what CDOT omits compared to GW**

To explain why GW is non-convex and where CDOT gains its convexity, the paper provides a clean decomposition within the same coordinate system. Defining dispersion as $\mathcal{V}(\pi) = \iint \big(\mathrm{Var}_\pi[d_\mathcal{X}(x,X)\mid Y=y] + \mathrm{Var}_\pi[d_\mathcal{Y}(y,Y)\mid X=x]\big)\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$, it captures the "conditional variance of distances" induced by the coupling. Theorem 3.7 proves that the GW structural cost is exactly equal to the CDOT structural cost plus this dispersion:

$$\mathcal{R}_{\mathrm{GW},2}(\pi) - \mathcal{R}(\pi) = \mathcal{V}(\pi).$$

Thus, GW equals "Convex structural term + Concave dispersion penalty." This provides a geometric interpretation of non-convexity: on the $(\mathcal{R}, \mathcal{V})$ plane, GW contour lines are diagonal $x+y=c$, which intersect non-convex "dart-shaped" feasible regions at local optima. CDOT contour lines are vertical $x=c$, heading straight toward the global optimum along the horizontal axis (Figure 2). This also explains an empirical phenomenon: GW prefers nearly deterministic couplings (to minimize dispersion), while CDOT does not penalize dispersion and tends toward more diffuse soft couplings, requiring a post-hoc LAP for hard matching.

**3. Frank–Wolfe + Glued Measure: Connecting algorithm output to the global optimum**

Previous GW methods could prove empirical risk approaches population risk, but they could not guarantee that the "algorithm-returned $\hat\pi$ itself" approaches the global optimum. The convexity of CDOT allows for two-step control. First, discrete CDOT is a convex QP, and standard FW (with step size $\gamma_t=2/(t+2)$) achieves global convergence at $O(1/T)$, controlling optimization error. Second, a glued measure $\Phi_n(\hat\pi)(dx,dy) = \int Q_X(dx\mid \hat x) Q_Y(dy\mid \hat y) \hat\pi(d\hat x, d\hat y)$ (where $Q_X, Q_Y$ are optimal couplings for $\mathbb{P}_X\leftrightarrow\hat{\mathbb{P}_X}$ and $\mathbb{P}_Y\leftrightarrow\hat{\mathbb{P}_Y}$) lifts the discrete solution back to the population space, allowing statistical error to be bounded by $W_1$. Together, Theorem 5.6 provides a finite-sample bound for the algorithm output:

$$\big|\mathcal{L}_\alpha(\Phi_n(\hat\pi)) - \min \mathcal{L}_\alpha\big| \le \tfrac{32\alpha n_{\min}}{T+3} + C\,\big(W_1^{d_\mathcal{X}}(\mathbb{P}_X,\hat{\mathbb{P}}_X) + W_1^{d_\mathcal{Y}}(\mathbb{P}_Y,\hat{\mathbb{P}}_Y)\big),$$

where the first term is optimization error and the second is statistical error. Corollary 5.7 further provides almost sure risk consistency as $n_{\min}/T_n\to 0$—a guarantee previously unattainable for algorithm outputs for the GW family on mm-spaces.

### Loss & Training

The empirical objective is $\hat{\mathcal{L}}_\alpha(\pi) = (1-\alpha)\langle C_f, \pi\rangle_F + \tfrac{\alpha}{2}\, n_X n_Y \|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}}_Y}\|_F^2$. The fusion weight defaults to $\alpha=0.5$ (applicable to synthetic and real data), and iterations $T \in \{50,100,200\}$ are sufficient to reduce optimization error to negligible levels. Complexity per step is $\mathcal{O}(n^3)$ (comparable to FGW). The paper introduces a lazy gradient FW variant that utilizes the quadratic structure for incremental gradient updates, reducing constant factors; this is three orders of magnitude lower than the $\mathcal{O}(n^6)$ of GW-SDP. Applications requiring hard matching append a Hungarian algorithm layer after convergence.

## Key Experimental Results

### Main Results

Synthetic 2D clustering point clouds ($N=4n$, 100 trials, reporting MSE) comparing CDOT with FGW, Entropic FGW, IsoRank, Spectral, and COPT:

| $n$ | CDOT | FGW | EFGW | IsoRank | Spectral | COPT |
|----|------|------|------|---------|----------|------|
| 100 | **0.0077** | 0.0146 | 0.0098 | 0.0141 | 1.3447 | 0.6664 |
| 300 | **0.0027** | 0.0055 | 0.0038 | 0.0053 | 1.3276 | 0.6670 |
| 500 | **0.0016** | 0.0034 | 0.0025 | 0.0033 | 1.3373 | 0.6670 |

CDOT achieves the lowest MSE across all sample sizes, with MSE decreasing monotonically as $n$ increases—directly validating the statistical consistency of Theorem 5.6. Spectral/COPT methods, which ignore feature information, fail to learn correct alignments.

OASIS-3 brain connectome node alignment accuracy (pairwise matching of 100 subjects):

| Method | Diffusion Distance | Geodesic Distance | Topology |
|------|----------------|---------------|----------|
| CDOT | **0.6136** | 0.4640 | – |
| FGW  | 0.1853 | **0.5375** | – |
| EFGW | 0.4097 | 0.4583 | – |
| IsoRank | – | – | **0.4055** |
| Spectral / COPT | – | – | 0.0737 / 0.0253 |

CDOT leads significantly under diffusion distance (0.61 vs FGW 0.19). FGW performs slightly better on geodesic distance because it directly penalizes pairwise distance distortion, which is sensitive to single shortest-path geometric information. Diffusion distance averages paths, reducing the "pairwise difference contrast" required by GW, whereas CDOT's operator aggregation effectively utilizes this global signal.

Graph classification benchmarks (node counts 17–40):

| Dataset | CDOT | FGW | GW | COPT |
|--------|------|-----|----|----- |
| MUTAG  | **0.862** | 0.825 | 0.718 | 0.633 |
| IMDB-B | **0.642** | 0.602 | – | 0.637 |
| PROTEINS | **0.755** | 0.736 | 0.661 | 0.695 |
| NCI1   | **0.748** | 0.730 | 0.571 | 0.599 |
| ENZYMES| **0.513** | 0.445 | 0.238 | 0.235 |

CDOT outperforms others in 5/5 datasets, with improvements over FGW ranging from 1.7–6.8 percentage points, and larger gaps over GW/COPT.

### Ablation Study

| Configuration | Key Phenomenon | Description |
|------|----------|------|
| $\alpha$ sweep | Most stable near $\alpha=0.5$ | Balance required between feature and structure terms; excessive structure weight biases toward pure geometry. |
| Iterations $T=50/100/200$ | $T=200$ is sufficient | Consistent with the $O(1/T)$ rate of Theorem 5.6. |
| Distance normalization | Stable across datasets using max normalization | Operator scale is sensitive to the HS norm. |
| Soft vs. LAP post-processing | Soft coupling is inherently diffuse | Explained by the dispersion gap: CDOT doesn't penalize dispersion, so LAP is needed for hard matching. |
| Distance choice (Diffusion vs. Geodesic) | CDOT prefers diffusion, FGW prefers geodesic | Validates the difference between CDOT's aggregated distance profiles and FGW's pairwise differences. |

### Key Findings

- **Convexity is genuinely beneficial**: CDOT systematically outperforms FGW on synthetic data. EFGW (which smooths GW into "near-convexity" via entropy) performs similarly to CDOT, suggesting non-convexity is the primary bottleneck for FGW.
- **The dispersion gap is not just theoretical**: The performance difference between diffusion and geodesic distances in the brain connectome corresponds to geometries with high/low distance variance. CDOT performs better on high-variance geometries because it does not penalize dispersion.
- **Algorithm outputs exhibit inherent consistency**: CDOT consistently wins on real datasets like brain connectomes and graph classification, typically with lower variance (std 0.04 on IMDB-B vs. FGW 0.07), suggesting FW on a convex landscape is far more stable than FGW's local optima traps.

## Highlights & Insights

- **"Operatorizing Geometry" is the key to convexifying GW**: Elevating distance matrices and couplings to operators turns "structural alignment" into operator intertwining error, eliminating the bilinear $\pi\otimes\pi$. This approach of "lifting to the operator level and using linear spectral theory" has precedents in kernel mean embedding and HSIC, but applying it to convexify GW is novel, enabled by $T_\pi$ being a linear operator of $\pi$.
- **Dispersion gap is a precise decomposition of non-convexity**: Formulating $\mathcal{R}_{\mathrm{GW}} = \mathcal{R}_{\mathrm{CDOT}} + \mathcal{V}$ (Convex + Concave) and providing the $(\mathcal{R}, \mathcal{V})$ plane contour visualization (Figure 2) upgrades "GW local minima" from an empirical observation to a structural explanation. This analysis can be extended to other $\pi\otimes\pi$ objectives like sliced or low-rank GW.
- **Glued measure is a standard technique to link discrete solutions to population optima**: Lifting $\hat\pi$ back to the population space via $\Phi_n$ and bounding statistical error with $W_1$ is a trick that can be reused in other consistency proofs for algorithm outputs (e.g., entropic OT estimation, neural OT).

## Limitations & Future Work

- The authors acknowledge that: (i) Zero-discriminability is not fully achieved—$d_{\mathrm{CT}}^{(\alpha)} = 0$ does not imply mm-space identity, but rather a "fractional structural equivalence." A full characterization of the zero set is left for future work; (ii) The statistical rate on $\mathbb{R}^d$ is $O(n^{-1/d})$, and without additional structural assumptions in the infinite-dimensional setting, no polynomial rate exists.
- Personal observations: (i) The resulting couplings are highly diffuse; almost all hard-matching applications require Hungarian post-processing, yet the approximation gap between soft and hard matchings is not systematically quantified; (ii) $\mathcal{O}(n^3)$ per step remains a bottleneck for graphs with thousands of nodes; lazy FW only provides constant acceleration rather than the sub-cubic order of sliced or low-rank variants; (iii) Assumption 5.5 (Lipschitz optimal coupling conditional distribution under $W_1$) is quite strong and requires case-by-case verification in discrete graph tasks.
- Future directions: Use the dispersion decomposition as a "diagnostic tool" to design "semi-convex CDOT" with controllable concave terms to recover GW's hard-matching preference while keeping global optima; combine with entropy regularization or slicing for $\mathcal{O}(n^2)$ approximations; extend the operator framework to non-Euclidean metrics (tree metrics, Wasserstein-on-Wasserstein).

## Related Work & Insights

- **vs FGW (Vayer et al., 2020)**: FGW aligns pairwise distances using $\mathbb{E}_{\pi\otimes\pi}|d_\mathcal{X}-d_\mathcal{Y}|^2$, which is non-convex. CDOT uses operator intertwining, which is convex. Both share $\mathcal{O}(n^3)$ complexity, but CDOT adds the benefits of pseudo-metrics, consistency, and finite-sample bounds.
- **vs Entropic GW (Peyré et al., 2016)**: Entropy regularization smooths GW into something "near-convex" but breaks the original pseudo-metric properties and depends on $\varepsilon$ tuning. CDOT is inherently convex in its **original unregularized** form.
- **vs GW-SDP (Chen et al., 2024)**: Achieves convexity via SDP relaxation but at the cost of $\mathcal{O}(n^6)$ complexity and loss of pseudo-metric properties. CDOT maintains $\mathcal{O}(n^3)$ and remains a pseudo-metric.
- **vs Sliced GW / Sliced FGW**: Slicing avoids non-convexity through 1D projections but either fails to output explicit couplings (calculating only discrepancy) or sacrifices metric properties. CDOT preserves the "Explicit coupling + Metricity + Convexity" triad.
- **Insight**: The paradigm of "lifting geometric information to operators and then manipulating in the RKHS / $L^2$ operator layer" is highly suitable for graph kernels, scene graph alignment, and cross-modal OT where objectives must be both convex and geometrically aware.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first truly convex geometric-aware OT objective; the dispersion gap provides an exact decomposition of GW's non-convexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + Brain connectome + 5 graph datasets cover typical heterogeneous alignment scenarios, but lacks larger-scale graphs (>1k nodes) and high-frequency applications like 3D shapes.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure; Table 1 is information-dense; a coherent narrative from convexity to pseudo-metrics and risk bounds.
- Value: ⭐⭐⭐⭐ Fills a critical gap in the GW family regarding "Convexity + Consistency + Finite-sample bounds." Theoretically significant and practically ready to replace FGW in graph alignment and brain analysis tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Convex Decomposition via Feature Fields](../../CVPR2026/3d_vision/learning_convex_decomposition_via_feature_fields.md)
- [\[ICML 2026\] AvAtar: Learning to Align via Active Optimal Transport](avatar_learning_to_align_via_active_optimal_transport.md)
- [\[ICML 2026\] Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)
- [\[ECCV 2024\] Differentiable Convex Polyhedra Optimization from Multi-view Images](../../ECCV2024/3d_vision/differentiable_convex_polyhedra_optimization_from_multi-view_images.md)
- [\[CVPR 2025\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](../../CVPR2025/3d_vision/3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)

</div>

<!-- RELATED:END -->
