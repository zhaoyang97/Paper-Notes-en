---
title: >-
  [Paper Note] Multi-Objective Bayesian Optimization via Adaptive $\varepsilon$-Constraints Decomposition
description: >-
  [ICML 2026][Optimization][Multi-Objective Bayesian Optimization] STAGE-BO reformulates MOBO into a sequence of "$\varepsilon$-constrained single-objective Bayesian subproblems" where thresholds are adaptively selected vi…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Multi-Objective Bayesian Optimization"
  - "$\\varepsilon$-constraint method"
  - "Pareto coverage"
  - "fill distance"
  - "Thompson Sampling"
date: 2026-05-08
content_hash: 90e5f41c31fb81c2
---

# Multi-Objective Bayesian Optimization via Adaptive $\varepsilon$-Constraints Decomposition

**Conference**: ICML 2026  
**arXiv**: [2604.15959](https://arxiv.org/abs/2604.15959)  
**Code**: https://github.com/YangYaohong1/STAGE-BO  
**Area**: Bayesian Optimization / Multi-Objective Optimization  
**Keywords**: Multi-Objective Bayesian Optimization, $\varepsilon$-constraint method, Pareto coverage, fill distance, Thompson Sampling

## TL;DR
STAGE-BO reformulates MOBO into a sequence of "$\varepsilon$-constrained single-objective Bayesian subproblems" where thresholds are adaptively selected via fill distance. By solving these using cEI, the method achieves uniform Pareto front coverage without calculating hypervolume (HV), and naturally supports hard constraints and user preferences.

## Background & Motivation

**Background**: The mainstream approach for Multi-Objective Bayesian Optimization (MOBO) involves fitting a Gaussian Process (GP) for each objective and using an acquisition function to guide the next expensive black-box evaluation. Most acquisition functions are designed around hypervolume (HV) improvement, such as qEHVI, JESMO, and TSEMO.

**Limitations of Prior Work**: Relying solely on HV comes with two significant costs. First, the exact computation of HV grows exponentially with the number of objectives $m$, becoming computationally infeasible when $m \ge 4$. Second, theoretical work by Auger et al. indicates that the asymptotic point density for HV maximization is proportional to the square root of the negative slope of the Pareto front $\propto\sqrt{-F'(\mathbf{x})}$. This causes solutions to cluster in "knee" regions while sparsely covering flat areas, resulting in Inverted Generational Distance (IGD) that can be an order of magnitude worse than optimal methods.

**Key Challenge**: Existing "accelerated coverage" solutions either still rely on HV (DGEMO, PDBO, MOBO-OSD) or follow scalarization routes (ParEGO, TS-TCH). In scalarization, a uniform distribution of weights does not equate to a uniform distribution of points on the Pareto front, often leading to clusters and geometric voids. The fundamental contradiction is the lack of "explicit sampling targeting geometric gaps on the front."

**Goal**: To develop a MOBO algorithm that (i) does not requires HV calculation, (ii) provides uniform front coverage, and (iii) supports hard constraints and preferences within a unified framework.

**Key Insight**: The authors revisit a classic observation from the $\varepsilon$-constraint method: any Pareto optimal point can be recovered by "optimizing only one objective while imposing $\ge\varepsilon$ inequality constraints on the others" (Haimes, 1971). The real difficulty lies in selecting $\varepsilon$. If $\varepsilon$ is chosen specifically to "fill the largest hole on the front," the coverage problem is automatically solved.

**Core Idea**: In each iteration, Thompson Sampling is used to estimate a surrogate Pareto front $\widetilde{\mathcal{P}}_{f}^{t}$. The point $\mathbf{Y}_c$ on the surrogate front with the maximum min-distance to current observations is identified. Its coordinates are used as $\varepsilon$-constraints, and the resulting constrained subproblem is solved using cEI—HV is never computed.

## Method

### Overall Architecture
A single iteration of STAGE-BO consists of four steps, taking the existing dataset $\mathcal{D}_t=\{(\mathbf{x}_i,\mathbf{y}_i)\}$ as input and outputting the next evaluation point $\mathbf{x}_{t+1}$:

1.  Fit $m$ independent GPs using $\mathcal{D}_t$, each corresponding to an objective $f_i$.
2.  Obtain a joint sampled path $\tilde{F}^t(\mathbf{x})$ via Thompson Sampling, then use NSGA-II to find the Pareto front $\widetilde{\mathcal{P}}_{f}^{t}$ on this sampled path.
3.  Select the target point $\mathbf{Y}_c$ on the surrogate front that is "furthest from existing observations," and determine the main objective $k=t\bmod m+1$ via a round-robin strategy.
4.  Use all coordinates of $\mathbf{Y}_c$ except for the $k$-th dimension as $\varepsilon$-constraint thresholds to construct a constrained BO subproblem, optimized using cEI to obtain the next query point.

The entire process involves no HV calculation; the primary computational cost is the NSGA-II search on cheap surrogate functions.

### Key Designs

1.  **Fill-distance driven $\varepsilon$ target selection**:
    -   **Function**: Selects the target position $\mathbf{Y}_c$ in the objective space that most needs to be filled from the surrogate Pareto front.
    -   **Mechanism**: The authors adopt the fill distance metric $\text{FD}(\mathbf{Y}_t)=\max_{\mathbf{y}\in\mathcal{P}_f}\min_{\mathbf{y}'\in\mathbf{Y}_t}\|\mathbf{y}-\mathbf{y}'\|$ from Zhang et al. (2024), replacing the true Pareto front with the surrogate front obtained through Thompson Sampling. Thus, $\mathbf{Y}_c=\arg\max_{\mathbf{y}'\in\widetilde{\mathcal{P}}_f^t}\min_{\mathbf{y}\in\mathbf{Y}_t}\|\mathbf{y}-\mathbf{y}'\|$ represents the geometrically worst-covered position. A theorem in the paper proves that the solution set of optimal FD provides an upper bound for IGD ($\text{IGD}(\mathbf{Y}^{\text{FD}})\le \text{FD}(\mathbf{Y}^{\text{FD}})$), so minimizing FD directly guarantees IGD.
    -   **Design Motivation**: To replace HV-driven geometric bias and explicitly encode "uniform coverage" as the optimization objective. Thompson Sampling paths are used instead of the posterior mean to preserve GP uncertainty and avoid overly greedy behavior.

2.  **$\varepsilon$-constraint decomposition + clipping stabilizer**:
    -   **Function**: Converts the multi-objective problem into $T$ single-objective constrained subproblems, optimizing one $f_k$ at a time while other $f_j$ are fixed by thresholds $\varepsilon_j=\widehat{\mathbf{Y}}_{c,j}$.
    -   **Mechanism**: The subproblem is formulated as $\max_{\mathbf{x}\in\mathcal{X}} f_k(\mathbf{x})+s\sum_j f_j(\mathbf{x})$ s.t. $f_j(\mathbf{x})\ge \varepsilon_j$ for $j\ne k$, where $s\approx 10^{-3}$ is used to exclude weak Pareto optimal solutions. To prevent early divergence where "thresholds exceed the best observed values causing an empty feasible region," clipping is introduced: if $\mathbf{Y}_{c,j}\ge \max_t \mathbf{Y}_{t,j}$, then $\widehat{\mathbf{Y}}_{c,j}$ is reduced to the current maximum observed value. The main objective $k$ is rotated via round-robin to ensure every objective is pushed.
    -   **Design Motivation**: Classic results of $\varepsilon$-constraints guarantee that the optimal solution of the subproblem is Pareto optimal. Clipping prevents the algorithm from getting stuck during early data-sparse stages. Ablations show clipping primarily acts as a stabilizer.

3.  **Constrained EI (cEI) acquisition function + natural extrapolation to constrained / preference MOBO**:
    -   **Function**: Uses $\alpha(\mathbf{x})=\text{EI}(\mathbf{x})\times\text{PoF}(\mathbf{x})$ in each subproblem to simultaneously consider improvement and feasibility probability.
    -   **Mechanism**: EI is written as $\mathbb{E}[\max(0, f_k(\mathbf{x})+s\sum_{j\ne k} f_j(\mathbf{x})-f_k^*-s\sum_{j\ne k}f_j^*)]$, and the Probability of Feasibility $\text{PoF}(\mathbf{x})=\prod_{j\ne k}\Pr(f_j(\mathbf{x})\ge \widehat{\mathbf{Y}}_{c,j})$ is analytically computable under the independent GP assumption. When hard constraints $g_l(\mathbf{x})\ge 0$ exist, their feasibility probabilities are simply multiplied into the PoF. When a user provides a preference ROI $[a_i,b_i]$, the "lower bound" and "upper bound" are treated as two candidate constraint sets and written into the subproblem in an OR format, making the method robust to overly optimistic or conservative ROIs.
    -   **Design Motivation**: cEI is the most mature and analytically friendly acquisition function in constrained BO. This same decomposition framework naturally accommodates constraints and preferences, eliminating the engineering burden of designing separate algorithms for each setting.

### Loss & Training
STAGE-BO does not involve neural network training. Key hyperparameters are concentrated in two places: the inner NSGA-II runs with default settings on cheap GP sample paths, and the scalarization coefficient $s\approx 10^{-3}$ is used to exclude weak Pareto solutions. Query points in each round are determined solely by cEI optimization and do not depend on HV calculation.

## Key Experimental Results

### Main Results
The authors compared against 8 SOTAs on 6 unconstrained, 4 constrained, and 4 preference benchmarks, as well as a real-world hyperparameter optimization task for DP-SGD privacy-utility. Results are reported as mean + standard error over 10 independent runs. Representative trends are summarized below:

| Benchmark Type | Representative Task | Metric | STAGE-BO vs Strongest Baseline |
|---|---|---|---|
| Unconstrained (Synthetic) | ZDT1 ($d=10,m=2$) | IGD | Approx. one order of magnitude lower IGD than qEHVI; HV comparable to qEHVI |
| Unconstrained (High-dim) | DTLZ7 ($d=6,m=5$, discontinuous) | IGD / HV | IGD significantly leads; HV comparable to JESMO/MOBO-OSD; qEHVI failed to run $m\ge 4$ |
| Unconstrained (Engineering) | Water planning ($d=3,m=6$) | IGD | Stable convergence at $m=6$; HV-only methods suffer from exploding costs |
| Constrained | MW7 / Disc brake / Gear train | IGD | Consistently outperforms qEHVI, qParEGO, qPOTS, and COMBOO |
| Preference ROI | ZDT3, DTLZ2, VehicleSafety | HV & IGD | Both HV and IGD within ROI outperform TS-TCH and others |
| Real-word | DP-SGD on Dutch ($d=5,m=2$) | HV | Maintains highest HV throughout, demonstrating utility in privacy-utility trade-offs |

### Ablation Study
| Configuration | Key Observation | Description |
|---|---|---|
| Full STAGE-BO | Best IGD/HV | Complete version: Thompson Sampling + FD target + cEI |
| Posterior mean instead of TS | Significant performance drop | Posterior mean is overly greedy, suppressing uncertainty needed for exploration |
| Disable clipping | Comparable on most tasks | Clipping primarily serves as numerical stabilization |
| Change main objective strategy | Almost no impact | Round-robin is not critical; framework is insensitive to the rotation strategy |
| Different constrained BO acquisition | Still applicable | The decomposition framework does not strictly depend on cEI itself |

### Key Findings
- The source of IGD improvement is "explicit gap-finding" rather than a stronger surrogate model—this aligns with the theoretical analysis of HV bias: HV-based methods under-sample flat regions, leading to poor IGD.
- When the number of objectives $m\ge 4$, HV-based methods (particularly qEHVI) lose usability due to HV calculation overhead, whereas STAGE-BO's time complexity is nearly linear with respect to $m$.
- The "Upper/Lower bound OR constraint" design for preferences is critical: when the ROI is too aggressive (outside the true front), the lower bound acts as a safety; when the ROI is too conservative, the upper bound drives the search toward better regions.

## Highlights & Insights
- Reintroduces the $\varepsilon$-constraint method from "classic MOO textbooks" to MOBO and solves the 50-year-old problem of "how to choose $\varepsilon$" using fill distance. This is an elegant combination of a "classic theoretical idea + modern uncertainty metrics."
- Avoiding HV calculations becomes a double advantage: it bypasses the computational curse of dimensionality for HV in high dimensions and avoids its geometric bias. This "doing less to gain more" perspective is valuable for any "metric-driven" algorithm design.
- The unified framework handles "unconstrained / constrained / preference" settings simply by adding factors to the PoF or OR constraints to the subproblem, making it engineering-efficient and directly reusable in any cEI-compatible constrained BO framework.

## Limitations & Future Work
- The algorithm relies heavily on the quality of the surrogate Pareto front: if the GP hasn't learned well, $\mathbf{Y}_c$ might be poorly selected, wasting iterations on impossible regions. NSGA-II may fail in many-objective scenarios ($m>6$); the authors suggest switching to NSGA-III.
- Gap detection is based on current observation locations, making it sensitive to evaluation noise; noise-robust geometric metrics are an obvious next step.
- The scalarization coefficient $s$ in the subproblem is an under-discussed hyperparameter—too small may allow weak Pareto solutions, while too large might bias cEI toward a simple sum-of-objectives.
- The OR constraints in the preference setting make the feasible region non-convex, warranting further study on the search efficiency of NSGA-II on such geometries.

## Related Work & Insights
- **vs qEHVI / TSEMO (HV-based)**: They maximize HV improvement, which is constrained by Auger's geometric bias and suffers from computational collapse when $m\ge 4$; STAGE-BO bypasses HV entirely for uniform coverage and scalability.
- **vs ParEGO / TS-TCH (scalarization)**: Scalarization uses random weights; uniform weights do not yield uniform solutions. STAGE-BO finds gaps directly in the objective space geometry to solve clustering at the source.
- **vs DGEMO / MOBO-OSD / PDBO (Diversity Enhancement)**: These methods still use HV as the final selection signal or measure diversity in the input space; STAGE-BO measures diversity using fill distance in the output space, precisely aligning with the goal of Pareto front coverage.
- **vs qPOTS / COMBOO (Constrained MOBO)**: qPOTS measures diversity in the input space, and COMBOO uses UCB-based early stopping for feasibility; STAGE-BO seamlessly integrates hard constraints as PoF factors, offering stronger framework unification.

## Rating
- Novelty: ⭐⭐⭐⭐ While based on the classic $\varepsilon$-constraint method, the integration with fill distance and the unified framework for unconstrained/constrained/preference settings is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 14 benchmarks + DP-SGD real task + multiple ablations, covering $m$ from 2 to 6.
- Writing Quality: ⭐⭐⭐⭐ Theorem 4.2 provides the FD/IGD relationship as a theoretical anchor; sections for method, constraints, and preferences are clearly structured.
- Value: ⭐⭐⭐⭐ Highly practical for engineering optimization (materials, robotics, private ML tuning); open-source code lowers the barrier to entry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization](accelerated_multiple_wasserstein_gradient_flows_for_multi-objective_distribution.md)
- [\[NeurIPS 2025\] MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions](../../NeurIPS2025/optimization/mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](cost-aware_stopping_for_bayesian_optimization.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](../../AAAI2026/optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)

</div>

<!-- RELATED:END -->
