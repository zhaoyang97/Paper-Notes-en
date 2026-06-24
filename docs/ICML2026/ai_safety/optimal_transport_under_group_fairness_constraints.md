---
title: >-
  [Paper Note] Optimal Transport under Group Fairness Constraints
description: >-
  [ICML 2026 Spotlight][AI Safety][Group Fairness] This paper explicitly encodes "group fairness" as a $K_s \times K_w$ inter-group matching probability target $\mathbf{F}$. It proposes three solutions: **FairSinkhorn** for exact solving, **Penalized OT** for convex relaxation, and **Bi-level Cost Learning**. It provides finite sample complexity $O(1/\sqrt{n})$ and fairness bias bounds $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$, outlining the "cost-fairness" trade-off frontier o…
tags:
  - "ICML 2026 Spotlight"
  - "AI Safety"
  - "Group Fairness"
  - "Optimal Transport"
  - "Sinkhorn"
  - "Bi-level Optimization"
  - "Cost Learning"
date: 2026-05-08
content_hash: 8c4a8613d2b0f80d
---

# Optimal Transport under Group Fairness Constraints

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2601.07144](https://arxiv.org/abs/2601.07144)  
**Code**: https://github.com/LinusBleistein/fair_ot (Available)  
**Area**: AI Safety / Algorithmic Fairness / Optimal Transport  
**Keywords**: Group Fairness, Optimal Transport, Sinkhorn, Bi-level Optimization, Cost Learning  

## TL;DR
This paper explicitly encodes "group fairness" as a $K_s \times K_w$ inter-group matching probability target $\mathbf{F}$. It proposes three solutions: **FairSinkhorn** for exact solving, **Penalized OT** for convex relaxation, and **Bi-level Cost Learning**. It provides finite sample complexity $O(1/\sqrt{n})$ and fairness bias bounds $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$, outlining the "cost-fairness" trade-off frontier on synthetic and semi-synthetic (dating app) datasets.

## Background & Motivation

**Background**: Algorithmic matching (admissions, hiring, dating, kidney transplants, urban resource allocation) is increasingly determined by centralized algorithms. Optimal Transport (OT) has become a mainstream tool due to its excellent modeling properties in economics and social sciences. Given two distributions $\mu, \eta$ and a cost $c$, entropy-regularized OT solves $\min_\pi \int c(x,y)\,d\pi + \varepsilon \mathbf{KL}(\pi|\mu\otimes\eta)$, efficiently iterated via the Sinkhorn algorithm.

**Limitations of Prior Work**: When features $X, Y$ are strongly correlated with sensitive attributes $S, W$ (gender, race, socioeconomic status), standard OT produces "block-diagonal" highly homogeneous matches—such as assigning high-income students almost exclusively to elite schools. Existing fairness × matching literature focuses almost entirely on **individual fairness** (similar individuals get similar outcomes) or **problem-specific participation quotas** (kidney transplant KPD), lacking a "group matching probability" framework specifically for OT that can be flexibly specified by a central planner.

**Key Challenge**: (a) Most existing fair-OT works treat OT as a tool for downstream fair prediction (e.g., Wasserstein barycenter projection) rather than studying the fairness of the **transport plan itself**; (b) Exact fairness often comes at a massive cost ("price of fairness"), requiring a relaxation framework for fine-grained trade-off control; (c) Whether fairness targets "imprinted" into the matching can be **reused** on new samples remains an open question.

**Goal**: (1) Formalize the new group fairness definition of "inter-group matching probability = target $\mathbf{F}$"; (2) Provide an exact solution algorithm + two types of relaxation methods; (3) Accompany relaxation methods with finite sample theoretical guarantees.

**Key Insight**: It is observed that the fairness constraint $\pi_{SW}(s,w) = \mathbf{F}_{sw}$ is **linear** over the transport plan $\mathbf{\Pi}$. Therefore, the Lagrange dual can still be written in a Sinkhorn-style multiplicative form. Meanwhile, entropy regularization ensures uniqueness and differentiability, making the bi-level optimization of "inducing fairness by learning costs" well-posed.

**Core Idea**: Use an "inter-group probability target matrix $\mathbf{F}$" to uniformly express real-world rules like limiting homophily, minimum minority quotas, the 4/5 hiring rule, and the EU's 40% gender quota for boards. Provide an algorithmic spectrum from exact to relaxed.

## Method

### Overall Architecture

Standard entropy-regularized OT produces "block-diagonal" matches when sensitive attributes and features are strongly correlated, locking disadvantaged groups into their original circles. This paper enables planners to directly specify "what percentage of group A should match with resource group B" using an inter-group matching probability matrix $\mathbf{F}$, translating it into linear constraints on the transport plan. The inputs are two samples with sensitive attributes $(\mathbf{x}_i, \mathbf{s}_i)_{i=1}^n \sim \mu$, $(\mathbf{y}_j, \mathbf{w}_j)_{j=1}^m \sim \eta$, a cost matrix $\mathbf{C}_{ij} = c(\mathbf{x}_i, \mathbf{y}_j)$, entropy regularization $\varepsilon$, and a fairness target $\mathbf{F} \in \Pi(\mathbf{p}, \mathbf{q})$. The output is a transport plan $\mathbf{\Pi} \in \mathbb{R}_+^{n\times m}$ whose row and column sums satisfy marginal constraints, while inter-group mass $\sum_{i: s_i=s, j: w_j=w} \mathbf{\Pi}_{ij}$ stays close to $\mathbf{F}_{sw}$. To achieve this, the authors trace three complementary paths from "hard-constrained exact solutions" to "reusable indirect geometric reshaping."

### Key Designs

**1. FairSinkhorn: Grafting Exact Fairness into the Sinkhorn Multiplicative Structure**

The first path treats fairness as a hard constraint for exact solutions, addressing the need for planners who require a scheme meeting $\mathbf{F}$ perfectly. The constraint is written as $\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] = \mathbf{F}_{sw}$, where $\mathbf{B}_{sw}$ is a 0/1 matrix marking sample group pairs $(s,w)$. A key property of this constraint is its **linearity**. Thus, after introducing Lagrange dual variables $\mathbf{h} \in \mathbb{R}^{K_s \times K_w}$, the optimal solution retains the Sinkhorn multiplicative form $\mathbf{\Pi} = \text{diag}(e^{\mathbf{f}/\varepsilon})(\mathbf{K} \odot \mathbf{H}) \text{diag}(e^{\mathbf{g}/\varepsilon})$: where $\mathbf{K} = e^{-\mathbf{C}/\varepsilon-1}$ is the standard Sinkhorn kernel, and $\mathbf{H} = \sum_{sw} e^{h_{sw}/\varepsilon} \mathbf{B}_{sw}$ is a "fairness coefficient" matrix that is block-constant by $(s,w)$. The algorithm only needs to insert a group-level re-projection step $\mathbf{L}^{(t+1)} \leftarrow \mathbf{F} \oslash \Phi(\mathbf{u}^{(t+1)}, \mathbf{v}^{(t+1)})$ to update $\mathbf{H}$ alongside standard row/column normalization, where $\Phi$ aggregates group-level mass. Since this adds only one block-level normalization to fixed-point iterations, the complexity is of the same order as standard Sinkhorn, and convergence speed is nearly identical, allowing drop-in replacement in existing pipelines. It provides a "perfect fairness" baseline, though the cost is flattening group-level geometric information, which may significantly increase transport costs—a problem the next two paths address.

**2. Penalized OT: A Convex "Cost-Fairness" Knob**

Many scenarios do not require exact fairness but only "close enough" results. The second path replaces hard constraints with a squared penalty $\mathcal{L}_\mathbf{F}(\mathbf{\Pi}) = \sum_{(s,w)} (\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] - \mathbf{F}_{sw})^2$, incorporated into the objective $\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}] + \varepsilon \mathbf{KL}(\mathbf{\Pi}) + \lambda \mathcal{L}_\mathbf{F}(\mathbf{\Pi})$, where $\lambda$ is the "fairness intensity knob." Since $\mathcal{L}_\mathbf{F}$ is convex, the overall objective remains strongly convex with a unique solution. Furthermore, $\mathbf{\Pi}$ smoothly slides from Sinkhorn ($\lambda=0$) toward FairSinkhorn ($\lambda\to\infty$) along a convex curve, allowing planners to pick a point at an acceptable cost. Solving uses **Generalized Conditional Gradient**: each iteration linearizes the penalty term around the current $\mathbf{\Pi}^t$ to obtain a modified cost $\mathbf{C} + \nabla \mathcal{L}_\mathbf{F}(\mathbf{\Pi}^t)$, solves this subproblem with standard Sinkhorn as the search direction, and uses line search for a convex combination to guarantee descent. Theoretically, the authors prove $\mathbb{E}|m^\star(\mu_n, \eta_n) - m^\star(\mu, \eta)| \lesssim 1/\sqrt{n}$, which is of the same order as standard entropy-regularized OT—meaning the addition of fairness penalties **does not lose statistical efficiency**. The proof involves sandwiching the empirical optimal value $m_n^\star$ between expectations of linearized subproblems, building on toolchains from rakotomamonjy2015 / genevay2019 / rigollet2022. This path wins on convexity and interpretability, though it requires re-solving for every new batch.

**3. Bi-level Cost Learning: Learning a "Fairness-Inducing" Geometry for Reuse**

The first two paths directly carve $\mathbf{\Pi}$ and require re-computation for new data. The third path changes strategy: instead of moving $\mathbf{\Pi}$, it learns a parameterized cost $c_\theta$ such that its induced entropy-regularized OT solution $\mathbf{\Pi}_\varepsilon(c_\theta)$ naturally satisfies fairness targets. Once learned, it can be cached and reused. This is formulated as a bi-level optimization $\min_\theta \mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) + \frac{1}{\lambda} \mathscr{D}(c_\theta, c_\text{base})$, with the inner constraint $\mathbf{\Pi}_\varepsilon(c_\theta) = \arg\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}_\theta] + \varepsilon \mathbf{KL}(\mathbf{\Pi})$, where $\mathscr{D}$ ensures the learned cost does not deviate too far from a baseline cost (e.g., squared Euclidean). The inner entropy-regularized OT has a unique solution due to strong convexity, making the bi-level objective well-posed; gradients are backpropagated via iterative or implicit differentiation. Two parameterizations are used: **Mahalanobis** $c_\mathbf{M}(x,y) = (x-y)^\top \mathbf{M} (x-y)$ is interpretable, revealing which feature directions are emphasized or suppressed (the "culprit directions" of unfairness); **Neural costs** $c_\theta(x,y) = \|\phi_{\theta_1}(x) - \phi_{\theta_2}(y)\|_2^2$ are more flexible for non-linear geometric transforms. The cost is weaker generalization guarantees: the authors prove that for any $\theta$ in the parameter family, the fairness bias on new samples satisfies $\sup_\theta \mathbb{E}[|\mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) - \mathcal{L}_\mathbf{F}(\pi_\varepsilon^\star(c_\theta))|] \lesssim \exp(5R_\Theta/\varepsilon)/\sqrt{n}$, which explodes exponentially as $\varepsilon$ decreases.

### Loss & Training

Training burden increases across the three paths: FairSinkhorn requires only $T$ fixed-point iterations with no learnable parameters; Penalized OT is a convex problem monotonically controlled by $\lambda$; Cost learning uses SGD/Adam to optimize $\theta$, running an inner entropy-regularized Sinkhorn at each step. The regularization $\mathscr{D}(c_\theta, c_\text{base})$ uses $\|\mathbf{M} - \mathbf{I}\|_F^2$ for Mahalanobis and $\ell_2$ norm of weights for neural costs. $\varepsilon$ and $\lambda$ are scanned via grid search to plot the complete trade-off curve.

## Key Experimental Results

### Main Results

**Synthetic Experiments**: Gaussians (two groups of students/schools generated via GMM, privileged closer to elite) and Circles (minority groups on a radius 2 ring). Target $\mathbf{F} = \begin{bmatrix} 0.20 & 0.30 \\ 0.28 & 0.22 \end{bmatrix}$, i.e., 60% of disadvantaged students matching elite schools.

| Method | Fairness Violation $\mathcal{L}_\mathbf{F}$ | Transport Cost (Relative to Sinkhorn) | Notes |
|:---|:---|:---|:---|
| Sinkhorn (vanilla) | High (Large block-diagonal bias) | 0 (Baseline) | Completely ignores fairness |
| Sinkhorn + Larger $\varepsilon$ | Still high | Increasing | Entropy reg **cannot** approximate $\mathbf{F}$ |
| FairSinkhorn | $\approx 0$ (Exact) | Significantly increased | Perfect fairness, highest cost |
| Penalized OT (varying $\lambda$) | Smooth interpolation | Smooth interpolation | Convex curve, most flexible trade-off |
| Cost learning - Mahalanobis | Medium ($\ge 10^{-2}$ on Circles) | Medium | Linear transform insufficient for geometry |
| Cost learning - MLP | Low | Medium (Similar to Penalized on Gaussians) | Non-linear geometry, matches Penalized on Circles |

**Semi-synthetic Dating App Experiment**: Kaggle dataset subsampled to match US demographics. Sensitive attributes are 7 income levels; feasible matching is determined by sexual orientation. Target $\mathbf{F}_{sw} = \mathbb{P}(S=s_i) \mathbb{P}(W=w_j)$ (i.e., independence = completely breaking income homophily). Results match synthetic tests: trade-off curves are convex, Penalized and Cost Learning nearly overlap, with good scalability to high dimensions and multiple groups.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|:---|:---|:---|
| Adjusting only $\varepsilon$ (vanilla OT) | Fairness violation **does not converge** to $\mathbf{F}$ | Entropy reg only makes the plan "blurrier," not automatically toward $\mathbf{F}$ |
| $\lambda$ from small to large (Penalized) | Convex curve along Sinkhorn → FairSinkhorn | Verifies trade-off is controllable and continuous |
| Mahalanobis vs MLP (Cost Learning) | MLP significantly lower on Circles | Linear geometry is powerless against ring distributions |
| Train cost → reuse on test set | Inference time 1-2 orders of magnitude faster than Penalized, slight fairness loss | Learned costs are reusable; fairness remains far better than vanilla OT |

### Key Findings
- **Entropy regularization is no substitute for fairness constraints**: Even a large $\varepsilon$ will not bring vanilla OT closer to $\mathbf{F}$, confirming the necessity of explicit fairness mechanisms.
- **Penalized OT is almost "overwhelmingly" the most flexible**: It searches over all couplings $\Pi$, whereas Cost Learning is restricted to a narrower subset of "entropy-reg OT solutions + parametric cost family"; however, the price is re-computation for every batch.
- **The core value of Cost Learning is "transferability"**: Learn once, use many times, making it ideal for real-time matching systems (hiring, dating); additionally, the Mahalanobis matrix itself serves as a fairness diagnostic tool.
- **Theory-Experiment Loop**: The $O(1/\sqrt{n})$ for Penalized and $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$ for Cost Learning are "reverse-verified" in experiments—the fairness gap on new samples indeed narrows as sample size increases.

## Highlights & Insights
- **The fairness target abstraction is very general**: The 4/5 hiring rule, EU 40% female board quota, and French minimum scholarship quotas can all be expressed as an $\mathbf{F}$, plugging regulations directly into the algorithm. This is the most significant conceptual contribution.
- **Linear fairness constraints → Sinkhorn needs only one extra step**: The derivation of FairSinkhorn is exceptionally clean, maintaining one additional block-constant matrix $\mathbf{T}^{(t)}$ beyond $\mathbf{u}, \mathbf{v}$, with almost zero overhead, enabling drop-in replacement.
- **Convex penalties do not lose statistical efficiency**: Proving that the sample complexity remains $O(1/\sqrt{n})$ after adding fairness penalties is a conclusion of independent value for the entire "constrained OT" field.
- **Transferable Trick**: The bi-level "learn cost to induce plan properties" paradigm can be extended beyond fairness—e.g., replacing $\mathcal{L}_\mathbf{F}$ with sparsity, monotonicity, or domain invariance to obtain a family of "constraint-induced OT" algorithms.

## Limitations & Future Work
- **Limited to group fairness**: Individual fairness ("similar individuals get similar matches") requires a different formalization and cannot be simply covered by $\mathbf{F}$.
- **No discussion of fairness-bias trade-off**: When $\mathbf{F}$ itself is mis-specified (planner misjudges true population distribution), the robustness of the three methods is unquantified.
- **Downstream utility not modeled**: Matching students to universities is just the matching; subsequent wages/academic success are the true welfare metrics, left for future work.
- **Complexity depends on $\exp(5R_\Theta/\varepsilon)$**: The bias bound for cost learning **explodes exponentially** as $\varepsilon$ decreases, so generalization guarantees are weak in small regularization regimes; practical use requires careful tuning between fairness, cost, and $\varepsilon$.
- **Refinement ideas**: (a) Make $\mathbf{F}$ learnable (planner specifies only upper bounds/inequality constraints); (b) Introduce causal graphs to explicitly model sensitive attribute → feature correlation for counterfactual fair OT; (c) Hybridize with stable matching (Gale-Shapley) for algorithms that are both stable and group-fair.

## Related Work & Insights
- **vs Gale-Shapley fair variants (karni2021, devic2023)**: They handle iterative individual matching + individual fairness; Ours is continuous mass-splitting OT + group quotas, targeting different application domains (preference-driven markets vs central-planning resource allocation).
- **vs KPD fairness (ashlagi2014, dickerson2014, zhang2025)**: KPD fairness is essentially "participation quotas" fit for specific integer programming; Ours makes fairness a general $\mathbf{F}$ matrix, adapting to the broader continuous relaxation framework of OT.
- **vs Wasserstein-barycenter fairness (gouic2020, chzhen2020, divol2024)**: They use OT as a tool for fair **prediction** (barycenter projection → correcting model output); Ours makes the **transport plan itself** fair. These are orthogonal and can be combined.
- **vs nguyen2024 (fair Wasserstein barycenter)**: They impose approximate equality constraints on sliced-Wasserstein distance; Ours imposes explicit mass constraints between subgroups, with direct semantics corresponding more easily to regulatory rules.
- **vs Constraints in OT (courty2016, blondel2018, liu2022, korman2015)**: Ours is an application of constrained OT to the specific semantics of "group fairness" and contributes sample complexity results for penalized fair OT, filling a gap in constrained OT theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces the "inter-group matching target $\mathbf{F}$" as a brand new fairness definition in OT and systematically provides three solution paths with theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + semi-synthetic covers 2D/High-D and 2-group/7-group; trade-off curves and transferability are verified, though lacking an end-to-end case study of a real critical application (hiring, admissions).
- Writing Quality: ⭐⭐⭐⭐⭐ The storyline (school → fairness target → exact → penalized → cost learning) progresses logically; definitions, propositions, algorithms, and theorems advance in a near-perfect pace.
- Value: ⭐⭐⭐⭐⭐ Provides a mathematically clear, computationally efficient, and reusable toolbox for "algorithmic governance + quota regulation" that benefits regulators, platforms, and researchers alike.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bypassing the Transport Plan: Dynamic Reweighting for Out-of-Distribution Detection with Optimal Transport](../../CVPR2026/ai_safety/bypassing_the_transport_plan_dynamic_reweighting_for_out-of-distribution_detecti.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)
- [\[ICLR 2026\] Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](../../ICLR2026/ai_safety/optimal_transport-induced_samples_against_out-of-distribution_overconfidence.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[ICML 2026\] Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)

</div>

<!-- RELATED:END -->
