---
title: >-
  [Paper Note] Optimal Transport under Group Fairness Constraints
description: >-
  [ICML 2026][AI Safety][Sinkhorn] This paper explicitly encodes "group fairness" as a $K_s \times K_w$ inter-group matching probability target $\mathbf{F}$. It proposes three solutions: **FairSinkhorn** for exact solving, **Penalized OT** for convex relaxation, and **Bilevel Cost Learning**. The study provides finite sample complexity $O(1/\sqrt{n})$ a
tags:
  - ICML 2026
  - AI Safety
  - Sinkhorn
date: 2026-05-08
content_hash: 2e6bd6d444d81414
---
# Optimal Transport under Group Fairness Constraints

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2601.07144](https://arxiv.org/abs/2601.07144)  
**Code**: https://github.com/LinusBleistein/fair_ot (Available)  
**Area**: AI Safety / Algorithmic Fairness / Optimal Transport  
**Keywords**: Group Fairness, Optimal Transport, Sinkhorn, Bilevel Optimization, Cost Learning

## TL;DR
This paper explicitly encodes "group fairness" as a $K_s \times K_w$ inter-group matching probability target $\mathbf{F}$. It proposes three solutions: **FairSinkhorn** for exact solving, **Penalized OT** for convex relaxation, and **Bilevel Cost Learning**. The study provides finite sample complexity $O(1/\sqrt{n})$ and fairness bias bounds $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$, mapping the "cost-fairness" trade-off frontier on synthetic and semi-synthetic (dating app) datasets.

## Background & Motivation

**Background**: Algorithmic matching (admissions, hiring, dating, kidney transplants, urban resource allocation) is increasingly determined by centralized algorithms. Optimal Transport (OT) has become a mainstream tool due to its desirable modeling properties in economics and social sciences. Given two distributions $\mu, \eta$ and a cost $c$, entropy-regularized OT solves $\min_\pi \int c(x,y)\,d\pi + \varepsilon \mathbf{KL}(\pi|\mu\otimes\eta)$ and iterates efficiently using the Sinkhorn algorithm.

**Limitations of Prior Work**: When features $X, Y$ are strongly correlated with sensitive attributes $S, W$ (gender, race, socioeconomic status), standard OT produces "block-diagonal" highly homogenous matchings—for example, assigning high-income students almost exclusively to elite schools. Existing fairness $\times$ matching literature focuses almost entirely on **individual fairness** (similar individuals get similar outcomes) or **problem-specific participation quotas** (kidney transplants KPD), lacking a "group matching probability" framework for OT that can be flexibly specified by a central planner.

**Key Challenge**: (a) Most existing fair-OT work treats OT as a tool for downstream fair prediction (e.g., Wasserstein barycenter projection) rather than studying the fairness of the **transport plan itself**; (b) Exact fairness often comes at a high "price of fairness," requiring a relaxation framework for fine-grained trade-off control; (c) Whether fairness objectives can be **reused** on new samples after being "imprinted" into the matching remains an open question.

**Goal**: (1) Formalize the new group fairness definition of "inter-group matching probability = target $\mathbf{F}$"; (2) Provide an exact solution algorithm and two types of relaxation methods; (3) Accompany the relaxation methods with finite sample theoretical guarantees.

**Key Insight**: It is observed that the fairness constraint $\pi_{SW}(s,w) = \mathbf{F}_{sw}$ is **linear** over the transport plan $\mathbf{\Pi}$. Therefore, the Lagrange dual still maintains a Sinkhorn-style multiplicative form. Meanwhile, entropy regularization ensures uniqueness and differentiability, making the bilevel optimization of "inducing fairness by learning costs" a well-posed problem.

**Core Idea**: Use an "inter-group probability target matrix $\mathbf{F}$" to uniformly express real-world rules such as limiting homologous marriage, minimum minority quotas, the 4/5 hiring rule, or the EU's 40% gender quota for boards, providing an algorithmic spectrum from exact to relaxed solutions.

## Method

### Overall Architecture

Standard entropy-regularized OT produces "block-diagonal" matches when sensitive attributes are strongly correlated with features, locking marginalized groups into their original circles. This work enables planners to use an inter-group matching probability matrix $\mathbf{F}$ to directly specify "what percentage of group A should be matched with resource B," translating this into linear constraints on the transport plan. The inputs are two samples with sensitive attributes $(\mathbf{x}_i, \mathbf{s}_i)_{i=1}^n \sim \mu$, $(\mathbf{y}_j, \mathbf{w}_j)_{j=1}^m \sim \eta$, a cost matrix $\mathbf{C}_{ij} = c(\mathbf{x}_i, \mathbf{y}_j)$, entropy regularization $\varepsilon$, and a fairness target $\mathbf{F} \in \Pi(\mathbf{p}, \mathbf{q})$. The output is a transport plan $\mathbf{\Pi} \in \mathbb{R}_+^{n\times m}$ whose row and column sums satisfy marginal constraints while ensuring inter-group mass $\sum_{i: s_i=s, j: w_j=w} \mathbf{\Pi}_{ij}$ stays as close as possible to $\mathbf{F}_{sw}$.

### Key Designs

**1. FairSinkhorn: Grafting Exact Fairness into Sinkhorn's Multiplicative Structure**

The first path targets scenarios where planners require a solution that satisfies $\mathbf{F}$ exactly. The constraint is written as $\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] = \mathbf{F}_{sw}$, where $\mathbf{B}_{sw}$ is a 0/1 matrix marking sample group pairs $(s,w)$. A key property of this constraint is its **linearity**. By introducing Lagrange dual variables $\mathbf{h} \in \mathbb{R}^{K_s \times K_w}$, the optimal solution maintains the multiplicative form $\mathbf{\Pi} = \text{diag}(e^{\mathbf{f}/\varepsilon})(\mathbf{K} \odot \mathbf{H}) \text{diag}(e^{\mathbf{g}/\varepsilon})$, where $\mathbf{K} = e^{-\mathbf{C}/\varepsilon-1}$ is the standard Sinkhorn kernel and $\mathbf{H} = \sum_{sw} e^{h_{sw}/\varepsilon} \mathbf{B}_{sw}$ is a "fairness coefficient" matrix that is block-constant per $(s,w)$. The algorithm simply inserts a group-level re-projection step $\mathbf{L}^{(t+1)} \leftarrow \mathbf{F} \oslash \Phi(\mathbf{u}^{(t+1)}, \mathbf{v}^{(t+1)})$ to update $\mathbf{H}$ between standard row/column normalizations $\mathbf{u}, \mathbf{v}$, where $\Phi$ aggregates the group-level mass of the current transport. The complexity is of the same order as standard Sinkhorn. It provides a "perfect fairness" baseline, though at the cost of erasing inter-group geometric information, which the subsequent relaxation paths address.

**2. Penalized OT: A Convex "Cost-Fairness" Knob**

In many scenarios, "close enough" fairness suffices. The second path replaces the hard constraint with a squared penalty $\mathcal{L}_\mathbf{F}(\mathbf{\Pi}) = \sum_{(s,w)} (\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] - \mathbf{F}_{sw})^2$, yielding the objective $\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}] + \varepsilon \mathbf{KL}(\mathbf{\Pi}) + \lambda \mathcal{L}_\mathbf{F}(\mathbf{\Pi})$. Here, $\lambda$ acts as the "fairness intensity knob." Since $\mathcal{L}_\mathbf{F}$ is convex, the overall problem remains strongly convex with a unique solution. The plan $\mathbf{\Pi}$ slides smoothly from Sinkhorn ($\lambda=0$) toward FairSinkhorn ($\lambda\to\infty$) along a convex curve. Optimization is performed via **generalized conditional gradient**: each iteration linearizes the penalty around the current $\mathbf{\Pi}^t$ to obtain a modified cost $\mathbf{C} + \nabla \mathcal{L}_\mathbf{F}(\mathbf{\Pi}^t)$, solves the subproblem with standard Sinkhorn, and uses line search for convex combinations. The authors prove $\mathbb{E}|m^\star(\mu_n, \eta_n) - m^\star(\mu, \eta)| \lesssim 1/\sqrt{n}$, which is the same order as standard entropy-regularized OT, meaning the fairness penalty **does not lose statistical efficiency**.

**3. Bilevel Cost Learning: Learning a Reusable "Fairness-Inducing" Geometry**

The third path avoids modifying $\mathbf{\Pi}$ directly. Instead, it learns a parameterized cost $c_\theta$ such that its induced entropy-regularized OT solution $\mathbf{\Pi}_\varepsilon(c_\theta)$ naturally satisfies the fairness target. This is formulated as a bilevel optimization $\min_\theta \mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) + \frac{1}{\lambda} \mathscr{D}(c_\theta, c_\text{base})$, with the inner layer $\mathbf{\Pi}_\varepsilon(c_\theta) = \arg\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}_\theta] + \varepsilon \mathbf{KL}(\mathbf{\Pi})$. Gradients are backpropagated through iterative or implicit differentiation. Two parameterizations are used: **Mahalanobis** $c_\mathbf{M}(x,y) = (x-y)^\top \mathbf{M} (x-y)$, which is interpretable and reveals which feature directions are emphasized or suppressed; and **Neural Cost** $c_\theta(x,y) = \|\phi_{\theta_1}(x) - \phi_{\theta_2}(y)\|_2^2$, which handles non-linear geometric transformations. The theoretical generalization bound is $\sup_\theta \mathbb{E}[|\mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) - \mathcal{L}_\mathbf{F}(\pi_\varepsilon^\star(c_\theta))|] \lesssim \exp(5R_\Theta/\varepsilon)/\sqrt{n}$, which explodes exponentially as $\varepsilon$ decreases.

### Loss & Training

Training burdens increase across the three paths: FairSinkhorn requires only $T$ fixed-point updates with no learnable parameters; Penalized OT is a convex problem controlled monotonically by $\lambda$; Cost Learning uses SGD/Adam to optimize $\theta$, running an inner entropy-regularized Sinkhorn at each step. The regularization $\mathscr{D}(c_\theta, c_\text{base})$ uses $\|\mathbf{M} - \mathbf{I}\|_F^2$ for Mahalanobis and $\ell_2$ norm of network weights for neural costs.

## Key Experimental Results

### Main Results

**Synthetic Experiments**: Gaussians (two groups of students/schools generated via GMM, where the privileged are closer to elite schools) and Circles (minority group on a ring of radius 2). Target $\mathbf{F} = \begin{bmatrix} 0.20 & 0.30 \\ 0.28 & 0.22 \end{bmatrix}$.

| Method | Fairness Violation $\mathcal{L}_\mathbf{F}$ | Transport Cost (Relative to Sinkhorn) | Remarks |
| :--- | :--- | :--- | :--- |
| Sinkhorn (vanilla) | High (Large block-diagonal bias) | 0 (Baseline) | Ignores fairness |
| Sinkhorn + Large $\varepsilon$ | Still High | Increased | Entropy alone **cannot** approach $\mathbf{F}$ |
| FairSinkhorn | $\approx 0$ (Exact) | Significantly higher | Perfect fairness, highest cost |
| Penalized OT (varying $\lambda$) | Smooth interpolation | Smooth interpolation | Convex curve, most flexible trade-off |
| Cost learning - Mahalanobis | Medium ($\ge 10^{-2}$ on Circles) | Medium | Limited by linear geometry |
| Cost learning - MLP | Low | Medium (Matching Penalized on Gaussians) | Non-linear geometry needed for Circles |

**Semi-synthetic Dating App Experiment**: Subsampled from a Kaggle dataset to match US demographics, with income as the sensitive attribute (7 tiers) and orientation determining feasible matching. Target $\mathbf{F}_{sw} = \mathbb{P}(S=s_i) \mathbb{P}(W=w_j)$ (Independency = breaking income homogamy). Results show Penalized and Cost Learning curves are nearly identical, with good scalability to high dimensions and multiple groups.

### Ablation Study

| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Tuning $\varepsilon$ only (vanilla) | Violation **does not converge** to $\mathbf{F}$ | Entropy makes plans "blurrier" but doesn't target $\mathbf{F}$ |
| Increasing $\lambda$ (Penalized) | Follows Sinkhorn → FairSinkhorn curve | Trade-off is controllable and continuous |
| Mahalanobis vs MLP | MLP significantly lower on Circles | Linear geometry cannot handle ring distributions |
| Cost reuse on test set | Inference 1-2 orders faster than Penalized | Learned cost is reusable; gap is MLP > Mahalanobis |

### Key Findings
- **Entropy is not a substitute for fairness**: Larger $\varepsilon$ does not automatically steer vanilla OT toward $\mathbf{F}$, necessitating explicit fairness mechanisms.
- **Penalized OT is most flexible**: It searches over all couplings $\Pi$, whereas Cost Learning is restricted to the subset of entropy-regularized OT solutions under a parameter family.
- **Value of Cost Learning is transferability**: Learned once and reused, it is ideal for real-time systems. Mahalanobis matrices also serve as fairness diagnostic tools.
- **Theory-Experiment alignment**: The $O(1/\sqrt{n})$ and $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$ complexity results are empirically validated by the narrowing fairness gap as sample sizes increase.

## Highlights & Insights
- **Generality of Fairness Target**: Real-world regulations (4/5 rule, EU board quotas) can be mathematically plugged in via $\mathbf{F}$.
- **Linearity Simplifies Exact Solution**: FairSinkhorn's derivation is clean, requiring only one extra block-normalization step with almost zero overhead.
- **Efficiency of Convex Penalty**: The proof that fairness penalties do not degrade statistical complexity to $O(1/\sqrt{n})$ is a valuable contribution to constrained OT.
- **Transferable Paradigm**: The bilevel "learn cost to induce plan properties" framework can extend beyond fairness to sparsity, monotonicity, or domain invariance.

## Limitations & Future Work
- **Group Fairness Restricted**: Individual fairness requires different formalizations and cannot use $\mathbf{F}$ directly.
- **Fairness-bias Trade-off**: The robustness of these methods when $\mathbf{F}$ is mis-specified (planner's error in population distribution) remains unquantified.
- **Downstream Utility**: Matching is the proxy; future work should model long-term welfare indices like salary or academic achievement.
- **Exponential Complexity**: Generalization bounds for cost learning explode as $\varepsilon \to 0$, necessitating careful tuning between fairness, cost, and regularization.

## Related Work & Insights
- **vs Gale-Shapley variants**: These handle iterative individual matching; Ours handles continuous mass group quotas in centralized resource allocation.
- **vs KPD (Kidney) fairness**: KPD quotas are often integer-specific; Ours provides a general matrix $\mathbf{F}$ for the broader continuous OT framework.
- **vs Wasserstein-barycenter fairness**: Those works use OT for fair **prediction** (correcting outputs); Ours ensures the **transport plan itself** is fair. The directions are orthogonal and combinable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bypassing the Transport Plan: Dynamic Reweighting for Out-of-Distribution Detection with Optimal Transport](../../CVPR2026/ai_safety/bypassing_the_transport_plan_dynamic_reweighting_for_out-of-distribution_detecti.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)
- [\[CVPR 2026\] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](../../CVPR2026/ai_safety/subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)
- [\[ICML 2026\] Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)
- [\[AAAI 2026\] Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](../../AAAI2026/ai_safety/truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)

</div>

<!-- RELATED:END -->
