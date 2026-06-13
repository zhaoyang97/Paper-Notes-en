---
title: >-
  [Paper Note] Provably Data-Driven Projection Method for Quadratic Programming
description: >-
  [AAAI 2026][quadratic programming] This work extends data-driven projection matrix learning from linear programming (LP) to convex quadratic programming (QP). By proposing an "unrolled active set method" to model the com…
tags:
  - "AAAI 2026"
  - "quadratic programming"
  - "projection method"
  - "generalization guarantee"
  - "pseudo-dimension"
  - "data-driven optimization"
date: 2026-05-08
content_hash: c8d0aea8e3f440b1
---

# Provably Data-Driven Projection Method for Quadratic Programming

**Conference**: AAAI 2026
**arXiv**: [2509.04524](https://arxiv.org/abs/2509.04524)  
**Code**: None  
**Area**: Other
**Keywords**: quadratic programming, projection method, generalization guarantee, pseudo-dimension, data-driven optimization

## TL;DR

This work extends data-driven projection matrix learning from linear programming (LP) to convex quadratic programming (QP). By proposing an "unrolled active set method" to model the computation of QP optimal values within the Goldberg–Jerrum (GJ) framework, it establishes a pseudo-dimension upper bound and generalization guarantees for projection matrix learning.

## Background & Motivation

LP and QP are the most fundamental forms of convex optimization, with broad applications in industry and science. In practice, LP/QP instances often involve millions of variables and constraints, making direct solving extremely time-consuming. **Dimensionality-reduction projection methods** are solver-agnostic acceleration strategies: a projection matrix $\boldsymbol{P} \in \mathbb{R}^{n \times k}$ (with $k \ll n$) maps the original high-dimensional problem to a lower-dimensional space, where it is solved quickly before mapping the solution back to the original space.

Traditional **random projection** methods ignore the geometric properties of problem instances, potentially leading to poor solution quality after projection. Sakaue et al. (2024) proposed a **data-driven** approach: assuming that multiple LP instances are sampled from an application-specific distribution $\mathcal{D}$, one can **learn an optimal projection matrix** from observed instances such that the objective value of the projected LP closely approximates that of the original LP. They also provide generalization guarantees from a learning-theoretic perspective.

The **core motivation** of this paper is to extend the above framework from LP to general convex QP. Although the formal extension appears straightforward—replacing the linear objective with a quadratic one—the geometry of QP optimal solutions fundamentally differs from LP:

**LP optimal solutions lie at vertices of the feasible polytope**, and the computation of optimal values can be characterized by enumerating vertices;

**QP optimal solutions can lie anywhere in the feasible region**, not restricted to vertices, making direct localization and computation of optimal objective values highly challenging;

3. When the quadratic matrix $\boldsymbol{Q}$ is singular (only positive semidefinite), there may exist infinitely many optimal solutions.

Extending to QP therefore requires developing new analytical tools to address these structural differences.

## Method

### Overall Architecture

Given a QP instance $\boldsymbol{\pi} = (\boldsymbol{Q}, \boldsymbol{c}, \boldsymbol{A}, \boldsymbol{b})$, the original formulation is:

$$\text{OPT}(\boldsymbol{\pi}) = \min_{\boldsymbol{x} \in \mathbb{R}^n} \left\{ \frac{1}{2} \boldsymbol{x}^\top \boldsymbol{Q} \boldsymbol{x} + \boldsymbol{c}^\top \boldsymbol{x} \mid \boldsymbol{A}\boldsymbol{x} \leq \boldsymbol{b} \right\}$$

Via projection matrix $\boldsymbol{P} \in \mathbb{R}^{n \times k}$, substituting $\boldsymbol{x} = \boldsymbol{P}\boldsymbol{y}$ yields the projected QP (PQP):

$$\ell(\boldsymbol{P}, \boldsymbol{\pi}) = \min_{\boldsymbol{y} \in \mathbb{R}^k} \left\{ \frac{1}{2} \boldsymbol{y}^\top \boldsymbol{P}^\top \boldsymbol{Q} \boldsymbol{P} \boldsymbol{y} + \boldsymbol{c}^\top \boldsymbol{P}\boldsymbol{y} \mid \boldsymbol{A}\boldsymbol{P}\boldsymbol{y} \leq \boldsymbol{b} \right\}$$

The goal is to learn an optimal projection matrix $\hat{\boldsymbol{P}}_S$ via empirical risk minimization (ERM) from $N$ observed instances $S = \{\boldsymbol{\pi}_1, \dots, \boldsymbol{\pi}_N\}$, with theoretical generalization guarantees on new instances. The analytical approach is: **bound the pseudo-dimension of the loss function class $\mathcal{L}$, then derive generalization bounds via Pollard's theorem.**

The technical approach proceeds in four steps:

1. Construct a perturbed objective $\ell_{\boldsymbol{P},\gamma}$ to make the problem strictly convex;
2. Apply Carathéodory's theorem to localize the optimal solution of the perturbed PQP;
3. Design the "Unrolled Active Set Method" to exactly compute the optimal value;
4. Extend the pseudo-dimension upper bound from the perturbed function class to the original one.

### Key Designs

1. **Tikhonov Regularization Perturbation (Lemma 5.1, Proposition 5.2)**

   Replacing $\boldsymbol{Q}$ with $\boldsymbol{Q}_\gamma = \boldsymbol{Q} + \gamma \boldsymbol{I}_n$ yields a perturbed QP. This perturbation has two key properties:
   - The perturbed objective is **strictly convex**, ensuring the PQP has a **unique optimal solution**;
   - The objective value deviation is bounded: $0 \leq \ell(\boldsymbol{P}, \boldsymbol{\pi}_\gamma) - \ell(\boldsymbol{P}, \boldsymbol{\pi}) \leq \frac{\gamma R^2}{2}$, where $R$ is an upper bound on the feasible region's radius.

   This implies that as $\gamma \to 0^+$, the perturbed function class can **approximate the original function class to arbitrary precision**.

2. **Solution Localization via Carathéodory's Theorem (Lemma 5.3)**

   The optimal solution $\boldsymbol{y}^*$ of the perturbed PQP satisfies KKT conditions. Using complementary slackness and stationarity conditions in the KKT system, the gradient can be expressed as a conic combination of active constraint normals. The key insight is to apply the **conic Carathéodory theorem**: one can find a subset $\mathcal{B} \subseteq \mathcal{A}(\boldsymbol{y}^*)$ such that the rows of $\boldsymbol{A}_\mathcal{B}$ are linearly independent, and $\boldsymbol{y}^*$ is the unique solution to the following equality-constrained QP:

   $\min_{\boldsymbol{y}} \left\{ \frac{1}{2} \boldsymbol{y}^\top \tilde{\boldsymbol{Q}} \boldsymbol{y} + \tilde{\boldsymbol{c}}^\top \boldsymbol{y} \mid \tilde{\boldsymbol{A}}_\mathcal{B} \boldsymbol{y} = \boldsymbol{b}_\mathcal{B} \right\}$

   This allows $\boldsymbol{y}^*$ to be computed directly via the inverse of the KKT matrix.

3. **Unrolled Active Set Method (Algorithm 1, Lemmas 5.4–5.5)**

   This is the paper's core algorithmic contribution. The algorithm enumerates all possible active sets $\mathcal{A} \subset \{1, \dots, m\}$ (with $|\mathcal{A}| \leq \min\{m, k\}$), and for each candidate subset:
   - Constructs the KKT matrix $\boldsymbol{K}$ and checks invertibility;
   - If invertible, computes the candidate solution $(\boldsymbol{y}_\mathrm{cand}, \boldsymbol{\lambda}_\mathrm{cand})$ via $\boldsymbol{K}^{-1}$;
   - Verifies primal feasibility ($\tilde{\boldsymbol{A}}_j^\top \boldsymbol{y}_\mathrm{cand} \leq \boldsymbol{b}_j$) and dual feasibility ($\boldsymbol{\lambda}_{\mathrm{cand},j} \geq 0$);
   - If all conditions are satisfied, outputs the optimal value.

   This algorithm is proven to be a **GJ algorithm** with degree $\mathcal{O}(m+k)$ and predicate complexity $\mathcal{O}(m \min(2^m, (em/k)^k))$.

### Loss & Training

The "training" in this paper consists of learning the projection matrix $\boldsymbol{P}$ via ERM, i.e., minimizing the empirical loss:

$$\hat{\boldsymbol{P}}_S \in \arg\min_{\boldsymbol{P} \in \mathcal{P}} \frac{1}{N} \sum_{i=1}^N \ell(\boldsymbol{P}, \boldsymbol{\pi}_i)$$

Theoretically, gradients with respect to $\boldsymbol{P}$ can be computed via the Envelope Theorem. Regarding generalization guarantees:

- **Pseudo-dimension upper bound (Theorem 5.7)**: $\text{Pdim}(\mathcal{L}) = \mathcal{O}(nk \min(m, k \log m))$
- **Pseudo-dimension lower bound (Proposition 5.8)**: $\text{Pdim}(\mathcal{L}) = \Omega(nk)$
- This is **strictly tighter** than the prior LP-specific upper bound $\mathcal{O}(nk^2 \log mk)$, and applies simultaneously to both QP and LP.

## Key Experimental Results

This is a **purely theoretical work** with no numerical experiments. All main results are presented as theorems and propositions.

### Main Results

| Setting | Pseudo-dim. Upper Bound | Applicability | Source |
|---------|------------------------|---------------|--------|
| LP (Sakaue 2024) | $\mathcal{O}(nk^2 \log mk)$ | LP only | Prior work |
| QP+LP (Ours) | $\mathcal{O}(nk \min(m, k\log m))$ | QP and LP | Theorem 5.7 |
| Lower bound | $\Omega(nk)$ | QP | Proposition 5.8 |

### Ablation Study

| Extended Setting | Pseudo-dim. Upper Bound | Notes |
|-----------------|------------------------|-------|
| Solution matching | $\mathcal{O}(nk \min(m, k\log m))$ | Objective changed to $\ell_2$ distance of solutions (Theorem 6.1) |
| Instance-aware (neural network) | $\mathcal{O}(W(L\log(U+mk) + \min(m, k\log m)))$ | Learning the map $\boldsymbol{\pi} \mapsto \boldsymbol{P}$ (Theorem 6.2) |

### Key Findings

- The paper's upper bound **strictly improves** prior results even in the special LP case (a factor of $k$ reduction)
- The gap between upper and lower bounds is $\min(m, k\log m)$; closing this gap is an open problem
- The pseudo-dimension in the instance-aware setting scales linearly or logarithmically with network parameter count $W$ and depth $L$

## Highlights & Insights

1. **Elegance of the technical core**: Converting the semidefinite QP to a strictly convex problem via Tikhonov regularization, localizing the solution using Carathéodory's theorem, and mapping back to the original problem—the "perturb → analyze → take limit" technical pipeline is remarkably elegant.

2. **Unified framework**: The results apply simultaneously to LP and QP, and are tighter than the prior LP-specific results, reflecting a deeper structural understanding.

3. **Creative use of the GJ framework**: "Unrolling" the active set method into a GJ algorithm cleverly bridges combinatorial optimization with learning-theoretic tools.

## Limitations & Future Work

1. **Purely theoretical, lacking empirical validation**: The paper does not validate the practical effectiveness of data-driven projection on real-world large-scale QP instances, leaving the tightness of theoretical bounds empirically unverified.
2. **Remaining gap between upper and lower bounds**: $\Omega(nk)$ vs. $\mathcal{O}(nk \min(m, k\log m))$; whether this can be further reduced remains open.
3. **More general optimization problems not covered**: The authors mention conic programming and mixed-integer programming (MIP) as future directions in the conclusion.
4. **Computational cost of ERM training not discussed**: The optimization problem for learning the projection matrix itself may also be computationally difficult.

## Related Work & Insights

- **Sakaue & Oki (2024)**: Generalization guarantees for data-driven LP projection matrix learning—the direct predecessor of this work
- **Bartlett et al. (2022)**: Improved GJ framework—the theoretical tool underlying this paper's analysis
- **Balcan et al. (2020)**: Unified framework for data-driven algorithm design—the broader research paradigm
- **Iwata et al. (2025)**: Instance-aware LP projection learning—one of the extensions considered in this paper
- The technical pipeline (regularization → localization → unrolling as GJ algorithm → pseudo-dimension bound) may offer inspiration for analyzing generalization in other parameterized optimization problems

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The extension from LP to QP is natural yet technically nontrivial; the use of Carathéodory's theorem and the unrolled active set method are original contributions
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Complete pseudo-dimension upper and lower bounds with multiple extended settings; rigorous theoretical analysis
- **Experimental Thoroughness**: ⭐⭐ — Purely theoretical; no experiments; practical acceleration effect unknown
- **Writing Quality**: ⭐⭐⭐⭐ — The four-step technical pipeline is logically clear; intuition and formal proofs are presented separately

This paper extends data-driven projection matrix learning from LP to convex QP by proposing an unrolled active set method and casting it as a GJ algorithm, thereby establishing pseudo-dimension upper bounds and ERM generalization guarantees for learning projection matrices.

## Background & Motivation

1. **Background**: Solving large-scale LP/QP is a central problem in operations research. Projection methods accelerate solving via dimensionality reduction—learning a projection matrix $\boldsymbol{P} \in \mathbb{R}^{n \times k}$ (with $k \ll n$) to map the high-dimensional problem to a lower-dimensional space.
2. **Limitations of Prior Work**: Sakaue et al. established generalization guarantees for data-driven LP projection matrix learning, but QP optimal solutions are not restricted to vertices of the feasible polytope (unlike LP), making direct extension fundamentally geometrically challenging.
3. **Key Challenge**: LP optimal solutions lie at vertices and can be analyzed via vertex enumeration of the piecewise structure of the optimal value function; QP optimal solutions can lie anywhere inside the feasible region, precluding direct enumeration.
4. **Key Insight**: Apply Carathéodory's theorem to confine QP solutions within feasible regions corresponding to special active sets, then construct a GJ algorithm via the unrolled active set method.

## Method

### Core Problem

Learn a projection matrix $\boldsymbol{P}$ to minimize the expected projected QP optimal value $\ell(\boldsymbol{P}, \boldsymbol{\pi})$ over the problem distribution $\mathcal{D}$. ERM-based learning requires proving that the pseudo-dimension of the function class $\mathcal{L} = \{\ell_{\boldsymbol{P}} \mid \boldsymbol{P} \in \mathcal{P}\}$ is bounded.

### Key Technical Steps

1. **Perturbed objective**: Construct $\ell_{\boldsymbol{P},\gamma}$ by making $\boldsymbol{Q}$ strictly positive definite (adding $\gamma \boldsymbol{I}$), ensuring a unique solution, with the perturbed problem approximating the original to arbitrary precision.

2. **Solution localization (Carathéodory's theorem)**: The QP optimal solution can be uniquely determined by at most $k$ active constraints. It therefore suffices to check $\min(2^m, (em/k)^k)$ candidate active sets.

3. **Unrolled active set method**: For each candidate active set, the optimal solution is computed analytically via KKT conditions (involving matrix inversion), followed by verification of feasibility and optimality. The entire procedure can be expressed as a GJ algorithm.

4. **Pseudo-dimension upper bound**: The GJ algorithm has predicate complexity $\Lambda = O(m \cdot \min(2^m, (em/k)^k))$ and degree $\Delta = O(m+k)$, yielding pseudo-dimension $\text{Pdim}(\mathcal{L}) = O(nk \log(m \cdot \min(2^m, (em/k)^k) \cdot (m+k)))$.

### Main Theorem

**Theorem 5.7 (QP Generalization Guarantee)**: Given $N$ QP instances, the projection matrix learned by ERM satisfies with probability $1-\delta$:
$$\mathbb{E}[\ell_{\hat{P}}(\boldsymbol{\pi})] \leq \inf_{\boldsymbol{P}} \mathbb{E}[\ell_{\boldsymbol{P}}(\boldsymbol{\pi})] + \epsilon$$
The required sample complexity is $N = O\left(\frac{H^2}{\epsilon^2}(nk\log(m(m+k)) + \log(1/\delta))\right)$.

### Extensions

- **Solution matching setting** (Theorem 6.1): Learn $\boldsymbol{P}$ so that the projected solution $\boldsymbol{P}\boldsymbol{y}^*$ is close to the original optimal solution $\boldsymbol{x}^*$
- **Instance-aware setting** (Theorem 6.2): Use a neural network to map QP instances to customized projection matrices

## Key Experimental Results

This paper is a purely theoretical contribution with no experimental section. The core result is the establishment of generalization bounds.

### Theoretical Comparison

| Setting | Pseudo-dim. Upper Bound | Source |
|---------|------------------------|--------|
| LP (Sakaue et al.) | $O(nk \log(nkm))$ | Prior work |
| **QP (Ours)** | $O(nk \log(m(m+k) \cdot \min(2^m, (em/k)^k)))$ | Theorem 5.7 |
| QP lower bound | $\Omega(nk)$ | Proposition 5.8 |

## Highlights & Insights
- **The essential difficulty from LP to QP**: LP optimal solutions lie at vertices → enumerable; QP optimal solutions lie in the interior → require active set analysis. The application of Carathéodory's theorem elegantly discretizes a continuous optimization problem
- **Unrolled active set as a GJ algorithm**: "Unrolling" the classical active set method into a finite-step conditional branching program, enabling it to be analyzed within the GJ framework for pseudo-dimension bounds—a new technical tool
- **Results strictly generalize LP bounds**: The setting degenerates to the LP case when $\boldsymbol{Q}=0$, and the bounds obtained here are tighter

## Limitations & Future Work
- Purely theoretical work, lacking empirical validation (e.g., performance on practical QP problems such as portfolio optimization)
- The $\min(2^m, (em/k)^k)$ term in the pseudo-dimension upper bound remains large when the number of constraints $m$ is large
- A gap remains between upper and lower bounds ($\Omega(nk)$ vs. $O(nk \log(\cdots))$); tightness remains to be improved
- Only inequality-constrained QP is considered; extension to equality constraints is not discussed

## Related Work & Insights
- **vs. Sakaue et al. (LP projection)**: This paper is a direct generalization, but technically requires entirely new solution localization procedures and GJ algorithm construction
- **vs. Learning to Optimize**: The distinctive advantage of data-driven projection methods is that solutions are guaranteed to be feasible (unlike directly approximating optimal solutions with neural networks)

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical extension from LP to QP is nontrivial; the unrolled active set method is a new tool
- Experimental Thoroughness: ⭐⭐ Purely theoretical, no experiments
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and clear; proof steps are logically structured
- Value: ⭐⭐⭐ Solid theoretical contribution but limited practical impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](cost-free_neutrality_for_the_river_method.md)
- [\[ICLR 2026\] t-SNE Exaggerates Clusters, Provably](../../ICLR2026/others/t-sne_exaggerates_clusters_provably.md)
- [\[AAAI 2026\] TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models](taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)
- [\[AAAI 2026\] DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion](diffmm_efficient_method_for_accurate_noisy_and_sparse_trajectory_map_matching_vi.md)
- [\[AAAI 2026\] Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server](bipartite_mode_matching_for_vision_training_set_search_from_a_hierarchical_data_.md)

</div>

<!-- RELATED:END -->
