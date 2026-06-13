---
title: >-
  [Paper Note] Optimal Transport under Group Fairness Constraints
description: >-
  [ICML 2026][AI Safety][Group Fairness] This paper explicitly encodes "group fairness" as a target inter-group matching probability $\mathbf{F}$ of size $K_s \times K_w$. It proposes three solutions: **FairSinkhorn** for…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Group Fairness"
  - "Optimal Transport"
  - "Sinkhorn"
  - "Bilevel Optimization"
  - "Cost Learning"
date: 2026-05-08
content_hash: b26c66215fab499d
---

# Optimal Transport under Group Fairness Constraints

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2601.07144](https://arxiv.org/abs/2601.07144)  
**Code**: https://github.com/LinusBleistein/fair_ot (Available)  
**Area**: AI Safety / Algorithmic Fairness / Optimal Transport  
**Keywords**: Group Fairness, Optimal Transport, Sinkhorn, Bilevel Optimization, Cost Learning

## TL;DR
This paper explicitly encodes "group fairness" as a target inter-group matching probability $\mathbf{F}$ of size $K_s \times K_w$. It proposes three solutions: **FairSinkhorn** for exact solution, **Penalized OT** for convex relaxation, and **Bilevel Cost Learning**. The authors provide finite sample complexity bounds of $O(1/\sqrt{n})$ and fairness bias bounds of $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$, delineating the "cost-fairness" trade-off frontier on synthetic and semi-synthetic (dating app) datasets.

## Background & Motivation

**Background**: Algorithmic matching (admissions, recruitment, dating, kidney transplants, urban resource allocation) is increasingly determined by centralized algorithms. Optimal Transport (OT) has become a primary tool due to its favorable modeling properties in economics and social sciences. Given two distributions $\mu, \eta$ and a cost $c$, entropy-regularized OT solves $\min_\pi \int c(x,y)\,d\pi + \varepsilon \mathbf{KL}(\pi|\mu\otimes\eta)$ using the efficient iterative Sinkhorn algorithm.

**Limitations of Prior Work**: When features $X, Y$ are strongly correlated with sensitive attributes $S, W$ (gender, race, socioeconomic status), standard OT yields "block-diagonal" highly homogeneous mappings—for example, assigning almost all high-income students to elite schools. Existing fairness $\times$ matching literature focuses almost exclusively on **individual fairness** (similar individuals get similar outcomes) or **task-specific participation quotas** (e.g., kidney exchange KPD), lacking a "group matching probability" framework for OT that can be flexibly specified by a central planner.

**Key Challenge**: (a) Most existing fair-OT works treat OT as a tool for downstream fair prediction (e.g., Wasserstein barycenter projection) rather than studying the fairness of the **transport plan itself**; (b) Exact fairness often incurs a substantial "price of fairness," requiring a relaxation framework for fine-grained trade-off control; (c) Whether a fairness goal can be **reused** on new samples after being "imprinted" into the matching remains an open question.

**Goal**: (1) Formalize the new group fairness definition of "inter-group matching probability = target $\mathbf{F}$"; (2) Provide an exact algorithm and two types of relaxation methods; (3) Supply finite sample theoretical guarantees for the relaxation methods.

**Key Insight**: The fairness constraint $\pi_{SW}(s,w) = \mathbf{F}_{sw}$ is **linear** with respect to the transport plan $\mathbf{\Pi}$, allowing the Lagrange dual to maintain a Sinkhorn-style multiplicative form. Simultaneously, entropy regularization ensures uniqueness and differentiability, making the bilevel optimization of "inducing fairness by learning costs" a well-posed problem.

**Core Idea**: A "group probability target matrix $\mathbf{F}$" is used to unify real-world rules such as restricting homophily, minimum minority quotas, the 4/5ths rule for hiring, and the EU's 40% gender quota for boards, providing an algorithmic spectrum from exact to relaxed solutions.

## Method

### Overall Architecture

Input: Two samples with sensitive attributes $(\mathbf{x}_i, \mathbf{s}_i)_{i=1}^n \sim \mu$ and $(\mathbf{y}_j, \mathbf{w}_j)_{j=1}^m \sim \eta$, a cost matrix $\mathbf{C}_{ij} = c(\mathbf{x}_i, \mathbf{y}_j)$, entropy regularization $\varepsilon$, and a fairness target $\mathbf{F} \in \Pi(\mathbf{p}, \mathbf{q})$ (where margins match sensitive attribute distributions).

Output: A transport plan $\mathbf{\Pi} \in \mathbb{R}_+^{n\times m}$ with row/column sums of $1/n \cdot \mathbf{1}_n$ and $1/m \cdot \mathbf{1}_m$, respectively, such that the inter-group mass $\sum_{i: s_i=s, j: w_j=w} \mathbf{\Pi}_{ij}$ is close to $\mathbf{F}_{sw}$.

Three solution paths form a spectrum: **FairSinkhorn** (Exact = Hard constraint) → **Penalized OT** (Direct penalty = Convex relaxation) → **Cost Learning** (Indirect geometry reshaping = Reusable bilevel optimization).

### Key Designs

1.  **FairSinkhorn: An Extra "Group-Level Reprojection" in Sinkhorn**
    *   **Function**: Incorporates exact inter-group constraints $\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] = \mathbf{F}_{sw}$ into entropy-regularized OT, where $\mathbf{B}_{sw}$ is a 0/1 indicator matrix for group pairs.
    *   **Mechanism**: By introducing Lagrange dual variables $\mathbf{h} \in \mathbb{R}^{K_s \times K_w}$, the optimal solution takes the form $\mathbf{\Pi} = \text{diag}(e^{\mathbf{f}/\varepsilon})(\mathbf{K} \odot \mathbf{H}) \text{diag}(e^{\mathbf{g}/\varepsilon})$, where $\mathbf{K} = e^{-\mathbf{C}/\varepsilon-1}$ is the standard Sinkhorn kernel and $\mathbf{H} = \sum_{sw} e^{h_{sw}/\varepsilon} \mathbf{B}_{sw}$ is a block-constant "fairness coefficient" matrix. The algorithm inserts a step $\mathbf{L}^{(t+1)} \leftarrow \mathbf{F} \oslash \Phi(\mathbf{u}^{(t+1)}, \mathbf{v}^{(t+1)})$ to update $\mathbf{H}$ between standard row/column normalization steps, where $\Phi$ aggregates the group-level mass of the current transport.
    *   **Design Motivation**: Since fairness constraints are linear, they can be "grafted" onto the Sinkhorn multiplicative structure with the same complexity order. This serves as the "perfect fairness" baseline, though it may significantly increase transport costs by overriding inter-group geometry.

2.  **Penalized OT: A Convex "Cost-Fairness" Knob**
    *   **Function**: Replaces hard constraints with a squared penalty $\mathcal{L}_\mathbf{F}(\mathbf{\Pi}) = \sum_{(s,w)} (\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] - \mathbf{F}_{sw})^2 \le \rho$ and adds it to the objective: $\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}] + \varepsilon \mathbf{KL}(\mathbf{\Pi}) + \lambda \mathcal{L}_\mathbf{F}(\mathbf{\Pi})$, where $\lambda$ is the "fairness strength."
    *   **Mechanism**: Since $\mathcal{L}_\mathbf{F}$ is convex, the problem remains strongly convex with a unique solution. It is solved using **Generalized Conditional Gradient**: each iteration linearizes the penalty around $\mathbf{\Pi}^t$ to obtain a modified cost $\mathbf{C} + \nabla \mathcal{L}_\mathbf{F}(\mathbf{\Pi}^t)$, solves this subproblem with standard Sinkhorn, and performs a line search. The authors prove $\mathbb{E}|m^\star(\mu_n, \eta_n) - m^\star(\mu, \eta)| \lesssim 1/\sqrt{n}$, equivalent to standard entropy-regularized OT—meaning the fairness penalty **does not degrade statistical efficiency**.
    *   **Design Motivation**: Exact fairness is too rigid; many scenarios only require "approximate" fairness. $\lambda$ allows $\mathbf{\Pi}$ to move smoothly along a convex curve between standard Sinkhorn ($\lambda=0$) and FairSinkhorn ($\lambda\to\infty$).

3.  **Bilevel Cost Learning: Learning a "Fairness-Inducing" Geometry**
    *   **Function**: Instead of directly modifying $\mathbf{\Pi}$, it learns a parameterized cost $c_\theta$ such that the induced entropy-regularized OT solution $\mathbf{\Pi}_\varepsilon(c_\theta)$ naturally satisfies the fairness target.
    *   **Mechanism**: Formulated as a bilevel optimization $\min_\theta \mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) + \frac{1}{\lambda} \mathscr{D}(c_\theta, c_\text{base})$ s.t. $\mathbf{\Pi}_\varepsilon(c_\theta) = \arg\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}_\theta] + \varepsilon \mathbf{KL}(\mathbf{\Pi})$. Two parameterizations are used: **Mahalanobis** $c_\mathbf{M}(x,y) = (x-y)^\top \mathbf{M} (x-y)$ and **Neural Cost** $c_\theta(x,y) = \|\phi_{\theta_1}(x) - \phi_{\theta_2}(y)\|_2^2$. Gradients are obtained via iterative or implicit differentiation. The authors prove that for any $\theta$ in the family, the fairness bias on new samples satisfies $\sup_\theta \mathbb{E}[|\mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) - \mathcal{L}_\mathbf{F}(\pi_\varepsilon^\star(c_\theta))|] \lesssim \exp(5R_\Theta/\varepsilon)/\sqrt{n}$.
    *   **Design Motivation**: Penalized OT must be re-solved for every new batch. **Cost Learning** allows the learned cost to be cached and reused on new data, saving inference time and generalizing fairness to unseen samples. Additionally, Mahalanobis costs identify which feature directions contribute to unfairness.

### Loss & Training
The exact version (FairSinkhorn) requires only $T$ fixed-point iterations with no learnable parameters. Penalized OT is a convex problem where $\lambda$ controls fairness. The cost learning version uses SGD/Adam to optimize $\theta$, running an inner entropy-regularized Sinkhorn at each step. The regularization $\mathscr{D}(c_\theta, c_\text{base})$ is implemented using $\|\mathbf{M} - \mathbf{I}\|_F^2$ or $\ell_2$ weight norms. Trade-off curves are generated by scanning $\varepsilon$ and $\lambda$.

## Key Experimental Results

### Main Results

**Synthetic Experiments**: Gaussians (two groups of students/schools generated via GMMs, privileged groups closer to elite schools) and Circles (minority group on a radius-2 ring). Target $\mathbf{F} = \begin{bmatrix} 0.20 & 0.30 \\ 0.28 & 0.22 \end{bmatrix}$, i.e., 60% of disadvantaged students matching elite schools.

| Method | Fairness Violation $\mathcal{L}_\mathbf{F}$ | Transport Cost (Rel. to Sinkhorn) | Notes |
| :--- | :--- | :--- | :--- |
| Sinkhorn (vanilla) | High (Large block-diagonal bias) | 0 (Baseline) | Completely ignores fairness |
| Sinkhorn + Large $\varepsilon$ | Still High | Increased | Entropy reg. **cannot** approximate $\mathbf{F}$ |
| FairSinkhorn | $\approx 0$ (Exact) | Significantly Increased | Perfect fairness, highest cost |
| Penalized OT (varying $\lambda$) | Smooth Interpolation | Smooth Interpolation | Convex curve, most flexible trade-off |
| Cost learning - Mahalanobis | Medium ($\ge 10^{-2}$ on Circles) | Medium | Limited by linear geometry |
| Cost learning - MLP | Low | Medium (Similar to Penalized on Gaussians) | Non-linear geometry, matches Penalized on Circles |

**Semi-synthetic Dating App Experiment**: Kaggle dataset sub-sampled to match US demographics. Sensitive attributes are 7 income levels; feasible matching is determined by sexual orientation. Target $\mathbf{F}_{sw} = \mathbb{P}(S=s_i) \mathbb{P}(W=w_j)$ (Independency = breaking income homophily). Results match synthetic experiments: trade-off curves are convex, and Cost Learning performs similarly to Penalized OT while being more scalable.

### Ablation Study

| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Tuning $\varepsilon$ only (vanilla OT) | Fairness violation **does not converge** to $\mathbf{F}$ | Entropy reg. only blurs the plan, doesn't shift it toward $\mathbf{F}$ |
| Increasing $\lambda$ (Penalized) | Convex curve from Sinkhorn → FairSinkhorn | Confirms controllable and continuous trade-off |
| Mahalanobis vs MLP (Cost Learning) | MLP significantly lower on Circles | Linear geometry is powerless against ring distributions |
| Train cost → reuse on test set | Inference 1-2 orders faster than Penalized; slight fairness loss | Learned cost is reusable; gap between MLP/Mahalanobis exists |

### Key Findings
- **Entropy regularization is not a substitute for fairness constraints**: Even large $\varepsilon$ values do not lead vanilla OT toward $\mathbf{F}$.
- **Penalized OT is the most flexible**: It searches across all couplings $\Pi$, whereas Cost Learning is restricted to a narrower subset defined by the cost family. However, Penalized OT requires re-calculation for every batch.
- **The value of Cost Learning is "Transferability"**: Particularly useful for real-time systems (recruitment, dating). Mahalanobis matrices also serve as diagnostic tools for fairness.
- **Closing the Loop**: The theoretical $O(1/\sqrt{n})$ and $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$ results match experimental observations—the fairness gap narrows as sample size increases.

## Highlights & Insights
- **The fairness target abstraction is highly general**: Regulations like the 4/5ths rule or board quotas can be expressed as $\mathbf{F}$, directly bridging policy and algorithms.
- **Linear constraints imply minimal Sinkhorn overhead**: FairSinkhorn maintains a block-constant matrix $\mathbf{T}^{(t)}$ with nearly zero extra cost, allowing drop-in replacement.
- **Convex penalties maintain statistical efficiency**: Proving that adding fairness penalties preserves $O(1/\sqrt{n})$ complexity is a significant independent contribution to constrained OT.
- **Transferable paradigm**: The bilevel "learning cost to induce behavior" framework can be generalized beyond fairness to targets like sparsity or domain invariance.

## Limitations & Future Work
- **Group fairness only**: Individual fairness requires different formalizations.
- **Fairness-bias trade-off**: Robustness is not quantified for when $\mathbf{F}$ is mis-specified.
- **Downstream utility**: Matching is only the first step; actual outcomes (wages, GPA) are not modeled.
- **Exponential dependence on $\varepsilon$**: The Cost Learning bound explodes as $\varepsilon \to 0$, necessitating careful parameter tuning.
- **Potential improvements**: Making $\mathbf{F}$ learnable (inequality constraints), utilizing causal graphs for counterfactual fairness, and hybridizing with stable matching (Gale-Shapley).

## Related Work & Insights
- **vs Gale-Shapley fair variants**: Those handle iterative individual matching; this work focuses on continuous mass OT and inter-group quotas.
- **vs KPD fairness**: KPD quotas are often restricted to integer programming; this work applies a general $\mathbf{F}$ matrix to a continuous relaxation framework.
- **vs Wasserstein-barycenter fairness**: Those use OT as a FAIR **prediction** tool; this work ensures the **transport plan itself** is fair.
- **vs constrained OT**: This work grounds constrained OT in the specific semantics of "group fairness" and provides the missing sample complexity theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](../../ICLR2026/ai_safety/optimal_transport-induced_samples_against_out-of-distribution_overconfidence.md)
- [\[CVPR 2026\] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](../../CVPR2026/ai_safety/subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)
- [\[ICML 2026\] Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[AAAI 2026\] Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](../../AAAI2026/ai_safety/truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)

</div>

<!-- RELATED:END -->
