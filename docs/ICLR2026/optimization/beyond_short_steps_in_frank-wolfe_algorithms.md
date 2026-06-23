---
title: >-
  [Paper Note] Beyond Short Steps in Frank-Wolfe Algorithms
description: >-
  [ICLR 2026][Optimization & Theory][Frank-Wolfe] This work upgrades the analysis of the Frank-Wolfe (FW) algorithm from "short steps considering only primal progress" to a "unified primal-dual gap framework." Based on this, it proposes an Optimistic FW algorithm borrowing the "optimism" concept from online learning (featuring an $O(LD^2/t)$ primal-dual convergence bo
tags:
  - ICLR 2026
  - Optimization & Theory
  - Frank-Wolfe
date: 2026-05-08
content_hash: 0d0da27f47cd4d42
---
# Beyond Short Steps in Frank-Wolfe Algorithms

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=HBmZlcD8Ue](https://openreview.net/forum?id=HBmZlcD8Ue)  
**Code**: To be confirmed (Authors promise public release based on FrankWolfe.jl after publication)  
**Area**: optimization  
**Keywords**: Frank-Wolfe, primal-dual analysis, optimistic online learning, short step, convergence rate

## TL;DR
This work upgrades the analysis of the Frank-Wolfe (FW) algorithm from "short steps considering only primal progress" to a "unified primal-dual gap framework." Based on this, it proposes an Optimistic FW algorithm borrowing the "optimism" concept from online learning (featuring an $O(LD^2/t)$ primal-dual convergence bound and a computable stopping criterion). Furthermore, it derives a class of "primal-dual short steps" that generalize short steps to the dual gap and are transferable to gradient descent. Experimentally, the optimistic variant significantly outperforms heavy-ball FW, vanilla FW, and even adaptive line search in convergence order.

## Background & Motivation
**Background**: The Frank-Wolfe algorithm (also known as conditional gradient) is designed to solve $\min_{x\in X} f(x)$, where $X$ is a compact convex set and $f$ is convex and $L$-smooth. Its selling point is being "projection-free": each step only calls a linear minimization oracle (LMO) $v\leftarrow \arg\min_{v\in X}\langle c,v\rangle$. In scenarios where projecting onto the constraint set is expensive (optimal transport, network pruning, adversarial attacks, non-negative matrix factorization, etc.), the per-step cost is extremely low, thus it is widely used. The most common step-size strategy for FW is the **short step**: along the segment from $x_t$ to the FW vertex $v_t$, it maximizes the "guaranteed primal progress" according to the quadratic upper bound given by smoothness, yielding $\gamma=\min\{1,\frac{\langle\nabla f(x_t),x_t-v_t\rangle}{L\|x_t-v_t\|^2}\}$.

**Limitations of Prior Work**: In classic FW analysis, step-size strategies and convergence rates are often guessed via heuristics and then proven using induction. While short steps ensure a monotonic decrease in **primal progress** $f(x_t)-f(x_{t+1})$, they **do not guarantee a monotonic decrease in the primal-dual gap**. The problem is that the computable dual gap is exactly what serves as a stopping criterion, telling "how far from the optimum" the current state is, rather than the uncomputable $f(x_t)-f(x^*)$. In other words, the metric being optimized (primal progress) and the metric truly cared about (dual gap) are misaligned.

**Key Challenge**: All traditional FW approaches use "smoothness" in logically identical ways—calculating a quadratic upper bound and using its guaranteed progress to cancel out errors elsewhere in the analysis. This usage is passive: it only extracts information provided by the previous gradient $\nabla f(x_t)$ and fails to utilize predictable structures like "what the next gradient might look like." Consequently, the convergence order is fixed and cannot adapt to the actual shape of the problem.

**Goal**: Split into two sub-problems—(1) Can an FW algorithm be designed with direct guarantees for the "dual gap," its own stopping criterion, and better environmental adaptability? (2) Can the old idea of "short steps" be shifted from "maximizing primal progress" to "maximizing dual gap progress," and can it be transferred beyond FW to gradient descent?

**Key Insight**: The authors found that by placing FW into a **primal-dual analysis framework**, step sizes and convergence rates are no longer heuristics but "emerge" as natural products of the analysis. Simultaneously, this framework naturally aligns with anytime online-to-batch conversion in **online learning**, allowing mature "optimism" (using historical information to predict future loss) to be directly imported to enhance FW.

**Core Idea**: Use the "dual gap" instead of "primal progress" as the unified benchmark for analysis and step sizes. Under this benchmark, one can create an Optimistic FW using gradient prediction for optimistic updates and generalize short steps into "primal-dual short steps."

## Method

### Overall Architecture
The entire paper is built upon a unified **primal-dual analysis skeleton**, with the two main contributions being derivatives of this skeleton. The construction of the skeleton involves maintaining a computable lower bound $L_t$ for the optimal value $f(x^*)$ at time $t$ (Note: **the choice of lower bound inversely defines the algorithm's vertex $v_t$**) and defining the primal-dual gap
$$G_t \;\overset{\text{def}}{=}\; f(x_{t+1}) - L_t,$$
where taking the "one step ahead" $f(x_{t+1})$ instead of $f(x_t)$ is a subtle but critical shift that allows both new and old algorithms to obtain very concise primal-dual proofs. Combined with weights $a_t>0$ and $A_t=\sum_{\ell=0}^t a_\ell$, if one can prove the per-step recursion
$$A_t G_t - A_{t-1}G_{t-1}\le E_t,\qquad(\star)$$
one immediately obtains $f(x_{t+1})-f(x^*)\le G_t\le \frac{1}{A_t}\sum_{i=0}^t E_i$. Thus, "algorithm design" is reduced to "minimizing the error term $E_t$ and maximizing $A_t$." For example, taking $a_t=2t+2, \gamma_t=a_t/A_t=\frac{2}{t+2}$, both vanilla FW and HB-FW satisfy $(\star)$ with $E_t=\frac{LD^2a_t^2}{2A_t}$, yielding the classic bound $G_t\le \frac{2LD^2}{t+2}$. The FW gap $g(x_\ell)\overset{\text{def}}{=}\max_{v\in X}\langle\nabla f(x_\ell),x_\ell-v\rangle$ itself is a primal-dual gap and can be directly used as a stopping criterion.

Branching from this skeleton: **Section 3** replaces the "passive use of smoothness" with "optimistic prediction" to obtain Optimistic FW; **Section 4** keeps the analysis but changes the step size—shifting the goal of short steps from primal progress to the dual gap progress in $(\star)$, resulting in primal-dual short steps, which are then transferred to gradient descent.

### Key Designs

**1. Unified Primal-Dual Analysis Skeleton: Establishing "Dual Gap" as the Sole Metric**

Addressing the pain point of "misalignment between optimized and concerned metrics," the authors stop revolving around the uncomputable $f(x_t)-f(x^*)$ and instead anchor everything to the computable $G_t=f(x_{t+1})-L_t$. The key insight is that the construction of the lower bound $L_t$ determines the algorithm itself: starting from $A_t f(x^*)\ge \sum_\ell a_\ell f(x_\ell)+\sum_\ell a_\ell\langle\nabla f(x_\ell),x^*-x_\ell\rangle$ (given by convexity), if one takes $\min_v$ over the **cumulative** linear lower bound, one gets the HB-FW vertex $v_t\leftarrow\arg\min_v\sum_{i}a_i\langle\nabla f(x_i),v\rangle$; if one takes $\min_v$ for **each term separately**, it reduces to the vanilla FW vertex $v_t\leftarrow\arg\min_v\langle\nabla f(x_t),v\rangle$. The benefit of this formulation is that convergence rates are not forced through induction but land naturally from $(\star)$, while providing primal-dual analysis with better constants and built-in stopping criteria for a set of algorithms that previously only had primal analysis (e.g., composite extensions of HB-FW, FW with decaying regularization, see Appendix E). It also explains "why the dual gap of HB-FW is often tighter": the HB lower bound calculates the FW vertex over functions **accumulated across rounds**, which cannot be decomposed per round. Consequently, in classic FW zigzag scenarios (where the optimum lies reasonably inside a face while LMO only returns vertices), the accumulated gradient is closer to being perpendicular to the optimal face, making the lower bound $L_t^{HB}$ closer to $f(x^*)$.

**2. Optimistic Frank-Wolfe: Predicting the Next Gradient for Active Smoothness Exploitation**

To address "traditional FW only passively calculating quadratic upper bounds for smoothness," this work borrows optimism from online learning—at each step, using a prediction (hint) for the **next gradient** to minimize a regularized lower model of the objective. The more accurate the prediction, the better the convergence rate; for $L$-smooth functions, the previous gradient $g_t$ is a sufficiently good predictor for the next gradient. The algorithm (Algorithm 2) is based on optimistic versions of OMD / FTRL and is deliberately broadened to **non-differentiable regularization** (using Bregman divergence with subgradient selection $D_\psi(x,y,\phi)=\psi(x)-\psi(y)-\langle\phi,x-y\rangle, \phi\in\partial\psi(y)$) to cover common FW $\psi=I_X$ indicator function scenarios. Even if $\psi$ is not strongly convex (it can be 0 within the feasible region), the optimistic method still achieves the optimal order. The OFTRL variant update is $v_t\in\arg\min_v\{\sum_{i=1}^{t-1}a_i\langle g_{i+1},v\rangle+a_t\langle g_t,v\rangle+\psi(v)\}$, synthesized via anytime online-to-batch conversion as $x_t=\frac{A_{t-1}}{A_t}y_{t-1}+\frac{a_t}{A_t}v_t$ (taking $a_t=2t, A_t=t(t+1)$, i.e., $x_t=\frac{t-1}{t+1}y_{t-1}+\frac{2}{t+1}v_t$), where $y_{t-1}$ can be any point satisfying $f(y_{t-1})\le f(x_{t-1})$, leaving space for heuristics like line search. The main result (Theorem 3.1) gives the dual gap bound:
$$\mathbb{E}[f(x_t)-f(x^*)]\le \mathbb{E}[G_t^{OP}]\le \frac{\psi(x^*)-\psi(x_1)}{t(t+1)}+\frac{4LD^2}{t+1}+\frac{\sqrt{2}\,\sigma D}{2},$$
where $\sigma^2$ is the unbiased stochastic gradient variance ($\sigma=0$ in the deterministic case). The dominant term $O(LD^2/t)$ matches the optimal order, while terms involving $\psi$ are mere byproducts of step-size selection and can be made to decay at any fast polynomial rate by modifying step sizes. A subtle point is that $G_t^{OP}$ cannot be calculated exactly due to reliance on hints, but a tight approximation per step can be given, or LMO can be called occasionally for exact calculation (Remark C.6).

**3. Primal-Dual Short Steps: Shifting the Goal to Dual Gap Progress and Transferring to Gradient Descent**

Addressing the problem that "short steps only make primal progress monotonic but not the dual gap," this work modifies the step size without changing the analysis framework: since the analysis ultimately lands on the dual gap in $(\star)$, the **guaranteed progress of the dual gap** is directly maximized. For FW / HB-FW, analysis yields $A_tG_t-A_{t-1}G_{t-1}\le \frac{La_t^2}{2A_t}\|v_t-x_t\|^2$. Letting $\gamma_t=a_t/A_t$:
$$G_t\le(1-\gamma_t)G_{t-1}+\gamma_t^2\tfrac{L}{2}\|v_t-x_t\|^2,$$
where the right side is quadratic with respect to $\gamma_t$. Minimizing this yields the **primal-dual short step** $\gamma_t=\min\{1,\frac{G_{t-1}}{L\|v_t-x_t\|^2}\}$, maintaining the optimal bound $G_t\le\frac{4LD^2}{t+2}$ (Proposition 4.1; the dual gap bound is convex in $\gamma$, allowing line search solutions). The authors highlight: if the gap definition is switched back to the primal gap $G_t=f(x_{t+1})-f(x^*)$, this primal-dual short step reduces to the standard short step—they are two ends of the same idea under different metrics. Furthermore, this logic transfers to **gradient descent** $x_{t+1}\leftarrow x_t-a_t\nabla f(x_t)$. After defining a similar dual gap, the optimal step size $a_t$ is solved via a simple cubic equation (convex in $a_t$, solvable via line search), proving $a_t\ge\frac{1}{2L}$ and yielding $G_t\le\frac{LD^2}{t}$ convergence (Proposition 4.2). A notable difference: FW's convexity holds for $\gamma_t=a_t/A_t$, while GD's holds for $a_t$ itself.

### Loss & Training
This work is purely theoretical with numerical validation, without learnable parameters or training losses. The algorithmic "objective" is the regularized linear subproblem solved at each step (LMO for FW, $\arg\min_v\{\langle w,v\rangle+\psi(v)\}$ for optimistic variants). Key hyperparameters are the weight sequence $a_t$ and its derived step size $\gamma_t=a_t/A_t$: classic/optimistic variants use open-loop $a_t=2t$ or $2t+2$, while the primal-dual short step adapts via closed-form or line search.

## Key Experimental Results

Experiments were implemented in Julia using FrankWolfe.jl and run on an Apple M1 MacBook Pro. Baselines include Optimistic FW (OFTRL), Heavy-Ball FW, Vanilla FW (open-loop step size), and Vanilla + Adaptive Line Search (Pokutta 2024, default in FrankWolfe.jl). Test functions include quadratic, ill-conditioned quadratic, and portfolio benchmarks. Feasible sets include probability simplex and $k$-sparse polytopes. All plots are log-log, where the slope represents the polynomial convergence order.

### Main Results

| Method | Step Size / Mechanism | Dual Gap Convergence Order (log-log slope) | Remarks |
|--------|------|------|------|
| Vanilla FW | Open-loop $\gamma_t=\frac{2}{t+2}$ | Baseline | Simple but prone to zigzagging |
| Vanilla FW (adaptive) | Adaptive line search | Usually better than vanilla, but **sometimes slower** | Restricted by Guélat–Marcotte lower bound; open-loop is not |
| Heavy-Ball FW | Cumulative gradient lower bound | Superior to vanilla | Tighter lower bound $L_t^{HB}$ (especially during zigzagging) |
| **Optimistic FW** | Optimistic gradient prediction | **Steepest slope, optimal order** | Fastest in both iterations and wall-clock time |

Conclusion: In settings like $f(x)=\|x-x_0\|_2^2$ on a probability simplex (random $x_0$ outside, Figure 1) and $f(x)=\|Ax-b\|_2^2$ on a $k=10$ sparse polytope (Figure 2), the optimistic variant is significantly faster in both iterations and time. Its "convergence order" (slope) is strictly superior to other methods, including adaptive line search.

### Ablation Study

| Configuration | Key Finding | Description |
|------|---------|------|
| Primal-dual short step vs. standard short step | **Roughly equivalent, no breakthrough** | Tighter dual gap leads to smaller step sizes, so convergence is not faster (echoes Guélat–Marcotte bound, Appendix F Fig. 3) |
| Optimism vs. Heavy-ball bound | **Advantage stems from optimism itself** | Applying HB lower bound to vanilla FW is weaker than vanilla; the acceleration comes from the optimistic trajectory (Fig. 4) |

### Key Findings
- **Optimism is the primary driver**: The authors performed a controlled experiment—simply switching to a stronger heavy-ball lower bound does not explain the acceleration. Applying the HB bound to vanilla FW performed worse than standard FW, indicating that the improvement comes from the **better iteration trajectory** of optimistic updates, not just a tighter bound.
- **Better gap metrics do not necessarily yield faster convergence**: While primal-dual short steps make the dual gap monotonic, because this metric is tighter, the calculated step sizes are often smaller. Resulting performance ties with standard short steps, confirming the unavoidable Guélat–Marcotte lower bound for short-step/line-search methods.
- **Adaptive line search may backfire**: In some instances, it was slower than vanilla (consistent with theoretical findings in Bach 2021, Wirth et al. 2023/2024). Open-loop step sizes like optimism/heavy-ball are not constrained by this lower bound.

## Highlights & Insights
- **Metric-shifting brings unification**: Anchoring analysis and step size to the computable primal-dual gap $G_t$ allows a single $(\star)$ recursion to cover FW, HB-FW, and variants with composite terms/decaying regularization. Convergence rates "emerge" rather than being guessed.
- **Cleanly bridging online learning optimism and projection-free optimization**: Using anytime online-to-batch conversion to link OFTRL/OMD optimistic updates to FW, while extending to non-differentiable regularization ($\psi$ can be non-strongly convex or 0), covers standard FW cases without losing optimal order.
- **A thought-provoking negative result**: "Tighter gap metrics" are not equivalent to "faster convergence." Primal-dual short steps tighten the metric but fail to speed up due to smaller steps. This warns against focusing solely on metric monotonicity/tightness during acceleration design.
- **GD Transferability**: Primal-dual short steps are not FW-exclusive. Transferring to gradient descent requires solving a cubic equation and ensuring $a_t\ge\frac{1}{2L}$ to yield $O(LD^2/t)$, demonstrating a general step-size design paradigm.

## Limitations & Future Work
- **Acknowledged Limitations**: Primal-dual short steps did not outperform existing short steps/line search in experiments; the main practical gain comes from optimism.
- **Inability to calculate optimistic gaps exactly**: $G_t^{OP}$ depends on the hint for the next gradient, requiring tight approximations or extra LMO calls for exact values, which adds overhead compared to the zero-cost vanilla FW gap.
- **Scale and Tasks**: Experiments focus on theory-verifying small-scale convex problems (simplex, $k$-sparse polytopes). Performance in large-scale ML tasks remains to be verified.
- **Open Questions**: Whether local acceleration in Nesterov's sense can be achieved for non-strongly convex but smooth functions remains open.
- **Improvement Directions**: The freedom to replace $y_{t-1}$ with any point $f(y_{t-1})\le f(x_{t-1})$ could be used to overlay line search; designing hints for stronger gradient predictors might further lower constants in the optimistic bound.

## Related Work & Insights
- **vs. Classic Short Step (Frank–Wolfe 1956)**: Classic maximizes "primal progress" for $f$ monotonicity; the new PD-short step maximizes "dual gap progress" for $G_t$ monotonicity. They overlap when the gap is replaced by primal gap, but the latter inherently provides stopping criteria.
- **vs. Jaggi (2013) FW gap**: Jaggi provided the first $O(LD^2/t)$ guarantee for primal-dual gaps; this work incorporates it into the general $(\star)$ skeleton and simplifies proofs using the "one step ahead" $G_t$.
- **vs. Heavy-Ball FW (Li et al. 2021)**: HB-FW uses convex combinations of cumulative gradients; this work generalizes its analysis to composite terms and clarifies that the tight HB bound is not the root of optimistic acceleration.
- **vs. Adaptive Line Search (Pedregosa et al. 2020 / Pokutta 2024)**: Line search utilizes local curvature but at high cost and is subject to the Guélat–Marcotte lower bound; optimistic/open-loop steps bypass this.
- **vs. Optimism in Online Learning (Rakhlin & Sridharan 2013; Cutkosky 2019)**: This work grafts optimistic OMD/FTRL and anytime online-to-batch conversion onto FW, extending it to non-differentiable regularization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes dual gap as a unified metric and introduces online learning optimism to FW analysis in a clean way.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various functions/feasible sets with controlled ablations, though scale remains at the theoretical verification stage.
- Writing Quality: ⭐⭐⭐⭐ Clear progression (skeleton -> optimism -> short steps); honest reporting of negative results.
- Value: ⭐⭐⭐⭐⭐ Provides a reusable primal-dual analysis paradigm + faster Optimistic FW + GD-transferable step sizes; highly significant for the FW community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Fast Frank–Wolfe Algorithms with Adaptive Bregman Step-Size for Weakly Convex Functions](fast_frankwolfe_algorithms_with_adaptive_bregman_step-size_for_weakly_convex_fun.md)
- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)
- [\[ICLR 2026\] CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](calm_co-evolution_of_algorithms_and_language_model_for_automatic_heuristic_desig.md)
- [\[ICLR 2026\] Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers](beyond_the_heatmap_a_rigorous_evaluation_of_component_impact_in_mcts-based_tsp_s.md)
- [\[ICLR 2026\] DR-Submodular Maximization with Stochastic Biased Gradients: Classical and Quantum Gradient Algorithms](dr-submodular_maximization_with_stochastic_biased_gradients_classical_and_quantu.md)

</div>

<!-- RELATED:END -->
