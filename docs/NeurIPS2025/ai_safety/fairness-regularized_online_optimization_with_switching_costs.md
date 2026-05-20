---
title: >-
  [Paper Note] Fairness-Regularized Online Optimization with Switching Costs
description: >-
  [NeurIPS 2025][AI Safety][Long-term fairness regularization] This paper is the first to rigorously integrate long-term fairness and action smoothness into a unified online optimization framework. It first establishes tha…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Long-term fairness regularization"
  - "switching costs"
  - "smoothed online optimization"
  - "competitive ratio"
  - "Mirror Descent"
date: 2026-05-08
content_hash: 5565e4469b321839
---

# Fairness-Regularized Online Optimization with Switching Costs

**Conference**: NeurIPS 2025
**arXiv**: [2512.11131](https://arxiv.org/abs/2512.11131)  
**Code**: None  
**Area**: AI Safety / Online Optimization / Fair Scheduling
**Keywords**: Long-term fairness regularization, switching costs, smoothed online optimization, competitive ratio, Mirror Descent

## TL;DR

This paper is the first to rigorously integrate long-term fairness and action smoothness into a unified online optimization framework. It first establishes that the original problem is fundamentally intractable under standard dynamic benchmarks, then proposes FairOBD, which online-izes the fairness cost via auxiliary variables and dual mirror descent, achieving an asymptotically optimal competitive ratio under the more principled $(R, \delta)$-constrained benchmark.

## Background & Motivation

Two categories of constraints are particularly prevalent in real-world online optimization systems.

The first concerns instantaneous efficiency, such as electricity costs, latency, utilization, and load imbalance penalties.

The second concerns action variation, i.e., switching costs.

In systems such as power grid scheduling, data center resource provisioning, robot control, and object tracking, large action swings at each step are generally undesirable.

As a result, smoothed online optimization has developed into a relatively mature theoretical subfield.

However, real-world applications demand more than just being "cheap" and "smooth."

An increasing number of scenarios also require long-term fairness.

For example, when scheduling AI workloads across geographically distributed data centers, electricity consumption distributes pollution and public health risks unevenly across regions.

If an algorithm solely optimizes for minimum electricity cost, certain regions may persistently bear disproportionate externality costs.

Prior work on fair online optimization typically encodes fairness as a long-term regularizer or long-term constraint.

The critical limitation of such approaches is that they universally rely on a key assumption: the objective function at each round depends only on the current action, not on the history of past actions.

Once switching costs are introduced, this assumption immediately breaks down, since the cost incurred at the current step is inherently coupled with the action taken at the previous step.

This gives rise to the central theoretical difficulty of this paper.

Long-term fairness terms are by nature only evaluable after the entire episode concludes.

Switching costs endow each round's cost function with memory.

Consequently, neither the standard smoothed online optimization framework—which only arbitrates between current hitting costs—nor the classical fair online allocation framework—which naively converts long-term budgets into independent per-round optimizations via dual variables—applies directly.

The authors therefore pose two foundational questions.

First, is it at all possible to achieve meaningful regret or competitive ratio guarantees for this problem under a standard dynamic benchmark?

Second, if the standard benchmark is fundamentally inadequate, what is a more principled yet theoretically meaningful alternative?

The paper's contributions unfold in precisely this order: hardness results first, followed by algorithm design and a new benchmark.

## Method

### Overall Architecture

The paper considers fairness-regularized smoothed online convex optimization.

At each time step $t$, the algorithm observes a hitting cost $f_t(\cdot)$ and a context matrix $A_t$, then commits to an irrevocable action $x_t$.

The total cost comprises three components: the hitting cost (instantaneous cost), the switching cost $d(x_t, x_{t-1}) = \frac{\beta_1}{2}\|x_t - x_{t-1}\|^2$, and a long-term fairness cost $g\!\left(\frac{1}{T}\sum_{t=1}^T A_t x_t\right)$.

Here $A_t x_t$ can be interpreted as the cost vector imposed on each fairness subject by the current action—in the data center case, this corresponds to health risk exposure in each region.

FairOBD proceeds in three steps.

First, an auxiliary variable $z_t$ is introduced to decompose the fairness term, which is otherwise only computable at the end of the global horizon, into a per-round form.

Second, a dual variable $\kappa_t$ is introduced to convert the long-term fairness constraint into a linear regularizer applied to the action at each round.

Third, when selecting actions online, the algorithm jointly balances hitting cost, switching cost, and fairness regularization, while updating $\kappa_t$ via mirror descent.

### Key Designs

1. **Hardness Results as Prerequisites**

    - Function: Establish the theoretical limits of the problem to avoid pursuing the wrong benchmark.
    - Mechanism: The authors prove that even without switching costs, any online algorithm incurs a constant worst-case cost gap relative to the offline optimum, corresponding to linear regret in total cost; the competitive ratio lower bound reaches $\Omega(T)$.
    - Design Motivation: This step is crucial. It demonstrates that the combination of long-term fairness regularization and a dynamic offline optimum is inherently too strong—any online algorithm is condemned under a naive benchmark.

2. **$(R, \delta)$-OPT Benchmark**

    - Function: Impose a reasonable yet non-trivial structural constraint on the offline benchmark.
    - Mechanism: The episode is divided into frames of length $R$, and the offline optimum is required to satisfy the constraint that the cumulative deviation between the average fairness vector within each frame and the global average fairness vector does not exceed $\delta$.
    - Intuitively, this restricts the degree to which the offline optimum can arbitrarily shift its long-term fairness direction over time.
    - Design Motivation: If the offline optimum could freely switch to a completely different fairness direction in each segment, no online algorithm could ever catch up. In practice, however, many real workloads exhibit periodic patterns such as diurnal cycles, making this constraint entirely natural.

3. **Auxiliary Variable Decomposition of the Long-Term Fairness Term**

    - Function: Transform the fairness cost, which is only computable after $T$ steps, into a per-round optimizable objective.
    - Mechanism: An auxiliary variable $z_t$ is introduced as the fairness budget, with the long-term constraint $\sum_t A_t x_t = \sum_t z_t$ imposed. The fairness term can then be written as a sum of per-round terms $g(z_t)$, handled online via Lagrangian relaxation.
    - Design Motivation: This is the pivotal step that "unfolds" the long-term regularizer. Without $z_t$, the algorithm would have to blindly optimize an objective whose value is only revealed in the future.

4. **FairOBD Balanced Descent Update**

    - Function: Simultaneously account for hitting, switching, and fairness costs.
    - Mechanism: At each round, FairOBD solves a joint objective comprising the current hitting cost $f_t(x_t)$, a weighted switching cost $\lambda_1 d(x_t, x_{t-1})$, a quadratic regularizer $\frac{\lambda_2}{2}\|x_t - v_t\|^2$ toward the hitting minimizer $v_t$, and a fairness regularizer $\kappa_t \cdot A_t x_t$. Simultaneously, $z_t$ is obtained by minimizing $g(z_t) - \kappa_t z_t$, and $\kappa_t$ is updated via mirror descent using the gradient $d_t = z_t - A_t x_t$.
    - Design Motivation: Minimizing the hitting cost alone drives the action toward the instantaneous optimum; minimizing only the fairness cost sacrifices system efficiency; and the switching cost demands that actions remain stable. The name FairOBD itself reflects its nature as a balanced descent.

5. **Novel Proof Technique**

    - Function: Overcome the hidden assumption in prior analyses that per-round objective functions must be independent.
    - Mechanism: An intermediate benchmark is introduced in the appendix to bridge FairOBD and the offline optimum, ensuring that both sides compare against a per-round objective that is properly aligned.
    - Design Motivation: This is the technically hardest contribution of the paper. Without this bridge, the memory induced by switching costs causes standard primal-dual or mirror descent analyses to break down entirely.

### Loss & Training

This paper presents an online algorithm rather than a learned model; thus it is more precise to speak of decision update rules rather than a training loss.

Key hyperparameters include $\lambda_1$, $\lambda_2$, the mirror descent learning rate $\eta$, and the reference function $h(\cdot)$. When $h(\kappa) = \frac{1}{2}\|\kappa\|^2$, the dual update reduces to the familiar gradient step.

Theoretically, the authors prescribe $\eta = \mathcal{O}(T^{-1/3})$.

Without switching costs, FairOBD achieves an asymptotic competitive ratio of 1 relative to $(R, \delta)$-OPT.

With switching costs, under an appropriate choice of $\lambda_1$ and $\lambda_2$, the asymptotic competitive ratio is

$$\frac{1}{2}\!\left(1 + \sqrt{1 + 4\beta_1/m}\right).$$

This result is of the same order as the classical optimal bound for smoothed online optimization without fairness, demonstrating that the incorporation of long-term fairness does not fundamentally worsen the asymptotic order of the competitive ratio.

## Key Experimental Results

### Main Results

The experimental setting is AI inference workload scheduling across seven data centers.

The authors use one week of normalized LLM inference request traces, routing demand to data centers in Arizona, Iowa, Illinois, Texas, Virginia, Washington, and Wyoming.

The total cost comprises electricity costs, load imbalance penalties, configuration switching overhead, and a long-term public health fairness cost.

Baselines include: OPT (strongest offline optimum), FairOPT (offline optimum optimizing only fairness), HITMIN (online algorithm optimizing only hitting cost), ROBD (strong smoothed online optimization baseline without fairness), DMD (classical dual mirror descent targeting long-term fairness but ignoring switching memory), and FairOBD (the proposed method).

| Method | Hitting Cost | Switching Cost | Fairness Cost | Total Cost | Remarks |
|--------|-------------|----------------|---------------|------------|---------|
| OPT | 168.88 | 22.92 | 20.33 | 212.13 | Strongest offline lower bound; unreachable by online algorithms |
| FairOPT | 177.20 | 351.49 | 16.67 | 545.35 | Best fairness, but catastrophic switching overhead |
| HITMIN | 159.75 | 43.75 | 70.06 | 273.56 | Optimizes instantaneous cost only; fairness collapses |
| ROBD | 163.63 | 23.16 | 55.93 | 242.71 | Excellent smoothness, but no long-term fairness awareness |
| DMD | 169.46 | 93.53 | 27.08 | 290.07 | Improved fairness, but switching cost is excessively high |
| **FairOBD** ($\eta=10^{-2}$) | **169.39** | **25.67** | **21.35** | **216.41** | Lowest total cost among all online methods |

This table is particularly informative.

FairOPT appears to be the most fair, yet its switching cost is extremely large, demonstrating that optimizing only long-term fairness renders the system highly unstable.

HITMIN is the opposite extreme: the lowest hitting cost, but fairness cost inflated to 70.06.

ROBD achieves excellent switching behavior, yet its fairness cost of 55.93 shows that classical smoothed online optimization cannot incidentally learn fairness.

The value of FairOBD lies precisely here: it does not drive any single cost component to its individual optimum, but pulls all three costs onto a more favorable Pareto frontier, yielding the lowest total cost.

### Ablation Study

The paper does not conduct module-level ablations in the style of deep learning work, but instead validates robustness to the learning rate—a more critical experiment for online algorithms.

| FairOBD Learning Rate | Hitting Cost | Switching Cost | Fairness Cost | Total Cost | Observation |
|----------------------|-------------|----------------|---------------|------------|-------------|
| $\eta=10^{-2}$ | 169.39 | 25.67 | 21.35 | 216.41 | Best default configuration |
| $\eta=10^{-3}$ | 167.80 | 27.52 | 25.52 | 220.84 | Slightly higher total cost, still outperforms all online baselines |
| $\eta=10^{-4}$ | 167.47 | 28.17 | 26.34 | 221.97 | More conservative updates; fairness cost slightly degrades |

Two observations emerge from these results.

First, FairOBD is not brittle with respect to the learning rate.

Second, as the learning rate decreases, the hitting cost slightly improves while the fairness cost increases, indicating that slower dual variable updates cause the long-term fairness budget to lag behind environmental changes.

In other words, the theoretical terms in the paper are not merely aesthetic formulas—they directly explain the observed experimental phenomena.

| Theoretical Result | Condition | Result | Implication |
|-------------------|-----------|--------|-------------|
| Theorem 4.1 | No switching cost | Constant worst-case cost gap persists | Linear regret in total cost |
| Theorem 4.2 | No switching cost | Competitive ratio lower bound $\Omega(T)$ | Standard dynamic benchmark is unusable |
| Theorem 5.1 | $\beta_1 = 0$ | Asymptotic competitive ratio 1 | Fairness term can be online-ized |
| Theorem 5.2 | With switching cost | Asymptotic competitive ratio $\frac{1}{2}(1+\sqrt{1+4\beta_1/m})$ | Same order as classical smoothed online optimization |

### Key Findings

- The paper's strongest contribution is proving that the original problem is intractable before proposing a new benchmark and algorithm, rather than assuming the original benchmark is appropriate.
- In the data center case, fairness and action smoothness are genuinely in tension: FairOPT and HITMIN exhibit this conflict at opposite extremes.
- Among all online methods, FairOBD achieves the lowest fairness cost while maintaining low switching overhead, confirming that the auxiliary variable decomposition and dual updates do not destabilize the system.
- ROBD and DMD each solve only half of the problem; the value of FairOBD is precisely its unification of long-term fairness and action memory.

## Highlights & Insights

- **Using impossibility proofs as a prerequisite for algorithm design.** This is more rigorous than directly proposing a new algorithm, as it first demonstrates that the existing benchmark is inappropriate for the problem.
- **The $(R, \delta)$ benchmark is highly instructive.** In many online systems, a meaningful offline comparison should not be entirely unconstrained, especially when the long-term regularizer is intrinsically dependent on the global trajectory.
- **The auxiliary variable $z_t$ perspective is practically transferable.** Interpreting fairness as a per-round "budget allocation" rather than a terminal settlement is readily generalizable to other long-term constrained settings such as quota management, carbon emission control, and risk exposure.
- **The theoretical results do not degrade in order due to the addition of the fairness term.** This is a clean result, demonstrating that fairness does not necessarily render theoretical performance unacceptable—the key is whether the modeling approach is appropriate.
- **The paper formalizes an AI social responsibility problem into a computable, provable, and deployable framework.** This has substantially greater engineering value than treating fairness as a high-level aspiration.

## Limitations & Future Work

First, the paper adopts a squared $L_2$ form for the switching cost. While this is theoretically standard and amenable to strong convexity analysis, real-world migration costs, migration latency, and cold-start overhead may not conform to this structure.

Second, although the $(R, \delta)$ benchmark is principled, it implies that the theoretical guarantees rest on the structural assumption that "the environment is not infinitely adversarial." In highly non-periodic, non-stationary, or bursty context-switching scenarios, $\delta$ may be large, weakening the theoretical guarantees.

Third, the experiments demonstrate only a single application domain, namely geographically distributed data center scheduling. While the scenario is highly representative, supplementary case studies in power grids, logistics, or medical resource allocation would strengthen the paper's generality.

Fourth, the algorithm does not exploit predictive information. In practice, workload forecasts, price predictions, or weather data are often available; incorporating "partially trustworthy predictions" into FairOBD is a natural and important next direction.

Finally, the paper focuses primarily on worst-case performance. Relaxing to a stochastic or semi-predictive setting may enable further reduction of the gap relative to the offline optimum.

## Related Work & Insights

- **vs. classical smoothed online optimization**: The latter is well-suited to balancing hitting and switching costs but assumes the absence of long-term fairness; this paper effectively extends this line of work with a social responsibility dimension.
- **vs. DMD-style fair online allocation algorithms**: Such methods handle long-term fairness budgets but require per-round objectives to be history-independent; the true contribution of this paper is legitimately incorporating switching memory into the proof framework.
- **vs. reinforcement learning approaches for long-term constraints**: RL can model long-horizon objectives but typically lacks worst-case guarantees; FairOBD provides explicit bounds under an adversarial setting.
- **Takeaway for practitioners**: When building fair systems, fairness should not be treated as a post-hoc evaluation metric. As soon as long-term fairness enters the objective function, it fundamentally changes both the benchmark and the algorithm design for online optimization.
- **Transferable directions**: Problems involving carbon fairness, medical resource allocation, fair cache replacement, and cross-regional edge inference can all benefit from the "auxiliary variable + constrained dynamic benchmark" modeling approach introduced here.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The problem formulation, hardness proofs, $(R, \delta)$ benchmark, and FairOBD together constitute a complete new framework.
- Experimental Thoroughness: ⭐⭐⭐☆ Strong theory, representative experimental setting, but the application domain remains narrow.
- Writing Quality: ⭐⭐⭐⭐☆ Logically rigorous with a complete theoretical chain, though the entry barrier is high for readers unfamiliar with online convex optimization.
- Value: ⭐⭐⭐⭐⭐ This work genuinely makes fairness an "optimizable objective," with significant long-term value for AI infrastructure and responsible scheduling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](private_zeroth-order_optimization_with_public_data.md)
- [\[AAAI 2026\] Alternative Fairness and Accuracy Optimization in Criminal Justice](../../AAAI2026/ai_safety/alternative_fairness_and_accuracy_optimization_in_criminal_j.md)
- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)

</div>

<!-- RELATED:END -->
