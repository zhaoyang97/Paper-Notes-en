---
title: >-
  [Paper Note] Avoiding Catastrophe in Online Learning by Asking for Help
description: >-
  [ICML2025][Online Learning][Catastrophic Risk] A novel theoretical framework for online learning is proposed to address catastrophic (irreversible) errors: payoffs are defined as catastrophe avoidance probabilities, and the objective function is the product of payoffs (overall catastrophe avoidance probability). By introducing an instructor querying mechanism and a Local Generalization assumption, the paper establishes an impossibility result (catastrophe is inevitable withou…
tags:
  - "ICML2025"
  - "Online Learning"
  - "AI Safety"
  - "Catastrophic Risk"
  - "Querying Mechanism"
  - "Regret Bounds"
  - "Local Generalization"
  - "VC Dimension"
  - "Littlestone Dimension"
date: 2026-05-08
content_hash: ff92dabd43aaf14b
---

# Avoiding Catastrophe in Online Learning by Asking for Help

**Conference**: ICML2025  
**arXiv**: [2402.08062](https://arxiv.org/abs/2402.08062)  
**Code**: None  
**Area**: Online Learning / AI Safety  
**Keywords**: Online Learning, Catastrophic Risk, Querying Mechanism, Regret Bounds, Local Generalization, VC Dimension, Littlestone Dimension

## TL;DR
A novel theoretical framework for online learning is proposed to address catastrophic (irreversible) errors: payoffs are defined as catastrophe avoidance probabilities, and the objective function is the product of payoffs (overall catastrophe avoidance probability). By introducing an instructor querying mechanism and a Local Generalization assumption, the paper establishes an impossibility result (catastrophe is inevitable without query) and a possibility result (if the policy class is learnable, both regret and query rate vanish simultaneously), elevating sublinear regret in standard online learning to subconstant regret.

## Background & Motivation

**Background**: Online learning theory is highly mature—classical results show that a finite Littlestone dimension is a necessary and sufficient condition for sublinear regret, and under $\sigma$-smooth inputs, a finite VC dimension suffices. However, almost all theoretical results implicitly assume a crucial premise: **all errors are recoverable**. Algorithms rely on trial-and-error strategies to "try all actions to see which is best," assuming that no single error is destructive.

**Limitations of Prior Work**: In high-stakes scenarios such as medical AI, autonomous driving, and autonomous weapons, a single error can be catastrophic and irreversible. It is unacceptable for a surgical robot to "try all possible moves to see which is best," or for an autonomous vehicle to learn by crashing on real roads. However, existing theoretical frameworks—whether bandits, MDPs, or online learning—cannot formally guarantee catastrophe avoidance.

**Key Challenge**: Learning requires exploration, yet exploration can lead to catastrophe. Complete avoidance of exploration prevents learning, while unrestrained exploration can cause irreversible harm. This exploration-safety trade-off is unsolvable under standard frameworks.

**Goal**: (1) Generally, is avoiding catastrophe possible? What is the cost? (2) Under what conditions can an agent achieve both vanishing regret and a vanishing query rate? (3) Can safe decisions be made without relying on Bayesian inference?

**Key Insight**: The authors observe that "instructors" (human doctors, drivers, operators) are typically available to assist in reality. If the agent can request help when uncertain, it can bypass the dilemma of needing to learn through trial and error. The key is for the agent to know when to ask for help, which is achieved through the Local Generalization assumption (similar inputs exhibit similar safety characteristics).

**Core Idea**: If a policy class is learnable in standard online learning without catastrophic risks, it remains learnable under the catastrophic risk setting when incorporating instructor queries and Local Generalization—with both regret and the query rate vanishing.

## Method

### Overall Architecture
At each time step, the agent observes an input $x_t$ and either selects an action or queries the instructor. The per-round payoff $\mu_t(x_t, y_t) \in [0,1]$ represents the probability of avoiding catastrophe in that round. The objective is to maximize $\prod_{t=1}^{T} \mu_t(x_t, y_t)$, which is the overall probability of avoiding catastrophe. The agent **does not observe payoffs** (information about the instructor's actions is only obtained through queries). The core algorithm consists of two components: (1) running a modified Hedge algorithm for "familiar" inputs, and (2) querying the instructor for "unfamiliar" inputs.

### Key Designs

1. **Impossibility Result (Theorem 4.1)**:

    - **Function**: Proving a general lower bound—regardless of the algorithm, if the number of queries is sublinear, the regret must be unbounded.
    - **Mechanism**: Partitioning the input space $[0,1]$ into $f(T)$ equal-sized independent intervals, where in each interval one action is optimal (payoff = 1) and the other action is penalized. If the number of queries is much smaller than the number of intervals, the agent has to guess in most intervals, guessing incorrectly about half of the time. By choosing $f(T) = \max(\sqrt{\sup E[|Q_T|] \cdot T}, 1)$ to maximize the lower bound, the regret of $\Omega(L\sqrt{T/(\sup E[|Q_T|]+1)})$ is obtained.
    - **Design Motivation**: Establishing the fundamental difficulty of the problem—demonstrating that catastrophe avoidance is not "free" and requires additional conditions (such as structural assumptions on the policy class).

2. **Local Generalization Assumption**:

    - **Function**: Providing a knowledge transfer assumption that is computationally more tractable than Bayesian inference.
    - **Mechanism**: Assuming there exists a constant $L > 0$ such that $|\mu_t^m(x) - \mu_t(x, \pi^m(x'))| \le L\|x - x'\|$, meaning that executing the instructor's action on similar inputs yields a payoff gap linearly bounded by the input distance. The agent can determine whether an input is "familiar" by computing the nearest-neighbor distance between the current input and previously queried inputs.
    - **Design Motivation**: Bayesian inference requires prior distributions and posterior computation, which are intractable in complex environments. Local Generalization only requires a distance metric, making it applicable to any input space with a distance concept, and the agent does not need to know the value of $L$.

3. **Algorithm 1: Combination of Hedge & Querying**:

    - **Function**: Achieving vanishing regret and vanishing query rate simultaneously under the condition of a finite VC dimension or Littlestone dimension.
    - **Mechanism**: The algorithm operates in two layers: **Hedge Layer**—runs label-efficient Hedge using an $\varepsilon$-cover $\tilde{\Pi}$ of the policy class (actively querying the instructor with probability $p = 1/\sqrt{\varepsilon T}$ at each step) to guarantee bounded errors; **Safety Layer**—for the policy $\pi_t$ selected by Hedge, checks if the nearest-neighbor distance between the current input and previously queried inputs with the same action exceeds $\varepsilon^{1/n}$, querying the instructor if it does. By choosing $\varepsilon = T^{-2n/(2n+1)}$, the multiplicative regret is $E[R_T^\times] \in O(T^{-1/(2n+1)} \log T)$ and the query count is $E[|Q_T|] \in O(T^{(4n+1)/(4n+2)})$, both respectively vanishing and sublinear.
    - **Design Motivation**: Hedge guarantees "not making too many mistakes," while Local Generalization guarantees "the cost is small when mistakes are made." Their combination achieves subconstant regret. Key insight: the agent does not need to observe the payoff, only requiring distance computations to determine safety.

### Loss & Training
This is a theoretical work with no explicit training stage. The core optimization objective is to maximize the product of payoffs $\prod_{t=1}^T \mu_t(x_t, y_t)$, which is equivalent to minimizing the multiplicative regret $R_T^\times = \log \prod \mu_t^m(x_t) - \log \prod \mu_t(x_t, y_t)$. The authors also provide bounds for the additive regret $R_T^+ = \sum \mu_t^m(x_t) - \sum \mu_t(x_t, y_t)$ and prove that they are tightly related in this setting (Lemma A.1).

## Key Experimental Results

### Comparison of Main Theoretical Results

| Setting | Input Type | Policy Class Condition | Regret Bound | Query Rate | Catastrophe Avoidance |
|------|---------|-----------|-----------|--------|---------|
| Standard Online Learning | Adversarial | Finite Littlestone Dim | $o(T)$ (Sublinear) | N/A | Not Addressed |
| Standard Online Learning | $\sigma$-smooth | Finite VC Dim | $o(T)$ | N/A | Not Addressed |
| Ours (No Structure) | i.i.d. | Unconstrained | $\Omega(L\sqrt{T/q}) \to \infty$ | Regret explodes if query rate is sublinear | **Impossible** |
| **Ours (Structured)** | $\sigma$-smooth | VC Dim = $d$ | $O(T^{-1/(2n+1)} \log T) \to 0$ | $O(T^{(4n+1)/(4n+2)}/T) \to 0$ | **Possible** |
| **Ours (Structured)** | Adversarial | Littlestone Dim = $d$ | $O(T^{-1/(2n+1)} \log T) \to 0$ | Same as above $\to 0$ | **Possible** |

### Comparison between Our Model and Standard Online Learning Model

| Dimension | Standard Model | Our Model |
|------|---------|---------|
| Objective Function | Sum of payoffs (Additive) | Product of payoffs (Multiplicative) |
| Regret Goal | Sublinear $o(T)$ | **Subconstant $\to 0$** |
| Feedback | Observed every round | **Only via queries** |
| Instructor | None | Yes, fixed policy |
| Local Generalization | None | Yes |
| Catastrophic Errors | Non-existent / Not addressed | **Core Focus** |

### Key Findings
- **Querying is a necessary condition for catastrophe avoidance** (Corollary 4.1.1): Even if the instructor can avoid catastrophe with probability 1, any algorithm with a sublinear query rate will have a catastrophe probability approaching 1 in the worst case—there is no "free lunch".
- **Elegant equivalence of learnability**: Learnability in standard online learning (finite VC/Littlestone dimension) translates perfectly to the catastrophe setting—the same structural conditions are sufficient to guarantee catastrophe avoidance, requiring no additional assumptions.
- **Subconstant vs. sublinear regret**: Traditional $o(T)$ regret implies that the "average error per round vanishes," whereas the vanishing regret in this paper implies that the "**total** probability of catastrophe vanishes"—differing by a factor of $T$, which stems from the combination of the instructor and Local Generalization.
- **Elegant properties of the algorithm**: It does not require knowledge of $\sigma$, $L$, or $T$ (via the doubling trick), applies to unbounded input spaces, and holds simultaneously for all $\mu$ that satisfy Local Generalization.

## Highlights & Insights
- **Mathematical elegance of the product of payoffs as the objective function**: A payoff close to 0 in any single round collapses the product, perfectly encoding the irreversible semantics of "a single catastrophe is enough." This is more fundamental than simply increasing the penalty weight in the loss function, as it shifts the required regret order from sublinear to subconstant.
- **The equivalence theorem of "if learnable, then safely learnable"**: This is the core contribution of the paper—reducing AI safety problems to standard learning theory problems and demonstrating that safety does not require inherently stronger assumptions, but only a querying mechanism and distance computation capability.
- **Local Generalization as an alternative to Bayesian inference**: Prior theoretical works addressing irreversible risk almost exclusively relied on Bayesian frameworks (requiring prior distributions and computable posteriors). The assumption in this paper requires only the existence of a distance metric, significantly expanding the applicability of the theory.

## Limitations & Future Work
- The theoretical framework is highly abstract, and the time complexity of Algorithm 1 is $\Omega(|\tilde{\Pi}|)$ (the policy cover can be exponentially large); practical deployment requires an oracle-efficient version.
- The instructor's policy is assumed to be fixed, but in reality, instructors (such as human doctors) may make mistakes, experience fatigue, or evolve their strategies.
- The Local Generalization assumption requires the existence of an input embedding that satisfies the conditions, but how to find or verify such an embedding remains an open problem.
- The MDP/reinforcement learning setting is not addressed, which the paper lists as a core open problem at the end.
- This is a purely theoretical work without experimental validation; the constants in the bounds may be large.

## Related Work & Insights
- **vs. Safe RL (Liu et al. 2021, Stradi et al. 2024)**: Safe RL assumes known safe policies, periodic resets, and observable safety costs—none of which are required in the framework of this paper. The agent in this work has no prior knowledge, does not reset, and does not observe payoffs, allowing catastrophe avoidance under weaker conditions.
- **vs. Bayesian Querying Methods (Cohen et al. 2021)**: Bayesian methods require a finite set of environments and computable posteriors, whereas this paper handles uncountable policy classes and continuous, unbounded input spaces. Conversely, Local Generalization only requires distance computations.
- **vs. Standard Online Learning (Littlestone 1988, Haghtalab et al. 2024)**: The positive results in this paper build upon these classical results—inheriting standard learnability via Hedge + cover techniques, and then elevating the regret order from sublinear to subconstant through the querying mechanism and Local Generalization.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ A completely new theoretical framework; the objective function, regret definition, impossibility results, and positive algorithms are all original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐ Purely theoretical work with no experiments, but the proofs of the theorems are rigorous and complete, including both upper and lower bounds as well as multiple variants.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely clear presentation, with intuitive explanations and formal proofs interweaving seamlessly, making it highly readable.
- **Value**: ⭐⭐⭐⭐⭐ Fundamental contribution to the theoretical foundations of AI safety, the equivalence of "learnable implies safely learnable" is poised to become a classic result.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Computable Universal Online Learning](../../NeurIPS2025/learning_theory/computable_universal_online_learning.md)
- [\[ICLR 2026\] Online Decision-Focused Learning](../../ICLR2026/learning_theory/online_decision-focused_learning.md)
- [\[ICML 2025\] Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](improved_and_oracle-efficient_online_ell_1-multicalibration.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/learning_theory/conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)

</div>

<!-- RELATED:END -->
