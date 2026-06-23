---
title: >-
  [Paper Note] Computing Equilibrium beyond Unilateral Deviation
description: >-
  [ICLR 2026][learning_theory][Paper Note] Addressing the blind spot where Nash/Correlated Equilibria only prevent "unilateral deviation" but fail against "multilateral coalition deviation," this paper proposes a **guaranteed-to-exist** solution concept called MASE (Minimum Average-Strong Equilibrium, which minimizes the maximum average utility any coalition ca
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 621b3e32fb433863
---
# Computing Equilibrium beyond Unilateral Deviation

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=XXNexSaay2](https://openreview.net/forum?id=XXNexSaay2)  
**Code**: TBD  
**Area**: Learning Theory / Algorithmic Game Theory  
**Keywords**: Game theoretic equilibrium, coalition deviation, computational complexity, treewidth, no-regret learning

## TL;DR
Addressing the blind spot where Nash/Correlated Equilibria only prevent "unilateral deviation" but fail against "multilateral coalition deviation," this paper proposes a **guaranteed-to-exist** solution concept called MASE (Minimum Average-Strong Equilibrium, which minimizes the maximum average utility any coalition can gain). It proves the computational complexity is determined by the **treewidth** of the "Utility Dependency Graph" (NP-hard + exponential lower bound under SETH) and provides an algorithm with a runtime perfectly matching this lower bound.

## Background & Motivation

**Background**: The most commonly used solution concepts in game theory—Nash Equilibrium (NE), Correlated Equilibrium (CE), Coarse Correlated Equilibrium (CCE), and Stackelberg Equilibrium—essentially guarantee one thing: **no single player** can improve their utility by unilaterally changing their strategy. They characterize stability under "unilateral deviation."

**Limitations of Prior Work**: In reality, players often **form coalitions**. A group of players can coordinate to change strategies together such that everyone in the coalition benefits—NE/CE provide no guarantees against such "multilateral deviation." The classic example is the Prisoner's Dilemma: the unique NE is (Defect, Defect), with a social welfare of only 0.4; however, if two players coordinate to (Cooperate, Cooperate), both improve (0.6 each). NE completely fails to prevent such collusion.

**Key Challenge**: Game theory long ago proposed concepts for coalition deviation—Strong Nash Equilibrium (strong NE), $k$-Strong NE, and coalition-proof equilibrium—but they have a fatal flaw: **they almost never exist in general games**. This paper uses a lemma to prove that even in simple games like the Prisoner's Dilemma, Strong NE and Strong Correlated Equilibrium do not exist (assigning any positive weight to (C,C)/(C,D)/(D,C) makes a single-player coalition profitable, while playing (D,D) triggers a deviation by the grand coalition {1,2} to (C,C)). The fundamental reason is that **objectives of different coalitions conflict**, and no single strategy can make all coalitions simultaneously unprofitable.

**Goal**: Rather than searching for a strategy "immune to all coalition deviations" (which does not exist), this paper takes a step back—seeking a strategy that "minimizes the incentive to deviate." By the Weierstrass Theorem, a minimum must exist, thus making this defined solution concept **always exist**.

**Key Insight**: Quantify "stability against coalition $S$" as the **average gain** from deviation (the average utility improvement of all members in the coalition), then take the maximum over all possible coalitions and the minimum over all strategies.

**Core Idea**: Replace "requiring the gain to be exactly $\le 0$" with "minimizing the worst-case average deviation gain," transforming a **feasibility problem** (finding an immune point) into an **optimization problem** (finding the most stable point) to ensure existence. Furthermore, it is discovered that the difficulty of this optimization problem is precisely characterized by the **treewidth** of the game's interaction structure.

## Method

### Overall Architecture

The paper addresses two tasks: (1) Defining a universally existent solution concept MASE that captures coalition deviations; (2) Determining its computational complexity and providing an algorithm that matches that complexity.

**Defining MASE.** Given a game $(N, \{A_i\}, \{U_i\}, S)$ with $N$ players, action sets $A_i$, utility functions $U_i: A \to [0,1]$, and a **set of allowed coalitions** $S$ (each a subset of players; if only unilateral deviations are allowed, $S=\{\{1\},\dots,\{N\}\}$). MASE seeks a **correlated strategy** $\pi \in \Delta_A$ (a distribution over joint actions) such that:

$$\pi^* \in \arg\min_{\pi \in \Delta_A} \ \max_{S \in S} \ \max_{\hat{a}_S \in A_S} \ \frac{1}{|S|} \sum_{i \in S} \mathbb{E}_{a \sim \pi}\big[U_i(\hat{a}_S, a_{-S}) - U_i(a)\big].$$

Intuition: The inner $\max$ identifies the "most motivated coalition $S$ and its best joint deviation $\hat{a}_S$," measuring the **average gain** it can achieve (sum of utility increases divided by $|S|$); the outer $\min$ selects a strategy that minimizes this worst average gain. If this value is $\le 0$, no coalition can strictly benefit all its members. This solution is called **Minimum Average-Strong Equilibrium (MASE)**. An $\epsilon$-MASE is a strategy with an objective value within $+\epsilon$ of the optimum.

**Two Hardness Results + One Matching Algorithm.** This paper first proves that MASE is hard to compute in general (NP-hard, with complexity growing exponentially with treewidth), then provides an algorithm with a runtime that is exponential in treewidth, showing this lower bound is tight and treewidth is the **intrinsic parameter** characterizing the problem's complexity. On the algorithmic side, MASE is rewritten as a zero-sum meta-game between a "Coordinator" and a "Deviator," solved using FTPL (Follow the Perturbed Leader) for no-regret learning, utilizing **tree decomposition** of the utility dependency graph to perform dynamic programming, compressing the exponentially large strategy space into a tractable form.

### Key Designs

**1. MASE: Trading Strict Immunity for Existence**

Strong equilibria generally do not exist because they require "for every coalition deviation, at least one member does not benefit"—a hard constraint where multiple coalition constraints often conflict. The design motivation for MASE is to bypass this non-existence: relax hard constraints into a soft objective to **minimize** the worst-case average deviation gain. Since the objective function is continuous on the compact simplex $\Delta_A$, a minimum must exist by the Weierstrass Theorem, making MASE well-defined for any game.

A subtle but crucial design choice: coalition gains are measured by the **average** ($\frac{1}{|S|}\sum_{i\in S}$) rather than "requiring every member to benefit." Compared to Strong NE, which asks "if there exists a deviation where everyone strictly improves," $\epsilon$-MASE focuses on "how much the coalition can improve on average." From another perspective, this is equivalent to allowing coalition members to **redistribute gains internally**, then seeing how much incentive remains to deviate—which makes MASE both mathematically well-defined and economically meaningful. The paper also notes the algorithm generalizes to any weighted average within coalitions.

**2. Utility Dependency Graph: Translating Game Complexity to Graph Structure**

To analyze the complexity of MASE, one must quantify the interaction complexity between players. Ours defines the **Utility Dependency Graph** $G=(V,E)$: vertices $V=[N]$ are players; for each player $k$, let $N(k)$ be the set of players affecting $U_k$, then form a clique within $N(k)$. That is, $E = \bigcup_{k\in[N]}\{(i,j) \mid i,j\in N(k), i\ne j\}$.

Note this is **different** from the graph in "graphical games": in graphical games, $i$ and $j$ are connected if and only if one directly affects the other's utility; here, $i$ and $j$ are connected if they **jointly affect some third party $k$**, even if $i$ and $j$ do not affect each other. This definition is elegant because it captures the coupling structure needed to evaluate joint coalition deviations (enumerating $\hat a_S \in A_S$), directly mapping to the treewidth lower bound.

**3. Treewidth Lower Bound: The Fundamental Barrier**

Using the Utility Dependency Graph, the paper pins the difficulty of MASE precisely to its **treewidth $\mathrm{tw}(G)$** (a measure of how close a graph is to a tree: trees have treewidth 1, cliques have $N-1$). Two hardness results:

- **NP-hardness (Thm 4.1)**: Even when $S$ contains only singletons (coalition size 1) and $1/\epsilon$ is polynomial in the number of players, computing $\epsilon$-MASE is NP-hard. This stands in stark contrast to CCE—where finding a feasible point with deviation gap $\le 0$ is polynomial-time solvable—whereas MASE requires **minimizing** this gap, which is inherently harder. Combined with the LP characterization in the appendix, the problem is NP-complete.
- **Fine-grained Lower Bound (Thm 4.3)**: Under the Strong Exponential Time Hypothesis (SETH), MASE cannot be computed in $O^*\big((A-\zeta)^{\mathrm{tw}(G)}\big)$ time (for any $\zeta>0$, where $A$ is the max action set size). Thus, the exponential dependence on treewidth is **unavoidable**. The approximate version ($\frac{1}{9N^2}$-approximation) shares the same lower bound under the additional assumption $\mathsf{BPP}=\mathsf{P}$ (as the reduction requires derandomization for sampling joint actions).

Intuitively: high treewidth means each player's utility is intertwined with many others, making even a single evaluation of coalition deviation (enumerating $\hat a_S \in A_S$) expensive. When treewidth is 0 (each player's utility depends only on their own action), MASE is trivially solvable.

**4. Coordinator–Deviator Meta-Game + Tree Decomposition DP: Complexity-Matching Algorithm**

The positive result is an algorithm with runtime $O\big(T\cdot|S|\cdot|T|\cdot A^{\mathrm{tw}(G)+1}\big)$, where the exponential factor aligns perfectly with the lower bound. Three pillars:

*Compact Representation (Thm 5.1)*: While MASE lives in a space of size $|A|$ (exponentially large in $N$), there always exists an $\epsilon$-MASE representable as a linear combination of $\sum_{S\in S}|S|\cdot A^{\mathrm{tw}(G)+1}$ pure strategies—making computation feasible via sparse representation.

*Zero-Sum Meta-Game Reconstruction*: MASE is formulated as a zero-sum game between a Coordinator (choosing $\pi\in\Delta_A$ to minimize coalition gain) and a Deviator (choosing a deviation distribution $\mu$ to maximize it), $\min_\pi\max_\mu F(\pi,\mu)$, where $F$ is bilinear in $(\pi,\mu)$. Both parties use **FTPL (Follow the Perturbed Leader)** for no-regret updates: each step's decision is a **pure strategy** (allowing compact representation), with exponential noise $\mathrm{Exp}(\eta)$ added for regularization to stabilize iterations.

*Dynamic Programming on Tree Decomposition*: The core of updating $\pi$ is selecting a pure joint action $a^{(t+1)}$ that minimizes the objective. The challenge is overlapping dependency sets $N(i), N(j)$, where local optima must be globally consistent. The solution introduces a **tree decomposition** $T=\{B_1,\dots,B_K\}$ of the utility dependency graph (each bag $B_k\subseteq V$ covers vertices/edges, with the running intersection property). Since each $N(i)$ is contained within some bag, a DP vector $d^{(t+1)}(B,a_B)$ (local objective + child optima, enforced consistency on overlaps) can be maintained bottom-up, followed by top-down reconstruction of the global optimal joint action. This compresses "optimization over an exponential space" into "enumerating $A^{|B|}\le A^{\mathrm{tw}(G)+1}$ for each bag." FTPL achieves $O(\sqrt{T})$ regret (with $\eta=1/\sqrt{T}$), and the average strategy converges to MASE, providing a finite-time convergence guarantee (Thm 5.4).

### Loss Function / Training Strategy

The algorithm essentially solves the zero-sum meta-game $\min_\pi\max_\mu F(\pi,\mu)$ using no-regret learning, where the "objective" is the duality gap of the meta-game. Both the Coordinator and Deviator run FTPL with step size $\eta=1/\sqrt{T}$, each achieving $O(\sqrt T)$ regret (Thm 5.2/5.3). Their combined average strategy $\bar\pi=\frac1T\sum_t\pi^{(t)}$ is an $O\!\big(\frac{|T|\,\mathrm{tw}(G)\log A}{\sqrt T}\big)$-MASE. Noise $n^{(t+1)}\sim\mathrm{Exp}(\eta)$ acts as OMD-style regularization, controlling the expected movement between iterations to ensure stability.

## Key Experimental Results

Experiments compare MASE with several no-regret learning baselines (FTRL, Hedge, Independent FTPL, OMD) across three metrics, using ground-truth MASE obtained via Linear Programming (LP):

- **Exploitability (Unilateral)**: Max gain a single player can achieve by deviating; $\le 0$ indicates NE/CE.
- **Coalition Exploitability**: Max average gain when coalitions deviate simultaneously ($S$ includes all non-empty subsets).
- **Social Welfare**: Sum of utilities of all players.

### Main Results (Classic Small Games)

| Game | Method | Converges To | Social Welfare | Coalition Robustness |
|------|------|--------|---------|-----------|
| Prisoner's Dilemma | Baselines (NE/CCE) | (D,D) | 0.4 | Vulnerable to Coalitions |
| Prisoner's Dilemma | MASE | (C,D)/(D,C) 0.5 each | **1.0** | More Robust |
| Stag Hunt | Baselines | Inferior NE | Lower | Vulnerable to Deviation |
| Stag Hunt | MASE | **Superior NE** | Higher | More Robust |

Key phenomenon: In the Prisoner's Dilemma, baselines converge to the unique NE/CCE (D,D), while MASE utilizes correlated strategies to improve social welfare from 0.4 to 1.0. In the Stag Hunt game with two NEs, baselines converge to the inferior one, whereas MASE selects the superior one. MASE maintains unilateral exploitability similar to baselines but is significantly more robust against multilateral deviations.

### Analysis Experiments (Exploitability–Social Welfare Trade-off & Large Games)

| Configuration | Key Findings |
|------|---------|
| Trade-off Frontier (Prisoner's Dilemma) | Allowing exploitability to rise from 0.0 to 0.1 increases social welfare from 0.4 to 1.0; Pareto frontier is efficiently computed via weighted objective (6.1) |
| Trade-off Frontier (Stag Hunt) | Since the existing NE already maximizes welfare, welfare remains optimal for all weights $w$ |
| Random Polymatrix Games | As $N$, $A$, and per-player interactions $c$ increase, standard no-regret algorithms' coalition exploitability **continuously rises**; larger $c$ corresponds to higher treewidth |

### Key Findings
- Equilibria reached by classic no-regret algorithms become **increasingly vulnerable** to coalition deviations as game scale increases, highlighting the necessity of explicitly minimizing coalition exploitability.
- A computable Pareto trade-off exists between exploitability and social welfare: using weighted objective (6.1) (individual rationality weight $w$ + grand coalition weight $1-w$) allows scanning $w$ to find the frontier, adjusting "how much stability to sacrifice for welfare."
- The correspondence between difficulty and treewidth is empirically validated: larger average interaction counts $c$ (approximating treewidth) lead to higher coalition exploitability in baselines.

## Highlights & Insights
- **Turning Non-existence into Existence**: The deadlock of Strong Equilibria non-existence is resolved by "minimizing worst-case average gain"—Weierstrass ensures a minimum exists, providing a simple yet powerful pivot for the entire paper.
- **Clever Utility Dependency Graph**: It doesn't use "who affects whom" (graphical games) but "who jointly affects a third party," perfectly aligning with the coupling structure of coalition deviations, which makes treewidth a precise complexity parameter.
- **Mutual Support of Lower Bounds and Algorithms**: The NP-hard + SETH lower bound shows exponential dependence on treewidth is unavoidable, while the algorithm runtime is exactly $A^{\mathrm{tw}(G)+1}$, aligning theoretical hardness with reachable upper bounds—a model for fixed-parameter complexity characterization.
- **Transferable Methodology**: The paradigm of "hard constraint solution non-existence → soft objective minimization + parameterized algorithm via structural parameters (treewidth)" can be transferred to other equilibrium/stability problems plagued by non-existence.

## Limitations & Future Work
- **Uniform Averaging within Coalitions**: MASE defines coalition gain as the uniform average of member gains. The authors suggest extending to richer objectives like "minimizing the minimum gain within a coalition" and characterizing their bounds.
- **Reliance on Given Tree Decomposition**: The algorithm assumes the tree decomposition of the utility dependency graph is **given**; complexity is analyzed relative to this decomposition. In reality, finding an optimal tree decomposition is NP-hard.
- **Succinct Representation Assumption**: Results are established for succinct games (polymatrix, congestion games, etc.) where utilities can be encoded in polynomial bits. In general games, MASE degrades to an LP with size exponential in $N$.
- **Limited Experimental Scale**: Primarily validated on Prisoner's Dilemma, Stag Hunt, and random polymatrix games; lacks testing on larger-scale or real-world application scenarios.

## Related Work & Insights
- **vs Strong NE / Coalition-Proof Equilibrium**: These require strict immunity to all coalition deviations, resulting in non-existence in general games; MASE shifts to minimizing deviation incentive, ensuring universal existence and computability.
- **vs CCE / No-Regret Learning**: CCE seeks a feasible point where "deviation gap $\le 0$," which is poly-time computable; MASE requires **minimizing** that gap, making it inherently harder (NP-complete). Ours also empirically finds that standard no-regret algorithms (FTRL/Hedge/OMD/Independent FTPL) converge to equilibria that are increasingly vulnerable to coalitions.
- **vs Graphical Games**: The edge rules for Utility Dependency Graphs differ—graphical games connect "direct influence," whereas this paper connects "joint influence on a third party," a distinction that makes treewidth the exact parameter for MASE complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes the universally existent MASE concept and provides a tight fixed-parameter complexity characterization using treewidth; both definition and analysis are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Clear validation on classic and polymatrix games, though scale and real-world coverage are limited (primarily a theoretical paper).
- Writing Quality: ⭐⭐⭐⭐⭐ Progresses logically from the non-existence lemma to definitions, lower bounds, and algorithms; the logical chain is complete and self-consistent.
- Value: ⭐⭐⭐⭐⭐ Provides the first solution concept for "coalition deviation" that balances existence and computability, precisely characterizing its complexity boundaries—foundational for algorithmic game theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Differentially Private Equilibrium Finding in Polymatrix Games](differentially_private_equilibrium_finding_in_polymatrix_games.md)
- [\[ICLR 2026\] Quantum Machine Learning Advantages Beyond Hardness of Evaluation](quantum_machine_learning_advantages_beyond_hardness_of_evaluation.md)
- [\[NeurIPS 2025\] The Parameterized Complexity of Computing the VC-Dimension](../../NeurIPS2025/learning_theory/the_parameterized_complexity_of_computing_the_vc-dimension.md)
- [\[ICLR 2026\] Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond](bounds_of_chain-of-thought_robustness_reasoning_steps_embed_norms_and_beyond.md)

</div>

<!-- RELATED:END -->
