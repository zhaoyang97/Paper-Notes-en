---
title: >-
  [Paper Note] Optimal Welfare in Noncooperative Network Formation under Attack
description: >-
  [AAAI 2026][Network formation games] In the noncooperative network formation game model proposed by Goyal et al. (WINE 2016), this paper proves that equilibrium networks created by selfish agents maintain asymptotically…
tags:
  - "AAAI 2026"
  - "Network formation games"
  - "Nash equilibrium"
  - "social welfare"
  - "attack and immunization"
  - "game theory"
  - "Price of Anarchy"
date: 2026-05-08
content_hash: 3b13305b11e8862e
---

# Optimal Welfare in Noncooperative Network Formation under Attack

**Conference**: AAAI 2026
**arXiv**: [2511.10845](https://arxiv.org/abs/2511.10845)  
**Authors**: Natan Doubez, Pascal Lenzner, Marcus Wunderlich
**Code**: Not available  
**Area**: Others (Game Theory / Network Formation)
**Keywords**: Network formation games, Nash equilibrium, social welfare, attack and immunization, game theory, Price of Anarchy

## TL;DR

In the noncooperative network formation game model proposed by Goyal et al. (WINE 2016), this paper proves that equilibrium networks created by selfish agents maintain asymptotically optimal social welfare $n^2 - O(n)$ under a broad class of attackers — including maximum disruption — called super-quadratic disruption (SQD) attackers, thereby resolving a long-standing open problem.

## Background & Motivation

### State of the Field
Communication networks (e.g., the Internet, IoT) are critical to economic activity and daily life, yet they are also targets for adversaries. Modern networks are not controlled by a single authority but are instead managed and interconnected by multiple selfish entities. As a result, decisions about interconnection and defense against attacks are decentralized. Goyal et al. (WINE 2016) proposed an elegant game-theoretic model to capture this setting: agents selfishly decide which edges to establish and whether to purchase immunization, while an attacker infects vulnerable nodes and spreads infection along vulnerable connected components.

### Limitations of Prior Work
- Goyal et al. proved that non-trivial equilibrium social welfare under maximum carnage and random attack is $n^2 - O(n^{5/3})$, but this bound is not tight.
- Analysis of the **maximum disruption attacker** (which aims to minimize post-attack social welfare) was left as an open problem.
- Only a $n^2 - O(n^{5/3})$ bound for maximum disruption was given for connected equilibria; the general case remained unsolved.
- When using a "payoff minus cost" utility function, the $n^2$ term obscures the looseness of the bound; under a pure cost function, existing bounds are far from tight.

### Core Problem
To provide tight social welfare bounds for all three attacker types, in particular to resolve the open problem of the maximum disruption attacker, and to investigate more general attacker classes.

## Method

### Game Model
- **Game instance**: $(n, C_E, C_I, \mathcal{A})$ with $n$ agents, edge cost $C_E > 1$, immunization cost $C_I > 0$, and attacker $\mathcal{A}$.
- **Strategies**: Each agent $i$ selects a set of edges $X_i$ to purchase and an immunization decision $y_i \in \{0,1\}$.
- **Attack mechanism**: The attacker selects a vulnerable node; all nodes in its vulnerable connected component (the connected component of the subgraph induced by non-immunized nodes) are infected and removed.
- **Utility function**: $u_i(\mathbf{s}) = \mathbb{E}[CC_i(\mathbf{s})] - |X_i|C_E - y_i C_I$, i.e., the expected size of the reachable component after the attack minus costs.
- **Three attacker types**: maximum carnage (maximizes number of infected nodes), random attack (attacks a random vulnerable node), maximum disruption (minimizes social welfare).

### Key Concept: $f$-Adversary and Super-Quadratic Disruption (SQD) Attackers
The paper introduces the $f$-adversary framework: the attacker's objective is defined by a function $f: \{0,\ldots,n\} \to \mathbb{R}^+$, selecting the attack to minimize $U_f(T, \mathbf{s}) = \sum_{K \in \mathcal{K}(T)} f(|K|)$ over vulnerable regions $T$. Maximum carnage corresponds to $f(x) = x$; maximum disruption corresponds to $f(x) = x^2$.

**Super-Quadratic Disruption (SQD) attackers** require $f$ to satisfy: (1) strict convexity; (2) $f(x)/x^2$ is non-decreasing. This class subsumes maximum disruption and all stronger attackers.

### Improved Bounds for Maximum Carnage and Random Attack (Theorem 1)
- Using the **block-cut decomposition** of equilibrium networks, it is shown that the number of vulnerable cut vertices on any path satisfies $p \leq 2C_E + 1$ (Lemma 1).
- The notion of a **centroid** of a connected component is introduced (Definition 2), and it is proven that every connected component has a centroid (Lemma 2).
- By immunizing centroid nodes and applying layered analysis with Jensen's inequality, a lower bound on social welfare is derived.
- A tight bound of $n^2 - O(n)$ is established.

### Core Analysis for SQD Attackers (Theorem 2)
The proof proceeds in several steps:
1. **SQD attackers prefer to split large components** (Corollaries 2–3): Strict convexity ensures attackers target vulnerable regions that sever connectivity (i.e., cut vertex positions).
2. **Attacked regions are singletons** (Lemma 4): In connected components containing immunized nodes, the attacked vulnerable region is always a single node.
3. **Structural constraints on attacked cut vertices** (Lemmas 5, 7, 9): If an attacked cut vertex exists, the total number of agents is bounded above by $(C_I + C_E + 2)(2C_I + 3C_E)$.
4. **No isolated nodes** (Lemma 13): In sufficiently large equilibria, no agent is isolated.
5. **Large equilibria are connected** (Lemma 14): Non-trivial equilibria are necessarily connected for sufficiently large $n$.
6. **Conclusion**: In connected equilibria, the attacked region is a singleton and is not a cut vertex, so the post-attack component has size $n-1$, the cost is $O(n)$, and social welfare is $n^2 - O(n)$.

### Counterintuitive Result: Tailored Attackers Can Yield Low Welfare (Theorem 3)
A specific $f$-adversary is constructed (with $f$ a hand-crafted non-convex function) such that, at $C_E = C_I = 6$, a Nash equilibrium exists in which $n-9$ agents are isolated, yielding social welfare of only $\Theta(n)$. This demonstrates that **an attacker explicitly designed to minimize social welfare cannot inflict the greatest harm** — other attackers exist that drive equilibrium welfare even lower.

## Key Experimental Results

### Table 1: Comparison of Social Welfare Bounds

| Attacker Type | Goyal et al. (2016) | This Paper |
|---|---|---|
| Maximum Carnage | $n^2 - O(n^{5/3})$ | $n^2 - O(n)$ ✓ Tight |
| Random Attack | $n^2 - O(n^{5/3})$ | $n^2 - O(n)$ ✓ Tight |
| Maximum Disruption | Open problem | $n^2 - O(n)$ ✓ Tight |
| SQD Attackers | Not studied | $n^2 - O(n)$ ✓ Tight |
| Tailored Attacker | Not studied | $\Theta(n)$ |

Note: The optimal social welfare without any attacker is $n^2 - O(n)$ (Bala & Goyal 2000); thus equilibrium welfare under SQD is asymptotically identical to the attacker-free setting.

### Table 2: Summary of Key Structural Properties

| Property | Max. Carnage / Random | SQD (incl. Max. Disruption) | Tailored Attacker |
|---|---|---|---|
| Vulnerable region is a tree | ✓ (Lemma 3) | ✓ (Lemma 3) | ✓ |
| Edge count $\leq 2n-4$ | ✓ (Lemma 3) | ✓ (Lemma 3) | ✓ |
| Large equilibria are connected | Implicit in proof | ✓ (Lemma 14) | ✗ |
| Attacked region is a singleton | ✓ | ✓ (Lemma 4) | ✗ |
| No isolated nodes | ✓ | ✓ (Lemma 13) | ✗ |
| Optimal social welfare | ✓ $n^2-O(n)$ | ✓ $n^2-O(n)$ | ✗ $\Theta(n)$ |

## Highlights & Insights

- **Resolution of an open problem**: The paper fully resolves the open problem on social welfare analysis under the maximum disruption attacker posed by Goyal et al. (WINE 2016).
- **More general attacker class**: The SQD (super-quadratic disruption) framework unifies maximum disruption and all convex attack functions growing at least as fast as $x^2$, proving asymptotically optimal equilibrium welfare across the entire class.
- **Improved bound from $O(n^{5/3})$ to $O(n)$**: Tight bounds are also established for maximum carnage and random attack, reducing the error term from $O(n^{5/3})$ to $O(n)$, consistent with the attacker-free setting.
- **Counterintuitive finding**: The attacker explicitly designed to minimize social welfare (maximum disruption) cannot inflict maximum harm; tailored attackers exist that drive equilibrium welfare down to $\Theta(n)$.
- **Elegant structural analysis**: Through block-cut decomposition, centroids, and layered analysis, the paper establishes a clean chain of structural properties of equilibrium networks (no cut vertex is attacked → no isolated nodes → large equilibria are connected → attacked region is a singleton), forming a rigorous and well-organized proof.

## Limitations & Future Work

- **Nash equilibrium only**: Stronger solution concepts (e.g., strong Nash equilibrium, coalition stability) are not considered; agent deviations are restricted to unilateral moves.
- **Parameter constraints**: Main results require $C_E, C_I > 1$; the low-cost regime is not fully covered.
- **Asymptotic results**: The $n^2 - O(n)$ welfare bound is asymptotic and does not apply to small networks ($n \leq (C_E+C_I+2)(2C_I+3C_E)$).
- **Attack model limitations**: The attacker can only infect a single node; simultaneous multi-point attacks or more complex attack strategies are not considered.
- **Deterministic propagation**: Infection spreads deterministically along vulnerable regions; probabilistic propagation models are not addressed (Chen et al. 2019 extended the random attack setting to probabilistic spread).
- **Incomplete characterization of SQD**: The full characterization of which $f$-adversary classes guarantee optimal welfare remains open; the root cause of low welfare under tailored attackers (non-convexity of $f$?) warrants further investigation.

## Related Work & Insights

- **Bala & Goyal (2000)**: Foundational work on the attacker-free setting; equilibria are empty graphs or trees with social welfare $n^2 - O(n)$. This paper shows that asymptotic welfare is preserved even under attack.
- **Goyal et al. (WINE 2016)**: Introduced the attack-and-immunization model, proved the $n^2 - O(n^{5/3})$ bound, and left the maximum disruption case as an open problem; this paper fully resolves it.
- **Friedrich et al. (SAGT 2017)**: Studied the computational complexity of best-response computation, providing polynomial-time algorithms for maximum carnage and random attack.
- **Àlvarez & Messegué (2023)**: Gave a polynomial-time best-response algorithm for maximum disruption, complementing the theoretical analysis of this paper.
- **Chen et al. (IJCAI 2019)**: Extended the model to probabilistic propagation under random attack only, showing equilibrium welfare $\Theta(n^2)$ with $O(n)$ edges.
- **Echzell et al. (IJCAI 2020)**: Network formation games with min-cut objectives; a different optimization target.
- **Berger et al. (AAAI 2025)**: Strategic network creation with greedy routing; same area but a different problem.

## Rating

- Novelty: ⭐⭐⭐⭐ — Resolves an open problem of more than 6 years, introduces the SQD attacker framework, and the counterintuitive tailored opponent result is impressive.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical work; no computational experiments or simulation validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — The theorem-lemma chain is well organized, the intuition provided in Example 1 is excellent, and the logical progression is clear throughout.
- Value: ⭐⭐⭐⭐ — Provides strong theoretical guarantees for decentralized network security: networks built by selfish agents are inherently robust, offering insights for the design of multi-agent AI systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Network Dismantling Without Handcrafted Inputs](learning_network_dismantling_without_handcrafted_inputs.md)
- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[AAAI 2026\] DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes](detonation_decoupled_torch_network-aware_training_on_interlinked_online_nodes.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

</div>

<!-- RELATED:END -->
