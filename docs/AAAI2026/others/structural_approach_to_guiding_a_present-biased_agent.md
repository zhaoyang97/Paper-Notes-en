---
title: >-
  [Paper Note] Structural Approach to Guiding a Present-Biased Agent
description: >-
  [AAAI 2026][Present Bias] This paper systematically investigates the parameterized complexity of the T-path-Editing problem within the principal-agent extension of the Kleinberg-Oren model. It presents FPT algorithms parameterized by treewidth and path-cost diversity, establishes tight hardness results, and comprehensively characterizes the tractability-intractability boundary for guiding a present-biased agent to complete critical tasks.
tags:
  - AAAI 2026
  - Present Bias
  - Parameterized Complexity
  - Treewidth
  - Task Graph Modification
  - Principal-Agent Problem
date: 2026-05-08
content_hash: dc04936f85c08349
---

# Structural Approach to Guiding a Present-Biased Agent

**Conference**: AAAI 2026
**arXiv**: [2601.07763](https://arxiv.org/abs/2601.07763)
**Code**: None
**Area**: Other
**Keywords**: Present Bias, Parameterized Complexity, Treewidth, Task Graph Modification, Principal-Agent Problem

## TL;DR

This paper systematically investigates the parameterized complexity of the T-path-Editing problem within the principal-agent extension of the Kleinberg-Oren model. It presents FPT algorithms parameterized by treewidth and path-cost diversity, establishes tight hardness results, and comprehensively characterizes the tractability-intractability boundary for guiding a present-biased agent to complete critical tasks.

## Background & Motivation

1. **State of the Field**: Present bias is a classic phenomenon in behavioral economics—individuals tend to overweight immediate outcomes while discounting future rewards, leading to procrastination and suboptimal abandonment of long-term beneficial plans. Kleinberg and Oren (2014) proposed a graph-theoretic model to formalize this behavior: a present-biased agent navigates a task DAG, making locally optimal decisions at each step based on discounted future costs, potentially deviating repeatedly from the intended plan.

2. **Limitations of Prior Work**: Belova et al. (2024) introduced a two-agent extension in which a fully rational principal attempts to guide the present-biased agent to complete a set of critical tasks via graph modifications (edge deletions or additions). However, that work only preliminarily analyzed the T-path-Deletion and T-path-Addition subproblems, leaving several open questions unresolved.

3. **Root Cause**: T-path-Editing is NP-hard in the general case, yet in practical applications (e.g., educational curricula, workflows, digital assistants), task graphs typically exhibit structural properties such as limited hierarchical depth or a small number of bottleneck nodes. The key challenge is how to exploit these structural properties to obtain efficient algorithms.

4. **Paper Goals**: To comprehensively characterize the parameterized complexity landscape of T-path-Editing with respect to various graph parameters, including treewidth, vertex cover, feedback vertex set, path length, and treedepth.

5. **Starting Point**: The paper adopts a parameterized complexity-theoretic perspective, decomposing the hardness of the problem according to structural parameters and identifying precise tractability boundaries.

6. **Core Idea**: By combining dynamic programming over tree decompositions with constraints on path-cost diversity, FPT algorithms are achieved under conditions of "simple graph structure + limited cost diversity," while the necessity of these conditions is formally established.

## Method

### Overall Architecture

The core of the paper is a parameterized algorithmic framework. Given a time-inconsistent planning model $M = (G, w, s, t, \beta, r)$—a DAG $G$, edge weights $w$, source $s$, target $t$, bias factor $\beta \in (0,1]$, and reward $r$—an agent at vertex $v$ evaluates all $(v,t)$-paths and selects the one minimizing the perceived cost $\zeta_M(P) = w(e_1) + \beta \cdot \sum_{i=2}^k w(e_i)$; if $\zeta_M(P) > \beta \cdot r$, the agent abandons the task. The goal is to find a minimum number of graph edits (deletions and additions) such that the agent follows a path containing all critical edges $T$ and reaches $t$.

### Key Designs

1. **XP Algorithm Parameterized by Vertex Cover (Theorem 2)**:
   - **Function**: Shows that T-path-Editing is solvable in polynomial time when the vertex cover number $\text{vc}$ of $G+A$ is constant.
   - **Mechanism**: The key observation is that paths in a DAG with bounded vertex cover have length at most $2\cdot\text{vc}-1$. The algorithm enumerates all possible agent paths $P$ (at most $m^{O(\text{vc})}$) and the shortest $(v,t)$-paths $R_v$ for each vertex in the critical vertex set $S = V(P) \cup C$. The inner layer reduces to a matching problem: edges to be deleted are classified into those between critical vertices that must be removed due to distance or preference violations ($X_1$), and edge pairs on non-critical vertices ($\mathcal{X}_v$), the latter of which reduces to minimum vertex cover in a bipartite graph (solvable in polynomial time via König's theorem). Total runtime is $m^{O(\text{vc}^2)}$.
   - **Design Motivation**: Vertex cover is one of the most restrictive parameters, guaranteeing bounded path lengths and making exhaustive enumeration feasible. It serves as the most intuitive starting point among all structured algorithms.

2. **FPT Algorithm Parameterized by Treewidth (Theorem 1, Main Result)**:
   - **Function**: The central positive result, establishing that T-path-Editing is fixed-parameter tractable when parameterized by treewidth $\text{tw}$ and path-cost diversity $|L|$.
   - **Mechanism**: The algorithm runs in time $|L|^{O(\text{tw})} \cdot m^{O(1)}$, where $L = |\bigcup_{v \in V(G)} \{w(P) \mid P \text{ is a } (v,t)\text{-path in } G+A\}|$ denotes the number of distinct path-cost values. The key distinction from Theorem 2 is that the algorithm avoids directly enumerating agent paths (which would require lexicographic comparison of arbitrary edges) and instead only compares edges in $G+A$ against edges in $T$. Dynamic programming over the tree decomposition maintains, for each bag, a guess of the distance from each vertex to $t$, enforcing consistency of distances and correctness of the agent's path preference constraints.
   - **Design Motivation**: Treewidth captures the "tree-likeness" of a graph, and many practical planning graphs (hierarchical structures, modular designs) exhibit low treewidth. Combining this with the bounded-$|L|$ condition substantially broadens the tractable regime.

3. **W[1]-Hardness and Para-NP Hardness Results**:
   - **Function**: Establishes lower bounds on intractability, proving that the parameter choices in the positive results are necessary.
   - **Mechanism**: Through parameterized gap reductions from the Modified $k$-Sum problem ($\text{W}[1]$-complete), the paper proves: (a) T-path-Editing is $\text{W}[1]$-hard parameterized by the number of vertices $n$; (b) it remains $\text{W}[1]$-hard with respect to the combined parameter path length $p$ + vertex cover $\text{vc}$. The reduction constructs a carefully engineered gadget graph in which the cost of the agent's chosen path precisely encodes a solution to the $k$-Sum problem. When $\text{tw}=2$, the problem is already NP-hard (as the reduction yields a series-parallel graph), while at $\text{tw}=1$ the problem is trivial (paths on trees are unique).
   - **Design Motivation**: These hardness results explain why both graph-structural parameters and cost diversity must be simultaneously constrained—bounding any single parameter alone is insufficient to overcome the inherent difficulty of the problem.

### Loss & Training

Not applicable—this is a purely theoretical and algorithmic paper. The core technical tools are: parameterized algorithm design (DP over tree decompositions), parameterized complexity theory ($\text{W}$-hierarchy, Para-NP hardness), and the Kleinberg-Oren model from behavioral economics.

## Key Experimental Results

### Main Results

This is a theoretical paper; the core results are presented as theorems:

| Parameter | Algorithmic Complexity | Type |
|-----------|----------------------|------|
| tw + $|L|$ | $|L|^{O(\text{tw})} \cdot m^{O(1)}$ | FPT |
| vc | $m^{O(\text{vc}^2)}$ | XP |
| fvs | $m^{O(\text{fvs}^2)}$ | XP |
| td | $m^{O(\text{td} \cdot 2^{\text{td}})}$ | XP |
| fvs + $|W|$ | FPT | FPT |
| td + $|W|$ | FPT | FPT |
| $n$ (parallel edges allowed) | W[1]-hard | Intractable |
| $p$ + vc | W[1]-hard | Intractable |
| tw = 2 | NP-hard | Intractable |

### Ablation Study

Not applicable. However, Lemma 1 establishes relationships among parameters: $|L| \leq \min\{(p+1)^{|W|}, 1 + p \cdot \max W\}$ and $|L| \leq \min\{m^{p+1}, m^{2\cdot\text{vc}}, m^{2^{\text{td}+1}}, m^{2\cdot\text{fvs}+1}\}$, demonstrating that Theorem 1 subsumes the XP algorithms for the cases of bounded vertex cover, feedback vertex set, and treedepth.

### Key Findings

- The problem is trivial at $\text{tw}=1$ (trees) but already NP-hard at $\text{tw}=2$, representing the tightest structural boundary.
- Constraining any single parameter alone (e.g., only vc or only $p$) is insufficient to obtain FPT; both structural parameters and cost diversity must be simultaneously bounded.
- The key algorithmic breakthrough enabling the FPT result is avoiding enumeration of agent paths and instead performing comparisons only against edges in $T$.
- Planning graphs in practical applications typically satisfy the conditions of low treewidth and limited cost variety (e.g., curriculum design, multi-stage workflows).

## Highlights & Insights

- **Complete Parameterized Complexity Landscape**: The paper systematically explores all natural combinations of graph parameters, precisely delineating the tractability/intractability boundary and resolving the open questions left by Belova et al. Such comprehensiveness is relatively rare in parameterized algorithm papers.
- **Intersection of Behavioral Economics and Algorithm Design**: Incorporating agents' irrational behavior (present bias) into algorithmic design reflects the trend of human-AI collaborative design in intelligent systems. The principal-agent framework naturally models the problem of "how a system designer guides an imperfect agent."
- **The Bob and Alice Example** is highly intuitive: the scenario of an AI engineer developing an LLM agent is used to explain the abstract graph-theoretic model, lending a strong sense of practical relevance to the theoretical results. This perspective connecting theoretical problems to LLM agent deployment is notably timely.
- The reduction technique of using the Modified $k$-Sum problem to encode agent choice behavior is elegant—by carefully setting $\beta$ and $r$, the agent's preferences are made to correspond precisely to solutions of a subset-sum problem.

## Limitations & Future Work

- The runtime $|L|^{O(\text{tw})}$ in the main theorem may be large in practice due to the potential magnitude of $|L|$; tighter analysis is needed to sharpen this bound.
- The model assumes the agent's behavior is fully deterministic and known (with fixed $\beta$ and lexicographic ordering); in practice, agent behavior may be stochastic.
- Open question: Does an algorithm with runtime $f(\text{tw}) \cdot m^{g(p)}$ exist? That is, can the dependence on $|L|$ be reduced to a function of path length alone?
- The Kleinberg-Oren model does not capture all known psychological aspects of time-inconsistent behavior.

## Related Work & Insights

- **vs. Kleinberg-Oren (2014)**: The original model studies only the behavior of a single agent without principal intervention. The present paper builds on the two-agent setting introduced by Belova et al. and provides a comprehensive algorithmic characterization.
- **vs. Belova et al. (2024)**: The prior work introduced the T-path-Deletion and T-path-Addition problems but provided only preliminary analysis. This paper unifies them as T-path-Editing and delivers a complete complexity landscape, resolving the open questions left therein.
- **vs. Classical Graph Modification Problems**: T-path-Editing differs from standard edge deletion/addition problems in that the constraints concern agent behavior (requiring simulation of the agent's decision process on the modified graph) rather than structural graph properties (such as connectivity), rendering the problem substantially more complex.

## Rating

- Novelty: ⭐⭐⭐⭐ — A novel perspective, systematically applying parameterized complexity to the agent-guidance problem in a behavioral economics model.
- Experimental Thoroughness: ⭐⭐⭐ — A purely theoretical work; theorem proofs are rigorous but no empirical validation of algorithmic efficiency is provided.
- Writing Quality: ⭐⭐⭐⭐⭐ — The motivating example is elegantly designed, the complexity landscape is presented with clarity, and the proof structure is well-organized.
- Value: ⭐⭐⭐⭐ — Makes theoretical contributions to the algorithmic game theory and parameterized algorithms communities, though practical application pathways remain unclear.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [\[AAAI 2026\] Agent-SAMA: State-Aware Mobile Assistant](agent-sama_state-aware_mobile_assistant.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)

<!-- RELATED:END -->
