---
title: >-
  [Paper Note] Good-for-MDP State Reduction for Stochastic LTL Planning
description: >-
  [AAAI 2026][Reinforcement Learning][LTL planning] This paper proposes a novel Good-for-MDP (GFM) automaton state reduction technique that significantly reduces automaton state counts via a GFM→DBA→DCA→GFG minimization→0/…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "LTL planning"
  - "Markov decision processes"
  - "automaton state reduction"
  - "Good-for-MDP"
  - "formal methods"
date: 2026-05-08
content_hash: ed1442605b4c773b
---

# Good-for-MDP State Reduction for Stochastic LTL Planning

**Conference**: AAAI 2026
**arXiv**: [2511.09073](https://arxiv.org/abs/2511.09073)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: LTL planning, Markov decision processes, automaton state reduction, Good-for-MDP, formal methods

## TL;DR

This paper proposes a novel Good-for-MDP (GFM) automaton state reduction technique that significantly reduces automaton state counts via a GFM→DBA→DCA→GFG minimization→0/1-PA transformation pipeline. Additionally, for formulas of the form $\textsf{GF}\varphi$ where $\varphi$ is a co-safety formula, a direct singly-exponential construction is provided, achieving an exponential reduction in state count compared to the general doubly-exponential construction.

## Background & Motivation

**Core pipeline for LTL temporal logic planning**: In stochastic planning, the environment is modeled as a Markov decision process (MDP) $\mathcal{M}$, and the task objective is described by a linear temporal logic (LTL) formula $\varphi$. The core solution pipeline is:
1. Translate the LTL formula into an automaton $\mathcal{A}$
2. Construct the product MDP $\mathcal{M} \times \mathcal{A}$
3. Identify accepting maximal end components (MECs) in the product MDP and synthesize a policy that maximizes the probability of reaching them

**The role of GFM automata**:
- **Deterministic automata (DRA)**: Can be used directly in product construction, but translating LTL to DRA involves complex techniques such as the Safra construction
- **GFM automata**: A restricted class of nondeterministic Büchi automata (NBA) whose nondeterminism can be resolved by the agent during policy synthesis without affecting the optimal satisfaction probability. In practice, GFM automata are substantially smaller than DRAs and serve as a core component in state-of-the-art tools such as PRISM

**Core bottleneck — automaton size**: The primary factor limiting the scalability of this planning approach is the number of states in the GFM automaton. More states yield a larger product MDP and slower policy synthesis.

**Research goals**:
1. Propose a general GFM state-space reduction technique
2. Provide a more efficient direct construction for the common class of $\textsf{GF}\varphi$ formulas

## Method

### Overall Architecture

The state reduction pipeline consists of four steps (see Figure 1):

$$\mathcal{A}_\text{GFM} \xrightarrow{\text{Poly}} \mathcal{A}_\text{DBA} \xrightarrow{\text{Const}} \mathcal{A}_\text{DCA} \xrightarrow{\text{Poly}} \mathcal{A}_\text{GFG-Min} \xrightarrow{\text{Poly}} \mathcal{A}_\text{PA}$$

All steps run in polynomial or constant time.

### Key Designs

#### 1. **Step 1: GFM→DBA (Encoding Nondeterminism into MDP Actions)**

Core Idea: The nondeterminism of a GFM automaton can be resolved by the agent — nondeterministic choices $[k]$ are encoded into the MDP's action space, rendering the automaton deterministic.

- **Modified MDP**: The action space is extended from $\text{Act}$ to $\text{Act} \times [k]$, and the label alphabet from $\Sigma$ to $\Sigma \times [k]$
- **Modified automaton**: The transition function becomes deterministic over the extended alphabet: $\delta_\text{DBA}(q, \langle\sigma, i\rangle) = q'$, where $q'$ is the $i$-th state in $\delta(q,\sigma)$

**Key Lemma** (Lemma 1): The products $\mathcal{M} \times \mathcal{A}_\text{GFM}$ and $\mathcal{M}' \times \mathcal{A}_\text{DBA}$ are identical (same state space, equivalent optimal policies).

Design Motivation: Subsequent GFG minimization techniques require deterministic or GFG inputs. By "outsourcing" the nondeterminism of GFM to MDP actions, the automaton becomes deterministic, enabling minimization.

#### 2. **Steps 2–3: DBA→DCA→GFG Minimization**

**Step 2** (constant time): The acceptance condition ($\alpha$) of the DBA is reinterpreted as a rejection condition for the DCA — complementing the language while leaving the transition structure unchanged. This is a purely semantic operation.

**Step 3** (polynomial time): The GFG co-Büchi automaton minimization algorithm of Radi & Kupferman (2022) is applied. Key observations:
- A DCA is deterministic and therefore also GFG
- GFG co-Büchi automata can be minimized in polynomial time
- The minimized output is a minimal GFG-NCA $\mathcal{A}_\text{GFG-Min}$

**Why GFG minimization can be applied across paradigms**: GFG automaton minimization techniques (originally developed for adversarial games) had not previously been applied in the MDP setting. A key contribution of this paper is establishing a bridge between GFM (probabilistic setting) and GFG (adversarial setting) via the DBA intermediate representation.

#### 3. **Step 4: GFG-Min→0/1-PA (Probabilistic Automaton)**

Core Idea: Li et al. (2025) proved that a minimal GFG-NCA can be converted into a language-equivalent 0/1-PA by uniformly randomizing nondeterministic transitions:

$$\delta_\text{PA}(q,\sigma)(q') = \frac{1}{|\delta_\text{NBA}(q,\sigma)|}$$

**GFM property of 0/1-PA** (Theorem 2, new contribution): This paper is the first to prove that a 0/1-PA can serve as a GFM automaton. The key distinction is that in the product MDP $\mathcal{M}' \otimes \mathcal{A}_\text{PA}$, successor state selection is governed by a randomized policy (rather than agent choice), so the action space remains $\text{Act}$ (without $[k]$), yielding a smaller product.

**Main Theorem** (Theorem 4): An optimal policy for the original problem $\mathcal{M}$ achieving $\text{Psem}(\mathcal{M}, \mathcal{A}_\text{GFM})$ can be extracted from an optimal policy on $\mathcal{M}' \otimes \mathcal{A}_\text{PA}$.

#### 4. **Direct Singly-Exponential Construction for $\textsf{GF}\varphi$ (Section 6)**

$\textsf{GF}\varphi$ (where $\varphi$ is a co-safety formula) expresses repeated reachability — the property that a finite pattern recurs infinitely often — and is common in verification and reinforcement learning.

**Construction**: Given the NFA $\mathcal{N}_\varphi = (Q, q_0, \delta, F)$ corresponding to $\varphi$, a GFM automaton $\mathcal{A}$ is constructed as follows:
- **State space**: $Q_\mathcal{A} \subseteq (Q \setminus F) \cup \{q_0\}$ (accepting states removed, initial state retained)
- **Transition rules**: (1) Successors of non-accepting states are preserved; (2) Any state may reset to $q_0$
- **Acceptance condition**: A transition is marked accepting when it reaches $q_0$ and corresponds to reaching an accepting state in the NFA

Intuition: For $\textsf{GF}\varphi$, the agent may "forget" past finite trajectories at any point and restart tracking whether $\varphi$ is satisfied. The Büchi condition ensures $\varphi$ is satisfied infinitely often.

**Theorem 5**: $\mathcal{A}$ recognizes $[\textsf{GF}\varphi]$, is GFM, and has only $2^{\mathcal{O}(|\varphi|)}$ states — an exponential improvement over the general $2^{2^{\mathcal{O}(|\varphi|)}}$ construction.

### Loss & Training

This paper does not involve learning or training. Policy synthesis is performed via standard MDP reachability analysis: MECs containing accepting transitions are identified in the product MDP, and a memoryless policy maximizing the probability of reaching these MECs is computed.

## Key Experimental Results

### Main Results

GFM state reduction results across 6 LTL pattern classes (Table 1), reporting automaton state counts / runtimes:

| Pattern | Owl | Owl-Red | Slim | Slim-Red |
|---------|-----|---------|------|----------|
| TDR[6] | 64 (0.46s) | **64** (0.03s) | 65 (0.09s) | **34** (0.01s) |
| TDR[8] | 256 (0.48s) | 256 (0.69s) | 257 (0.14s) | **130** (0.05s) |
| BRP[6] | 69 (0.55s) | **17** (0.02s) | 317 (0.14s) | 65 (0.17s) |
| BRP[8] | 261 (0.86s) | **21** (0.14s) | 1277 (0.33s) | 146 (3.49s) |
| EHP | 13 (0.43s) | **9** (0.01s) | 127 (0.14s) | 54 (0.31s) |
| NU[5] | 150 (1.41s) | **56** (0.17s) | 232 (0.82s) | 159 (0.76s) |
| LFR[7] | 72 (0.92s) | 73 (2.99s) | 73 (1.57s) | **34** (1.76s) |

### Ablation Study

Direct $\textsf{GF}\varphi$ construction vs. state-of-the-art reduction methods (Table 2):

| Pattern | Owl-Red (incl. construction) | Slim-Red (incl. construction) | GFM-GF (direct construction) |
|---------|------------------------------|-------------------------------|-------------------------------|
| TDR[6] | 64 (0.49s) | 34 (0.10s) | **7** (0.08s) |
| TDR[8] | 256 (1.17s) | 130 (0.19s) | **9** (0.08s) |
| TDR[10] | 1024 (23.67s) | 514 (0.91s) | **11** (0.09s) |
| LIB[6] | **14** (0.63s) | 66 (129s) | 13 (0.09s) |
| LIB[8] | 18 (5.95s) | timeout | **17** (0.20s) |
| LIB[9] | 20 (37.78s) | timeout | **19** (0.23s) |

### Key Findings

1. **Reduced automata achieve the smallest state counts across all patterns** (bold values in Table 1), with low reduction overhead (typically <1s)
2. **BRP patterns exhibit the most significant reduction**: BRP[8] drops from 261 to 21 states (12.4× reduction), as this pattern contains a large number of redundant transitions
3. **Direct $\textsf{GF}\varphi$ construction achieves exponential improvement**: TDR[10] decreases from 1024 states (Owl-Red) to just 11 states (GFM-GF) — linear vs. exponential growth
4. **Runtime advantage is equally pronounced**: TDR[10] requires 23.67s under Owl-Red but only 0.09s under GFM-GF — a 260× speedup
5. **Slim constructions benefit more from reduction due to lack of prior optimization**: Slim-Red shows larger relative gains than Owl-Red, as Owl already incorporates advanced formula simplification
6. **Slim-Red times out on LIB patterns**: GFG minimization incurs substantial overhead on certain complex patterns (LIB[6]: 128s), whereas the direct construction completes within 0.2s

## Highlights & Insights

1. **Cross-paradigm technical bridge**: GFG minimization (from adversarial game theory) is introduced into the GFM setting (probabilistic planning) via DBA/DCA intermediate representations, enabling technology transfer between seemingly unrelated fields
2. **0/1-PA as a new class of GFM automaton**: This paper is the first to prove that probabilistic automata can replace GFM automata in MDP product construction — nondeterminism is resolved via randomization rather than agent choice — resulting in a smaller action space in the product MDP
3. **Theory and practice in tandem**: The paper provides a rigorous correctness proof chain (Theorem 2→3→4) alongside comprehensive benchmark experiments covering 8+ literature sources
4. **Elegant intuition for the $\textsf{GF}\varphi$ construction**: Repeated reachability properties can be handled by the simple strategy of "resetting to the initial state at any time" — the Büchi condition naturally guarantees infinite satisfaction

## Limitations & Future Work

1. **GFG minimization itself can be time-consuming**: Reduction times on complex patterns such as NU[6] (27.75s) and LFR[8] (39.11s) are non-negligible
2. **Restricted to Büchi/co-Büchi conditions**: GFM reduction for Rabin conditions is not experimentally validated (the paper mentions generalizability but does not implement it)
3. **Not yet integrated into mainstream tools**: The proposed reduction pipeline has not been incorporated into PRISM, Owl, or similar tools
4. **Direct $\textsf{GF}\varphi$ construction limited to co-safety subformulas**: Full LTL still requires doubly-exponential construction
5. **No end-to-end planning experiments**: Only automaton state counts and runtimes are reported; the impact on downstream MDP policy synthesis performance is not evaluated
6. **Integration with online solvers**: How reduced automata interact with online POMDP solvers is not explored

## Related Work & Insights

- **LTL-to-GFM**: Sickert et al. 2016 ("slim" construction), Hahn et al. 2020 (formal definition of GFM) → this paper applies further reduction to their outputs
- **GFG minimization**: Radi & Kupferman 2022 → this paper transfers the technique from the adversarial setting to the probabilistic setting
- **0/1-PA theory**: Li et al. 2025 (semantic determinism proof) → this paper establishes the GFM property of 0/1-PAs
- **LTL in reinforcement learning**: Jackermeier & Abate 2025 use $\textsf{GF}\varphi$ → the direct construction proposed here can directly accelerate their approach
- This work can inspire automatic compression and efficiency optimization of automaton representations in RL

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The GFM-GFG bridge and the GFM property of 0/1-PAs are both original theoretical contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive benchmarks covering 6 pattern classes from 8+ sources, but end-to-end planning experiments are absent
- Writing Quality: ⭐⭐⭐⭐ — The transformation pipeline is clearly presented, though the high density of formal definitions requires a background in automata theory
- Value: ⭐⭐⭐⭐⭐ — A foundational contribution with direct implications for LTL planning, probabilistic verification, and reinforcement learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[AAAI 2026\] Efficient Multiagent Planning via Shared Action Suggestions](efficient_multiagent_planning_via_shared_action_suggestions.md)
- [\[NeurIPS 2025\] Horizon Reduction Makes RL Scalable](../../NeurIPS2025/reinforcement_learning/horizon_reduction_makes_rl_scalable.md)
- [\[ICLR 2026\] SUSD: Structured Unsupervised Skill Discovery through State Factorization](../../ICLR2026/reinforcement_learning/susd_structured_unsupervised_skill_discovery_through_state_factorization.md)
- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)

</div>

<!-- RELATED:END -->
