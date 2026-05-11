---
title: >-
  [Paper Note] Deviation Dynamics in Cardinal Hedonic Games
description: >-
  [AAAI 2026][Hedonic games] This paper establishes meta-theorems for deviation dynamics in cardinal hedonic games, showing that the computational complexity of determining whether deviation dynamics may or must converge c…
tags:
  - "AAAI 2026"
  - "Hedonic games"
  - "deviation dynamics"
  - "stable partitions"
  - "computational complexity"
  - "individual rationality"
date: 2026-05-08
content_hash: 1871d33237f21e48
---

# Deviation Dynamics in Cardinal Hedonic Games

**Conference**: AAAI 2026
**arXiv**: [2511.11531](https://arxiv.org/abs/2511.11531)
**Code**: None
**Area**: Other
**Keywords**: Hedonic games, deviation dynamics, stable partitions, computational complexity, individual rationality

## TL;DR

This paper establishes meta-theorems for deviation dynamics in cardinal hedonic games, showing that the computational complexity of determining whether deviation dynamics may or must converge can be derived directly from instances in which no stable outcome exists. The paper further proposes methods for finding individually rational and contractually individually stable partitions via deviation dynamics in additively separable hedonic games.

## Background & Motivation

**Background**: Hedonic games are a classical model of coalition formation in which each agent holds preferences over different groupings. Cardinal hedonic games form an important subclass, encompassing variants such as additively separable hedonic games (ASHGs), fractional hedonic games (FHGs), and modified fractional hedonic games (MFHGs). Computing stable partitions in these games is a central problem in algorithmic game theory.

**Limitations of Prior Work**: (1) In many hedonic games, stable partitions may not exist—certain instances admit no grouping that achieves stability. (2) These non-existence instances have typically been used to establish NP-hardness of computing stable partitions. (3) A systematic theoretical framework for analyzing whether deviation dynamics—wherein agents sequentially make utility-improving deviations—converge to a stable partition has been lacking.

**Key Challenge**: Intuitively, there should be a deep connection between the non-existence of stable partitions and the non-convergence of deviation dynamics, yet this connection had not previously been formalized.

**Goal**: (1) Formalize the relationship between "no stable partition exists" and "deviation dynamics do not converge" as meta-theorems. (2) Analyze the convergence speed of deviation dynamics under different initialization conditions. (3) Exploit deviation dynamics as an algorithmic tool for finding specific types of stable partitions.

**Key Insight**: The authors propose systematically formalizing the convergence analysis of deviation dynamics—given a class of hedonic games and a stability concept, if one can construct instances admitting no stable partition, complexity results regarding the convergence of deviation dynamics follow automatically.

**Core Idea**: Establish meta-theorems mapping from "No-instances" (instances with no stable partition) to the complexity of deciding convergence of deviation dynamics, providing a unified treatment of nearly all single-agent deviation-based stability concepts in ASHGs, FHGs, and MFHGs.

## Method

### Overall Architecture

The analytical framework consists of two components: (1) the meta-theorem component, which establishes reductions from the non-existence of stable partitions to the complexity of deciding convergence of deviation dynamics; and (2) the algorithmic component, which designs deviation-dynamics-based algorithms for finding specific types of stable partitions.

### Key Designs

1. **Meta-Theorems**:

    - **Function**: Provide general-purpose tools for the computational complexity analysis of deviation dynamics.
    - **Mechanism**: The core insight is that if deciding the existence of a stable partition (under some stability concept) is NP-hard for a given class of hedonic games, then deciding whether deviation dynamics *possibly converge* (there exists some deviation order under which dynamics converge) and whether they *necessarily converge* (all deviation orders converge) is also hard. Concretely, No-instances can be used to construct cycles in deviation dynamics such that certain deviation orders loop indefinitely rather than converge.
    - **Design Motivation**: Prior work required separate complexity proofs for each (game class, stability concept) pair. The proposed meta-theorems cover multiple combinations at once, substantially reducing the burden of complexity analysis.

2. **CIS Deviation Dynamics in ASHGs**:

    - **Function**: Use deviation dynamics as an algorithmic tool to find contractually individually stable (CIS) partitions.
    - **Mechanism**: Starting from the singleton partition (each agent in its own coalition), CIS deviations are performed—an agent deviates only if doing so does not harm the members of its current or target coalition. Two results are established: (a) *possible convergence*—there exists a deviation sequence that converges to a CIS partition in a linear number of steps ($O(n)$); (b) *worst case*—certain deviation sequences may require an exponential number of steps to converge.
    - **Design Motivation**: CIS is a relatively weak yet practically meaningful stability concept that respects the contractual spirit of not harming others. Finding CIS partitions has practical value, and initializing from the singleton partition is a natural conservative starting point.

3. **Unified Treatment Across Game Variants**:

    - **Function**: Extend results to the three main variants: ASHGs, FHGs, and MFHGs.
    - **Mechanism**: The conditions of the meta-theorems are sufficiently general to apply to all single-agent deviation-based stability concepts (e.g., individual stability (IS), Nash stability (NS), contractual individual stability (CIS), individual rationality (IR)). By verifying that each game variant satisfies the prerequisites, the corresponding complexity results follow directly.
    - **Design Motivation**: Results for different game variants and stability concepts were previously scattered across the literature without a unified perspective. The proposed framework offers a "prove once, apply everywhere" paradigm.

### Loss & Training

Not applicable. This is a theoretical computer science paper involving complexity proofs and algorithm design; no machine learning training is involved.

## Key Experimental Results

### Main Results

This is a theoretical paper whose primary contributions are mathematical theorems and complexity proofs rather than empirical results.

| Game Type | Stability Concept | Possible Convergence | Necessary Convergence | Notes |
|-----------|------------------|---------------------|----------------------|-------|
| ASHG | NS | NP-hard | coNP-hard | Derived via meta-theorem |
| ASHG | IS | NP-hard | coNP-hard | Derived via meta-theorem |
| FHG | NS | NP-hard | coNP-hard | Derived via meta-theorem |
| MFHG | NS | NP-hard | coNP-hard | Derived via meta-theorem |
| ASHG | CIS | P (linear steps) | Exp worst-case | Algorithmic design |

### Ablation Study

| Analysis Dimension | Result | Notes |
|-------------------|--------|-------|
| CIS from singleton | Possible convergence in $O(n)$ steps | Efficient deviation sequences exist |
| CIS worst case | Potentially exponential steps | Certain deviation orders are highly inefficient |
| IR deviation dynamics | Convergence in finite steps | IR is a weaker stability concept |

### Key Findings

- The meta-theorems reveal a deep connection between the existence of stable partitions and the convergence of deviation dynamics—constructions witnessing non-existence can "automatically" yield evidence of non-convergence in dynamics.
- CIS deviation dynamics exhibit an interesting duality: linear-step convergence in the best case (highly efficient) versus exponential steps in the worst case (highly inefficient), underscoring the critical importance of deviation order selection.
- The meta-theorems apply broadly, covering nearly all common stability concepts appearing in the literature.

## Highlights & Insights

- The **unifying power of the meta-theorems** is the paper's most significant contribution. Consolidating complexity results that previously required individual proofs into a single framework constitutes both an elegant theoretical simplification and a ready-made tool for analyzing future variants.
- The perspective of **deviation dynamics as algorithms** is practically valuable—reinterpreting game-theoretic dynamics as computational processes yields a natural algorithmic paradigm for finding stable partitions.
- The linear/exponential duality of CIS deviation dynamics highlights the importance of deviation order selection strategies and may inspire new heuristic algorithms.

## Limitations & Future Work

- The proof techniques underlying the meta-theorems rely on specific reduction structures; it is unclear whether they can be extended to more general game classes (e.g., non-cardinal hedonic games).
- Whether the exponential lower bound for CIS deviation dynamics is tight remains open.
- Computational experiments at practical scales to validate the utility of deviation-dynamics-based algorithms are absent.
- Extending the meta-theorems to multi-agent deviation settings (e.g., group deviations) is a promising direction.

## Related Work & Insights

- **vs. traditional complexity analysis**: Conventional approaches prove results separately for each (game, stability) combination; the meta-theorem framework enables batch proofs, representing a significant methodological advance.
- **vs. optimization-based algorithms for stable partitions**: IP/LP methods directly solve for stable partitions, whereas deviation dynamics simulate a decentralized natural process and are better suited to distributed settings.
- **vs. game-theoretic applications in machine learning**: The partition problem in hedonic games shares structural similarities with clustering; CIS deviation dynamics can be viewed as a special form of local-search clustering algorithm.

## Rating

- Novelty: ⭐⭐⭐⭐ — The meta-theorems establish novel theoretical connections and unify scattered complexity results.
- Experimental Thoroughness: ⭐⭐⭐ — As a theoretical paper, proofs are rigorous but computational experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ — Standard theoretical paper writing with clear structure and rigorous logic.
- Value: ⭐⭐⭐⭐ — Makes an important contribution to the theoretical toolkit of the hedonic games literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Phase Transition for Opinion Dynamics with Competing Biases](a_phase_transition_for_opinion_dynamics_with_competing_biase.md)
- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](../../NeurIPS2025/others/evolutionary_prediction_games.md)
- [\[NeurIPS 2025\] Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games](../../NeurIPS2025/others/efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->
