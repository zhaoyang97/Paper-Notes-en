---
title: >-
  [Paper Note] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables
description: >-
  [AAAI 2026][symmetry breaking] By introducing auxiliary variables to encode lexicographic order in place of large-integer encodings, this work fundamentally redesigns the VeriPB proof system, achieving order-of-magnitude speedups in both proof generation and verification for certified SAT symmetry breaking, both theoretically and empirically.
tags:
  - AAAI 2026
  - symmetry breaking
  - proof logging
  - SAT solving
  - pseudo-Boolean
  - VeriPB
  - certified solving
date: 2026-05-08
content_hash: 5678f81758369716
---

# Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables

**Conference**: AAAI 2026
**arXiv**: [2511.16637](https://arxiv.org/abs/2511.16637)
**Code**: [satsuma](https://doi.org/10.5281/zenodo.17607863), [VeriPB](https://doi.org/10.5281/zenodo.17608873), [CakePB](https://doi.org/10.5281/zenodo.17609070)
**Area**: Other
**Keywords**: symmetry breaking, proof logging, SAT solving, pseudo-Boolean, VeriPB, certified solving

## TL;DR

By introducing auxiliary variables to encode lexicographic order in place of large-integer encodings, this work fundamentally redesigns the VeriPB proof system, achieving order-of-magnitude speedups in both proof generation and verification for certified SAT symmetry breaking, both theoretically and empirically.

## Background & Motivation

- **Background**: Symmetry breaking is a key technique in combinatorial solving to avoid redundant exploration of equivalent search spaces.
- **Limitations of Prior Work**: The SAT community emphasizes provable correctness—solvers in competitions must produce machine-verifiable proofs. Certified symmetry breaking has long been an open challenge: the DRAT format cannot handle multiple symmetries. Bogaerts et al. (2023) proposed a fully general method based on pseudo-Boolean inequalities, but it encodes lexicographic order using large integers $\sum_{i=1}^n 2^{n-i}x_i \leq \sum_{i=1}^n 2^{n-i}y_i$, where coefficients require $n^2$ bits, resulting in linear overhead in proof generation and costly arbitrary-precision arithmetic during verification.

## Core Problem

How can the performance bottleneck of large-integer encodings in symmetry breaking proofs be eliminated while retaining full generality?

## Method

### Mechanism

Auxiliary variables $s_1, \ldots, s_n$ are introduced to encode the lexicographic order (analogous to a clause-based encoding where $s_i$ indicates equality of the first $i$ variables), replacing a single large-coefficient pseudo-Boolean inequality.

### Technical Challenge

Auxiliary variables do not appear in the premises, which breaks the "implication" invariant of the original proof system. This necessitates a fundamental redesign of the proof system.

### Key Design: Separation of Specification and Order

The order encoding is split into two components:

1. **Specification $\mathcal{S}_\preceq(\vec{x}, \vec{y}, \vec{a})$**: Defines how auxiliary variables $\vec{a}$ are computed from the primary variables (analogous to a circuit definition); can be placed in the premises.
2. **Order $\mathcal{O}_\preceq(\vec{x}, \vec{y}, \vec{a})$**: Order constraints expressed in terms of auxiliary variables.

**Definition**: $\alpha \preceq \beta$ if and only if there exists an assignment $\varrho$ satisfying both $\mathcal{S}_\preceq$ and $\mathcal{O}_\preceq$ simultaneously.

### Modified Proof Rules

- **Dominance rule**: The proof obligation becomes $\mathcal{C} \cup \mathcal{D} \cup \{\neg C\} \cup \mathcal{S}_\preceq(\vec{z}|_\omega, \vec{z}, \vec{a}) \vdash \mathcal{C}|_\omega \cup \mathcal{O}_\preceq(\vec{z}|_\omega, \vec{z}, \vec{a})$
- **Redundance rule**: $\mathcal{S}_\preceq$ is similarly incorporated as an additional premise.
- Reflexivity and transitivity proof obligations are adjusted accordingly.

### Asymptotic Complexity Improvements

| Operation | Prior Method | Ours |
|-----------|-------------|------|
| Defining lexicographic order ($n$ variables) | $O(n^2)$ | $O(n)$ |
| Symmetry breaking (support $k$) logging | $O(nk)$ | $O(k)$ |
| Symmetry breaking checking | $O(n^2 + nk^2)$ | $O(n)$ |

## Key Experimental Results

### Constructed Benchmarks (5 families)

- The ratio of proof generation time to solving time (without proof) approaches a constant for the proposed method.
- The prior method exhibits clearly degrading asymptotic behavior on PHP/RPHP families.
- For proof verification, the proposed method demonstrates better scaling across all 5 families.

### SAT Competition Instances (2020–2024, 982 instances with symmetry)

| Metric | Ours | Prior Method |
|--------|------|-------------|
| Successful proof generation | 982/982 | 930/982 |
| VeriPB verification passed | 893 | 806 |
| CakePB verification passed | 799 | 732 |

The proposed method is never worse than the prior method by more than a constant factor and is frequently faster by orders of magnitude.

## Highlights & Insights

- Resolves a long-standing performance bottleneck in certified symmetry breaking.
- Theoretical improvements (at least a linear-factor speedup) are thoroughly validated experimentally.
- A complete end-to-end formally verified chain is established: satsuma → VeriPB → CakePB, providing formal correctness guarantees.
- Implemented in state-of-the-art tools, offering direct practical value.

## Limitations & Future Work

- Proof verification may still be asymptotically slower than symmetry breaking itself (each symmetry requires reasoning over all order variables).
- Enabling proof generation still incurs a constant-factor overhead, primarily due to disk I/O.
- Only static symmetry breaking is addressed; dynamic symmetry exploitation during search is not considered.
- The efficiency of CakePB's formally verified arithmetic library still has room for improvement.

## Related Work & Insights

| Dimension | Bogaerts et al. (2023) | Ours |
|-----------|----------------------|------|
| Order encoding | Large-integer PB inequalities | Auxiliary variable clauses |
| Coefficient bit-width | $O(n^2)$ | $O(n)$ |
| Generality | Fully general | Fully general |
| Practical feasibility | Infeasible for large symmetries | Feasible for all 982 instances |

**Insights**:
- "Replacing large coefficients with auxiliary variables" is a general optimization strategy in PB reasoning.
- The Specification/Order separation pattern for proof system design may prove useful in other domains of certified solving.

## Rating

⭐⭐⭐⭐ — Strong integration of theory and practice, resolves a long-standing open problem, complete and formally verified toolchain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Certified but Fooled! Breaking Certified Defences with Ghost Certificates](certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](certified_branch-and-bound_maxsat_solving_extended_version.md)
- [\[AAAI 2026\] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer](synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)
- [\[ICLR 2026\] Speculative Actions: A Lossless Framework for Faster AI Agents](../../ICLR2026/others/speculative_actions_faster_ai_agents.md)

</div>

<!-- RELATED:END -->
