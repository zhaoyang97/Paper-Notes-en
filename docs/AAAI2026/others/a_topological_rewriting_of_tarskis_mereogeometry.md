---
title: >-
  [Paper Note] A Topological Rewriting of Tarski's Mereogeometry
description: >-
  [AAAI 2026][Tarski Geometry] This work extends the λ-MM library within the Coq theorem prover to recast Tarski's solid geometry—grounded in Leśniewski's mereology—into a fully formalized system with a complete topological structure. It proves that mereological classes correspond to regular open sets, satisfy Kuratowski's interior axioms, and exhibit the Hausdorff (T2) separation property, thereby providing a unified mereological–geometric–topological theoretical framework for qualitative spatial reasoning.
tags:
  - AAAI 2026
  - Tarski Geometry
  - Mereology
  - Topological Space
  - Coq Theorem Proving
  - Regular Open Sets
  - Qualitative Spatial Reasoning
date: 2026-05-08
content_hash: f51a342613ca2249
---

# A Topological Rewriting of Tarski's Mereogeometry

**Conference**: AAAI 2026
**arXiv**: [2511.12727](https://arxiv.org/abs/2511.12727)
**Code**: [λ-MM Coq Library](https://www.univ-smb.fr/listic/technologies/logiciels/modeles-decidables-pour-la-specification-d-ontologies/extending-tarskis-mereo-geometry-with-topological-definitions/)
**Area**: Other
**Keywords**: Tarski Geometry, Mereology, Topological Space, Coq Theorem Proving, Regular Open Sets, Qualitative Spatial Reasoning

## TL;DR

This work extends the λ-MM library within the Coq theorem prover to recast Tarski's solid geometry—grounded in Leśniewski's mereology—into a fully formalized system with a complete topological structure. It proves that mereological classes correspond to regular open sets, satisfy Kuratowski's interior axioms, and exhibit the Hausdorff (T2) separation property, thereby providing a unified mereological–geometric–topological theoretical framework for qualitative spatial reasoning.

## Background & Motivation

**Background**: In qualitative spatial reasoning (QSR), the combination of mereology and topology forms the foundation of spatial theories such as RCC8 and contact algebras. These theories follow Whitehead's point-free geometry, substituting "regions" for "points" as the primitive entities.

**Limitations of Prior Work**: Existing approaches face multiple challenges: (1) the ontological status of boundaries is ambiguous (spatial objects or abstract entities?); (2) decidability questions remain unresolved; (3) insufficient reasoning mechanisms limit the practical utility of spatial representations; (4) naively mapping part–whole relations into first-order logic results in limited expressive power.

**Key Challenge**: Qualitative spatial models based on Goodman-style mereology and pseudo-topology lack genuine Euclidean geometry and a complete topological space, and thus cannot support advanced geometric reasoning. While Tarski's solid geometry defines solids, balls, and interior points, it lacks a formal bridge to modern topology.

**Key Insight**: The paper leverages the existing λ-MM Coq library—an implementation of Leśniewski's mereology based on inductive type theory—to construct algebraic topological relations on top of it, proving that the mereological structure itself entails a topology, and formalizing Tarski's geometry as a subspace of that topology.

**Core Idea**: Proving that mereological classes are equivalent to regular open sets, thereby constructing a complete topological space from within mereology itself, without appealing to set theory.

## Method

### Overall Architecture

The paper proceeds in two major parts:
- **Part I**: Adds meet/join operators and topological definitions (interior, closure, boundary) to the λ-MM library, and proves that the collection of individual names in mereology constitutes a topological space $\langle N, \mathbb{T}_{MM} \rangle$ satisfying Kuratowski's interior axioms.
- **Part II**: Defines Tarski's geometric notions—solids, balls, and interior points—as subspaces of the above topology, proves Tarski's three postulates, and extends the theory to establish the T2 (Hausdorff) separation property.

### Key Designs

1. **Bridging Mereology and Topology (Regular Open Set Correspondence)**:

    - **Function**: Establishes a one-to-one correspondence between mereological classes (m-classes) in λ-MM and regular open sets.
    - **Mechanism**: Individual names in mereology are interpreted as open sets of the topology; the part–whole relation $pt$ corresponds to set inclusion $\subseteq$; and mereological $b\_sum$/$b\_product$/$compl$ correspond respectively to topological union/intersection/complement. The interior operator is defined as $\eta\;P\;(interior\;Q) \triangleq \eta\;Q\;Q \wedge \eta\;P\;Q$, meaning that the interior of an individual is itself (since all individual names are open sets).
    - **Design Motivation**: The topological construction is achieved entirely within the nominalist system, without introducing external set-theoretic semantics.

2. **Coq Formalization of Kuratowski's Axioms**:

    - **Function**: Creates a `Topological` type class in Coq, encoding the four Kuratowski interior axioms as type class fields, with machine-verified proofs.
    - **Mechanism**: The four axioms are: preservation of the whole space $int(\mathbb{T}_{MM}) = \mathbb{T}_{MM}$; sub-inclusion $int(A) \subseteq A$; idempotency $int(int(A)) = int(A)$; and preservation under intersection $int(A \cap B) = int(A) \cap int(B)$. Proofs are completed semi-automatically using CoqHammer (with the Vampire and Eprover automated reasoners).
    - **Design Motivation**: Machine-verified proofs are more reliable than manual ones, and the generic nature of the type class ensures that changing the underlying data does not affect the structure.

3. **Tarski's Geometry as a Topological Subspace**:

    - **Function**: Redefines saturated interior points over a subbasis of balls, proves Tarski's three postulates, and establishes the T2 (Hausdorff) separation axiom.
    - **Mechanism**: Tarski defines "points" as classes of concentric balls (intensional classes), and solids as mereological sums of balls. The paper shows that these definitions are consistent with regular open sets in the subspace topology, thereby reducing Tarski's five axioms to three and supplementing them with the T2 property.
    - **Design Motivation**: The T2 property guarantees unique separation of points, which is a prerequisite for many spatial reasoning algorithms; axiom reduction improves the practical utility of the system.

### Formalization System / Proof Strategy

All proofs are carried out in Coq 8.x, with semi-automated proofs completed via CoqHammer invoking the Vampire and Eprover automated theorem provers. The λ-MM library is based on the Calculus of Inductive Constructions (CIC) restricted to classical logic. The formalization code is publicly available.

## Key Experimental Results

### Formalization Achievements

| Achievement | Content | Remarks |
|---|---|---|
| Kuratowski Interior Axioms | All 4 proved | Machine-verified in Coq |
| Tarski's Postulates | 3 proved (reduced from original 5) | 40% reduction in axiom count |
| T2 (Hausdorff) Property | First proved in this framework | Guarantees unique separation of points |
| Closure / Boundary / Regularity | Fully formalized with proofs | Complete topological toolkit |

### Theoretical Contribution Comparison

| Aspect | RCC8 / Contact Algebras | λ-MM (Original) | This Work |
|---|---|---|---|
| Topological Space | Pseudo-topology | None | Full Kuratowski topology |
| Geometry | None | Tarski geometry (partial) | Tarski geometry + T2 |
| Formal Verification | None | Coq mereology | Coq mereology + topology + geometry |
| Boundary Definition | Ambiguous | None | Precise formalization |
| Decidability | Partial fragments | Yes (inductive types) | Inherited |

### Key Findings

- The equivalence between mereological classes and regular open sets implies that mereology "naturally" entails topological structure, without any external introduction.
- Simplifying Tarski's axiom system (5→3) via subspace construction reduces the complexity of formalization.
- The proof of the T2 property demonstrates that the system possesses sufficient separation capability to support fine-grained spatial reasoning.

## Highlights & Insights

- **Mereology as Topology**: This is not merely "adding a topological layer to mereology," but rather a proof that the algebraic structure of mereology itself constitutes a topological space, unifying two traditionally separate theories.
- **Value of Formal Verification**: In the spatial reasoning community, errors in manual proofs have a long history; Coq machine verification provides an incontestable correctness guarantee.
- **Generic Type Class Design**: The topological structure is implemented as a parameterized type class, so substituting the underlying data (e.g., different mereological variants) does not require rewriting proofs.

## Limitations & Future Work

- **Highly Theoretical**: The paper contains virtually no computational experiments or application demonstrations; its impact on practical spatial reasoning systems requires further validation.
- **Decidability Not Fully Addressed**: Although inductive types provide a constructive foundation, the decidable fragments of the extended theory are not explicitly characterized.
- **Relation to RCC8**: The paper claims advantages over contact algebra approaches such as RCC8, but lacks direct expressiveness comparisons or concrete reasoning task benchmarks.
- **Accessibility of Coq Code**: Although open-source, the formalization proofs carry a steep learning curve, limiting broader adoption.

## Related Work & Insights

- **vs. RCC8**: RCC8 builds spatial reasoning on contact relations; this work builds on mereology and proves that it subsumes more topological information (e.g., boundary, closure), yielding greater expressive power, though practical tooling is less mature than RCC8.
- **vs. Whitehead's Point-Free Geometry**: The paper inherits the point-free idea of replacing points with regions, but provides the first complete formalization thereof in Coq.
- **vs. Traditional Tarski Geometry Formalization**: Prior Coq formalizations of Tarski's geometry (e.g., GeoCoq) are point-based; this work is the first formalization grounded in point-free mereology.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Establishing a formal equivalence between mereology and topology is an original contribution.
- **Experimental Thoroughness**: ⭐⭐ Purely theoretical work; no computational experiments or application evaluations.
- **Writing Quality**: ⭐⭐⭐ Mathematically rigorous but with an extremely high reading threshold; difficult to follow for non-specialists.
- **Value**: ⭐⭐⭐ Makes an important contribution to the theoretical foundations of qualitative spatial reasoning, though near-term practical applications are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Topological Descriptors for Graph Products](../../NeurIPS2025/others/on_topological_descriptors_for_graph_products.md)
- [\[AAAI 2026\] Deviation Dynamics in Cardinal Hedonic Games](deviation_dynamics_in_cardinal_hedonic_games.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[AAAI 2026\] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details](learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)
- [\[AAAI 2026\] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets](intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)

</div>

<!-- RELATED:END -->
