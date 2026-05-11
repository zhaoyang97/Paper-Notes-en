---
title: >-
  [Paper Note] Model Change for Description Logic Concepts
description: >-
  [AAAI 2026][Description Logic] This paper studies the problem of modifying description logic concepts in response to new model evidence represented as pointed interpretations. It defines three operations — eviction…
tags:
  - "AAAI 2026"
  - "Description Logic"
  - "Model Change"
  - "Belief Revision"
  - "EL/ALC"
  - "Satisfiability"
date: 2026-05-08
content_hash: 7dadf53820624988
---

# Model Change for Description Logic Concepts

**Conference**: AAAI 2026
**arXiv**: [2603.05562](https://arxiv.org/abs/2603.05562)
**Code**: None
**Area**: Other
**Keywords**: Description Logic, Model Change, Belief Revision, EL/ALC, Satisfiability

## TL;DR
This paper studies the problem of modifying description logic concepts in response to new model evidence represented as pointed interpretations. It defines three operations — eviction, reception, and revision — and establishes positive and negative compatibility results for the EL and ALC description logics.

## Background & Motivation

**Background**: Belief revision is a central problem in AI and knowledge representation. Classical belief revision theory (the AGM framework) primarily addresses knowledge change at the propositional logic level. Description Logic (DL), as the foundational formalism for ontologies and the Semantic Web, requires analogous knowledge update operations at the concept level.

**Limitations of Prior Work**: Existing work on belief revision in description logics has focused mainly on changes to TBoxes (terminological components) and ABoxes (assertional components). A systematic theoretical framework for modifying concepts themselves — i.e., DL formulas — in response to new model evidence is lacking. In particular, there is no principled account of how to minimally revise a concept when newly observed individuals (pointed interpretations) are inconsistent with it.

**Key Challenge**: Intuitively, revision might be decomposed as "first remove, then add," but the authors demonstrate that this intuition is incorrect — revision cannot be reduced to a simple composition of eviction followed by reception. This reveals an intrinsic complexity in model change for description logic concepts.

**Goal**: (1) Formalize the problem of model change for DL concepts; (2) Define three basic change operations and study their interrelationships; (3) Establish compatibility results for two important DL fragments, EL and ALC.

**Key Insight**: Adopting a model-theoretic perspective, the paper models concept change as adjustments to the set of pointed interpretations accepted by a concept, drawing on the AGM postulate tradition from classical belief revision theory.

**Core Idea**: The problem of DL concept change is decomposed into three atomic operations (eviction/reception/revision). The paper proves the irreducibility of revision and establishes realizability conditions for these operations in EL and ALC.

## Method

### Overall Architecture
Given a DL concept $C$ and a set of pointed interpretations (each consisting of an interpretation structure together with a designated element), three change operations are defined: eviction removes a specified model from the model set of $C$; reception incorporates a new model into the model set of $C$; revision performs both removal and incorporation simultaneously. The output is a new DL concept $C'$.

### Key Designs

1. **Eviction**:

    - Function: Remove unwanted pointed interpretations from the concept's model set.
    - Mechanism: Given a concept $C$ and a model $\mathcal{I}$ to be removed, find a new concept $C'$ such that $\text{Mod}(C') = \text{Mod}(C) \setminus \{\mathcal{I}\}$ or a principled approximation thereof.
    - Design Motivation: Corresponds to the scenario of discovering that a particular individual should not belong to the concept.

2. **Reception**:

    - Function: Incorporate new pointed interpretations into the concept's model set.
    - Mechanism: Find a new concept $C'$ such that $\text{Mod}(C') \supseteq \text{Mod}(C) \cup \{\mathcal{I}\}$, while keeping $C'$ as close as possible to $C$.
    - Design Motivation: Corresponds to the scenario of discovering that a new individual should belong to the concept.

3. **Revision**:

    - Function: Simultaneously perform model removal and incorporation.
    - Mechanism: The authors prove that revision cannot be straightforwardly decomposed into eviction followed by reception, since the intermediate concept may be inexpressible in the DL, or the composed operation may fail to satisfy the postulates for revision.
    - Design Motivation: To establish a complete theory of concept change capable of handling more complex knowledge update requirements.

### Theoretical Results

For **EL** (supporting only existential quantification and conjunction): positive compatibility results are obtained for eviction and reception — under reasonable conditions, these operations are realizable within EL. For **ALC** (supporting conjunction, disjunction, negation, and existential/universal quantification): the compatibility analysis is more involved, and the authors derive conditions under which the revision operation is compatible. The key negative result is the irreducibility of revision — in general, no eviction-then-reception decomposition satisfying all reasonable postulates exists.

## Key Experimental Results

### Theoretical Complexity Analysis

| Operation | EL | ALC |
|------|-----|-----|
| Eviction Compatibility | ✓ Positive | Partially positive |
| Reception Compatibility | ✓ Positive | Partially positive |
| Revision Decomposability | ✗ Counterexample exists | ✗ Counterexample exists |

### Key Findings
- The irreducibility of revision is the central theoretical contribution of this paper.
- Change operations in EL are relatively tractable; in ALC, the presence of negation and disjunction introduces additional complexity.
- Compatibility results depend on the expressive power of the DL and the chosen principle of minimal change.

## Highlights & Insights
- The finding that **revision is irreducible** is a deep result that refutes the "delete-then-add" intuition and reveals an essential difficulty in DL concept change.
- The paper successfully extends classical belief revision theory to the concept level of description logics, providing a theoretical foundation for ontology maintenance.
- The framework is transferable to other DL fragments (e.g., SHIQ, SROIQ), offering theoretical guidance for knowledge base evolution in the Semantic Web.

## Limitations & Future Work
- The paper is primarily a theoretical contribution; it lacks concrete algorithmic implementations and experimental validation on real ontology repositories.
- Only EL and ALC are considered; compatibility for more expressive fragments (e.g., the OWL 2 family) remains to be investigated.
- The computational complexity analysis of the change operations is insufficiently developed, and practical usability requires further evaluation.
- Future work could explore approximation algorithms or heuristic methods to make the theoretical results applicable to large-scale ontology repositories.

## Related Work & Insights
- **vs. AGM Belief Revision**: AGM addresses propositional logic; this paper extends the framework to description logics, where the challenge lies in the limited expressiveness of DL rendering certain operations unrealizable.
- **vs. TBox/ABox Revision**: Prior work handles the addition and deletion of axioms in TBoxes; this paper focuses on changes to concepts themselves, operating at a finer granularity.
- This paper provides a theoretical foundation for knowledge graph evolution and ontology version management.

## Rating
- Novelty: ⭐⭐⭐⭐ First formalization of model change as operations on DL concepts; irreducibility of revision is a novel finding.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; no empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Logically rigorous with clear formal definitions.
- Value: ⭐⭐⭐ Solid theoretical contribution, though with a relatively narrow application scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[AAAI 2026\] Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction](description_logics_with_two_types_of_definite_descriptions_complexity_expressive.md)
- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](measuring_model_performance_in_the_presence_of_an_intervention.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)

</div>

<!-- RELATED:END -->
