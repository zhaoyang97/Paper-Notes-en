---
title: >-
  [Paper Note] Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction
description: >-
  [AAAI 2026][Description Logics] This paper introduces two extensions of the description logic ALC with definite descriptions — local definite descriptions $\{ι C\}$ and global definite descriptions $ι C.D$ — and proves that the satisfiability problems of all three extended logics are ExpTime-complete. Furthermore, it establishes that global definite descriptions are strictly more expressive than local ones ($\mathcal{ALC}\iota_L < \mathcal{ALC}\iota_G = \mathcal{ALC}\iota$)…
tags:
  - "AAAI 2026"
  - "Description Logics"
  - "Definite Descriptions"
  - "Computational Complexity"
  - "Expressiveness"
  - "Tableau Calculus"
date: 2026-05-08
content_hash: d9a265ac4ef89ddd
---

# Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction

**Conference**: AAAI 2026
**arXiv**: [2512.06604](https://arxiv.org/abs/2512.06604)  
**Code**: [github.com/ExtenDD/two-types-of-DDs-AAAI-2026](https://github.com/ExtenDD/two-types-of-DDs-AAAI-2026)  
**Area**: Other
**Keywords**: Description Logics, Definite Descriptions, Computational Complexity, Expressiveness, Tableau Calculus

## TL;DR

This paper introduces two extensions of the description logic ALC with definite descriptions — local definite descriptions $\{ι C\}$ and global definite descriptions $ι C.D$ — and proves that the satisfiability problems of all three extended logics are ExpTime-complete. Furthermore, it establishes that global definite descriptions are strictly more expressive than local ones ($\mathcal{ALC}\iota_L < \mathcal{ALC}\iota_G = \mathcal{ALC}\iota$), and provides tableau calculi decision procedures along with experimental evaluation.

## Background & Motivation

**Definite Descriptions (DDs)** are expressions of the form "the unique $x$ satisfying property $C$," enabling reference to objects via their distinguishing characteristics. In Knowledge Representation and Reasoning (KRR), definite descriptions can precisely identify individuals while encoding structural constraints — a capability that opaque identifiers (such as database primary keys) lack.

For example, "the tallest building" can be expressed as:
$$\{ι(\mathsf{building} \sqcap \forall \mathsf{tallThan}.\neg \mathsf{building})\}$$

The **research motivation** stems from several open problems concerning definite descriptions in description logics:

**Unknown computational complexity**: The satisfiability complexity of ALC extended with local and global definite descriptions has not been systematically studied.

**Unclear expressiveness**: The expressiveness relationship between the two types of definite descriptions has not been characterized.

**Missing bisimulation**: Existing bisimulations apply only to settings with nominals and the universal role ($\mathcal{ALCO}_u^\iota$); bisimulations for pure ALC+DD have not been proposed.

**No reasoning systems**: Despite practical motivation, no DL reasoner currently supports definite descriptions.

**Distinction between the two types of definite descriptions**:
- **Local definite description $\{ι C\}$**: Denotes the singleton set consisting of "the unique individual satisfying $C$" (if it exists); it is a concept whose extension is that unique individual itself.
- **Global definite description $ι C.D$**: Denotes the entire domain "if the unique individual satisfying $C$ also satisfies $D$." For example, $ι(\mathsf{building} \sqcap \forall \mathsf{tallThan}.\neg \mathsf{building}).\exists \mathsf{locIn}.\{Dubai\}$ expresses "the tallest building is located in Dubai."

## Method

### Overall Architecture

The theoretical contributions of this paper are organized at three levels:
1. **Complexity analysis**: Proving that concept and ontology satisfiability for all three logical systems are ExpTime-complete.
2. **Expressiveness analysis**: Establishing $\mathcal{ALC}\iota_L < \mathcal{ALC}\iota_G = \mathcal{ALC}\iota$ by defining novel bisimulation notions.
3. **Decision procedures**: Designing and implementing tableau calculi.

### Key Designs

1. **Semantic Definitions and Complexity Proofs**:

   **Semantics**:
   - Local DD: $(\{ι C\})^\mathcal{I} = \{d\}$ if $C^\mathcal{I} = \{d\}$, otherwise $\emptyset$
   - Global DD: $(ι C.D)^\mathcal{I} = \Delta^\mathcal{I}$ if $C^\mathcal{I} = \{d\} \subseteq D^\mathcal{I}$, otherwise $\emptyset$

   **ExpTime upper bound**: Ontology satisfiability in $\mathcal{ALC}\iota$ is polynomially reduced to the already known ExpTime-complete problem for $\mathcal{ALCO}_u^\iota$. The key translation replaces each global DD $ι C.D$ with $\exists u.(\{ι C\} \sqcap D)$, where $u$ is the universal role.

   **ExpTime lower bound**: The known ExpTime-hard problem of ALC concept satisfiability w.r.t. a TBox is log-space reduced to pure concept satisfiability in $\mathcal{ALC}\iota_L$ (and $\mathcal{ALC}\iota_G$). The construction cleverly uses definite descriptions to "internalize" TBox axioms:
   $C' = C \sqcap \bigsqcap_{(D \sqsubseteq E) \in \mathcal{T}} ((\neg D \sqcup E) \sqcap \{ι(\neg(\neg D \sqcup E) \sqcup A_{D \sqsubseteq E})\})$
   where each fresh atom $A_{D \sqsubseteq E}$ ensures that the TBox axiom is satisfied at every individual.

2. **Bisimulation Definitions and Expressiveness Analysis**:

   A core theoretical contribution of this paper is the definition of appropriate bisimulation relations for $\mathcal{ALC}\iota_L$ and $\mathcal{ALC}\iota_G$.

   **$\mathcal{ALC}\iota_L$ bisimulation**: In addition to the standard ALC bisimulation conditions (Atom, Forth, Back), a **NamesL** condition is added:
   $\text{Names}(Dom(Z), \mathcal{I}) = \text{Names}(Rng(Z), \mathcal{J})$
   That is, the sets of "named individuals" within the domain of the bisimulation relation must coincide.

   **$\mathcal{ALC}\iota_G$ bisimulation**: The NamesL condition is replaced by a **NamesG** condition:
   $\text{Names}(\Delta^\mathcal{I}, \mathcal{I}) = \text{Names}(\Delta^\mathcal{J}, \mathcal{J})$
   That is, the sets of "named individuals" across the entire domain must coincide.

   By constructing a counterexample (interpretations $\mathcal{I}$ and $\mathcal{J}$ in Example 4), the paper demonstrates that an $\mathcal{ALC}\iota_L$ bisimulation can exist where no $\mathcal{ALC}\iota_G$ bisimulation exists, thereby establishing $\mathcal{ALC}\iota_L < \mathcal{ALC}\iota_G$.

   **Key technical result**: Theorem 9 establishes the connection between "named individuals" and "naming sets." The identification of named individuals reduces to standard ALC bisimulation checking — an individual that is not ALC-bisimilar to any other individual is a named individual. Algorithms 1 and 2 operationalize the bisimulation verification process.

3. **Tableau Calculus Decision Procedures**:

   The paper designs $\mathtt{TAB}_{\mathcal{ALC}\iota}$, comprising rules for standard ALC constructs and dedicated rules for definite descriptions:

   **Global DD rules**:
   - $(ι_1^g)$: Introduces a new individual satisfying both $C$ and $D$.
   - $(ι_2^g)$: Enforces uniqueness — merges the theories of any two individuals satisfying $C$.
   - $(\neg ι^g)$: Handles negation — for each individual, either it does not satisfy $C$, or it does not satisfy $D$, or there exist two distinct individuals satisfying $C$.
   - $(cut_\iota^g)$: A case-splitting rule — ensures a definite judgment about $C$ for each individual; critical for completeness.

   **Local DD rules**: Analogous treatment, but the negation rule applies only locally to the current individual.

   **Blocking condition**: Pattern-based blocking is used to ensure termination — if an individual $a'$ whose theory contains the required conditions already exists, it serves as a proxy for an $r$-successor.

   **Termination proof** (Theorem 16): The number of individuals on each branch is bounded by $2^{4|C|}$, and each individual satisfies at most $4|C|$ concepts, so the tableau is finite.

### Loss & Training

(Not applicable — this is a purely theoretical contribution with no training process.)

## Key Experimental Results

### Main Results

A Python-implemented prover is used to evaluate the effect of different types and proportions of definite descriptions on reasoning efficiency:

| DD Type | DD count / binary operator count | 0.1·k | 0.3·k | 0.5·k |
|---------|-----------------------------------|-------|-------|-------|
| Global DD | Mean runtime | 0.239s | 0.750s | 0.777s |
| Global DD | Std. deviation | 0.879s | 2.16s | 1.81s |
| Global DD | Timeouts (/150) | 21 | 31 | 42 |
| Local DD | Mean runtime | 0.371s | 0.356s | 0.411s |
| Local DD | Std. deviation | 1.31s | 0.92s | 1.59s |
| Local DD | Timeouts (/150) | 15 | 32 | 28 |

### Ablation Study

Scalability analysis (concept size vs. runtime):

| Configuration | Runtime trend | Timeouts | Notes |
|---------------|--------------|----------|-------|
| Global DD only | Polynomial growth (highest) | More | Consistent with theoretical analysis |
| Local DD only | Polynomial growth (moderate) | Moderate | Local processing is simpler |
| No DD | Polynomial growth (lowest) | 2/200 | Baseline ALC |

### Key Findings

1. **Runtime and timeout counts scale proportionally with DD quantity**: More DDs yield harder reasoning instances.
2. **Global DDs are more challenging than local DDs**: Higher mean runtime and more timeouts — consistent with the expressiveness analysis.
3. **Practical runtime exhibits polynomial growth**: Despite the theoretical ExpTime-complete complexity, empirical runtime on practical concepts grows polynomially.
4. **DD overhead is manageable**: Even with a large number of DDs, reasoning time remains within a manageable range, confirming the practical feasibility of the extension.
5. The time for generating and parsing concepts scales linearly with concept size (100 atoms: ~0.5s parsing).

## Highlights & Insights

- **Theoretical completeness**: The paper forms a coherent theoretical framework spanning syntax, semantics, complexity, expressiveness, and decision procedures.
- **Elegant bisimulation design**: Although the NamesL and NamesG conditions appear to require quantification over all concepts, Theorem 9 cleverly reduces them to computable ALC bisimulation checks.
- **Elegant counterexample construction**: Using only $\Delta^\mathcal{I} = \{a,b\}$ and $\Delta^\mathcal{J} = \{c,d,e\}$, the paper distinguishes the expressiveness gap between the two bisimulation notions.
- **Uniform tableau rule design**: Despite the semantic differences between the two types of DDs, analogous tableau rules suffice to handle both.
- **Necessity of the $(cut_\iota^g)$ rule**: A concrete unsatisfiable concept demonstrates that this rule is indispensable for completeness.

## Limitations & Future Work

1. **Implementation efficiency requires optimization**: The Python implementation is not systematically optimized, resulting in a notable number of timeouts.
2. **Only concept satisfiability is tested**: Satisfiability problems involving ontologies are not evaluated.
3. **Random concept generation may not be representative**: Concept structures in real-world applications may differ significantly from randomly generated ones.
4. **No integration with existing DL reasoners**: Such as HermiT or FaCT++.
5. **Richer DL bases unexplored**: Extensions such as SHIQ and SROIQ have not been considered.

## Related Work & Insights

This paper sits at the intersection of philosophical logic (Russell's theory of definite descriptions), description logics (ALC and its extensions), and modal logic (bisimulation theory). It has direct implications for the design of ontology and query languages — definite descriptions provide a semantically richer means of individual reference than opaque identifiers. Future work may optimize algorithms and implementations, extend the framework to more expressive DL fragments, and explore usage patterns of definite descriptions in real-world ontologies.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel bisimulation definitions and expressiveness separation results)
- Experimental Thoroughness: ⭐⭐⭐ (Preliminary experiments validate theoretical feasibility, limited in scale)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematically rigorous, proofs thorough, structure clear)
- Value: ⭐⭐⭐⭐ (Fills a foundational theoretical gap in the study of DDs within DLs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)
- [\[AAAI 2026\] Automated Reproducibility Has a Problem Statement Problem](automated_reproducibility_has_a_problem_statement_problem.md)
- [\[AAAI 2026\] On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting](on_the_edge_of_core_non-emptiness_an_automated_reasoning_approach_to_approval-ba.md)
- [\[ICML 2026\] Complexity as Advantage: A Regret-Based Perspective on Emergent Structure](../../ICML2026/others/complexity_as_advantage_a_regret-based_perspective_on_emergent_structure.md)

</div>

<!-- RELATED:END -->
