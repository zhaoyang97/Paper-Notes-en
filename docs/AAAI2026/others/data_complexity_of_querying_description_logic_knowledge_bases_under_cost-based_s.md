---
title: >-
  [Paper Note] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics
description: >-
  [AAAI 2026][Description Logics] This paper systematically investigates the data complexity of query answering over weighted description logic knowledge bases under cost-based semantics. It establishes that optimal-cost semantics is decidable within $\Delta_2^p$, and delivers a surprising positive result: for DL-Lite$_{\text{bool}}^{\mathcal{H}}$ ontologies with a fixed cost bound, both certain answers to instance queries and possible answers to conjunctive queries admit first-order rewritings, achieving the lowest possible data complexity (AC$^0$).
tags:
  - AAAI 2026
  - Description Logics
  - Cost-Based Semantics
  - Inconsistency Tolerance
  - Ontology-Mediated Query Answering
  - Computational Complexity
date: 2026-05-08
content_hash: c1e2db7a3eab262b
---

# Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics

**Conference**: AAAI 2026
**arXiv**: [2511.07095](https://arxiv.org/abs/2511.07095)
**Code**: None
**Area**: Other (Knowledge Representation & Reasoning / Description Logics)
**Keywords**: Description Logics, Cost-Based Semantics, Inconsistency Tolerance, Ontology-Mediated Query Answering, Computational Complexity

## TL;DR

This paper systematically investigates the data complexity of query answering over weighted description logic knowledge bases under cost-based semantics. It establishes that optimal-cost semantics is decidable within $\Delta_2^p$, and delivers a surprising positive result: for DL-Lite$_{\text{bool}}^{\mathcal{H}}$ ontologies with a fixed cost bound, both certain answers to instance queries and possible answers to conjunctive queries admit first-order rewritings, achieving the lowest possible data complexity (AC$^0$).

## Background & Motivation

### Ontology-Mediated Query Answering (OMQA)

OMQA is a central task in knowledge representation: given an ontology (TBox) $\mathcal{T}$, an ABox $\mathcal{A}$, and a query $q(\vec{x})$, the goal is to identify all "certain answers"—tuples that hold in every model of the knowledge base $(\mathcal{T}, \mathcal{A})$. Description Logics (DLs) are the most widely used formal languages for ontologies.

### The Inconsistency Problem

When a knowledge base is inconsistent (i.e., $\mathcal{T}$ and $\mathcal{A}$ are contradictory), classical semantics render **every** query answer trivially entailed (ex falso quodlibet), causing OMQA to degenerate.

### Limitations of Prior Work & Inconsistency-Tolerant Methods

**Repair-based semantics** (e.g., AR semantics) consider maximal consistent subsets of the ABox, but implicitly assume the TBox is fully reliable and that inconsistency originates solely from ABox errors. In practice, TBox axioms may also admit exceptions ("soft" constraints).

### Cost-Based Semantics: A Quantitative Approach

Bienvenu, Bourgaux & Jean (2024) proposed **cost-based semantics**:
- Each axiom and assertion is assigned a weight (penalty); weight $\infty$ denotes a "hard" constraint.
- The **cost** of an interpretation is based on the weighted sum of violated axioms and assertions.
- Four query answering semantics are defined: bounded-cost certain/possible and optimal-cost certain/possible.

### Paper Goals

Two significant gaps remain in prior work:

1. No non-trivial upper bound is known for the **optimal-cost certain semantics** (the most practically useful semantics).
2. The DLs studied lack inverse roles and role inclusions, excluding the DL-Lite family—the most commonly used DLs in OMQA.

## Method

### Overall Architecture

The core contributions are a series of **complexity results** covering a spectrum of DLs from the lightweight DL-Lite$_{\text{core}}$ to the expressive $\mathcal{ALCHIO}$. The methodology consists of three parts: upper bound proofs, lower bound proofs, and positive first-order rewritability results.

### Key Designs

#### 1. **Formalization of Cost-Based Semantics**

**Weighted Knowledge Base (WKB)**: $\mathcal{K}_\omega = (\mathcal{T}, \mathcal{A})_\omega$, where $\omega: \mathcal{T} \cup \mathcal{A} \mapsto \mathbb{N}_{>0} \cup \{\infty\}$.

**Cost of an interpretation**:
$$\omega(\mathcal{I}) = \sum_{\tau \in \mathcal{T}} \omega(\tau)|vio_\tau(\mathcal{I})| + \sum_{\alpha \in vio_\mathcal{A}(\mathcal{I})} \omega(\alpha)$$

TBox axiom costs accumulate with the number of violations (counted by elements), while ABox assertion costs are binary (violated or not).

**Four query answering semantics**:
- $k$-bounded-cost certain: holds in all interpretations of cost $\leq k$
- $k$-bounded-cost possible: holds in some interpretation of cost $\leq k$
- Optimal-cost certain: holds in all interpretations of minimum cost
- Optimal-cost possible: holds in some interpretation of minimum cost

#### 2. **General Upper Bounds: Quotient Construction and Small Model Property** (Section 4)

**Core technique**: Given an interpretation $\mathcal{I}$ that satisfies (or does not satisfy) a query $q$, construct another interpretation $\mathcal{J}$ such that:
- The query satisfaction status is preserved relative to $\mathcal{I}$
- The cost does not exceed that of $\mathcal{I}$
- The domain size is polynomially bounded in $|\mathcal{A}|$ and, crucially, **independent of the cost bound $k$**

**Key difficulty**: Preserving "query non-satisfaction" (for CQA$_c$) is substantially harder than preserving satisfaction. The former requires a **parameterized quotient construction** (Lemma 6) that introduces a "guard set" $\mathcal{V} \subseteq \mathcal{T}$ to control which violations are preserved exactly.

**Iterative construction**:
1. Initialize $\mathcal{V}_0 = \emptyset$
2. If $\omega(\mathcal{J}_{\mathcal{V}_i}) > k$, identify an axiom $\tau$ responsible for violations exceeding those in $\mathcal{I}$
3. Add $\tau$ to $\mathcal{V}_{i+1}$ and reconstruct
4. Convergence is guaranteed within $|\mathcal{T}|$ steps

**Theorem**: CQA$_p$ and CQA$_c$ for $\mathcal{ALCHIO}$ lie in NP and coNP, respectively; optimal-cost semantics lies in $\Delta_2^p$ (via binary search for the optimal cost).

#### 3. **Lower Bounds: NP/coNP-Hardness Already at $k=1$** (Section 5)

**For $\mathcal{EL}_\bot$** (Theorem 3): Via reduction from 3-SAT. A WKB is constructed such that the sole weight-1 axiom $\mathsf{Bool} \sqsubseteq \mathsf{True}$ encodes Boolean variable assignments; the formula is satisfiable if and only if the WKB is 1-satisfiable. A key improvement over prior work reduces the threshold from $k \geq 3$ to $k \geq 1$.

**For DL-Lite$_{\text{core}}$** (Theorem 5): Via reduction from 3-coloring, using six role names $\{r_1,r_2,g_1,g_2,b_1,b_2\}$ to encode three colors, with incompatibility axioms ensuring adjacent nodes receive distinct colors.

**For optimal-cost semantics** (Theorem 6): $\Delta_2^p$-hardness is established via reduction from "lexicographically maximal satisfying assignment," exploiting exponentially large weights under binary encoding to simulate lexicographic comparison.

#### 4. **Surprising Result: AC$^0$ Rewritability for the DL-Lite Family** (Section 6)

**Theorem 8** (most technically demanding): For any fixed $k \geq 1$, CQA$^k_p$ and IQA$^k_c$ under DL-Lite$_{\text{bool}}^{\mathcal{H}}$ lie in AC$^0$.

**Approach**: First-order query rewriting—given a weighted TBox $\mathcal{T}_{\omega_\mathcal{T}}$, a query $q$, and a cost bound $k$, a first-order query $q'$ is constructed such that for any weighted ABox $\mathcal{A}_{\omega_\mathcal{A}}$:

$$(\mathcal{T}, \mathcal{A})_\omega \models^k_p q \text{ iff } \mathcal{I}_{\mathcal{A}_{\omega_\mathcal{A}}} \models q'$$

**Core technique—minimal model property**:

**ABox types**: An extension of 1-types incorporating cardinality information $\exists^{>k}\mathsf{r}$ (i.e., role $r$ has more than $k$ distinct $r$-successors), capturing structural information under cost constraints.

**Rare types**: An ABox type $t$ is "rare" if it has at most $2k$ instances. Non-rare types always admit a violation-free interpretation (otherwise more than $k$ violations would occur, exceeding the cost bound).

**Core**:
$$\mathsf{core}(\mathcal{A}) = \mathsf{pc}(\mathcal{A}) \cup \{b \mid a \leadsto_i b, a \in \mathsf{pc}(\mathcal{A}), i \leq k+1\}$$
where $\mathsf{pc}(\mathcal{A})$ contains individuals of rare types and those appearing in the query. Neighborhood exploration depth is $k+1$ (since each level introduces at most one violation).

The **core size** is constant with respect to $|\mathcal{A}|$, so all relevant violation and query satisfaction information is concentrated in a constant-size subdomain, enumerable by finitely many first-order sub-queries.

**Strategy**: The rewritten query $q'$ is a disjunction of sub-queries corresponding to all p-strategies (or c-strategies). Each strategy $(M, \Gamma, \mathcal{V}_\nu)$ specifies a core-sized interpretation, its individual part, and the allowed ABox violations. Sub-queries check whether the ABox types of remaining individuals are "safe" (interpretable without introducing additional violations).

### Summary of Complexity Results

| Reasoning Task | Fixed $k$ | Variable $k$ | Optimal Cost |
|---|---|---|---|
| $\mathcal{EL}_\bot$ / $\mathcal{ALCHIO}$ | NP/coNP-complete | NP/coNP-complete | $\Delta_2^p$-complete |
| DL-Lite$_{\text{core}}$ / DL-Lite$_{\text{bool}}^{\mathcal{H}}$ | **AC$^0$** (CQA$_p^k$, IQA$_c^k$) / coNP (CQA$_c^k$) | NP/coNP-complete | $\Delta_2^p$-complete |

## Key Experimental Results

### Main Results (Table 1)

| Reasoning Task | DL-Lite$_{\text{core}}$ Fixed $k$ | DL-Lite$_{\text{core}}$ Variable $k$ | $\mathcal{EL}_\bot$ Fixed $k$ |
|---|---|---|---|
| BCS$^k$ / IQA$_p^k$ | AC$^0$ (upper bound) | NP-complete | NP-complete ($k \geq 1$) |
| IQA$_c^k$ | AC$^0$ (upper bound) | coNP-complete | coNP-complete ($k \geq 1$) |
| CQA$_c^k$ | coNP-complete ($k \geq 1$) | coNP-complete | coNP-complete ($k \geq 1$) |
| Optimal-cost certain/possible | $\Delta_2^p$-complete | — | $\Delta_2^p$-complete |

### Ablation Study / Key Comparisons

| Dimension | Ours vs. Prev. SOTA (BBJ 2024) |
|---|---|
| Upper bound for optimal-cost certain | **$\Delta_2^p$** (first non-trivial upper bound) | No upper bound |
| $k$ threshold for NP/coNP-hardness | **$k \geq 1$** | $k \geq 3$ |
| DL coverage | Includes DL-Lite family | Only $\mathcal{EL}_\bot$ to $\mathcal{ALCO}$ |
| Positive result for DL-Lite at fixed $k$ | **AC$^0$** | None |

### Key Findings

1. **The precise complexity of optimal-cost semantics is $\Delta_2^p$**—applicable across a broad range of DLs from DL-Lite$_{\text{core}}$ to $\mathcal{ALCHIO}$.
2. **The tractability of DL-Lite at fixed cost bounds is a surprising result**: the data complexity of CQA$_p^k$ and IQA$_c^k$ matches that of classical DL-Lite query answering (AC$^0$).
3. **NP-hardness already holds at $k=1$**: a substantial improvement over the prior lower bound of $k \geq 3$.
4. **CQA$_c^k$ for DL-Lite$_{\text{core}}$ is coNP-complete**—even when CQA$_p^k$ and IQA$_c^k$ are tractable.

## Highlights & Insights

1. The **cost-independent small model property** is the central technical breakthrough: the iterative quotient construction yields a polynomial domain size bound that is independent of the cost bound $k$, which is the key to obtaining NP/coNP upper bounds rather than higher ones.
2. The **design of the minimal model property for AC$^0$ results is elegant**: the notion of "rare types" compresses infinitely many ABox configurations into a constant number of categories, and the introduction of $\exists^{>k}\mathsf{r}$ into ABox types precisely captures structural information under cost constraints.
3. **First-order rewritability implies practical implementability**: in principle, query answering under cost-based semantics can be compiled into standard SQL queries and executed over a database.
4. **The improved lower bound ($k \geq 1$)** demonstrates that intractability is not an artifact of high cost values.

## Limitations & Future Work

1. The AC$^0$ result **does not cover CQA$_c^k$** (shown to be coNP-complete), limiting the scope of the positive results.
2. **$\Delta_2^p$-hardness relies on binary-encoded weights**—complexity may be lower under unary encoding.
3. **DLs with functional constraints or number restrictions** (e.g., $\mathcal{ALCQI}$) are not addressed and would require fundamentally different techniques.
4. **No practical implementation of the rewriting is provided**; a gap remains between the theoretical AC$^0$ result and an efficient implementation.
5. **The treatment of DL-Lite$_\mathcal{R}$** (with negative role inclusions) remains ongoing work.

## Related Work & Insights

- Cost-based semantics unifies classical OMQA and weighted repair-based AR semantics.
- The work is closely related to the bounded description technique of Lutz & Manière (2024), which inspired the iterative quotient construction.
- Insight: in inconsistency handling, quantitative approaches (weights/costs) can provide richer semantics than qualitative ones (simple consistent/inconsistent distinctions) while preserving tractability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The AC$^0$ result exceeds expectations; the quotient construction technique is deep)
- Experimental Thoroughness: ⭐⭐⭐ (Purely theoretical; no implementation or experiments, but the complexity landscape is complete)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured, though the high technical density is challenging for non-specialists)
- Value: ⭐⭐⭐⭐ (Significant theoretical contribution to description logics and knowledge representation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction](description_logics_with_two_types_of_definite_descriptions_complexity_expressive.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)
- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](cost-free_neutrality_for_the_river_method.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)

</div>

<!-- RELATED:END -->
