---
title: >-
  [Paper Note] Towards a Common Framework for Autoformalization
description: >-
  [AAAI 2026][LLM Evaluation][autoformalization] This paper systematically surveys existing work on autoformalization across mathematics, logical reasoning, planning, and knowledge representation…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "autoformalization"
  - "formalization"
  - "LLM"
  - "semantic parsing"
  - "formal verification"
date: 2026-05-08
content_hash: a23a9e69e90f62b4
---

# Towards a Common Framework for Autoformalization

**Conference**: AAAI 2026
**arXiv**: [2509.09810](https://arxiv.org/abs/2509.09810)  
**Code**: None  
**Area**: LLM Evaluation
**Keywords**: autoformalization, formalization, LLM, semantic parsing, formal verification

## TL;DR

This paper systematically surveys existing work on autoformalization across mathematics, logical reasoning, planning, and knowledge representation, and proposes a unified cross-disciplinary definitional framework. Autoformalization is defined as the semantically equivalent transformation from informal language to formal reasoning languages, with the goal of facilitating methodology sharing across research communities and accelerating the development of next-generation AI reasoning systems.

## Background & Motivation

### Problem Definition

Autoformalization was originally introduced in 2018 to refer to the automatic conversion of informal mathematics into machine-verifiable formal languages using interactive theorem provers (e.g., Isabelle/HOL, Lean, Rocq, Mizar). With the rapid advancement of LLMs, the term has expanded beyond mathematics to broadly denote the task of converting natural language input into formal logical representations.

### Root Cause

Although autoformalization has been studied across multiple domains, their independent development has led to significant issues:

**Terminological inconsistency**: Different communities hold subtly different understandings of "autoformalization," and some researchers do not consider their work to fall under this umbrella.

**Limited method reuse**: Independent development across domains restricts opportunities to share methodologies, benchmarks, and theoretical frameworks.

**Absence of unified evaluation standards**: The lack of cross-domain evaluation criteria makes it difficult to compare the effectiveness of different approaches.

**Unreliable LLM reasoning**: LLMs as general-purpose reasoners suffer from hallucination, whereas autoformalization can leverage LLMs' strengths in "translation" while delegating reasoning to formal tools.

### Limitations of Prior Work

- Current Chain-of-Thought and reinforcement-learning-enhanced "reasoning" models remain unreliable on complex logical inference tasks.
- Reasoning processes are difficult or impossible to formally verify, posing trust issues in high-stakes applications.
- Autoformalization, by mapping informal language to formal representations, offers a more reliable and interpretable alternative.

## Method

### Overall Architecture

The authors systematically survey 81 research papers spanning four subfields and subsequently propose a unified definition.

### Key Designs

#### 1. **Literature Survey and Taxonomy**: Systematic categorization of autoformalization applications

The surveyed papers are organized into four categories:

- **Mathematical Formalization** (32 papers): Uses interactive theorem provers (Isabelle/HOL, Lean, etc.); all papers employ the term "autoformalization."
- **Logical Reasoning and Declarative Programming** (30 papers): Target formalisms include first-order logic, Prolog, ASP, and temporal logics (LTL, CTL); only 8 papers use the term.
- **Planning** (16 papers): Translates natural language into the PDDL planning language; none use the term.
- **Knowledge Representation**: Translates natural language into OWL/RDF ontologies; none use the term.

Despite terminological differences, the definitions across domains exhibit remarkable similarity—the core task in all cases is the automatic translation of natural language into a target language that supports logical and automated reasoning.

#### 2. **Unified Definition**: Formalizing the transformation from informal language $L_i$ to formal reasoning language $L_f$

**Definition 1 (Formalization)**:

> A **formalization** from informal language $L_i$ to formal reasoning language $L_f$ with respect to semantic equivalence criterion $E$ is a transformation of expressions from a domain-specific subset of $L_i$ into well-formed and valid expressions of $L_f$ that are semantically equivalent under $E$. **Autoformalization** is formalization performed automatically by a computational system.

The definition involves four core parameters:

| Parameter | Meaning | Examples |
|-----------|---------|---------|
| Informal language $L_i$ | A set of meaningful expressions | English, mathematical language |
| Domain subset of $L_i$ | A specific subset of expressions of interest | Game-theoretic descriptions, planning scenarios |
| Formal reasoning language $L_f$ | An enumerable set of expressions defined by syntactic or formation rules | Lean, FOL, PDDL, OWL |
| Semantic equivalence criterion $E$ | The standard by which informal and formal expressions "mean the same thing" | Theorem statements preserving all essential mathematical detail |

The **formal reasoning language** $L_f$ must satisfy two conditions:
- It must be capable of representing all relevant information in the domain-specific subset.
- It must support reasoning, allowing properties of interest to be verified at acceptable computational complexity.

#### 3. **Verification Criterion $V$**: A computable approximation of semantic equivalence

Since $E$ involves the meaning of informal language and is inherently ambiguous, the authors introduce a verification criterion $V$ as a computable approximation of $E$:

- $E$ defines the semantic standard by which informal and formal expressions "mean the same thing."
- $V$ provides a practical means of verifying whether a given pair satisfies this equivalence.

Verification approaches vary across domains:
- Mathematics: BLEU scores + human evaluation (costly and unscalable)
- Logical reasoning: Comparing derived answers against correct answers (automated but without guarantees of semantic equivalence)
- Game theory: Querying all action combinations and comparing payoff values (automated and practically guaranteeing semantic equivalence, but applicable only to a narrow class of problems)
- Planning: Cross-validating plan validity (a relaxed but practical notion of equivalence)
- Knowledge representation: Manual verification by ontology engineers + automated consistency checking

### Case Studies

The authors demonstrate the flexibility of the definition through five case studies:

| Case | $L_i$ | $L_f$ | Verification |
|------|--------|--------|-------------|
| Mathematical proofs | Mathematical natural language | Isabelle/HOL | BLEU + human |
| First-order logic | Questions about objects/shapes | Prover9 (FOL) | Answer comparison |
| Logic programming | Game descriptions | SWI-Prolog | Game query verification |
| Planning | Natural language action descriptions | PDDL | Plan cross-validation |
| Knowledge representation | Domain descriptions | OWL | Human + reasoner |

### Loss & Training

This paper is a survey and framework paper; no model training is involved. The core contribution is conceptual.

## Key Experimental Results

### Main Results

This is a survey and framework paper and does not include quantitative experiments in the traditional sense. The primary "experiment" consists of the systematic review of 81 papers.

| Subfield | Number of Papers | Use of "autoformalization" | Example Target Languages |
|----------|-----------------|---------------------------|--------------------------|
| Mathematical formalization | 32 | All | Isabelle, Lean, Rocq, Mizar |
| Logical reasoning | 30 | 8 | FOL, Prolog, ASP, LTL, CTL |
| Planning | 16 | 0 | PDDL |
| Knowledge representation | Several | 0 | OWL, RDF, RDFS |

### Ablation Study

| Definition Parameter | Problem if Removed | Explanation |
|----------------------|--------------------|-------------|
| Domain subset | Scope becomes too broad to evaluate | In practice, only expressions from specific domains are formalized |
| Semantic equivalence $E$ | No basis for judging correctness | "Looks correct" does not imply semantic faithfulness |
| Verification criterion $V$ | Cannot be practically tested | $E$ is generally not computable and requires approximation |

### Key Findings

1. **Cross-domain consistency**: Despite terminological differences, the definitions across the four subfields are fundamentally similar.
2. **Semantic verification is the core challenge**: Fully automated semantic equivalence verification is in principle impossible given the inherently informal nature of natural language, but meta-level reasoning can substantially reduce the need for human supervision.
3. **Role of LLMs**: Autoformalization leverages LLMs' strengths in translation rather than direct reasoning, representing a more reliable paradigm.

## Highlights & Insights

1. **Insightful observation**: The paper identifies that different communities are effectively working on the same problem without recognizing each other—a valuable cross-disciplinary perspective.
2. **Precise definition**: The proposed four-parameter definition ($L_i$, domain subset, $L_f$, $E$) is sufficiently general to cover all existing subfields while remaining concrete enough to be useful.
3. **Practical significance**: The framework can facilitate methodology transfer—for example, the automated PDDL verification techniques from planning research may inform evaluation in mathematical formalization.
4. **Strategic vision**: Positioning autoformalization as a bridge between LLMs and formal reasoning tools, rather than relying on LLMs for direct reasoning, reflects a clear and well-motivated perspective.

## Limitations & Future Work

1. **Lack of quantitative validation**: The purely conceptual framework is not empirically validated to demonstrate that it actually facilitates cross-domain methodology transfer.
2. **Survey completeness**: The authors acknowledge that the survey does not claim to be exhaustive.
3. **Operationalizability of the definition**: Although the definition is general, instantiating its parameters in specific domains still requires substantial domain expertise.
4. **Semantic verification remains an open problem**: The authors acknowledge that fully automated semantic verification is in principle impossible but do not propose a concrete path forward.
5. **Scalability**: Current systems perform adequately on curated benchmarks but frequently fail in real-world practice.

## Related Work & Insights

- **Semantic parsing**: Autoformalization can be viewed as an instance of semantic parsing, but with the target language restricted to formal reasoning and automated inference languages.
- **AI for Mathematics**: Closely related to the AI for Math research direction, but this paper's contribution lies in extending the scope beyond mathematics.
- **Neuro-symbolic AI**: Autoformalization serves as an important bridge between neural networks and symbolic reasoning.
- **Insights**: Verification strategies across domains can inform one another—for example, automated verification methods from game theory may be applicable in other subfields.

## Rating

- Novelty: ⭐⭐⭐⭐ (The unified framework perspective is novel, though there is no technical innovation)
- Experimental Thoroughness: ⭐⭐⭐ (Survey in nature; case analyses are thorough but no quantitative experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rigorous definitions, rich case studies)
- Value: ⭐⭐⭐⭐ (Significant contribution to promoting cross-disciplinary exchange)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees](../../ICML2026/llm_evaluation/decompose_structure_and_repair_a_neuro-symbolic_framework_for_autoformalization_.md)
- [\[ACL 2026\] Common to Whom? Regional Cultural Commonsense and LLM Bias in India](../../ACL2026/llm_evaluation/common_to_whom_regional_cultural_commonsense_and_llm_bias_in_india.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)
- [\[AAAI 2026\] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)
- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](../../ACL2026/llm_evaluation/beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)

</div>

<!-- RELATED:END -->
