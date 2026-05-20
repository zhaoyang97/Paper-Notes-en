---
title: >-
  [Paper Note] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy
description: >-
  [AAAI 2026][Robotics][NL-FOL translation] This paper critically examines existing evaluation methodologies for natural language to first-order logic (FOL) translation — specifically FOLIO and MALLS — exposing fundamental…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "NL-FOL translation"
  - "autoformalization"
  - "first-order logic"
  - "LLM evaluation"
  - "benchmark"
  - "semantic understanding"
date: 2026-05-08
content_hash: d728104deb56d020
---

# Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy

**Conference**: AAAI 2026
**arXiv**: [2511.11816](https://arxiv.org/abs/2511.11816)  
**Code**: [dslab-uniud/NL-FOL-LT](https://github.com/dslab-uniud/NL-FOL-LT)  
**Area**: Natural Language Processing / Formal Logic Reasoning
**Keywords**: NL-FOL translation, autoformalization, first-order logic, LLM evaluation, benchmark, semantic understanding

## TL;DR

This paper critically examines existing evaluation methodologies for natural language to first-order logic (FOL) translation — specifically FOLIO and MALLS — exposing fundamental flaws in their datasets and evaluation protocols. The authors propose a novel benchmarking strategy that decomposes the translation task into ontology extraction (OE) and logical translation (LT), augmented with "most similar selection" and "ranking" subtasks. Experiments demonstrate that conversational LLMs (o3-mini, GPT-4o-mini, Qwen3 series) exhibit strong NL-FOL translation capabilities and genuine logical semantic understanding, while embedding-based models perform significantly worse.

## Background & Motivation

- **Background**: First-order logic (FOL), owing to its unambiguity and expressive power, serves as a robust formal system for representing natural language concepts, with important applications in system property specification and verification, AI safety monitoring, and related domains — for instance, automatically translating LLM outputs into logical formulae for real-time monitoring, or assisting in the construction of formal world models and safety specifications.

- **Limitations of Prior Work**: FOLIO reports GPT-4 zero-shot accuracy at approximately 52%, while MALLS claims a logical equivalence (LE) score of approximately 80% for the same model — two of the most comprehensive studies in the field reaching diametrically opposed conclusions, underscoring the urgent need to clarify LLMs' actual capabilities in NL-FOL translation.

- **Key Challenge**:
    - *Flaws in FOLIO's evaluation protocol*: FOLIO equates translation correctness with reasoning conclusion correctness (i.e., correctness is satisfied as long as the entailment relation from premises to conclusion is preserved), which constitutes a coarse proxy metric — whether all sentences are mistranslated or only one is, a failed reasoning step yields the same score, making it impossible to precisely measure translation quality.
    - *Fundamental flaw in MALLS' LE score*: MALLS' logical equivalence (LE) score treats FOL formulae as propositional logic, assigning fixed truth values to predicate symbols and generating truth tables — a theoretically incorrect approach. For example, existentially and universally quantified formulae receive identical LE scores (=1), entirely disregarding quantifier semantics; semantic relationships among predicates are also ignored.
    - *Annotation errors in MALLS dataset*: The annotation guidelines for MALLS' manually verified test set contain serious issues — inconsistent rules for quantifier usage (suggesting that "a turtle has a shell" may use either existential or universal quantifiers) and the erroneous claim that logical connectives are interchangeable (→, ∧, ↔ are "sometimes interchangeable"), resulting in theoretically incorrect ground truth labels.
    - *Lack of mechanisms to distinguish genuine understanding from surface pattern matching*: Existing evaluation protocols conflate ontology extraction and logical translation, making it impossible to determine whether a model truly understands logical semantics or achieves high scores through memorization, pattern matching, or dataset contamination.

## Method

### Core Design: Decomposing NL-FOL Translation into Two Stages with Multi-Dimensional Probing

**Stage 1 — Ontology Extraction (OE)**: Identifies the logical signature (predicates, functions, constants) and associates semantic meanings with each symbol.

**Stage 2 — Logical Translation (LT)**: Given a fixed signature, defines FOL formulae that capture the semantics of the natural language input.

This decomposition is critical: (i) it enables separate diagnosis of which stage a model fails at; (ii) fixing the signature allows automated equivalence verification via SMT solvers; (iii) it is applicable to practical scenarios where domain experts provide predefined ontologies.

### Three Evaluation Tasks

**1. Logical Translation Task**: Given a triple $(p, \varphi, \Omega=(\sigma,\gamma))$, the model receives natural language $p$ and ontology $\Omega$, generates formula $\varphi'$, and the Z3 SMT solver verifies whether $\varphi' \equiv \varphi$. Providing the ontology is an intentional design choice to isolate logical translation ability and eliminate ontology extraction as a confounding factor.

**2. Most Similar Selection Task**: Eight types of random perturbations are applied to the reference formula $\varphi$ (replacing Boolean connectives such as ∧↔∨, switching quantifiers ∀↔∃, adding/removing negations) to construct a candidate set $\mathcal{F}_{ms}$, from which the model selects the formula semantically closest to $p$. Both FOL variants and NL variants (obtained by translating formulae into English sentences via a translation function T()) are included. The task is inherently resistant to data contamination and memorization through online random generation of candidates.

**3. Ranking Task**: A candidate set $\mathcal{F}_r$ is constructed containing the original formula $\varphi$, three perturbations, an equivalent transformation $\varphi_{eq}$ (generated via random applications of De Morgan's laws, double negation, commutativity, distributivity, implication expansion, etc.), the negation $\neg\varphi$, and its negation normal form $(\neg\varphi)_{nnf}$. The model ranks candidates by semantic similarity. Success criteria: (i) Ranking-Eq: $\varphi$ and $\varphi_{eq}$ ranked in the top two; (ii) Ranking-Neg: $\neg\varphi$ and $(\neg\varphi)_{nnf}$ ranked in the bottom two; (iii) Ranking-Both: both conditions satisfied simultaneously.

### Models and Datasets

- **Conversational models**: GPT-4o-mini, o3-mini, Qwen3-8B, Qwen3-30B-A3B (all using thinking mode)
- **Embedding models**: Qwen3-Embedding-8B (plain/inst variants), Gemini-Embedding-001
- **Datasets**: $\mathcal{D}_{Stanford}$ (159 high-quality private instances from the Grade Grinder logic teaching platform); $\mathcal{D}_{FOLIO}$ (1,565 instances from the FOLIO training set with XOR removed)

## Key Experimental Results

### Table 1: Conversational Model Logical Translation Performance (Z3 Equivalence Verification, Mean ± Std. Dev. over 5 Runs)

| Dataset | GPT-4o-mini | o3-mini | Qwen3-8B | Qwen3-30B |
|---|---|---|---|---|
| $\mathcal{D}_{Stanford}$ | .84±.03 | **.94±.00** | .84±.01 | .85±.01 |
| $\mathcal{D}_{FOLIO}$ | .73±.01 | **.80±.00** | .72±.00 | .74±.01 |

**Key Findings**: o3-mini leads by a substantial margin with 94%/80% translation accuracy. All models perform 11–14 percentage points higher on Stanford than on FOLIO, partly attributable to approximately 6% annotation errors in the FOLIO training set (systematic inspection of 302 instances where both models agreed on an incorrect answer revealed 93 ground truth errors).

### Table 2: Conversational vs. Embedding Models — Multi-Task Performance Comparison (FOL Variants)

| Task | o3-mini | GPT-4o-mini | Qwen3-30B | Qwen-Emb-inst | Gemini-Emb |
|---|---|---|---|---|---|
| Most Similar (Stanford) | **1.00** | .88 | .98 | .55 | .52 |
| Most Similar (FOLIO) | **.95** | .89 | .94 | .76 | .49 |
| Ranking-Eq (Stanford) | **.98** | .57 | .91 | .31 | .29 |
| Ranking-Neg (Stanford) | **.91** | .51 | .62 | .71 | .64 |
| Ranking-Both (Stanford) | **.89** | .32 | .57 | .23 | .20 |
| Ranking-Both (FOLIO) | **.82** | .44 | .66 | .37 | .32 |

**Key Findings**: (1) Conversational models comprehensively outperform embedding models. (2) o3-mini approaches perfection on the most similar task (1.00). (3) The ranking task offers far greater discriminative power than the most similar task — GPT-4o-mini achieves only 32% on Ranking-Both, far behind o3-mini's 89%. (4) Embedding models perform substantially worse on FOL variants than on NL variants, while conversational models show the opposite pattern, indicating that the latter can reliably parse FOL semantics. (5) Ranking-Eq is generally easier than Ranking-Neg, suggesting models are better at identifying equivalent formulae than negations. (6) Traditional BLEU and LE scores exhibit low and unstable correlation with Z3 equivalence verification (point-biserial correlation $r_{pb}$ fluctuates between .44 and .83), confirming their unreliability.

## Highlights & Insights

- **Theoretically grounded evaluation framework**: The paper is the first to systematically critique the methodological flaws of the two most influential works in NL-FOL translation from a theoretical perspective, proposing a principled alternative based on reliable SMT solver verification.
- **OE/LT decoupling design**: Decomposing the translation task into ontology extraction and logical translation enables more precise and controllable evaluation, and better reflects practical settings where domain experts provide predefined ontologies.
- **Contamination-resistant multi-dimensional probing**: The most similar and ranking tasks naturally resist data contamination and memorization bias through online random generation of perturbed candidate sets, effectively distinguishing genuine semantic understanding from surface pattern matching.
- **Cross-paradigm comparison**: The paper is the first systematic comparison of conversational and embedding-based LLMs on NL-FOL translation, revealing significant deficiencies of embedding models in formal logic comprehension.
- **Uncovering dataset quality issues**: Systematic analysis reveals at least 6% annotation errors in the FOLIO training set, providing an important warning for subsequent research.

## Limitations & Future Work

- **Only LT is evaluated**: The paper assumes a known ontology and does not evaluate ontology extraction (OE) capability, which is equally critical in practical applications.
- **No prompt engineering optimization**: All experiments use zero-shot automatic CoT without exploring few-shot prompting, system prompt tuning, or other performance-enhancing strategies; the reported results represent a lower bound on model capability.
- **Limited dataset scale and coverage**: Only two datasets are used (159 Stanford instances + 1,565 FOLIO instances), and the Stanford dataset is private and cannot be publicly reproduced.
- **Restricted FOL complexity range**: More challenging scenarios involving higher-order logic, nested quantifiers, and function symbols are not addressed.
- **Preliminary evaluation of embedding models**: Only three embedding model configurations are tested; other similarity metrics are not explored in depth.
- **Natural language ambiguity not considered**: The evaluation assumes each NL sentence has a unique correct FOL translation, whereas ambiguity is pervasive in practice.

## Related Work & Insights

- **Autoformalization**: The automatic translation of natural language into a given formal system; early work focused on formalizing mathematical proofs (Wu et al. 2022; ProofNET) in Lean/Isabelle, with recent extensions to formal languages such as SQL and LTL.
- **Evolution of NL-FOL translation methods**: Approaches have evolved from rule-based methods (Abzianidze 2017; Bos & Markert 2005) to BERT/RoBERTa (Tian et al. 2021) and then to LLMs (Lu et al. 2022; MALLS), yet evaluation standards remain inconsistent across the literature.
- **FOLIO benchmark**: Proposes a joint NLI and formalization evaluation framework with 487 stories, but suffers from coarse evaluation granularity.
- **MALLS benchmark**: Uses GPT-4-generated synthetic NL-FOL pairs (28K instances) and proposes LE and BLEU metrics, whose theoretical foundations are refuted in this work.
- **Reasoning-enhanced pipelines**: NL-FOL translation has been used to enhance general reasoning (Pan et al. 2023; Ye et al. 2023) and automatic logical fallacy detection (Lalwani et al. 2025).
- **Grade Grinder corpus**: The Stanford dataset used in this paper originates from student submissions on the *Language, Proof and Logic* teaching platform between 2001 and 2010.

## Rating

| Dimension | Score | Notes |
|---|---|---|
| Novelty | ★★★★☆ | Innovation lies in evaluation methodology rather than algorithmic contribution; the OE/LT decomposition and multi-dimensional probing design are highly original |
| Technical Depth | ★★★★☆ | Theoretical critique of FOLIO and MALLS is rigorous; the Z3 solver replacement of traditional metrics rests on solid theoretical foundations |
| Experimental Thoroughness | ★★★★☆ | 6 models × 2 datasets × 5 runs × multiple tasks provides comprehensive coverage, though dataset scale is limited |
| Writing Quality | ★★★★★ | Argumentation is clear and coherent, with a complete logical chain from problem analysis to solution design to experimental validation |
| Impact | ★★★★☆ | Establishes more reliable evaluation standards for the NL-FOL translation community, with potential to advance formal verification in AI safety |
| Overall | 8.2/10 | A high-quality contribution focused on evaluation methodology reform, providing an important new perspective for assessing LLMs' logical reasoning capabilities |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](../../ICLR2026/robotics/whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)
- [\[NeurIPS 2025\] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs](../../NeurIPS2025/robotics/toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)
- [\[AAAI 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[ICLR 2026\] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](../../ICLR2026/robotics/socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/robotics/tracing_and_reversing_edits_in_llms.md)

</div>

<!-- RELATED:END -->
