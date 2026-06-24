---
title: >-
  [Paper Note] Semantic-Aware Logical Reasoning via a Semiotic Framework
description: >-
  [ACL 2026][Reasoning][Symbolic Reasoning] Proposes LogicAgent, a logical reasoning framework based on the Greimas Semiotic Square, achieving SOTA logical reasoning performance under the dual challenges of semantic and logical complexity through multi-perspective semantic analysis and reflective verification.
tags:
  - "ACL 2026"
  - "Reasoning"
  - "Symbolic Reasoning"
  - "Greimas Semiotic Square"
  - "Logical Reasoning"
  - "Semantic Complexity"
  - "Multi-perspective Reasoning"
date: 2026-05-08
content_hash: 1d11a411edd04639
---

# Semantic-Aware Logical Reasoning via a Semiotic Framework

**Conference**: ACL 2026  
**arXiv**: [2509.24765](https://arxiv.org/abs/2509.24765)  
**Code**: [GitHub](https://github.com/AI4SS/Logic-Agent)  
**Area**: LLM Reasoning / Logical Reasoning  
**Keywords**: Symbolic Reasoning, Greimas Semiotic Square, Logical Reasoning, Semantic Complexity, Multi-perspective Reasoning

## TL;DR

Proposes LogicAgent, a logical reasoning framework based on the Greimas Semiotic Square, achieving SOTA logical reasoning performance under the dual challenges of semantic and logical complexity through multi-perspective semantic analysis and reflective verification.

## Background & Motivation

**Background**: The logical reasoning ability of LLMs is one of their core capabilities. Existing methods are mainly categorized into three types: linear reasoning (CoT), aggregate reasoning (multi-trajectory aggregation like ToT/CR), and symbolic reasoning (combining FOL solvers like Logic-LM). These methods perform well on benchmarks with clear logical structures.

**Limitations of Prior Work**: Almost all existing methods focus on **logical complexity** (reasoning depth, number of steps) while neglecting **semantic complexity** (abstract propositions, ambiguous contexts, opposing stances). In real-world reasoning, semantic ambiguity and abstraction are often intertwined with logical complexity—for example, a philosophical proposition like "Is justice always beneficial?" requires not only deep reasoning but also multi-perspective interpretation of abstract concepts.

**Key Challenge**: Most existing benchmarks (ProntoQA, ProofWriter, etc.) are generated based on templates with clear and unambiguous propositions, failing to test the model's reasoning robustness in semantically complex scenarios. In the real world, the coupling of semantic and logical complexity is the true difficulty of reasoning.

**Goal**: Construct a reasoning framework that simultaneously addresses semantic and logical complexity and provide a benchmark to evaluate this coupled challenge.

**Core Idea**: Borrow from the Greimas Semiotic Square in structuralist semantics—expanding a proposition into a quaternary structure (original $S_1$, contradiction $\lnot S_1$, contrary $S_2$, and contradiction of the contrary $\lnot S_2$)—to perform reasoning and cross-verification from multiple perspectives, thereby enhancing reasoning robustness under semantic ambiguity.

## Method

### Overall Architecture

LogicAgent aims to solve the problem where single-perspective reasoning easily gets locked into one interpretation when semantic and logical complexities are intertwined. It first expands a proposition into the quaternary structure of the Greimas Semiotic Square, then performs formal deductive reasoning for each perspective, and finally uses the structural relationships inherent in the square for cross-arbitration. The pipeline consists of three stages: **Semantic Structuring** expands proposition $S_1$ into four related propositions $\lnot S_1$, $S_2$, and $\lnot S_2$ and verifies FOL consistency; **Logical Reasoning** translates premises into FOL, plans paths, and performs step-by-step deduction for each perspective's judgment; **Reflective Verification** uses a three-tier progressive mechanism to compare conclusions across perspectives and output a consistent final answer.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Original Prop S1"] --> SEM
    subgraph SEM["Semantic Structuring Stage"]
        direction TB
        B["Construct Quaternary Structure via Rules<br/>¬S1 / S2 / ¬S2 (Covers 6 Logical Forms)"] --> C["EIC + Triple Filtering (Truth Table / CFG / LLM)"]
    end
    SEM --> LOG
    subgraph LOG["Logical Reasoning Stage"]
        direction TB
        D["Translator: NL Premises → FOL"] --> E["Planner: Set Goals / Select Premises / Define Rules"]
        E --> F["Solver: Step-by-step Deduction<br/>True/False/Uncertain for 4 Perspectives"]
    end
    LOG --> G
    subgraph REF["Reflective Verification Stage"]
        direction TB
        G{"Judgment Relation S1 & ¬S1"}
        G -->|Complementary| H["Direct Parsing: Accept Directly"]
        G -->|One is Uncertain| I["Fast Reflection: Check Consistency via Trajectory"]
        G -->|Same Judgment (Contradiction)| J["Deep Reflection: Arbitration via S2⇒¬S1 Stance"]
    end
    H --> K["Final Answer"]
    I --> K
    J --> K
```

### Key Designs

**1. Semantic Structuring: Expanding a Single Proposition into a Quaternary Semantic Space to Reveal Latent Ambiguity**

Natural language propositions often imply multiple interpretations; locking into one too early can miss opposing stances—for instance, in "Is justice always beneficial?", both affirmation and negation have merits. This stage takes the original proposition $S_1$ and constructs its contradictory proposition $\lnot S_1$ (strict negation), contrary proposition $S_2$ (cannot both be true but can both be false), and the contradiction of the contrary $\lnot S_2$ based on a unified rule table. The table uses 6 rules covering universal, existential, implication, conjunction, disjunction, and biconditional forms. To avoid "vacuous truth" loopholes in the null domain for contrary relations, an Existential Import Check (EIC) is introduced to ensure logical correctness. All candidate propositions then pass through triple filtering (truth table verification, CFG syntax check, and LLM semantic verification) to ensure the quaternary structure is both syntactically valid and semantically relevant. Consequently, subsequent reasoning unfolds in a structured multi-perspective space rather than focusing solely on the original proposition.

**2. Logical Reasoning: Formal Symbolic Deduction for Each Proposition in the Square**

Pure LLM reasoning is unreliable and prone to skipping steps or self-contradiction; therefore, this stage delegates reasoning to three functional units with clear divisions of labor. The Translator converts natural language premises into FOL using unified mapping specifications (entities to unary predicates, actions to binary predicates, and evaluative properties to predicates); the Planner constructs a reasoning blueprint, sets goals, selects relevant premises, and identifies reasoning rules to be used (e.g., Modus Ponens / Modus Tollens); the Solver performs step-by-step deduction according to the blueprint, outputting transparent reasoning trajectories and a three-valued judgment of True / False / Uncertain. The language understanding of the LLM grounds ambiguous natural language into symbols, while the rigorous deduction of symbolic logic ensures every step is traceable, mutually compensating for the unreliability of end-to-end LLMs.

**3. Reflective Verification: Cross-Arbitration via Structural Relations of the Square to Resolve Inconsistencies**

After the four perspectives provide their judgments, how can a reliable conclusion be synthesized? This stage designs a three-tier progressive mechanism to precisely match different inconsistency patterns. When $S_1$ and $\lnot S_1$ yield complementary judgments (one true, one false), **Direct Parsing** is used to accept the result; when one side is Uncertain, **Fast Reflection** is triggered, letting the LLM review the reasoning trajectory to check internal consistency; if $S_1$ and $\lnot S_1$ yield the same judgment (a contradiction), **Deep Reflection** is triggered, utilizing the entailment relationships $S_1 \Rightarrow \lnot S_2$ and $S_2 \Rightarrow \lnot S_1$ of the square to bring in the reasoning results of $S_2$ and $\lnot S_2$ for arbitration. The three structural relations of the semiotic square—contradiction, contrariety, and entailment—form a natural cross-validation network that exposes and corrects contradictions when reasoning errors occur.

### Loss & Training

LogicAgent is a training-free reasoning framework implemented via prompt engineering based on existing LLMs (Qwen2.5-32B, GPT-4o). CFG syntax checking uses the `nltk` library, and the decoding temperature is set to 0.

## Key Experimental Results

### Main Results

| Benchmark | LogicAgent | Best Baseline | Gain |
|------|-----------|---------|------|
| RepublicQA (Qwen2.5) | 82.50 | 76.00 (SymbCoT) | +6.50 |
| RepublicQA (GPT-4o) | 87.00 | 82.50 (Aristotle) | +4.50 |
| ProntoQA | 97.80 | 95.20 (SymbCoT) | +2.60 |
| ProofWriter | 71.95 | 64.67 (SymbCoT) | +7.28 |
| FOLIO | 79.90 | 72.54 (ToT) | +7.97 |
| ProverQA | 68.60 | 62.40 (Logic-LM) | +6.20 |
| **Average** | **79.56** | - | **+7.05** |

### Ablation Study

| Configuration | Avg | Description |
|------|-----|------|
| Full LogicAgent | 76.36 | Complete Model |
| w/o Square | 67.58 | Largest drop; multi-perspective reasoning is crucial |
| w/o Plan | 69.70 | Planning significantly aids complex reasoning |
| w/o Reflect | - | Reflective verification further improves reliability |

### Key Findings
- The semantic complexity indicators of RepublicQA surpass existing benchmarks (FKGL=11.94 at college level, contrary construction rate of 0.70 far exceeding 0-0.30 of others).
- Logic-LM performs close to naive baselines on RepublicQA, indicating that pure symbolic augmentation fails under semantic ambiguity.
- The semiotic square provides the greatest contribution (an average drop of ~8.8 points when removed), validating the core value of multi-perspective reasoning.
- LogicAgent consistently improves performance on both simple (ProntoQA) and complex (ProverQA) benchmarks, demonstrating good generalization.

## Highlights & Insights
- **Interdisciplinary Fusion of Linguistic Theory and AI Reasoning**: Migrating the Greimas Semiotic Square from structuralist semantics to computational logical reasoning possesses both theoretical depth and practical effectiveness.
- **First Systematic Approach to Semantic Complexity**: Defined multi-dimensional semantic complexity metrics and constructed a dedicated benchmark, filling an important gap.
- **Progressive Design of Three-tier Reflection**: Ranging from direct parsing to fast reflection and then deep reflection, it precisely matches different inconsistency patterns.
- **Rigor of Existential Import Check (EIC)**: Ensures the logical correctness of contrary relations within the FOL framework, avoiding logical loopholes in null domains.

## Limitations & Future Work
- RepublicQA focuses on philosophical/ethical domains; coverage of scientific and commonsense reasoning is limited.
- The framework depends on the LLM's ability to correctly execute FOL translation and semiotic square construction; weak models may produce low-quality intermediate results.
- Deep reflection introduces additional reasoning overhead (requiring complete reasoning for $S_2$ and $\lnot S_2$).
- The three-valued logic (True/False/Uncertain) setting might not be flexible enough; probabilistic reasoning could be explored in the future.
- Future work could combine the semiotic square with test-time compute.

## Related Work & Insights
- **vs SymbCoT**: SymbCoT combines CoT and symbolic reasoning but lacks multi-perspective verification; LogicAgent improves significantly through cross-verification of the semiotic square.
- **vs Logic-LM**: Logic-LM directly calls FOL solvers, which has limited effectiveness under semantic ambiguity; LogicAgent addresses ambiguity via semantic structuring first.
- **vs Aristotle**: Aristotle combines aggregation and symbolic reasoning but lacks a systematic reflection mechanism; LogicAgent's three-tier reflection is more effective at contradiction detection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The application of the Greimas Semiotic Square in AI reasoning is highly original, and the RepublicQA benchmark is a unique contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, multiple baselines, and ablation analysis, though model coverage (only 2 LLMs) is slightly limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, with clear definitions and theorems, though the abundance of symbols creates a high barrier to entry.
- Value: ⭐⭐⭐⭐ Introduces the dimension of semantic complexity to logical reasoning; both the framework and benchmark provide independent contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)
- [\[ACL 2025\] Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](../../ACL2025/llm_reasoning/aristotle_logical_reasoning.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ICLR 2026\] ActivationReasoning: Logical Reasoning in Latent Activation Spaces](../../ICLR2026/llm_reasoning/activationreasoning_logical_reasoning_in_latent_activation_spaces.md)

</div>

<!-- RELATED:END -->
