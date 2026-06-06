---
title: >-
  [Paper Note] Language Model as Planner and Formalizer under Constraints
description: >-
  [ACL 2026][LLM Reasoning][Constrained Planning] This paper proposes the CoPE benchmark, which injects formally categorized natural language constraints into classic planning environments. It reveals that a single sentenc…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Constrained Planning"
  - "LLM-as-Planner"
  - "LLM-as-Formalizer"
  - "Benchmark"
  - "PDDL"
date: 2026-05-08
content_hash: a9bfba3c29545926
---

# Language Model as Planner and Formalizer under Constraints

**Conference**: ACL 2026  
**arXiv**: [2510.05486](https://arxiv.org/abs/2510.05486)  
**Code**: [GitHub](https://github.com/CassieHuang22/LLM-as-Formalizer-constraints)  
**Area**: LLM Evaluation  
**Keywords**: Constrained Planning, LLM-as-Planner, LLM-as-Formalizer, Benchmark, PDDL

## TL;DR

This paper proposes the CoPE benchmark, which injects formally categorized natural language constraints into classic planning environments. It reveals that a single sentence of constraint can halve the planning performance of the strongest current LLMs, exposing a severe lack of robustness in LLM planning.

## Background & Motivation

**Background**: There are two mainstream paradigms for LLMs in the planning field: LLM-as-Planner, which generates action sequences directly end-to-end, and LLM-as-Formalizer, which converts natural language descriptions into formal languages like PDDL before using a solver to derive solutions. Both methods have shown impressive capabilities on standard planning benchmarks.

**Limitations of Prior Work**: However, existing benchmarks (such as BlocksWorld, Gripper, etc.) have mostly existed for decades, featuring simple and homogeneous environment descriptions that are highly likely to be covered by LLM training data. This simplicity may lead to a significant **overestimation** of LLM planning capabilities, posing risks in downstream safety-sensitive scenarios.

**Key Challenge**: Real-world planning instructions typically include **personalized requirements and constraints** imposed by users or resources, whereas standard benchmarks completely lack these elements. Existing enhancement methods only add noise or lexical perturbations without changing the semantics itself.

**Goal**: To build a constrained planning benchmark enhanced at the semantic level to systematically evaluate the planning and formalization capabilities of LLMs under constraints. **Key Insight**: Formalize constraints into four categories (Initial, Goal, Action, State) using linguistic and pragmatic approaches to ensure completeness. **Core Idea**: A simple one-sentence constraint can significantly degrade LLM performance, and this degradation is further exacerbated as problem complexity increases and vocabulary becomes confused.

## Method

### Overall Architecture

CoPE (Constrained Planning Environments) manually labels natural language constraints and their ground-truth encodings in four formal languages for each problem in the BlocksWorld and CoinCollector domains. Evaluation process: Given a domain description $D_d$, problem description $D_p$, PDDL header $\mathcal{DF}'$, and constraint $\mathcal{C}$, the LLM generates a plan (Planner) or formal code (Formalizer). Finally, the VAL validator is used to verify the correctness of the plan.

### Key Designs

1.  **Formal Definition of Four Constraint Categories**:
    - Function: Strictly classifies natural language constraints into Initial (modifying the initial state), Goal (modifying the goal state), Action (restricting legal action sequences), and State (restricting legal state trajectories).
    - Mechanism: Defined based on the set relationship between the original action/state space (primitive) and the modified space (modified). It is proven that the State constraint subclass covers all possible constraints, ensuring **completeness**.
    - Design Motivation: Different formal languages (PDDL, PDDL3, LTL, SMT) have varying expressive power for different constraint categories. Formal classification supports systematic analysis.

2.  **Comparative Evaluation of Multiple Formal Languages**:
    - Function: Encodes constraints into PDDL 1.2, PDDL3, LTL, and SMT (Z3) to evaluate the expressive and solving capabilities of each formal language.
    - Mechanism: Employs three technical routes: Generation (direct generation), Editing (generating unconstrained code first then editing), and Revision (up to 3 iterations of syntax error correction).
    - Design Motivation: Different constraint types are naturally suited to different formal languages; for example, PDDL3 excels in state constraint syntax, while SMT is proficient in modeling state predicates. Systematic comparison provides guidance for future toolchain selection.

3.  **Robustness Extension Experiments**:
    - Function: Evaluates complexity scaling and data contamination via BlocksWorld-XL (50 blocks) and MysteryBlocksWorld (lexical confusion).
    - Mechanism: The XL version tests performance after increasing the entity space, while the Mystery version replaces all types, predicates, and action names with meaningless placeholders.
    - Design Motivation: To verify whether constraints amplify the existing vulnerabilities of LLMs under complex problems and lexical perturbations.

### Loss & Training

This work is an evaluation study and does not involve model training. The core evaluation metric is **plan correctness**—whether the predicted plan can successfully transition from the initial state to the goal state within the ground-truth PDDL environment.

## Key Experimental Results

### Main Results

| Dataset | Method | Unconstrained | Constrained | Drop |
|--------|------|--------|--------|----------|
| BlocksWorld | LLM-as-Planner (Gemini-3-Flash) | ~85% | ~55% | ~30% |
| BlocksWorld | LLM-as-PDDL-Formalizer (Gemini) | ~70% | ~40% | ~30% |
| CoinCollector | LLM-as-Planner (Gemini) | ~90% | ~60% | ~30% |
| BlocksWorld | PDDL3 Formalizer | Lower than PDDL | Even lower | High syntax/compile errors |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Generation | Baseline | Direct generation of constraint code |
| Editing | Partial Improvement | Generating unconstrained version first before editing |
| Revision | Further Improvement | Iterative correction based on solver errors |
| BlocksWorld-XL (50 blocks) | Performance Plunge | Constraint impact becomes more severe as complexity scales |
| MysteryBlocksWorld | Formalizer Robustness Disappears | Double blow from constraints and lexical confusion |

### Key Findings
- A single sentence of constraint consistently halves performance; all LLM, method, and language combinations are affected.
- LLM-as-Planner generally outperforms Formalizer when unconstrained, but Formalizer is more robust to problem complexity.
- Although PDDL3 has syntax support for constraints, it performs worse than standard PDDL due to scarce training data.
- Once constraints are introduced, the original robustness of the Formalizer against complexity and lexical perturbation **completely disappears**.

## Highlights & Insights
- The formal definition of constraint classification is very rigorous and includes a proof of completeness, serving as a theoretical foundation for subsequent work.
- Experimental design covers 4 LLMs × 4 formal languages × 3 techniques × 4 constraint types × 4 datasets, providing rich dimensions of analysis.
- It reveals an important conclusion: **Simple semantic modifications challenge LLMs more effectively than lexical noise**, offering new ideas for benchmark design.
- CoPE's design philosophy—fighting data contamination through semantic enhancement rather than data perturbation—is worth adopting in other NLP evaluation tasks.

## Limitations & Future Work
- Constraint types only consider single constraints, without discussing conjunctions, negations, or ambiguities of constraints; real-world constraints are more diverse.
- BlocksWorld and CoinCollector domains are still relatively simple, with a large gap from real-world planning scenarios (e.g., robotic manipulation, resource scheduling).
- The evaluation metric (plan correctness) may have false positives—plans being coincidentally correct without the code truly encoding the constraint—though verification shows this proportion is negligible.
- Future directions: Supporting more complex combinations of constraints, expanding to more domains, and developing constraint-aware planning toolchains.
- The safety risks of autonomous agents in downstream tasks deserve attention; formal representations provide transparency for human auditing and formal verification.

## Related Work & Insights
- **vs. Standard IPC Benchmarks**: CoPE challenges LLMs through semantic modification rather than just adding noise, exposing true capabilities more effectively.
- **vs. LLM+P (Liu et al., 2023)**: Both follow the Formalizer route, but LLM+P does not consider constraints; CoPE reveals its limitations.
- **vs. Mystery BlocksWorld**: CoPE demonstrates that constraints weaken the robustness of Formalizers more than lexical confusion.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic evaluation benchmark for constrained planning in LLMs, with rigorous formal classification.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed analysis covering multiple models, languages, techniques, and domains.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions, logical structure, and rich visualization.
- Value: ⭐⭐⭐⭐ Sounds an alarm for LLM planning research and points out the important research direction from simple benchmarks to realistic constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](../../ICLR2026/llm_reasoning/estimating_the_empowerment_of_language_model_agents.md)
- [\[ICML 2026\] On the Generalization Gap in Self-Evolving Language Model Reasoning](../../ICML2026/llm_reasoning/on_the_generalization_gap_in_self-evolving_language_model_reasoning.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)

</div>

<!-- RELATED:END -->
