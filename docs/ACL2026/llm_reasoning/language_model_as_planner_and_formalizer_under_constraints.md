---
title: >-
  [Paper Note] Language Model as Planner and Formalizer under Constraints
description: >-
  [ACL 2026][Reasoning][Constrained Planning] This paper proposes the CoPE benchmark, which injects formally classified natural language constraints into classic planning environments. It reveals that a single constraint can halve the planning performance of current state-of-the-art LLMs, exposing a severe lack of robustness in LLM planning.
tags:
  - "ACL 2026"
  - "Reasoning"
  - "Constrained Planning"
  - "LLM-as-Planner"
  - "LLM-as-Formalizer"
  - "Benchmarking"
  - "PDDL"
date: 2026-05-08
content_hash: da9cc9554d09ed8e
---

# Language Model as Planner and Formalizer under Constraints

**Conference**: ACL 2026  
**arXiv**: [2510.05486](https://arxiv.org/abs/2510.05486)  
**Code**: [GitHub](https://github.com/CassieHuang22/LLM-as-Formalizer-constraints)  
**Area**: LLM Evaluation  
**Keywords**: Constrained Planning, LLM-as-Planner, LLM-as-Formalizer, Benchmarking, PDDL

## TL;DR

This paper proposes the CoPE benchmark, which injects formally classified natural language constraints into classic planning environments. It reveals that a single constraint can halve the planning performance of current state-of-the-art LLMs, exposing a severe lack of robustness in LLM planning.

## Background & Motivation

**Background**: There are two mainstream paradigms for LLM planning: LLM-as-Planner directly generates action sequences end-to-end, while LLM-as-Formalizer translates natural language descriptions into formal languages like PDDL for solver-based derivation. Both have demonstrated impressive capabilities on standard planning benchmarks.

**Limitations of Prior Work**: However, existing benchmarks (e.g., BlocksWorld, Gripper) were mostly created decades ago, featuring simple and homogeneous environment descriptions that are highly likely covered by LLM training data. This simplicity may lead to an **overestimation** of LLM planning capabilities, posing risks in downstream safety-sensitive scenarios.

**Key Challenge**: Real-world planning instructions typically involve **personalized requirements and constraints** imposed by users or resources, which are entirely absent from standard benchmarks. Existing augmentation methods only add noise or lexical perturbations without altering the underlying semantics.

**Goal**: To construct a semantically enhanced constrained planning benchmark to systematically evaluate LLM capabilities in planning and formalization under constraints. **Key Insight**: Constraints are formalized into four categories (Initial, Goal, Action, State) using linguistic and pragmatic methods to ensure completeness. **Core Idea**: A simple natural language constraint can significantly degrade LLM performance, and this degradation is further exacerbated by increased problem complexity and lexical confusion.

## Method

### Overall Architecture

CoPE (Constrained Planning Environments) provides manually annotated natural language constraints and their ground-truth encodings in four formal languages for each problem across the BlocksWorld and CoinCollector domains. The evaluation process: given a domain description $D_d$, problem description $D_p$, PDDL header $\mathcal{DF}'$, and constraints $\mathcal{C}$, the LLM generates either a plan (Planner) or formal code (Formalizer), followed by verification via the VAL validator.

### Key Designs

**1. Formal Definition of Four Constraint Categories: Establishing a complete classification first**

To evaluate LLM performance under constraints, a mutually exclusive and collectively exhaustive classification is required. CoPE strictly categorizes constraints based on the part of the planning problem they affect: Initial (modifies initial state), Goal (modifies goal state), Action (restricts valid action sequences), and State (restricts valid state trajectories). Definitions are based on set relations between the primitive space and the modified space, proving that the State sub-category formally covers all possible constraints. This provides a clean coordinate system to analyze which constraints are hardest and which formal languages are most suitable.

**2. Multi-Formal Language Comparative Evaluation: Comparing four formal languages on the same constraints**

The success of the Formalizer approach depends on the choice of formal language. CoPE encodes each constraint into ground-truths for PDDL 1.2, PDDL3, LTL, and SMT (Z3). Evaluation includes three strategies: Generation (one-shot code generation), Editing (modifying a constraint-free version), and Revision (iterative correction based on solver errors for up to 3 rounds). This horizontal comparison identifies toolchain preferences—e.g., PDDL3 is designed for state constraints, while SMT excels at modeling state predicates as satisfiability problems.

**3. Robustness Expansion Experiments: Compounding constraints with complexity and lexical pollution**

To test if constraints amplify existing LLM vulnerabilities, CoPE includes two stress tests: BlocksWorld-XL increases the number of blocks to 50 to test entity space expansion, and MysteryBlocksWorld replaces all types, predicates, and action names with meaningless placeholders to sever reliance on training data familiarity. These experiments reveal that the complexity and lexical robustness of Formalizers vanish when constraints are introduced.

### Loss & Training

As an evaluation-focused work, no model training is involved. The core metric is **plan correctness**—whether the predicted plan successfully transitions from the initial state to the goal state within the ground-truth PDDL environment.

## Key Experimental Results

### Main Results

| Dataset | Method | Unconstrained | Constrained | Gain (Drop) |
|---------|--------|---------------|-------------|-------------|
| BlocksWorld | LLM-as-Planner (Gemini-3-Flash) | ~85% | ~55% | -30% |
| BlocksWorld | LLM-as-PDDL-Formalizer (Gemini) | ~70% | ~40% | -30% |
| CoinCollector | LLM-as-Planner (Gemini) | ~90% | ~60% | -30% |
| BlocksWorld | PDDL3 Formalizer | Lower than PDDL | Even Lower | Syntax/Compile Errors |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Generation | Baseline | Direct generation of constrained code |
| Editing | Partial Gain | Edit after generating unconstrained version |
| Revision | Further Gain | Iterative correction based on solver errors |
| BlocksWorld-XL (50 blocks) | Performance Plunge | Constraint impact worse with higher complexity |
| MysteryBlocksWorld | Formalizer Robustness Vanishes | Impact of constraints + lexical confusion |

### Key Findings
- Single-sentence constraints consistently halve performance across all LLMs, methods, and language combinations.
- LLM-as-Planner generally outperforms Formalizers in unconstrained settings, but Formalizers are more robust to problem complexity.
- Despite native syntax for constraints, PDDL3 performs worse than standard PDDL due to scarce training data.
- The inherent complexity and lexical robustness of Formalizers **completely disappear** upon introducing constraints.

## Highlights & Insights
- Rigorous formal definitions of constraint types with proven completeness provide a theoretical foundation for future work.
- Extensive experimental design covers 4 LLMs × 4 formal languages × 3 techniques × 4 constraint types × 4 datasets.
- Reveals a critical conclusion: **Simple semantic modifications challenge LLMs more effectively than lexical noise**, offering new directions for benchmark design.
- The philosophy of using semantic enhancement rather than data perturbation to combat data contamination is valuable for other NLP evaluation tasks.

## Limitations & Future Work
- Only single constraints are considered; conjunctions, negations, and ambiguities of constraints are not discussed.
- BlocksWorld and CoinCollector domains remain somewhat simple compared to real-world robotics or scheduling.
- The evaluation metric (plan correctness) might have false positives (plan correct by chance, but code is wrong), though verification suggests this is negligible.
- Future directions: Support complex constraint combinations, extend to more domains, and develop constraint-aware planning toolchains.
- Safety risks of autonomous agents in downstream tasks warrant attention; formal representations could provide transparency for human auditing and formal verification.

## Related Work & Insights
- **vs Standard IPC Benchmarks**: CoPE challenges LLMs via semantic modifications rather than just adding noise, better exposing true capabilities.
- **vs LLM+P (Liu et al., 2023)**: Both follow the Formalizer path, but LLM+P ignores constraints; CoPE reveals this limitation.
- **vs Mystery BlocksWorld**: CoPE demonstrates that constraints weaken Formalizer robustness more significantly than lexical confusion does.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic evaluation benchmark for constrained planning in LLMs with rigorous classification.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exhaustive analysis across models, languages, techniques, and domains.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions, logical structure, and rich visualizations.
- Value: ⭐⭐⭐⭐ Sounds an alarm for LLM planning research and points towards the importance of realistic constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICML 2026\] On the Generalization Gap in Self-Evolving Language Model Reasoning](../../ICML2026/llm_reasoning/on_the_generalization_gap_in_self-evolving_language_model_reasoning.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](../../AAAI2026/llm_reasoning/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)

</div>

<!-- RELATED:END -->
