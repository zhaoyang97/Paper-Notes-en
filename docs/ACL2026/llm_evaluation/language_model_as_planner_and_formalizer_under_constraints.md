---
title: >-
  [Paper Note] Language Model as Planner and Formalizer under Constraints
description: >-
  [ACL 2026][LLM Evaluation][Constrained Planning] This paper introduces the CoPE benchmark, which injects formally categorized natural language constraints into classical planning environments, revealing that a single constraint sentence can halve the planning performance of state-of-the-art LLMs, exposing critical deficiencies in LLM planning robustness.
tags:
  - ACL 2026
  - LLM Evaluation
  - Constrained Planning
  - LLM-as-Planner
  - LLM-as-Formalizer
  - Benchmark
  - PDDL
date: 2026-05-08
content_hash: 9a5d4dc4a05759cc
---

# Language Model as Planner and Formalizer under Constraints

**Conference**: ACL 2026
**arXiv**: [2510.05486](https://arxiv.org/abs/2510.05486)
**Code**: [GitHub](https://github.com/CassieHuang22/LLM-as-Formalizer-constraints)
**Area**: LLM Evaluation
**Keywords**: Constrained Planning, LLM-as-Planner, LLM-as-Formalizer, Benchmark, PDDL

## TL;DR

This paper introduces the CoPE benchmark, which injects formally categorized natural language constraints into classical planning environments, revealing that a single constraint sentence can halve the planning performance of state-of-the-art LLMs, exposing critical deficiencies in LLM planning robustness.

## Background & Motivation

**State of the Field**: Two dominant paradigms exist for LLM-based planning — LLM-as-Planner directly generates action sequences end-to-end, while LLM-as-Formalizer translates natural language descriptions into formal languages such as PDDL before deriving solutions via a solver. Both approaches demonstrate notable capability on standard planning benchmarks.

**Limitations of Prior Work**: Existing benchmarks (e.g., BlocksWorld, Gripper) were largely developed decades ago, feature simple and homogeneous environment descriptions, and are highly likely to be covered by LLM training data. This simplicity may lead to significant **overestimation** of LLM planning capabilities, posing risks in downstream safety-critical applications.

**Root Cause**: Real-world planning instructions typically involve **personalized requirements and constraints** imposed by users or resources, yet standard benchmarks entirely lack such elements. Existing augmentation methods introduce only noise or lexical perturbations without altering the underlying semantics.

**Paper Goals**: To construct a semantically enriched constrained planning benchmark that systematically evaluates LLM planning and formalization capabilities under constraint conditions. **Starting Point**: Constraints are formalized into four categories (Initial, Goal, Action, State) using linguistic and pragmatic methods to ensure taxonomic completeness. **Core Idea**: A single constraint sentence can substantially degrade LLM performance, and this degradation is further amplified as problem complexity increases and lexical obfuscation is applied.

## Method

### Overall Architecture

CoPE (Constrained Planning Environments) manually annotates natural language constraints and their ground-truth encodings in four formal languages for each problem instance across two domains: BlocksWorld and CoinCollector. The evaluation pipeline proceeds as follows: given a domain description $D_d$, a problem description $D_p$, a PDDL header $\mathcal{DF}'$, and a constraint $\mathcal{C}$, the LLM generates either a plan (Planner) or formalized code (Formalizer), which is then validated by the VAL verifier for plan correctness.

### Key Designs

1. **Formal Definition of Four Constraint Categories**:

    - Function: Natural language constraints are strictly classified into four types — Initial (modifying the initial state), Goal (modifying the goal state), Action (restricting valid action sequences), and State (restricting valid state trajectories).
    - Mechanism: Categories are defined via set-theoretic relations between primitive and modified action/state spaces, with a proof that the State constraint subcategory subsumes all possible constraints, ensuring **completeness** of the taxonomy.
    - Design Motivation: Different formal languages (PDDL, PDDL3, LTL, SMT) vary in their expressive power across constraint categories; a formal taxonomy enables systematic analysis.

2. **Multi-Formalism Comparative Evaluation**:

    - Function: Constraints are encoded in PDDL 1.2, PDDL3, LTL, and SMT (Z3) respectively, evaluating the expressive and solving capabilities of each formalism.
    - Mechanism: Three technical strategies are compared — Generation (direct generation), Editing (generate unconstrained code then edit), and Revision (iterative correction of syntax errors, up to 3 rounds).
    - Design Motivation: Different constraint types are naturally suited to different formalisms — e.g., PDDL3 excels at state constraint syntax while SMT is well-suited for state predicate modeling — and systematic comparison provides guidance for future toolchain selection.

3. **Robustness Extension Experiments**:

    - Function: BlocksWorld-XL (50 blocks) and MysteryBlocksWorld (lexical obfuscation) are used to evaluate performance under increased complexity and potential data contamination.
    - Mechanism: The XL variant tests performance as the entity space grows; the Mystery variant replaces all type, predicate, and action names with meaningless placeholders.
    - Design Motivation: To verify whether constraints amplify existing LLM fragility under complex problems and lexical perturbation.

### Loss & Training

This paper is an evaluation study and does not involve model training. The primary evaluation metric is **plan correctness** — whether the predicted plan successfully transitions from the initial state to the goal state in the ground-truth PDDL environment.

## Key Experimental Results

### Main Results

| Dataset | Method | w/o Constraint | w/ Constraint | Drop |
|--------|------|--------|--------|----------|
| BlocksWorld | LLM-as-Planner (Gemini-3-Flash) | ~85% | ~55% | ~30% |
| BlocksWorld | LLM-as-PDDL-Formalizer (Gemini) | ~70% | ~40% | ~30% |
| CoinCollector | LLM-as-Planner (Gemini) | ~90% | ~60% | ~30% |
| BlocksWorld | PDDL3 Formalizer | Below PDDL | Even lower | More syntax/compilation errors |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Generation | Baseline | Direct generation of constraint code |
| Editing | Partial improvement | Generate unconstrained version then edit |
| Revision | Further improvement | Iterative correction based on solver errors |
| BlocksWorld-XL (50 blocks) | Sharp performance drop | Constraint impact more severe at higher complexity |
| MysteryBlocksWorld | Formalizer robustness vanishes | Dual impact of constraints and lexical obfuscation |

### Key Findings
- A single constraint sentence consistently halves performance across all LLM, method, and formalism combinations.
- LLM-as-Planner generally outperforms Formalizer in the unconstrained setting, but Formalizer is more robust to problem complexity.
- Despite dedicated constraint syntax support, PDDL3 underperforms standard PDDL due to scarcity of training data.
- Upon the introduction of constraints, the Formalizer's previously observed robustness to both problem complexity and lexical perturbation **completely disappears**.

## Highlights & Insights
- The formal definition of constraint categories is rigorous and proves completeness, providing a solid theoretical foundation for future work.
- The experimental design spans 4 LLMs × 4 formalisms × 3 techniques × 4 constraint types × 4 datasets, yielding a rich multidimensional analysis.
- A key finding is revealed: **simple semantic modifications challenge LLMs more effectively than lexical noise**, offering new directions for benchmark design.
- The design philosophy of CoPE — using semantic enrichment rather than data perturbation to combat data contamination — is worth adopting in other NLP evaluation tasks.

## Limitations & Future Work
- Only single constraints are considered; conjunctions, negations, and ambiguous constraints are not discussed, while real-world scenarios involve far more diverse constraint structures.
- The BlocksWorld and CoinCollector domains remain relatively simple and are far removed from real-world planning scenarios such as robotic manipulation or resource scheduling.
- The plan correctness metric may admit false positives — plans that happen to be correct without truly encoding the constraint — though validation shows this proportion is negligible.
- Future directions include support for more complex constraint combinations, extension to additional domains, and development of constraint-aware planning toolchains.
- Safety risks posed by autonomous agents in downstream tasks warrant attention; formal representations can provide transparency for human auditing and formal verification.

## Related Work & Insights
- **vs. Standard IPC Benchmarks**: CoPE challenges LLMs through semantic modification rather than noise injection alone, more effectively exposing true capabilities.
- **vs. LLM+P (Liu et al., 2023)**: A representative Formalizer approach that does not account for constraints; CoPE reveals its limitations.
- **vs. Mystery BlocksWorld**: CoPE demonstrates that constraints undermine Formalizer robustness more severely than lexical obfuscation.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic constrained planning benchmark for LLM evaluation, with rigorous formal categorization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models × formalisms × techniques × domains with highly detailed analysis.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear, structure is well-organized, and figures are abundant.
- Value: ⭐⭐⭐⭐ Sounds an important alarm for LLM planning research, pointing to a critical direction from simple benchmarks toward realistic constrained settings.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](../../NeurIPS2025/llm_evaluation/model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](../../ICLR2026/llm_evaluation/how_reliable_is_language_model_micro-benchmarking.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

<!-- RELATED:END -->
