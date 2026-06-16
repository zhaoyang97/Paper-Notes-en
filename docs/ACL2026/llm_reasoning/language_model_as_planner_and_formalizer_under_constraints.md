---
title: >-
  [Paper Note] Language Model as Planner and Formalizer under Constraints
description: >-
  [ACL 2026][LLM Reasoning][LLM-as-Planner] This paper proposes the CoPE benchmark, which injects formally classified natural language constraints into classic planning environments. It reveals that a single constraint sentence can halve the planning performance of even the strongest LLMs, exposing a severe lack of robustness in LLM planning.
tags:
  - ACL 2026
  - LLM Reasoning
  - LLM-as-Planner
  - LLM-as-Formalizer
  - PDDL
date: 2026-05-08
content_hash: d528500ad0d594b9
---
# Language Model as Planner and Formalizer under Constraints

**Conference**: ACL 2026  
**arXiv**: [2510.05486](https://arxiv.org/abs/2510.05486)  
**Code**: [GitHub](https://github.com/CassieHuang22/LLM-as-Formalizer-constraints)  
**Area**: LLM Evaluation  
**Keywords**: Constrained Planning, LLM-as-Planner, LLM-as-Formalizer, Benchmarking, PDDL

## TL;DR

This paper proposes the CoPE benchmark, which injects formally classified natural language constraints into classic planning environments. It reveals that a single constraint sentence can halve the planning performance of even the strongest LLMs, exposing a severe lack of robustness in LLM planning.

## Background & Motivation

**Background**: There are two mainstream paradigms for LLMs in planning: LLM-as-Planner, which generates action sequences end-to-end, and LLM-as-Formalizer, which translates natural language descriptions into formal languages like PDDL for solver-based derivation. Both have shown respectable capabilities on standard planning benchmarks.

**Limitations of Prior Work**: However, existing benchmarks (e.g., BlocksWorld, Gripper) are decades old, feature simple and homogeneous environment descriptions, and are highly likely to be covered by LLM training data. This simplicity leads to a potential **overestimation** of LLM planning capabilities, posing risks in downstream safety-sensitive scenarios.

**Key Challenge**: Real-world planning instructions typically include **personalized requirements and constraints** imposed by users or resources, which are entirely absent from standard benchmarks. Existing enhancement methods only add noise or lexical perturbations without altering the underlying semantics.

**Goal**: Construct a semantically enhanced constrained planning benchmark to systematically evaluate the planning and formalization capabilities of LLMs under constraints. **Key Insight**: Formalize constraints into four categories (Initial, Goal, Action, State) using linguistic and pragmatic methods to ensure exhaustive classification. **Core Idea**: Simple one-sentence constraints significantly degrade LLM performance, and this degradation is further exacerbated by increased problem complexity and lexical confusion.

## Method

### Overall Architecture

CoPE (Constrained Planning Environments) provides manually annotated natural language constraints and their ground-truth encodings in four formal languages for each problem across the BlocksWorld and CoinCollector domains. Evaluation process: Given a domain description $D_d$, problem description $D_p$, PDDL header $\mathcal{DF}'$, and constraints $\mathcal{C}$, the LLM generates a plan (Planner) or formal code (Formalizer), which is finally verified for correctness using the VAL validator.

### Key Designs

**1. Formal Definition of Four Constraint Categories: Establishing a complete classification first**

To systematically evaluate LLM performance under constraints, the constraints themselves must follow a mutually exclusive and collectively exhaustive (MECE) classification. CoPE strictly categorizes natural language constraints into four types based on the component of the planning problem they affect: Initial (modifies initial state), Goal (modifies goal state), Action (restricts valid action sequences), and State (restricts valid state trajectories). This classification is defined based on set relations between the primitive action/state space and the modified space, proving that the State sub-class formally covers all possible constraints. This rigorous coordinate system allows for clean analysis of which constraints are most difficult or which formal language is best suited for specific types.

**2. Multi-Formal Language Comparative Evaluation: Comparing four formal languages on the same set of constraints**

The success of the Formalizer approach depends heavily on the choice of formal language for encoding constraints. CoPE encodes each constraint into ground-truth versions for PDDL 1.2, PDDL3, LTL, and SMT (Z3). Three technical routes are used to observe LLM generation behavior: Generation (direct one-shot generation of constrained code), Editing (generating an unconstrained version first, then adding constraints), and Revision (iterative correction up to 3 times based on solver errors). Since different constraints naturally favor different languages—e.g., PDDL3 is designed for state constraints, while SMT excels at modeling state predicates as satisfiability problems—this horizontal comparison identifies optimal toolchains for specific constraint types.

**3. Robustness Extension Experiments: Superimposing constraints with "Complexity / Lexical Pollution"**

Testing constraints on small-scale problems is insufficient; the real danger lies in whether constraints amplify existing LLM vulnerabilities. CoPE designs two stress-test versions: BlocksWorld-XL expands the block count to 50 to see if constraints become more lethal as the entity space grows; MysteryBlocksWorld replaces all types, predicates, and action names with meaningless placeholders to eliminate reliance on familiar vocabulary from training data. These experiments transform the "constraint effect" from a single difficulty slice into a pressure curve, revealing that the inherent complexity and lexical robustness of the Formalizer approach almost entirely vanish under constraints.

### Loss & Training

As an evaluation-centric work, no model training is involved. The core evaluation metric is **plan correctness**, which measures whether the predicted plan can successfully transition from the initial state to the goal state in the ground-truth PDDL environment.

## Key Experimental Results

### Main Results

| Dataset | Method | Unconstrained | Constrained | Gain (Decrease) |
|---------|--------|---------------|-------------|-----------------|
| BlocksWorld | LLM-as-Planner (Gemini-3-Flash) | ~85% | ~55% | -30% |
| BlocksWorld | LLM-as-PDDL-Formalizer (Gemini) | ~70% | ~40% | -30% |
| CoinCollector | LLM-as-Planner (Gemini) | ~90% | ~60% | -30% |
| BlocksWorld | PDDL3 Formalizer | Lower than PDDL | Even lower | High syntax errors |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Generation | Baseline | Direct generation of constrained code |
| Editing | Marginal Gain | Edit after unconstrained generation |
| Revision | Further Gain | Iterative correction based on solver errors |
| BlocksWorld-XL (50 blocks) | Performance Plunge | Constraint impact worsens with complexity |
| MysteryBlocksWorld | Robustness Vanishes | Double blow: constraints + lexical confusion |

### Key Findings
- One-sentence constraints consistently halve performance across all LLMs, methods, and language combinations.
- LLM-as-Planner generally outperforms Formalizer when unconstrained, but Formalizer is more robust to problem complexity.
- Despite having native syntax for constraints, PDDL3 performs worse than standard PDDL due to scarce training data.
- The inherent complexity and lexical robustness of Formalizers **vanish completely** once constraints are introduced.

## Highlights & Insights
- The formal definition of constraint types is rigorous and proven exhaustive, providing a theoretical foundation for future work.
- The experimental design is exceptionally thorough, covering 4 LLMs × 4 Formal Languages × 3 Techniques × 4 Constraint Types × 4 Datasets.
- It reveals an important conclusion: **Simple semantic modifications challenge LLMs more effectively than lexical noise**, offering new directions for benchmark design.
- The philosophy of CoPE—combating data contamination through semantic enhancement rather than data perturbation—is a valuable lesson for other NLP evaluation tasks.

## Limitations & Future Work
- Only single constraints are considered; conjunctions, negations, and ambiguities of constraints are not discussed.
- The BlocksWorld and CoinCollector domains are still relatively simple compared to real-world planning (e.g., robotics, resource scheduling).
- There is a possibility of false positives in plan correctness (plan is correct by chance but code is wrong), though verification suggests this is negligible.
- Future directions: Supporting complex constraint combinations, expanding to more domains, and developing constraint-aware planning toolchains.
- The safety risks of autonomous agents in downstream tasks warrant attention; formal representations could provide transparency for human auditing and formal verification.

## Related Work & Insights
- **vs. Standard IPC Benchmarks**: CoPE challenges LLMs through semantic modification rather than noise, better exposing true capabilities.
- **vs. LLM+P (Liu et al., 2023)**: Both follow the Formalizer route, but CoPE reveals limitations by incorporating constraints.
- **vs. Mystery BlocksWorld**: CoPE demonstrates that constraints are more effective than lexical confusion at undermining Formalizer robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic LLM evaluation benchmark for constrained planning with rigorous formalization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed analysis across multiple models, languages, techniques, and domains.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions, logical structure, and rich visualization.
- Value: ⭐⭐⭐⭐ Serves as a wake-up call for LLM planning research, highlighting the shift from simple benchmarks to realistic constraints.

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
