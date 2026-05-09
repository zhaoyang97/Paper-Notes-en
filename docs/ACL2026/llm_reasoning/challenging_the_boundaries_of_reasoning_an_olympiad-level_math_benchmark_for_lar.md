---
title: >-
  [Paper Note] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models
description: >-
  [ACL 2026][LLM Reasoning][mathematical reasoning benchmark] This paper introduces OlymMATH, the first olympiad-level mathematical benchmark that unifies natural language evaluation and formal theorem proving. It comprises 350 bilingual (Chinese–English) problems, spanning OlymMATH-EASY/HARD (200 problems with numerical answers) and OlymMATH-LEAN (150 problems formalized in Lean 4). Experiments reveal that the strongest model achieves only 58.4% accuracy on the HARD subset.
tags:
  - ACL 2026
  - LLM Reasoning
  - mathematical reasoning benchmark
  - olympiad mathematics
  - formal verification
  - Lean4
  - bilingual evaluation
date: 2026-05-08
content_hash: d4018d6238c4e89e
---

# Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models

**Conference**: ACL 2026
**arXiv**: [2503.21380](https://arxiv.org/abs/2503.21380)
**Code**: [GitHub](https://github.com/RUCAIBox/OlymMATH)
**Area**: LLM Mathematical Reasoning
**Keywords**: mathematical reasoning benchmark, olympiad mathematics, formal verification, Lean4, bilingual evaluation

## TL;DR

This paper introduces OlymMATH, the first olympiad-level mathematical benchmark that unifies natural language evaluation and formal theorem proving. It comprises 350 bilingual (Chinese–English) problems, spanning OlymMATH-EASY/HARD (200 problems with numerical answers) and OlymMATH-LEAN (150 problems formalized in Lean 4). Experiments reveal that the strongest model achieves only 58.4% accuracy on the HARD subset.

## Background & Motivation

**State of the Field**: The rapid advancement of reasoning models (DeepSeek-R1, o3-mini, Gemini 2.5 Pro, etc.) has led to saturation on existing mathematical benchmarks such as GSM8K and MATH, creating an urgent need for more challenging evaluation frameworks.

**Limitations of Prior Work**: (1) Olympiad-level benchmarks suffer from insufficient scale (e.g., AIME contains only 30 problems, where a single problem accounts for 3.33% accuracy); (2) Some benchmarks rely on LLM-as-judge to evaluate proof problems, introducing evaluation hallucination risks; (3) Data contamination is a serious concern — problems scraped from online sources such as AoPS may already appear in pretraining corpora; (4) Nearly all benchmarks support only English, lacking multilingual evaluation.

**Root Cause**: There is a need for a mathematical reasoning benchmark that simultaneously satisfies high difficulty, large scale, low contamination, bilingual support, and dual-paradigm evaluation (answer verification + process verification) — yet no existing benchmark satisfies all these criteria.

**Paper Goals**: To construct the first olympiad-level bilingual mathematical benchmark that unifies both natural language and formal proof paradigms.

**Starting Point**: Problems are manually collected from printed publications (rather than online sources) to minimize data contamination.

**Core Idea**: OlymMATH-EASY/HARD provides computation problems verified via SymPy rule-based checking (outcome evaluation), while OlymMATH-LEAN provides Lean 4 formalized problems verified by a theorem prover (process evaluation), with the two paradigms complementing each other.

## Method

### Overall Architecture

OlymMATH consists of three non-overlapping subsets: (1) OlymMATH-EASY (100 problems, relatively accessible olympiad level) and (2) OlymMATH-HARD (100 problems, difficult olympiad level), both consisting of computation problems with numerical answers supporting automated SymPy verification, each available in bilingual Chinese–English parallel versions; and (3) OlymMATH-LEAN (150 problems) formalized in Lean 4, accompanied by bilingual natural language statements and solutions. The benchmark covers four major areas: number theory, algebra, combinatorics, and geometry.

### Key Designs

1. **Unified Dual-Paradigm Evaluation**:

    - Function: Simultaneously assesses answer correctness and reasoning process quality.
    - Mechanism: Computation problems use SymPy rule-based answer verification (scalable and objective, but unable to assess reasoning quality); formal proof problems use the Lean 4 verifier to check proof processes (rigorous and auditable, but requiring formal specification expertise). The two paradigms provide complementary coverage.
    - Design Motivation: Answer-only verification cannot detect "heuristic guessing" behavior (where models directly guess answers without rigorous derivation); formal verification addresses this blind spot.

2. **Data Contamination Prevention**:

    - Function: Ensures evaluation results reflect genuine reasoning capabilities.
    - Mechanism: All problems are manually collected from printed publications (professional journals and textbooks), deliberately excluding online sources. N-gram contamination analysis shows that OlymMATH's contamination indicators are significantly lower than those of PolyMath.
    - Design Motivation: Existing benchmarks face serious contamination risks (Omni-MATH scrapes from AoPS; PolyMath directly uses AIME/CNMO problems).

3. **Bilingual Parallel Evaluation**:

    - Function: Reveals differences in model reasoning across languages.
    - Mechanism: Each problem has semantically equivalent English and Chinese versions. Experiments consistently reveal a performance gap between English and Chinese.
    - Design Motivation: Multilingual reasoning capability is a practical requirement for LLM deployment, yet systematic evaluation has previously been lacking.

## Key Experimental Results

### Main Results (OlymMATH-HARD EN)

| Model | Accuracy |
|-------|----------|
| Gemini 2.5 Pro | **58.4%** |
| o3-mini | 31.2% |
| DeepSeek-R1 | 19.5% |

### Key Findings
- The strongest model achieves only 58.4% accuracy on the HARD subset, demonstrating that olympiad-level mathematics remains highly challenging.
- All models consistently achieve higher accuracy on English than Chinese versions, revealing a gap in multilingual reasoning capability.
- Case analysis identifies "heuristic guessing" behavior in models — skipping rigorous derivation and directly probing for answers.
- N-gram contamination analysis confirms that OlymMATH has a lower data contamination risk compared to PolyMath.
- Over 582K reasoning trajectories are released to support community-level analysis.

## Highlights & Insights
- **Dual-paradigm unification is the key innovation**: This is the first benchmark to integrate outcome evaluation and process evaluation within a single framework.
- **The discovery of "heuristic guessing" is significant**: Models may arrive at correct answers through non-rigorous pathways, causing answer-only verification to overestimate capability.
- **The printed-publication sourcing strategy is effective**: It minimizes contamination risk while maintaining problem quality.
- **582K reasoning trajectories are a valuable resource**: They support community analysis of reasoning patterns across different models.

## Limitations & Future Work
- **Scale remains limited**: 350 problems is an improvement over AIME but is still not large.
- **High barrier to Lean 4 formalization**: This constrains the types and quantity of problems that can be included.
- **Coverage limited to four areas**: Mathematical branches such as probability and statistics are not covered.
- Future directions include expansion to additional mathematical branches, automated formalization pipelines, and construction of training data from reasoning trajectories.

## Related Work & Insights
- **vs. AIME**: Only 30 English-only problems, offering poor statistical reliability and insufficient difficulty ceiling.
- **vs. Omni-MATH**: Large scale but scraped from AoPS (contamination risk), and relies on LLM-as-judge for proof evaluation.
- **vs. miniF2F**: A formal benchmark but English-only, with problems drawn from well-known competitions, resulting in high contamination risk.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first olympiad-level benchmark combining dual paradigms and bilingual evaluation, with a clear design rationale.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model evaluation, contamination analysis, case studies, and 582K open-sourced reasoning trajectories.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough comparison with prior work and clearly motivated problem statement.
- Value: ⭐⭐⭐⭐⭐ Sets a new standard for mathematical reasoning evaluation; the dual-paradigm design has far-reaching implications.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](../../ICLR2026/llm_reasoning/nudging_the_boundaries_of_llm_reasoning.md)

<!-- RELATED:END -->
