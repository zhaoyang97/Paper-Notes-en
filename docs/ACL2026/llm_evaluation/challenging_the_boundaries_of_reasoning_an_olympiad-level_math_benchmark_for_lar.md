---
title: >-
  [Paper Note] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Lean4] OlymMATH is proposed as the first Olympiad-level mathematical benchmark that unifies natural language evaluation and formal theorem proving. It contains 350 bilingual (Chinese and English) problems, covering OlymMATH-EASY/HARD (200 problems with numerical answers) and OlymMATH-LEAN (150 Lean 4 formalized problems), rev
tags:
  - ACL 2026
  - LLM Evaluation
  - Lean4
date: 2026-05-08
content_hash: f337fd047470a359
---
# Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2503.21380](https://arxiv.org/abs/2503.21380)  
**Code**: [GitHub](https://github.com/RUCAIBox/OlymMATH)  
**Area**: LLM Mathematical Reasoning  
**Keywords**: Mathematical reasoning benchmark, Olympiad math, Formal verification, Lean4, Bilingual evaluation

## TL;DR

OlymMATH is proposed as the first Olympiad-level mathematical benchmark that unifies natural language evaluation and formal theorem proving. It contains 350 bilingual (Chinese and English) problems, covering OlymMATH-EASY/HARD (200 problems with numerical answers) and OlymMATH-LEAN (150 Lean 4 formalized problems), revealing that the strongest models achieve only 58.4% accuracy on the HARD subset.

## Background & Motivation

**Background**: The rapid advancement of reasoning models (e.g., DeepSeek-R1, o3-mini, Gemini 2.5 Pro) has led to the saturation of existing mathematical benchmarks like GSM8K and MATH, necessitating more challenging evaluation frameworks.

**Limitations of Prior Work**: (1) Olympiad-level benchmarks lack sufficient scale (e.g., AIME consists of only 30 problems, where a single question difference affects accuracy by 3.33%); (2) some benchmarks rely on LLM-as-judge for proof problems, risking evaluation hallucinations; (3) data leakage issues are severe, as problems scraped from online sources like AoPS may already exist in pre-training data; (4) nearly all benchmarks support only English, lacking multilingual evaluation.

**Key Challenge**: There is a need for a mathematical reasoning benchmark that simultaneously satisfies high difficulty, large scale, low leakage, bilingual support, and dual-paradigm (answer verification + process verification)—yet no existing benchmark meets all these criteria.

**Goal**: To construct the first Olympiad-level bilingual mathematical benchmark that unifies natural language and formal proof paradigms.

**Key Insight**: Problems are manually collected from printed publications (non-web sources) to minimize data leakage.

**Core Idea**: OlymMATH-EASY/HARD provides computational problems verified with sympy rules (result evaluation), while OlymMATH-LEAN provides Lean 4 formalized problems verified by a theorem prover (process evaluation), complementing each other.

## Method

### Overall Architecture

OlymMATH addresses the core issue that existing mathematical benchmarks are either saturated, too small, or verify only answers without validating the reasoning process. It divides an Olympiad-level problem set into three non-overlapping subsets and evaluates them collaboratively across two dimensions. On one side, computational problems with numerical answers are used for objective result verification; on the other, Lean 4 formalized problems provide rigorous process verification. Given an Olympiad problem, the system outputs a dual signal identifying both "answer correctness" and "reasoning rigor." Among the three subsets, OlymMATH-EASY (100 easier problems) and OlymMATH-HARD (100 difficult problems) consist of numerical problems automatically verifiable by sympy with parallel Chinese and English versions. OlymMATH-LEAN (150 problems) is formalized in Lean 4 and includes bilingual natural language statements and solutions, covering number theory, algebra, combinatorics, and geometry.

### Key Designs

**1. Unified Dual-Paradigm Evaluation: Result and Process Verification Complementarity**

Relying solely on answer correctness has a hidden blind spot: models might bypass rigorous derivation and use "heuristic guessing" to find the correct value. Result-only evaluation misinterprets such behavior as true reasoning. OlymMATH uses two paradigms to address this: computational problems use sympy to compare final answers (objective and scalable but result-focused), while formal proof problems use the Lean 4 verifier to step-check the proof process (rigorous and auditable but requiring formalization capabilities). The former ensures coverage breadth, while the latter blocks "answer guessing" loopholes. Together, they assess both "answer correctness" and "procedural rigor."

**2. Data Leakage Prevention: Problems Sourced from Printed Publications**

Leakage risks in existing benchmarks are significant: Omni-MATH scrapes from AoPS and PolyMath uses AIME/CNMO, both likely present in pre-training corpora, leading to inflated scores reflecting memory rather than reasoning. OlymMATH avoids web sources by manually collecting all problems from printed publications such as professional journals and textbooks. N-gram leakage analysis confirms this strategy—OlymMATH shows significantly lower leakage indicators than PolyMath, more accurately reflecting true reasoning levels.

**3. Bilingual Parallel Evaluation: Semantically Equivalent Chinese and English Versions**

Multilingual reasoning is a practical requirement for LLM deployment, yet previous benchmarks are almost exclusively English-only, failing to reveal if models retain capability when the language changes. OlymMATH prepares semantically equivalent English and Chinese versions for every problem, treating language as a controlled variable. Experiments show stable and consistent performance gaps between the two versions, a phenomenon that unilingual benchmarks cannot expose.

## Key Experimental Results

### Main Results (OlymMATH-HARD EN)

| Model | Accuracy |
|------|--------|
| Gemini 2.5 Pro | **58.4%** |
| o3-mini | 31.2% |
| DeepSeek-R1 | 19.5% |

### Key Findings
- The strongest model achieves only 58.4% accuracy on the HARD subset, indicating Olympiad-level math remains extremely challenging.
- All models consistently show higher accuracy in English than in Chinese, revealing a gap in multilingual reasoning capabilities.
- Case studies reveal "heuristic guessing" behaviors where models skip rigorous derivation to probe for answers directly.
- N-gram leakage analysis confirms OlymMATH has a lower risk of data leakage compared to PolyMath.
- The release of 582K+ reasoning trajectories supports deep community analysis of reasoning patterns.

## Highlights & Insights
- **Dual-paradigm unification is a key innovation**: This is the first benchmark to integrate result evaluation and process evaluation within a single framework.
- **Discovery of "heuristic guessing" is significant**: Models may reach correct answers through non-rigorous paths; answer-only verification overestimates capability.
- **Printed publication sourcing strategy is effective**: This strategy minimizes leakage risks while ensuring the high quality of questions.
- **582K reasoning trajectories are a valuable resource**: These trajectories enable the community to analyze the reasoning modes of different models.

## Limitations & Future Work
- **Scale remains limited**: While 350 problems are better than AIME, the scale is still relatively small.
- **High barrier for Lean 4 formalization**: The difficulty of formalization limits the expansion of problem types and quantity.
- **Limited domain coverage**: Branches such as probability and statistics are not yet covered.
- Future directions: Expanding to more mathematical branches, developing automated formalization pipelines, and constructing training data based on reasoning trajectories.

## Related Work & Insights
- **vs AIME**: AIME has only 30 problems and is English-only, resulting in poor statistical reliability and an insufficient difficulty ceiling.
- **vs Omni-MATH**: Omni-MATH is larger but scraped from AoPS with leakage risks, and it relies on LLM-as-judge for proof evaluation.
- **vs miniF2F**: A formalization benchmark that is English-only and uses well-known competition problems, which carry high leakage risks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First dual-paradigm + bilingual Olympiad-level math benchmark with a clear design philosophy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model evaluation, leakage analysis, case studies, and 582K reasoning trajectories open-sourced.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough comparison with prior work and clear problem motivation.
- Value: ⭐⭐⭐⭐⭐ Sets a new standard for mathematical reasoning evaluation; dual-paradigm design has far-reaching influence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](../../ICML2025/llm_evaluation/leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)
- [\[ACL 2026\] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models](revisiting_a_pain_in_the_neck_a_semantic_reasoning_benchmark_for_language_models.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)

</div>

<!-- RELATED:END -->
