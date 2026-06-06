---
title: >-
  [Paper Note] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Mathematical Reasoning Benchmark] OlymMATH is proposed as the first olympiad-level math benchmark to unify natural language evaluation and formal theorem proving. It comprises 350 bilingual (Ch…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Mathematical Reasoning Benchmark"
  - "Olympiad Mathematics"
  - "Formal Verification"
  - "Lean4"
  - "Bilingual Evaluation"
date: 2026-05-08
content_hash: 2ae5b020f7561c36
---

# Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2503.21380](https://arxiv.org/abs/2503.21380)  
**Code**: [GitHub](https://github.com/RUCAIBox/OlymMATH)  
**Area**: LLM Mathematical Reasoning  
**Keywords**: Mathematical Reasoning Benchmark, Olympiad Mathematics, Formal Verification, Lean4, Bilingual Evaluation

## TL;DR

OlymMATH is proposed as the first olympiad-level math benchmark to unify natural language evaluation and formal theorem proving. It comprises 350 bilingual (Chinese-English) problems across OlymMATH-EASY/HARD (200 numerical answer problems) and OlymMATH-LEAN (150 Lean 4 formalized problems), revealing that even the strongest models achieve only 58.4% accuracy on the HARD subset.

## Background & Motivation

**Background**: Rapid progress in reasoning models (e.g., DeepSeek-R1, o3-mini, Gemini 2.5 Pro) has led to the saturation of existing mathematical benchmarks like GSM8K and MATH, necessitating more challenging evaluation frameworks.

**Limitations of Prior Work**: (1) Olympiad-level benchmarks lack scale (e.g., AIME has only 30 problems, where a single problem variance affects accuracy by 3.33%); (2) Some benchmarks rely on LLM-as-judge for proof evaluation, risking evaluation hallucinations; (3) Severe data leakage—problems scraped from online sources like AoPS may already exist in pre-training data; (4) Almost all benchmarks are English-only, lacking multilingual assessment.

**Key Challenge**: There is a need for an olympiad-level benchmark that simultaneously satisfies high difficulty, large scale, low leakage, bilingual support, and dual-paradigm (answer verification + process verification)—criteria that no existing benchmark meets.

**Goal**: To construct the first olympiad-level bilingual math benchmark that unifies natural language and formal proof paradigms.

**Key Insight**: Manually collect problems from printed publications (non-web sources) to minimize data leakage.

**Core Idea**: OlymMATH-EASY/HARD provides computational problems verified via sympy rules (result evaluation), while OlymMATH-LEAN provides Lean 4 formalized problems verified by a theorem prover (process evaluation), rendering the two complementary.

## Method

### Overall Architecture

OlymMATH includes three non-overlapping subsets: (1) OlymMATH-EASY (100 problems, easier olympiad level) and (2) OlymMATH-HARD (100 problems, difficult olympiad level), both consisting of computational problems with numerical answers supporting sympy auto-verification and parallel Chinese-English versions; (3) OlymMATH-LEAN (150 problems), formalized in Lean 4 with bilingual natural language statements and solutions. The benchmark covers four major domains: Number Theory, Algebra, Combinatorics, and Geometry.

### Key Designs

1.  **Dual-Paradigm Evaluation Unification**:
    - **Function**: Simultaneously evaluates result correctness and reasoning process quality.
    - **Mechanism**: Computational problems use sympy rules to verify answers (scalable and objective, but lacks process assessment); formal proof problems use the Lean 4 verifier to check the proof process (rigorous and auditable, but requires specialized formalization skills). These complement each other.
    - **Design Motivation**: Answer verification alone cannot detect "heuristic guessing" (where models bypass rigorous derivation to guess answers). Formal verification addresses this blind spot.

2.  **Data Leakage Protection**:
    - **Function**: Ensures evaluation results reflect genuine reasoning capabilities.
    - **Mechanism**: All problems are manually collected from printed publications (professional journals and textbooks), intentionally excluding web sources. N-gram leakage analysis shows that OlymMATH leakage metrics are significantly lower than those of PolyMath.
    - **Design Motivation**: Existing benchmarks (e.g., Omni-MATH scraped from AoPS, PolyMath using AIME/CNMO) face high leakage risks.

3.  **Bilingual Parallel Evaluation**:
    - **Function**: Reveals reasoning disparities across different languages.
    - **Mechanism**: Every problem has English and Chinese versions with semantic equivalence. Experiments identified consistent performance gaps between English and Chinese outputs.
    - **Design Motivation**: Multilingual reasoning is a real-world requirement for LLM deployment, yet systematic evaluation has been historically lacking.

## Key Experimental Results

### Main Results (OlymMATH-HARD EN)

| Model | Accuracy |
|------|--------|
| Gemini 2.5 Pro | **58.4%** |
| o3-mini | 31.2% |
| DeepSeek-R1 | 19.5% |

### Key Findings
- The strongest model achieves only 58.4% accuracy on the HARD subset, indicating that olympiad-level math remains extremely challenging.
- Accuracy for the English versions of all models is consistently higher than the Chinese versions, revealing a gap in multilingual reasoning capabilities.
- Case studies identified "heuristic guessing" behavior—models skip rigorous derivation to attempt answers directly.
- N-gram leakage analysis confirms that OlymMATH has lower data leakage risk compared to PolyMath.
- The open-sourcing of 582K+ reasoning trajectories supports in-depth community analysis of reasoning patterns.

## Highlights & Insights
- **Dual-paradigm unification is a key innovation**: It is the first to fuse result evaluation and process evaluation within a single benchmark.
- **Discovery of "heuristic guessing" is significant**: Models may obtain correct answers through non-rigorous paths; answer-only verification tends to overestimate capability.
- **Printed publication sourcing strategy is effective**: It minimizes leakage risk while guaranteeing high problem quality.
- **The 582K reasoning trajectories are a valuable resource**: They enable the community to analyze the reasoning modes of different models.

## Limitations & Future Work
- **Scale remains limited**: While 350 problems outperform AIME, the size is still modest.
- **High barrier for Lean 4 formalization**: This limits the expansion of problem types and quantities.
- **Domain coverage**: Mathematical branches such as probability and statistics are not yet covered.
- **Future directions**: Expansion to more mathematical branches, automated formalization pipelines, and the construction of training data based on reasoning trajectories.

## Related Work & Insights
- **vs AIME**: AIME contains only 30 problems in English, offering poor statistical reliability and an insufficient difficulty ceiling.
- **vs Omni-MATH**: While large in scale, Omni-MATH's scraping from AoPS presents leakage risks and relies on LLM-as-judge for proof evaluation.
- **vs miniF2F**: A formal benchmark that is English-only and uses well-known competition problems with high leakage risks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First dual-paradigm and bilingual olympiad-level math benchmark with a clear design philosophy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model evaluation, leakage analysis, case studies, and open-sourced trajectories.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Thorough comparison with prior work and clear problem motivation.
- **Value**: ⭐⭐⭐⭐⭐ Sets a new standard for mathematical reasoning evaluation with a far-reaching dual-paradigm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models](revisiting_a_pain_in_the_neck_a_semantic_reasoning_benchmark_for_language_models.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)
- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)

</div>

<!-- RELATED:END -->
