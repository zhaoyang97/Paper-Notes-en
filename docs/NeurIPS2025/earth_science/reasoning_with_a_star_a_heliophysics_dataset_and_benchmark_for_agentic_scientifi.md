---
title: >-
  [Paper Note] Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning
description: >-
  [NeurIPS 2025][Earth Science][scientific reasoning] This paper introduces Reasoning With a Star (RWS), a 158-question scientific reasoning benchmark derived from NASA heliophysics summer school problem sets…
tags:
  - "NeurIPS 2025"
  - "Earth Science"
  - "scientific reasoning"
  - "multi-agent"
  - "heliophysics"
  - "systems engineering"
  - "benchmark"
date: 2026-05-08
content_hash: 68cfe251992ffc12
---

# Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2511.20694](https://arxiv.org/abs/2511.20694)  
**Code**: HuggingFace - SpaceML/ReasoningWithAStar  
**Area**: Earth Science
**Keywords**: scientific reasoning, multi-agent, heliophysics, systems engineering, benchmark

## TL;DR
This paper introduces Reasoning With a Star (RWS), a 158-question scientific reasoning benchmark derived from NASA heliophysics summer school problem sets, spanning three answer types (numeric/symbolic/textual). Paired with a unit-aware grader, it evaluates four multi-agent coordination paradigms (HMAW/PACE/PHASE/SCHEMA) and finds that no single paradigm dominates across all tasks — the systems-engineering-inspired SCHEMA achieves the strongest performance on tasks requiring rigorous constraint validation.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance on reasoning benchmarks (GSM8K, MATH, GPQA), yet remain deficient in domain-specific reasoning that requires physical assumptions, unit consistency, and proper scientific formatting. Heliophysics, as an interdisciplinary field with implications for climate, communications, and space safety, is conspicuously absent from LLM reasoning benchmarks.

**Limitations of Prior Work**: (1) Existing benchmarks do not test scientific reasoning — they do not require stating assumptions, maintaining unit consistency, or providing correctly formatted outputs; (2) Multi-agent systems lack systematic comparison on genuine scientific reasoning tasks; (3) Single-shot LLMs are prone to "reasoning hallucinations" and algebraic errors.

**Key Challenge**: Scientific reasoning is not a single logical leap — it demands domain expertise, iterative refinement, and assumption verification. Yet no clear design principles exist for multi-agent systems in this context.

**Goal**: (1) Construct a benchmark targeting real-world scientific reasoning; (2) Systematically compare multi-agent paradigms and their appropriate use cases.

**Key Insight**: Apply systems engineering principles ("complexity must be earned, not assumed") to agent design, and evaluate pure scientific reasoning capability without retrieval-augmented generation (RAG).

**Core Idea**: Different agent coordination paradigms suit different types of reasoning tasks — arithmetic reasoning benefits from lightweight self-critique, while scientific reasoning requires systems-engineering-style constraint tracking and verification.

## Method

### Overall Architecture
RWS benchmark (158 heliophysics problems; numeric/symbolic/textual answer types) → programmatic grader (unit-aware numeric tolerance, CAS symbolic equivalence, schema validation) → evaluation of single-shot baseline + 4 multi-agent paradigms → cross-benchmark comparison (+ GSM8K/MATH/GPQA/HumanEval/SWE-bench).

### Key Designs

1. **RWS Dataset Construction**:

    - Function: Constructs a benchmark from NASA/UCAR LWS Summer School problem sets.
    - Mechanism: OCR → manual cleaning → JSONL format. Each problem includes a question description, intermediate reasoning steps, final answer, answer type (numeric: 38 / symbolic: 52 / textual: 68), formatting hints, and metadata. Physical assumptions are preserved in the problem and reasoning step text.
    - Design Motivation: Heliophysics problems require stating assumptions (e.g., adiabatic expansion, neglecting certain loss terms), maintaining unit consistency, and providing correct scientific formatting — dimensions untested by existing benchmarks.

2. **Programmatic Grader**:

    - Function: Automatically verifies the correctness of scientific answers.
    - Mechanism: Numeric answers — 5% tolerance + unit checking; symbolic answers — algebraic equivalence via SymPy; textual answers — semantic equivalence judgment. When automatic grading fails, a two-agent pipeline (Parser + Judge) performs secondary verification.
    - Design Motivation: Scientific answer correctness cannot be assessed by simple string matching — algebraically equivalent expressions may differ in form, and units must be correct.

3. **Four Multi-Agent Paradigms**:

    - **HMAW** (Hierarchical Multi-Agent Workflow): CEO/manager/worker hierarchical handoff.
    - **PACE** (Plan-Answer-Critique-Enclose): Self-critique loop after answer generation.
    - **PHASE** (Plan-Hypothesize-Analyze-Solve-Evaluate): Hypothesis formulation prior to solving and verification.
    - **SCHEMA**: Systems-engineering-inspired design incorporating requirements tracking, assumption management, and interface checking.
    - Design Motivation: These paradigms span a spectrum from simple hierarchical delegation to complex engineering workflows, enabling systematic comparison of their applicable scenarios.

### Loss & Training
No training is performed. All paradigms use Gemini 2.5 Pro as the base model. A SWE-agent adapter is employed for code benchmarks.

## Key Experimental Results

### Main Results (Single-Shot Baseline)

| Model | RWS Accuracy |
|-------|-------------|
| Gemini 2.5 Pro | **35.44%** |
| OpenAI OSS 120B | 32.91% |
| Meta Llama 3.3 | 31.01% |
| Mistral 24.11 | 27.22% |

### Multi-Agent Paradigm Comparison

| Dataset | HMAW | PACE | PHASE | SCHEMA |
|---------|------|------|-------|--------|
| GSM8K | 91.1 | **93.4** | 92.4 | 86.4 |
| MATH | 78.3 | **81.5** | 77.8 | 71.4 |
| GPQA | **79.0** | 77.1 | 77.2 | 73.4 |
| RWS | 39.5 | 41.9 | 42.5 | **44.3** |
| HumanEval | 30.5 | 37.8 | 36.0 | **43.3** |
| SWE-bench Verified | 53.8 | 55.7 | 60.5 | **63.2** |

### Key Findings
- **No single paradigm is optimal across all tasks** — this is the most important conclusion.
- **PACE achieves the strongest performance on mathematical reasoning** (GSM8K 93.4%, MATH 81.5%): a lightweight self-critique loop suffices to correct computational errors.
- **HMAW is strongest on GPQA** (79.0%): simple hierarchical handoff is adequate for classification-style scientific QA.
- **SCHEMA leads on RWS/HumanEval/SWE-bench** (44.3%/43.3%/63.2%): tasks requiring format constraints, assumption tracking, and requirements verification benefit from the systems engineering approach.
- **All multi-agent paradigms outperform single-shot on RWS** (39–44% vs. 35%) — even simple coordination enhances scientific reasoning.
- **Absolute accuracy on RWS remains low** (<45%) — scientific reasoning continues to pose significant challenges for current LLMs.

## Highlights & Insights
- Applying the systems engineering principle **"complexity must be earned, not assumed"** to agent design is particularly insightful: adding more agents is not inherently beneficial; the key is matching the coordination strategy to task requirements.
- The **RWS grader design** offers substantial value: unit-aware numeric tolerance + CAS symbolic equivalence + semantic judgment + LLM secondary verification. This pipeline is transferable to other scientific benchmarks.
- The **cross-benchmark comparative methodology** is methodologically instructive: evaluating paradigms across diverse task types under unified conditions yields interpretable guidance for paradigm selection.

## Limitations & Future Work
- **Small dataset scale** (158 problems): may be insufficient to draw statistically robust conclusions.
- **Single-domain coverage**: generalization to other scientific fields requires further validation.
- **The no-RAG setting limits practical applicability**: real-world scientific reasoning requires access to reference materials and formula sheets.
- **Recommended directions**: extend RWS to other space science domains and incorporate RAG support.

## Related Work & Insights
- **vs. GSM8K/MATH**: These benchmarks test mathematical reasoning; RWS tests scientific reasoning (requiring assumption declaration and unit consistency) — they assess distinct dimensions.
- **vs. GPQA**: GPQA covers scientific QA but is primarily multiple-choice; RWS requires generating specific numerical values, formulas, and explanations.
- **vs. SWE-bench**: SWE-bench tests software engineering capability; RWS tests scientific reasoning — yet SCHEMA achieves top performance on both, suggesting that constraint validation is a generalizable capability.

## Rating
- Novelty: ⭐⭐⭐⭐ Heliophysics reasoning benchmark + systematic multi-agent paradigm comparison
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-benchmark comparison over 6 datasets, though RWS itself is small in scale
- Writing Quality: ⭐⭐⭐⭐ Clear structure with systems engineering principles woven consistently throughout
- Value: ⭐⭐⭐⭐ Provides interpretable guidance for agent paradigm selection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)
- [\[NeurIPS 2025\] Adaptive Online Emulation for Accelerating Complex Physical Simulations](adaptive_online_emulation_for_accelerating_complex_physical_simulations.md)
- [\[NeurIPS 2025\] Predicting Public Health Impacts of Electricity Usage](predicting_public_health_impacts_of_electricity_usage.md)
- [\[NeurIPS 2025\] A Probabilistic U-Net Approach to Downscaling Climate Simulations](a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)
- [\[AAAI 2026\] MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics](../../AAAI2026/earth_science/mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)

</div>

<!-- RELATED:END -->
