---
title: >-
  [Paper Note] GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision
description: >-
  [CVPR 2026][LLM Efficiency][3D Vision Code Generation] The first PhD-level code generation benchmark for 3D geometric computer vision, GeoCodeBench…
tags:
  - "CVPR 2026"
  - "LLM Efficiency"
  - "3D Vision Code Generation"
  - "LLM Evaluation"
  - "Geometric Algorithm Implementation"
  - "PhD-level Benchmark"
  - "Unit Testing"
date: 2026-05-08
content_hash: 9efcc652796e42ec
---

# GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision

**Conference**: CVPR 2026  
**arXiv**: [2603.30038](https://arxiv.org/abs/2603.30038)  
**Code**: [https://geocodebench.github.io/](https://geocodebench.github.io/)  
**Area**: LLM Efficiency / Code Generation Evaluation  
**Keywords**: 3D Vision Code Generation, LLM Evaluation, Geometric Algorithm Implementation, PhD-level Benchmark, Unit Testing

## TL;DR
The first PhD-level code generation benchmark for 3D geometric computer vision, GeoCodeBench, comprising 100 function completion tasks curated from top-venue 2025 papers and codebases, with automated diverse unit tests. The strongest model GPT-5 achieves only 36.6% pass rate, revealing a significant gap in LLM scientific-level 3D code implementation.

## Background & Motivation

**Background**: AI-assisted programming has reshaped software practices and research workflows, but existing models still struggle with complex 3D geometric vision code. If models could reliably write such code, 3D vision research would be fundamentally transformed (automated prototyping, accelerated research cycles, democratized algorithm development).

**Limitations of Prior Work**: (1) Existing code benchmarks (HumanEval/MBPP/SWE-bench) do not cover 3D geometric implementations—they target general software engineering or competitive programming; (2) Scientific 3D vision code requires mathematically precise geometric operators, physical modeling, and multi-view reasoning—far beyond general-purpose capabilities; (3) Paper-to-code long-context scientific understanding remains an open problem.

**Key Challenge**: LLMs can generate general-purpose code but cannot reliably implement core functions in 3D geometric vision—how large is this gap? Where are the bottlenecks?

**Key Insight**: Simulating real research scenarios—giving models paper text plus function skeletons, requiring implementation completion, with automatic evaluation via unit tests.

**Core Idea**: (1) Extract core functions from official repositories of 2025 top-venue papers; (2) Automated tool nomination plus manual curation to ensure quality; (3) Diverse boundary tests covering geometric degenerate configurations; (4) Two-level capability taxonomy for evaluation.

## Method

### Overall Architecture
Paper PDF (OCR → structured JSON) + Code repository (automated candidate extraction → manual curation → function masking) + Unit tests (auto-generation → manual review) → LLM receives (paper + masked code + execution template) → fills implementation → sandbox execution + testing → PassRate scoring.

### Key Designs

1. **Benchmark Construction Pipeline**:

    - **Paper Processing**: MinerU OCR automatically extracts text/formulas/figures → organized into JSON by sections
    - **Code Processing**: Cursor automatically recommends candidate functions (10–20 per repository) → **manual review by 3D vision researchers** → retains 3–5 core geometric functions → function bodies replaced with `****EMPTY****` placeholders
    - **Unit Tests**: Cursor auto-generates 10 test cases (multiple parameter configurations) → manual review ensures reliability. Standardized execution templates (import/input-output definitions) are also provided
    - Design Motivation: Automated nomination is efficient but may select trivial/auxiliary functions → manual curation ensures every task is a "paper-core 3D geometric component"

2. **Two-Level Capability Taxonomy**:

    - **General 3D Capability** (foundational geometric knowledge):
        - Geometric Transformations (24%): coordinate conversions, projections, normals, rotation parameterizations
        - Mechanics/Optics Formulation (31%): spherical harmonics, BRDF, equations of motion, radiometric quantities
    - **Research Capability** (research-level reasoning):
        - Novel Algorithm Implementation (34%): function-level implementation of paper-core novel ideas
        - Geometric Logic Routing (11%): composing existing operators into new pipelines—many influential papers are structured this way
    - Design Motivation: Separating foundational and research capabilities to diagnose model weaknesses

3. **Evaluation Metric**:

    - PassRate = $\frac{1}{N}\sum_{i=1}^{N}\frac{p_i}{T_i}$, where $p_i$ is the number of passed tests and $T_i$ is the total number of tests
    - Context ablation: Method-only vs. full-text input

### Paper Sources
Covering 3DGS, pose estimation, SLAM, reconstruction, physics-based modeling, NeRF, 3D segmentation, and other subfields. All papers are from CVPR/ICCV/ICLR 2025, maximizing reduction of data leakage risk.

## Key Experimental Results

### Main Results (8 Representative Models)

| Model | Company | Overall | General | Research | Geo.Trans. | Algorithm |
|-------|---------|---------|---------|----------|------------|-----------|
| **GPT-5** | OpenAI | **36.6%** | **42.8%** | **29.1%** | 41.7% | 29.1% |
| Claude-Sonnet-4.5 | Anthropic | 31.1% | 37.2% | 23.7% | 38.3% | 19.7% |
| Gemini-2.5-Pro | Google | 30.4% | 33.8% | 26.2% | 41.9% | 25.3% |
| Kimi-K2-Instruct | Moonshot | 30.4% | 34.6% | 25.1% | 36.7% | 23.1% |
| Doubao-Seed-1.6 | ByteDance | 26.9% | 29.7% | 23.4% | 40.9% | 22.9% |
| Qwen3-Coder-480B | Alibaba | 23.5% | 22.7% | 24.6% | 29.0% | 21.8% |
| DeepSeek-R1 | DeepSeek | 21.0% | - | - | - | - |

### Ablation Study

| Input Context | PassRate | Note |
|---------------|----------|------|
| Full text | Baseline | Includes introduction, related work, etc. |
| **Truncated to Method** | **Statistically significantly better** | Irrelevant context interferes with reasoning |
| Abstract only | Significant drop | Insufficient technical details |

### Key Findings
- **Best model achieves only 36.6%**: GPT-5 is far from reliable on PhD-level 3D code
- **Research tasks are harder but positively correlated with General**: Geometric fundamentals are necessary but not sufficient for research-level implementation
- **Truncating to Method section actually performs better**: This indicates LLMs face severe difficulties in long-context scientific paper understanding—more text equals more interference rather than more useful information
- **Creative correctness**: In some successful cases, models used completely different but mathematically equivalent approaches to pass tests—demonstrating genuine problem-solving beyond copying
- **Geometric Logic Routing (11% of tasks)** reflects how many classic 3D vision papers are constructed—composing existing operators—requiring higher-level system design capabilities

## Highlights & Insights
- **First 3D vision code benchmark**: Fills the gap in AI coding evaluation for the scientific 3D domain. The community-driven, extensible design enables continuous growth with new papers
- **"More context is not better" finding**: Raises sharp questions about LLMs' long-context scientific understanding capabilities. Method truncation outperforms full text → LLMs may be misled by noise in introduction/related work sections
- **Paper-to-code research paradigm**: GeoCodeBench's evaluation setup directly simulates the real research workflow of "read paper → implement algorithm," representing a first step toward an "automated 3D vision scientist"
- **Unit test engineering contribution**: The diverse, boundary-case-covering automated tests for each function are themselves valuable pedagogical materials for 3D geometry

## Limitations & Future Work
- The scale of 100 functions remains limited—continuous expansion is needed
- Restriction to 2025 papers may require updates over time to avoid data leakage
- Unit test coverage may be incomplete—passing tests does not necessarily guarantee a fully correct implementation
- Only evaluates function-level completion—full paper reproduction (including training loops, data pipelines) is more challenging

## Related Work & Insights
- **vs HumanEval/MBPP**: General programming benchmarks without domain knowledge. GeoCodeBench requires deep 3D geometric reasoning
- **vs SWE-bench**: Repository-level issue resolution; GeoCodeBench is function-level paper-to-code
- **vs PaperBench**: Full paper reproduction evaluation; GeoCodeBench focuses on function-level core components—complementary
- **vs ResearchCodeBench**: Also masks key paper code, but does not focus on 3D geometry and tests are less diverse

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First 3D vision code benchmark with an insightful two-level capability taxonomy
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models, context ablation, category analysis, creative case studies
- Writing Quality: ⭐⭐⭐⭐⭐ Transparent and reproducible construction pipeline
- Value: ⭐⭐⭐⭐⭐ Long-term impact on 3D vision automation research and LLM scientific coding evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](../../ACL2026/llm_efficiency/humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)
- [\[AAAI 2026\] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](../../AAAI2026/llm_efficiency/resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)
- [\[AAAI 2026\] InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](../../AAAI2026/llm_efficiency/intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)
- [\[CVPR 2026\] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)

</div>

<!-- RELATED:END -->
