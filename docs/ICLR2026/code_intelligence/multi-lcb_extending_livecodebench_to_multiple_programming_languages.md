---
title: >-
  [Paper Note] Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages
description: >-
  [ICLR 2026][Code Intelligence][LiveCodeBench] Multi-LCB extends the Python-only LiveCodeBench to 12 programming languages via a transformation pipeline that converts functional LeetCode tasks into a unified STDIN/STDOUT format. It enables cross-language comparisons on identical problems without compromising contamination control, revealing prevalent "Python overfitting" and language-specific data contamination in current LLMs.
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "LiveCodeBench"
  - "Multi-lingual Code Generation"
  - "Contamination-aware Evaluation"
  - "Pass@1"
  - "STDIN/STDOUT"
  - "Python Overfitting"
date: 2026-05-08
content_hash: 7431c8962af48caf
---

# Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages

**Conference**: ICLR 2026  
**Code**: [https://github.com/Multi-LCB/Multi-LCB](https://github.com/Multi-LCB/Multi-LCB)  
**Area**: Code Intelligence / Code Generation Benchmarks  
**Keywords**: LiveCodeBench, Multi-lingual Code Generation, Contamination-aware Evaluation, Pass@1, STDIN/STDOUT, Python Overfitting  

## TL;DR
Multi-LCB extends the Python-only LiveCodeBench to 12 programming languages via a transformation pipeline that converts functional LeetCode tasks into a unified STDIN/STDOUT format. It enables cross-language comparisons on identical problems without compromising contamination control, revealing prevalent "Python overfitting" and language-specific data contamination in current LLMs.

## Background & Motivation
**Background**: LiveCodeBench (LCB) has become a de facto standard for evaluating LLM code generation capabilities, utilized by entities like DeepMind and DeepSeek. It features continuous scraping of contest problems, filtering by release date, and support for contamination-aware evaluation.

**Limitations of Prior Work**: LCB only evaluates Python. However, in real-world software engineering, developers switch between multiple languages such as C++, Java, JavaScript, and Rust, each with its own syntax, semantics, and idioms. A model proficient only in Python may not perform equally well in C++ for system programming or Java for enterprise software.

**Key Challenge**: Existing multi-lingual benchmarks are either static snapshots that are already saturated and contaminated (MBXP, MultiPL-E, and HumanEval-XL require rewriting function signatures and unit tests for each language, which is labor-intensive and sensitive to syntax/runtime differences), utilize different task sets for different languages (McEval, BigCodeBench) preventing direct cross-language comparison, or lack continuous updates (xCodeEval). **No benchmark simultaneously achieves: identical tasks across languages + controllable contamination + continuous automated updates.**

**Goal**: To losslessly replicate every LCB task across 12 languages while fully inheriting LCB's contamination control and evaluation protocols, allowing for a comparison of cross-lingual capabilities on identical problems as LCB evolves.

**Key Insight**: **Unified I/O protocols instead of test translation**. By retaining only the natural language description and converting all hidden tests into a language-agnostic STDIN/STDOUT format, a single evaluation harness can drive any target language. This bypasses the unsustainable engineering effort of rewriting unit tests for every language.

## Method

### Overall Architecture
Multi-LCB is a transformation pipeline that repackages the LCB dataset into a multi-lingual benchmark. It loads a version of the LCB code generation dataset from HuggingFace (retaining metadata from LeetCode, AtCoder, and Codeforces), converts functional LeetCode tasks into STDIN/STDOUT, wraps each problem into a prompt for the target language, and finally compiles/executes the generated code in an isolated sandbox against official hidden tests to calculate Pass@1.

```mermaid
flowchart LR
    A[LCB Dataset<br/>HuggingFace] --> B{Task Format}
    B -->|STDIN/STDOUT<br/>AtCoder/Codeforces| D[Unified Evaluation Pool]
    B -->|Functional<br/>LeetCode| C[Test Converter<br/>Prompt Rewriting + Test Conversion]
    C --> D
    D --> E[Target Language Prompt<br/>12 Languages]
    E --> F[LLM Generated Code]
    F --> G[Sandbox Compile/Execute<br/>Official Hidden Tests]
    G --> H[Pass@1]
```

### Key Designs

**1. Unified STDIN/STDOUT Conversion for Functional Tasks**: LCB uses two native formats: STDIN/STDOUT for AtCoder/Codeforces and Functional for LeetCode. Extending the latter is a challenge because LeetCode starter code is tightly coupled with its test harness. The authors designed an automated pipeline consisting of: (1) **Prompt rewriting**, which rearranges example cases into STDIN/STDOUT format; and (2) **Test conversion**, which transforms hidden tests into a unified form categorized by I/O structure (scalars, 1D arrays, and 2D arrays). Since contest problems are inherently language-agnostic, this conversion introduces no inconsistencies.

**2. Zero-shot Prompt Protocol and Pass@1 Scoring**: Code generation follows LCB’s original zero-shot strategy. The prompt includes a system message (e.g., "You are an expert Python programmer..."), a natural language problem description, explicit STDIN/STDOUT specifications, and input/output examples. A problem is marked correct only if it compiles and passes all hidden tests within 6 seconds and 4GB of memory. The primary metric is Pass@1.

**3. Automatic Tracking and Language Selection**: Multi-LCB automatically tracks future LCB updates. It covers 12 languages: C++, C#, Python, Java, Rust, Go, TypeScript, JavaScript, Ruby, PHP, Kotlin, and Scala. These were selected based on popularity (GitHub/StackOverflow rankings), infrastructure stability, and paradigm diversity. Execution occurs in isolated containers with specific toolchains (e.g., GCC 13, Rust 1.79, OpenJDK 21).

## Key Experimental Results
The authors evaluated 24 public LLMs (7B–685B) using Dataset v6 (tasks released after 2025-02-01).

### Main Results (Pass@1 % · Selected Models · *=Reasoning Model)

| Model | Python | C++ | Java | Go | Rust | Scala | 12-Lang Avg |
|------|--------|-----|------|-----|------|-------|-------------|
| GPT-OSS-120B* (Medium) | 71.1 | **72.3** | 70.4 | **69.9** | 70.5 | 54.1 | **67.8** |
| Qwen3-235B-A22B-Thk-2507* | **74.0** | 75.8 | **73.9** | 56.7 | 47.7 | 57.6 | 64.0 |
| DeepSeek-R1-0528* | 66.3 | 68.0 | 67.8 | 55.0 | 63.1 | 62.3 | 63.1 |
| Qwen3-30B-A3B-Thk-2507* | 64.0 | 65.7 | 62.4 | 44.1 | 51.7 | 43.6 | 53.2 |
| OpenRsn-Nmt-32B* | 64.4 | 44.2 | 40.8 | 11.5 | 2.8 | 6.0 | 22.7 |

### Fidelity (Multi-LCB Python vs. Official LCB Leaderboard)

| Model | ORIG | OUR | Δ |
|------|------|-----|-----|
| Qwen3-235B-A22B-Thk-2507 | 74.1 | 74.0 | −0.1 |
| DeepSeek-R1-0528 | 68.7 | 66.3 | −2.4 |
| **Mean Absolute Deviation** | — | — | **≈3%** |

### Key Findings
- **Python is not a reliable proxy for other languages**: Models like GPT-OSS-120B* significantly outperform others in Go/Rust/Ruby despite lower Python scores compared to Qwen3 models.
- **Widespread Python Overfitting**: Scatter plots of Python vs. average multi-lingual performance show most models above the $y=x$ line. Models without explicit multi-lingual training show gaps exceeding 60%.
- **Clear Difficulty Gradients**: Python has the highest average Pass@1 (0.482), followed by Java/C++ (≈0.44), while Scala remains the most difficult (<0.29).
- **Language-Specific Contamination**: Scores for problems released before the model cutoff are systematically higher, dropping sharply for post-cutoff problems.

## Highlights & Insights
- **Engineering Leverage**: Replacing "per-language translation" with a "unified protocol" reduces the complexity of adapting $N$ languages $\times$ $M$ problems to a one-time I/O conversion.
- **Quantifying Overfitting**: Identical tasks across languages allow differences in scores to be attributed solely to language proficiency rather than task difficulty.
- **Granular Analysis**: Conducting contamination analysis at a monthly and per-language granularity reveals that contamination follows the distribution of pre-training corpora.

## Limitations & Future Work
- **Language Coverage**: The benchmark does not yet include languages like Swift, Haskell, or R.
- **Task Domain**: Problems are rooted in competitive programming, which relates only indirectly to industrial software engineering.
- **Platform Assumptions**: The viability of STDIN/STDOUT conversion depends on the language-agnostic nature of contest problems.

## Related Work & Insights
- **Comparison**: Traditional benchmarks like HumanEval or MBPP are static and Python-centric. Multi-LCB solves the "identical task" and "automated tracking" problems simultaneously via STDIN/STDOUT unification.
- **Insight**: When scaling an evaluation dimension (like language), "unifying the interface protocol" is more sustainable than "customized adaptation for every instance."

## Rating
- **Novelty**: ⭐⭐⭐ (Extends LCB; the combination of STDIN/STDOUT unification and automatic tracking is highly effective.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Evaluates 24 models over 12 languages with 10 iterations each.)
- **Writing Quality**: ⭐⭐⭐⭐ (Logical flow from design to findings; clear visualizations.)
- **Value**: ⭐⭐⭐⭐ (Exposes "Python overfitting" and provides a self-updating target for multi-lingual code model evaluation.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Gradient-Based Program Synthesis with Neurally Interpreted Languages](gradient-based_program_synthesis_with_neurally_interpreted_languages.md)
- [\[AAAI 2026\] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](../../AAAI2026/code_intelligence/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)
- [\[ICLR 2026\] The Matthew Effect of AI Programming Assistants: A Hidden Bias in Software Evolution](the_matthew_effect_of_ai_programming_assistants_a_hidden_bias_in_software_evolut.md)
- [\[ICLR 2026\] VisCoder2: Building Multi-Language Visualization Coding Agents](viscoder2_building_multi-language_visualization_coding_agents.md)
- [\[ICLR 2026\] CARD: Towards Conditional Design of Multi-agent Topological Structures](card_towards_conditional_design_of_multi-agent_topological_structures.md)

</div>

<!-- RELATED:END -->
