---
title: >-
  [Paper Note] CLARC: C/C++ Benchmark for Robust Code Search
description: >-
  [ICLR 2026][AIGC Detection][Code Retrieval] This paper introduces CLARC, the first compilable C/C++ code retrieval benchmark comprising 6…
tags:
  - "ICLR 2026"
  - "AIGC Detection"
  - "Code Retrieval"
  - "C/C++ Benchmark"
  - "Compilation Verification"
  - "Code Embedding"
  - "Assembly Language"
  - "Robustness"
date: 2026-05-08
content_hash: e775b641f91e4bd5
---

# CLARC: C/C++ Benchmark for Robust Code Search

**Conference**: ICLR 2026
**arXiv**: [2603.04484](https://arxiv.org/abs/2603.04484)  
**Code**: [GitHub](https://github.com/ClarcTeam/CLARC) / [HuggingFace](https://huggingface.co/datasets/ClarcTeam/CLARC)  
**Area**: AIGC Detection
**Keywords**: Code Retrieval, C/C++ Benchmark, Compilation Verification, Code Embedding, Assembly Language, Robustness

## TL;DR
This paper introduces CLARC, the first compilable C/C++ code retrieval benchmark comprising 6,717 query–code pairs. An automated pipeline extracts code from GitHub and employs LLMs combined with hypothesis testing to generate and validate queries. The benchmark covers four retrieval scenarios—standard, anonymized, assembly, and WebAssembly—and reveals that existing code embedding models over-rely on lexical features (NDCG@10 drops from 0.89 to 0.67 after anonymization) and perform poorly on binary-level retrieval.

## Background & Motivation
**Background**: Code retrieval benchmarks predominantly target Python and Java (e.g., CodeSearchNet, CoSQA, COIR), with embedding-based retrieval models (Voyage-code-3, OpenAI embeddings, etc.) serving as the standard for large-scale retrieval.

**Limitations of Prior Work**:
   - **Language Coverage Bias**: C/C++ is central to systems programming, yet mainstream benchmarks neglect or underemphasize text-to-code retrieval for C/C++.
   - **Non-Compilable Code**: Many code snippets in existing benchmarks lack necessary includes or dependencies and cannot be compiled, disconnecting them from real engineering practice.
   - **Unexposed Lexical Dependency**: Benchmarks rarely test robustness under identifier renaming or anonymization, meaning high scores may reflect variable-name matching rather than semantic understanding.
   - **Absence of Binary-Level Evaluation**: Security auditing and reverse engineering require searching code at the assembly or binary level, yet no benchmark assesses this capability.

**Key Challenge**: Code retrieval models claim to understand "code semantics," but if they rely on lexical features such as variable and function names, they will fail on obfuscated or assembly-level code.

**Core Idea**: Construct a full-stack code retrieval benchmark spanning source code to binary, and systematically probe semantic understanding versus lexical matching through anonymization and compilation to low-level representations.

## Method

### Dataset Construction Pipeline (Four Stages)

1. **Data Collection**
    - Functions are mined from 144 popular C/C++ GitHub repositories (45 for evaluation, 99 for training).
    - A whitelist of compilation environments (standard library headers) is established.
    - Each function is extracted along with its full dependency context (call graph and required definitions).
    - **Key Filter**: Only functions that compile successfully in the prepared environment are retained, ensuring code completeness.

2. **Dependency Complexity Classification**
    - Group 1 (526 pairs): Self-contained functions with no custom-type or auxiliary-function dependencies (avg. 12.8 LOC).
    - Group 2 (469 pairs): Functions depending on custom types but not auxiliary functions (13.3 LOC).
    - Group 3 (250 pairs): Functions depending on both custom types and auxiliary functions (71.5 LOC; longer and more complex).
    - **Design Motivation**: Retrieval difficulty varies with dependency complexity—code requiring contextual understanding is harder to retrieve.

3. **Robustness Settings (Four Scenarios)**
    - **Standard Source Code**: Original C/C++ code.
    - **Anonymized**: All variable, function, and type names replaced with meaningless identifiers (e.g., `var_0`, `func_1`), stripping lexical cues and retaining only semantics.
    - **Assembly**: Compiled to x86 assembly, simulating reverse engineering scenarios.
    - **WebAssembly**: Compiled to `.wat` format, simulating Web security auditing scenarios.
    - **Design Motivation**: Progressively stripping high-level features tests the model's genuine understanding at different levels of abstraction.

4. **Query Generation and Validation**
    - An LLM generates natural language queries (code summarization).
    - **Hypothesis-Testing Validation**: Rather than relying solely on LLM quality, human annotators score queries, and statistical hypothesis testing verifies that scores are significantly above a random baseline.
    - **Design Motivation**: Ensures that query–code correspondence is not an LLM hallucination.

### Evaluation Metrics
- **NDCG@10**: Ranking quality.
- **Hole@10**: Proportion of relevant documents absent from the top 10 (lower is better)—measures severe omissions.

## Key Experimental Results

### Main Results (6 Models × 4 Scenarios)

| Model | Standard NDCG@10 | Anonymized NDCG@10 | Drop |
|-------|-----------------|-------------------|------|
| Voyage-code-3 | **0.89** | 0.67 | −24.7% |
| OpenAI-embed-large | 0.85 | 0.60 | −29.4% |
| CodeT5+ | 0.72 | 0.55 | −23.6% |
| OASIS | 0.68 | 0.54 | −20.6% |
| Nomic-emb-code | 0.78 | 0.58 | −25.6% |

### Assembly / WebAssembly Retrieval

| Metric | Average Performance |
|--------|-------------------|
| Assembly Hole@10 (best model) | ~17.1% |
| WebAssembly Hole@10 | Higher (worse performance) |

### Analysis by Dependency Complexity

| Group | Standard NDCG@10 | Anonymized NDCG@10 |
|-------|-----------------|-------------------|
| Group 1 (self-contained) | Highest | Largest drop |
| Group 3 (complex dependencies) | Second | Smaller drop |

### Key Findings
- **All models exhibit consistent, substantial drops after anonymization** (20–30%), directly demonstrating reliance on lexical features such as variable and function names rather than genuine semantic understanding.
- **Assembly-level retrieval poses a significant challenge**: Even the best model misses 17.1% of relevant documents, rendering code retrieval largely impractical for security auditing and reverse engineering.
- Group 1 (self-contained functions) suffers the largest drop after anonymization, likely because such functions rely more heavily on descriptive naming.
- Commercial models (Voyage-code-3) lead on standard scenarios but lose their advantage after anonymization, suggesting that their "semantic understanding" partly stems from superior lexical matching.
- OASIS, specifically optimized for robustness, shows the smallest degradation but also achieves a lower absolute performance level.

## Highlights & Insights
- **Systematic Falsification of "Lexical Dependency"**: The clean experimental design of anonymization directly quantifies how much model performance derives from lexical features versus genuine semantic understanding—an important warning for the code retrieval community.
- **Full-Stack Coverage**: The progressive abstraction stripping from high-level source code → anonymized code → assembly → WebAssembly constitutes an elegant evaluation framework.
- **Value of Compilability Guarantees**: Ensuring code completeness with full context brings the benchmark closer to real engineering practice and enables dependency-complexity analysis.
- **Reusable Automated Pipeline**: The same pipeline can be applied to build analogous benchmarks for other languages and projects.

## Limitations & Future Work
- Coverage is limited to C/C++; analogous benchmarks for Python, Java, Rust, and other languages are absent.
- Assembly-level evaluation targets only the x86 architecture; ARM, RISC-V, and other architectures may exhibit different characteristics.
- Queries are LLM-generated and may not fully reflect the distribution of real developer search intent.
- The evaluation set of 1,245 pairs is relatively small compared to benchmarks such as CodeSearchNet.
- Multi-stage retrieval (coarse ranking followed by re-ranking) and RAG scenarios remain unexplored.

## Related Work & Insights
- **vs. CodeSearchNet**: Primarily covers Python, Go, Ruby, etc.; lacks C/C++ text-to-code retrieval and does not verify compilability.
- **vs. COIR**: A multi-task code retrieval benchmark that includes no anonymization or assembly-level robustness testing.
- **vs. XCodeEval**: Includes C++ but not sourced from real-world projects.
- **Implications for Code Embedding Models**: The substantial degradation after anonymization indicates a need for better semantic modeling approaches—such as program analysis, control-flow graphs, and data-flow analysis—rather than purely text-based embeddings.

## Rating
- Novelty: ⭐⭐⭐⭐ First compilable C/C++ code retrieval benchmark spanning source code to binary; the progressive anonymization–compilation design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six models × four scenarios × three dependency complexity levels, with hypothesis-testing validation.
- Writing Quality: ⭐⭐⭐⭐ Pipeline description is detailed and statistical validation is rigorous.
- Value: ⭐⭐⭐⭐⭐ The dataset contribution provides lasting value to the code retrieval and security communities and exposes critical model weaknesses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[AAAI 2026\] BAID: A Benchmark for Bias Assessment of AI Detectors](../../AAAI2026/aigc_detection/baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)
- [\[NeurIPS 2025\] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](../../NeurIPS2025/aigc_detection/synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)
- [\[ICLR 2026\] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)
- [\[ICLR 2026\] Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity](death_of_the_novelty_beyond_n-gram_novelty_as_a_metric_for_textual_creativity.md)

</div>

<!-- RELATED:END -->
