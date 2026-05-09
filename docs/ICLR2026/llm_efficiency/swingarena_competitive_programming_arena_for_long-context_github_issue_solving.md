---
title: >-
  [Paper Note] SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving
description: >-
  [ICLR 2026][LLM Efficiency][Adversarial Evaluation] This paper proposes SwingArena, an adversarial evaluation framework in which two LLMs alternately play the roles of patch submitter and test reviewer on real GitHub issues, with end-to-end verification through repository-native CI pipelines (compilation / lint / regression tests). Evaluated on 400 instances across C++, Python, Rust, and Go, the framework reveals behavioral divergence between models in terms of "aggressive patch generation" versus "defensive quality assurance."
tags:
  - ICLR 2026
  - LLM Efficiency
  - Adversarial Evaluation
  - CI Pipeline
  - Submitter-Reviewer
  - Retrieval-Augmented Code Generation (RACG)
  - Multilingual Code Benchmark
date: 2026-05-08
content_hash: 48a8d3be7213ba6d
---

# SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving

**Conference**: ICLR 2026
**arXiv**: [2505.23932](https://arxiv.org/abs/2505.23932)
**Code**: [GitHub](https://github.com/) / [HuggingFace Dataset](https://huggingface.co/)
**Area**: LLM Efficiency
**Keywords**: Adversarial Evaluation, CI Pipeline, Submitter-Reviewer, Retrieval-Augmented Code Generation (RACG), Multilingual Code Benchmark

## TL;DR
This paper proposes SwingArena, an adversarial evaluation framework in which two LLMs alternately play the roles of patch submitter and test reviewer on real GitHub issues, with end-to-end verification through repository-native CI pipelines (compilation / lint / regression tests). Evaluated on 400 instances across C++, Python, Rust, and Go, the framework reveals behavioral divergence between models in terms of "aggressive patch generation" versus "defensive quality assurance."

## Background & Motivation

**Background**: Evaluation of LLM coding capabilities has evolved from function-level snippets in HumanEval/MBPP to repository-level issue resolution in SWE-Bench. SWE-Bench anchors evaluation on real GitHub issues, which represents a significant advance; however, its core judgment criterion remains whether a patch can pass a set of predefined unit tests.

**Limitations of Prior Work**: Existing benchmarks exhibit three critical blind spots. First, static benchmarks rely on fixed, predictable test cases and cannot simulate the dynamic process in real development where reviewers actively construct corner cases to challenge patch quality. Second, nearly all benchmarks adopt a single-agent paradigm—the model generates a patch and a test suite scores it—lacking the iterative adversarial game between submitter and reviewer. Third, benchmarks such as SWE-Bench focus exclusively on Python, neglecting equally mainstream languages such as C++, Rust, and Go, and do not execute a complete CI pipeline (multi-gate checks including compilation, linting, style validation, security scanning, and regression testing), which is severely misaligned with industrial practice.

**Key Challenge**: Real software development is inherently adversarial—contributors submit PRs and reviewers not only inspect code logic but also actively write targeted tests to expose weaknesses in patches. This dynamic game is a dimension that static evaluation cannot capture. Furthermore, real-world repositories often contain tens of thousands of lines of code with relevant information scattered across multiple files, making efficient retrieval of key code context within a limited token window a core challenge.

**Goal**: (1) Design an adversarial dual-agent evaluation protocol that enables models to demonstrate both patch generation and test construction capabilities simultaneously; (2) Build a real-world CI evaluation benchmark spanning four languages, elevating the evaluation criterion from "passing unit tests" to "passing a complete CI pipeline"; (3) Provide a multilingual RACG baseline that uniformly addresses the long-context challenge.

**Key Insight**: The authors observe that in real PR reviews, the reviewer role is essentially an "adversarial test generator"—the goal is not to confirm correctness but to find defects. This adversarial dynamic can be naturally modeled with a dual-agent framework: one agent generates the patch (submitter) and the other generates targeted tests (reviewer), with alternating roles and iterative competition.

**Core Idea**: Upgrade LLM code evaluation from "static generation + fixed tests" to "dual-agent adversarial + full CI verification," using submitter-reviewer role swapping to capture the dynamic collaborative nature of real software development.

## Method

### Overall Architecture
SwingArena operates at three levels: a data construction layer (mining real GitHub issues and ensuring CI reproducibility), an adversarial evaluation layer (iterative battle between submitter and reviewer agents), and a retrieval support layer (RACG providing coding context to the models). The input is a GitHub issue description with a buggy program and the corresponding repository code; the output consists of metrics such as win rate and CI pass rate across multiple rounds of battle between the two agents.

### Key Designs

1. **Adversarial Battle Protocol**:

    - Function: Simulates the dynamic adversarial game of real-world PR reviews, simultaneously evaluating models' "patching capability" and "verification capability."
    - Mechanism: Two LLMs play the roles of Submitter and Reviewer respectively. The Submitter generates a fix patch based on the issue description and retrieved code context; the Reviewer writes targeted test cases based on the patch diff, focusing on probing edge cases and logical defects. The scoring rule is clear—if the Submitter's patch passes all CI checks (including the Reviewer's tests), it scores +1; otherwise −1. If the Reviewer's tests pass on the golden patch but fail on the Submitter's patch, the Reviewer scores +1; if the tests fail even on the golden patch, the Reviewer scores −1. Each battle consists of 10 rounds, with each agent acting as submitter 5 times and reviewer 5 times, ensuring role symmetry.
    - Design Motivation: Role swapping is the key—the same model must demonstrate both generation and verification capabilities, adding a dimension beyond static evaluation. Quality gate constraints on the Reviewer (tests must pass the golden patch, modifying production code is prohibited, line count is limited, non-deterministic logic is forbidden) prevent exploitative behavior such as writing trivial tests to game the score.

2. **RACG Retrieval-Augmented Code Generation**:

    - Function: Within a limited token window, provides the most relevant code context to the model, uniformly handling the long-context challenge across four languages.
    - Mechanism: A three-stage pipeline—FileRetriever performs coarse file-level ranking via BM25 sparse retrieval, selecting top-k candidate files from the issue description to source code; CodeChunker performs syntax-aware chunking on candidate files (splitting by function/class/code block, with language-specific parsing rules for C++/Python/Rust/Go, falling back to regex chunking on parse failure); CodeReranker encodes queries and code chunks as dense vectors using CodeBERT and reranks by cosine similarity, incorporating language-aware scoring (prioritizing definitions over references), proximity bias (boosting chunks near already-selected ones), and cross-file deduplication. A Token Budget Manager then dynamically allocates the budget, selecting coarse-grained chunks when space is sufficient and switching to fine-grained chunks under tight constraints.
    - Design Motivation: Existing code RAG solutions (e.g., SWE-Agent's AST parsing) typically support only Python and lack cross-language support and token budget management. RACG's syntax-aware chunking ensures semantic completeness of code blocks (no truncation mid-function), while dynamic budget management ensures fairness across models with different context window sizes.

3. **Data Construction and CI Reproducibility**:

    - Function: Constructs a high-quality, CI-reproducible evaluation dataset from real GitHub repositories.
    - Mechanism: A four-stage filtering pipeline—PR-Issue pairs are mined from high-star repositories via the GitHub API (approximately 2,300 pairs); CI test filtering retains only instances where all CI checks pass; LLM-as-Judge (Grok-3-beta) evaluates clarity and difficulty of issue descriptions (requiring justifications for assessments); and human expert validation corrects LLM evaluation biases. The CI environment for each repository is fully reproduced in an isolated Docker container, supporting GitHub Actions and Travis CI, with language-specific build systems (e.g., Rust's cargo) fully preserved.
    - Design Motivation: Data quality directly determines evaluation credibility. The two-stage LLM filtering + human validation strategy balances efficiency and accuracy. Docker isolation ensures zero contamination across tasks, and temperature=0 with fixed random seeds guarantees reproducibility.

## Key Experimental Results

### Main Results: Closed-Source Model Adversarial Battle

| Configuration | Submitter | Reviewer | RPR | SPR | Win Rate |
|---|---|---|---|---|---|
| GPT-4o vs GPT-4o | GPT-4o | GPT-4o | 0.71 | 0.68 | 0.97 |
| Claude vs Claude | Claude | Claude | 0.62 | 0.62 | 1.00 |
| Gemini vs Gemini | Gemini | Gemini | 0.72 | 0.63 | 0.91 |
| DeepSeek vs DeepSeek | DeepSeek | DeepSeek | 0.70 | 0.66 | 0.96 |
| GPT-4o vs Claude | GPT-4o | Claude | 0.65 | 0.55 | 0.90 |
| Claude vs GPT-4o | Claude | GPT-4o | 0.66 | 0.55 | 0.89 |
| Gemini vs DeepSeek | Gemini | DeepSeek | 0.64 | 0.64 | 1.00 |
| DeepSeek vs Gemini | DeepSeek | Gemini | 0.68 | 0.64 | 0.96 |

RPR = proportion of Reviewer tests passing on the golden patch; SPR = proportion of Submitter patches passing CI; Win Rate = proportion of cases where the submitter ultimately passes all checks.

### Multilingual Best@3 and RACG Ablation

| Model / Config | Avg | C++ | Go | Rust | Python |
|---|---|---|---|---|---|
| DeepSeek-V3 | **0.59** | 0.64 | 0.61 | **0.58** | 0.52 |
| Gemini-2.0 | 0.57 | 0.64 | 0.58 | 0.51 | **0.57** |
| GPT-4o | 0.57 | 0.63 | 0.53 | 0.56 | 0.54 |
| Claude-3.5 | 0.55 | 0.63 | 0.55 | 0.52 | 0.50 |

| RACG Ablation Config | Best@3 | Win Rate |
|---|---|---|
| C++ w/ RACG | 0.42 | 0.84 |
| C++ w/o RACG | 0.38 | 0.77 |
| Python w/ RACG | 0.46 | 0.84 |
| Python w/o RACG | 0.44 | 0.71 |
| Rust w/ RACG | 0.58 | 0.75 |
| Rust w/o RACG | 0.49 | 0.72 |
| Go w/ RACG | 0.45 | 0.80 |
| Go w/o RACG | 0.37 | 0.71 |
| BM25-only | 0.38 | 0.62 |
| Top-20 Related + Rerank | 0.43 | 0.73 |

### Key Findings
- **GPT-4o is the most aggressive patch generator**: Its Win Rate as submitter is ≥0.90 regardless of the opponent, indicating extremely strong patch "offensiveness." However, its SPR (0.55–0.68) is not the highest, suggesting that part of the high Win Rate stems from the opposing reviewer's tests being insufficiently rigorous.
- **DeepSeek-V3 and Gemini lead in CI stability**: DeepSeek consistently maintains high RPR/SPR (0.60–0.70 / 0.55–0.66), and Gemini achieves an RPR of 0.72 in self-play. These two models tend toward a "defensive" style—their patches may not be the most aggressive, but their CI pass rates are the most stable.
- **All models perform best on C++, with Python and Rust relatively weaker**: This may be because C++ CI pipelines are more standardized, while Python's non-deterministic tests and Rust's strict compiler introduce additional challenges.
- **Reviewer identity subtly influences battle outcomes**: The slight asymmetry between GPT-4o vs. Claude (0.90) and Claude vs. GPT-4o (0.89) indicates that differences in reviewer "strictness" style affect the submitter's final performance.
- **RACG provides the largest gain on Go** (Best@3 from 0.37 to 0.45, Win Rate from 0.71 to 0.80) and a significant gain on Rust (Best@3 from 0.49 to 0.58), indicating that retrieval augmentation is more critical for languages with more dispersed code context.
- **Retrieval granularity analysis** shows that moving from BM25 to class-level chunking improves the Top-10 file hit rate from 20.7% to 48.7%; however, class-level chunks are often too large and exceed the context window, making block-level reranking the optimal practical trade-off.

## Highlights & Insights
- **Adversarial evaluation reveals dimensions invisible to static benchmarks**: The same model may behave very differently in submitter versus reviewer roles (e.g., GPT-4o excels at attacking but is mediocre at verification); this behavioral divergence can only be exposed under a dual-agent interaction setting. This design principle can be transferred to any scenario requiring simultaneous evaluation of "generation" and "discrimination" capabilities.
- **Using the complete CI pipeline as the evaluation criterion** is the core upgrade of this paper: moving from "passing unit tests" to "passing compilation + lint + style checks + regression tests + reviewer tests" more closely reflects industrial standards for code quality. The finding that 24% of failures stem from non-functional requirement violations (style/security) demonstrates that functional correctness alone is far from sufficient.
- **RACG's token budget management strategy** cleverly switches between coarse and fine granularity based on remaining window space, ensuring fairness across models with different context window sizes. This adaptive packing idea has broad reference value for any RAG scenario.
- **Failure mode analysis** reveals that 31% of failures stem from cross-file consistency issues (the model fixed the main file but forgot to update header files/API definitions), pointing to a fundamental shortcoming of current LLMs in "architecture-level reasoning."

## Limitations & Future Work
- **Retrieval bottleneck**: The fixed Top-5 file retrieval limit in RACG may become a bottleneck for complex issues—the paper's failure analysis attributes 26% of errors to failure to retrieve the correct target file. A more dynamic retrieval strategy (e.g., adaptively adjusting $k$ based on issue complexity) is an obvious improvement direction.
- **High evaluation cost**: Each battle pair requires multiple CI executions (Docker build + full test suite), making large-scale evaluation demanding in both compute and time. Future work could explore lightweight CI agents or incremental verification to reduce costs.
- **Insufficient language and scale coverage**: Only 4 languages are supported, leaving out mainstream languages such as Java and TypeScript; 400 evaluation instances (100 per language) may be insufficient for fine-grained statistical analysis, and confidence intervals for some paired cross-experiment results are not reported.
- **Insufficient evaluation of open-source models**: The main experiments focus on 4 closed-source models, with open-source models only supplementally evaluated using Qwen2.5-Coder-7B/14B and Seed-Coder-8B; systematic evaluation of mainstream open-source code models such as CodeLlama and StarCoder2 is absent.
- **Reviewer quality gates may be overly strict**: Constraints such as prohibiting non-determinism and limiting line count prevent exploits but may also filter out valuable concurrency and performance tests.

## Related Work & Insights
- **vs. SWE-Bench**: SWE-Bench is a Python-only static issue repair benchmark judged by predefined unit tests. SwingArena extends it along three dimensions—multilingual, complete CI, and adversarial dual-agent. However, SWE-Bench's data scale (2,294 instances in Python alone) far exceeds SwingArena's 100 instances per language.
- **vs. Multi-SWE-Bench / SWE-PolyBench**: These extensions add multilingual support but still rely on manual Docker configuration and adopt static evaluation, lacking the adversarial interaction dimension.
- **vs. Agent-as-a-Judge**: The idea of using an agent to evaluate another agent is similar, but Agent-as-a-Judge focuses on scoring consistency, whereas SwingArena focuses on exposing capability differences across dimensions through adversarial interaction.
- **vs. Agentless**: Agentless uses BM25 + AST for code localization; SwingArena's RACG builds upon this by adding CodeBERT reranking and token budget management, constituting a more complete retrieval baseline.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The adversarial CI evaluation framework is pioneering in code LLM benchmarking; the submitter-reviewer role swapping is a novel and intuitive design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 closed-source models × 4 languages × multiple pairing configurations, with ablation studies and failure analysis included; however, open-source model evaluation is insufficiently systematic and some experiments lack error estimates.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, battle protocol and scoring mechanism are precisely defined, and the appendix's failure mode analysis is valuable; some sections contain redundancy.
- Value: ⭐⭐⭐⭐ Provides a comprehensive upgrade paradigm for code LLM evaluation—from static to dynamic, from single-language to multilingual, and from unit tests to complete CI.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[ICLR 2026\] When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)
- [\[NeurIPS 2025\] Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context](../../NeurIPS2025/llm_efficiency/technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)
- [\[AAAI 2026\] Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](../../AAAI2026/llm_efficiency/harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)
- [\[NeurIPS 2025\] The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](../../NeurIPS2025/llm_efficiency/the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)

<!-- RELATED:END -->
