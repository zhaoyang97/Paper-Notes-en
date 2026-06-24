---
title: >-
  [Paper Note] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models
description: >-
  [ICLR 2026][Model Compression][Benchmark Evaluation] The authors propose BeyondBench, an evaluation framework that algorithmically and dynamically generates mathematical problems (44 tasks / 117 variants / 3 difficulty levels). This ensures each test is free from training data contamination. After evaluating 101 language models (0.5B to 141B parameters), it was found that even the strongest models only achieve a 56% accuracy rate on the Hard Suite…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Benchmark Evaluation"
  - "Data Contamination"
  - "Reasoning Ability"
  - "Algorithmic Task Generation"
  - "NP-Complete Problems"
date: 2026-05-08
content_hash: 49007d789630a208
---

# BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models

**Conference**: ICLR 2026  
**arXiv**: [2509.24210](https://arxiv.org/abs/2509.24210)  
**Code**: [GitHub](https://github.com/ctrl-gaurav/BeyondBench) / [PyPI](https://pypi.org/project/beyondbench/) / [Leaderboard](https://ctrl-gaurav.github.io/BeyondBench/)  
**Area**: LLM Evaluation / Model Compression  
**Keywords**: Benchmark Evaluation, Data Contamination, Reasoning Ability, Algorithmic Task Generation, NP-Complete Problems

## TL;DR
The authors propose BeyondBench, an evaluation framework that algorithmically and dynamically generates mathematical problems (44 tasks / 117 variants / 3 difficulty levels). This ensures each test is free from training data contamination. After evaluating 101 language models (0.5B to 141B parameters), it was found that even the strongest models only achieve a 56% accuracy rate on the Hard Suite, and performance drops significantly when tools are not utilized.

## Background & Motivation
Language model evaluation faces an increasingly severe issue of **data contamination**: as model training data scales (covering vast amounts of internet text), static benchmark questions may already exist in the training sets. This allows models to achieve high scores through "memorization" rather than "reasoning," leading to inflated benchmark scores that fail to reflect true reasoning capabilities.

Existing benchmarks (e.g., GSM8K, MATH, ARC) are static datasets that are likely "absorbed" into training data once made public. While some efforts attempt to mitigate this through data deduplication, the fundamental issue remains that static datasets have limited scale and cannot fundamentally prevent contamination.

**Key Challenge**: There is a need to evaluate the "true reasoning ability" of models, but any public, fixed set of questions carries the risk of being contaminated.

**Key Insight**: This work completely abandons static question banks in favor of **algorithmic dynamic generation**. New problem instances are generated online for every evaluation, with a problem space exceeding $10^{15}$ combinations, making the coverage in any pre-training corpus move toward zero. Furthermore, every problem has a deterministically verifiable solution to ensure objective evaluation.

## Method

### Overall Architecture
BeyondBench is an installable Python evaluation package (`pip install beyondbench`). The core idea is to replace the "task bank" with a "task generator": for each evaluation, new problem instances are algorithmically generated online according to specified suites and difficulty levels. After being sent to the model under evaluation, the outputs are validated item-by-item against deterministic ground-truth answers. The framework calculates accuracy, instruction-following compliance, and token efficiency. The framework itself does not train any models and supports three types of backends: OpenAI/Gemini/Anthropic APIs, vLLM local inference, and HuggingFace Transformers.

### Key Designs

**1. Three-tier Difficulty Task Suites: Building a Reasoning Ladder by Computational Complexity**

To distinguish between "arithmetic proficiency" and "algorithmic thinking," difficulty must increase along an interpretable axis. BeyondBench decomposes reasoning into three suites of increasing difficulty corresponding to computational complexity levels. The Easy Suite contains 29 tasks covering basic arithmetic and statistics such as sorting, summation, mean, median, and GCD/LCM, representing basic operations solvable in polynomial time. The Medium Suite contains 5 tasks and 49 variants, shifting to sequence pattern recognition (Fibonacci variants, sequence discovery, pattern matching), testing inductive reasoning rather than brute calculation. The Hard Suite contains 10 tasks and 68 variants, introducing NP-complete and constraint satisfaction problems such as Graph Coloring, Knapsack, Traveling Salesman variants, and Satisfiability (SAT). These are inherently computationally difficult, forcing models to engage in combinatorial search or heuristic reasoning. This progressive hierarchy exposes layers of reasoning ability; experiments demonstrate a cliff-like performance drop from Easy to Hard, helping determine if a model is "reasoning" or "pattern matching."

**2. Triple Contamination-Resistance Guarantees: Making Questions Impossible to "Memorize"**

The foundation of the framework is the impossibility of questions appearing in training corpora, achieved through three overlapping mechanisms. First is the massive problem space: instance combinations for each task exceed $10^{15}$, far surpassing the coverage of any static corpus. Second is the deterministic verifiable solution: every generated instance has a mathematically unique correct answer, meaning verification does not rely on human judges and has no scoring ambiguity. Third is isomorphic transformation: the framework applies semantically equivalent but syntactically different rewrites to the same problem (e.g., renumbering graph nodes, replacing variable names) to generate variants that "look different but are essentially the same," further reducing the probability of winning through surface-level memory. Together, these ensure that the shortcut of "recalling training data" is ineffective on BeyondBench, forcing scores to reflect real-time reasoning.

**3. Multi-dimensional Evaluation Metrics: Decomposing "Success" into Interpretable Facets**

Relying solely on accuracy can hide significant information. Therefore, the framework records four types of signals for every evaluation: accuracy (calculated by task and suite), instruction-following compliance (measuring if the model outputs answers in the required format), token efficiency (reflecting the tokens consumed to reach an answer), and three-fold evaluation (averaging results across three runs for each configuration). Decoupling accuracy from instruction-following reveals whether a model is "capable and obedient" or "capable but format-agnostic." Three-fold averaging dampens random fluctuations from dynamic generation, ensuring scores are comparable.

**4. Out-of-the-box Evaluation Toolchain: Benchmarking as an Accessible Utility**

BeyondBench is more than a paper; it is a set of installable engineering utilities (`pip install beyondbench`) that lowers the barrier for large-scale evaluation to a single package installation. The CLI command `beyondbench evaluate --model-id xxx --suite easy` runs an evaluation in one line; the Python API allows for programmatic control; `beyondbench serve` launches a FastAPI service to provide REST-based evaluation; and `beyondbench results compare` enables horizontal comparison across models. This toolchain, combined with unified wrappers for OpenAI/Gemini/Anthropic, vLLM, and HuggingFace backends, supported the large-scale evaluation of 101 models.

## Key Experimental Results

### Main Results: Large-scale Evaluation of 101 Models
The study evaluated 85 open-source and 16 closed-source models, ranging from 0.5B to 141B parameters:

**Top 5 Leaderboard (Using Tools/Reasoning Tokens)**:

| Rank | Model | Hard Suite Accuracy | Easy Suite Accuracy |
|------|------|-----------------|-----------------|
| 🥇 | GPT-5* | Not Specified | 96.15% |
| 🥈 | GPT-5-Nano* | Not Specified | 93.58% |
| 🥉 | GPT-5-Mini* | Not Specified | 94.23% |
| 4 | o3* | Not Specified | 94.96% |
| 5 | o4-Mini* | Not Specified | 95.30% |

(*Models using reasoning/thought tokens)

**Representative Model Performance on Hard Suite**:

| Model | Hard Suite Accuracy |
|------|-----------------|
| Gemini-2.5-pro | 56.21% |
| Qwen2.5-72B | 33.37% |
| Llama-3.3-70B | 27.16% |

### Impact of Tool-Use vs. No-Tool

| Model | Overall Accuracy Drop (No Tools) |
|------|----------------------|
| GPT-5 | -16.81% |
| GPT-5-mini | -15.86% (or -28.05%) |
| GPT-5-nano | -43.95% (or -47.59%) |

Tool-use (such as code execution) has a massive impact on reasoning performance, particularly for smaller models.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Easy→Medium→Hard | Step-wise Performance Drop | Performance drops sharply from polynomial to exponential complexity |
| Model Scaling Effect | Larger Models Usually Better | However, the relationship is not strictly linear |
| Quantization Impact | Various Schemes Tested | Quantization affects different tasks inconsistently |
| Instruction-Following vs. Accuracy | Inconsistent | High accuracy does not guarantee perfect instruction-following |

### Key Findings
- **Reasoning Performance Degrades Sharply with Complexity**: Even the strongest models show significant performance drops from Easy to Hard, suggesting current LLM "reasoning" relies more on pattern matching than genuine algorithmic thinking.
- **Tool-Use is Critical**: Performance on mathematical and algorithmic problems drops significantly without code execution tools, especially for smaller models.
- **Scaling Effects Exist but are Limited**: Larger models perform better on the Hard Suite, but the gap between 70B and 141B models is much smaller than the gaps observed in the Easy Suite.
- **Open-source vs. Closed-source Gap**: Closed-source models (especially those with reasoning capabilities like o3 or GPT-5) lead open-source models significantly on the Hard Suite.

## Highlights & Insights
- **Evaluation Paradigm Shift**: The transition from "static banks" to "dynamic generation" is a major methodological advancement that fundamentally solves the data contamination problem.
- **Unprecedented Scale**: The horizontal comparison of 101 models provides an unprecedented overview of the landscape.
- **Engineering Completeness**: More than just a paper, it provides a complete open-source toolset—Python package, CLI, API server, and online leaderboard—lowering usage barriers.
- **NP-complete Problems as Reasoning Upper Bound**: Using computationally hard problems from theoretical computer science to test LLMs provides valuable insights into the upper bounds of reasoning capabilities.
- **Tool-use vs. No-tool Performance Comparison**: This reveals the gap between a model's true understanding of a problem and its ability to translate it into code.

## Limitations & Future Work
- All tasks are mathematical/algorithmic, failing to cover natural language reasoning, commonsense reasoning, or causal reasoning.
- Dynamically generated problem formats may differ from formats common in pre-training data, leading to potential format bias.
- Problems in the Easy Suite might be too simple (basic arithmetic), offering limited discriminative power.
- Reliance on deterministic answers precludes the evaluation of open-ended reasoning.
- Three-fold evaluation improves robustness but increases evaluation costs.
- NP problems in the Hard Suite might favor models using brute-force search (via code execution), which does not necessarily reflect deep "reasoning."

## Related Work & Insights
- Compared to static math benchmarks like GSM8K and MATH, BeyondBench fundamentally avoids contamination.
- Similar to dynamic benchmarks like LiveBench, but BeyondBench offers a larger problem space ($>10^{15}$) and covers NP-complete problems.
- Compared to synthetic reasoning benchmarks like PrOntoQA, BeyondBench focuses on a broader range of algorithmic reasoning rather than a single reasoning type.
- Insight: Future benchmark designs should shift toward the "dynamic generation + deterministic verification" paradigm instead of relying on manually annotated static datasets.
- The distinction between "reasoning ability" and "tool-use capability" is vital for understanding and developing true intelligence in LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ — Dynamic evaluation is not a new concept, but the systematicity and scale are unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 101 models, 3 difficulty levels, multiple quantization schemes, and Tool-use vs. No-tool comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Abstracts and framework descriptions are clear, though certain detailed evaluations were restricted by document conversion issues.
- Value: ⭐⭐⭐⭐⭐ — High practical value for the LLM evaluation community; the tools are open-sourced and ready for immediate use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling](towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation](../../ACL2026/model_compression/introlm_introspective_language_models_via_prefilling-time_self-evaluation.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)

</div>

<!-- RELATED:END -->
