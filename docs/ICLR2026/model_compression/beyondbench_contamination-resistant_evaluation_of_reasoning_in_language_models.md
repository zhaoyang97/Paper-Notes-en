---
title: >-
  [Paper Note] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models
description: >-
  [ICLR 2026][Model Compression][Benchmark Evaluation] This paper proposes BeyondBench, an evaluation framework that algorithmically generates mathematical problems on-the-fly (44 tasks / 117 variants / 3 difficulty levels) to ensure each evaluation instance is free from training data contamination. It evaluates 101 language models (0.5B–141B parameters), finding that even the strongest models achieve only 56% accuracy on the Hard Suite, with substantial performance drops when tools are unavailable.
tags:
  - ICLR 2026
  - Model Compression
  - Benchmark Evaluation
  - Data Contamination
  - Reasoning
  - Algorithm Problem Generation
  - NP-Complete Problems
date: 2026-05-08
content_hash: af631802635d97e4
---

# BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.24210](https://arxiv.org/abs/2509.24210)
**Code**: [GitHub](https://github.com/ctrl-gaurav/BeyondBench) / [PyPI](https://pypi.org/project/beyondbench/) / [Leaderboard](https://ctrl-gaurav.github.io/BeyondBench/)
**Area**: LLM Evaluation / Model Compression
**Keywords**: Benchmark Evaluation, Data Contamination, Reasoning, Algorithm Problem Generation, NP-Complete Problems

## TL;DR
This paper proposes BeyondBench, an evaluation framework that algorithmically generates mathematical problems on-the-fly (44 tasks / 117 variants / 3 difficulty levels) to ensure each evaluation instance is free from training data contamination. It evaluates 101 language models (0.5B–141B parameters), finding that even the strongest models achieve only 56% accuracy on the Hard Suite, with substantial performance drops when tools are unavailable.

## Background & Motivation

- **Background**: Language model evaluation faces an increasingly severe **data contamination** problem. As training corpora grow to encompass vast amounts of internet text, the questions in static benchmarks may already appear in training data, enabling models to achieve high scores through memorization rather than genuine reasoning—leading to inflated benchmark numbers that fail to reflect true reasoning ability.
- **Limitations of Prior Work**: Existing benchmarks (e.g., GSM8K, MATH, ARC) are static datasets that, once released, risk being absorbed into the training data of subsequent models. While some works attempt deduplication as a mitigation, the fundamental issue is that static datasets are finite and cannot structurally eliminate contamination.
- **Key Challenge**: We need to evaluate models' *genuine reasoning ability*, yet any publicly fixed question set is inherently at risk of contamination.
- **Key Insight**: This paper abandons static question banks entirely and instead adopts **algorithmic dynamic generation**—generating fresh problem instances online at every evaluation, with a problem space exceeding $10^{15}$ combinations, driving the coverage probability of any pretraining corpus toward zero. Each problem also admits a deterministically verifiable solution, ensuring objective evaluation.

## Method

### Overall Architecture
BeyondBench is an installable Python package (`pip install beyondbench`) supporting multiple backends (OpenAI, Gemini, Anthropic APIs; vLLM local inference; HuggingFace Transformers). The workflow is: (1) generate problem instances online according to the specified Suite and difficulty level; (2) submit problems to the target model; (3) parse model responses and compare against deterministic ground-truth answers; (4) compute metrics including accuracy, instruction-following compliance, and token efficiency.

### Key Designs

1. **Three-Tier Difficulty Suites**:

   - **Easy Suite (29 tasks)**: Fundamental arithmetic and statistical problems—sorting, summation, mean, median, GCD/LCM, etc.—assessing basic mathematical operation ability.
   - **Medium Suite (5 tasks, 49 variants)**: Sequence pattern recognition and reasoning problems, including Fibonacci variants, sequence rule discovery, and pattern matching, requiring inductive reasoning.
   - **Hard Suite (10 tasks, 68 variants)**: NP-complete and constraint satisfaction problems—graph coloring, knapsack, TSP variants, SAT problems—which are computationally hard and require combinatorial search or heuristic reasoning.

2. **Contamination Resistance: Three-Layer Guarantee**:

   - **Vast problem space**: Each task's instance space exceeds $10^{15}$, making static dataset coverage impossible.
   - **Deterministically verifiable solutions**: Every generated instance has a mathematically unique correct answer, eliminating evaluation ambiguity.
   - **Isomorphic transformations**: Problems can be subjected to semantically equivalent but syntactically distinct transformations (e.g., renumbering graph nodes, renaming variables), producing instances that appear different but are structurally identical, further reducing memorization-based matching.

3. **Multi-Dimensional Evaluation Metrics**:

   - Accuracy: reported per task and per Suite.
   - Instruction-following compliance: whether models output answers in the required format.
   - Token efficiency analysis: number of tokens consumed to reach an answer.
   - Three-fold evaluation: each configuration is run three times and averaged for robustness.

4. **Complete Toolchain**:

   - CLI: `beyondbench evaluate --model-id xxx --suite easy`
   - Python API: programmatic control over the evaluation pipeline.
   - FastAPI server: `beyondbench serve` exposes a REST API.
   - Result comparison: `beyondbench results compare` for cross-model comparison.

### Loss & Training
Not applicable—BeyondBench is a pure evaluation framework involving no training.

## Key Experimental Results

### Main Results: Large-Scale Evaluation of 101 Models
The study evaluates 85 open-source and 16 closed-source models ranging from 0.5B to 141B parameters.

**Top-5 Leaderboard (with tool use / reasoning tokens)**:

| Rank | Model | Hard Suite Acc. | Easy Suite Acc. |
|------|-------|-----------------|-----------------|
| 🥇 | GPT-5* | N/A | 96.15% |
| 🥈 | GPT-5-Nano* | N/A | 93.58% |
| 🥉 | GPT-5-Mini* | N/A | 94.23% |
| 4 | o3* | N/A | 94.96% |
| 5 | o4-Mini* | N/A | 95.30% |

(*Models using reasoning/thinking tokens)

**Hard Suite Performance of Representative Models**:

| Model | Hard Suite Accuracy |
|-------|---------------------|
| Gemini-2.5-pro | 56.21% |
| Qwen2.5-72B | 33.37% |
| Llama-3.3-70B | 27.16% |

### Impact of Tool Use vs. No Tool Use

| Model | Overall Accuracy Drop (No Tools) |
|-------|----------------------------------|
| GPT-5 | −16.81% |
| GPT-5-mini | −15.86% (or −28.05%) |
| GPT-5-nano | −43.95% (or −47.59%) |

Tool use (e.g., code execution) has a substantial impact on reasoning performance, with the effect being especially pronounced for smaller models.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Easy → Medium → Hard | Monotonically decreasing performance | Complexity scales from polynomial to exponential; performance drops sharply |
| Model scale effect | Larger models generally perform better | Relationship is not strictly linear |
| Quantization impact | Multiple quantization schemes tested | Effect varies across tasks |
| Instruction-following vs. accuracy | Inconsistent | High accuracy does not guarantee perfect instruction-following compliance |

### Key Findings
- **Reasoning degrades sharply with complexity**: Even the strongest models exhibit drastic performance drops from Easy to Hard, suggesting that current LLM "reasoning" relies more on pattern matching than genuine algorithmic thinking.
- **Tool use is critical**: Without code execution tools, model performance on mathematical and algorithmic tasks drops substantially, particularly for smaller models.
- **Scale effects exist but are limited**: Larger models perform better on the Hard Suite, yet the gap between 70B and 141B models is far smaller than on the Easy Suite.
- **Open-source vs. closed-source gap**: Closed-source models—especially those with reasoning capabilities such as o3 and GPT-5—markedly outperform open-source counterparts on the Hard Suite.

## Highlights & Insights
- **Paradigm shift in evaluation**: The transition from "static question banks" to "dynamic generation" represents a significant methodological advance, fundamentally resolving the data contamination problem.
- **Unprecedented scale**: A cross-sectional comparison of 101 models provides a panoramic view previously unavailable to the community.
- **Engineering completeness**: Beyond being a research paper, BeyondBench is a fully functional open-source tool—Python package, CLI, API server, and online leaderboard—lowering the barrier to adoption.
- **NP-complete problems as a reasoning ceiling**: Using computationally hard problems from complexity theory to probe LLMs provides valuable insight into the upper limits of reasoning ability.
- **Tool-assisted vs. tool-free performance contrast**: This comparison reveals the gap between models truly understanding a problem and models merely transcribing it into executable code.

## Limitations & Future Work
- All tasks are mathematical/algorithmic in nature; natural language reasoning, commonsense reasoning, and causal reasoning are not covered.
- The format of dynamically generated problems may differ from problem formats commonly encountered during pretraining, introducing potential format bias.
- Easy Suite problems may be too simple (basic arithmetic) to offer meaningful discrimination.
- Reliance on deterministic answers precludes evaluation of open-ended reasoning capabilities.
- Three-fold evaluation improves robustness but increases evaluation cost.
- NP-complete problems in the Hard Suite may unduly favor models that employ brute-force search via code execution, not necessarily reflecting genuine "reasoning" ability.

## Related Work & Insights
- Compared with static mathematical benchmarks such as GSM8K and MATH, BeyondBench structurally eliminates contamination risk.
- Similar in spirit to dynamic benchmarks like LiveBench, BeyondBench offers a substantially larger problem space ($>10^{15}$) and covers NP-complete problems.
- Compared with synthetic reasoning benchmarks such as PrOntoQA, BeyondBench targets broader algorithmic reasoning rather than a single reasoning type.
- **Insight**: Future benchmark design should more broadly adopt the paradigm of "dynamic generation + deterministic verification" rather than relying on statically annotated datasets.
- The distinction between *reasoning ability* and *tool-use ability* is crucial for understanding and advancing genuine intelligence in LLMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Dynamic generation for evaluation is not an entirely new concept, but the systematicity and scale here are unprecedented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 101 models, 3 difficulty levels, multiple quantization schemes, and tool vs. no-tool comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Abstract and framework descriptions are clear; full-text assessment is limited due to HTML conversion failure.
- **Value**: ⭐⭐⭐⭐⭐ — Significant practical value for the LLM evaluation community; tooling is open-source and immediately usable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](../../ACL2026/model_compression/selar_selective_latent_reasoning_in_large_language_models.md)

<!-- RELATED:END -->
