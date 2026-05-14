---
title: >-
  [Paper Note] ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track)][Image Generation][Algorithm Engineering Benchmark] This paper introduces ALE-Bench, the first AI benchmark targeting scored algorithm engineering contests (AtCoder Heuristic C…
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track)"
  - "Image Generation"
  - "Algorithm Engineering Benchmark"
  - "Long-Horizon Reasoning"
  - "Scored Programming Contests"
  - "Iterative Optimization"
  - "LLM Agent"
date: 2026-05-08
content_hash: 248db1ea0c215c13
---

# ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering

**Conference**: NeurIPS 2025 (Datasets & Benchmarks Track)  
**arXiv**: [2506.09050](https://arxiv.org/abs/2506.09050)  
**Code**: [GitHub](https://github.com/SakanaAI/ALE-Bench)  
**Area**: Image Generation  
**Keywords**: Algorithm Engineering Benchmark, Long-Horizon Reasoning, Scored Programming Contests, Iterative Optimization, LLM Agent

## TL;DR

This paper introduces ALE-Bench, the first AI benchmark targeting scored algorithm engineering contests (AtCoder Heuristic Contest). It curates 40 NP-hard optimization problems and provides an interactive agent evaluation framework. The strongest model, o3-high, achieves only human-average performance in a one-shot setting, with significant gaps between AI and human experts in cross-problem consistency and long-horizon iterative improvement.

## Background & Motivation

**Background**: Programming benchmarks such as HumanEval, CodeContests, and LiveCodeBench focus on short-horizon, pass/fail exact-solution tasks. LLMs have approached top human performance on these benchmarks, and saturation is becoming a concern.

**Limitations of Prior Work**: Real-world problems are filled with NP-hard optimization challenges—logistics routing, factory scheduling, power grid balancing—that admit no exact solutions. Human experts spend days or weeks iteratively refining heuristics such as simulated annealing and beam search. Existing programming benchmarks cannot evaluate AI on such "long-horizon, score-driven" tasks, as they measure only single-submission performance rather than iterative improvement.

**Key Challenge**: AtCoder Heuristic Contest (AHC) is one of the world's largest scored algorithm competitions, attracting roughly 1,000 participants per round who spend weeks iteratively optimizing their solutions. However, no systematic evaluation platform exists to assess AI performance under this paradigm of long-horizon reasoning combined with repeated trial and error.

**Key Insight**: This work standardizes 40 AHC problems into an AI-accessible benchmark, provides a Session-based interactive interface that simulates the real contest workflow (read problem → test run → visualize → submit), and designs ALE-Agent as a strong baseline. The core idea is to introduce "long-horizon iterative improvement capability" as a new evaluation dimension for AI systems.

## Method

### Overall Architecture

ALE-Bench consists of three components: (1) **Dataset**—40 AHC problems including problem statements, Rust-based scorers, visualization tools, and human leaderboards, covering diverse types such as path planning, scheduling, puzzle solving, multi-agent control, and Bayesian inference; (2) **Evaluation Framework**—a Python library providing a Session interface through which AI iterates over a read → test-run (score feedback) → visualize → final-submit loop, with a Docker sandbox ensuring a consistent execution environment; (3) **ALE-Agent**—a prototype agent designed specifically for algorithm engineering.

### Key Designs

1. **Session Interaction System**:
    - Function: Provides AI with a complete interactive environment simulating a real AHC contest, supporting four operations (view problem, test run, visualize, final submit).
    - Mechanism: A timer starts when the Session object is created; AI can repeatedly execute test runs within the time limit to obtain feedback scores. Code runs in a Docker sandbox supporting C++, Python, and Rust. Visualization supports both static images and interactive web modes. Final submission triggers private evaluation and computes a Performance metric (Elo-style rating, range 0–3,500).
    - Design Motivation: To faithfully reproduce the human contestant workflow and ensure AI and humans are compared under identical conditions. Amazon EC2 C6i instances standardize the CPU environment, eliminating hardware variation from scores.

2. **ALE-Agent (Algorithm Engineering Agent)**:
    - Function: A strong baseline agent built on the ALE-Bench framework that substantially improves LLM algorithm engineering capability through domain knowledge injection and diversity-driven search.
    - Mechanism: **Method 1—Domain Knowledge Injection**: Expert knowledge of optimization algorithms such as simulated annealing and beam search is embedded directly into the prompt, covering search space design, objective function construction, neighborhood generation, and acceleration techniques. **Method 2—Diversity-Driven Search**: A best-first tree search that expands $k=30$ child nodes from the current best node in parallel, retaining multiple promising search paths in a beam-search fashion to avoid premature convergence while amortizing API latency.
    - Design Motivation: Algorithm engineering has well-established "standard techniques" (simulated annealing, greedy search, etc.), and selecting the correct high-level strategy is critical. However, even with the right strategy, implementation details and hyperparameter tuning greatly affect results, motivating diversity search to explore different implementation variants.

3. **Evaluation Metric System**:
    - Function: Provides two-level evaluation—fine-grained (per-problem Performance) and aggregate (average performance, rating)—supporting fair comparison between AI and humans.
    - Mechanism: Performance is computed from rankings using an Elo-rating method, ranging from 0 to 3,500 and comparable across problems. The aggregate metric of choice is average performance (rating can be inflated by individual high-scoring problems, overestimating AI capability). Both a lite version (10 problems) and a full version (40 problems) are provided.
    - Design Motivation: Scored tasks cannot be evaluated with pass/fail; continuous Performance metrics are required. The rating system was designed for humans (encouraging risk-taking strategies) and is biased for AI evaluation—a single exceptionally high score can substantially inflate the rating.

### Loss & Training

As an evaluation benchmark, no training is involved. ALE-Agent employs an iterative prompting strategy: each round incorporates a history summary, the current best code, and improvement hints. Test-run feedback (score and error messages) is appended directly to the next prompt.

## Key Experimental Results

### Main Results

| Model / Setting | Avg Perf (short) | Avg Perf (long) | Avg Perf (overall) | Rating | Top% |
|---|---|---|---|---|---|
| o3-high (one-shot) | 1116 | 946 | 1044 | 1456 | 43.2% |
| Claude 3.7 Sonnet (one-shot) | 851 | 810 | 833 | 1197 | 63.2% |
| GPT-4.1 mini (iterative) | 1293 | 1114 | 1217 | 1636 | 30.5% |
| o4-mini-high (iterative) | 1677 | 1307 | 1520 | 2104 | 11.8% |
| Human Average | — | — | 1260 | 1414 | — |

### Ablation Study

| Configuration | Avg Perf (lite) | Notes |
|---|---|---|
| Self-Refine Baseline | 1264 | Simple iterative refinement |
| + Domain Knowledge (M1) | Marginal gain | Improved high-level strategy selection |
| + Diversity Search (M2) | 1879 | Large gain from exploring multiple paths |
| ALE-Agent (M1+M2) | **1879** | Corresponding rating 2222, top 8.6% |
| Best of 150 independent one-shot runs | < iterative | Confirms benchmark measures iterative reasoning, not parallel exploration |

### Key Findings

- Iterative settings significantly outperform one-shot: all models improve average performance by 400+ points under iterative evaluation.
- Reasoning models (e.g., o3-high) clearly outperform non-reasoning models, yet even the strongest o3-high in one-shot only reaches human-average performance.
- C++ code quality consistently exceeds Python and Rust: C++ outperforms Python by approximately 44 points and Rust by approximately 57 points in average performance.
- AI performs relatively well on simulated-annealing-type problems but lags most behind humans on planning problems requiring problem-specific insight.
- ALE-Agent participated in AHC046 and ranked 154th (performance 1,915), validating the benchmark's fidelity and the agent's real-world competitiveness.

## Highlights & Insights

- ALE-Bench fills an important gap in AI evaluation: existing benchmarks measure "problem-solving ability," whereas ALE-Bench measures "iterative improvement ability"—one of the most critical capabilities human experts exercise in real-world work. The absence of a score ceiling ensures discriminative power even if AI surpasses the human top score.
- ALE-Agent's diversity search strategy (beam=30 parallel generation) is an interesting inference-time scaling approach that simultaneously serves the practical purpose of amortizing API latency.
- The results expose a core weakness of current AI: not "single-trial peak performance" but "cross-problem consistency"—AI achieves very high scores on some problems while performing mediocrely on others, whereas human experts exhibit a more uniform score distribution.

## Limitations & Future Work

- The dataset scale is limited (40 problems), although the long iterative horizon per problem reduces variance.
- AI primarily uses textual feedback and does not fully exploit visualization tools—human contestants rely heavily on visual debugging, and multimodal agents could potentially narrow the gap further.
- The rating metric tends to overestimate AI capability (a single high-scoring problem can substantially inflate the rating); the paper recommends using average performance instead.
- Future directions include generating synthetic problems to scale the dataset, training RL policies for long-horizon reasoning, and incorporating multimodal understanding of visualization outputs.

## Related Work & Insights

- **vs. SWE-Bench**: SWE-Bench evaluates code comprehension and repair (pass/fail); ALE-Bench evaluates algorithm design and iterative optimization (scored). The two are orthogonal and complementary.
- **vs. MLE-Bench**: MLE-Bench draws from Kaggle ML competitions (requiring GPUs, emphasizing data science); ALE-Bench focuses on algorithm engineering (CPU-only, lower cost).
- **vs. FunSearch**: FunSearch uses LLMs to improve local fragments of human-written template code; ALE-Bench requires AI to design and iteratively improve complete solutions from scratch.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First scored long-horizon algorithm engineering benchmark with a uniquely targeted evaluation dimension.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 models × 3 languages × 3 settings with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; appendix is lengthy but information-rich.
- Value: ⭐⭐⭐⭐⭐ An important new dimension for AI capability assessment with potential for continuous updates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations](../../AAAI2026/image_generation/longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)
- [\[NeurIPS 2025\] CORAL: Disentangling Latent Representations in Long-Tailed Diffusion](coral_disentangling_latent_representations_in_longtailed_dif.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](../../ICCV2025/image_generation/ale_attribute_leakage_free_editing.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)

</div>

<!-- RELATED:END -->
