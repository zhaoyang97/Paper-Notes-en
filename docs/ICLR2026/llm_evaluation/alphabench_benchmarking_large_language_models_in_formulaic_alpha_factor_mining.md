---
title: >-
  [Paper Note] AlphaBench: Benchmarking Large Language Models in Formulaic Alpha Factor Mining
description: >-
  [ICLR 2026][LLM Evaluation][LLM Benchmark] AlphaBench is the first benchmark to systematically evaluate Large Language Models (LLMs) in "Formulaic Alpha Factor Mining" (FAFM). It decomposes the real workflow of quantitative researchers into three major tasks: factor generation, factor evaluation, and factor searching. By cross-evaluating over ten open-source an
tags:
  - ICLR 2026
  - LLM Evaluation
  - LLM Benchmark
date: 2026-05-08
content_hash: ab019ce4c824400d
---
# AlphaBench: Benchmarking Large Language Models in Formulaic Alpha Factor Mining

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=d97Q8r7ZKZ](https://openreview.net/forum?id=d97Q8r7ZKZ)  
**Code**: https://alphabench.cc/  
**Area**: LLM Evaluation / Quantitative Finance / Formulaic Alpha Factor Mining  
**Keywords**: Alpha Factor Mining, LLM Benchmark, Factor Generation, Factor Evaluation, Factor Searching

## TL;DR
AlphaBench is the first benchmark to systematically evaluate Large Language Models (LLMs) in "Formulaic Alpha Factor Mining" (FAFM). It decomposes the real workflow of quantitative researchers into three major tasks: factor generation, factor evaluation, and factor searching. By cross-evaluating over ten open-source and closed-source models in a real-world backtesting environment (Qlib + CSI300), the study finds that LLMs can reliably generate valid factors but perform close to random guessing when judging factor quality (evaluation task).

## Background & Motivation
**Background**: In quantitative investment, an alpha factor is a mathematical expression that extracts predictive signals from market data—composed of operators (`Mean`, `Corr`, `Std`, `Rank`, etc.) and variables (`$close`, `$volume`, etc.). After calculating factor values for each stock at each time point, one can rank stocks for portfolio selection. Factor quality is typically measured by the Information Coefficient (IC) or RankIC, representing the correlation between factor values and future returns. **Formulaic Alpha Factor Mining (FAFM)** is the continuous discovery of new, predictive factor formulas because old factors decay over time (alpha decay).

**Limitations of Prior Work**: Traditional factors were handcrafted by human experts based on financial intuition (e.g., Alpha101, Alpha158 libraries), which is limited by prior knowledge and cannot adapt quickly to market changes. Subsequent automated search using machine learning—Genetic Programming (AutoAlpha), Reinforcement Learning (AlphaGen), and Symbolic Regression—can discover new signals but entails high engineering and computational costs. Given their strengths in symbolic reasoning and code generation, LLMs are naturally suited for automated factor design, leading to the emergence of LLM-driven mining works like FAMA, QuantAgent, AlphaAgent, and Alpha-GPT.

**Key Challenge**: Despite the high interest, the real capabilities of LLMs in FAFM remain unclear. Generated factors might be biased, produce invalid/unexecutable formulas, or lack robustness under market regime shifts. Crucially, **there is no standardized benchmark** to measure where LLMs succeed or fail in the factor mining pipeline. While code generation and scientific discovery have rigorous benchmarks, FAFM does not. Furthermore, how different LLM configurations (model type, prompting paradigm, reasoning strategy) affect heavy-duty searching has not been systematically studied.

**Goal**: This paper aims to fill this gap by formalizing the role of LLMs in FAFM and designing a multi-task, multi-metric evaluation covering the entire factor lifecycle. It seeks to clarify the strengths and weaknesses of different LLMs and the impact of configuration variables.

**Key Concept**: The core idea is to decompose the real workflow of a quantitative researcher—generating, evaluating, and searching for factors—into three quantifiable benchmark tasks. All produced factors are executed and verified using the Qlib backtesting engine with real CSI300 historical data, creating the first capability map for "LLMs in Quant."

## Method

### Overall Architecture
AlphaBench is not a new model but an **evaluation protocol + dataset + indicator system**. It abstracts the FAFM workflow into three core tasks, each corresponding to a key stage in the real-world quantitative process, with sub-tasks measuring different capability dimensions:

- **Factor Generation**: Translating natural language descriptions into valid factor formulas (687 instructions).
- **Factor Evaluation**: Using LLMs as "judges" to predict factor quality without full backtesting to accelerate screening (1170 instructions).
- **Factor Searching**: Iteratively searching within a combinatorial space rather than one-shot generation (covering 3 search algorithms and 27 instructions).

All factors are required to be generated in **Qlib-compatible format** for direct execution and scoring. Search tasks use Qlib's built-in Alpha158 as the initial factor pool and utilize daily CSI300 stock data (2020–2025) covering bull, bear, and volatile markets. The authors also systematically ablate two configuration variables: model selection (over ten models including Gemini, GPT, DeepSeek, LLaMA, Qwen) and prompting methodology (vanilla vs. Chain-of-Thought).

### Key Designs

**1. Factor Generation: Measuring semantic alignment from language to executable formulas**

This task tests whether LLMs can correctly translate financial intent into valid, semantically correct formulas. It is split into: **Text2Alpha**, which provides broad descriptions (e.g., momentum, mean reversion) for translation without examples to test conceptual understanding; and **Directional Mining**, which provides specific themes (e.g., volatility-based signals) to test constraint following and creativity. Evaluation uses three metrics: Reliability (executable code), Stability (consistency across outputs), and Accuracy (matching user intent). An Overall score is the mean of these three. Cases are categorized by difficulty (easy/medium/hard). **Design Motivation**: Correct syntax is easy; semantic alignment under complex instructions is the true bottleneck.

**2. Factor Evaluation: LLM as a zero-shot "judge" and drilling down to atomic capabilities**

Since backtesting every candidate is expensive, a key goal is using LLMs to pre-judge quality using factor structure and economic intuition **without backtesting**. This includes: **Ranking**, where the model selects the top-$k$ from a pool (measured by precision@$k$ and rank correlation); and **Scoring**, where the model provides absolute scores (e.g., predicted IC or Sharpe), measured by Signal Accuracy and MAE. To further dissect behavior, evaluation is split into **atomic tasks**: **Signal Classification** (meaningful signal vs. noise) and **Pairwise Selection** (identifying the better of two factors). This decomposition helps locate exactly where models fail.

**3. Factor Searching: Three search paradigms + dual-axis evaluation**

Factor mining balances exploration and convergence. AlphaBench benchmarks three LLM-driven search paradigms: **Chain-of-Experience (CoE)**, sequential refinement from a seed factor using historical metrics; **Tree-of-Thought (ToT)**, branched exploration with pruning; and **Evolutionary Algorithm (EA)**, population iteration via LLM-simulated crossover and mutation. Evaluation follows two axes: **Search Cost** (rounds, token usage, reliability rate) and **Search Quality** (improvement over the initial seed for CoE/ToT; overall population performance for EA).

**4. Configuration Ablation: Model × Prompt × Temperature × Search Capacity**

AlphaBench treats configuration variables as first-class citizens. For generation and evaluation, it compares vanilla prompts vs. **Chain-of-Thought (CoT)**. For searching, it scans across different **temperatures** (exploration-exploitation trade-off) and **EA capacities** (population size). Supervised Fine-Tuning (SFT) experiments on GPT-4o-Mini verify if small amounts of labeled data fix evaluation weaknesses and test cross-market (CSI300 ↔ SP500) transferability.

## Key Experimental Results

Environment: Qlib framework + Alpha158 pool + CSI300 daily data 2020–2025. Models include closed-source (Gemini-1.5 series, GPT-4o-Mini, GPT-4o) and open-source (DeepSeek-V3, DeepSeek-R1-Distill-Qwen, LLaMA-3.1, Qwen-2.5).

### Main Results: Cross-Task Evaluation

**Factor Generation** (vanilla prompt, Overall = mean of Reliability/Stability/Accuracy):

| Model | Reliability | Stability | Accuracy | Overall |
|------|------|------|------|------|
| GPT-4o | 1.00 | 0.62 | 0.56 | **0.72** |
| Gemini-1.5-Flash | 0.99 | 0.57 | 0.57 | 0.71 |
| GPT-4o-Mini | 0.93 | 0.59 | 0.44 | 0.65 |
| LLaMA3.1-70B-Instruct | 0.95 | 0.52 | 0.38 | 0.62 |
| DeepSeek-V3 | 0.91 | 0.35 | 0.31 | 0.52 |

**Findings**: Large commercial models lead. **Reliability is generally high, but Accuracy is notably lower**—models produce valid syntax but struggle to align expressions with specific intents (especially hard instructions). CoT provides marginal gains and can sometimes decrease Stability.

**Factor Evaluation** (Vanilla / CoT, averaged across regimes):

| Model | Precision (Rank) | Signal ACC | MAE | Overall |
|------|------|------|------|------|
| Gemini-1.5-Pro | 0.24 | 0.36 | 1.66 | 0.48 |
| GPT-4o | 0.24 | 0.32 | 1.67 | 0.47 |
| Gemini-1.5-Flash | 0.25 / 0.14 | 0.32 / 0.33 | 1.67 / 1.64 | 0.47 / 0.44 |
| DeepSeek-V3 | 0.19 | 0.16 | 1.60 | 0.40 |

**Findings**: **Evaluation performance is surprisingly poor.** No model excels in both ranking and scoring. Overall scores hover around 0.40-0.48. Atomic tasks (Signal Classification, Pairwise Selection) show accuracy close to random guessing.

**Factor Searching** (Quality / Cost normalized to [0,1]):

| Model | Search Quality | Search Cost |
|------|------|------|
| GPT-4o | **0.656** | **0.940** |
| Gemini-1.5-Flash | 0.646 | 0.850 |
| LLaMA3.1-70B-Instruct | 0.624 | 0.850 |
| DeepSeek-V3 | 0.494 | 0.800 |

**Findings**: There is a clear trade-off between quality and cost. GPT-4o achieves the best balance between high search effectiveness and cost efficiency.

### Ablation Study

| Configuration | Key Finding |
|------|---------|
| Vanilla vs. CoT | CoT provides marginal gains in generation but often fails or degrades performance in evaluation tasks. |
| Temperature (Search) | High temp (1.5) increases diversity but lowers reliability and stability. Low temp (0.75) is more efficient for searching. |
| SFT (GPT-4o-Mini) | Small amounts of labeled data significantly improve Pairwise Selection (0.48→0.83 on CSI300) but have limited impact on Signal Classification. |
| Cross-market Transfer | Experience from CSI300 SFT transfers to SP500 for Pairwise Selection. Authors attribute this to the "noisier" nature of the Chinese market providing more learnable patterns. |

### Key Findings
- **Capability Disconnection**: LLMs excel at "generating valid formulas" (high Reliability) but fail at "judging factor quality" (random-level Accuracy).
- **Diminishing Returns on Scale**: Scaling model size provides limited benefits for complex evaluation tasks.
- **CoT is not a Panacea**: In FAFM tasks requiring execution verification, forced step-by-step reasoning is often unhelpful or harmful.
- **SFT Mends Evaluation Gaps**: The weakness in evaluation is likely a lack of alignment data rather than an inherent capability ceiling.

## Highlights & Insights
- **Mapping Real Workflows**: Decomposing FAFM into generation/evaluation/searching ensures the benchmark reflects real-world quantitative research.
- **Diagnostic Decomposition**: Splitting complex tasks into atomic tasks (Signal Classification vs. Pairwise Selection) allows for precise "capability anatomy."
- **Actionable Industry Insight**: The "easy generation, hard evaluation" finding suggests practitioners can use LLMs as generators but should rely on backtesting or specialized SFT for evaluation.
- **Cost/Quality Dual Axis**: Including token usage and reliability makes the benchmark practical for engineering trade-offs.

## Limitations & Future Work
- **Ceiling of Evaluation Metrics**: Zero-shot IC/Sharpe prediction is inherently difficult; it remains unclear how much of the poor performance is due to task difficulty versus model capability.
- **Market Scope**: Primarily focuses on CSI300. Wider market structures (e.g., crypto, commodities) need validation.
- **Search Paradigms**: Current benchmarks focus on classic methods (CoE/ToT/EA); future work could explore multi-agent collaborative search.

## Related Work & Insights
- **Comparison with ML Mining**: Traditional symbolic regression or RL searchers are computationally heavy; this study positions where LLMs can effectively augment parts of that pipeline.
- **Comparison with Financial Benchmarks**: Unlike FinQA or Pixiu which focus on text/QA, AlphaBench treats FAFM as a program synthesis problem, bridging code generation and quantitative finance.

## Rating
- Novelty: ⭐⭐⭐⭐☆ First systematic benchmark for FAFM.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across models, tasks, and market regimes.
- Writing Quality: ⭐⭐⭐⭐☆ Clear definitions and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Provides direct guidance for deploying LLMs in quantitative finance.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors](simbench_benchmarking_the_ability_of_large_language_models_to_simulate_human_beh.md)
- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](../../ACL2025/llm_evaluation/ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](../../ACL2025/llm_evaluation/codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](../../ICML2026/llm_evaluation/politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)

</div>

<!-- RELATED:END -->
