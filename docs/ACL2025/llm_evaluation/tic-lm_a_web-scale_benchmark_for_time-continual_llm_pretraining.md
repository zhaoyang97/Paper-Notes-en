---
title: >-
  [Paper Note] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining
description: >-
  [ACL 2025][LLM Evaluation][Continual Pretraining] This paper proposes TiC-LM, a large-scale temporal continual learning benchmark based on 114 months of Common Crawl data (2.9T tokens). Through over 150 experiments, the work systematically evaluates the performance of optimizers, data replay, and regularization methods in continual pretraining scenarios. The findings reveal that an autoregressive learning rate schedule combined with fixed-ratio data replay can closely approac…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Continual Pretraining"
  - "Temporal Distribution Shift"
  - "Data Replay"
  - "Common Crawl"
  - "Forgetting Problem"
date: 2026-05-08
content_hash: e715f1cfd3ab9587
---

# TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining

**Conference**: ACL 2025  
**arXiv**: [2504.02107](https://arxiv.org/abs/2504.02107)  
**Code**: [GitHub](https://github.com/apple/ml-tic-lm)  
**Area**: LLM Evaluation  
**Keywords**: Continual Pretraining, Temporal Distribution Shift, Data Replay, Common Crawl, Forgetting Problem

## TL;DR

This paper proposes TiC-LM, a large-scale temporal continual learning benchmark based on 114 months of Common Crawl data (2.9T tokens). Through over 150 experiments, the work systematically evaluates the performance of optimizers, data replay, and regularization methods in continual pretraining scenarios. The findings reveal that an autoregressive learning rate schedule combined with fixed-ratio data replay can closely approach the performance of training from scratch with 2.6 times less computational cost.

## Background & Motivation

**Background**: Large language models (LLMs) are typically trained from scratch on massive historical web data, but the training data suffers from knowledge cutoff issues, causing model performance on new data to decay over time. Meanwhile, retraining LLMs from scratch is computationally expensive.

**Limitations of Prior Work**: Existing research on continual language modeling has severe limitations in scale and scope: (1) Most previous works only train and evaluate on a single domain (e.g., Wikipedia, news, Twitter); (2) Although more recent large-scale studies are conducted on general web data, they do not focus on temporal distribution shifts, and the training epochs do not exceed 3; (3) There is a lack of systematic evaluations across multiple timesteps and domains.

**Key Challenge**: While practical LLM training utilizes general web data and requires robust performance across various tasks, existing benchmarks fail to simulate this scenario. A benchmark of sufficient scale and time span is required to genuinely investigate effective strategies for LLM continual pretraining.

**Goal**: (1) To determine whether continual pretraining can match periodic retraining from scratch at a lower cost; (2) To investigate whether forgetting is the core challenge when continually training on general web data; (3) To analyze whether the impact of forgetting varies across different domains.

**Key Insight**: Inspired by TiC-CLIP, this work constructs a web-scale benchmark based on the complete Common Crawl time series, utilizing monthly CC dumps as natural temporal distribution shift units.

**Core Idea**: By constructing a temporally-stratified web corpus benchmark spanning 114 months and 2.9T tokens, this work systematically investigates the trade-offs of different strategies in learning new knowledge versus retaining old knowledge during LLM continual pretraining.

## Method

### Overall Architecture

TiC-LM simulates a setup where monthly CC dumps are revealed sequentially. The LLM is first pretrained on the initial month's data and then continually updated every month within a fixed token budget (with optional replay of historical data). The evaluation covers temporally-stratified benchmarks on both general CC data and specific domains (Wikipedia, StackExchange, and CodeDocs).

### Key Designs

1. **TiC-CommonCrawl (TiC-CC) Dataset**:

    - Function: Provides large-scale, temporally-stratified training and evaluation data
    - Mechanism: 114 CC monthly dumps from May 2013 to July 2024 are collected and processed based on the DataComp-LM pipeline: resiliparse is used to extract plain text, heuristic filters from RefinedWeb are applied, and fuzzy deduplication is performed within each month (rather than across months). The classifier filter of DCLM-Baseline is excluded to maintain causality. The dataset totals 29T tokens, and a subset of 2.9T tokens is used for experiments.
    - Design Motivation: To maintain causality (avoiding processing past data with future data) and preserve natural temporal distribution shifts. This provides over 100 times more tokens and 10 times more timesteps compared to previous benchmarks.

2. **Multi-dimensional Temporally-Stratified Evaluation System**:

    - Function: Comprehensively evaluates the performance of continually trained models across different times and domains
    - Mechanism: Four categories of evaluation are designed: (1) TiC-CC: monthly held-out perplexity of general CC data (including Wiki and News subsets); (2) TiC-Wikipedia: a 10-year span evaluation based on full Wikipedia dumps, utilizing proper noun perplexity to capture factual changes; (3) TiC-StackExchange: QA answer perplexity across a span of 8 to 170 months; (4) TiC-CodeDocs: perplexity of 16 versions of official NumPy and PyTorch documentation. Additionally, 22 static downstream tasks from DCLM Core are included.
    - Design Motivation: To distinguish between In-Domain (ID), backward, and forward performance, capturing the trade-offs between forgetting and adapting to new data, recognizing that the need for replay may vary across different domains.

3. **Systematic Evaluation of Continual Learning Methods**:

    - Function: Compares three major categories of continual learning strategies under a unified framework
    - Mechanism: (1) **Optimizer methods**: Cyclic Cosine (decay within each month), Cyclic Cosine + AR (autoregressive decay of peak learning rate across epochs), Rsqrt (reciprocal square root infinite schedule), and Schedule-Free (iterative averaging optimizer); (2) **Data replay**: A proportion $\alpha_t$ of tokens is allocated to the current month, and the rest is distributed uniformly to historical months, testing both $\alpha_t=1/t$ (proportional) and $\alpha_t=1/2$ (fixed half); (3) **Regularization**: Learning without Forgetting (LwF, penalizing output differences via KL divergence) and Elastic Weight Consolidation (EWC, weighting parameter updates with the Fisher information matrix).
    - Design Motivation: To comprehensively cover the primary methodological paradigms in sequential/continual learning, facilitating a fair comparison under a unified large-scale setting.

### Loss & Training

The standard autoregressive language modeling loss (cross-entropy for next-token prediction) is utilized. Experiments are conducted at 1B and 3B parameter scales with training token volumes of 220B and 440B. The Oracle baseline is trained from scratch every two years (using a total of 1.16T tokens) and serves as the upper bound for continual training methods. A regret matrix (relative perplexity of each checkpoint on monthly evaluation data) is used to visualize the trade-off between ID and backward performance.

## Key Experimental Results

### Main Results

| Method (3B, 440B tokens) | TiC-CC Bwd↓ | TiC-CC ID↓ | TiC-CC-Wiki Bwd↓ | TiC-CC-News Bwd↓ |
|------------------------|-------------|------------|-------------------|-------------------|
| Cyclic Cosine | 0.082 | -0.011 | 0.029 | 0.071 |
| Cyclic Cosine + AR | 0.058 | -0.002 | 0.014 | 0.044 |
| Replay (α=1/2) | 0.007 | 0.007 | 0.010 | 0.005 |
| Replay (α=1/2) + AR | **-0.002** | 0.016 | **-0.001** | **-0.009** |
| Oracle Series (1.16T) | -0.003 | 0.035 | 0.004 | -0.008 |

*Values represent relative log-perplexity compared to Oracle-2024-07. Lower values are better, and negative values indicate performance exceeding the Oracle.*

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| α=1/t vs α=1/2 | 1/t has the lowest Bwd but worst ID | Equal-proportion replay yields too low a ratio of current-month data after 114 months |
| With or without AR schedule | AR consistently reduces Bwd, slightly increases ID | Decaying the learning rate across epochs helps mitigate forgetting |
| LwF / EWC | Minimal to no improvement | Regularization methods exhibit limited effectiveness in large-scale continual training on web data |
| 1B vs 3B | Consistent trends | Larger models benefit slightly more significantly |
| 220B vs 440B | 440B is overall superior | Larger token budgets improve all configurations |

### Key Findings

- **Replay (α=1/2) + AR can approach the performance of the Oracle series with 2.6 times less computation**, serving as the most balanced choice among all continual training methods.
- Forgetting is indeed a severe challenge on general CC data: methods without replay suffer a significant drop in performance on historical monthly data.
- The optimal replay ratio varies across domains: general web data requires substantial replay to prevent forgetting, but in rapidly evolving domains like StackOverflow and PyTorch, replaying old data can instead degrade performance.
- Regularization methods (e.g., LwF/EWC) are nearly ineffective in large-scale web data continual training, contrasting sharply with their effectiveness in small-scale continual learning settings.
- Irrelevant passages ranked higher by stronger retrievers are more disruptive.

## Highlights & Insights

- Unprecedented scale: 2.9T tokens across 114 timesteps, which is more than 100 times larger than previous continual learning benchmarks.
- Domain dependency discovery: The impact of forgetting and the requirement for replay are highly dependent on the evolution speed of the domain, offering crucial guidance for practical deployment.
- Comprehensive and systematic: Over 150 experiments covering diverse methods, model scales, and evaluation dimensions.
- Causal data design: Strictly ensures that the processing of each month's data only uses data from that month or before, preventing information leakage.

## Limitations & Future Work

- Continual learning effects after instruction tuning remain unexplored.
- English-only data is used, without considering multilingual scenarios.
- Experiments are only conducted at 1B and 3B scales; continual training strategies for larger models (7B+) might differ.
- Costs of data expiration and deletion (legal and privacy regulations) are not considered.
- The impact of cross-month deduplication is only preliminarily explored, which could affect the accuracy of forgetting evaluations.

## Related Work & Insights

- TiC-CLIP (Garg et al., 2024) is a similar work in temporal continual learning within the vision domain; this paper extends its concepts to LLM pretraining.
- DataComp-LM (Li et al., 2024a) provides the foundation for the data processing pipeline.
- Unlike works on web data continual training like Gupta et al. (2023) and Ibrahim et al. (2024), this study focuses on temporal distribution shifts and long-term continual updates.
- Confirming the effectiveness of replay strategies provides a direct reference for practical LLM update deployments (e.g., monthly/quarterly updates).

## Rating

- **Novelty**: 8/10 — The benchmark design fills an important gap, but the methods evaluated are pre-existing.
- **Technical Depth**: 7/10 — Primarily focused on systematic evaluation with limited methodological innovation.
- **Experimental Thoroughness**: 9/10 — Over 150 experiments, providing a comprehensive comparison across multiple scales, domains, and methods.
- **Writing Quality**: 8/10 — Clear structure with highly informative figure and table designs.
- **Value**: 9/10 — Offers direct guidance for the engineering practices of continual LLM updates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] WebWalker: Benchmarking LLMs in Web Traversal](webwalker_benchmarking_llms_in_web_traversal.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ICML 2025\] DataDecide: How to Predict Best Pretraining Data with Small Experiments](../../ICML2025/llm_evaluation/datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [\[ACL 2025\] Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph](benchmarking_uncertainty_quantification_methods_for_large_language_models_with_l.md)

</div>

<!-- RELATED:END -->
