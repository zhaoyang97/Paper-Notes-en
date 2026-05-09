---
title: >-
  [Paper Note] Chart Deep Research in LVLMs via Parallel Relative Policy Optimization
description: >-
  [ICLR2026][chart understanding] This paper proposes PRPO (Parallel Relative Policy Optimization), which addresses GRPO's training bottlenecks under multi-dimensional reward interference and heterogeneous data gradient conflicts through two-level parallel decoupled optimization — across reward dimensions and data types. It also introduces MCDR-Bench, which leverages an "error uniqueness principle" to transform subjective generation evaluation into objective error identification, enabling quantitative assessment of chart deep research capabilities.
tags:
  - ICLR2026
  - chart understanding
  - deep research
  - RLHF
  - policy optimization
  - benchmark
date: 2026-05-08
content_hash: 9a6de6a6533d36ec
---

# Chart Deep Research in LVLMs via Parallel Relative Policy Optimization

**Conference**: ICLR2026
**arXiv**: [2603.06677](https://arxiv.org/abs/2603.06677)
**Code**: To be confirmed
**Area**: Others
**Keywords**: chart understanding, deep research, RLHF, policy optimization, benchmark

## TL;DR
This paper proposes PRPO (Parallel Relative Policy Optimization), which addresses GRPO's training bottlenecks under multi-dimensional reward interference and heterogeneous data gradient conflicts through two-level parallel decoupled optimization — across reward dimensions and data types. It also introduces MCDR-Bench, which leverages an "error uniqueness principle" to transform subjective generation evaluation into objective error identification, enabling quantitative assessment of chart deep research capabilities.

## Background & Motivation

**Background**: Chart understanding has evolved from simple data extraction to reasoning and analysis. Existing methods (ChartQA, PlotQA, etc.) primarily handle shallow tasks — visual recognition and factual QA — while capabilities for genuine "deep research" (trend analysis, causal reasoning, strategic recommendations) remain severely underdeveloped.

**Limitations of Prior Work**: (a) **Training bottleneck** — Chart deep research requires simultaneous mastery of background knowledge integration, fact extraction, relation construction, deep reasoning, and predictive planning, yet GRPO compresses multi-dimensional rewards into a single scalar, causing signal interference and mutual cancellation; gradient conflicts from heterogeneous data allow simple tasks to dominate training. (b) **Evaluation bottleneck** — Existing benchmarks only assess factual QA and cannot evaluate end-to-end analytical reasoning; subjective generation tasks incur high annotation costs and exhibit large answer diversity.

**Key Challenge**: Tension between coordinated multi-dimensional capability development and single-objective optimization — GRPO aggregates all dimensional rewards into one scalar, compressing variance and weakening the discriminative power of optimization signals, preventing balanced development across dimensions.

**Goal**: (a) How to achieve balanced training under multi-dimensional rewards and heterogeneous data? (b) How to objectively evaluate chart deep research capabilities?

**Key Insight**: Introducing the concept of "parallelism" into policy optimization — parallel optimization across reward dimensions and data capability partitions — to decouple the sources of conflict. On the evaluation side, controlled error injection transforms subjective generation into objective classification.

**Core Idea**: Building on GRPO, PRPO introduces two-level parallel decoupling (Reward-PRPO decomposing reward dimensions + Data-PRPO partitioning data types) to eliminate signal interference and gradient conflicts in multi-dimensional training.

## Method

### Overall Architecture
PRPO is a unified framework combining Reward-PRPO and Data-PRPO. Given a chart and question as input, the model generates a deep analysis. During training: (1) Data-PRPO partitions training samples by capability dimension (visual understanding, logical reasoning, data analysis, etc.) and computes advantages independently within each partition; (2) Reward-PRPO separately computes advantages for each reward dimension (background knowledge, factual accuracy, relation construction, reasoning depth, prediction quality) within each partition and performs weighted optimization. For evaluation, MCDR-Bench transforms generation evaluation into error identification through a 5-stage annotation pipeline with controlled error injection.

### Key Designs

1. **Reward-PRPO (Reward Dimension Parallelism)**:

    - Function: Decomposes optimization along reward dimensions, computing advantages independently per dimension.
    - Mechanism: For $K$ reward dimensions, advantages are computed as $\hat{A}_i^{(k)} = (R_i^{(k)} - \bar{R}^{(k)}) / \sigma^{(k)}$, then combined with weighted aggregation: $J_{\text{Reward-PRPO}} = \sum_{k=1}^K \lambda_k \mathbb{E}[\cdots L_{\text{clip}}(r_{i,t}, \hat{A}_i^{(k)})]$
    - Design Motivation: GRPO compresses multi-dimensional rewards as $R_i = \sum_k R_i^{(k)}$, causing advantages in some dimensions to be cancelled by disadvantages in others. Reward-PRPO preserves independent optimization signals per dimension, allowing the model to learn each dimension separately.

2. **Data-PRPO (Data Type Parallelism)**:

    - Function: Partitions data by capability dimension, normalizing advantages independently within each partition.
    - Mechanism: A `capability_uid` assigns samples to $M$ capability partitions $\{P(Q^{(m)})\}_{m=1}^M$, with partition-level statistics for normalization: $\hat{A}_i^{(m)} = (R_i - \bar{R}^{(m)}) / \sigma^{(m)}$
    - Outlier Handling: Iteratively detects samples where $|\hat{A}_i^{(t)}| > \tau$ and demotes them to rollout-level individual optimization, preventing outliers from corrupting partition statistics.
    - Design Motivation: Reward distributions vary drastically across capability dimensions (simple visual recognition vs. complex causal reasoning). Global normalization allows high-variance simple tasks to dominate gradients; partition-level normalization ensures each capability type competes within its own scale.

3. **MCDR-Bench (Evaluation Framework)**:

    - Function: A benchmark for quantitative evaluation of chart deep research capabilities.
    - Construction Pipeline: Phase 1 — 5-stage multi-agent annotation (background acquisition → fact extraction → relation construction → deep research report → predictive planning) with human review; Phase 2 — controlled error injection based on the "error uniqueness principle," transforming subjective generation into objective error identification.
    - Scale: 1,021 high-complexity charts → 3,084 high-difficulty samples covering 5 capability dimensions.
    - Design Motivation: Subjective generation tasks are difficult to score objectively. By injecting a single known error into an otherwise correct report and asking the model to identify it, the task becomes objectively decidable and enables precise diagnosis of capability deficiencies per dimension.

### Loss & Training
The unified PRPO objective: for partition $m$ and reward dimension $k$, the advantage is $\hat{A}_i^{(k,m)} = (R_i^{(k)} - \bar{R}^{(k,m)}) / \sigma^{(k,m)}$, and the total objective is a two-level weighted sum: $J_{\text{PRPO}} = \sum_m \lambda_m \sum_k \lambda_k \mathbb{E}[\cdots L_{\text{clip}}(r_{i,t}, \hat{A}_i^{(k,m)})]$. The base model is Qwen2.5-VL-7B-Instruct.

## Key Experimental Results

### Main Results (MCDR-Bench)

| Model | BG | FE | RL | DR | F/P | Overall |
|------|-----|-----|-----|-----|-----|---------|
| GPT-4o | 27.2 | 21.9 | 41.0 | 47.5 | 60.0 | 35.8 |
| Claude-3.7 Sonnet | 68.8 | 57.3 | 89.5 | 85.0 | 87.0 | 75.0 |
| Gemini-2.5-Pro | 81.2 | 87.3 | 91.4 | 93.8 | 93.0 | **89.3** |
| Qwen2.5-VL-7B (base) | 23.4 | 39.4 | 51.0 | 37.6 | 45.6 | 40.0 |
| + GRPO | 41.2 | 51.7 | 75.4 | 66.1 | 77.4 | 61.7 |
| + PRPO | **50.7** | **61.4** | **81.8** | **72.8** | **84.0** | **69.6** |
| + PRPO Think | **62.9** | **65.2** | **88.9** | **80.9** | **87.2** | **76.3** |

### Ablation Study (ChartQAPRO Cross-Validation)

| Configuration | Factoid | MCQ | Conv. | FactChk | Hypo. | Overall |
|------|---------|-----|-------|---------|-------|---------|
| Qwen2.5-VL-7B base | 27.5 | 37.9 | 55.2 | 46.7 | 44.4 | 36.3 |
| + ChartReasoner-GRPO | - | - | - | - | - | 40.0 |
| + PRPO | **36.2** | **50.5** | 49.6 | **53.3** | **53.7** | **43.0** |

### Key Findings
- **PRPO comprehensively outperforms GRPO**: On MCDR-Bench, PRPO exceeds GRPO by +7.91% (direct) and +13.26% (Think), with consistent improvements across all 5 dimensions.
- **Think mode amplifies gains**: PRPO + Think further improves over direct PRPO by +6.64%, indicating that models trained with PRPO release greater potential under chain-of-thought reasoning.
- **7B model approaches proprietary large models**: PRPO Think's 76.3% surpasses Claude-3.7 Sonnet (75.0%) and approaches Gemini-2.5-Pro (gap of only 13 points), despite being 10–100× smaller.
- **Cross-benchmark generalization**: PRPO also outperforms GRPO by +6.64% on ChartQAPRO, ruling out overfitting to MCDR-Bench.
- **Largest gain in FE (Fact Extraction)**: From 39.4 → 61.4 (+22.0), indicating that PRPO's dimension-wise optimization most significantly benefits information extraction.

## Highlights & Insights
- **"Parallel decoupling" is a general strategy for multi-dimensional optimization conflicts**: Reward-PRPO decouples along reward dimensions; Data-PRPO decouples along data types. This design philosophy is transferable to any multi-objective RL scenario (e.g., correctness vs. efficiency vs. safety in code generation).
- **Error injection evaluation paradigm is elegant**: Transforming subjective generation into objective classification reduces annotation costs while enabling fine-grained diagnosis. This evaluation approach generalizes to any long-form generation task (e.g., RAG accuracy, report quality assessment).
- **Outlier demotion mechanism is practically useful**: Data-PRPO is not a rigid partition — samples detected as unsuitable for their current partition are automatically demoted to individual optimization, balancing partition efficiency with per-instance fairness.

## Limitations & Future Work
- **Single base model**: All experiments use Qwen2.5-VL-7B. Effectiveness on larger models (72B+) or different architectures remains unvalidated.
- **Manually defined capability partitions**: Data-PRPO's `capability_uid` requires predefined capability categories; automatic capability partition discovery is a promising direction for improvement.
- **Sensitivity of reward dimension weights $\lambda_k$**: The paper does not thoroughly discuss weight sensitivity. Adaptive adjustment of dimension weights (e.g., based on per-dimension convergence rates) may yield further gains.
- **Chart domain only**: Although PRPO's parallel optimization idea is general, experiments are limited to charts — validation in general multi-task VLM training is warranted.

## Related Work & Insights
- **vs. GRPO/DAPO**: GRPO uses group-level normalization but a single reward scalar. DAPO addresses entropy collapse but does not handle multi-dimensional conflicts. PRPO's key addition is "two-level parallelism" — across dimensions and data types.
- **vs. ChartReasoner**: ChartReasoner applies SFT+GRPO for structured reasoning. PRPO does not modify the reasoning structure, only the optimization strategy — making it more lightweight and general.
- **vs. PPO/DPO**: PPO requires an additional value model; DPO avoids the reward model but does not naturally accommodate multi-dimensional rewards. PRPO natively supports multiple dimensions within the GRPO framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual innovation of parallel decoupled training + error injection evaluation, though Reward-PRPO is essentially standard multi-objective optimization decomposition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation on MCDR-Bench + ChartQAPRO with comprehensive comparisons to proprietary and open-source models, but lacks experiments on additional base models.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear and mathematical derivations are rigorous, though Sections 3–4 are slightly redundant in structure.
- Value: ⭐⭐⭐⭐ PRPO's parallel optimization idea offers general reference value for multi-dimensional RLHF training; MCDR-Bench fills the evaluation gap for chart deep research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](../../AAAI2026/others/or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)
- [\[NeurIPS 2025\] Note 4: WebThinker — Empowering Reasoning Models with Deep Research Capabilities](../../NeurIPS2025/others/webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)
- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)
- [\[ICLR 2026\] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](enhancing_generative_auto_bidding.md)
- [\[ICLR 2026\] Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)

</div>

<!-- RELATED:END -->
