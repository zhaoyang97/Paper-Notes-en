---
title: >-
  [Paper Note] VL-RouterBench: A Benchmark for Vision-Language Model Routing
description: >-
  [CVPR 2026][Multimodal VLM][model routing] This paper introduces VL-RouterBench, the first systematic routing benchmark for vision-language models, encompassing 14 datasets, 17 candidate models, and 519,180 sample-model pairs. It evaluates 10 routing methods and reveals a significant gap between the current best router and the ideal Oracle.
tags:
  - CVPR 2026
  - Multimodal VLM
  - model routing
  - VLM
  - benchmark
  - efficiency-quality tradeoff
  - multi-model selection
date: 2026-05-08
content_hash: ae5fb5785b686d75
---

# VL-RouterBench: A Benchmark for Vision-Language Model Routing

**Conference**: CVPR 2026
**arXiv**: [2512.23562](https://arxiv.org/abs/2512.23562)
**Code**: [https://github.com/VL-RouterBench](https://github.com/VL-RouterBench)
**Area**: Multimodal VLM
**Keywords**: model routing, VLM, benchmark, efficiency-quality tradeoff, multi-model selection

## TL;DR
This paper introduces VL-RouterBench, the first systematic routing benchmark for vision-language models, encompassing 14 datasets, 17 candidate models, and 519,180 sample-model pairs. It evaluates 10 routing methods and reveals a significant gap between the current best router and the ideal Oracle.

## Background & Motivation
**Background**: Multi-model routing has evolved from an engineering optimization into a critical infrastructure component. Different VLMs vary substantially in inference cost and capability, and no single model can simultaneously ensure performance and efficiency across all query types. Routing research in the LLM domain has matured (RouterBench, RouterEval, RouterArena, etc.), yet a systematic benchmark for VLM routing remains absent.

**Limitations of Prior Work**: VLM routing faces several unique challenges: (a) highly diverse task types (VQA, visual reasoning, chart OCR, etc.), each emphasizing different capabilities; (b) multimodal fusion mechanisms remain an open problem, with large variation across VLMs in modality interaction and semantic representation; (c) vision-modality-specific issues such as visual semantic density and cross-modal alignment.

**Key Challenge**: Existing LLM routing benchmarks focus on text-only routing and cannot be directly adapted to VLM scenarios — defining "what constitutes an optimal routing decision" for VLMs within a unified framework is substantially harder.

**Goal**: Construct a VLM-dedicated routing benchmark that provides a unified pipeline for data preparation, training, and evaluation, thereby promoting reproducibility and comparability in VLM routing research.

**Key Insight**: Build quality-cost matrices from raw VLM inference and scoring logs, and design an accuracy-cost-aware soft-label training strategy.

**Core Idea**: Establish the first VLM routing benchmark covering 30,540 samples × 17 models, providing a complete pipeline from data preparation through training to evaluation.

## Method

### Overall Architecture
The VL-RouterBench pipeline consists of three stages: (1) **Routing data preparation** — collecting inference logs from VLMEvalKit to construct quality matrix $Y$ and cost matrix $C$; (2) **Router training** — employing an accuracy-cost-aware soft-label strategy with support for both feature-level and end-to-end architectures; (3) **Routing evaluation** — multi-dimensional assessment using average accuracy, average cost, throughput, and a composite Rank Score.

### Key Designs

1. **Quality-Cost Matrix Construction**:

    - Function: Establishes correctness labels and inference costs for each sample-model pair.
    - Mechanism: Employs rule-based evaluation (multiple-choice/answer matching) to ensure consistency. The cost formula is $C_{i,j} = n_{i,j}^{in} \cdot c_j^{in} + n_{i,j}^{out} \cdot c_j^{out}$, derived from token statistics in actual inference logs and public pricing tables.
    - Design Motivation: Avoids bias introduced by subjective judgments and ensures fair, reproducible evaluation.

2. **Accuracy-Cost-Aware Soft-Label Strategy**:

    - Function: Explicitly controls the accuracy-cost tradeoff during training via a tunable parameter $\lambda$.
    - Mechanism: Models router training as a multi-objective optimization and derives analytic soft labels: $t_i^{(\lambda)}(j) = \frac{\mathbf{1}\{Y_{i,j}=1\} \cdot \exp(-\lambda \cdot C_{i,j})}{\sum_{j:Y_{i,j}=1} \exp(-\lambda \cdot C_{i,j})}$. When $\lambda=0$, only accuracy is considered; as $\lambda \to \infty$, strong preference is placed on low-cost models.
    - Design Motivation: More flexible than hard labels, allowing probability mass to be distributed among correct models according to cost.

3. **Rank Score Composite Evaluation**:

    - Function: Unifies accuracy and cost into a single comparable score.
    - Mechanism: Cost is log-normalized to $[0, 100]$, then combined with accuracy via harmonic mean: $S(\beta) = \frac{(1+\beta)\cdot\bar{A}\cdot C_{norm}}{\beta\cdot\bar{A}+C_{norm}}$.
    - Design Motivation: Accuracy and cost operate on different scales; normalization is required before cross-configuration comparison.

### Router Architectures
- **Feature-level routers**: Frozen text/visual encoders extract embeddings fed into lightweight classifiers (KNN/MLP/Linear, etc.).
- **End-to-end routers**: Methods such as RouterDC and VLC directly predict model selection from multimodal inputs.

## Key Experimental Results

### Main Results — Routing Method Comparison

| Router | Avg. Acc.↑ | Avg. Cost↓ | Rank Score↑ | Rank |
|--------|-----------|-----------|------------|------|
| Oracle | 95.60 | $0.37 | 93.68 | 0 |
| Strongest | 78.01 | $2.72 | - | - |
| RouterDC (1st) | - | - | Highest | 1 |
| VLC (2nd) | - | - | - | 2 |
| MLP (3rd) | - | - | - | 3 |

### Ablation Study — Modality Fusion

| Fusion Strategy | Description |
|----------------|-------------|
| Text features only | Sub-optimal; lacks visual discriminative signals |
| Visual features only | Weakest; lacks task instruction information |
| Normalized concatenation | Best; simple and effective |

### Key Findings
- **Significant routing gains**: Learned routing systems consistently achieve more stable accuracy than any single model, often at comparable or lower cost.
- **Multimodal features are effective**: Simple normalized concatenation of text and visual embeddings is sufficient to support highly competitive routers, consistently outperforming unimodal counterparts.
- **Gap from Oracle**: Even the best router exhibits a notable gap from the Oracle, indicating substantial room for improvement in exploiting visual cues and modeling textual structure.
- **Model coverage**: 17 candidate models ranging from 1B to 78B parameters, spanning two orders of magnitude.

## Highlights & Insights
- **First VLM routing benchmark**: Fills the gap in unified routing evaluation for VLMs; the pipeline is complete (data → training → evaluation) and highly extensible.
- **Mathematical elegance of the soft-label strategy**: Analytic soft labels are derived via Lagrangian optimization, offering theoretical guarantees and allowing continuous control of the accuracy-cost tradeoff through a single parameter $\lambda$.
- **Diagnostic value of the Oracle gap**: Clearly identifies improvement directions in finer visual cue exploitation and textual structure modeling.

## Limitations & Future Work
- Only single-image inputs are considered; multi-image and video VLM scenarios are not covered.
- Correctness evaluation relies solely on rule-based matching (multiple-choice/answer matching), excluding open-ended generation tasks.
- Cost estimation is based on token count × pricing table, without accounting for actual latency or throughput variation.
- The router introduces additional feature extraction and classification overhead at inference time, which may not be worthwhile for models with already low inference costs.
- Robustness of routers on out-of-distribution data is not explored.

## Related Work & Insights
- **vs. RouterBench/RouterEval**: These target LLM text routing; VL-RouterBench is the first benchmark for VLMs, adding evaluation of visual modality and cross-modal fusion.
- **vs. RouterArena**: RouterArena covers multi-dimensional metrics and automated leaderboards; VL-RouterBench adopts its Rank Score design and extends it to the multimodal setting.
- **GPT-5 built-in routing**: The industry has already incorporated routing as a unified interface feature, underscoring the practical value of routing research.

## Rating
- Novelty: ⭐⭐⭐⭐ First VLM routing benchmark, filling a research gap
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 datasets × 17 models × 10 routing methods, comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Well-structured, clear derivations
- Value: ⭐⭐⭐⭐ Direct practical value for efficient VLM deployment

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Adaptive Vision-Language Model Routing for Computer Use Agents](adaptive_visionlanguage_model_routing_for_computer.md)
- [\[CVPR 2026\] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)
- [\[CVPR 2026\] GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmarking_vision_language_models_for_multimodal_graph_learning.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)

<!-- RELATED:END -->
