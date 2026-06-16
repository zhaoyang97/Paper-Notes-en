---
title: >-
  [Paper Note] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models
description: >-
  [ICML 2026][Autonomous Driving][LLM/VLM/TSLLM] TSRBench constructs a time series reasoning benchmark covering 14 domains, 4 major dimensions (Perception/Reasoning/Prediction/Decision-making), 15 tasks, and 4,125 questions. It supports four input modalities (Text, Visual, Text+Image, Embedding) and systematically evaluates 30+ mainstream LLMs, VLMs, and TSLLMs. It r
tags:
  - ICML 2026
  - Autonomous Driving
  - LLM/VLM/TSLLM
  - scaling law
date: 2026-05-08
content_hash: 5ef1323ad39fe322
---
# TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models

**Conference**: ICML 2026  
**arXiv**: [2601.18744](https://arxiv.org/abs/2601.18744)  
**Code**: https://tsrbench.github.io/ (Available)  
**Area**: Evaluation Benchmark / Time Series Reasoning / Multimodal VLM / Generalist Models  
**Keywords**: Time series reasoning, multimodal evaluation, LLM/VLM/TSLLM, 4 dimensions 15 tasks, scaling law

## TL;DR
TSRBench constructs a time series reasoning benchmark covering 14 domains, 4 major dimensions (Perception/Reasoning/Prediction/Decision-making), 15 tasks, and 4,125 questions. It supports four input modalities (Text, Visual, Text+Image, Embedding) and systematically evaluates 30+ mainstream LLMs, VLMs, and TSLLMs. It reveals that "scaling holds in perception/reasoning but fails in prediction" and that "text and visual modalities are highly complementary, yet current models struggle to fuse them."

## Background & Motivation

**Background**: Time series are ubiquitous in high-risk domains such as finance, healthcare, industry, and transportation. Reasoning over time series is considered a core capability for generalist models to solve real-world problems. Existing evaluations largely follow traditional time series analysis paradigms (forecasting, classification, anomaly detection, imputation), treating sequences as isolated numerical strings. Recent works like TimeMMD / CiK / TimeSeriesExam / MTBench / TSR-SUITE have begun introducing context and LLM/VLMs, but they either focus on the prediction dimension or evaluate a limited set of tasks within specific vertical domains.

**Limitations of Prior Work**: (1) Imbalanced task dimensions, with most benchmarks covering only 1-2 capability dimensions (e.g., TimeMMD only tests prediction, TimeSeriesExam only tests perception); (2) Limited domain diversity, allowing models to exploit domain-specific priors; (3) Single modality, as few benchmarks simultaneously support "textual numerical strings / line charts / interleaved text-image / time series embeddings" as input formats; (4) Lack of quantitative answers to generalist-level questions such as whether "semantic understanding vs. numerical prediction" is decoupled, or if "text vs. vision" is complementary.

**Key Challenge**: While the capability evaluation of generalist LLM/VLMs is rapidly expanding into mathematics, science, and embodiment scenarios, time series—the modality most closely coupled with the physical world—has long been excluded from "general reasoning" evaluation frameworks. Existing time series benchmarks focus on task metrics rather than "reasoning capability dimensions."

**Goal**: To construct a time series reasoning benchmark that can pressure-test four capability dimensions (Perception / Reasoning / Prediction / Decision-making) across multiple domains and modalities with reliable ground truth. This benchmark aims to systematically compare LLM / VLM / TSLLM families under different modal inputs to answer key questions regarding scaling, modality complementarity, and capability decoupling.

**Key Insight**: Explicitly decompose "time series reasoning" into 4 capability dimensions × 15 specific tasks. Collect questions based on four principles: high text-series alignment, domain diversity, verifiable ground truth (GT), and the use of synthetic data to ensure precise numerical answers where necessary. On the input side, render sequences as 100 PPI line charts for VLMs and run T, V, and T+V modalities for proprietary models to directly measure modality fusion capabilities.

**Core Idea**: Use a unified benchmark with comprehensive capability dimensions, modalities, and verifiable GT to transform the vague question of "what capabilities do generalist models lack in time series" into quantifiable propositions through radar charts, scaling curves, and modality ablations.

## Method

### Overall Architecture
The core problem TSRBench aims to answer is "which specific capabilities do generalist models lack in time series." To achieve this, it moves beyond traditional task metrics like prediction or classification, mapping the proposition onto a "capability dimension × modality" grid. The pipeline consists of question bank construction, input rendering, and evaluation protocols: first, collecting questions from 14 real-world domains and verifiable synthetic scenarios, generating 4,125 questions with machine-verifiable GT across 15 templates; then, rendering the same sequence as text strings, line charts, and embeddings for LLM / VLM / TSLLM evaluation, with an additional text+image joint input for proprietary models; finally, all models are evaluated using reasoning traces and accuracy scores, enabling scaling curves, modality ablations, and capability radars across over 30 models.

### Key Designs

**1. A 4-Dimension 15-Task Capability Grid: Decomposing "Reasoning" to Locate Weaknesses**

Previous benchmarks treated "time series reasoning" as a single entity. TSRBench decomposes this into a two-dimensional grid. The vertical axis consists of 4 capability dimensions: Perception (identifying low-level statistical properties like trends/cycles/stationarity, including 4 tasks: PR/NU/AD/CA), Reasoning (7 tasks), Prediction (TSF/EP), and Decision-Making (QualDM/QuantDM). The reasoning dimension is further subdivided into Causality (CD), Abduction (AR), Temporal Relations (TR), Numerical Reasoning (NR), Deduction (DR), and Induction (IR). Notably, IR requires models to "abstract rules before predicting specific future events" rather than simple curve-fitting. Decision-making is split into qualitative and quantitative (requiring simulation with multiple rules). This allows conclusions to reach the sub-task level—for example, identifying if a model is strong in AR/TR but weak in CD.

**2. Unified Multimodal Input and Fusion Testing: Evaluating Three Views for the Same Question**

To provide a controlled comparison between "VLM sight" and "LLM reading," TSRBench generates three inputs for the same sequence: (a) raw numerical text sequences for LLMs; (b) code-rendered line charts for VLMs, featuring vertical sub-plot stacking for multivariate data and grid labels; (c) projector embeddings for TSLLMs. Line chart resolution is fixed at 100 PPI after ablation as the "sweet spot" between token cost and detail visibility. For proprietary multimodal models, T, V, and T+V experiments are conducted to quantitatively measure whether adding visual information actually improves performance.

**3. Verifiable GT + Synthetic Data Backstop: Enabling 0/1 Scoring for Numerical Reasoning**

Real-world data often lacks precise answers, making it difficult to reliably evaluate tasks like NR/DR that require strict scoring. TSRBench applies four principles: high text-series alignment (context must be necessary for reasoning), domain diversity (balanced across 14 domains), verifiable unambiguous GT, and synthetic data for numerical reasoning. For questions requiring precision, controlled simulations of chaotic physical systems or algorithmic trading backtests generate noise-free, exact answers. This retains the complexity of real data while providing a subset of questions that can directly verify numerical accuracy.

### Loss & Training
This is an evaluation benchmark, so there is no training loss. Evaluation protocol: All models use reasoning traces; proprietary models are tested on T / V / T+V inputs; o4-mini / GPT-5 / GPT-5-mini are reported with both low and high reasoning efforts; accuracy serves as the unified metric across all 15 tasks.

## Key Experimental Results

### Main Results

| Model | Input | Perception (PR) | Reasoning (TR) | Prediction (EP) | Decision (QualDM) | Overall |
|------|------|------|------|------|------|------|
| GPT-5 | T | 75.7 | 68.8 | 79.7 | 31.9 | 55.5 |
| o4-mini | T | 73.1 | 34.4 | 73.3 | 30.4 | 47.7 |
| GPT-5-mini | T | 72.2 | 39.4 | 67.8 | 35.5 | 46.6 |
| DeepSeek-V3.2 | T | 67.7 | 19.4 | 47.2 | 33.1 | 39.1 |
| Qwen3-235B-A22B | T | 66.0 | 28.1 | 48.9 | 34.8 | 42.2 |
| Qwen2.5-3B | T | 46.4 | 21.2 | 58.3 | 22.7 | 33.2 |

GPT-5 (T) leads with an overall score of 55.5%, but sub-tasks requiring strict rule application (NR/DR/IR) remain significantly lower than perception tasks. The smallest model, Qwen2.5-3B, scores 33.2%, leaving ample room for scaling.

### Ablation Study

| Dimension | Key Observation | Description |
|------|---------|------|
| Model Scale | Scaling holds for Perception / Reasoning | Both LLMs and VLMs show a stable positive correlation between scale and accuracy in these dimensions. |
| Model Scale | Scaling fails for Prediction | TSF tasks show negligible improvement across different scales, creating a "plateau" relative to other dimensions. |
| Task Correlation | TSF has weak correlation with other tasks | Strong reasoning does not equal strong context-aware forecasting; semantic understanding and numerical prediction are decoupled. |
| Modality Fusion (T+V) | Modalities are complementary but fusion fails | T and V solve different subsets of problems, but T+V rarely outperforms the better of the two in proprietary models. |
| Viz Resolution | 100 PPI is the sweet spot | Higher resolution increases token cost sharply; lower resolution leads to significant loss of detail. |

### Key Findings
- **Scaling holds for Perception/Reasoning but breaks for Prediction**: While performance in the first two dimensions rises smoothly with model scale, TSF tasks show a flat curve. This suggests that current LLM/VLM pre-training objectives and data have not truly improved numerical prediction capabilities.
- **TSF shows weak correlation with other tasks**: A model's strength in reasoning tasks does not predict its TSF performance. Time series forecasting should be treated as an independent capability family rather than a "byproduct" of general reasoning.
- **Text and Vision are highly complementary but fusion fails**: The sets of questions solved by T vs. V vary greatly. However, joint T+V input yields almost no "1+1>2" effect in mainstream multimodal models, exposing a lack of alignment mechanisms for "two views of the same signal."
- **Decision-Making is a universal weakness**: All models score significantly lower in QualDM / QuantDM than in perception and reasoning, reflecting a gap between "understanding + reasoning" and "executing decisions based on time series."

## Highlights & Insights
- **Capability Grid Mapping**: Explicitly decomposing "time series reasoning" into a 4×15 grid is a major methodological upgrade over MTBench or TimeSeriesExam. It allows for natural expansion into radar and scaling analysis.
- **Multimodal Comparative Design**: The unified pipeline for three inputs (T/V/T+V) for the same question provides the cleanest setting to measure "whether modality fusion actually works," revealing that even proprietary VLMs fail to fuse dual views effectively.
- **Synthetic Data for Numerical Reasoning**: Using chaotic systems effectively fills the gap where real data lacks precise GT. This ensures tasks requiring 0/1 accuracy (NR/DR) are effective pressure tests, a strategy transferable to any benchmark emphasizing numerical precision.

## Limitations & Future Work
- The evaluation uses only the accuracy metric. For TSF tasks typically evaluated with MSE/MAPE, converting them to multiple-choice format may lose signals regarding "near misses."
- Current VLM inputs only test line charts; the impact of other representations (heatmaps, spectrograms) remains unexplored.
- The decision-making dimension only evaluates "selecting the best" from discrete options, lacking coverage of long-horizon sequential decisions or reinforcement learning-style scenarios.
- Multilingual time series reasoning (e.g., non-English financial/medical reports paired with series) is not yet included; cross-lingual context-series alignment is a valuable future dimension.

## Related Work & Insights
- **vs. TimeMMD / CiK**: These focus on context-aware forecasting, covering only the Prediction dimension. This work expands to 4 dimensions and 15 tasks.
- **vs. TimeSeriesExam**: They use synthetic data for perception only. This work uses synthetic data for numerical reasoning but extends the scope to reasoning, prediction, and decision-making.
- **vs. MTBench / TSR-SUITE etc.**: These cover specific "slices" of the reasoning space. TSRBench provides the first complete evaluation matrix for generalist models via the 4×15 grid and multimodal comparison across 30+ models.

## Rating
- Novelty: ⭐⭐⭐⭐ The explicit capability grid and quad-modal input design are systematic contributions, though individual techniques are based on engineering integration.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 30+ models, 4 modalities, 15 tasks across multiple reasoning effort levels, with ablations on resolution and tool enhancements.
- Writing Quality: ⭐⭐⭐⭐ Clear dimension breakdown and task definitions, though the density of sub-task definitions can be high for initial reading.
- Value: ⭐⭐⭐⭐⭐ Provides the first standardized matrix for positioning generalist models in time series. The findings on scaling breakage and fusion failure have direct implications for future model and architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](../../AAAI2026/autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)
- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](../../CVPR2025/autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](../../CVPR2026/autonomous_driving/towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[ICML 2026\] Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion](constrained_multi-objective_reinforcement_learning_with_max-min_criterion.md)

</div>

<!-- RELATED:END -->
