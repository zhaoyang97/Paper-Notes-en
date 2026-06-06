---
title: >-
  [Paper Note] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models
description: >-
  [ICML 2026][Autonomous Driving][Time series reasoning] TSRBench constructs a time series reasoning benchmark covering 14 domains, 4 major dimensions (Perception, Reasoning, Prediction, and Decision-making), 15 tasks…
tags:
  - "ICML 2026"
  - "Autonomous Driving"
  - "Time series reasoning"
  - "multi-modal evaluation"
  - "LLM/VLM/TSLLM"
  - "4 dimensions & 15 tasks"
  - "scaling law"
date: 2026-05-08
content_hash: fde37bcce26b72a3
---

# TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models

**Conference**: ICML 2026  
**arXiv**: [2601.18744](https://arxiv.org/abs/2601.18744)  
**Code**: https://tsrbench.github.io/ (Available)  
**Area**: Benchmark / Time Series Reasoning / Multi-modal VLM / Generalist Models  
**Keywords**: Time series reasoning, multi-modal evaluation, LLM/VLM/TSLLM, 4 dimensions & 15 tasks, scaling law

## TL;DR
TSRBench constructs a time series reasoning benchmark covering 14 domains, 4 major dimensions (Perception, Reasoning, Prediction, and Decision-making), 15 tasks, and 4125 questions. It supports four input modalities—text, visualization, interleaved textnd image, and embedding. Systematic evaluation of 30+ mainstream LLMs, VLMs, and TSLLMs reveals that "scaling still holds for perception and reasoning but fails for prediction," and while "textual and visual modalities are highly complementary, current models struggle to fuse them effectively."

## Background & Motivation

**Background**: Time series data are ubiquitous in high-risk sectors such as finance, healthcare, industry, and transportation. Reasoning about time series is considered a core capability for generalist models to solve real-world problems. Existing evaluations largely follow traditional time series analysis paradigms (forecasting, classification, anomaly detection, imputation), treating sequences as isolated numerical strings. Recent benchmarks like TimeMMD, CiK, TimeSeriesExam, MTBench, and TSR-SUITE have begun introducing context and LLMs/VLMs, but they often focus on a single dimension like prediction or cover only one or two categories of tasks in vertical domains.

**Limitations of Prior Work**: (1) Task dimensions are highly imbalanced, with most benchmarks covering only 1–2 capability dimensions (e.g., TimeMMD only tests prediction, TimeSeriesExam only tests perception); (2) Domain diversity is limited, allowing models to rely on domain-specific priors; (3) Modalities are singular, with almost no benchmark simultaneously supporting numerical strings, line charts, interleaved text-image, and time series embeddings; (4) There is a lack of quantitative answers to horizontal questions, such as whether semantic understanding and numerical prediction are decoupled, or if textual and visual modalities are complementary.

**Key Challenge**: While the capability evaluation of generalist LLMs/VLMs is rapidly expanding into mathematics, science, and embodied AI, time series—the modality most closely coupled with the physical world—has long been excluded from "general reasoning" evaluation frameworks. Existing time series benchmarks focus on task metrics rather than "dimensions of reasoning capability."

**Goal**: To construct a multi-domain, multi-modal time series reasoning benchmark that can stress-test four capability dimensions—Perception, Reasoning, Prediction, and Decision-making—with reliable ground truth (GT). The goal is to systematically compare LLM, VLM, and TSLLM families across different modalities and answer key questions regarding scaling, modality complementarity, and capability decoupling.

**Key Insight**: Explicitly decompose "time series reasoning" into 4 capability dimensions $\times$ 15 specific tasks. Questions are collected based on four principles: high text-time series alignment, domain diversity, verifiable GT, and the use of synthetic data where necessary to ensure precise numerical answers. Inputs are uniformly rendered as 100 PPI line charts for VLMs, and specialized models are evaluated across T, V, and T+V modalities to measure fusion capability.

**Core Idea**: By using a unified benchmark with comprehensive dimensions, modalities, and verifiable GT, the ambiguous question of "what capabilities generalist models lack in time series" is transformed into a quantifiable proposition that can be analyzed via radar charts, scaling curves, and modality ablations.

## Method

### Overall Architecture
TSRBench consists of three components: (1) Question Bank Construction: Collecting time series data from 14 real-world domains and synthetic scenarios (e.g., chaotic physical systems, algorithmic trading) to generate 4125 multiple-choice or open-ended questions across 4 dimensions and 15 task templates, totaling 15,250 channels. (2) Input Renderer: Converting the same time series into "pure numerical text" for LLMs, "100 PPI line charts (vertically stacked subplots for multivariate data, shared x-axis, shown grids, and sequence names)" for VLMs, and "projector embeddings" for TSLLMs. Proprietary models are also tested with combined T+V inputs. (3) Evaluation Protocol: All models use reasoning (CoT) and are evaluated using accuracy as the unified metric across 30+ models (6 proprietary, 13 LLM, 13 VLM, 4 TSLLM).

### Key Designs

1. **Capability Decomposition into Four Dimensions and Fifteen Tasks**:
    - **Function**: Decomposes "time series reasoning" into a 4$\times$15 grid: Perception (PR/NU/AD/CA), Reasoning (ER/CD/AR/TR/NR/DR/IR), Prediction (TSF/EP), and Decision-making (QualDM/QuantDM).
    - **Mechanism**: The perception layer identifies statistical properties (trend, seasonality, noise). The reasoning layer is subdivided into seven categories, including Explanation (ER), Causal Discovery (CD), Abduction (AR), etc., where Inductive Reasoning (IR) emphasizes abstracting rules before predicting events. The prediction layer converts numerical forecasting into multiple-choice to ease direct numerical output for generalists. The decision-making layer distinguishes between qualitative decisions and quantitative simulation comparisons.
    - **Design Motivation**: Previous benchmarks treated "reasoning" as a monolith, making it impossible to locate specific weaknesses. Subdividing it into 7 tasks allows for fine-grained conclusions, such as "GPT-5 is strong in AR/TR but weak in CD," providing clear directions for future model design.

2. **Unified Multi-modal Input and Modality Fusion Testing**:
    - **Function**: Ensures a "fair" evaluation across LLM, VLM, and TSLLM families and measures the complementarity of text and vision.
    - **Mechanism**: For each sequence, the system generates (a) numerical text, (b) rendered line charts (single variables in one plot, multivariate data in stacked subplots sharing x-axes, with grids and labels), and (c) embeddings. Resolution is fixed at 100 PPI after ablation. Proprietary models are tested on T, V, and T+V to directly measure the gain from adding visual information.
    - **Design Motivation**: Current claims that "VLMs solve everything via vision" or "LLMs only need numbers" lack controlled comparisons. TSRBench answers these questions by embedding three modalities into the same problem.

3. **Verifiable GT and Synthetic Data Support**:
    - **Function**: Guarantees that every one of the 4125 questions has a machine-scorable, unambiguous ground truth, especially for precise numerical reasoning.
    - **Mechanism**: Adheres to four principles: high text-TS alignment (context must be essential), domain diversity (14 domains to prevent over-fitting), verifiable GT (via high-fidelity Python simulations or direct extraction), and synthetic data for numerical reasoning (using chaotic systems to provide "noise-free exact answers").
    - **Design Motivation**: Real-world data often lacks precise GT for numerical reasoning tasks. Explicitly introducing synthetic data maintains complexity while providing a subset for reliable precision testing, reducing the risk of "incorrect answers due to ambiguous questions."

### Loss & Training
This work is a benchmark, not a training method; thus, no training loss is defined. Evaluation Protocol: All models use reasoning. Proprietary models are tested with T, V, and T+V inputs. Models like o4-mini and GPT-5 are tested at both low and high reasoning efforts. Accuracy is the unified metric for all 15 tasks, with ablations performed on model scale, chart resolution, tool use, and reasoning effort.

## Key Experimental Results

### Main Results

| Model | Input | Perception (PR) | Reasoning (TR) | Prediction (EP) | Decision (QualDM) | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5 | T | 75.7 | 68.8 | 79.7 | 31.9 | 55.5 |
| o4-mini | T | 73.1 | 34.4 | 73.3 | 30.4 | 47.7 |
| GPT-5-mini | T | 72.2 | 39.4 | 67.8 | 35.5 | 46.6 |
| DeepSeek-V3.2 | T | 67.7 | 19.4 | 47.2 | 33.1 | 39.1 |
| Qwen3-235B-A22B | T | 66.0 | 28.1 | 48.9 | 34.8 | 42.2 |
| GPT-OSS-120B | T | 66.8 | 31.3 | 59.7 | 33.7 | – |
| Qwen2.5-3B | T | 46.4 | 21.2 | 58.3 | 22.7 | 33.2 |

GPT-5 (T) leads with 55.5% overall, but still performs significantly worse on NR/DR/IR tasks requiring strict rule application compared to perception tasks. The smallest model, Qwen2.5-3B, scores only 33.2%, indicating significant room for scaling.

### Ablation Study

| Dimension | Key Findings | Description |
| :--- | :--- | :--- |
| Model Scale | Scaling holds for Perception/Reasoning | Both LLMs and VLMs show a stable positive correlation between scale and accuracy in these dimensions. |
| Model Scale | Scaling fails for Prediction | TSF tasks show almost no significant improvement across scales, creating a breakpoint compared to other dimensions. |
| Task Correlation | TSF correlation is weak | Strong reasoning does not equate to strong context-aware forecasting; semantic understanding and numerical prediction are decoupled. |
| Modality Fusion (T+V) | Complementarity vs. Fusion Failure | T and V solve different subsets of problems, but T+V rarely outperforms the stronger of the two in proprietary models. |
| Resolution | 100 PPI is the sweet spot | Higher resolutions sharply increase token cost, while lower resolutions lead to significant loss of detail. |

### Key Findings
- **Scaling holds for perception/reasoning but fails for prediction**: While accuracy improves with scale in reasoning, TSF performance remains flat, suggesting current pre-training does not truly improve numerical prediction.
- **TSF lacks correlation with other tasks**: A model's strength in reasoning does not predict its TSF performance, implying time series prediction should be treated as an independent capability for training and evaluation.
- **High complementarity but fusion failure in text and vision**: The sets of problems solved by T and V differ greatly, but T+V inputs do not create a "$1+1>2$" effect, exposing a lack of alignment mechanisms in multi-modal attention for dual-view signals.
- **Decision-making is a universal bottleneck**: All models score significantly lower in QualDM/QuantDM than in perception or reasoning, highlighting the gap between "understanding/reasoning" and "acting on time series."

## Highlights & Insights
- **Capability Grid Partitioning**: Decomposing "time series reasoning" into a 4$\times$15 grid is a major methodological upgrade over MTBench or TimeSeriesExam. It allows for natural radar charts, scaling curves, and modality ablations.
- **Multi-modal Control Design**: Using the same question with three input types (T/V/T+V) provides a clean setup to measure the utility of modality fusion. It provides quantitative evidence for the industry's suspicion that even proprietary VLMs have not mastered dual-view fusion.
- **Synthetic Data for Numerical Reasoning**: Filling the gap of missing GT in real-world data with controlled simulations (chaotic systems, etc.) enables stress testing for tasks like NR and DR. This approach is transferable to any benchmark emphasizing "numerical precision."

## Limitations & Future Work
- Using accuracy as the sole metric for TSF tasks (converted to MCQ) loses signals regarding "close but incorrect" answers. Future versions could introduce tiered scoring.
- Only line charts were tested for visualization; other representations like heatmaps, spectrograms, or polar plots are not yet covered.
- The decision dimension only evaluates "selecting the best" among discrete options, missing long-horizon sequential decisions or RL-style scenarios.
- Time series reasoning in multi-lingual contexts (e.g., non-English financial/medical reports) remains an unexplored dimension for context-TS alignment.

## Related Work & Insights
- **vs. TimeMMD / CiK**: These focus on context-aware forecasting (Prediction dimension), whereas this work extends analysis to 4 dimensions and 15 tasks.
- **vs. TimeSeriesExam**: While they use synthetic data for perception, this work extends the scope to reasoning, prediction, and decision-making.
- **vs. MTBench / SciTS / TSR-SUITE**: These cover only slices of the time series reasoning space (single modality, narrow domain, etc.). TSRBench provides the first complete evaluation matrix for generalist models via its 4$\times$15 grid and quad-modality inputs.

## Rating
- Novelty: ⭐⭐⭐⭐ The explicit decomposition of capability dimensions and multi-modal input design is a systematic contribution, though individual techniques are engineering integrations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 30+ models, 4 modalities, 15 tasks, and multiple reasoning effort levels, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear dimension breakdown and task definitions, though the density of sub-tasks can be overwhelming.
- Value: ⭐⭐⭐⭐⭐ Provides the first standardized matrix for positioning generalist models in time series reasoning. The findings on scaling failure and fusion bottlenecks offer direct guidance for future architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/autonomous_driving/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[AAAI 2026\] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](../../AAAI2026/autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](../../CVPR2026/autonomous_driving/towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[ICML 2026\] Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion](constrained_multi-objective_reinforcement_learning_with_max-min_criterion.md)

</div>

<!-- RELATED:END -->
