---
title: >-
  [Paper Note] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models
description: >-
  [ICML 2026][Time Series][Time Series Reasoning] TSRBench constructs a time series reasoning benchmark covering 14 domains, 4 major dimensions (perception/reasoning/prediction/decision-making), 15 tasks, 4125 questions…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Reasoning"
  - "Multimodal Benchmarking"
  - "LLM/VLM/TSLLM"
  - "4 Dimensions 15 Tasks"
  - "Scaling Law"
date: 2026-05-08
content_hash: bf6e258fdcd3f990
---

# TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models

**Conference**: ICML 2026  
**arXiv**: [2601.18744](https://arxiv.org/abs/2601.18744)  
**Code**: https://tsrbench.github.io/ (available)  
**Area**: Benchmarking / Time Series Reasoning / Multimodal VLM / Generalist Models  
**Keywords**: Time Series Reasoning, Multimodal Benchmarking, LLM/VLM/TSLLM, 4 Dimensions 15 Tasks, Scaling Law

## TL;DR
TSRBench constructs a time series reasoning benchmark covering 14 domains, 4 major dimensions (perception/reasoning/prediction/decision-making), 15 tasks, 4125 questions, and supports four input modalities: text, visualization, text+image, and embedding. It systematically evaluates 30+ mainstream LLMs, VLMs, and TSLLMs, revealing key findings such as "scaling holds for perception/reasoning but fails for prediction" and "text and visualization modalities are highly complementary, but current models can hardly fuse them."

## Background & Motivation

**Background**: Time series are ubiquitous in high-risk domains such as finance, healthcare, industry, and transportation. Reasoning over time series is considered a core capability for generalist models to solve real-world problems. Existing benchmarks largely follow traditional time series analysis paradigms (forecasting, classification, anomaly detection, imputation), treating sequences as isolated strings of numbers. Recent works like TimeMMD / CiK / TimeSeriesExam / MTBench / TSR-SUITE have begun introducing context and LLM/VLMs, but either focus on a single dimension (prediction) or only test one or two task types in a few vertical domains.

**Limitations of Prior Work**: (1) Task dimensions are highly imbalanced; most benchmarks cover only 1-2 capability dimensions (e.g., TimeMMD only tests prediction, TimeSeriesExam only tests perception); (2) Limited domain diversity, allowing models to exploit domain-specific priors; (3) Single modality—almost no benchmark supports all four input forms: "textual number string / line chart / mixed text-image / time series embedding"; (4) Lack of quantitative answers to generalist-level questions such as whether "semantic understanding vs. numerical prediction" are decoupled, or whether "text vs. vision" are complementary.

**Key Challenge**: Capability evaluation of generalist LLMs/VLMs is rapidly expanding into mathematics, science, and embodied scenarios, yet time series—the modality most tightly coupled to the real physical world—has long been excluded from "general reasoning" evaluation frameworks. Existing time series benchmarks focus only on task metrics, not on "reasoning capability dimensions."

**Goal**: To construct a time series reasoning benchmark that can stress-test the four major capability dimensions (perception/reasoning/prediction/decision-making), across multiple domains and modalities, with reliable ground truth. The benchmark systematically compares the performance of LLM, VLM, and TSLLM model families under different input modalities, addressing key questions on scaling, modality complementarity, and capability decoupling.

**Key Insight**: Explicitly decompose "time series reasoning" into 4 capability dimensions × 15 specific tasks, collecting questions based on four principles: high text-sequence alignment, domain diversity, verifiable ground truth, and use of synthetic data for precise numerical answers when necessary. On the input side, all sequences are rendered as 100 PPI line charts for VLMs, and proprietary models are evaluated with T, V, and T+V modalities to directly measure modality fusion capability.

**Core Idea**: Use a unified benchmark with complete capability and modality coverage and verifiable ground truth to break down the vague question of "what capabilities are generalist models lacking on time series" into quantifiable propositions that can be visualized as radar charts, scaling curves, and modality ablations.

## Method

### Overall Architecture
TSRBench consists of three components: (1) Question bank construction, collecting time series data from 14 real domains plus synthetic, verifiable scenarios (chaotic physical systems, algorithmic trading), then generating multiple-choice or open-ended questions based on 4 dimensions and 15 task templates, totaling 4125 questions and 15,250 channels; (2) Input renderer, converting each time series into "pure text number string" for LLMs, "100 PPI line chart (multivariate data vertically stacked, shared time axis, grid and series names shown)" for VLMs, and "projector embedding" for TSLLMs, with proprietary models additionally provided T+V joint input; (3) Evaluation protocol, where all models are set to reasoning mode, accuracy is used as the unified metric, and 6 proprietary + 13 LLM + 13 VLM + 4 TSLLM models (30+ in total) are evaluated.

### Key Designs

1. **Capability Decomposition: Four Dimensions, Fifteen Tasks**:

    - **Function**: Decomposes "time series reasoning" into a 4×15 grid: Perception (PR/NU/AD/CA, 4 tasks), Reasoning (ER/CD/AR/TR/NR/DR/IR, 7 tasks), Prediction (TSF/EP, 2 tasks), Decision-Making (QualDM/QuantDM, 2 tasks), with each task corresponding to an independent capability subspace.
    - **Mechanism**: The perception layer assesses recognition of underlying statistical properties such as trend, periodicity, stationarity, mean, noise, and anomalies. The reasoning layer is further divided into seven types: etiology (ER), causal discovery (CD), abduction (AR), temporal relation (TR), numerical reasoning (NR), deduction (DR), and induction (IR), with IR emphasizing "abstracting rules first, then predicting specific future events" rather than curve-fitting. The prediction layer reformulates numerical prediction as multiple-choice to lower the difficulty for generalist models to output numbers directly. The decision-making layer separates qualitative decisions from quantitative decisions that require "simulating and comparing multiple candidate rules."
    - **Design Motivation**: Previous benchmarks scored "reasoning" as a whole, making it impossible to pinpoint specific model weaknesses. By subdividing reasoning into seven sub-tasks, one can directly observe fine-grained conclusions such as "GPT-5 is strong in AR/TR/NR but still weak in CD," providing guidance for future model design.

2. **Unified Multimodal Input and Modality Fusion Testing**:

    - **Function**: Enables "fair" evaluation of the same time series data across LLM/VLM/TSLLM models and quantitatively measures whether "text + vision" are complementary.
    - **Mechanism**: For each sequence, simultaneously generates (a) textual number sequence, (b) code-rendered line chart (one chart per univariate, multivariate vertically stacked sharing x-axis, always with grid and series names), and (c) embedding. Resolution is fixed at 100 PPI (after ablation) to balance token cost and feature visibility. For proprietary models supporting multimodal input, T, V, and T+V experiments are conducted to directly compare "does adding the chart actually help."
    - **Design Motivation**: Existing claims that "VLMs can solve all time series problems by looking at charts" or "LLMs only need number strings" lack controlled comparisons under the same prompt and question. TSRBench embeds all three modalities into the same question, quantitatively answering the modality complementarity and fusion failure issues.

3. **Verifiable Ground Truth + Synthetic Data for Numerical Reasoning**:

    - **Function**: Ensures that each of the 4125 questions has "machine-gradable, unambiguous" ground truth, especially for precise numerical reasoning.
    - **Mechanism**: Four collection principles—high text-sequence alignment (context must be essential for reasoning, not decorative), domain diversity (14 domains to prevent overfitting), verifiable unambiguous GT (either generated directly via high-fidelity Python simulation or extracted from sequence/context), and synthetic data to supplement numerical reasoning (using chaotic physical systems and algorithmic trading backtests as controllable simulations, providing "noise-free precise answers" as stress tests for numerical reasoning and deductive logic).
    - **Design Motivation**: Real-world data, while complex, often lacks precise GT, making reliable evaluation of tasks requiring 0/1 grading (like numerical reasoning) difficult. Explicitly introducing synthetic data retains real data complexity while providing a subset for direct evaluation of numerical accuracy, reducing the risk that "model errors are due to ambiguous questions."

### Loss & Training
This work is a benchmark, not a training method, so there is no training loss. Evaluation protocol: all models are set to reasoning mode; proprietary models are evaluated with T / V / T+V inputs; o4-mini / GPT-5 / GPT-5-mini are evaluated under both low and high reasoning settings; accuracy is used as the unified metric for all 15 tasks, with ablations on model size, visualization resolution, tool usage, and reasoning effort.

## Key Experimental Results

### Main Results

| Model | Input | Perception (PR) | Reasoning (TR) | Prediction (EP) | Decision (QualDM) | Overall |
|------|------|------|------|------|------|------|
| GPT-5 | T | 75.7 | 68.8 | 79.7 | 31.9 | 55.5 |
| o4-mini | T | 73.1 | 34.4 | 73.3 | 30.4 | 47.7 |
| GPT-5-mini | T | 72.2 | 39.4 | 67.8 | 35.5 | 46.6 |
| DeepSeek-V3.2 | T | 67.7 | 19.4 | 47.2 | 33.1 | 39.1 |
| Qwen3-235B-A22B | T | 66.0 | 28.1 | 48.9 | 34.8 | 42.2 |
| GPT-OSS-120B | T | 66.8 | 31.3 | 59.7 | 33.7 | – |
| Qwen2.5-3B | T | 46.4 | 21.2 | 58.3 | 22.7 | 33.2 |

GPT-5 (T) leads with 55.5% overall, but still lags behind perception tasks in sub-tasks requiring strict rule application (NR / DR / IR). The smallest Qwen2.5-3B achieves only 33.2 overall, leaving significant scaling headroom.

### Ablation Study

| Dimension | Key Observation | Description |
|------|---------|------|
| Model Size | Scaling holds for Perception / Reasoning | Both LLMs and VLMs show stable positive correlation between model size and accuracy in perception and reasoning |
| Model Size | Scaling fails for Prediction | TSF task shows almost no improvement with model size, forming a breakpoint with the other three dimensions |
| Task Correlation | TSF weakly correlated with other tasks | Strong reasoning does not imply strong context-aware forecasting; semantic understanding and numerical prediction are decoupled |
| Modality Fusion (T+V) | Text and vision are complementary but fusion fails | For the same question, T and V solve different subsets; T+V usually does not outperform the stronger of the two in proprietary models |
| Visualization Resolution | 100 PPI is optimal | Higher resolution sharply increases token cost, lower resolution loses significant detail |

### Key Findings
- **Scaling holds for perception/reasoning but breaks for prediction**: Perception and reasoning accuracy increases smoothly with model size, but TSF task forms a "plateau curve," indicating that current LLM/VLM pretraining objectives and data do not truly improve numerical prediction ability.
- **TSF is weakly correlated with other tasks**: A model strong on all reasoning tasks does not predict its TSF performance, suggesting that time series forecasting should be trained and evaluated as an independent capability, not as a byproduct of general reasoning.
- **Text and vision are highly complementary but fusion fails**: T and V solve very different sets of questions; T+V input rarely yields a 1+1>2 effect in mainstream multimodal models, exposing a lack of alignment mechanisms for "two views of the same signal" in current multimodal attention.
- **Decision-Making is a universal weakness**: All models score significantly lower on QualDM / QuantDM than on perception and reasoning, reflecting a large gap between "understanding + reasoning" and "making decisions based on time series."

## Highlights & Insights
- **Grid-based capability decomposition**: Explicitly decomposing "time series reasoning" into a 4×15 capability grid is the biggest methodological upgrade over MTBench / TimeSeriesExam, enabling natural radar/scaling curves and modality ablations. Future work can directly target specific weak sub-tasks.
- **Multimodal contrastive design**: Unified rendering pipeline for three input types (T/V/T+V) per question provides the cleanest setup for measuring "is modality fusion actually useful," directly supporting the long-suspected but previously unquantified conclusion that "even proprietary VLMs have not learned to fuse dual views."
- **Synthetic data for numerical reasoning**: Using chaotic systems and algorithmic trading backtests as controllable simulations fills the gap where real data lacks precise GT, enabling NR / DR tasks to truly stress-test numerical reasoning. This approach is transferable to any benchmark emphasizing "numerical precision."

## Limitations & Future Work
- Evaluation uses accuracy as the sole metric; for TSF tasks that typically use MSE / MAPE, converting to multiple-choice loses the "almost correct" signal. Future work could introduce tiered scoring or correlation analysis with continuous metrics.
- Current VLM input only tests line charts; other visualizations such as heatmaps, spectrograms, and polar plots are not yet covered.
- The decision-making dimension only evaluates "choose the best among discrete options," not sequence decision-making or RL-style long-horizon decisions; the difficulty ceiling for QuantDM can be further raised.
- Time series reasoning in multilingual scenarios (e.g., non-English financial/medical reports + time series) is not included; cross-lingual context-sequence alignment is a promising future direction.

## Related Work & Insights
- **vs TimeMMD / CiK**: These focus on context-aware time series prediction, covering only the Prediction dimension; this work expands to 4 dimensions and 15 tasks.
- **vs TimeSeriesExam**: They use synthetic data but only test perception; this work also uses synthetic data for numerical reasoning but expands to reasoning, prediction, and decision-making.
- **vs MTBench / EngineMT-QA / SciTS / TimeMQA / TSR-SUITE**: These benchmarks each cover only a slice of the time series reasoning space (narrow domain, single modality, lacking decision-making). TSRBench, with its 4×15 capability grid, four modalities, and 30+ models, provides the first complete time series reasoning evaluation matrix for generalist models.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicit decomposition of capability dimensions and unified four-modality input design are rare systematic contributions in time series benchmarking, though individual techniques are mostly engineering integrations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 30+ models, 4 modalities, 15 tasks, multiple reasoning effort levels, and ablations on visualization resolution and tool augmentation.
- Writing Quality: ⭐⭐⭐⭐ Clear dimension breakdown, concrete task definitions, and quantitatively supported findings; however, the high density of sub-task definitions may be overwhelming on first read.
- Value: ⭐⭐⭐⭐⭐ Provides the first standardized matrix for positioning generalist models' capabilities in time series reasoning; the conclusions "scaling breaks for prediction" and "modality fusion fails" directly inform future foundation model and multimodal architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](../../ICLR2026/time_series/swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)
- [\[ICML 2026\] Adaptive Time Series Reasoning via Segment Selection](adaptive_time_series_reasoning_via_segment_selection.md)
- [\[ICCV 2025\] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models](../../ICCV2025/time_series/vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)
- [\[ICLR 2026\] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting](../../ICLR2026/time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)

</div>

<!-- RELATED:END -->
