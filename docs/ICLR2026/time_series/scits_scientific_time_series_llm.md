---
title: >-
  [Paper Note] SciTS: Scientific Time Series Understanding and Generation with LLMs
description: >-
  [ICLR 2026][Time Series][scientific time series] This paper proposes SciTS—a scientific time series benchmark spanning 12 scientific domains, 43 tasks, and 54K+ samples—and introduces the TimeOmni framework, which unifies understanding and generation tasks via multi-patch expert routing and an LLM backbone, achieving the best overall performance across the full benchmark.
tags:
  - ICLR 2026
  - Time Series
  - scientific time series
  - LLM
  - benchmark
  - unified multi-task model
  - patch expert
date: 2026-05-08
content_hash: 59bd24bd550d7a42
---

# SciTS: Scientific Time Series Understanding and Generation with LLMs

**Conference**: ICLR 2026
**arXiv**: [2510.03255](https://arxiv.org/abs/2510.03255)
**Code**: [https://github.com/OpenTSLab/TimeOmni](https://github.com/OpenTSLab/TimeOmni)
**Area**: Time Series
**Keywords**: scientific time series, LLM, benchmark, unified multi-task model, patch expert

## TL;DR

This paper proposes SciTS—a scientific time series benchmark spanning 12 scientific domains, 43 tasks, and 54K+ samples—and introduces the TimeOmni framework, which unifies understanding and generation tasks via multi-patch expert routing and an LLM backbone, achieving the best overall performance across the full benchmark.

## Background & Motivation

**Background**: The scientific reasoning capabilities of LLMs have attracted considerable attention, yet time series—a fundamental modality of scientific data—remain largely overlooked. Existing multimodal LLMs either encode numerical sequences as text (resulting in excessively long sequences) or convert them to images (sacrificing numerical precision), neither of which is adequate for comprehensive scientific time series understanding.

**Limitations of Prior Work**: Existing unified time series models typically focus on a single task type, such as forecasting or analysis. More critically, they are primarily trained and evaluated on periodic commercial data (weather, traffic, finance), leaving their effectiveness on aperiodic, highly heterogeneous scientific signals (gravitational waves, EEG, bioacoustics) largely unknown.

**Key Challenge**: Scientific time series exhibit extreme diversity—sampling frequencies range from daily to MHz, lengths from a few points to millions, dimensionality from univariate to 58 channels, and tasks from classification to synthesis. Existing models and benchmarks cannot accommodate this diversity.

**Goal**: (1) Construct the most comprehensive benchmark for scientific time series; (2) Systematically evaluate 17 SOTA models on scientific time series tasks; (3) Propose TimeOmni as a working example to explore the key ingredients for LLMs to handle scientific time series.

**Key Insight**: Scientific time series (astronomical light curves, seismic waveforms, EEG, etc.) are fundamentally different from those in commercial domains, necessitating dedicated benchmarks and methods. General-purpose LLMs may generalize more effectively than specialized time series models.

**Core Idea**: Build the large-scale SciTS benchmark for comprehensive evaluation, and propose TimeOmni as an exploratory solution—employing multi-patch expert routing to adaptively select patch sizes for signals of varying scales, while unifying understanding and generation tasks under a single framework.

## Method

### Overall Architecture

TimeOmni takes a time series signal $\mathbf{X} \in \mathbb{R}^{T' \times N}$ and a task prompt as input. The time series is first flattened along the temporal dimension, then encoded by a Time Series Encoder (Router + Patch Expert + Patch Reprogramming) into $\mathbf{X}_{enc} \in \mathbb{R}^{T_{enc} \times D_{llm}}$ (where $T_{enc}$ is typically 100–200). The prompt is encoded via a text tokenizer. Both are concatenated and fed into a pretrained LLM. Understanding tasks produce text via a softmax output head; generation tasks produce time series via a linear regression head.

### Key Designs

1. **Router + Patch Expert Family**:

    - Function: Adaptively selects an appropriate patch size for scientific signals of varying lengths and resolutions.
    - Mechanism: Given a flattened input of length $T = NT'$, the Router selects patch size $D_{patch}$ such that the encoded sequence length falls within 100–200, i.e., $T/200 < D_{patch} < T/100$. The Patch Expert reshapes the input to $\mathbb{R}^{\lceil T/D_{patch} \rceil \times D_{patch}}$ and maps it to a unified dimension $D_{enc}$ via 1D convolution. Different patch sizes correspond to different Patch Experts.
    - Design Motivation: Scientific signal lengths span $10^0$ to $10^7$; a fixed patch size cannot accommodate this range. The multi-patch expert design ensures the number of encoded tokens remains manageable (100–200), neither exceeding LLM context limits nor discarding essential information.

2. **Patch Reprogramming**:

    - Function: Reprojects time series embeddings into the LLM's vocabulary space.
    - Mechanism: The LLM vocabulary embeddings $\mathbf{E} \in \mathbb{R}^{vocab\_size \times D_{llm}}$ are first linearly projected to $\mathbb{R}^{1000 \times D_{llm}}$. Multi-head cross-attention is then applied with $\mathbf{X}_{patch}$ as query and $\mathbf{E}$ as key/value, followed by a linear projection to produce the encoded output. This effectively re-represents time series using the LLM's semantic space.
    - Design Motivation: Directly feeding time series embeddings into an LLM creates a modality gap. Cross-attention over LLM vocabulary embeddings implicitly aligns the representational spaces of time series and language.

3. **Dual Output Heads + Prompt Ordering Strategy**:

    - Function: Unifies understanding (text output) and generation (time series output) tasks.
    - Mechanism: Understanding tasks adopt a Prompt-as-suffix strategy (signal first, prompt second), with text tokens generated via softmax. Generation tasks adopt a Prompt-as-prefix strategy (prompt first, signal second), with outputs mapped to the target length via a flatten-and-linear layer. Multiple regression heads are predefined to cover different output lengths, and the model automatically selects the closest match.
    - Design Motivation: Different task types depend on input information differently—understanding requires observing data before the question, while generation requires understanding the task requirements before processing the data.

### Loss & Training

TimeOmni is initialized from Qwen3-8B and fine-tuned with DoRA. Understanding tasks use the standard cross-entropy loss for language modeling; generation tasks use a regression loss. Training data consists of the 54K+ samples from SciTS.

## Key Experimental Results

### Main Results

| Model Category | Representative Model | Understanding AvgRk | Generation AvgRk | Task Coverage | Success Rate |
|---|---|---|---|---|---|
| Text LLM | GPT-4.1-mini | 6.1 | 6.7 | ~90% | Moderate |
| MLLM | Gemini2.5-Flash | 5.8 | — | ~95% | Moderate |
| Time Series Model | UniTS | 7.9 | — | ~30% | High (on supported tasks) |
| TimeOmni | Qwen3-8B base | **1.9** | **1.4** | **100%** | **100%** |

### Ablation Study

| Analysis Dimension | Key Findings |
|---|---|
| Text vs. image input to LLM | Image input generally outperforms text input on understanding tasks (better compression of long sequences) |
| General LLM vs. specialized time series model | LLMs generalize more effectively to unseen scientific domains |
| Open-source vs. closed-source LLM | Closed-source models achieve higher task coverage and success rates |
| Multi-patch expert vs. fixed patch | Multi-expert routing is critical for signals of varying scales |

### Key Findings

- SciTS is highly challenging: even the strongest closed-source LLMs achieve low F1 scores (<15%) on domains such as astronomy and neuroscience, with bioacoustics and radar results falling below 10%.
- General-purpose LLMs generalize better than specialized time series models on unseen scientific domains—specialized models achieve high success rates on supported tasks but cover an extremely narrow range of tasks.
- TimeOmni is the only model to achieve 100% task coverage and 100% instance success rate, demonstrating the advantage of explicit temporal modeling combined with an LLM backbone.

## Highlights & Insights

- **The SciTS benchmark** is itself a significant contribution—spanning 12 scientific domains, 7 task types, and multiple orders of magnitude in frequency, length, and dimensionality, it fills a critical gap in scientific time series evaluation.
- **The multi-patch expert routing** design is elegant and effective: by constraining the number of encoded tokens to 100–200, it gracefully addresses the challenge of extreme length variation in scientific signals.
- **The finding that general LLMs generalize better than specialized models** is noteworthy—suggesting that in data-scarce scientific domains, the pretrained knowledge encoded in LLMs is more valuable than specialized architectures.

## Limitations & Future Work

- TimeOmni requires fine-tuning on SciTS; zero-shot generalization to scientific domains not covered by SciTS remains unverified.
- Flattening multivariate signals into a single dimension may discard inter-channel correlation information.
- Predefined fixed regression head sets for generation tasks limit flexibility.
- The proportion and quality of synthetic data in the benchmark may affect the representativeness of evaluations.

## Related Work & Insights

- **vs. UniTS (Gao et al., 2024)**: UniTS also unifies question answering and forecasting, but its architecture is standalone and incompatible with LLM training; TimeOmni can be directly embedded within a general-purpose LLM.
- **vs. Time-MoE (Shi et al., 2025)**: Time-MoE employs MoE for time series forecasting but supports only forecasting tasks; TimeOmni covers 7 task types.
- **vs. SFE (Zhou et al., 2025)**: SFE uses an image-format scientific benchmark, sacrificing numerical precision; SciTS retains complete information by using the raw time series format.

## Rating

- Novelty: ⭐⭐⭐⭐ The benchmark contribution is outstanding; the TimeOmni methodology is somewhat incremental in its combinations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across 17 models, 43 tasks, and 12 domains.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough benchmark description.
- Value: ⭐⭐⭐⭐⭐ The SciTS benchmark holds long-term value for the scientific AI community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)

<!-- RELATED:END -->
