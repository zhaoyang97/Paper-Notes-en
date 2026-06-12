---
title: >-
  [Paper Note] TimeOmni-VL: Unified Models for Time Series Understanding and Generation
description: >-
  [ICML 2026][Time Series][Time Series Forecasting] TimeOmni-VL achieves both time series **understanding and generation** within a unified multimodal framework by converting time series into **high-fidelity images** (Bi-T…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Forecasting"
  - "Time Series Imputation"
  - "Unified Multimodal"
  - "Visual Representation"
  - "Understanding-Generation"
date: 2026-05-08
content_hash: 849fc5a93dd994ab
---

# TimeOmni-VL: Unified Models for Time Series Understanding and Generation

**Conference**: ICML 2026  
**arXiv**: [2602.17149](https://arxiv.org/abs/2602.17149)  
**Code**: TBD  
**Area**: Time Series / Unified Multimodal Models  
**Keywords**: Time Series Forecasting, Time Series Imputation, Unified Multimodal, Visual Representation, Understanding-Generation

## TL;DR
TimeOmni-VL achieves both time series **understanding and generation** within a unified multimodal framework by converting time series into **high-fidelity images** (Bi-TSI) and introducing an **understanding-guided generation mechanism** (CoT as diffusion conditioning). It reaches state-of-the-art performance in both forecasting and imputation.

## Background & Motivation

**Background**: Time series modeling has long been divided into two branches: **generative models** (Time Series Foundation Models, TSFM), which pursue numerical accuracy in forecasting and imputation, and **understanding models**, which leverage LLMs to provide human-readable explanations of temporal dynamics. These paths have largely operated in isolation.

**Limitations of Prior Work**: Generative models often lack structural understanding, relying on shallow pattern matching. Understanding models struggle with numerical fidelity—text tokenizers split "123" into "1", "2", and "3", destroying numerical continuity. Vision-based methods like VisionTS are competitive but essentially rely on pixel texture matching, ignoring intrinsic properties like periodicity and seasonality. Pure-text LLMs are limited by counting abilities and cannot reliably generate sequences with hundreds or thousands of steps.

**Key Challenge**: While the vision domain has integrated understanding and generation through Unified Multimodal Models (UMM) based on the insight that "strong understanding is the foundation of high-quality generation," this paradigm remains unexplored in time series. The primary obstacles are the lack of **high-fidelity bidirectional mapping** and **mechanisms for understanding-guided generation**.

**Goal**: To achieve integrated time series understanding and generation within a unified visual framework, maintaining numerical precision while learning semantic features.

**Key Insight**: Inspired by the success of visual foundation models, can time series be represented visually to allow UMMs to natively support both understanding and generation?

**Core Idea**: Encode time series into high-fidelity images (Bi-TSI) and explicitly constrain the generation process using Chain-of-Thought (CoT) from understanding tasks, transforming temporal semantics into control signals for generation.

## Method

### Overall Architecture
A three-layer architecture:
1.  **Conversion Layer**: Transforms sequences into TS-images ($896 \times 896$) via Bi-TSI.
2.  **Understanding Layer**: Based on the understanding branch of Bagel-7B UMM, generating CoT for six types of tasks from the TS-image.
3.  **Generation Layer**: A diffusion decoding module conditioned on the CoT, outputting target TS-images which are then inversely mapped back to numerical sequences.

The key innovation lies in the **interconnection between understanding and generation**, where the understanding CoT directly serves as a conditional variable for the diffusion process during generation.

### Key Designs

1.  **Bi-TSI (High-fidelity Bidirectional Mapping)**:
    - **Function**: Achieves near-lossless round-trip conversion between time series and images.
    - **Mechanism**: Unlike the simple texture rendering in VisionTS, Bi-TSI explicitly manages **periodic grid folding**. Given a time series $X \in \mathbb{R}^{T \times N}$ and periodicity $f$, each variable is folded into an $f \times N_p$ periodic grid ($N_p = T/f$). Variables are rendered into vertical strip areas and stacked vertically. Mapping constraints $H/N \geq f$ and $W \geq L/f$ are maintained to prevent implicit downsampling. Robust Fidelity Normalization (RFN) is introduced: $\sigma = \alpha \frac{\text{Median}(|X - \mu|)}{c_{\text{MAD}}} + (1 - \alpha) \text{Std}(X)$, followed by a bounded mapping $X_{\text{norm}} = \tanh\left(\frac{X - \mu}{\kappa \sigma}\right)$ to balance outliers and details.
    - **Design Motivation**: Std scaling is sensitive to spikes, while MAD scaling fails on flat segments—RFN combines their strengths. The high $896 \times 896$ resolution (16x the area of VisionTS++'s $224 \times 224$) prevents information loss from downsampling.

2.  **Understanding-Guided Generation**:
    - **Function**: Explicitly transforms temporal semantics into control signals for generation via CoT.
    - **Mechanism**: For a generation task (prediction length $P$), the model first processes an understanding instruction to produce a CoT $R = (r_1, \ldots, r_K)$ (e.g., "upward trend at step $t$", "seasonal cycle of 7 days"). $R$ is then used as a condition for the diffusion module's iterative denoising. The joint objective is $\mathcal{L} = \lambda_{\text{und}} \mathcal{L}_{\text{und}} + \lambda_{\text{gen}} \mathcal{L}_{\text{gen}}$, where the understanding loss is text token prediction and the generation loss is diffusion MSE.
    - **Design Motivation**: Pure generative models might pursue local numerical fitting while ignoring global trends. Integrating CoT allows the model to learn signals like "this period should continue to decline," improving quality by 8.2% in experiments.

3.  **TSUMM-Suite Dataset**:
    - **Function**: Constructs a paired dataset of six understanding tasks and two generation tasks.
    - **Mechanism**: A "generation-first" pipeline: define 40k forecasting and 40k imputation samples, then derive 9,409 understanding QA pairs from the same instances. Understanding tasks are split into two levels: **Layout-level** (variable localization, cycle recognition) and **Signal-level** (intra/inter-cycle pattern comparison, anomaly detection). Detailed CoT is generated via rules and LLMs.
    - **Design Motivation**: General VLMs (like Gemini-2.5-Flash) have near-zero accuracy on signal-level tasks for TS-images. Multi-level tasks force the model to evolve from "locating variables" to "recognizing non-linear dynamics."

## Key Experimental Results

### Main Results

| Task | Method | Short-term | Mid-term | Long-term |
|------|------|------|------|------|
| **Forecasting** | Gemini-2.5-Flash | 1.295 | 1.201 | 1.279 |
| | VisionTS++ | 0.915 | 0.682 | 0.690 |
| | **Ours (TimeOmni-VL)** | **0.878** | **0.816** | **0.784** |
| **Imputation** | Moment-large | 1.220 | 1.400 | 1.630 |
| | Bagel (zero-shot) | 17.411 | 12.239 | 11.849 |
| | **Ours (TimeOmni-VL)** | **0.713** | **0.757** | **0.842** |

### Ablation Study

| Configuration | Forecasting nMASE | Imputation nMASE | Note |
|------|---------|---------|------|
| No Understanding CoT | +8.2% degradation | +8.2% degradation | CoT conditioning significantly impacts performance |
| Heatmap instead of Bi-TSI | > 1.0 | > 1.0 | TS2I strategy is crucial for generation |
| No RFM | Signal saturation | — | Robust normalization is necessary |

### Key Findings
- **Effectiveness of Understanding Tasks**: The base Bagel-7B had 0% accuracy on layout-level QA1-QA4; after TimeOmni-VL fine-tuning, it reflects near 1.0 accuracy.
- **8.2% Consistency Gain**: Disabling CoT results in a consistent 8.2% nMASE drop in generation quality.
- **Criticality of TS2I Design**: Replacing Bi-TSI with Heatmaps leads to nMASE > 1.0.
- **Long-term Forecasting Breakthrough**: Unlike text-only models (e.g., Time-R1 failing at 480+ steps), TimeOmni-VL maintains 0.784 nMASE even at a 900-step horizon.

## Highlights & Insights
- **Conceptual Innovation**: First systematic migration of the visual "understanding as a control signal" paradigm to time series, breaking the rigid boundary between generation and understanding.
- **Robust Engineering**: Bi-TSI's periodic folding elegantly solves three often-neglected issues: extreme value sensitivity (via RFN), implicit downsampling (bi-directional constraints), and high-resolution requirements ($896 \times 896$).
- **Dataset Value**: The "Layout → Signal" progression in TSUMM-Suite design forces the model to move from surface recognition to deep understanding.
- **Practical Breakthrough**: SOTA performance in imputation (0.713 nMASE) indicates readiness for production scenarios involving missing value completion.

## Limitations & Future Work
- LLM counting capabilities still exhibit occasional failures at 500+ steps.
- Performance excels on GIFT-Eval, but generalization to entirely different domains (high-frequency finance, geophysical signals) remains untested.
- $896 \times 896$ images generate 3,000 visual tokens, resulting in inference latency several times higher than pure LLMs.
- Dependency on clear periodicity; limited adaptation to non-periodic or irregular data.
- **Future Improvements**: Integrate RL to correct counting biases; explore dynamic TS-image resolution; add adaptive period detection for non-stationary sequences.

## Related Work & Insights
- **vs VisionTS / VisionTS++**: Both use images, but TimeOmni-VL provides better fidelity through RFN and capacity constraints, and introduces explicit understanding task constraints.
- **vs Time-LLM / Time-R1**: Text-based methods are bottlenecked by tokenization destroying continuity; TimeOmni-VL bypasses this with pixel-level representation.
- **vs Chronos-2 / MOMENT**: Specialized models utilize statistics and regression but lack semantic understanding; TimeOmni-VL unifies both, bringing "multi-task collaborative learning" to the LLM era.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (First systematic implementation of the "understanding-guided generation" paradigm in time series).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Covers forecasting, imputation, understanding, and reasoning; deep ablation; 11k+ QA pairs).
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure and motivation; some constants in RFN lack detailed explanation).
- **Value**: ⭐⭐⭐⭐⭐ (Introduces a unified multimodal paradigm to the time series community and validates the transferability of the VLM paradigm).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICML 2026\] CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)
- [\[ICML 2026\] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](../../ICLR2026/time_series/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICML 2026\] Interpretability in Deep Time Series Models Demands Semantic Alignment](interpretability_in_deep_time_series_models_demands_semantic_alignment.md)

</div>

<!-- RELATED:END -->
