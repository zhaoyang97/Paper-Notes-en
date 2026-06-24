---
title: >-
  [Paper Note] Context is Key: A Benchmark for Forecasting with Essential Textual Information
description: >-
  [ICML 2025][Multimodal VLM][Context-aided forecasting] This paper proposes the Context is Key (CiK) benchmark—consisting of 71 manually designed forecasting tasks across 7 domains. Each task requires combining numerical history and natural language context to make accurate predictions. The paper also introduces the RCRPS evaluation metric and the Direct Prompt method, demonstrating that a simple prompting strategy on Llama-3.1-405B (RCRPS=0.159) significantly outperforms all…
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Context-aided forecasting"
  - "Time series benchmark"
  - "LLM forecaster"
  - "RCRPS"
  - "Multimodal"
date: 2026-05-08
content_hash: 89627da8b26b28ca
---

# Context is Key: A Benchmark for Forecasting with Essential Textual Information

**Conference**: ICML 2025  
**arXiv**: [2410.18959](https://arxiv.org/abs/2410.18959)  
**Code**: [GitHub](https://github.com/ServiceNow/context-is-key-forecasting)  
**Area**: Time Series / Multimodal Forecasting  
**Keywords**: Context-aided forecasting, Time series benchmark, LLM forecaster, RCRPS, Multimodal

## TL;DR

This paper proposes the Context is Key (CiK) benchmark—consisting of 71 manually designed forecasting tasks across 7 domains. Each task requires combining numerical history and natural language context to make accurate predictions. The paper also introduces the RCRPS evaluation metric and the Direct Prompt method, demonstrating that a simple prompting strategy on Llama-3.1-405B (RCRPS=0.159) significantly outperforms all statistical and time-series foundation models.

## Background & Motivation

**Background**: Time-series forecasting traditionally relies solely on historical numerical data. Recently, two new trends have emerged: (1) time-series foundation models (e.g., Chronos, Moirai, Lag-Llama) that learn generalizable forecasting capabilities across domains, and (2) LLM-adapted forecasters (e.g., LLMP, UniTime) that leverage natural language to integrate side information.

**Limitations of Prior Work**: Although LLM-based forecasters claim the ability to incorporate textual information, no systematic benchmark exists to evaluate this capability. In existing context-aided benchmarks (e.g., Zhang et al. 2023, Merrill et al. 2024), the textual context is not guaranteed to be helpful for forecasting—models can often achieve good predictions by relying on numerical data alone, which fails to distinguish whether they truly leverage the text.

**Key Challenge**: Evaluating multimodal forecasting capabilities requires a benchmark where "context is essential": it must be impossible to forecast accurately based on numbers alone without understanding the textual information.

**Goal**: (1) Establish a forecasting benchmark where textual context is necessary for every task; (2) systematically evaluate performance differences among various model types under context-aided settings; (3) propose proper scoring rules suitable for evaluation.

**Key Insight**: Meticulously design the text context and numerical data for each task by hand to ensure that the context contains crucial information for forecasting. For instance, forecasting solar power output requires knowing that "this is a solar panel" to infer that nighttime generation is zero. Human verification is conducted on 95% of the tasks to confirm context efficacy.

**Core Idea**: Forecasting requires not only observing "how numbers change" but also understanding "what is happening"—the CiK benchmark precisely measures a model's ability to do so.

## Method

### Overall Architecture

The CiK benchmark consists of 71 forecasting tasks across 7 domains (climatology, economics, energy, mechanics, public safety, transport, and retail), utilizing 2644 real-world time-series datasets. Each task includes: numerical history $\mathbf{X}_H$, natural language context $\mathbf{C}$, and the future target to forecast $\mathbf{X}_F$. The objective is to estimate $P(\mathbf{X}_F|\mathbf{X}_H, \mathbf{C})$. Crucially, each task is validated for context necessity (confirmed valid in 95% of instances) via humans and an LLM evaluation panel.

### Key Designs

1. **Five-Category Textual Context Taxonomy**

    - **Function**: Systematically define the types of context information to ensure the benchmark covers diverse auxiliary information in real-world scenarios.
    - **Mechanism**: (1) **Time-invariant information** $\mathbf{c}_I$: process descriptions and variable properties (e.g., "solar power generation" to infer zero at night); (2) **Future information** $\mathbf{c}_F$: future events or scenarios (e.g., "ATM offline for two days" to infer zero withdrawals); (3) **Historical information** $\mathbf{c}_H$: past statistics beyond the visible history (e.g., "peak of 100 in the same period last year"); (4) **Covariate information** $\mathbf{c}_{cov}$: numerical behavior of related variables; (5) **Causal information** $\mathbf{c}_{causal}$: causal relationships between covariates and target variables.
    - **Design Motivation**: Different context types place different demands on model capabilities—understanding time-invariant information requires common sense, while understanding causal information requires reasoning. This classification allows for fine-grained analysis of model performance across different context types.

2. **Region of Interest CRPS (RCRPS) Scoring Rule**

    - **Function**: An evaluation metric specifically designed for context-aided probabilistic forecasting.
    - **Mechanism**: Extends standard CRPS by: (1) **Region of Interest (RoI)**: assigning higher weights to time windows most relevant to the context; (2) **Constraint violation penalty** $\beta \cdot \text{CRPS}(v_\mathbf{C}(\tilde{\mathbf{X}}_F), 0)$: imposing an additional penalty ($\beta=10$) when the prediction violates contextual constraints; (3) **Normalization** $\alpha$: allowing fair aggregation across different task scales. RCRPS sums the CRPS inside and outside the RoI by assigning 1/2 weight to each.
    - **Design Motivation**: Standard CRPS treats all time steps equally, but the key in context-aided forecasting is whether the model predicts correctly within the "context-relevant time windows."

3. **Direct Prompt Forecasters**

    - **Function**: A simple yet powerful baseline for using LLMs directly as context-aided forecasters.
    - **Mechanism**: Formats the numerical time series as text (e.g., in table format) and concatenates it with the natural language context into a prompt, directly querying the LLM to output future numerical predictions. Multi-sample generation is used to construct the predictive distribution. It requires no fine-tuning or specialized tokenization, directly leveraging the LLMs' in-context learning capabilities.
    - **Design Motivation**: Compared to methods like LLMP that require training, Direct Prompt tests the "native" ability of LLMs to integrate numerical and textual data.

### Data Contamination Mitigation Strategy

To reduce the risk of LLM data contamination, the authors employ continuously updated live data sources, derivative series (e.g., incident logs $\rightarrow$ time series), and minor perturbations (adding noise/time shifts).

## Key Experimental Results

### Main Results — Weighted Average RCRPS↓

| Model Category | Model | Avg RCRPS↓ | Avg Rank↓ |
|---------|------|-----------|-----------|
| Direct Prompt | **Llama-3.1-405B** | **0.159** | **4.516** |
| Direct Prompt | GPT-4o | 0.274 | 4.381 |
| LLMP | Llama-3-70B (base) | 0.236 | 6.522 |
| Multimodal | UniTime | 0.370 | 14.675 |
| Time-series Foundation Model* | Chronos-Large | 0.326 | 12.298 |
| Time-series Foundation Model* | Moirai-Large | 0.520 | 12.873 |
| Statistical Model* | ARIMA | 0.475 | 12.721 |
| Statistical Model* | ETS | 0.530 | 15.001 |

### Analysis by Context Type (Direct Prompt RCRPS↓)

| Model | Time-invariant | Historical | Future | Covariates | Causal |
|------|---------|---------|---------|-------|------|
| Llama-405B | 0.174 | 0.146 | **0.075** | 0.164 | 0.398 |
| GPT-4o | 0.218 | **0.118** | 0.121 | 0.250 | 0.858 |
| Llama-70B | 0.336 | 0.180 | 0.194 | 0.228 | 0.629 |
| Qwen-2.5-7B | 0.290 | 0.176 | 0.287 | 0.240 | 0.525 |

### Key Findings

- **Direct Prompt Llama-3.1-405B dominantly leads**: Its RCRPS of 0.159 is drastically lower than the best time-series foundation model, Chronos, at 0.326 (a 51% improvement).
- **Textual context is the decisive factor**: Time-series foundation models cannot leverage context and remain systematically behind on tasks requiring context.
- **Model scale is key**: Scaling from Llama-405B $\rightarrow$ 70B $\rightarrow$ Qwen-0.5B raises RCRPS from 0.159 to 0.290 and 0.463, indicating that larger LLMs are significantly better at utilizing context.
- **Causal information is the most difficult**: All models yield their worst performance on causal reasoning tasks (0.360–0.858), demonstrating that causal reasoning remains a major bottleneck for LLMs.
- **Direct Prompt outperforms LLMP**: Training-free direct prompting performs better than LLMP which requires training, indicating that pre-trained LLMs possess sufficiently strong in-context learning capabilities.

## Highlights & Insights

- **"Context is key" directly targets the blind spot of time-series foundation models**: The example of solar generation being zero at night is highly intuitive—without knowing it is a solar panel, prediction is impossible. This reveals the fundamental limitation of purely numerical time-series models: they cannot encode domain knowledge and event constraints. CiK establishes clear evaluation standards for the multimodal time-series forecasting field.
- **Exquisite RCRPS design**: By combining region-of-interest weighting with constraint violation penalties, RCRPS precisely measures whether models "understand and utilize the context," avoiding the underestimation of context-relevant time steps inherent in standard CRPS.

## Limitations & Future Work

- All 71 tasks are manually designed, limiting the scale and generalizability across all real-world scenarios.
- LLMs' numerical understanding remains limited—performance is weaker on tasks requiring precise numerical calculation (e.g., causal reasoning).
- End-to-end trained multimodal time-series models (e.g., frameworks specifically co-trained on numbers and text) are yet to be tested.
- The efficacy of mitigation strategies against data contamination is not fully verified.
- High evaluation cost (5 instances $\times$ 25 samples $\times$ 71 tasks).

## Related Work & Insights

- **vs. Time-series foundation models like TimesFM/Chronos**: While strong in pure numerical prediction, CiK reveals that their inability to exploit textual context is a structural shortcoming.
- **vs. LLMP (Requeima et al., 2024)**: While LLMP requires training adapter layers, the zero-shot Direct Prompt approach performs even better, showing that pre-trained LLMs already encompass sufficient forecasting capability.
- **vs. Existing benchmarks (Zhang et al., 2023; Merrill et al., 2024)**: These benchmarks do not guarantee that the context is actually helpful, whereas CiK ensures context necessity through manual design and human verification.
- **Insight**: Future time-series models should natively support multimodal inputs—treating natural language context as a first-class citizen.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first time-series forecasting benchmark where context is indispensable, filling a major gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 20+ models on 71 tasks, with in-depth fine-grained analysis across context types.
- Writing Quality: ⭐⭐⭐⭐⭐ Intuitive motivation, vivid examples, and a clear design process for RCRPS.
- Value: ⭐⭐⭐⭐⭐ Provides crucial directional guidance to the time-series forecasting community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities](what_limits_virtual_agent_application_omnibench_a_scalable_multi-dimensional_ben.md)
- [\[ICML 2026\] FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs](../../ICML2026/multimodal_vlm/futureomni_evaluating_future_forecasting_from_omni-modal_context_for_multimodal_.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](../../ICCV2025/multimodal_vlm/mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[ICML 2025\] CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization](cocoa-mix_confusion-and-confidence-aware_mixture_model_for_context_optimization.md)

</div>

<!-- RELATED:END -->
