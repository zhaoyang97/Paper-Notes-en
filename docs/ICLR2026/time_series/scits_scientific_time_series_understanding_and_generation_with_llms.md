---
title: >-
  [Paper Note] Untitled
description: >-
  [ICLR2026][Time Series] Paper note
tags:
  - ICLR 2026
  - Time Series
date: 2026-05-08
content_hash: cf4e543b0492e135
---
## TL;DR
This work proposes the SciTS benchmark covering 43 tasks across 12 scientific domains with 54K+ instances (lengths from $10^0$ to $10^7$, frequencies up to 10 MHz), systematically evaluates 17 models and finds that general-purpose LLMs generalize better than specialized time-series models while text/image encodings each have distinct limitations, and accordingly designs the TimeOmni framework, which employs multi-patch experts with a routing mechanism and patch reprogramming to explicitly model temporal dynamics in joint training with an LLM backbone.

## Background & Motivation

**Background**: The scientific reasoning capabilities of LLMs have attracted considerable attention in recent years. Time series, as one of the most fundamental data modalities in scientific domains (physics, astronomy, biology, engineering, etc.), remains severely underexplored in current multimodal LLMs. Existing approaches either encode numerical sequences as text (producing extremely long sequences) or convert them to images (losing numerical precision), neither of which adequately supports the understanding and generation of scientific time series.

**Limitations of Prior Work**: (1) Existing time-series benchmarks are predominantly focused on conventional tasks such as forecasting and anomaly detection, lacking coverage of scientific domains (astronomy, geoscience, neuroscience, etc.); (2) unified time-series models either support only forecasting or only analysis, and cannot simultaneously handle understanding and generation; (3) scientific time-series signals exhibit extreme heterogeneity (astronomical light curves vs. EEG signals vs. seismic waveforms vs. radar communications), which existing models struggle to accommodate.

**Key Insight**: Construct the first comprehensive scientific time-series benchmark SciTS → identify problems through systematic evaluation → design a LLM-native time-series processing framework, TimeOmni.

**Key Challenges**: Scientific time-series signals span frequencies from $10^{-5}$ Hz to $10^7$ Hz, lengths from a few points to millions of samples, and dimensionality from 1 to 58. This extreme heterogeneity poses a fundamental challenge to unified modeling.

**Limitations of Prior Attempts**: Although UniTS integrates QA and forecasting, it relies on an independent architectural design that is incompatible with general-purpose LLM training pipelines. Specialized models such as Moirai and TimeMoE support only forecasting and cannot handle tasks such as imputation or event localization.

**Goal**: A unified framework is needed that leverages the reasoning and world knowledge of LLMs while explicitly modeling temporal dynamics, and that remains compatible with general-purpose LLM training pipelines.

## Method

### Overall Architecture
TimeOmni consists of three core components: a **time-series encoder** (comprising a router, a family of patch experts, and patch reprogramming), an **LLM backbone** (Qwen3-8B fine-tuned with DoRA), and **task-specific output heads** (softmax text generation for understanding tasks; a linear regression head for generation tasks). Given an input time-series signal $\mathbf{X} \in \mathbb{R}^{T' \times N}$, it is first flattened along the temporal dimension to $\mathbf{X}' \in \mathbb{R}^{NT' \times 1}$, then encoded to $\mathbf{X}_{\text{enc}} \in \mathbb{R}^{T_{\text{enc}} \times D_{\text{llm}}}$ (where $T_{\text{enc}}$ is typically 100–200), and concatenated with text prompt embeddings before being fed into the LLM backbone.

### Key Design 1: Multi-Patch Expert Routing

- **Function**: Automatically selects the most appropriate patch size for input signals of varying lengths and frequencies, dividing the raw signal into a fixed number of patches.
- **Mechanism**: The router selects patch size $D_{\text{patch}}$ based on the total flattened length $T = NT'$, ensuring the patch count remains between 100 and 200:
$$\frac{T}{200} < D_{\text{patch}} < \frac{T}{100}$$
A patch expert reshapes the signal from $\mathbb{R}^{T \times 1}$ to $\mathbb{R}^{\lceil T/D_{\text{patch}} \rceil \times D_{\text{patch}}}$, then maps it to $\mathbf{X}_{\text{patch}} \in \mathbb{R}^{\lceil T/D_{\text{patch}} \rceil \times D_{\text{enc}}}$ via 1D convolution.
- **Design Motivation**: Scientific time-series lengths span from $10^0$ to $10^7$, making a fixed patch size impractical—small patches cause the patch count to explode for long sequences (leading to out-of-memory errors), while large patches collapse short sequences into a single patch, discarding information. Multi-patch experts address this fundamental dilemma through scale-adaptive patching.

### Key Design 2: Patch Reprogramming

- **Function**: Reprograms time-series patch representations using the LLM's vocabulary embeddings, mapping temporal features into the LLM's semantic space.
- **Mechanism**: The LLM word embeddings $\mathbf{E} \in \mathbb{R}^{\text{vocab\_size} \times D_{\text{llm}}}$ are first projected to $\mathbb{R}^{1000 \times D_{\text{llm}}}$ via a linear layer. $\mathbf{X}_{\text{patch}}$ then interacts with $\mathbf{E}$ through a multi-head cross-attention mechanism:
$$\mathbf{X}_{\text{enc}} = \text{Linear}(\text{CrossAttn}(\mathbf{X}_{\text{patch}}, \mathbf{E}, \mathbf{E}))$$
where $\mathbf{X}_{\text{patch}}$ serves as the query and $\mathbf{E}$ serves as both key and value.
- **Design Motivation**: Directly feeding time-series embeddings into the LLM leads to modality misalignment. By leveraging the LLM's existing vocabulary embeddings as a "bridge," temporal features are re-expressed as vectors in a semantic space the LLM can interpret, eliminating the modality gap. Ablation experiments confirm that replacing the reprogramming module with a simple MLP leads to consistent performance degradation.

### Key Design 3: Prompt Strategy and Dual Output Heads

- **Function**: Adopts different prompt concatenation strategies and output heads according to the task type.
- **Mechanism**:
    - Understanding tasks (classification / anomaly detection / QA): A prompt-as-suffix strategy is adopted, i.e., $[\mathbf{X}_{\text{enc}}; \mathbf{P}]$—the model observes the signal before the question, with output generated as text tokens via softmax.
    - Generation tasks (forecasting / imputation / synthesis): A prompt-as-prefix strategy is adopted, i.e., $[\mathbf{P}; \mathbf{X}_{\text{enc}}]$—the model processes the instruction before the signal, with output mapped to the target time-series length via flattening followed by a linear layer.
- **Design Motivation**: Understanding tasks require the model to first "observe" the signal and then "answer" questions, mirroring the cognitive process of examining data before analysis; generation tasks require understanding task requirements prior to processing the input signal. Multiple regression heads covering different output lengths are predefined, selected by nearest-match with truncation as needed.

### Key Design 4: Multivariate Signal Handling

- **Function**: Handles multivariate scientific signals with dimensionality ranging from 1 to 58.
- **Mechanism**: A multivariate signal $\mathbf{X} \in \mathbb{R}^{T' \times N}$ is flattened along the temporal dimension to $\mathbf{X}' \in \mathbb{R}^{NT' \times 1}$, treated uniformly as a univariate long sequence. The router then automatically selects an appropriate patch size to accommodate the total flattened length.
- **Design Motivation**: This avoids designing separate encoders for each channel, reducing architectural complexity, while allowing patch experts to naturally capture cross-channel temporal dependencies through the flattening operation.

## Key Experimental Results

### Understanding Task Results (F1%, averaged per domain)

| Model | Astro. | Bioacoustics | Geosci. | Econ. | Meteor. | Manuf. | Neuro. | Physio. | Radar | Urban | Avg. Rank |
|------|------|---------|---------|------|------|------|---------|------|------|------|---------|
| GPT-4.1-mini | 41.4 | 6.7 | 67.0 | 90.4 | 45.3 | 31.7 | 13.5 | 26.8 | 17.6 | 64.4 | 6.1 |
| Gemini2.5-Flash | 40.2 | 10.3 | 67.6 | 87.8 | 51.8 | 28.8 | 12.7 | 31.8 | 17.2 | 64.6 | 5.5 |
| GPT-5-mini (multimodal) | 42.3 | 10.7 | 67.6 | 83.8 | 45.3 | 38.4 | 13.9 | 25.0 | 16.5 | 64.8 | 6.0 |
| UniTS | 38.2 | 8.1 | 0.0 | 27.1 | 9.8 | 48.5 | 25.9 | 22.9 | 10.6 | 67.4 | 7.9 |
| ChaTS | 11.3 | — | 64.8 | 79.2 | 51.2 | — | 22.7 | 30.9 | 13.9 | 65.4 | 9.2 |
| **TimeOmni** | **73.2** | **58.1** | **82.5** | **96.4** | **61.3** | **82.0** | **60.1** | **45.9** | **68.9** | **64.8** | **1.9** |

### Generation Task Results (swMAPE, lower is better)

| Model | Astro. | Geosci. | Meteor. | Econ. | Neuro. | Energy | Physio. | Urban | Math | Avg. Rank |
|------|------|---------|------|------|---------|------|------|------|------|---------|
| GPT-4.1-mini | 100.9 | 65.0 | 85.0 | 112.2 | 61.4 | 2.0e3 | 610.6 | 670.0 | 1.2e3 | 6.7 |
| Gemini2.5-Flash | 116.6 | 63.0 | 107.5 | 4.5 | 38.7 | 307.6 | 60.5 | 391.4 | 477.5 | 4.6 |
| Moirai-Large | — | — | 51.7 | 1.8 | — | — | — | — | 360.1 | 8.3 |
| UniTS | 3.3e6 | — | 42.0 | — | 147.3 | — | 216.3 | — | — | 9.8 |
| **TimeOmni** | **2.8** | **2.2** | **37.5** | **5.3** | **46.6** | **66.4** | **91.7** | **402.7** | **656.5** | **4.1** |

## Key Findings

1. **General-purpose LLMs generalize better than specialized TS models**: Across the 12 scientific domains in SciTS, general-purpose LLMs (e.g., GPT-4.1-mini, Gemini2.5-Flash) demonstrate stronger cross-domain generalization than specialized time-series models (Moirai, TimeMoE, etc.). Specialized models exhibit severe performance degradation on scientific signals outside their training distribution.

2. **Task-dependent complementarity of text vs. image encoding**: Image inputs outperform text inputs on understanding tasks (high-level understanding does not require precise numerical values, and images compress long sequences more efficiently); text inputs outperform image inputs on generation tasks (numerical precision is critical). This reveals the complementarity and respective limitations of both encoding strategies.

3. **SciTS is highly challenging**: F1 scores in bioacoustics and radar domains are generally below 10%; high-frequency long sequences (millions of sampling points) cause context overflow or instruction-following failures in a large number of models. Approximately 10% of tasks cannot be handled at all by open-source LLMs.

4. **TimeOmni achieves full coverage and full success**: TimeOmni is the only model that successfully processes all instances across all 43 tasks, while achieving optimal or near-optimal performance on both understanding (average rank 1.9) and generation (average rank 4.1) tasks.

5. **Ablation studies validate key design choices**: (1) Replacing patch reprogramming with an MLP leads to consistent performance degradation; (2) using a fixed patch size causes severe performance deterioration on sequences of extreme length; (3) fine-tuning Qwen2.5VL and TimeMoE fails to compensate for architectural limitations, indicating that the bottleneck lies in architecture rather than training data.

## Highlights & Insights

- **SciTS fills an important gap**: As the first time-series benchmark covering 12 scientific domains, it includes 7 task types and extremely heterogeneous signals spanning 12 orders of magnitude in frequency, providing a standardized evaluation platform for LLMs processing scientific time series.
- **The counterintuitive finding that "general > specialized"**: Specialized time-series models perform worse than general-purpose LLMs on aperiodic scientific signals, indicating that general reasoning capabilities and world knowledge are more important than domain-specific architectural design.
- **Theoretical elegance of the patch routing mechanism**: By enforcing the constraint $T/200 < D_{\text{patch}} < T/100$, signals of arbitrary length are uniformly mapped to 100–200 tokens, simultaneously avoiding excessively long sequences and ensuring adequate information density—a design that is both simple and effective.
- **Framework compatibility**: TimeOmni integrates seamlessly into general-purpose LLM training pipelines and supports joint training with other modalities (text/image/audio), laying the groundwork for truly scientific multimodal LLMs.

## Limitations & Future Work

- All baseline models are evaluated under zero-shot settings without domain-specific fine-tuning, which may underestimate the true capabilities of some models.
- TimeOmni is fine-tuned on Qwen3-8B, a relatively modest model scale; scaling effects remain insufficiently explored.
- SciTS data are primarily sourced from open-source datasets and simulated data, which may exhibit distributional differences from raw experimental data in real scientific research.
- Simple flattening of multivariate signals may discard inter-channel structural information (e.g., the spatial topology of EEG channels).
- The "thinking" mode of closed-source LLMs has not been evaluated (preliminary experiments suggest no improvement at substantial cost).

## Related Work & Insights

### vs. Chronos / Moirai / TimeMoE (specialized time-series models)
These models perform well on specific forecasting tasks (e.g., Moirai achieves the lowest swMAPE in economics and mathematics), but exhibit **extremely low task coverage** (supporting forecasting only) and cannot handle classification, QA, imputation, or other tasks. Evaluation on SciTS reveals their generalization bottleneck in scientific domains: architectures designed specifically for conventional periodic signals cannot adapt to heterogeneous scientific signals.

### vs. UniTS / ChaTS (unified time-series models)
UniTS attempts to integrate QA and forecasting but relies on an independent architecture that cannot be incorporated into LLM training; ChaTS supports analysis tasks but fails entirely on certain domains (bioacoustics, manufacturing). Through its LLM-native design, TimeOmni **achieves a unified treatment of both understanding and generation** while maintaining compatibility with LLM training pipelines.

### vs. Multimodal LLMs (GPT-5-mini / InternVL / QwenVL)
Image encoding offers advantages for high-level understanding tasks (compressing long sequences), but is severely limited for generation tasks requiring numerical precision. TimeOmni avoids the text/image encoding dilemma through an explicit time-series encoder, demonstrating superior performance on both task categories.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First comprehensive scientific TS benchmark combined with an LLM-native TS framework, filling a critical gap
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale systematic evaluation across 17 models × 43 tasks × 12 domains, supplemented by ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous benchmark design, highly informative figures and tables, clear motivation
- **Value**: ⭐⭐⭐⭐⭐ Significant contribution to LLM-based scientific applications; both the benchmark and framework are open-sourced

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](scits_scientific_time_series_llm.md)
- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)

</div>

<!-- RELATED:END -->
