---
title: >-
  [Paper Note] SEMPO: Lightweight Foundation Models for Time Series Forecasting
description: >-
  [NeurIPS 2025][Time Series][Time series foundation models] This paper proposes SEMPO — a lightweight time series foundation model with only 6.5M parameters pretrained on 83M time points — that combines energy-aware spectral decomposition with a mixture-of-prompts Transformer to surpass large foundation models with over 100× more parameters in zero-shot and few-shot forecasting.
tags:
  - NeurIPS 2025
  - Time Series
  - Time series foundation models
  - lightweight
  - spectral decomposition
  - mixture of prompts
  - zero-shot/few-shot forecasting
date: 2026-05-08
content_hash: aac2fae2d3264a6c
---

# SEMPO: Lightweight Foundation Models for Time Series Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2510.19710](https://arxiv.org/abs/2510.19710)
**Code**: [https://github.com/mala-lab/SEMPO](https://github.com/mala-lab/SEMPO)
**Area**: Time Series Forecasting
**Keywords**: Time series foundation models, lightweight, spectral decomposition, mixture of prompts, zero-shot/few-shot forecasting

## TL;DR

This paper proposes SEMPO — a lightweight time series foundation model with only 6.5M parameters pretrained on 83M time points — that combines energy-aware spectral decomposition with a mixture-of-prompts Transformer to surpass large foundation models with over 100× more parameters in zero-shot and few-shot forecasting.

## Background & Motivation

Time series foundation models (FMs) pretrained on large-scale multi-domain data are driving a paradigm shift in forecasting by enabling zero-shot/few-shot generalization. Existing approaches (e.g., Chronos with 710M parameters, Time-MoE with 453M, Moment with 385M) rely on massive architectures and enormous pretraining corpora (billions to hundreds of billions of time points), severely hindering deployment in resource-constrained environments.

**Core Limitations**:

**Low data utilization efficiency**: Pretraining of existing FMs is biased against low-energy frequency signals — the self-attention mechanism in Transformers naturally favors high-energy frequency components, while low-energy signals carrying stable temporal dynamics are overlooked (as illustrated in Figure 1, ChronosS completely ignores low-energy signals).

**Bloated model architectures**: To accommodate heterogeneous temporal patterns across domains, most approaches adopt large Transformers with MoE, resulting in parameter counts in the hundreds of millions.

**Key Challenge**: Generalizability vs. affordability — can generalization capability be maintained or even improved while drastically reducing model size and pretraining data volume?

**Key Insight**: A two-pronged approach —
1. Substantially improving **data utilization efficiency** via Energy-Aware Spectral Decomposition (EASD)
2. Achieving a **lightweight architecture** by replacing large MoE modules with compact dataset-specific prompts via the Mixture-of-Prompts Transformer (MoPFormer)

## Method

### Overall Architecture

SEMPO adopts an encoder-decoder architecture comprising four core components: the EASD module → Patchify & Project → MoPFormer backbone → reconstruction/forecasting head. Training proceeds in two stages: energy-aware pretraining followed by MoP fine-tuning.

### Key Designs

1. **Energy-Aware Spectral Decomposition (EASD)**:

    - **Energy-based splitting**: The input time series is transformed to the frequency domain via FFT, and the spectral energy of each frequency is computed as $\text{Energy}[f]=|Z[f]|^2$. A learnable threshold $\tau$ partitions the spectrum into high-energy components $Z_{\text{Hec}}$ and low-energy components $Z_{\text{Lec}}$, preventing low-energy signals from being overwhelmed by high-energy ones.
    - **Frequency-domain masking**: Within each branch, frequency thresholds $\delta_i$ and direction indicators $d_i$ are independently sampled to generate multiple frequency-band masks $M_i$, selectively suppressing high- or low-frequency bands. Independent sampling parameters for each branch (a decoupled design) promote spectral diversity.
    - Final fusion: $X_{\text{mask}} = \text{iFFT}(Z_{\text{Hec}} \odot M_{\text{Hec}} + Z_{\text{Lec}} \odot M_{\text{Lec}})$

2. **Mixture-of-Prompts Transformer (MoPFormer)**:

    - **Prompt expert pool**: $I=128$ lightweight prompt vectors $\mathbf{e}_i \in \mathbb{R}^{D_p}$ are randomly initialized.
    - **Adaptive router**: Gating scores $\mathbf{s}_{i,p}$ are computed per token via Linear+Softmax, and prompt experts are aggregated via weighted combination: $\tilde{\mathbf{e}}_p = \text{Reshape}(\text{MLP}(\sum_i \mathbf{s}_{i,p} \cdot \mathbf{e}_i))$
    - **Injection into self-attention**: The aggregated prompts are split into key-value pairs and concatenated to the original $K$ and $V$ matrices: $\text{SA} = \text{Attention}(Q=B, K=\text{Concat}(E_{\text{mix}}^K, B), V=\text{Concat}(E_{\text{mix}}^V, B))$
    - This injects dataset-specific knowledge without enlarging the base Transformer, incurring minimal additional parameters.

3. **Two-Stage Training**:

    - **Pretraining**: Self-supervised reconstruction objective (MSE) on multi-domain data without MoP.
    - **MoP fine-tuning**: The Transformer backbone is frozen; only the MoP module and forecasting head are trained using a multi-resolution forecasting strategy.

### Loss & Training

- Pretraining loss: $\mathcal{L}_{\text{pretrain}} = \|X_{1:L} - \hat{X}_{1:L}\|_2^2$
- Fine-tuning loss: $\mathcal{L}_{\text{tuning}} = \sum_{H_r} \|X_{L+1:L+H_r} - \hat{X}_{L+1:L+H_r}\|_2^2 + \|X_{1:L} - \hat{X}_{1:L}\|_2^2$
- Training requires only 4 A6000 GPUs for 10 hours, BF32 precision, batch size 2048.

## Key Experimental Results

### Main Results

**Zero-Shot Forecasting — TSLib Benchmark (Average MSE, $H \in \{96, 192, 336, 720\}$)**

| Model (Params/Data) | ETTh1 | ETTh2 | ETTm2 | Weather | Electricity |
|---------------------|-------|-------|-------|---------|-------------|
| SEMPO (6.5M/83M) | **0.410** | **0.341** | **0.286** | **0.248** | **0.196** |
| Time-MoE-B (113M/309B) | 0.445 | 0.566 | 0.538 | 0.279 | - |
| ChronosL (710M/84B) | 0.541 | 0.385 | 0.315 | 0.292 | 0.326 |
| Moment (385M/1.13B) | 0.708 | 0.392 | 0.319 | 0.291 | 0.861 |
| MoiraiB (91M/27B) | 0.433 | 0.360 | 0.339 | 0.312 | 0.207 |

**Few-Shot Forecasting (5% training data) — TSLib Benchmark**

| Model | ETTh1 | ETTh2 | ETTm1 | Weather | Traffic |
|-------|-------|-------|-------|---------|---------|
| SEMPO | **0.406** | **0.320** | **0.363** | **0.230** | **0.410** |
| TTM | 0.382 | 0.333 | 0.389 | 0.236 | 0.427 |
| Time-LLM | 0.627 | 0.382 | 0.425 | 0.260 | 0.423 |
| PatchTST | 0.694 | 0.827 | 0.526 | 0.269 | 0.418 |

### Ablation Study

| Configuration | ETTh1 MSE | ETTh2 MSE | Weather MSE | Electricity MSE |
|---------------|-----------|-----------|-------------|-----------------|
| SEMPO (full) | 0.410 | 0.341 | 0.248 | 0.196 |
| A.1 Multi-band masking (no energy split) | 0.462 | 0.423 | 0.261 | 0.204 |
| A.2 Random patch masking | 0.446 | 0.400 | 0.261 | 0.243 |
| B.1 Sparse MoE (3 experts, 8.5M params) | 0.441 | 0.358 | 0.253 | 0.223 |
| B.2 Prefix tuning (replacing MoP) | 0.430 | 0.359 | 0.268 | 0.217 |

### Key Findings

- With only 6.5M parameters and 83M pretraining time points, SEMPO achieves an average MSE reduction of 23.1% in zero-shot forecasting, outperforming large FMs with 100× more parameters and 1000× more data.
- EASD is critical: replacing it with standard multi-band masking increases average MSE by approximately 14%.
- The MoP module (6.5M parameters) outperforms a sparse MoE with more parameters (8.5M), demonstrating the efficiency of lightweight prompt-based adaptation.
- Spectral visualizations show that SEMPO effectively captures low-energy but persistent frequency signals, whereas ChronosS and MoiraiL predominantly focus on high-energy components.
- Gating score visualizations across datasets reveal similar routing patterns for same-domain datasets (e.g., ETTh1 and ETTm2) and markedly different patterns for cross-domain datasets (e.g., Traffic and Weather).

## Highlights & Insights

- **A paradigm of doing more with less**: A 6.5M-parameter model outperforms ChronosL at 710M parameters, with the key lying in data utilization efficiency and architectural design.
- **Discovery and resolution of energy bias**: Revealing the energy bias in Transformer pretraining is a significant contribution with implications for other frequency-domain modeling tasks.
- **Elegance of the MoP design**: Replacing a large MoE network with 128 prompt vectors and a router achieves extremely high parameter efficiency.
- **Two-stage training strategy**: Pretraining freezes MoP while fine-tuning freezes the backbone, resulting in a clear separation of responsibilities.

## Limitations & Future Work

- Only univariate (channel-independent) modeling is considered; cross-variate interactions are not captured.
- Although the pretraining data scale (83M) is far smaller than competing models, whether it is sufficiently compact for truly resource-constrained scenarios remains to be validated.
- The marginal improvement of few-shot over zero-shot on ETTh1/ETTh2 suggests room for improvement in adaptation on small-scale or low-variance datasets.
- Probabilistic forecasting is not explored; the model produces only point predictions.

## Related Work & Insights

- The energy-aware approach is transferable to other frequency-domain modeling scenarios (e.g., speech, signal processing).
- The MoP design suggests that for lightweight models requiring cross-domain generalization, a small set of learnable prompts with adaptive routing may be an effective alternative to MoE.
- Comparison with TTM (a lightweight mixer architecture) demonstrates that SEMPO holds advantages even within the lightweight FM category.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both EASD and MoP are novel and theoretically motivated designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 datasets, zero-shot/few-shot settings, two major benchmarks, comprehensive ablations and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures and tables, well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ Achieving state-of-the-art performance at minimal cost offers substantial practical value for time series forecasting in resource-constrained settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
