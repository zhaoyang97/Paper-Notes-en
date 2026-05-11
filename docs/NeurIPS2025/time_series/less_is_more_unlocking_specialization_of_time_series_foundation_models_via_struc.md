---
title: >-
  [Paper Note] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning
description: >-
  [NeurIPS 2025][Time Series][time series foundation models] This paper reveals that pretrained time series foundation models (TSFMs) exhibit inherent task-relevant sparsity…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "time series foundation models"
  - "structured pruning"
  - "model specialization"
  - "fine-tuning"
  - "sparsity"
date: 2026-05-08
content_hash: 00b77a519b6c1597
---

# Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning

**Conference**: NeurIPS 2025
**arXiv**: [2505.23195](https://arxiv.org/abs/2505.23195)
**Code**: [Available](https://github.com/SJTU-DMTai/Prune-then-Finetune)
**Area**: Time Series Forecasting / Model Compression
**Keywords**: time series foundation models, structured pruning, model specialization, fine-tuning, sparsity

## TL;DR

This paper reveals that pretrained time series foundation models (TSFMs) exhibit inherent task-relevant sparsity, and proposes a *Prune-then-Finetune* paradigm—removing task-irrelevant parameters via structured pruning so that a pruned-then-finetuned smaller model significantly outperforms direct fine-tuning of the full model, and even surpasses strong specialized baselines.

## Background & Motivation

- **Background**: TSFMs such as TimesFM, Moirai, and Time-MoE acquire strong zero-shot forecasting capabilities through large-scale pretraining.
- **Limitations of Prior Work**: Surprisingly, even after fine-tuning, TSFMs fail to consistently outperform small specialized models trained from scratch, such as PatchTST.
- **Key Challenge**: Empirical analysis reveals two key properties of TSFMs. (1) *Attention head output sparsity*: up to 50% of attention heads produce outputs whose relative residual norm is below 1%, contributing negligibly to predictions. (2) *FFN activation sparsity*: up to 60% of FFN intermediate channels are never activated (activation probability < 5%). Critically, sparsity patterns vary across datasets—Weather and ETTm1 exhibit distinct activation patterns—indicating that TSFMs have learned to selectively activate sub-networks for different tasks. Conventional fine-tuning preserves the full architecture and updates all parameters, potentially disrupting these valuable task-relevant substructures.
- **Goal**: Propose a structured pruning–based specialization paradigm that removes task-irrelevant parameters before fine-tuning, enabling better downstream adaptation with fewer parameters.

## Method

### Overall Architecture

The paper proposes a two-stage deployment pipeline called *Prune-then-Finetune*. The first stage performs **architectural specialization** via structured pruning (removing unimportant channels and attention heads); the second stage performs **weight specialization** via fine-tuning within the reduced parameter space.

### Key Designs

1. **Pruning Unit Definition**: The atomic pruning unit is the input/output channel of a linear layer. For each linear layer $f(\cdot)$, binary masks $\mathbf{m}^{\text{in}} \in \{0,1\}^{d_{\text{in}}}$ and $\mathbf{m}^{\text{out}} \in \{0,1\}^{d_{\text{out}}}$ are introduced: $\mathbf{h} = f(\mathbf{x} \odot \mathbf{m}^{\text{in}}) \odot \mathbf{m}^{\text{out}}$. Due to residual connections, the output channel mask and the input channel mask of the subsequent layer must be handled independently.

2. **Loss-Based Importance Scoring**: The importance of channel $i$ is defined as the change in loss upon its removal: $s_i = |\mathcal{L}(\mathbf{m} - \boldsymbol{e}_i) - \mathcal{L}(\mathbf{m})|$. A second-order Taylor expansion is used for approximation:
$$s_i \approx \left| -\frac{1}{N}\sum_{n=1}^{N}\frac{\partial \mathcal{L}_n}{\partial m_i} + \frac{1}{2N}\sum_{n=1}^{N}\left(\frac{\partial \mathcal{L}_n}{\partial m_i}\right)^2 \right|$$
   The second-order term is approximated via Fisher information. When a channel consistently produces zero activations, its importance score is automatically zero, which is consistent with the FFN sparsity analysis.

3. **Progressive Pruning**: To avoid removing too many units at once, pruning proceeds in mini-batches: importance scores are computed per batch, smoothed via exponential moving average $\tilde{s}^{(j)} = \alpha s^{(j)} + (1-\alpha)\tilde{s}^{(j-1)}$, and the $K$ globally least important channels are removed after each step. Pruning completes after one or more epochs, followed by fine-tuning of the remaining parameters.

### Loss & Training

- During pruning, the downstream forecasting loss (MSE) is used to compute channel importance scores.
- During fine-tuning, standard MSE optimization is applied to the remaining parameters.
- Hyperparameters (learning rate, pruning ratio) are tuned on the validation set.
- For autoregressive TSFMs, a single model per dataset is trained and reused across multiple prediction horizons; for non-autoregressive TSFMs, separate models are trained per horizon.

## Key Experimental Results

### Main Results (7 TSFMs × 6 Datasets)

Results on ETTm1 (MSE averaged over prediction horizons 96/192/336/720):

| Model | Finetune MSE | Prune+Finetune MSE | Gain | vs. PatchTST |
|---|---|---|---|---|
| Time-MoE_base | 0.391 | **0.385** | −1.5% | Win |
| Time-MoE_large | 0.374 | **0.369** | −1.3% | Win |
| Timer-XL | 0.410 | **0.396** | −3.4% | Win |
| Moirai_base | 0.418 | **0.407** | −2.6% | Win |
| Moirai_large | 0.429 | **0.423** | −1.4% | Win |
| Chronos-bolt | 0.422 | **0.421** | −0.2% | Win |
| PatchTST (baseline) | 0.398 | — | — | — |

### Ablation Study

| Configuration | Finding |
|---|---|
| Direct fine-tuning vs. Prune+Finetune | Prune+Finetune is superior in 83% of tasks |
| Maximum improvement | 22.8% (Time-MoE_base on ETTm2) |
| Maximum degradation | Only 1% (downside risk is well-controlled) |
| Win rate vs. PatchTST | Increases from 90% to **100%** (when selecting the best TSFM) |
| Prunable parameter ratio | Up to **90%** of parameters removed while performance improves |
| Inference speedup | Up to **7×** |

### Key Findings

- **Pruning improves performance, not merely compresses**: Unlike LLM pruning, which typically incurs performance degradation, pruning TSFMs consistently improves accuracy.
- **Longer horizons yield larger gains**: Autoregressive models benefit most at the 720-step horizon, due to reduced error accumulation.
- **Pruning acts as regularization**: It effectively constrains fine-tuning to the task-relevant parameter subspace, mitigating overfitting to noisy features.
- **Cross-domain transferability**: Models pruned on one dataset exhibit improved zero-shot performance on other datasets from the same domain.
- **LoRA is insufficient**: Parameter-efficient fine-tuning via LoRA does not outperform full fine-tuning, as the root cause lies in architectural redundancy rather than parameter count.

## Highlights & Insights

- The **"Less is More" insight is profound**: the performance bottleneck of TSFMs lies not in insufficient parameters, but in the interference of irrelevant parameters during specialization.
- The most significant conceptual shift in this paper is repositioning pruning from a *compression tool* to a *specialization tool*.
- The sparsity analysis (attention head output norms + FFN activation probabilities) provides strong intuitive grounding.
- The non-stationarity of time series makes persistently informative features scarce, which is the fundamental reason why pruning is particularly well-suited to TSFMs.

## Limitations & Future Work

- The pruning ratio must be tuned for each dataset, increasing deployment overhead.
- The current approach applies a globally uniform pruning strategy without accounting for layer-wise differences (shallow vs. deep layers).
- Progressive pruning incurs additional training cost (though it is generally faster than full fine-tuning).
- The combination of unstructured and structured pruning remains unexplored.

## Related Work & Insights

- The loss-based importance scoring from LLM-Pruner serves as the direct methodological foundation; however, LLM pruning typically results in performance degradation.
- The phenomenon of *pruning improving performance* revealed in this paper is rare in the LLM literature, highlighting a fundamental distinction between time series and language tasks.
- The proposed paradigm is broadly applicable: any foundation model with heterogeneous task relevance across parameters may benefit from pruning for downstream specialization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Reframing pruning as a specialization mechanism rather than a compression technique represents a valuable conceptual contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 7 TSFMs, 6 datasets, multi-dimensional analyses, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — The pipeline from empirical analysis to method to experiments is logically coherent.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a simple and effective standard deployment pipeline for TSFMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
