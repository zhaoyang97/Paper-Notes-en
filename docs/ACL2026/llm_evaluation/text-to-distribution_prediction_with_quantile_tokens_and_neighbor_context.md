---
title: >-
  [Paper Note] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context
description: >-
  [ACL 2026][LLM Evaluation][quantile regression] This paper proposes Quantile Token Regression, a method that inserts dedicated quantile tokens into the input sequence and incorporates retrieved neighbor instances along with their empirical distributions, enabling LLMs to predict full conditional distributions rather than single point estimates. The approach reduces MAPE by approximately 4 points over baselines and narrows prediction intervals by more than 2× on the Airbnb and StackSample datasets.
tags:
  - ACL 2026
  - LLM Evaluation
  - quantile regression
  - distribution prediction
  - retrieval augmentation
  - LLM fine-tuning
  - uncertainty estimation
date: 2026-05-08
content_hash: 092f09cd233e2029
---

# Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context

**Conference**: ACL 2026
**arXiv**: [2604.20216](https://arxiv.org/abs/2604.20216)
**Code**: [github.com/yilunzhu/text2distribution](https://github.com/yilunzhu/text2distribution)
**Area**: LLM Evaluation
**Keywords**: quantile regression, distribution prediction, retrieval augmentation, LLM fine-tuning, uncertainty estimation

## TL;DR
This paper proposes Quantile Token Regression, a method that inserts dedicated quantile tokens into the input sequence and incorporates retrieved neighbor instances along with their empirical distributions, enabling LLMs to predict full conditional distributions rather than single point estimates. The approach reduces MAPE by approximately 4 points over baselines and narrows prediction intervals by more than 2× on the Airbnb and StackSample datasets.

## Background & Motivation

**Background**: LLMs have demonstrated capabilities beyond text generation, performing well on time-series forecasting and regression tasks. Most LLM regression work focuses on point estimation; however, practical applications such as price prediction, demand forecasting, and risk assessment require predicting full probability distributions rather than central tendency values alone. Quantile regression provides a natural framework for distribution prediction, estimating conditional quantiles at different probability levels.

**Limitations of Prior Work**: Vedula et al. (2025) made an important step toward LLM-based distribution prediction by attaching multiple linear regression heads to a shared final hidden state to predict different quantiles. However, this architecture has three critical shortcomings: (1) all quantile predictions originate from the same representational bottleneck, forcing the model to compress all distributional information into a single vector; (2) distributions are predicted from query text alone, without explicit comparison to similar instances — whereas human distributional reasoning naturally relies on analogy; (3) prior retrieval-augmented methods provide only a single scalar label per neighbor, limiting the distributional supervision signal.

**Key Challenge**: Distribution prediction requires capturing differentiated features across quantiles (e.g., lower quantiles attending to popularity signals, upper quantiles attending to complexity indicators), yet the shared representation bottleneck forces all quantiles to extract information from the same features, resulting in an indirect input–output mapping.

**Goal**: Design an architecture in which each quantile has a dedicated representation and a direct input–output path, supported by the full empirical distribution of retrieved neighbors as local evidence.

**Key Insight**: The approach draws inspiration from how humans reason about distributions — by searching for similar items and comparing their price ranges. Dedicated tokens inserted into the self-attention mechanism allow each quantile to independently attend to different parts of the input.

**Core Idea**: Learnable quantile tokens $\langle Q_{\tau_1}\rangle, \ldots, \langle Q_{\tau_Q}\rangle$ are appended to the input sequence so that each quantile establishes a direct connection to the input via self-attention. Retrieved neighbors are augmented with their full empirical distributions (rather than a single label).

## Method

### Overall Architecture
Given a text input (e.g., an Airbnb listing description or a Stack Overflow question), the system first retrieves the top-$K$ semantically similar neighbor instances via dense embeddings. The titles and 9 representative empirical quantiles of each neighbor are concatenated into the input. Then 99 learnable quantile tokens are appended to the input sequence and fed into a pretrained Transformer (the Qwen3 series). The hidden states at each quantile token position are passed through a shared linear regressor to predict the corresponding quantile values, yielding a full 99-quantile conditional distribution.

### Key Designs

1. **Quantile Token Regression**:

    - **Function**: Creates a dedicated representation path for each quantile, replacing the conventional shared-hidden-state with multiple linear heads.
    - **Mechanism**: $Q$ special tokens $\langle Q_{\tau_k}\rangle$ are appended to the input sequence $X = (x_1, \ldots, x_n)$, forming $\widetilde{X} = (x_1, \ldots, x_n, \langle Q_{\tau_1}\rangle, \ldots, \langle Q_{\tau_Q}\rangle)$. After passing through the Transformer, the hidden state $h_{\tau_k}$ at each quantile token position is projected via a shared linear layer $\hat{q}_{\tau_k}(X) = w^\top h_{\tau_k} + b$ to predict the corresponding quantile. Each $\langle Q_{\tau_k}\rangle$ aggregates information across all Transformer layers through self-attention, realizing a direct input–output pathway.
    - **Design Motivation**: In conventional methods, all quantiles are derived from the same hidden state, preventing the model from learning differentiated attention patterns per quantile. Quantile tokens allow, for example, $\langle Q_{10}\rangle$ to attend to popularity signals (predicting fast responses) while $\langle Q_{90}\rangle$ attends to complexity indicators (predicting slow responses), with all quantiles jointly computed within the same attention operation to ensure consistency.

2. **Retrieval-Augmented Distribution Estimation**:

    - **Function**: Provides local evidence for distribution prediction by retrieving semantically similar neighbor instances together with their full empirical distributions.
    - **Mechanism**: Dense embeddings of each instance's full text are computed using Qwen3-Embedding-8B, and the top-$K$ ($K=8$) most similar neighbors are retrieved from the training set. The key innovation is that each neighbor is accompanied by its 9 representative empirical quantiles (1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, and 99th percentiles) rather than a single scalar label, providing the model with rich information about the shape, dispersion, and tail behavior of the neighbor's distribution.
    - **Design Motivation**: Prior retrieval-augmented methods attach only a single point label per neighbor, discarding distributional information. Similar inputs tend to exhibit similar outcome distributions (e.g., price distributions of similar listings); providing the full neighbor distribution enables the model to better estimate dispersion and tail behavior beyond central tendency.

3. **Theoretical Analysis of Loss Functions**:

    - **Function**: Provides a theoretical basis for loss function selection when learning distributions from empirical quantile supervision.
    - **Mechanism**: Four loss functions are analyzed: $\ell_1$ and $\ell_2$ Wasserstein losses are Fisher consistent for the target quantiles in large samples; Pinball-Q, which applies the pinball loss to empirical quantile targets, incurs a systematic bias of order $M_i^{-1/2}$; Pinball-Med uses only the empirical median as a scalar supervision signal, discarding distributional shape information. The theoretically predicted ranking $\ell_1 > \text{Pinball-Q} > \text{Pinball-Med}$ aligns with experimental results.
    - **Design Motivation**: The standard pinball loss is appropriate for learning quantiles from raw observations, but produces systematic bias when the "observations" are themselves empirical quantile estimators. The Wasserstein loss directly matches quantile functions, avoiding this issue.

### Loss & Training
The model is trained using the $\ell_1$ Wasserstein loss on empirical quantile targets, predicting $Q=99$ uniformly spaced quantiles. Qwen3 models (1.7B–14B parameters) are fine-tuned with LoRA. Training and prediction are performed in log space; outputs are exponentiated back to the original scale before computing evaluation metrics.

## Key Experimental Results

### Main Results (Airbnb Dataset, Qwen3-4B)

| Method | avg MAPE↓ | CRPSS↑ | RCIW@95↓ |
|--------|-----------|--------|----------|
| QR (K=0) | 30.31 | 0.4536 | 12.30 |
| QR (K=8) | 27.78 | 0.4700 | 15.08 |
| QT (K=8) | **26.89** | **0.4700** | **7.17** |

### StackSample Dataset (Qwen3-4B)

| Method | avg MAPE↓ | CRPSS↑ | RCIW@99↓ |
|--------|-----------|--------|----------|
| QR (K=0) | 266.65 | 0.0668 | 45480 |
| QR (K=8) | 98.56 | 0.3001 | 2110 |
| QT (K=8) | **84.30** | **0.3375** | **346.9** |

### Ablation Study (Loss Functions, Airbnb Dev, Qwen3-4B)

| Loss Function | avg MAPE↓ | RCIW@95↓ |
|---------------|-----------|----------|
| Pinball-Med | 32.80 | 151.78 |
| Pinball-Q | 32.66 | 151.27 |
| $\ell_2$ Wasserstein | 26.64 | 4.15 |
| $\ell_1$ Wasserstein | **26.55** | **3.55** |

### Key Findings
- Retrieval augmentation is particularly effective on the smaller StackSample dataset (avg MAPE reduced from 266.65 to 98.56, a 63% reduction), validating the assumption that similar inputs exhibit similar distributions.
- Quantile tokens reduce avg MAPE by 14% over the shared-representation baseline on StackSample and narrow prediction intervals by 6×.
- Returns diminish beyond a certain model scale: scaling from 1.7B to 4B reduces MAPE by 7%, while scaling from 8B to 14B yields only a 1% reduction.
- The $\ell_1$ Wasserstein loss achieves the best balance between accuracy and sharpness; pinball losses, while improving CRPSS, produce excessively wide prediction intervals.

## Highlights & Insights
- **Quantile tokens as dedicated probes**: Inserting learnable special tokens into the input sequence to create dedicated representation paths is a transferable idea applicable to any task requiring multiple differentiated outputs from the same input (e.g., multi-task learning, multi-granularity prediction).
- **Distribution-level retrieval augmentation**: Rather than retrieving similar samples alone, providing the full empirical distribution of each neighbor as context constitutes a richer form of retrieval augmentation than point-label-level retrieval, and is generalizable to settings such as time-series distribution forecasting.
- **Theory–practice alignment**: The theoretical analysis of loss functions accurately predicts the experimental ranking, offering principled guidance for loss function selection in practice.

## Limitations & Future Work
- Evaluation is conducted on only two datasets (Airbnb and StackSample); generalizability to broader domains remains to be validated.
- The quantile token approach does not guarantee monotonicity of predictions; post-processing is required to enforce constraints such as the 90th percentile ≥ 80th percentile.
- Increasing the number of neighbors $K$ incurs significant computational and memory overhead ($K=16$ requires approximately 2× memory), necessitating a performance–efficiency trade-off.
- Variance-aware weighting is not explored, which could further improve estimation quality for tail quantiles.

## Related Work & Insights
- **vs. Vedula et al. (2025)**: Shared hidden state with multiple linear heads vs. dedicated quantile tokens. The proposed method's advantage lies in differentiated attention patterns and narrower prediction intervals.
- **vs. Wang et al. (2025) retrieval-augmented regression**: Single label with single price output vs. full empirical distribution with full distributional prediction — a substantially richer information source.
- **vs. classical quantile regression**: Classical methods apply the pinball loss to raw observations; this work applies the Wasserstein loss to empirical quantiles. Both theoretical analysis and experiments demonstrate the superiority of the latter in this setting.

## Rating
- Novelty: ⭐⭐⭐⭐ — The quantile token design is elegant and concise, supported by rigorous theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, four model scales, and comprehensive ablations; dataset diversity could be further improved.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theory and experiments are tightly integrated; writing is clear and precise.
- Value: ⭐⭐⭐⭐ — Practically valuable for text-to-distribution prediction scenarios; the quantile token idea is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](../../ICCV2025/llm_evaluation/odp-bench_benchmarking_out-of-distribution_performance_prediction.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)

</div>

<!-- RELATED:END -->
