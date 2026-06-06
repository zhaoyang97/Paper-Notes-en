---
title: >-
  [Paper Note] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context
description: >-
  [ACL 2026][LLM/NLP][Quantile Regression] This paper proposes the Quantile Token Regression method, which enables LLMs to predict full conditional distributions rather than single point estimates by inserting specialized…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Quantile Regression"
  - "Distribution Prediction"
  - "Retrieval-Augmented"
  - "LLM Fine-tuning"
  - "Uncertainty Estimation"
date: 2026-05-08
content_hash: 946a6d003d240275
---

# Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context

**Conference**: ACL 2026  
**arXiv**: [2604.20216](https://arxiv.org/abs/2604.20216)  
**Code**: [github.com/yilunzhu/text2distribution](https://github.com/yilunzhu/text2distribution)  
**Area**: LLM Evaluation  
**Keywords**: Quantile Regression, Distribution Prediction, Retrieval-Augmented, LLM Fine-tuning, Uncertainty Estimation

## TL;DR
This paper proposes the Quantile Token Regression method, which enables LLMs to predict full conditional distributions rather than single point estimates by inserting specialized quantile tokens into the input sequence and incorporating retrieved neighbor instances with their empirical distributions. This approach reduces MAPE by approximately 4 points and narrows prediction intervals by more than 2x compared to baselines on Airbnb and StackSample datasets.

## Background & Motivation

**Background**: LLMs have demonstrated capabilities beyond text generation, performing well in time-series forecasting and regression tasks. Most LLM regression research focuses on point estimates, but real-world scenarios such as price prediction, demand estimation, and risk assessment require predicting the full probability distribution rather than just central tendency values. Quantile regression provides a natural framework for distribution prediction by estimating conditional quantiles at different probability levels.

**Limitations of Prior Work**: The work by Vedula et al. (2025) made a significant step toward LLM distribution prediction by attaching multiple linear regression heads to a shared final hidden state. However, this architecture has three key flaws: (1) All quantile predictions originate from the same representation bottleneck, forcing the model to compress all distributional information into a single vector; (2) Distributions are predicted based solely on the query text, lacking explicit comparisons with similar instances—an analogical reasoning naturally used by humans; (3) Previous retrieval-augmented methods provide only a single scalar label for each neighbor, limiting distributional supervision.

**Key Challenge**: Distribution prediction requires capturing distinct features for different quantiles (e.g., low quantiles focus on popularity signals, while high quantiles focus on complexity indicators). The shared representation bottleneck forces all quantiles to extract information from the same features, leading to indirect input-output mapping.

**Goal**: To design an architecture where each quantile has its own representation and direct input-output path, while providing local evidence support by retrieving the full empirical distributions of neighbors.

**Key Insight**: Borrowing from how humans reason about distributions—searching for similar items and comparing their price ranges to establish understanding. Simultaneously, by inserting specialized tokens into the self-attention mechanism, it allows each quantile to autonomously attend to different parts of the input.

**Core Idea**: Learnable quantile tokens $\langle Q_{\tau_1}\rangle, \ldots, \langle Q_{\tau_Q}\rangle$ are inserted at the end of the input sequence, allowing each quantile to establish a direct connection with the input through self-attention, while neighbors are augmented with their full empirical distributions (rather than a single label).

## Method

### Overall Architecture
Given a text input (e.g., an Airbnb listing description or a Stack Overflow question), the system first retrieves the Top-K semantically similar neighbor instances via dense embeddings. The titles and 9 representative empirical quantiles of these neighbors are prepended to the input. Then, 99 learnable quantile tokens are appended to the end of the input sequence and fed into a pre-trained Transformer (Qwen3 series). The hidden states of each quantile token are used by a shared linear regressor to predict the corresponding quantile value, outputting a complete 99-quantile conditional distribution.

### Key Designs

1. **Quantile Token Regression**:
    - **Function**: Creates dedicated representation paths for each quantile, replacing the traditional shared hidden state + multi-linear head architecture.
    - **Mechanism**: $Q$ special tokens $\langle Q_{\tau_k}\rangle$ are appended to the input sequence $X = (x_1, \ldots, x_n)$, forming $\widetilde{X} = (x_1, \ldots, x_n, \langle Q_{\tau_1}\rangle, \ldots, \langle Q_{\tau_Q}\rangle)$. After the Transformer, the hidden state $h_{\tau_k}$ at each quantile token position is used to predict the corresponding quantile via a shared linear layer $\hat{q}_{\tau_k}(X) = w^\top h_{\tau_k} + b$. Each $\langle Q_{\tau_k}\rangle$ collects information via self-attention across all Transformer layers, achieving a direct input-output pathway.
    - **Design Motivation**: In traditional methods, all quantiles are derived from the same hidden state, failing to learn differentiated attention patterns for different quantiles. Quantile tokens allow, for example, $\langle Q_{10}\rangle$ to focus on popularity signals (predicting fast answers) and $\langle Q_{90}\rangle$ to focus on complexity metrics (predicting slow answers), while all quantiles are generated jointly in the same attention calculation to ensure consistency.

2. **Retrieval-Augmented Distribution Estimation**:
    - **Function**: Provides local evidence for distribution prediction by retrieving semantically similar neighbor instances and their full empirical distributions.
    - **Mechanism**: Dense embeddings are computed for the full text of each instance using Qwen3-Embedding-8B, and the Top-K (K=8) most similar neighbors are retrieved from the training set. A key innovation is that each neighbor is accompanied by its 9 representative empirical quantiles (1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th percentiles) rather than a single scalar label. This provides the model with rich information about the neighbors' distribution shape, dispersion, and tail behavior.
    - **Design Motivation**: Previous retrieval methods only attached a single point label to neighbors, losing distributional information. Similar inputs tend to exhibit similar outcome distributions (e.g., price distributions for similar listings). Providing full neighbor distributions helps the model better estimate dispersion and tail behavior beyond the central tendency.

3. **Loss Function Theory**:
    - **Function**: Provides a theoretical basis for choosing loss functions for distribution learning under empirical quantile supervision.
    - **Mechanism**: Four loss functions were analyzed: $\ell_1$ and $\ell_2$ Wasserstein losses are Fisher consistent for target quantiles in large samples; Pinball-Q introduces a bias of magnitude $M_i^{-1/2}$ when applying pinball loss to empirical quantile targets; Pinball-Med uses only the empirical median as scalar supervision, losing distribution shape info. The theoretically predicted ranking $\ell_1 > \text{Pinball-Q} > \text{Pinball-Med}$ is consistent with experimental results.
    - **Design Motivation**: Standard pinball loss is suitable for learning quantiles from raw observations, but it produces systematic bias when "observations" themselves are empirical quantile estimators. Wasserstein loss directly matches the quantile function, avoiding this issue.

### Loss & Training
The $\ell_1$ Wasserstein loss is used to train on empirical quantile targets, predicting Q=99 uniformly distributed quantiles. Qwen3 models (1.7B-14B parameters) are fine-tuned using LoRA. Training and prediction are performed in log-space, and evaluation metrics are calculated after exponentiating back to the original scale during inference.

## Key Experimental Results

### Main Results (Airbnb Dataset, Qwen3-4B)

| Method | avg MAPE↓ | CRPSS↑ | RCIW@95↓ |
|------|-----------|--------|----------|
| QR (K=0) | 30.31 | 0.4536 | 12.30 |
| QR (K=8) | 27.78 | 0.4700 | 15.08 |
| QT (K=8) | **26.89** | **0.4700** | **7.17** |

### Main Results (StackSample Dataset, Qwen3-4B)

| Method | avg MAPE↓ | CRPSS↑ | RCIW@99↓ |
|------|-----------|--------|----------|
| QR (K=0) | 266.65 | 0.0668 | 45480 |
| QR (K=8) | 98.56 | 0.3001 | 2110 |
| QT (K=8) | **84.30** | **0.3375** | **346.9** |

### Ablation Study (Loss Function, Airbnb dev, Qwen3-4B)

| Loss Function | avg MAPE↓ | RCIW@95↓ |
|----------|-----------|----------|
| Pinball-Med | 32.80 | 151.78 |
| Pinball-Q | 32.66 | 151.27 |
| $\ell_2$ Wasserstein | 26.64 | 4.15 |
| $\ell_1$ Wasserstein | **26.55** | **3.55** |

### Key Findings
- Retrieval augmentation is particularly effective on the smaller StackSample dataset (avg MAPE decreased from 266.65 to 98.56, a 63% reduction), validating the hypothesis that "similar inputs have similar distributions."
- Compared to the shared representation baseline, the quantile token approach reduced avg MAPE by 14% on StackSample and narrowed prediction intervals by 6x.
- Yields diminish as model scale increases: MAPE dropped by 7% from 1.7B to 4B, but only by 1% from 8B to 14B.
- $\ell_1$ Wasserstein loss achieves the best balance between accuracy and sharpness; while pinball loss optimizes CRPSS, it leads to extremely wide prediction intervals.

## Highlights & Insights
- **Quantile tokens as dedicated probes**: Inserting learnable special tokens into input sequences to create dedicated representation paths is an idea transferable to any task requiring multiple differentiated outputs from the same input (e.g., multi-task learning, multi-granularity prediction).
- **Distribution-level augmentation of neighbors**: Not only retrieving similar samples but also using their full empirical distributions as context input provides significantly more information than "point-label level retrieval augmentation." This can be generalized to scenarios like time-series distribution prediction.
- **Alignment of theory and practice**: The theoretical analysis of loss functions accurately predicted the experimental rankings, providing principled guidance for selecting loss functions in practice.

## Limitations & Future Work
- Evaluations were conducted on only two datasets (Airbnb and StackSample); generalization across more domains remains to be verified.
- The quantile token method does not guarantee prediction monotonicity; post-processing is required to ensure, for example, that the 90th quantile $\ge$ 80th quantile.
- Increasing the number of neighbors K brings significant computational and memory overhead (K=16 requires ~2x memory), requiring a trade-off between performance and efficiency.
- Variance-aware weighting was not explored, which might further improve the estimation quality of tail quantiles.

## Related Work & Insights
- **vs Vedula et al. (2025)**: Shared hidden state + multi-linear heads vs. dedicated quantile tokens. The advantage of this work lies in differentiated attention patterns and narrower prediction intervals.
- **vs Wang et al. (2025) Retrieval-Augmented Regression**: Single label + single price output vs. full empirical distribution + full distribution prediction; the information density is significantly higher here.
- **vs Traditional Quantile Regression**: Traditional methods use pinball loss on raw observations; this work uses Wasserstein loss on empirical quantiles. Both theory and experiments indicate the latter is more suitable.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of quantile tokens is simple yet elegant, supported by solid theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, four model scales, and full ablations; dataset diversity could be further improved.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are tightly integrated with clear writing.
- Value: ⭐⭐⭐⭐ Practical value for text-to-distribution prediction tasks; the quantile token idea is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Wait, There's a Way Out: A Decision Mechanism for Conversational Derailment Prediction](wait_theres_a_way_out_a_decision_mechanism_for_forecasting_conversational_derail.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[ACL 2026\] Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion](leveraging_pretrained_language_models_as_energy_functions_for_glauber_dynamics_t.md)
- [\[ACL 2026\] Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing](overcoming_copyright_barriers_in_corpus_distribution_through_non-reversible_hash.md)
- [\[ACL 2026\] Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling](hot-start_from_pixels_low-resolution_visual_tokens_for_chinese_language_modeling.md)

</div>

<!-- RELATED:END -->
