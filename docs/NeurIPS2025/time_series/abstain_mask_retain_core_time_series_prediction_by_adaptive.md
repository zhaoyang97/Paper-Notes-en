---
title: >-
  [Paper Note] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency
description: >-
  [NeurIPS 2025 (Spotlight)][Time Series][time series forecasting] This paper reveals a counter-intuitive phenomenon in time series forecasting — that appropriately truncating historical inputs can improve prediction accuracy (termed the redundant feature learning problem) — and proposes AMRC based on information bottleneck theory. AMRC suppresses redundant feature learning via adaptive masking loss and representation consistency constraints, serving as a model-agnostic training framework that consistently improves performance across diverse architectures.
tags:
  - NeurIPS 2025 (Spotlight)
  - Time Series
  - time series forecasting
  - information bottleneck
  - adaptive masking
  - representation consistency
  - redundant features
date: 2026-05-08
content_hash: af0c721b3f9f8645
---

# Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2510.19980](https://arxiv.org/abs/2510.19980)
**Code**: [GitHub](https://github.com/MazelTovy/AMRC)
**Area**: Time Series Forecasting / Information Bottleneck
**Keywords**: time series forecasting, information bottleneck, adaptive masking, representation consistency, redundant features

## TL;DR
This paper reveals a counter-intuitive phenomenon in time series forecasting — that appropriately truncating historical inputs can improve prediction accuracy (termed the redundant feature learning problem) — and proposes AMRC based on information bottleneck theory. AMRC suppresses redundant feature learning via adaptive masking loss and representation consistency constraints, serving as a model-agnostic training framework that consistently improves performance across diverse architectures.

## Background & Motivation

**Background**: Mainstream time series forecasting models follow the "long-sequence information gain hypothesis," assuming that longer historical inputs provide richer temporal dependency information.

**Limitations of Prior Work**: Experiments reveal a counter-intuitive phenomenon: appropriately truncating the input sequence (e.g., masking the first $k$ time steps) reduces prediction MSE on over 50% of samples. This indicates that existing models learn substantial redundant features (noise and irrelevant fluctuations) during training — features that not only fail to improve performance but actively interfere with the extraction of effective signals.

## Core Problem
1. Why do longer historical sequences not always help, and how are existing models affected by redundant features?
2. How can redundant feature learning be automatically suppressed without manual input truncation?
3. Can a model-agnostic training framework be designed to address this issue?

## Method

### Overall Architecture
AMRC consists of two core components:
1. **Adaptive Masking Loss (AML)**: Dynamically identifies highly discriminative temporal segments during training, directing gradient optimization toward the most informative input regions.
2. **Embedding Similarity Penalty (ESP)**: Constrains the representation mapping among inputs, labels, and predictions to remain mutually consistent.

The two components are combined into the overall training objective: $\mathcal{L} = \mathcal{L}_{task} + \alpha \cdot \mathcal{L}_{AML} + \lambda \cdot \mathcal{L}_{ESP}$

### Key Designs

1. **Adaptive Masking Loss (AML)**:

   - **Function**: Dynamically searches for the optimal masking length for each batch during training, guiding the model representation to disregard redundant temporal segments.
   - **Mechanism**: For an input of length $L$, $m$ masking lengths $\{k_s\}_{s=1}^m \sim \text{Uniform}\{1,...,L\}$ are randomly sampled; the prediction loss $\ell_s = \mathcal{L}(f_\theta(\mathcal{M}_{k_s}(X)), Y)$ is computed for each mask; and the mask with the greatest gain is selected via $s^* = \arg\max_s (\ell - \ell_{s})$. The L2 distance between the original representation $Z$ and the optimal masked representation $\tilde{Z}_{s^*}$ is then minimized: $\mathcal{L}_{AML} = \beta \cdot \frac{1}{D_1 \times D_2} \|Z - \tilde{Z}_{s^*}\|^2$, where $\beta = \max(0, (\ell - \ell_{s^*})/\ell)$ activates only when a superior mask is found.
   - **Design Motivation**: Rather than simply truncating inputs, the model is guided to "forget" redundant information directly in representation space.

2. **Embedding Similarity Penalty (ESP)**:

   - **Function**: Constrains the geometric structure of the embedding space to remain consistent with the output space.
   - **Mechanism**: For pairs of samples within a batch, embedding distances $\Delta^E_{ij}$ and label distances $\Delta^O_{ij}$ are computed, and inconsistencies between them are penalized: $\mathcal{L}_{ESP} = \frac{1}{n^2} \sum_{i,j} |\Delta^E_{ij} - \Delta^O_{ij}|_+$
   - **Design Motivation**: t-SNE visualization reveals abnormal concentration of model embeddings (representation collapse) that is misaligned with the label distribution, indicating the encoding of task-irrelevant redundant features.

3. **Theoretical Foundation (Information Bottleneck)**:
   Starting from IB theory, the objective is to maximize $I(Z;Y) - \beta I(Z;X)$. Existing models primarily optimize $I(Z;Y)$, whereas AMRC explicitly minimizes the redundant component of $I(Z;X)$ through AML, providing a novel optimization pathway.

## Key Experimental Results

### Main Results: Prevalence of Redundant Feature Learning

| Dataset | Baseline | Original MSE | Best-Mask MSE | Improvement (% samples) |
|---------|----------|-------------|---------------|--------------------------|
| ETTh1 | iTransformer | 0.413 | 0.289 | 60.07% |
| Weather | iTransformer | 0.209 | 0.170 | 80.26% |
| ETTh2 | TSMixer | 0.324 | 0.289 | 42.13% |
| Solar-Energy | PatchTST | 0.374 | 0.344 | 51.66% |

### AMRC Performance Gains

| Model | Dataset | Original MSE | +AMRC MSE | Gain |
|-------|---------|-------------|-----------|------|
| SOFTS | ETTh1 | 0.408 | 0.389 | −4.7% |
| SOFTS | ETTm2 | 0.210 | 0.198 | −5.7% |
| iTransformer | Electricity | 0.176 | 0.163 | −7.4% |
| iTransformer | Weather | 0.209 | 0.201 | −3.8% |
| TimeMixer | ETTm1 | 0.466 | 0.447 | −4.1% |
| PatchTST | ETTm2 | 0.211 | 0.196 | −7.1% |

- Redundant feature learning is **architecture-agnostic**: it is consistently observed across MLP-based (TSMixer), Transformer-based (iTransformer, PatchTST), and attention-based (SOFTS) architectures.
- As a plug-in, AMRC yields significant improvements across 5 baseline models and 7 datasets.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Full AMRC | Best performance; joint optimization of AML + ESP |
| Only AML | Effective but inferior to the combined approach, indicating that representation consistency also matters |
| Only ESP | Improvements observed in isolation, validating the reality of representation collapse |
| Baseline | No additional loss terms; subject to redundant feature learning |

## Highlights & Insights
- **Counter-intuitive findings are highly illuminating**: A simple truncation experiment exposes a fundamental deficiency in mainstream time series models.
- **Novel application of information bottleneck**: The typically overlooked $I(Z;X)$ minimization path in IB theory is instantiated as a concrete and actionable training strategy.
- **Model-agnostic framework**: As a training-time plug-in, AMRC is applicable to any time series forecasting model.
- **Novel representation consistency constraint**: By enforcing consistency between pairwise distance distributions of predictions and labels, the method indirectly improves model generalization.

## Limitations & Future Work
- The optimal masking length search increases training overhead, requiring multiple masking lengths to be evaluated per batch.
- The definition of redundant features relies on MSE improvement and may behave differently under alternative forecasting objectives.
- Validation is limited to time series forecasting; the redundancy suppression idea may generalize to broader sequence modeling tasks.
- The hyperparameters $\lambda$ and $\alpha$ for ESP require tuning.

## Related Work & Insights
- **vs. TS2Vec / TS-CoT**: These methods improve representations via contrastive learning but do not explicitly address redundant feature learning.
- **vs. DECL**: Employs denoising contrastive learning, but denoising and redundant feature suppression operate at different levels.
- **vs. VIB (Alemi et al.)**: VIB optimizes the IB objective through variational inference but lacks customized design for temporal-dimension redundancy in time series.

The observation that "more data is not always better" warrants investigation in other domains (NLP, CV). The adaptive masking idea may inspire attention mechanism design in other tasks, and the representation consistency constraint may find application in multimodal alignment settings.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Both the discovery of redundant feature learning and the IB-theoretic solution are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dataset, multi-architecture evaluation with comprehensive ablation studies and visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem-driven narrative structure is clear, progressing naturally from discovery → analysis → solution.
- **Value**: ⭐⭐⭐⭐ — Strong practical value as a model-agnostic plug-in training framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](../../AAAI2026/time_series/mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [\[NeurIPS 2025\] WaLRUS: Wavelets for Long-range Representation Using SSMs](walrus_wavelets_for_long-range_representation_using_ssms.md)

</div>

<!-- RELATED:END -->
