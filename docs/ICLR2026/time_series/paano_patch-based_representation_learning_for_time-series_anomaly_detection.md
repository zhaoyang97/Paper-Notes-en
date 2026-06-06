---
title: >-
  [Paper Note] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection
description: >-
  [ICLR 2026][Time Series][Time series anomaly detection] This paper proposes PaAno, a lightweight patch-level representation learning method for time-series anomaly detection. It employs a 1D-CNN encoder trained with trip…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Time series anomaly detection"
  - "patch representation learning"
  - "lightweight CNN"
  - "memory bank"
  - "metric learning"
date: 2026-05-08
content_hash: 89998455ce131f0c
---

# PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection

**Conference**: ICLR 2026
**arXiv**: [2602.01359](https://arxiv.org/abs/2602.01359)  
**Code**: [Available](https://github.com/jinnnju/PaAno)  
**Area**: Time Series
**Keywords**: Time series anomaly detection, patch representation learning, lightweight CNN, memory bank, metric learning

## TL;DR

This paper proposes PaAno, a lightweight patch-level representation learning method for time-series anomaly detection. It employs a 1D-CNN encoder trained with triplet loss and pretext loss to learn a patch embedding space, and computes anomaly scores by measuring the distance between query patches and normal patches stored in a memory bank. PaAno achieves comprehensive state-of-the-art performance on the TSB-AD benchmark while requiring only 0.3M parameters and seconds of inference time.

## Background & Motivation

Time-series anomaly detection is critical in industrial monitoring, financial trading, and healthcare. Transformer-based and foundation model approaches (e.g., AnomalyTransformer, MOMENT, TimesFM) have increasingly dominated the field; however:

**Limitations of Prior Work**:

**Illusion of progress**: Sarfraz et al. and Liu & Paparrizos demonstrate that under rigorous evaluation protocols (removing point adjustment and avoiding threshold tuning), complex large models do not substantially outperform simple baselines.

**High computational cost**: Transformer and foundation models carry large parameter counts (0.5M–210M) and long runtimes (tens to thousands of seconds), making them unsuitable for real-time and resource-constrained settings.

**Diluted locality**: Global self-attention is insensitive to local context, yet anomaly detection fundamentally relies on local temporal patterns within short intervals.

**Positioning of PaAno**:

- Adopts a **representation learning** paradigm (rather than prediction or reconstruction), which is relatively underexplored in anomaly detection.
- Introduces a locality inductive bias, drawing on the success of patch-based methods such as PatchCore in visual anomaly detection.
- Core Idea: Normal time series exhibit repetitive local patterns; anomalies break these short-range regularities.

## Method

### Overall Architecture

PaAno consists of two phases—training and inference:

**Training phase**: (1) Extract overlapping fixed-length patches from the training sequence; (2) map patches to an embedding space via a 1D-CNN encoder; (3) train the encoder with triplet loss and pretext loss; (4) construct a memory bank of normal patch embeddings.

**Inference phase**: (1) Extract patches surrounding the query time step; (2) compute the distance between each patch and its nearest neighbor in the memory bank as the patch-level anomaly score; (3) aggregate scores into a time-step-level anomaly score.

### Key Designs

**1. Patch Extraction and Normalization**

Given a training sequence $\mathbf{X} = (\mathbf{x}_1, \ldots, \mathbf{x}_N)$, a sliding window of size $w$ and stride 1 is applied to extract the patch set $\mathcal{P} = \{\mathbf{p}_t\}_{t=1}^{N-w+1}$. Each patch undergoes instance normalization (zero mean, unit variance) to improve robustness against distribution shift and regime changes.

**2. Model Architecture (Three Components)**

- **Patch encoder $f_\theta$**: 4-layer 1D-CNN followed by global average pooling, producing a 64-dimensional embedding $\mathbf{h}$.
- **Projection head $g_\theta$**: Two-layer MLP (256-dimensional hidden size) projecting $\mathbf{h}$ to $\mathbf{z}$ for metric learning.
- **Classification head $c_\theta$**: Single-layer MLP predicting whether two patches are temporally consecutive.

Only the encoder $f_\theta$ is retained for inference after training.

**3. Triplet Loss (Primary Loss)**

For each anchor patch $\mathbf{p}_i$:

- **Positive sample** $\mathbf{p}_i^+$: the anchor patch randomly shifted within $r$ steps (non-zero shift), ensuring temporal pattern similarity.
- **Negative sample** $\mathbf{p}_i^-$: the patch in the minibatch with the largest cosine distance from $\mathbf{p}_i$ (hardest negative strategy).

$$\mathcal{L}_{\text{triplet}} = \frac{1}{M} \sum_{i=1}^{M} \max(0, \text{dist}(\mathbf{z}_i, \mathbf{z}_i^+) - \text{dist}(\mathbf{z}_i, \mathbf{z}_i^-) + \delta)$$

The objective is to keep embeddings of small-shifted patches close (robust to minor temporal offsets) while pushing apart patches with dissimilar patterns (sensitive to anomalies).

**4. Pretext Loss (Auxiliary Loss)**

Inspired by predicting spatial patch relationships in visual anomaly detection, a temporal counterpart is designed to predict whether two patches are temporally consecutive:

$$\mathcal{L}_{\text{pretext}} = \frac{1}{M} \sum_{i=1}^{M} \left[ -\log c_\theta(\mathbf{h}_i, \mathbf{h}_i^{\text{pre}}) - \frac{1}{U} \sum_{j=1}^{U} \log(1 - c_\theta(\mathbf{h}_i, \mathbf{h}_{i,j}^{\text{rand}})) \right]$$

where $\mathbf{p}_i^{\text{pre}}$ is the patch $w$ steps prior (temporally adjacent), and $\mathbf{p}_{i,j}^{\text{rand}}$ are randomly sampled patches. This loss is applied only in the early training phase (linearly decayed to 0 over the first 20 iterations) to accelerate the formation of a structured embedding space.

**5. Memory Bank Construction and Compression**

The full memory bank is $\mathcal{M} = \{f_\theta(\mathbf{p}_t) \mid \mathbf{p}_t \in \mathcal{P}\}$. To reduce storage and computation:

- K-means clustering into $K$ clusters is applied.
- The vector nearest to each cluster centroid is selected as the representative.
- The compressed memory bank $\hat{\mathcal{M}}$ retains only 10% of the original entries.

**6. Anomaly Score Computation**

For the $w$ patches covering query time step $t_*$, the anomaly score of each patch is the mean $k$-nearest-neighbor distance:

$$S(\mathbf{p}_t) = \frac{1}{k} \sum_{i=1}^{k} \text{dist}(f_\theta(\mathbf{p}_t), \mathbf{m}_t^{(i)})$$

The final anomaly score is the average of scores across all patches covering $t_*$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{triplet}} + \lambda \cdot \mathcal{L}_{\text{pretext}}$$

- $\lambda$ is linearly decayed from 1 to 0 over the first 20 iterations.
- AdamW optimizer with weight decay $10^{-4}$.
- Training runs for only 200 iterations with a minibatch size of 512.
- Results are averaged over 10 different random seeds.

## Key Experimental Results

### Main Results

Evaluation is conducted on the TSB-AD benchmark, comprising 530 univariate sequences (TSB-AD-U) and 180 multivariate sequences (TSB-AD-M), with comparisons against 48 baselines.

**Univariate Anomaly Detection (TSB-AD-U)**:

| Method | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Params | Runtime |
|--------|--------|---------|--------|---------|--------|---------|
| **PaAno** | **0.52** | **0.89** | **0.46** | **0.86** | 0.3M | 6.9s |
| KAN-AD | 0.43 | 0.82 | 0.41 | 0.80 | <0.1M | 12.1s |
| (Sub)-PCA | 0.42 | 0.76 | 0.37 | 0.71 | - | 1.5s |
| MOMENT (FT) | 0.39 | 0.76 | 0.30 | 0.69 | 109.6M | 43.6s |
| TimesFM | 0.30 | 0.74 | 0.28 | 0.67 | 203.5M | 83.8s |
| AnomalyTrans. | 0.12 | 0.56 | 0.08 | 0.50 | 4.8M | 48.9s |

**Multivariate Anomaly Detection (TSB-AD-M)**:

| Method | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Params | Runtime |
|--------|--------|---------|--------|---------|--------|---------|
| **PaAno** | **0.43** | **0.79** | **0.38** | **0.76** | 0.3M | 12.8s |
| KAN-AD | 0.41 | 0.75 | 0.38 | 0.73 | <0.1M | 31.9s |
| DeepAnT | 0.31 | 0.76 | 0.32 | 0.73 | <0.1M | 9.5s |
| PCA | 0.31 | 0.74 | 0.31 | 0.70 | - | 0.1s |
| CATCH | 0.30 | 0.73 | 0.24 | 0.67 | 210.8M | 40.1s |

### Ablation Study

Detailed ablations are provided in the appendix; key findings include:

- Triplet loss is the core contribution; removing it leads to a significant drop in VUS-PR.
- Early application of the pretext loss accelerates the structuring of the embedding space.
- Memory bank compression (to 10%) incurs negligible performance degradation.
- The method exhibits high hyperparameter robustness and requires no careful tuning.

### Key Findings

- PaAno achieves first place on all six evaluation metrics for both univariate and multivariate settings.
- Its parameter count of 0.3M is far smaller than Transformer-based methods (4.8M–210M).
- Runtime is 6.9–12.8s, compared to 42–1221s for foundation models.
- Traditional methods such as PCA and KShapeAD perform surprisingly well under rigorous evaluation, corroborating the "illusion of progress" phenomenon.
- Transformer-based methods such as AnomalyTransformer and DCdetector rank very low (20+) under strict evaluation protocols.

## Highlights & Insights

1. **Small is beautiful**: A 0.3M-parameter 1D-CNN outperforms million-parameter Transformers and foundation models, fundamentally challenging the "bigger is better" assumption.
2. **Locality first**: Anomaly detection requires fine-grained local awareness; global attention mechanisms dilute the critical signal.
3. **Successful transfer from vision to time series**: The patch-based representation learning and memory bank paradigm (analogous to PatchCore) proves equally effective in the temporal domain.
4. **Importance of rigorous evaluation**: Removing point adjustment and threshold tuning substantially reorders method rankings.

## Limitations & Future Work

- The semi-supervised setting assumes entirely normal training data, precluding scenarios with a small number of labeled anomalies.
- The patch size $w$ is a fixed hyperparameter; different anomaly types may require different window sizes.
- The memory bank approach may face storage challenges on very long training sequences, despite 10% compression.
- Evaluation is limited to anomaly detection; whether the patch embeddings transfer to tasks such as classification or forecasting remains unexplored.

## Related Work & Insights

- Inspired by visual anomaly detection methods (PaDiM, PatchCore, SPADE), PaAno transfers the patch-level representation and memory bank paradigm to the temporal domain.
- Complementary to prediction- and reconstruction-based approaches: PaAno follows a **representation learning** paradigm and does not require reconstructing the original signal.
- Insight: Simple yet targeted methods (local patches + metric learning) may outperform general-purpose large models on specific tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Effective transfer of patch representation learning from vision to time series; clean and well-motivated design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (48 baselines, 710 time series, rigorous evaluation protocol, multiple metrics)
- Writing Quality: ⭐⭐⭐⭐ (Well-motivated, with in-depth discussion of evaluation issues)
- Value: ⭐⭐⭐⭐⭐ (Achieves SOTA at minimal cost; highly practical and a strong candidate for industrial deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Contextual and Seasonal LSTMs for Time Series Anomaly Detection](contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICML 2026\] IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection](../../ICML2026/time_series/impact_influence_modeling_for_open-set_time_series_anomaly_detection.md)
- [\[ICML 2026\] AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection](../../ICML2026/time_series/anomseer_reinforcing_multimodal_llms_to_reason_for_time-series_anomaly_detection.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)

</div>

<!-- RELATED:END -->
