---
title: >-
  [Paper Note] Finding Time Series Anomalies using Granular-ball Vector Data Description
description: >-
  [AAAI 2026][Time Series] This paper proposes the Granular-ball One-Class Network (GBOC), which adaptively constructs density-guided Granular-ball Vector Data Descriptions (GVDD) in the latent space. By replacing traditio…
tags:
  - "AAAI 2026"
  - "Time Series"
date: 2026-05-08
content_hash: 8c2bb15b6796925c
---

# Finding Time Series Anomalies using Granular-ball Vector Data Description

**Conference**: AAAI 2026
**arXiv**: [2511.12147](https://arxiv.org/abs/2511.12147)
**Authors**: Lifeng Shen, Liang Peng, Ruiwen Liu, Shuyin Xia, Yi Liu
**Code**: [https://github.com/notshine/GBOC](https://github.com/notshine/GBOC)

## TL;DR

This paper proposes the Granular-ball One-Class Network (GBOC), which adaptively constructs density-guided Granular-ball Vector Data Descriptions (GVDD) in the latent space. By replacing traditional clustering or single-hypersphere assumptions, GBOC enables flexible modeling of normal time series behavior and robust anomaly detection.

## Background & Motivation

Time series anomaly detection is critical in complex cyber-physical systems such as industrial monitoring, data centers, and smart factories. Existing methods fall into three main categories:

**Nearest-neighbor methods** (e.g., KNN): Rely on local density or proximity. In group anomaly scenarios, a cluster of anomalous points may mutually serve as each other's "neighbors," leading to misclassification as normal; they also fail to capture global or temporal structure.

**Clustering methods** (e.g., KShapeAD, KMeansAD): Require a predefined number of clusters and assume normal data forms discrete, well-separated cluster structures. However, representations derived from sliding windows over time series typically exhibit **structural continuity** with smooth transitions rather than discrete boundaries, making rigid cluster partitioning ill-suited to such data.

**One-class classification methods** (e.g., DeepSVDD, SVDD): Model normal data as a single hypersphere. This oversimplified assumption fails to capture the diversity of multi-modal normal patterns.

**Memory-based methods** (e.g., MEMTO): Depend on the quality and representativeness of stored prototypes; performance degrades significantly when prototype coverage is insufficient.

**Core Motivation**: The latent representations of time series possess a continuous topological structure, and the rigid assumptions of traditional methods—predefined cluster counts, discrete boundaries, and single hyperspheres—are poorly suited to such data. An adaptive, parameter-free approach capable of flexibly modeling complex data distributions is therefore needed.

## Method

### Overall Architecture

GBOC consists of four steps: (1) time series encoding → (2) granular-ball construction → (3) granular-ball representation optimization → (4) anomaly inference.

### 1. Time Series Encoding

A sliding window segments the input time series into overlapping windows $\{\mathbf{w}_1, \mathbf{w}_2, \ldots\}$. Each window is mapped to a $d'$-dimensional latent representation $\mathbf{z}_i = f_\theta(\mathbf{w}_i)$ via a three-layer LSTM encoder. The final hidden states of each layer are concatenated to capture multi-level temporal dependencies. The encoder can be replaced with alternative architectures such as Transformers.

### 2. Granular-ball Vector Data Description (GVDD)

Granular-ball construction is performed in the latent space:

- **Initialization**: Data is coarsely partitioned into initial granular-balls using K-Means with $k_0 = \lfloor\sqrt{n}\rfloor$.
- **Recursive splitting**: Any granular-ball containing $\geq s_{\min}=8$ points is split into two sub-balls using 2-Means.
- **Splitting criterion**: Based on the granular-ball distribution measure $DM = s / |GB|$ (where $s$ is the sum of distances from all points to the center); a split is accepted only if the weighted child $DM_w < DM$.
- Recursion continues until no split yields a quality improvement.

Each granular-ball is defined by a center $\mathbf{c}$ (mean) and a radius $\mathbf{r}$ (maximum Euclidean distance), naturally occupying a granularity level between individual samples and global clusters, thereby preserving the local topological structure of the data.

### 3. Granular-ball Representation Optimization

**Low-quality granular-ball pruning**: A dynamic threshold $r_{th} = \mu \cdot \max\{\text{median}(r), \text{mean}(r)\}$ ($\mu=2$) is applied to remove overly diffuse or noisy granular-balls, retaining only structurally compact, high-confidence ones.

**Joint optimization objective**:

- **Granular-ball alignment loss** $\mathcal{L}_{gb}$: Pulls each sample toward its nearest granular-ball center to enhance compactness and discriminability in the latent space:
$$\mathcal{L}_{gb} = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{z}_i - \mathbf{c}_{s(i)}\|_2^2$$

- **Reconstruction loss** $\mathcal{L}_{rec}$: Preserves temporal fidelity via a lightweight MLP decoder to prevent representation collapse:
$$\mathcal{L}_{rec} = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{x}_i - g_\phi(\mathbf{z}_i)\|_2^2$$

- **Total loss**: $\mathcal{L} = \lambda \cdot \mathcal{L}_{rec} + (1-\lambda) \cdot \mathcal{L}_{gb}$, with $\lambda=0.5$.

### 4. Anomaly Inference

A test sample is first encoded, and its anomaly score is computed as the Euclidean distance to the nearest granular-ball center:
$$\text{Score}(\mathbf{z}) = \min_{\mathbf{c} \in \mathcal{C}} \|\mathbf{z} - \mathbf{c}\|_2$$

An empirical $3\sigma$ rule is used to determine the unsupervised threshold: a sample is flagged as anomalous if its score exceeds the evaluation set mean by more than three standard deviations.

## Experiments

### Experimental Setup

- **Datasets**: 7 univariate + 5 multivariate datasets spanning industrial systems (SMD), web services (IOPS, WSD), medical (UCR, LTDB, SVDB), environmental (TAO, SMAP, MSL), and synthetic (YAHOO) domains.
- **Baselines**: 14 methods, including non-deep-learning (PCA, KNN, IForest, MatrixProfile, KShapeAD) and deep learning (CNN, LSTMAD, TranAD, USAD, TimesNet, AnomalyTransformer, DeepSVDD, THOC, MEMTO).
- **Metrics**: VUS-PR, VUS-ROC, Affiliation-F1.
- **Hardware**: NVIDIA RTX 4090 GPU, 128 GB RAM.

### Main Results

**Table 1: Univariate Anomaly Detection (VUS-PR)**

| Method | SMD | TAO | YAHOO | UCR | IOPS | WSD |
|--------|-----|-----|-------|-----|------|-----|
| KNN | 0.766 | 0.940 | 0.281 | 0.856 | 0.222 | 0.011 |
| DeepSVDD | 0.812 | 0.945 | 0.967 | 0.996 | 0.236 | 0.404 |
| TimesNet | 0.680 | 0.932 | 0.577 | 0.023 | 0.184 | 0.354 |
| THOC | 0.272 | 0.938 | 0.048 | 0.513 | 0.407 | 0.025 |
| MEMTO | 0.314 | 0.932 | 0.074 | 0.630 | 0.180 | 0.021 |
| **GBOC** | **0.831** | **0.978** | **0.991** | **0.996** | **0.604** | **0.963** |

GBOC achieves the best VUS-PR on all six univariate datasets, with particularly notable margins on YAHOO (0.991 vs. runner-up 0.967), IOPS (0.604 vs. 0.407), and WSD (0.963 vs. 0.404).

**Table 3: Robustness to Drift and Noise (VUS-PR)**

| Method | I: Clean | II: Drift | III: Noise | IV: Drift+Noise |
|--------|----------|-----------|------------|-----------------|
| KShapeAD | 1.000 | 0.982 | 0.802 | 0.624 |
| DeepSVDD | 0.824 | 0.153 | 0.833 | 0.893 |
| MEMTO | 0.782 | 0.028 | 0.121 | 0.031 |
| **GBOC** | **1.000** | **0.977** | **0.952** | **0.921** |

GBOC achieves the best performance across all four scenarios. Under the most challenging Type IV condition (drift + noise), GBOC (0.921) substantially outperforms KShapeAD (0.624) and MEMTO (0.031), demonstrating strong robustness.

### Ablation Study

**Table 4: Granular-ball Component Ablation (VUS-PR)**

| GBC | Pruning | SMD | IOPS | UCR | YAHOO |
|-----|---------|-----|------|-----|-------|
| ✗ (K-Means) | ✗ | 0.755 | 0.554 | 0.921 | 0.823 |
| ✓ | ✗ | 0.781 | 0.566 | 0.972 | 0.795 |
| ✓ | ✓ | **0.831** | **0.604** | **0.996** | **0.991** |

- Removing granular-ball construction (replaced by K-Means) leads to substantial performance degradation, demonstrating that adaptive density-aware granular-ball construction is superior to fixed cluster structures.
- Removing pruning retains noisy regions and also degrades performance.

**Table 5: Loss Function Ablation**

Using $\mathcal{L}_{rec}$ or $\mathcal{L}_{gb}$ alone is consistently inferior to their combination. The effect is most pronounced on the noisier YAHOO dataset (joint: 0.991 vs. individual: 0.869 / 0.701).

## Highlights & Insights

- **First application of granular-ball computing to one-class anomaly detection**: GVDD is proposed to integrate granular-ball computing into one-class methods, filling a gap in the application of granular-ball computing to time series anomaly detection.
- **Adaptive, parameter-free modeling**: No predefined number of clusters or neighbors is required; granular-balls automatically split and are pruned according to data density, naturally accommodating continuous temporal structures.
- **Strong robustness under noise and drift**: By focusing on high-density, compact regions, GBOC effectively filters out noisy and low-quality areas, maintaining high performance across four scenarios of varying complexity.
- **Efficient inference**: The number of granular-balls is far smaller than the number of training samples; anomaly scoring requires only computing the distance to the nearest centroid.

## Limitations & Future Work

- The LSTM encoder has limited capacity to model extremely long time series; although the paper mentions Transformer as an alternative, no comparative experiments are provided.
- Granular-ball construction relies on K-Means initialization and recursive 2-Means splitting; sensitivity to initialization is not thoroughly analyzed.
- The pruning threshold $\mu=2$ and minimum support $s_{\min}=8$ are empirically set and may not be optimal across datasets of different scales.
- Evaluation is conducted exclusively in unsupervised settings; semi-supervised or few-shot anomaly scenarios are not explored.
- At inference time, all granular-ball centers must be traversed to find the minimum distance, which may impact real-time performance when the number of granular-balls is large.

## Related Work & Insights

- **Nearest-neighbor methods**: KNN (SubKNN), LOF — rely on local density, susceptible to group anomalies.
- **Clustering methods**: KShapeAD, KMeansAD, SAND — require predefined cluster counts and assume discrete patterns.
- **One-class classification**: SVDD, DeepSVDD — single hypersphere, unable to capture multi-modal normality.
- **Memory-augmented methods**: MEMTO — fixed memory structure, degrades when prototype coverage is incomplete.
- **Hierarchical clustering**: THOC — multi-scale vector data description, but still constrained by fixed structure.
- **Reconstruction/prediction methods**: USAD, LSTMAD, TranAD — prone to failure in noisy or non-stationary environments.
- **Granular-ball computing (GBC)**: Previously applied to clustering acceleration, density clustering, point cloud registration, and intent classification; this work introduces it to time series anomaly detection for the first time.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](../../ICLR2026/time_series/rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[AAAI 2026\] IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](../../ICLR2026/time_series/gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](../../NeurIPS2025/time_series/synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
