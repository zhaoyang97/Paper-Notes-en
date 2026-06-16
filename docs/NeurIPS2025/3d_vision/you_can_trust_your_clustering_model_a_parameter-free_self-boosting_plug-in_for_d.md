---
title: >-
  [Paper Note] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering
description: >-
  [NeurIPS 2025][3D Vision][Deep Clustering] This paper proposes DCBoost, a plug-and-play module requiring no additional hyperparameters…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Deep Clustering"
  - "Self-Boosting Plug-in"
  - "Local Structure"
  - "Global Structure Optimization"
  - "Parameter-free"
date: 2026-05-08
content_hash: 3bd0b0f4e5c2e781
---

# You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering

**Conference**: NeurIPS 2025
**arXiv**: [2511.21193](https://arxiv.org/abs/2511.21193)  
**Code**: [Available](https://github.com/l-h-y168/DCBoost)  
**Area**: Deep Clustering / Unsupervised Learning
**Keywords**: Deep Clustering, Self-Boosting Plug-in, Local Structure, Global Structure Optimization, Parameter-free

## TL;DR

This paper proposes DCBoost, a plug-and-play module requiring no additional hyperparameters, which selects high-confidence samples via adaptive k-NN and leverages reliable local structural information to guide global feature space optimization, substantially improving the performance of existing deep clustering models.

## Background & Motivation

Deep clustering aims to discover intrinsic structure in unlabeled data using deep networks. In recent years, integrating self-supervised learning (contrastive and non-contrastive) with clustering has achieved significant progress, with representative methods including ProPos, CC, and SCAN.

However, the authors identify a commonly overlooked issue shared across existing methods: **a substantial discrepancy between global and local structure quality**.

- **High local structure quality**: k-NN accuracy is often high (sometimes exceeding clustering accuracy), indicating that semantically similar samples are indeed locally grouped together in the feature space.
- **Poor global structure quality**: low intra-class similarity, high inter-class similarity, and extremely low Silhouette Coefficients (e.g., ProPos achieves only 0.10 on CIFAR-10), indicating blurred decision boundaries and severe feature space entanglement across classes.

t-SNE visualizations further confirm this: while coarse cluster structures are visible, substantial inter-class overlap persists. This observation motivates the authors to ask: **can reliable local structure be leveraged to repair unreliable global structure?**

## Method

### Overall Architecture

The core mechanism of DCBoost is "using local structure to guide global structure":

1. An arbitrary pre-trained deep clustering model serves as the **target network** $f_t(\cdot)$; a copy is maintained as the **online network** $f_o(\cdot)$, augmented with a randomly initialized nonlinear predictor $g(\cdot)$.
2. Two weak augmentations are applied to each sample to generate two views, which are passed through the online and target networks respectively to obtain L2-normalized features $z^o$ and $z^t$.
3. Within each batch, high-confidence samples are selected via adaptive k-NN (local structure mining).
4. These high-confidence samples are used to construct a discriminative loss that fine-tunes the model (local-guided global optimization).
5. At the end of each epoch, k-means is applied to all sample outputs from the target network to obtain pseudo-labels.
6. The target network parameters are updated from the online network via EMA (Exponential Moving Average).

### Key Designs

**Adaptive k-NN High-Confidence Sample Selection**:

The core idea is to select samples whose $k$ nearest neighbors all share the same pseudo-label. The key challenge lies in choosing $k$: too small a value admits noisy samples, while too large a value leaves too few high-confidence samples.

The authors propose a parameter-free adaptive selection strategy:

- Let the candidate set be $\mathbb{K} = \{1, 2, \ldots, 50\}$.
- For each candidate $k_s$, compute a score: $\text{score}_{k_s} = k_s \times \frac{n_s}{n_B}$, where $n_s$ is the number of high-confidence samples satisfying the criterion and $n_B$ is the batch size.
- The $k_s$ with the highest score is selected as the k value for the current batch.

This score can be geometrically interpreted as the area of a rectangle with width $k$ and height equal to the high-confidence sample ratio, automatically balancing quality and quantity. Early in training, $k$ tends to be small (admitting more samples); later, $k$ increases (focusing on higher-quality samples).

### Loss & Training

The total loss consists of three terms: $L = L_{pos} + L_{neg} + L_{ins}$

**1. Positive sample loss $L_{pos}$ (intra-class compactness)**:

$$L_{pos} = \frac{1}{2c_B} \sum_{c=1}^{c_B} \sum_{i,j \in \mathbb{X}_c} w_c d_{ij}^2$$

This pulls together high-confidence sample pairs sharing the same pseudo-label. Here $w_c$ is a class-balancing weight to mitigate class imbalance among high-confidence samples:

$$w_c = \frac{1}{\|\sum_{x \in \mathbb{X}_c} z^o\|_2 \cdot \|\sum_{x \in \mathbb{X}_c} z^t\|_2}$$

Classes with more samples yield aggregated features with larger norms, naturally receiving lower weights, thereby preventing dominant classes from overwhelming training.

**2. Negative sample loss $L_{neg}$ (inter-class separability)**:

$$L_{neg} = -\sum_{c_1 \neq c_2} \|v_{c_1}^o - v_{c_2}^t\|_2^2$$

Inter-class separation is enhanced by maximizing the Euclidean distance between prototypes of different classes, where prototypes are computed as normalized aggregated features of high-confidence samples.

**3. Instance consistency loss $L_{ins}$**:

$$L_{ins} = \|g(f_o(\mathcal{T}^1(x)) + \sigma\varepsilon) - f_t(\mathcal{T}^2(x))\|_2^2$$

Applied to all samples (not only high-confidence ones) to maintain instance-level consistency and prevent representation collapse. A small amount of Gaussian noise ($\sigma=0.001$) is added to enhance robustness.

## Key Experimental Results

### Main Results

Validated on 5 benchmark datasets with 6 baseline models, DCBoost consistently improves all baselines as a plug-in:

| Method | CIFAR-10 ACC | CIFAR-20 ACC | STL-10 ACC | ImageNet-10 ACC | ImageNet-Dogs ACC | Avg. Gain |
|--------|-------------|-------------|-----------|----------------|------------------|-----------|
| CC | 85.2 | 41.7 | 80.0 | 89.9 | 69.6 | - |
| CC+DCBoost | 88.2 | 48.9 | 80.8 | 90.7 | 70.6 | +3.1% |
| BYOL | 87.5 | 52.3 | 86.1 | 94.7 | 72.9 | - |
| BYOL+DCBoost | 91.5 | 54.7 | 90.2 | 95.7 | 77.1 | +4.6% |
| ProPos | 94.4 | 61.6 | 91.6 | 95.8 | 76.9 | - |
| ProPos+DCBoost | **96.0** | **63.9** | **93.6** | **97.1** | **79.7** | +3.0% |
| CDC | 94.7 | 61.6 | 93.0 | 97.3 | 79.2 | - |
| CDC+DCBoost | 95.1 | 62.7 | 93.4 | 97.3 | 79.7 | +0.8% |

Key findings: weaker baselines benefit more (BYOL +4.6%); the strong baseline ProPos still gains +3.0%; the Silhouette Coefficient improves from 0.10 to 0.74 (more than 7×).

### Ablation Study

Ablation results on top of ProPos (averaged across CIFAR-10 / CIFAR-20 / STL-10):

| Adaptive Selection | $L_{ins}$ | $L_{pos}$ | $L_{neg}$ | CIFAR-10 ACC | CIFAR-20 ACC | STL-10 ACC | Avg. |
|-------------------|-----------|-----------|-----------|-------------|-------------|-----------|------|
| - | - | - | - | 94.4 | 61.6 | 91.6 | 77.2 |
| ✗ | ✓ | ✓ | ✗ | 94.2 | 57.7 | 90.6 | 75.7 |
| ✓ | ✓ | ✓ | ✗ | 95.6 | 61.8 | 92.2 | 78.7 |
| ✓ | ✓ | ✗ | ✓ | 94.4 | 59.1 | 92.0 | 76.5 |
| ✓ | ✗ | ✓ | ✓ | 91.1 | 62.5 | 20.5 | 51.9 |
| ✗ | ✓ | ✓ | ✓ | 94.9 | 62.5 | 92.0 | 78.3 |
| ✓ | ✓ | ✓ | ✓ | **96.0** | **63.9** | **93.6** | **80.4** |

### Key Findings

1. **All loss terms are indispensable**: removing $L_{ins}$ causes STL-10 ACC to collapse from 93.6% to 20.5%, demonstrating that instance consistency is critical for preventing collapse; removing $L_{pos}$ yields virtually no improvement; removing $L_{neg}$ also leads to a notable decline.
2. **Adaptive selection is crucial**: applying all losses without selection reduces average performance from 80.4% to 78.3%; with selection, even incomplete loss combinations yield significant gains (75.7% → 78.7%).
3. **Class-balancing weights are effective**: no weights ($w_0$) performs worst; weights excluded from gradient computation ($w_1$) provide a substantial improvement; the full formulation ($w_{ours}$) achieves the best performance.
4. **Effective on CLIP-based models**: consistent improvements are observed on SIC and TAC (e.g., TAC on CIFAR-20 improves from 61.5% to 62.9%).
5. **Outperforms the comparable method CoNR**: DCBoost achieves larger and more consistent gains than CoNR, as CoNR exploits only small-range positive pairs whereas DCBoost makes fuller use of local structural information.
6. **Adaptive k selection outperforms manual tuning**: the adaptive method achieves 96.0% on CIFAR-10 (vs. 95.9% for the manually optimal k=30), while eliminating the risk of performance degradation from a poorly chosen k.

## Highlights & Insights

- **Insightful observation**: this work is the first to explicitly identify and formalize the overlooked phenomenon of "reliable local structure, unreliable global structure" in deep clustering, and translates it into a practical improvement strategy.
- **Elegant design**: no new hyperparameters are introduced; the adaptive k-NN scoring function (k × ratio) is concise and effective, with clear geometric intuition.
- **Strong generality**: applicable to both representation-based and clustering-head-based deep clustering methods, and even effective for CLIP-based approaches.
- **Zero additional overhead**: k-NN search is performed only within mini-batches, avoiding the cost of global k-NN computation.

## Limitations & Future Work

1. **Assumes the number of clusters is known**: in practical settings, the number of clusters is typically unknown and requires additional estimation mechanisms.
2. **Limited robustness to class imbalance**: although balancing weights are introduced, highly imbalanced or non-uniform data distributions may still pose challenges.
3. **Scalability to large datasets**: validation is primarily conducted on medium-scale datasets (up to 100K samples); performance on larger-scale data remains to be verified.
4. **Weak augmentations only**: both views employ weak augmentations; whether incorporating strong or mixed augmentation strategies could yield further improvements warrants exploration.
5. **Boundary effects of high-confidence sample selection**: samples near class boundaries are consistently excluded, potentially discarding useful hard sample information.

## Related Work & Insights

- **ProPos** (TPAMI 2023): one of the current state-of-the-art deep clustering methods, learning representations via prototype scattering and positive sampling; the primary enhancement target of this work.
- **BYOL** (NeurIPS 2020): a non-contrastive self-supervised learning method that updates the target network via EMA; DCBoost adopts a similar dual-network architecture.
- **CoNR** (shares motivation with this work): enhances intra-class compactness using contextual cues, but achieves smaller gains than DCBoost.
- **Inspiration**: the "local → global" paradigm of this method is generalizable to other unsupervised/semi-supervised tasks, such as unsupervised domain adaptation, where locally reliable features but globally misaligned distributions are also commonly observed.

## Rating

- **Novelty**: ★★★★☆ — The observation (local vs. global structural discrepancy) is novel, and the method design is concise and effective.
- **Technical Depth**: ★★★☆☆ — The technical components (k-NN selection + discriminative loss) are not overly complex, but are well combined.
- **Experimental Thoroughness**: ★★★★★ — Highly comprehensive, covering 5 datasets, 6 baselines, and extensive ablation and comparative experiments.
- **Practicality**: ★★★★★ — Plug-and-play, parameter-free, and broadly applicable; high practical value.
- **Writing Quality**: ★★★★☆ — Motivation is clearly articulated and experimental organization is well structured.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CGHair: Compact Gaussian Hair Reconstruction with Card Clustering](../../CVPR2026/3d_vision/cghair_compact_gaussian_hair_reconstruction_with_card_clustering.md)
- [\[AAAI 2026\] MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection](../../AAAI2026/3d_vision/monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)
- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[NeurIPS 2025\] RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes](rgb-only_supervised_camera_parameter_optimization_in_dynamic_scenes.md)
- [\[NeurIPS 2025\] CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning](cosmobench_a_multiscale_multiview_multitask_cosmology_benchmark_for_geometric_de.md)

</div>

<!-- RELATED:END -->
