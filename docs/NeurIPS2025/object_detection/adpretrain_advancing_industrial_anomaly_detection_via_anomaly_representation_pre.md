---
title: >-
  [Paper Note] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining
description: >-
  [NeurIPS 2025][Object Detection][anomaly representation pretraining] This work presents ADPretrain, the first dedicated representation pretraining framework for industrial anomaly detection. By learning residual feature…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "anomaly representation pretraining"
  - "contrastive learning"
  - "residual features"
  - "industrial anomaly detection"
  - "feature pretraining"
date: 2026-05-08
content_hash: bddd210ba24f4a99
---

# ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining

**Conference**: NeurIPS 2025
**arXiv**: [2511.05245](https://arxiv.org/abs/2511.05245)  
**Code**: [xcyao00/ADPretrain](https://github.com/xcyao00/ADPretrain)  
**Area**: Other
**Keywords**: anomaly representation pretraining, contrastive learning, residual features, industrial anomaly detection, feature pretraining
**Institution**: Shanghai Jiao Tong University, Nanjing Agricultural University

## TL;DR

This work presents ADPretrain, the first dedicated representation pretraining framework for industrial anomaly detection. By learning residual feature representations via angle-oriented and norm-oriented contrastive losses on the large-scale RealIAD dataset, the pretrained features consistently improve five mainstream embedding-based AD methods across five datasets and five backbone networks when substituted for the original features.

## Background & Motivation

**Background**: Current mainstream and state-of-the-art anomaly detection methods rely almost exclusively on ImageNet-pretrained feature networks (e.g., PaDiM, PatchCore, UniAD). However, ImageNet pretraining involves no notion of "normal vs. anomalous," creating a mismatch between pretraining objectives and anomaly detection goals.

**Limitations of Prior Work**: A notable distribution shift exists between natural images in ImageNet and industrial images in AD scenarios, which limits the effectiveness of directly applying pretrained features. Under unsupervised training with only normal samples, learning representations from scratch faces two key challenges: (a) mode collapse, where normal and anomalous features become indistinguishable, and (b) limited dataset scale in conventional AD benchmarks, which constrains representation quality. Fine-tuning approaches such as PANDA and MeanShift adapt pretrained features to AD data but remain dataset-specific and lack generalizability.

**Key Challenge**: The emergence of large-scale AD datasets offers an opportunity: RealIAD comprises 151,050 images (99,721 normal + 51,329 anomalous) with ground-truth masks, providing the data foundation for AD-specific pretraining.

**Goal**: No prior work has systematically investigated anomaly representation pretraining for anomaly detection tasks. This paper constitutes the first exploration of this problem.

## Method

### Overall Architecture

The core mechanism of ADPretrain is to learn discriminative pretrained features on the large-scale RealIAD dataset using residual features as the base representation, optimized via angle-oriented and norm-oriented contrastive losses. After pretraining, the features output by the Feature Projector can directly replace the original features in existing embedding-based AD methods.

The overall pipeline:
- Input image → Fixed backbone extracts multi-level features → Subtract matched normal reference features to obtain residual features → Feature Projector transformation → Contrastive loss optimization

### Key Design 1: Residual Feature-Based Representation

- **Design Motivation**: The goal is to obtain domain-invariant pretrained features such that the normal/anomalous representation distributions remain consistent even when the downstream AD dataset differs from the pretraining dataset.
- **Mechanism**: The residual feature from ResAD is adopted: $r_{h,w}^l = x_{h,w}^l - x_n^*$, where $x_n^*$ is the nearest-neighbor feature retrieved from a normal reference bank.
- **Key Details**: During training, reference samples are randomly selected per input to increase the diversity of residual features; at test time, 8 reference samples are used per image.
- Experiments confirm that residual features are better suited than raw backbone features for AD pretraining, with particularly pronounced differences under the FeatureNorm baseline.

### Key Design 2: Angle-Oriented and Norm-Oriented Contrastive Losses

**Angle-Oriented Contrastive Loss**:

- Based on InfoNCE with two key modifications:
    - (a) **Contrastive pairs are drawn only between normal and anomalous features**: Negatives are restricted to features with labels differing from the anchor (controlled by $\mathbb{I}_{[m_k \neq m_i]}$), preventing normal features from being incorrectly repelled from each other.
    - (b) **Cosine similarity is computed relative to the normal feature center**: Inspired by MeanShift, the normal feature center $c$ is subtracted to yield $\bar{x}_i = x_i - c$ before computing cosine similarity, preventing features from distributing uniformly on the origin-centered hypersphere.
- Only original image features are used as anchors; augmented image features are not used as anchors to avoid contrasting augmented images against anomalous data.

**Norm-Oriented Contrastive Loss**:

- Inspired by one-class classification (OCC) learning but reformulated contrastively:
    - **Normal features**: A compactness loss $\mathcal{L}_{con}$ constrains normal features within a hypersphere of radius $r=0.4$ centered at the origin.
    - **Anomalous features**: A margin $\Delta r=0.75$ is introduced to push anomalous features outside a hypersphere of radius $r'=r+\Delta r$; anomalous features already outside the hypersphere are not pushed further, preventing pretraining overfitting.
- The pseudo-Huber distance $\sqrt{\|x_i\|_2^2 + 1} - 1$ is used as a more robust norm measure.
- Gradient analysis shows that this loss adaptively assigns larger gradients to features lying outside the hypersphere.

### Key Design 3: Feature Projector with Learnable Key/Value Attention

- The architecture is Transformer-based, replacing self-attention with **Learnable Key/Value Attention (LKV-Attn)**.
- A set of $N_r=2048$ randomly initialized learnable reference representations $\mathcal{R} \in \mathbb{R}^{N_r \times C}$ serves as Keys and Values for all attention layers.
- Input features act as Queries and interact with the learnable references via cross-attention.
- The residual connection uses **subtraction** rather than addition: the attention output captures learned normal patterns, and subtraction further eliminates normal components from the residual feature distribution, enhancing discriminability between normal and anomalous features.
- Experiments demonstrate that LKV-Attn outperforms alternatives including linear/MLP projectors, self-attention, and standard cross-attention.

### Loss & Training

$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{2N} \lambda \cdot \mathbb{I}_{[i \leq N]} \cdot \mathcal{L}_{angle}(x_i) + \mathcal{L}_{norm}(x_i)$$

where $\lambda=1$ (default). The angle loss is computed only over original image features, while the norm loss is applied to all $2N$ features.

## Key Experimental Results

### Main Results (Tab. 1)

Comprehensive evaluation across 5 backbones (DINOv2-Base/Large, CLIP-Base/Large, ImageBind) × 5 AD methods (PaDiM, PatchCore, CFLOW, GLASS, UniAD) × 5 datasets (MVTecAD, VisA, BTAD, MVTec3D, MPDD):

| Setting | Representative Gain (AUROC/PRO) |
|---|---|
| DINOv2-Base + PatchCore + MVTecAD | +3.5/+4.7 |
| DINOv2-Base + UniAD + MVTecAD | **+26.0**/+9.7 |
| CLIP-Base + PatchCore + MPDD | +10.1/+28.7 |
| ImageBind + UniAD + MPDD | **+32.9**/+42.7 |
| DINOv2-Large + PaDiM + MPDD | +6.8/+3.3 |

- Gains are especially pronounced for combinations where original features perform poorly (e.g., UniAD + DINOv2-Base).
- **FeatureNorm baseline**: Original features are nearly unable to distinguish normal from anomalous samples by norm alone (AUROC ~50%), whereas pretrained features achieve 90%+ AUROC using norm alone.

### Ablation Study (Tab. 2, VisA + ImageBind)

| Component | PaDiM (AUROC/PRO) | PatchCore | FeatureNorm |
|---|---|---|---|
| Original features (no pretraining) | 92.6/86.3 | 91.6/81.3 | 49.2/44.5 |
| Non-residual + angle & norm loss | 93.5/86.1 | 93.6/85.1 | 82.9/83.9 |
| Residual + angle & norm loss (full) | **95.4/88.7** | **94.6/87.0** | **94.2/89.0** |
| Angle loss only | 93.9/85.2 | 93.2/83.8 | 83.9/83.0 |
| Norm loss only | 93.7/85.3 | 90.9/84.5 | 92.4/85.1 |

### Key Findings

1. **Residual vs. standard features**: Residual features show a substantial advantage under FeatureNorm (94.2 vs. 82.9), demonstrating stronger class-invariant properties.
2. **Complementarity of the two losses**: The angle loss benefits distance-based methods (PaDiM, PatchCore), while the norm loss excels under FeatureNorm; their combination achieves the best overall results.
3. **Backbone must be frozen**: Making the backbone learnable causes severe performance degradation (e.g., residual features + learnable backbone yields only 78.4/32.1 on PatchCore, vs. 94.6/87.0 with a frozen backbone).
4. **Sample efficiency (Tab. 3a)**: With only 10% of normal training samples, pretrained features yield even larger gains (e.g., PaDiM + MVTecAD: +15.4; PaDiM + MPDD: +28.3).
5. **Noise robustness (Tab. 3b)**: When 10% anomalous noise is introduced into the training set, pretrained features continue to deliver stable and often larger improvements.

## Highlights & Insights

1. **Novel problem formulation**: This is the first work to systematically define and address anomaly representation pretraining, filling an important gap in the AD literature.
2. **Dual-dimensional contrastive design**: Optimizing both the direction and magnitude of feature vectors simultaneously is theoretically complete and experimentally complementary.
3. **Compelling FeatureNorm baseline**: Achieving high AD performance using feature norm alone provides intuitive and strong evidence of pretrained feature quality — a capability absent in original features.
4. **Plug-and-play generalizability**: Pretrained features directly replace original features in five structurally diverse AD methods without modifying any downstream components.
5. **Subtractive residual connection**: Using subtraction in the Feature Projector's residual connection encourages the attention module to actively learn and remove normal patterns, which is coherent with the overall objective of amplifying normal–anomalous differences.

## Limitations & Future Work

1. **Dependence on large-scale annotated AD data**: Pretraining requires pixel-level mask annotations from RealIAD to obtain feature-level labels, and acquiring such large-scale annotated data is costly in real industrial deployments.
2. **Full backbone fine-tuning fails**: Experiments show that fine-tuning the entire backbone on RealIAD leads to performance collapse, indicating that current AD dataset scales remain insufficient to support full-parameter pretraining.
3. **Single pretraining dataset**: Pretraining is conducted solely on RealIAD; multi-dataset joint pretraining or data augmentation strategies to broaden pretraining diversity are not explored.
4. **Computational overhead underexplored**: Computing residual features requires nearest-neighbor retrieval from a reference bank for each image, which may become a bottleneck at scale.
5. **Generalization beyond industrial scenarios untested**: All downstream datasets are industrial manufacturing benchmarks; generalization to other AD domains such as medical imaging or remote sensing remains unknown.

## Related Work & Insights

| Direction | Representative Works | Relationship to This Paper |
|---|---|---|
| Pretrained feature-based AD | PaDiM, PatchCore, CFLOW, UniAD | Used as downstream validation methods that directly benefit from the pretrained feature replacement |
| Feature adaptation/fine-tuning | PANDA, MeanShift, FYD, GaussianFineTune | Fine-tuned networks are dataset-specific and lack generalizability; pretrained features in this work are more universal |
| Residual features | ResAD | The residual feature concept is borrowed, but the motivation and goal differ (pretraining vs. downstream detection) |
| Contrastive learning | SimCLR, MoCo, InfoNCE | Standard contrastive learning is adapted with AD-specific modifications (center shift + label-aware negatives) |
| OCC learning | DeepSVDD, FCDD | The norm loss draws on hypersphere constraint ideas, augmented with a contrastive anomaly-repulsion mechanism |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to define and systematically address anomaly representation pretraining; the angle+norm dual-dimensional contrastive design is theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across 5 backbones × 5 methods × 5 datasets; ablation studies cover all key components; additional analyses on sample efficiency and noise robustness.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, rigorous mathematical derivation, and gradient analysis enhancing interpretability of the loss functions.
- **Value**: ⭐⭐⭐⭐⭐ — Provides plug-and-play pretrained features with direct applicability across the AD community; the FeatureNorm baseline reveals the untapped potential of learned anomaly representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](../../AAAI2026/object_detection/rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[NeurIPS 2025\] Normal-Abnormal Guided Generalist Anomaly Detection](normal-abnormal_guided_generalist_anomaly_detection.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](../../ICLR2026/object_detection/paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)

</div>

<!-- RELATED:END -->
