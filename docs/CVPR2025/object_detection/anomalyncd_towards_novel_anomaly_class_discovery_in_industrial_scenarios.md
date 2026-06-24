---
title: >-
  [Paper Note] AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios
description: >-
  [CVPR 2025][Object Detection][novel class discovery] Proposes AnomalyNCD, the first self-supervised multi-class anomaly classification method for industrial scenarios: MEBin extracts major anomaly regions $\rightarrow$ mask-guided ViT focuses on weak-semantic anomalies $\rightarrow$ region fusion strategy achieves flexible region/image-level classification, improving F1 by 10.8% and NMI by 8.8% on MVTec AD.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "novel class discovery"
  - "anomaly classification"
  - "MEBin"
  - "mask-guided attention"
  - "industrial inspection"
date: 2026-05-08
content_hash: 414d14c7ef0d2982
---

# AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios

**Conference**: CVPR 2025  
**arXiv**: [2410.14379](https://arxiv.org/abs/2410.14379)  
**Code**: [GitHub](https://github.com/HUST-SLOW/AnomalyNCD)  
**Area**: Others  
**Keywords**: novel class discovery, anomaly classification, MEBin, mask-guided attention, industrial inspection

## TL;DR
Proposes AnomalyNCD, the first self-supervised multi-class anomaly classification method for industrial scenarios: MEBin extracts major anomaly regions $\rightarrow$ mask-guided ViT focuses on weak-semantic anomalies $\rightarrow$ region fusion strategy achieves flexible region/image-level classification, improving F1 by 10.8% and NMI by 8.8% on MVTec AD.

## Background & Motivation

**Background**: Mature methods (e.g., PatchCore, EfficientAD) exist for industrial anomaly detection, which can localize anomalies but fail to distinguish fine-grained anomaly classes (e.g., fracture vs. ablation). Downstream processing requires identifying anomaly categories and even discovering novel classes.

**Limitations of Prior Work**:
   - **Anomaly clustering methods** (AC, UniFormaly): Frozen feature extractors cannot learn anomaly-specific features.
   - **Generic NCD methods** (UNO, GCD, SimGCD): Assume objects are centered in images, which is inapplicable to industrial scenarios.
   - **Two major obstacles**:
     - ❶ Non-salient anomalies: Industrial anomalies are local damage and are not located at the image center.
     - ❷ Weak-semantic anomalies: Industrial anomalies have weak semantics; ViTs tend to focus on the background rather than the anomaly.

**Key Challenge**: The attention of the classification network (ViT) naturally focuses on salient objects rather than subtle anomalies, rendering standard NCD pipelines completely ineffective for industrial defects.

**Key Insight**: Designing MEBin to isolate anomalies from detection results $\rightarrow$ cropping them into anomaly-centered sub-images $\rightarrow$ using mask-guided attention to force the [CLS] token to focus on anomaly regions.

**Core Idea**: Anomaly-centered cropping + mask-guided ViT attention = enabling the classification network to "see" weak-semantic anomalies.

## Method

### Overall Architecture
1. Use anomaly detection methods (e.g., MuSc, PatchCore) to obtain anomaly probability maps.
2. MEBin binarizes the probability maps and extracts anomaly-centered sub-images.
3. Mask-Guided ViT (MGViT) learns discriminative features of anomalies.
4. Teacher-Student framework generates pseudo-labels for classification learning.
5. Region fusion strategy merges sub-image predictions into image-level classification.

### Key Designs

1. **Main Element Binarization (MEBin)**

    - **Function**: Stably extract major anomaly regions from anomaly detection results.
    - Three-step pipeline:
        - Step 1: Determine the threshold range $[s_{\min}, s_{\max}]$, where $s_{\min}$ is the maximum value of the minimum anomaly scores of all anomaly maps.
        - Step 2: Uniformly sample $\mathcal{T}=64$ thresholds for binarization.
        - Step 3: Find the most frequent number of connected components $\bar{\delta}_i$, and select the minimum threshold for complete segmentation.
    - Core Advantage: Adaptive threshold selection without validation sets, generalizable to various AD methods.
    - Contrast with Otsu: Otsu tends to over-detect, especially on normal images.

2. **Mask-Guided Vision Transformer (MGViT)**

    - **Function**: Guide the [CLS] token's attention to focus on the anomaly regions.
    - **Mechanism**: Insert masks into the self-attention of the last $L_m=9$ layers.
    - Comparison of three designs:
        - (a) Masking both CLS and patch tokens $\rightarrow$ suppresses context.
        - (b) Masking only patch tokens $\rightarrow$ also suppresses context.
        - **(c) Masking only the CLS token** (adopted) $\rightarrow$ patch tokens maintain global receptive fields.
    - Masked Attention: $\text{Attn} = \text{softmax}(\text{concat}(\mathbf{Q}^{cls}\mathbf{K}^\top + \bar{\mathcal{M}}, \mathbf{Q}^{patch}\mathbf{K}^\top))\mathbf{V}$
    - Where $\bar{\mathcal{M}}(i) = 0$ if $\mathcal{M}(i) > 0.5$, else $-\infty$.

3. **Pseudo-Label Correction (PLC)**

    - **Function**: Correct pseudo-labels of over-detected regions using anomaly scores.
    - Formula: $\hat{q}_{i,k} \leftarrow w_{i,k}\mathbf{e} + (1-w_{i,k})\hat{q}_{i,k}$, where $w_{i,k} = \max(0.5 - s_{i,k}, 0)$.
    - Effect: Recall for normal class improves by 14.9%.

4. **Region Fusion Strategy**

    - **Function**: Determine the image-level class based on sub-image classifications.
    - **Core Idea**: Area weighting (instead of simple averaging or anomaly score weighting).
    - Formula: $\alpha_{i,k}^u = \frac{\exp(a_{i,k}^u / \tau_\alpha)}{\sum_k \exp(a_{i,k}^u / \tau_\alpha)}$
    - **Design Motivation**: Over-detected regions have small areas but high anomaly scores; area weighting mitigates their impact.

### Loss & Training
$$\mathcal{L} = \lambda(\mathcal{L}_{rep}^l + \mathcal{L}_{cls}^l) + (1-\lambda)(\mathcal{L}_{rep} + \mathcal{L}_{cls}^u + \mu\mathcal{L}_{reg}^u)$$
- $\mathcal{L}_{rep}^l$: Supervised contrastive learning, $\mathcal{L}_{rep}$: Self-supervised contrastive learning.
- $\mathcal{L}_{cls}^l$: Cross-entropy with GT labels, $\mathcal{L}_{cls}^u$: Cross-entropy with pseudo-labels.
- $\mathcal{L}_{reg}^u$: Mean entropy maximization regularization.

## Key Experimental Results

### Main Results (Unsupervised setting, using only unlabeled images)

| Method | MVTec AD NMI↑ | MVTec AD ARI↑ | MVTec AD F1↑ |
|------|-------------|-------------|------------|
| SimGCD | 0.452 | 0.346 | — |
| AC (Anomaly Clustering) | 0.525 | 0.431 | — |
| **MuSc + AnomalyNCD** | **0.613** | **0.526** | **0.712** |

### Semi-supervised Setting (Using normally labeled images)

| AD Method + AnomalyNCD | MVTec AD NMI↑ | MVTec AD ARI↑ | MVTec AD F1↑ |
|---------------------|-------------|-------------|------------|
| PatchCore | 0.670 | 0.601 | 0.769 |
| CPR | **0.736** | **0.674** | **0.805** |

### Ablation Study

| Component | NMI | ARI | F1 |
|------|-----|-----|------|
| (a) w/o MGA | 0.598 | 0.494 | 0.698 |
| (b) all tokens | 0.507 | 0.382 | 0.600 |
| (c) patch tokens | 0.563 | 0.467 | 0.686 |
| **(d) class token (Ours)** | **0.613** | **0.526** | **0.712** |

| MEBin vs. Fixed Threshold | FPR↓ | FNR↓ | F1↑ |
|-----------------|------|------|------|
| Fixed threshold 0.5 | High | High | 0.640 |
| Otsu | Highest | Medium | 0.499 |
| **MEBin** | **0.153** | **0.035** | **0.712** |

### Key Findings
- MGA performs best when applied only to the CLS token (+5.0% NMI, +2.6% F1).
- Replacing mask attention in the last 9 layers is optimal ($L_m=9$).
- Area-weighted fusion outperforms average/score weighting.
- NMI reaches 0.871 under GT masks, indicating that the quality of AD methods is the bottleneck.
- Using labeled anomalous data ($\mathcal{D}_l$) yields a +3.0% NMI gain.

## Highlights & Insights
- **The first self-supervised multi-class anomaly classification method** for industrial scenarios, compatible with any AD method.
- MEBin provides adaptive threshold selection, generalizable to various AD methods.
- Elegant mask-guided attention design, requiring modification of only the CLS token's attention.
- Supports composite anomalies (a single image containing multiple anomaly types).

## Limitations & Future Work
- Performance is highly dependent on the quality of the upstream AD method (e.g., EfficientAD's extreme span of anomaly probability causes poor results).
- MEBin's computation is CPU-based (OpenCV connected components analysis), bottlenecking the inference by taking over 80% of the time.
- Requires the number of novel classes $\mathcal{C}_u$ as a prior.

## Rating
- Novelty: ⭐⭐⭐⭐ Pioneering integration of NCD with industrial anomaly classification, unique MEBin design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed ablation studies (7 ablations + cross-dataset evaluations + class-wise results).
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive diagrams.
- Value: ⭐⭐⭐⭐ A crucial cornerstone for downstream processing in industrial quality inspection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[CVPR 2025\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)
- [\[CVPR 2025\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](../../NeurIPS2025/object_detection/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)

</div>

<!-- RELATED:END -->
