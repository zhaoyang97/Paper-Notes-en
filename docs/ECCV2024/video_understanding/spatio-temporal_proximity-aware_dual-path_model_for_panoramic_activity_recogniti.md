---
title: >-
  [Paper Note] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition
description: >-
  [ECCV 2024][Video Understanding][Panoramic Activity Recognition] This paper proposes SPDP-Net, which models social relationships among individuals through spatio-temporal proximity and utilizes a Dual-Path Transformer (DPATr) architecture to synergistically recognize multi-granular activities along two paths: individual-to-global and individual-to-social. It significantly outperforms previous SOTA models on the JRDB-PAR dataset with an overall F1 score of 46.5%.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Panoramic Activity Recognition"
  - "Social Group Detection"
  - "Spatio-Temporal Proximity"
  - "Dual-Path Transformer"
  - "Multi-Granular Activity"
date: 2026-05-08
content_hash: 4c9f3ac8b522218b
---

# Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition

**Conference**: ECCV 2024  
**arXiv**: [2403.14113](https://arxiv.org/abs/2403.14113)  
**Code**: Not released  
**Area**: Other  
**Keywords**: Panoramic Activity Recognition, Social Group Detection, Spatio-Temporal Proximity, Dual-Path Transformer, Multi-Granular Activity

## TL;DR

This paper proposes SPDP-Net, which models social relationships among individuals through spatio-temporal proximity and utilizes a Dual-Path Transformer (DPATr) architecture to synergistically recognize multi-granular activities along two paths: individual-to-global and individual-to-social. It significantly outperforms previous SOTA models on the JRDB-PAR dataset with an overall F1 score of 46.5%.

## Background & Motivation

Panoramic Activity Recognition (PAR) aims to identify human activities across three granularities from panoramic videos: (i) individual actions, (ii) social group activities and group detection, and (iii) global activity. PAR faces two core challenges:

**Spatial proximity is insufficient to determine social relations**: Existing methods only exploit the spatial distance between individuals in a single frame to infer social relations. However, as shown in Fig. 1 of the paper, three individuals may appear to belong to the same group in the initial frame, but only two of them continue to walk together over time. This indicates that temporal proximity must be introduced to accurately determine social dynamics.

**Hierarchical modeling of multi-granular activities suffers from information bottlenecks**: Existing methods [JRDB-PAR, MUP] adopt an "individual -> social -> global" hierarchical structure. However, in reality, both global and social activities require individual information, and they interact with each other. A unidirectional hierarchical structure fails to fully capture this bidirectional dependency.

## Method

### Overall Architecture

SPDP-Net consists of two stages:

1. **Proximity-based Relation Encoding**: Enhances individual feature representations using spatio-temporal positional relations, and performs social group detection via a combination of feature similarity and spatio-temporal proximity.
2. **Multi-Granular Activity Recognition**: Synergistically models individual, social groups, and global activities through the dual-path architecture of DPATr.

The input panoramic video has its frame-level features extracted by a 2D CNN backbone (Inception-v3). Individual regional features are then cropped via RoIAlign and downsampled using 3D convolution to yield $F^{idv} \in \mathbb{R}^{N_i \times T \times d \times h \times w}$.

### Key Designs

1. **Panoramic Positional Embedding (PPE)**: Traditional positional encoding only encodes location within the cropped region, which discards the absolute positional context of the individual within the panoramic scene. PPE extracts individual regions from the full-scene sinusoidal positional embedding, thereby preserving the individual's spatio-temporal position within the panorama. Specifically, multi-head self-attention with PPE is sequentially applied to the individual features along the temporal, height, and width dimensions:

$$\bar{F}^{idv} = A^w(A^h(A^t(F^{idv}, e_{pn}), e_{pn}), e_{pn}) + F^{idv}$$

2. **Temporal Generalized IoU (TGIoU)**: Extends GIoU from the spatial to the temporal dimension, measuring the spatio-temporal proximity of two individuals across the entire video sequence:

$$R_p(i,j) = \text{TGIoU}(P^i, P^j) = \frac{1}{T}\sum_{t=1}^{T}\text{GIoU}(p_t^i, p_t^j)$$

Here, GIoU considers both the intersection over union (IoU) of the bounding boxes and their minimum enclosing box. By averaging GIoU across frames, TGIoU captures dynamic changes in social distance, reflecting real-world social relationships more accurately than single-frame spatial metrics or Euclidean distance.

3. **Social Relation Matrix**: Fuses the visual similarity matrix $R_s$ and the spatio-temporal proximity matrix $R_p$:

- $R_s = \text{Softmax}(W_\theta \bar{F}^{idv} (W_\phi \bar{F}^{idv})^\top)$ (learnable similarity of visual features)
- $R_p$ is directly computed via TGIoU (non-learnable physical positional relationship)
- The final relationship matrix $R = \frac{1}{2}(R_s + R_p)$

The number of social groups is regressed from the mean of the relation-enhanced features using an MLP, and the group partitioning is achieved through K-means clustering.

4. **Dual-Path Activity Transformer (DPATr)**: Composed of $L$ layers, each containing two paths based on Transformer encoders:

- **Individual-to-Global Path**: Prepends a learnable global token to the sequence of individual features, jointly encoding individual interactions and global context via self-attention.
- **Individual-to-Social Path**: Groups the individual features output by the global path according to group assignments $\mathcal{G}$, and prepends a learnable social token to each group to capture the activity dynamics of individual social groups.

Through multi-layer stacking, the two paths enhance each other to produce a synergistic effect, ultimately outputting individual ($\tilde{F}^{idv}$), social group ($F^{sg}$), and global ($F^{glb}$) activity features.

### Loss & Training

The total loss consists of six components:

$$\mathcal{L} = \mathcal{L}_{idv} + \mathcal{L}_R + \mathcal{L}_{aux} + \lambda_{sg}\mathcal{L}_{sg} + \lambda_{glb}\mathcal{L}_{glb} + \lambda_n\mathcal{L}_n$$

- $\mathcal{L}_{idv}$: binary cross-entropy loss for individual action recognition
- $\mathcal{L}_R$: supervision loss for the social relation matrix
- $\mathcal{L}_{aux}$: auxiliary individual action loss based on $\bar{F}^{idv}$
- $\mathcal{L}_{sg}, \mathcal{L}_{glb}$: binary cross-entropy losses for social group and global activities
- $\mathcal{L}_n$: L2 loss for estimating the number of groups
- Loss weight ratio: $\lambda_{sg}:\lambda_{glb}:\lambda_n = 3:2:5$

Training employs the Adam optimizer for 60 epochs, with a linear warmup for the first 15 epochs, followed by a fixed learning rate of $4\times10^{-5}$ and a weight decay of $10^{-2}$.

## Key Experimental Results

### Main Results

Comparison with SOTA methods on the JRDB-PAR dataset:

| Method | $\mathcal{F}_i$ (Individual) | $\mathcal{F}_p$ (Social) | $\mathcal{F}_g$ (Global) | $\mathcal{F}_a$ (Overall) |
|------|---------|---------|---------|---------|
| ARG | 33.2 | 8.2 | 50.7 | 30.7 |
| JRDB-PAR | 43.4 | 24.8 | 38.8 | 35.6 |
| MUP | 47.7 | 25.1 | 51.8 | 41.5 |
| **SPDP-Net** | **51.8** | **34.2** | **53.5** | **46.5** |

Comparison of social group detection performance:

| Method | IoU@0.5 | IoU@AUC | Mat.IoU |
|------|---------|---------|---------|
| JRDB-PAR | 37.9 | 25.2 | 22.3 |
| MUP | 46.9 | 34.2 | 28.5 |
| **SPDP-Net** | **56.4** | **42.5** | **34.3** |

### Ablation Study

| Configuration | $\mathcal{F}_a$ | IoU@0.5 | Description |
|------|---------|---------|------|
| Temporal Attention Only (w/o PPE) | 42.5 | - | Baseline |
| Spatial + Temporal Attention + PPE | **46.5** | - | PPE yields +3.1% |
| Only $R_s$ | - | 37.6 | Visual Similarity |
| Only $R_p$ | - | 55.8 | Spatio-temporal proximity, larger contribution |
| $R_s + R_p$ | - | **56.4** | Complementary |
| GIoU (Spatial) | - | 48.7 | Single-frame spatial metric |
| TGIoU (Spatio-temp) | - | **56.4** | Temporal expansion yields +7.7% |

### Key Findings

- Spatio-temporal proximity $R_p$ contributes significantly more to social group detection than visual similarity $R_s$ (IoU@0.5: 55.8 vs. 37.6).
- TGIoU yields a 7.7% improvement in IoU@0.5 compared to spatial GIoU, proving the time dimension is crucial for social relationship determination.
- The dual-path architecture of DPATr outperforms parallel, hierarchical, and inverse hierarchical structures across all granularities, verifying the hypothesis that multi-granular activities interact with each other.
- Using ground-truth group detection further boosts $\mathcal{F}_p$ by 19.3%, indicating that the accuracy of group detection is the bottleneck of social activity recognition.

## Highlights & Insights

1. **The design of TGIoU is simple yet effective**: It only requires temporal averaging over existing GIoU without adding extra parameters, yet significantly boosts social group detection performance.
2. **Clear design motivation of the dual-path architecture**: Experiments validate the intuition that "global context benefits social activities" and "individual information is crucial for both". DPATr elegantly realizes this bidirectional information flow through its dual-path design.
3. **Necessity of Panoramic Positional Embedding**: It solves the problem of losing global position information after cropping individual regions, which is a unique challenge in the PAR task.

## Limitations & Future Work

1. The model uses a frozen Inception-v3 backbone; replacing it with stronger pre-trained models (e.g., ViT) might yield further improvement.
2. Group number estimation still has room for improvement (performance significantly increases with ground-truth group counts); more precise group size prediction methods can be explored.
3. K-means clustering is non-differentiable, preventing end-to-end training of group assignment; differentiable clustering schemes could be explored.
4. The method is validated only on the single JRDB-PAR dataset; its generalization remains to be verified on other datasets.

## Related Work & Insights

- **JRDB-PAR [Han et al.]**: First proposed the PAR task and a hierarchical GCN solution, serving as the main comparison baseline of this paper.
- **MUP [Cao et al.]**: A unified multi-granular perception framework employing hierarchical aggregation.
- **Groupformer [Li et al.]**: Spatio-temporal Transformer for group activity recognition, which inspired the design of DPATr.
- **GIoU [Rezatofighi et al.]**: Extended to the temporal dimension in this work, demonstrating simplicity and effectiveness.

## Rating

- Novelty: ⭐⭐⭐⭐ — Both TGIoU and the dual-path architecture are reasonable and novel designs with clear motivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The ablation study is highly comprehensive, covering the contributions of each module; however, validation is limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, with well-executed method descriptions and experimental analyses.
- Overall Value: ⭐⭐⭐⭐ — Achieves significant breakthroughs in the emerging PAR task, providing valuable insights for social scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[CVPR 2025\] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition](../../CVPR2025/video_understanding/tamt_temporal-aware_model_tuning_for_cross-domain_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
