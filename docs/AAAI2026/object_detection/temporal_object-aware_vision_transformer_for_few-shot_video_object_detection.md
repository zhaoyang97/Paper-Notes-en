---
title: >-
  [Paper Note] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection
description: >-
  [AAAI 2026][Object Detection][Few-Shot Detection] This paper proposes an object-aware temporal modeling framework that achieves cross-frame temporal consistency through selective propagation of high-confidence detection…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "Few-Shot Detection"
  - "Video Object Detection"
  - "Temporal Modeling"
  - "OWL-ViT"
  - "Object-Awareness"
date: 2026-05-08
content_hash: 36ba08fffade3844
---

# Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection

**Conference**: AAAI 2026
**arXiv**: [2511.13784](https://arxiv.org/abs/2511.13784)  
**Code**: [https://github.com/yogesh-iitj/fs-video-vit](https://github.com/yogesh-iitj/fs-video-vit)  
**Area**: Object Detection
**Keywords**: Few-Shot Detection, Video Object Detection, Temporal Modeling, OWL-ViT, Object-Awareness

## TL;DR

This paper proposes an object-aware temporal modeling framework that achieves cross-frame temporal consistency through selective propagation of high-confidence detection features. Combined with a pretrained vision-language encoder (OWL-ViT) and a few-shot detection head, the method achieves an average improvement of 3.7%–5.3% AP across four video few-shot detection benchmarks.

## Background & Motivation

### Few-Shot Video Object Detection (FSVOD)

FSVOD aims to detect novel object categories in videos using only a small number of annotated samples. Compared to image-based few-shot detection, FSVOD faces additional challenges:

**Temporal Consistency**: Maintaining detection consistency across frames subject to occlusion, appearance changes, and motion blur.

**Novel Category Generalization**: Adapting to unseen categories without relying on complex region proposals.

**Inter-Frame Information Utilization**: Processing each frame independently discards valuable temporal cues.

### Limitations of Prior Work

| Method Type | Representative Methods | Limitations |
|-------------|----------------------|-------------|
| Non-temporal | FSOD, UP-DETR | Process frames independently; cannot exploit temporal information |
| Tracking-based | DeepSort+CLIP | Weak temporal matching; high false positive rate |
| Proposal-based | FSVOD, TACF | Region proposal networks not optimized for few-shot settings |
| Query-based | QDETRv | Difficulty distinguishing visually similar objects |

Although **Video OWL-ViT** leverages large-scale vision-language pretraining, its temporal modeling is frame-independent — detected objects are not considered across frames, leading to inconsistent detections.

### Core Problem

**Can high-confidence detection features from previous frames be selectively propagated to the current frame?** Such an approach would exploit temporal information to enhance detection while preventing noise accumulation by filtering out low-confidence features.

## Method

### Overall Architecture

The framework comprises three main components:

1. **Language-Aligned Visual Encoder**: Provides semantically rich visual representations.
2. **Temporal Fusion Decoder**: Selectively propagates high-confidence object features across frames.
3. **Few-Shot Matching and Detection Head**: Aligns query frame features with support samples.

### Key Designs

#### 1. Language-Aligned Visual Encoder: Advantages of OWL-ViT

**Why OWL-ViT over CNN backbones?**

Traditional CNN backbones (e.g., ResNet, used in FSVOD and TACF) are trained solely on visual classification and exhibit weak generalization to novel categories. OWL-ViT, pretrained on large-scale image-text pairs, offers:
- Stronger semantic transfer capability
- Better discrimination between visually similar but categorically distinct objects
- Improved handling of partially visible or occluded objects

The encoder transforms input images into patch embeddings: $\mathbf{E} \in \mathbb{R}^{P \times D}$, where $P$ is the number of patches and $D$ is the embedding dimension.

#### 2. Object-Aware Temporal Decoder: Core Innovation

**Support Set Encoding**:
- Patch-level features and objectness scores are extracted from each support image.
- The patch with the highest objectness score is selected as the object representation: $p^* = \arg\max_p s_{i,j}^p$
- A class prototype $\mathbf{z}_i$ is obtained by averaging over $K$ support samples.

**Temporal Fusion Mechanism (Core)**:

The first frame is processed directly. For subsequent frames $t > 0$, high-confidence detections are selected from the classification head output of the previous frame:

$$\mathbf{V}_{t-1}^{\text{det}} = \{\mathbf{v}_{t-1}^p \mid \hat{c}_{t-1,i}^p > \tau\}$$

where $\tau = 0.94$ is the confidence threshold. Cross-attention is then used to fuse previous-frame detection features into the current frame:

$$\mathbf{A}_t = \text{Softmax}\left(\frac{\mathbf{F}_t (\mathbf{V}_{t-1}^{\text{det}})^T}{\sqrt{D}}\right)$$

$$\hat{\mathbf{F}}_t = \mathbf{F}_t + \mathbf{A}_t \mathbf{V}_{t-1}^{\text{det}}$$

**Design Motivation**:
- **Selective Propagation**: Only high-confidence object features are propagated, preventing background noise accumulation.
- **Residual Connection**: $\hat{\mathbf{F}}_t = \mathbf{F}_t + \ldots$ ensures current-frame information is not overwritten.
- **Lightweight and Efficient**: A single cross-attention layer is sufficient; no complex pipelines or tracking modules are introduced.

#### 3. Few-Shot Detection Head: Parallel Classification and Localization

**Classification Head**:
- Projects temporally enhanced features into the classification space: $\mathbf{V}_t^{cls} = \mathbf{W}_{cls} \hat{\mathbf{F}}_t$
- Computes cosine similarity with support class prototypes:

$$s_{p,i} = \frac{\mathbf{v}_t^p \cdot \mathbf{z}_i}{\|\mathbf{v}_t^p\| \cdot \|\mathbf{z}_i\|}$$

- Applies softmax normalization to obtain probability distributions over $N$ categories and background.

**Localization Head**:
- Predicts bounding box coordinates via a multi-layer MLP: $\hat{\mathbf{B}}_t = \text{MLP}_{box}(\hat{\mathbf{F}}_t)$
- Each patch predicts four coordinate values $(x, y, w, h)$
- Patches whose maximum classification score exceeds threshold $\kappa = 0.98$ are treated as valid detections.

### Loss & Training

**End-to-End Training Objective**:

$$\mathcal{L} = \sum_{t=1}^{T} \left[\sum_{m=1}^{M_t} \lambda_{cls} \cdot \mathcal{L}_{cls}(\hat{\mathbf{c}}_t^{\sigma_t(m)}, c_t^m) + \sum_{m=1}^{M_t} \lambda_{box} \cdot \mathcal{L}_{box}(\hat{\mathbf{b}}_t^{\sigma_t(m)}, \mathbf{b}_t^m)\right]$$

- Classification loss: cross-entropy
- Localization loss: L1 + GIoU combination
- DETR-style Hungarian matching for prediction–ground-truth assignment
- $\lambda_{cls} = 2$, $\lambda_{box} = 5$

**Implementation Details**:
- Pretrained OWL ViT-L/16 encoder
- 4-head cross-attention with 1024-dimensional hidden states
- 2-layer MLP classification and localization heads with 512-dimensional hidden states
- AdamW optimizer, learning rate 1e-5, cosine schedule with linear warmup

## Key Experimental Results

### Main Results

**Performance Comparison under 5-Shot Setting (Table 1)**:

| Method | Type | FSVOD-500 AP | FSYTV-40 AP | VidOR AP | VidVRD AP |
|--------|------|-------------|------------|---------|---------|
| FR-CNN | Non-temporal | 18.2 | 9.3 | 21.6 | 14.3 |
| OWL-ViT | Non-temporal | 14.8 | 10.6 | 30.1 | 23.6 |
| UP-DETR | Non-temporal | 20.1 | 11.8 | 39.7 | 11.6 |
| ByteTrack+CLIP | Temporal | 14.7 | 7.9 | 24.9 | 25.3 |
| FSVOD | Temporal | 25.1 | 14.6 | 45.1 | 40.7 |
| TACF | Temporal | 26.9 | 15.9 | - | - |
| Video OWL-ViT | Temporal | 25.8 | 15.7 | 44.3 | 44.2 |
| QDETRv | Temporal | 26.1 | 14.1 | 43.4 | 42.8 |
| **Ours** | **Temporal** | **30.6** | **21.2** | **49.4** | **48.7** |

The proposed method achieves gains of +3.7%, +5.3%, +4.3%, and +4.5% across the four datasets.

### Ablation Study

**Effect of Temporal Fusion (Table 4)**:

| Temporal Fusion | FSVOD AP/AP50/AP75 | FSYTV AP/AP50/AP75 |
|----------------|-------------------|-------------------|
| ✗ | 25.4/37.1/26.9 | 19.2/26.8/21.4 |
| ✓ | **30.6/42.9/32.1** | **21.2/29.8/23.5** |

Temporal fusion yields +5.2% AP on FSVOD-500 and +2.0% AP on FSYTV-40.

**Effect of Threshold $\tau$**:
- As $\tau$ decreases from 0.98 to 0.70, the average number of propagated detections increases from 2 to 23.
- AP peaks at 30.6 when $\tau = 0.94$ and drops to 17.7 at $\tau = 0.70$.
- $\tau = 0.94$ represents the optimal balance between recall and precision.

**Performance under Different Shot Settings (Table 3)**:

| Setting | Method | FSVOD AP | FSYTV AP | VidOR AP | VidVRD AP |
|---------|--------|---------|---------|---------|---------|
| 1-shot | FSVOD | 20.7 | 11.4 | 36.8 | 34.5 |
| 1-shot | **Ours** | **27.4** | **18.3** | **45.2** | **45.3** |
| 10-shot | FSVOD | 27.2 | 16.1 | 47.3 | 44.7 |
| 10-shot | **Ours** | **33.2** | **23.5** | **51.8** | **50.3** |

The proposed method maintains consistent advantages across all shot settings.

### Key Findings

1. **Temporal modeling is critical for FSVOD**: The non-temporal UP-DETR achieves 20.1% AP versus the proposed method's 30.6% AP — a gap of 10.5%.
2. **Tracking + CLIP is insufficient**: ByteTrack+CLIP achieves only 14.7% AP, 15.9% below the proposed method, indicating that dedicated modules are needed for few-shot scenarios.
3. **High-confidence filtering is crucial**: $\tau = 0.94$ (propagating approximately 2 high-confidence detections) outperforms $\tau = 0.70$ (propagating 23 detections) by 12.9% AP.
4. **Excellent inference speed**: 14 FPS and 8.1 GB GPU memory, 40% faster than Video OWL-ViT.
5. **Error type analysis**: The proposed method exhibits the lowest classification errors (19) and localization errors (5) among all compared methods.

## Highlights & Insights

1. **Elegant design of selective propagation**: Propagating only high-confidence detection features effectively leverages temporal information while preventing noise accumulation — a concise and effective approach.
2. **First introduction of vision-language pretrained models into FSVOD**: The semantic knowledge embedded in OWL-ViT substantially improves generalization to novel categories.
3. **Elimination of proposal networks**: The complex process of adapting region proposal networks for few-shot settings is entirely bypassed.
4. **Consistent effectiveness across all shot settings**: Significant improvements from 1-shot to 10-shot demonstrate the robustness of the method.
5. **Balance between efficiency and accuracy**: An inference speed of 14 FPS is practical for real-world video applications.

## Limitations & Future Work

1. **Only the immediately preceding frame is utilized**: The current design propagates detections from only one previous frame; leveraging longer temporal memory may further improve performance.
2. **Fixed confidence threshold $\tau$**: A dynamic, adaptive threshold may offer greater flexibility.
3. **Simple averaging for support set prototypes**: More sophisticated prototype aggregation strategies (e.g., attention-weighted averaging) may yield better class representations.
4. **Handling of prolonged occlusion**: When an object is occluded across multiple consecutive frames, temporal propagation may be interrupted.
5. **Dependence on large-scale pretraining**: A portion of OWL-ViT's strong performance stems from large-scale pretraining data.

## Related Work & Insights

- **Video OWL-ViT** (Heigold et al. 2023): The base framework of this work, which lacks object-aware temporal modeling.
- **FSVOD** (Qi et al.): The first FSVOD benchmark and proposal-based method.
- **QDETRv** (Kumar et al. 2024): Query-based one-shot video object detection.
- **TACF** (Wentano et al.): Enhances temporal information through contextual fusion.
- **OWL-ViT** (Minderer et al. 2022): A vision-language model for open-vocabulary image detection.

**Insight**: In video understanding, "intelligent filtering" — propagating only useful information — is often more effective than "full propagation" of all frame features. This is consistent with the attention mechanism of the human visual system, which selectively focuses on salient objects in a scene.

## Rating

- Novelty: ⭐⭐⭐⭐ (Object-aware filtered propagation is a compelling idea, though the overall architectural improvements are incremental)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets, multiple shot settings, comprehensive ablations, efficiency analysis, and error analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with well-articulated problem formulation)
- Value: ⭐⭐⭐⭐ (Provides a strong baseline for the emerging FSVOD field; the direction of incorporating OWL-ViT is promising)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](../../ICLR2026/object_detection/fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](../../CVPR2026/object_detection/visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)
- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)

</div>

<!-- RELATED:END -->
