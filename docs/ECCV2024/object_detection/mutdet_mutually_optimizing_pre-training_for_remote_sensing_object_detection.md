---
title: >-
  [Paper Note] MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection
description: >-
  [ECCV 2024][Object Detection][Detection Pre-training] MutDet is proposed, a mutually optimizing pre-training framework for remote sensing oriented object detection. It systematically mitigates the feature discrepancy issue between object embeddings and detector features in detection pre-training via bidirectional cross-attention fusion of object embeddings and encoder features, a contrastive alignment loss, and an auxiliary Siamese head.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Detection Pre-training"
  - "Oriented Object Detection"
  - "Remote Sensing"
  - "DETR"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 74e7c2a61da1fc52
---

# MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2407.09920](https://arxiv.org/abs/2407.09920)  
**Code**: [Available](https://github.com/floatingstarZ/MutDet)  
**Area**: Object Detection (Remote Sensing Oriented Object Detection)  
**Keywords**: Detection Pre-training, Oriented Object Detection, Remote Sensing, DETR, Contrastive Learning

## TL;DR

MutDet is proposed, a mutually optimizing pre-training framework for remote sensing oriented object detection. It systematically mitigates the feature discrepancy issue between object embeddings and detector features in detection pre-training via bidirectional cross-attention fusion of object embeddings and encoder features, a contrastive alignment loss, and an auxiliary Siamese head.

## Background & Motivation

### Problem Introduction

DETR-series detectors have recently achieved success in remote sensing oriented object detection. However, the training and optimization of DETR are challenging and require large-scale annotated datasets. In remote sensing images, objects are densely distributed with arbitrary orientations, making annotation costs extremely high and large-scale annotated datasets difficult to obtain. **Detection pre-training**, which performs unsupervised pre-training of the detection module using pseudo-labels, serves as an effective way to address this issue.

### Limitations of Prior Work

Existing detection pre-training methods are mainly divided into two categories:

**Predictive methods** (e.g., UP-DETR, DETReg): These achieve pre-training by aligning the detector with object embeddings cropped and extracted from a pre-trained backbone. However, **there is a significant feature discrepancy between object embeddings and detector features**, because their feature extraction mechanisms differ—the detector predicts embeddings using image features and the DETR decoder, whereas object embeddings are extracted directly from cropped images by the entire backbone. The complex environments and dense object distributions in remote sensing images further exacerbate this discrepancy.

**Self-supervised learning methods** (e.g., AlignDet, PreSoCo): These achieve self-training by constraining the consistency of instance features under different views, but they miss the valuable visual knowledge contained in the pre-trained backbone.

### Core Motivation

The core motivation of MutDet is: **instead of simply fitting the detector to the object embeddings, the two are mutually optimized**. By utilizing bidirectional fusion to pull the distribution of the two types of features closer, the detector can learn visual knowledge from the pre-trained backbone more effectively, while contrastive learning is used to prevent feature collapse.

## Method

### Overall Architecture

Given an input image $I$, pseudo-labels are first generated offline (oriented bounding boxes are generated using SAM, object embeddings $O$ are extracted using a pre-trained backbone, reduced to 256 dimensions using PCA, and clustered into 256 classes via k-means). The image passes through a frozen backbone and a DETR encoder to obtain encoder features $F$. Subsequently, $F$ and $O$ are fed into the **Mutual Enhancement Module** to obtain the enhanced $F_{enh}$ and $O_{enh}$. $F_{enh}$ is then fed into the DETR decoder to predict embeddings $\hat{Z}_{dec}$, while the **Auxiliary Siamese Head** (utilizing the original $F$) is used to predict $\hat{Z}_{aux}$. Finally, multi-path alignment optimization is accomplished via the **Contrastive Alignment Loss**.

### Key Designs

#### 1. **Mutual Enhancement Module**

**Function**: Bidirectionally fuses object embeddings $O \in \mathbb{R}^{M \times C}$ and encoder features $F \in \mathbb{R}^{K \times C}$ to mitigate the feature discrepancy between them.

**Mechanism**: Three enhancement layers are used, each containing bidirectional cross-attention:

$$O' = \text{LN}(\text{MHSA}(O_i) + O_i), \quad O'' = \text{LN}(\text{MHCA}(O', F_i) + O')$$
$$O_{i+1} = \text{LN}(\text{MLP}(O'') + O''), \quad F_{i+1} = \text{LN}(\text{MHCA}(F_i, O_{i+1}) + F_i)$$

where $O$ is first updated via self-attention and cross-attention over $F$, and then $F$ is enhanced via cross-attention over the updated $O$.

**Design Motivation**: From the perspective of backpropagation, DETReg can only receive supervisory signals from object embeddings through a single path. In contrast, the Mutual Enhancement Module in MutDet fully fuses object embeddings with encoder features during feedforward propagation, allowing supervisory signals to affect various modules of the detector through multiple paths, thereby more effectively learning visual knowledge from the pre-trained backbone.

#### 2. **Contrastive Alignment Loss**

**Function**: Aligns the enhanced embeddings $O_{enh}$ with the predicted embeddings.

**Mechanism**: A bidirectional contrastive learning loss is adopted:

$$\mathcal{L}_{ca}(Z,O) = -\frac{2\tau}{M}\sum_{i=1}^{M}\left[\log\frac{\exp(\mathbf{z}_i \cdot \mathbf{o}_i / \tau)}{\sum_{k=1}^{M}\exp(\mathbf{z}_i \cdot \mathbf{o}_k / \tau)} + \log\frac{\exp(\mathbf{o}_i \cdot \mathbf{z}_i / \tau)}{\sum_{k=1}^{M}\exp(\mathbf{o}_i \cdot \mathbf{z}_k / \tau)}\right]$$

The detector's total alignment loss acts simultaneously on the encoder and decoder predictions:

$$\mathcal{L}_{ca}^{det} = \mathcal{L}_{ca}(\hat{Z}_{enc}^+, O_{enh}) + \mathcal{L}_{ca}(\hat{Z}_{dec}^+, O_{enh})$$

**Design Motivation**: (1) Instance-level contrastive alignment is equivalent to maximizing the mutual information between the distributions of object embeddings and predicted embeddings; (2) since the enhanced object embeddings become learnable, directly applying distillation loss would lead to feature collapse, which is effectively resolved by the negative samples in contrastive learning.

#### 3. **Auxiliary Siamese Head**

**Function**: Mitigates the task gap between pre-training and fine-tuning introduced by the Mutual Enhancement Module.

**Mechanism**: An auxiliary head sharing parameters with the DETR decoder is introduced. It takes the original unenhanced encoder features $F$ (instead of the enhanced $F_{enh}$) as input and directly supervises its output using pseudo-labels:

$$\mathcal{L}_{det}^{aux} = \mathcal{L}_{ca}^{aux} + \mathcal{L}_{cls}^{aux} + \mathcal{L}_{reg}^{aux} + \mathcal{L}_{ang}^{aux}$$

**Design Motivation**: During fine-tuning, object embeddings are unavailable, meaning the Mutual Enhancement Module cannot be utilized, which causes a task gap between pre-training and fine-tuning. The auxiliary Siamese head allows the decoder to simultaneously adapt to the distribution of the original features $F$, while parameter sharing acts as an implicit constraint guiding $F$ to approach $F_{enh}$. Compared to encoder feature distillation (+0.2 AP50) and decoder cross-distillation (+0.5 AP50), the auxiliary Siamese head achieves the best performance (+1.3 AP50).

### Loss & Training

The total loss is defined as: $\mathcal{L}_{mut} = \mathcal{L}_{det} + \mathcal{L}_{det}^{aux}$

where $\mathcal{L}_{det} = \mathcal{L}_{ca}^{det} + \mathcal{L}_{cls} + \mathcal{L}_{reg} + \mathcal{L}_{ang}$ includes contrastive alignment, focal loss classification, GIoU+L1 regression, and angle classification loss.

- **Pre-training**: Freeze the backbone (ImageNet pre-trained), train on DOTA-v1.0 for 36 epochs with the AdamW optimizer and lr=1e-4.
- **Fine-tuning**: Initialize with pre-trained weights and train on downstream datasets for 36 epochs.

## Key Experimental Results

### Main Results

**DIOR-R Dataset (20-class remote sensing oriented detection)**:

| Method | AP50 | AP75 | Gain vs. pre-training free |
|------|------|------|----------------------|
| Pre-training free | 65.7 | 45.7 | - |
| UP-DETR | 67.1 | 48.4 | +1.4 / +2.7 |
| AlignDet | 65.8 | 45.8 | +0.1 / +0.1 |
| DETReg | 67.9 | 49.1 | +2.2 / +3.4 |
| **MutDet (Ours)** | **70.7** | **51.2** | **+5.0 / +5.5** |

**DOTA-v1.0 Dataset (15 classes)**:

| Method | AP50 | AP75 |
|------|------|------|
| Pre-training free | 72.9 | 49.6 |
| DETReg | 72.6 | 49.4 |
| **MutDet (Ours)** | **74.5** | **51.6** |

Note: DOTA is used for both pre-training and fine-tuning. While other methods suffer from negative transfer, MutDet still achieves a 1.6% improvement.

### Ablation Study

**Component Ablation (DIOR-R, 12/24/36 epoch AP50)**:

| Configuration | 12 ep | 24 ep | 36 ep | Explanation |
|------|-------|-------|-------|------|
| DETReg baseline | 62.1 | 65.1 | 67.9 | Baseline |
| + Contrastive loss | 62.6 | 65.8 | 68.5 | +0.6 |
| + Enhanced embedding | 62.8 | 65.9 | 68.7 | +0.8 |
| + Enhanced feature | 65.5 | 67.0 | 69.5 | Core improvement +1.6 |
| + encoder loss + Siamese head | **66.9** | **69.8** | **70.7** | Full MutDet +2.8 |

**Low-data Regime (DIOR-R, k% labeled data)**:

| Labeled Data Ratio | Pre-training free | DETReg | MutDet | MutDet Gain |
|----------|-------------------|--------|--------|------------|
| 10% | 37.9 | 50.8 | **56.9** | +19.0 / +6.1 vs DETReg |
| 25% | 51.1 | 58.4 | **62.9** | +11.8 / +4.5 vs DETReg |
| 50% | 58.8 | 63.1 | **66.7** | +7.9 / +3.6 vs DETReg |
| 100% | 65.7 | 67.9 | **70.7** | +5.0 / +2.8 vs DETReg |

### Key Findings

1. **Less data yields larger pre-training gains**: With 10% of the data, MutDet outperforms DETReg by 6.1% in AP50; using 50% of the data matches the performance of 100% data without pre-training.
2. **MutDet is the only method that still achieves positive gains under pre-training and fine-tuning on the same DOTA dataset**, whereas other methods experience negative transfer.
3. The enhanced feature in the Mutual Enhancement Module is key to performance improvement (boosting AP50 directly by 3.4% at 12 epochs).
4. The auxiliary Siamese head outperforms the encoder distillation and decoder cross-distillation strategies.

## Highlights & Insights

- **First detection pre-training method for the remote sensing domain**, systematically transferring the pre-training paradigm of natural scenes to remote sensing.
- **Ingenious mutually optimizing concept**: Instead of unidirectional distillation/alignment, it allows object embeddings and detector features to bidirectionally fuse before alignment, fundamentally mitigating feature discrepancy.
- Utilizes SAM to generate high-quality remote sensing pseudo-labels (oriented boxes), replacing the traditional Selective Search.
- The design of the auxiliary Siamese head is simple yet effective, implicitly constraining the original features to approach the enhanced ones via parameter sharing.

## Limitations & Future Work

- The pre-training phase still relies on substantial computational resources (4 GPUs, 36 epochs); reducing the pre-training time is a direction worth exploring.
- Validated only on ARS-DETR; whether it is applicable to other DETR variants (such as RT-DETR) or non-DETR detectors remains to be verified.
- The Mutual Enhancement Module is completely discarded during the fine-tuning phase; whether a lightweight alternative can be designed to retain some gains is an open question.
- The current pseudo-label quality is affected by SAM's performance, which may be insufficient for extremely small objects.

## Related Work & Insights

- **DETReg / UP-DETR**: Direct baselines for this work; MutDet systematically improves the alignment mechanism upon them.
- **AlignDet / ProSeCo**: Points out the feature discrepancy issue but circumvents it using self-supervised methods, thereby missing valuable backbone knowledge.
- **SAM**: An effective tool in the remote sensing domain to replace Selective Search for generating pseudo-labels.
- The application of contrastive learning in detection pre-training is worthy of further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The mutually optimizing pre-training paradigm is novel, systematically addressing the feature discrepancy issue.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, multiple training setups (including low-data and few-epochs), and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-explained motivation and methodology.
- **Value**: ⭐⭐⭐⭐ — Pioneering work in remote sensing detection pre-training, delivering prominent improvements in low-data regimes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[CVPR 2026\] Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images](../../CVPR2026/object_detection/balanced_hierarchical_contrastive_learning_with_decoupled_queries_for_fine-grain.md)
- [\[ICCV 2025\] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/object_detection/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)

</div>

<!-- RELATED:END -->
