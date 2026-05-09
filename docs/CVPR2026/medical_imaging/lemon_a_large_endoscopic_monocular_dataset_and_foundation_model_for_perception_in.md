---
title: >-
  [Paper Note] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings
description: >-
  [CVPR 2026][Medical Imaging][Surgical Foundation Model] This paper presents LEMON, a large-scale endoscopic dataset comprising 4,194 surgical videos (938 hours), and proposes LemonFM, a self-supervised foundation model based on augmented knowledge distillation. LemonFM achieves state-of-the-art performance across four downstream surgical tasks: phase recognition, tool detection, action recognition, and semantic segmentation.
tags:
  - CVPR 2026
  - Medical Imaging
  - Surgical Foundation Model
  - Endoscopic Dataset
  - Self-Supervised Learning
  - Knowledge Distillation
  - Surgical Scene Understanding
date: 2026-05-08
content_hash: 6038e6f80df48474
---

# LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings

**Conference**: CVPR 2026
**arXiv**: [2503.19740](https://arxiv.org/abs/2503.19740)
**Code**: [https://github.com/visurg-ai/LEMON](https://github.com/visurg-ai/LEMON)
**Area**: Medical Imaging / Surgical Vision
**Keywords**: Surgical Foundation Model, Endoscopic Dataset, Self-Supervised Learning, Knowledge Distillation, Surgical Scene Understanding

## TL;DR

This paper presents LEMON, a large-scale endoscopic dataset comprising 4,194 surgical videos (938 hours), and proposes LemonFM, a self-supervised foundation model based on augmented knowledge distillation. LemonFM achieves state-of-the-art performance across four downstream surgical tasks: phase recognition, tool detection, action recognition, and semantic segmentation.

## Background & Motivation

Surgical vision is a core perceptual capability for autonomous surgical robots, requiring models to accurately understand surgical instruments, tissues, and operative phases. However, due to medical data privacy regulations and the difficulty of annotation, existing public surgical datasets are extremely limited in scale—most contain fewer than 100 videos and less than 30 hours of footage, resulting in poor model generalization.

Self-supervised learning offers a promising avenue to address annotation scarcity: pretraining foundation models on large-scale unannotated data can substantially reduce dependence on labeled data. The critical bottleneck, however, lies in **the data itself**—existing attempts either rely on private data (e.g., Endo-FM), making them non-reproducible, or use smaller public datasets (e.g., EndoViT) with limited effectiveness.

Recent works such as GenSurgery and SurgeNetXL attempt to collect surgical videos from the web, but they lack systematic data curation pipelines. The collected videos often contain substantial non-surgical content (e.g., conference presentations, patient interviews, device UI screens), and such noise may introduce spurious features that interfere with model learning.

The core observation of this paper is that **sufficient high-quality surgical videos exist online; the key challenge lies in systematic filtering, cleaning, and annotation**. The authors propose a complete multi-stage data curation pipeline that distills 4,194 high-quality surgical videos from 18K raw YouTube videos, upon which a new self-supervised foundation model is trained.

## Method

### Overall Architecture

The system comprises two core contributions: (1) a multi-stage curation pipeline for the LEMON dataset—processing raw YouTube videos through four stages of video classification, segment selection and clipping, preprocessing, and annotation to yield a clean surgical video collection; and (2) the LemonFM foundation model—a self-supervised pretraining approach based on augmented knowledge distillation within the DINO framework, producing transferable representations for diverse downstream surgical tasks.

### Key Designs

1. **Multi-Stage Data Curation Pipeline**:

    - **Function**: Systematically extract high-quality surgical content from noisy web videos.
    - **Mechanism**: A four-step process—first, a storyboard-based classifier (ResNet18) distinguishes surgical from non-surgical videos; next, a frame-level classifier localizes and clips surgical segments; then, YOLOv8 detects and masks non-surgical regions (e.g., UI overlays, logos) within surgical frames; finally, video titles and ChatGPT are used to match 35 surgical procedure types for annotation, with human quality control throughout.
    - **Design Motivation**: Prior works (GenSurgery/SurgeNetXL) used uncurated videos, where mixed non-surgical content introduces spurious features. Ablation experiments confirm that curated data improves phase recognition by 4.5 pp F1 over the uncurated counterpart.

2. **Augmented Knowledge Distillation**:

    - **Function**: Enables the foundation model to learn invariance to subtle motion and cross-patient appearance variation in surgical scenes.
    - **Mechanism**: Additional supervisory signals $W_i$ are introduced into the DINO teacher–student distillation framework. $W_i$ consists of two types of images: (a) temporally neighboring frames retrieved via KNN search in embedding space from different videos of the same surgical type, included only when their cosine distance to the input frame is less than three times the distance between the input frame and its adjacent frame; and (b) temporally adjacent frames of the input. Training minimizes the cross-entropy loss between teacher and student outputs over both the original augmented views $U_i, V_i$ and the augmented views $W_i$.
    - **Design Motivation**: Surgical scenes exhibit subtle inter-patient variation (e.g., organ coloration) and minor tool motion across adjacent frames. Standard DINO only performs distillation over different augmented views of the same image and cannot explicitly learn such cross-frame or cross-patient invariances. Ablation shows that augmented distillation improves semantic segmentation by 3.2 pp over vanilla DINO.

3. **LemonFM-Vid Video Classification Model**:

    - **Function**: Aggregates frame-level features into a video-level representation for video classification tasks.
    - **Mechanism**: Typicality-based weighted aggregation. Each frame's weight $\omega_j$ is determined by its typicality in embedding space—defined as the inverse of the mean cosine distance to its $K$ nearest neighbors, such that frames more similar to others receive higher weights. The final video embedding $v_e = \sum_j \omega_j \phi_j$ is classified via a single-layer MLP.
    - **Design Motivation**: Surgical videos contain numerous atypical frames (e.g., transition shots, blurred frames) that corrupt simple averaging. Typicality-based weighting allows the model to automatically focus on the most representative surgical frames.

### Loss & Training

The training loss is the cross-entropy between teacher and student networks: $\mathcal{L} = -\sum_i \sum_{u \in U_i} \sum_{v \in V_i \cup W_i, u \neq v} \sum_z P_t(z|u) \log P_s(z|v)$, with output dimension $C = 2^{16}$. The student network is updated via gradient descent; the teacher network is updated via EMA. The backbone is ConvNeXt-L, which ablation studies show outperforms ViT-L for surgical scenes (segmentation +10.7 pp mDice), attributed to the local connectivity inductive bias of convolutions better preserving fine-grained details such as instrument tips.

## Key Experimental Results

### Main Results

**Linear Probing (Frozen Backbone)**

| Dataset | Metric | LemonFM | Prev. SOTA (SurgeNetXL) | Gain |
|--------|------|---------|----------------------|------|
| AutoLaparo | Acc/F1 | **76.4/66.9** | 68.8/57.0 | +7.6/+9.9 |
| Cholec80 | Acc/F1 | **75.8/68.6** | 73.2/65.1 | +2.6/+3.5 |
| GraSP (Tool Detection) | mAP | **76.4** | 62.7 | +13.7 |
| CholecT50 (Action Recognition) | mAP | **50.4** | 45.3 | +5.1 |

**Full Fine-Tuning**

| Dataset | Metric | LemonFM | Prev. SOTA | Gain |
|--------|------|---------|---------|------|
| AutoLaparo | Acc/Jacc | **85.5/64.8** | 85.0/55.3 (SurgeNetXL/Endo-FM) | +9.5 pp Jacc |
| Cholec80 | Acc/Jacc | **92.7/85.1** | 90.3/79.3 (Trans-SVNet) | +5.8 pp Jacc |
| M2CAI16 | Acc/Jacc | **89.9/79.4** | 87.2/74.7 (Trans-SVNet) | +4.7 pp Jacc |
| CholecSeg8k (Segmentation) | mDice | **81.3** | 71.0 (EndoViT) | +10.3 pp |

### Ablation Study

| Configuration | AutoLaparo (Acc/F1) | CholecSeg8k (mDice) | Notes |
|------|-------------------|-------------------|------|
| ImageNet-1K Pretrained | 63.6/53.0 | 64.4 | General pretraining baseline |
| Cholec80 Pretrained | 54.0/46.9 | 64.1 | Small-scale surgical data |
| LEMON Uncurated + DINO | 71.7/61.4 | 67.4 | No data cleaning |
| LEMON Curated + DINO | 75.3/65.9 | 68.7 | With data cleaning |
| LEMON Curated + Aug. Distillation + ViT-L | 75.6/66.1 | 61.2 | ViT backbone |
| LEMON Curated + Aug. Distillation + ConvNeXt-L | **76.4/66.9** | **71.9** | Full model |

### Key Findings

- The data curation pipeline contributes substantially: curation vs. no curation yields +4.5 pp F1, indicating that data quality matters more than quantity.
- ConvNeXt-L substantially outperforms ViT-L (segmentation +10.7 pp), confirming that the local inductive bias of convolutions is better suited to fine-grained surgical structures.
- Discriminative self-supervision (DINO) significantly outperforms generative methods (MAE), with the gap being larger under frozen backbone evaluation.
- LemonFM fine-tuned on only 50% of labeled data still surpasses all other foundation models trained on 100% of data, demonstrating strong data efficiency.
- Standard deviations under 5-fold cross-validation are small (e.g., segmentation 72.7±3.3), indicating stable model behavior.

## Highlights & Insights

- **Elegant data curation pipeline design**: The three-tier filtering from video-level → frame-level → region-level, combining automated classifiers with human quality control, serves as a template for large-scale web data cleaning. The use of storyboards for video-level classification is particularly efficient.
- **Neighbor selection strategy in augmented distillation**: Using "cosine distance < 3× adjacent-frame distance" as a threshold for cross-video neighbor selection ensures visual similarity while avoiding over-matching. This adaptive threshold design is transferable to other video self-supervised learning tasks.
- **Surpassing SOTA with 50% data** is the most compelling experiment—demonstrating that improved pretraining quality can substantially reduce downstream annotation requirements, which is highly significant for annotation-expensive medical domains.

## Limitations & Future Work

- Data sourced exclusively from public YouTube videos; despite curation, quality remains inferior to standardized hospital-collected data.
- Coverage is limited to 35 minimally invasive surgical procedure types; open surgery and other modalities are not addressed.
- The current model is purely image-based and does not fully exploit video temporal information (though experiments demonstrate that image model + TCN is already competitive and faster).
- Surgical type classification achieves only 57.8% mAP; anatomically adjacent procedure types (e.g., myomectomy vs. hysterectomy) remain difficult to distinguish.

## Related Work & Insights

- **vs. Endo-FM**: Trained on private data, making it non-reproducible; LemonFM uses fully public data and achieves superior performance.
- **vs. SurgeNetXL**: Also collects data from the web but lacks cleaning; the gap under linear probing is substantial (GraSP: 62.7 vs. 76.4 mAP).
- **vs. EndoViT**: A generative MAE-based method that significantly underperforms discriminative approaches on segmentation (71.0 vs. 81.3 mDice).

## Rating

- **Novelty**: ⭐⭐⭐⭐ Augmented distillation is innovative, though the core framework builds on DINO; the primary contribution lies in dataset construction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four tasks, six datasets, linear probing + full fine-tuning + ablation + cross-validation + low-data regime experiments—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, detailed description of the curation pipeline, intuitive figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Largest public surgical dataset + SOTA foundation model + open-source code; of exceptional value to the surgical vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[CVPR 2026\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)

</div>

<!-- RELATED:END -->
