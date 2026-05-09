---
title: >-
  [Paper Note] Referring Expression Comprehension for Small Objects
description: >-
  [ICCV 2025][Autonomous Driving][Referring Expression Comprehension] This work proposes the SOREC dataset (100K referring expression–bounding box pairs for small objects) and the PIZA adapter module (Progressive-Iterative Zooming Adapter), enabling pretrained models such as GroundingDINO to autoregressively zoom in on extremely small targets, achieving substantial accuracy gains for small-object REC in autonomous driving scenarios.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Referring Expression Comprehension
  - Small Object Detection
  - Parameter-Efficient Fine-Tuning
  - Progressive Zooming
  - GroundingDINO
date: 2026-05-08
content_hash: aee92c06ba21d564
---

# Referring Expression Comprehension for Small Objects

**Conference**: ICCV 2025
**arXiv**: [2510.03701](https://arxiv.org/abs/2510.03701)
**Code**: [GitHub](https://github.com/mmaiLab/sorec)
**Area**: Autonomous Driving
**Keywords**: Referring Expression Comprehension, Small Object Detection, Parameter-Efficient Fine-Tuning, Progressive Zooming, GroundingDINO

## TL;DR

This work proposes the SOREC dataset (100K referring expression–bounding box pairs for small objects) and the PIZA adapter module (Progressive-Iterative Zooming Adapter), enabling pretrained models such as GroundingDINO to autoregressively zoom in on extremely small targets, achieving substantial accuracy gains for small-object REC in autonomous driving scenarios.

## Background & Motivation

Referring Expression Comprehension (REC) aims to localize a specific object in an image given a natural language description. State-of-the-art models have achieved over 90% accuracy on standard benchmarks such as RefCOCO, yet **extremely small object localization** remains a significant challenge.

**Lack of datasets**: Existing REC datasets (e.g., RefCOCO) predominantly contain medium-to-large objects and lack annotations for small targets. In SOREC, the average bounding box occupies only 0.05% of the image area—far smaller than conventional objects.

**Pretrain–finetune gap**: Large-scale pretrained models (e.g., GroundingDINO) perform well on normally sized objects, but low-resolution features are insufficient for fine-grained localization of extremely small targets. Naively rescaling images to the model's input resolution discards critical details.

**Urgent application demand**: Detecting small objects at a distance—pedestrians, traffic signs, street lights—is safety-critical in autonomous driving, yet existing methods perform poorly in such scenarios.

Core insight: When facing a small object, humans **first coarsely localize and then progressively zoom in**. PIZA mimics this behavior through autoregressive zooming.

## Method

### Overall Architecture

PIZA (Progressive-Iterative Zooming Adapter) formulates small-object localization as a search process. Given a pretrained model $F$, PIZA extends it to $F_{\bigcirc *}$, which autoregressively predicts a sequence of bounding boxes:

$$P = (\mathbf{b}_0, \mathbf{b}_1, \cdots, \mathbf{b}_T)$$

where $\mathbf{b}_0$ covers the entire image and $\mathbf{b}_T$ is the final localization of the small target. At each step:

$$\hat{\mathbf{b}}_{i+1} = F_{\bigcirc *}(\mathbf{x}_i, \mathbf{t}, \mathbf{b}_{0:i})$$

where $\mathbf{x}_i$ is the cropped image region at step $i$.

### Key Designs

1. **PIZA Module (Zooming-Step Embedding)**: Inspired by timestep embeddings in diffusion models, PIZA learns a *zooming-step embedding* to represent the progress of the search process. The pipeline is:

    - Extract 6-dimensional low-level features from the bounding box sequence $\mathbf{b}_{0:i}$ (normalized size, relative size, normalized width/height, center coordinates)
    - Derive embedding $\mathbf{h} \in \mathbb{R}^d$ via learnable Fourier embedding + Transformer encoder + average pooling
    - Two prediction heads: an EOS head (binary classification for search termination) and a Progress head predicting search progress $\hat{z} \in [0,1]$
    - Total parameter count: only 0.27M; feature dimension: 16

2. **Three Parameter-Efficient Fine-Tuning Integration Modes**: PIZA can be flexibly integrated into different PEFT methods:

    - **PIZA-CoOp**: inserts the zooming-step embedding $\mathbf{h}$ into the learnable prompt embedding sequence
    - **PIZA-LoRA**: injects $\mathbf{h}$ into the LoRA bottleneck layer: $W\mathbf{x} + BA\mathbf{x} + BC\mathbf{h}$
    - **PIZA-Adapter+**: adds $\mathbf{h}$ to the output of the channel-scaling layer in Adapter+

3. **Extended Training Dataset Construction**: Ground-truth search trajectories are constructed to supervise the autoregressive zooming process. Key steps:

    - Estimate the bounding box area ratio distribution $p(r)$ from the pretraining dataset
    - Determine the zooming factor $z_j^*$ at each step using an exponential weighting scheme, prioritizing precision at early steps
    - Gradually transition the aspect ratio from the full image to that of the target
    - Generate [CONT] and [EOS] labels to supervise the stopping decision

### Loss & Training

- Contrastive loss and localization loss based on GroundingDINO
- A random step from the search trajectory is sampled per mini-batch for forward computation
- AdamW optimizer, learning rate $2 \times 10^{-4}$, decayed by 0.5 at epoch 3, trained for 5 epochs total
- LoRA applied to all self-attention and cross-attention modules with rank=16
- Adapter+ inserted after each self-attention and FFN module

## Key Experimental Results

### Main Results

**Parameter-efficient fine-tuning results on SOREC (Train-L)**

| Method | Params | Val mAcc | Test-A mAcc | Test-B mAcc | Test-A Acc50 | Test-B Acc50 |
|--------|--------|---------|------------|------------|-------------|-------------|
| Zero-shot | 0 | 0.2 | 0.3 | 0.0 | 1.0 | 0.2 |
| Full fine-tuning | 173.0M | 37.4 | 43.8 | 30.5 | 69.6 | 55.6 |
| LoRA | 1.3M | 25.2 | 30.7 | 19.7 | 50.2 | 37.3 |
| PIZA-LoRA | 1.5M | 34.5 | 39.3 | 29.0 | 54.0 | 43.4 |
| Adapter+ | 3.3M | 34.6 | 40.7 | 27.6 | 65.9 | 51.3 |
| **PIZA-Adapter+** | 3.5M | **39.0** | **45.1** | **31.7** | **66.2** | **52.2** |

### Ablation Study

**Contribution of each PIZA component (PIZA-Adapter+, Train-S, mAcc/Acc50/Acc75)**

| Configuration | Val | Test-A | Test-B | Notes |
|---------------|-----|--------|--------|-------|
| w/o PIZA module | 26.0/48.1/24.8 | 32.0/55.0/33.3 | 20.3/40.4/17.9 | No zooming: baseline |
| w/o emb. insertion | 36.7/53.2/41.7 | 42.8/59.2/49.9 | 30.3/45.8/34.0 | No embedding injection |
| **Full PIZA-Adapter+** | **36.8/53.5/41.8** | **43.1/59.6/50.1** | **30.4/45.9/34.1** | Full model |

**Effect of adapter bottleneck dimension**

| Dimension d | Params | Val mAcc | Test-A mAcc |
|-------------|--------|---------|------------|
| 32 | 1.6M | 35.1 | 40.8 |
| 64 | 1.9M | 36.6 | 42.2 |
| 128 | 2.4M | 36.4 | 41.8 |
| 256 | 3.5M | **36.8** | **43.1** |

### Key Findings

- The zero-shot baseline is near zero (mAcc 0.2%), confirming that pretrained models completely fail on extremely small objects.
- PIZA-Adapter+ surpasses full fine-tuning with only 3.5M parameters (vs. 173M), demonstrating the effectiveness of progressive zooming.
- Removing the PIZA module causes mAcc to drop sharply from 36.8 to 26.0, confirming that autoregressive zooming is the core contribution.
- Test-A (traffic signs, etc.) consistently outperforms Test-B (other small objects) by more than 10% in accuracy.
- Performance improves continuously with larger training sets, indicating room for further dataset scaling.
- On average, only 2.11 zooming steps are required to complete localization.

## Highlights & Insights

- **Strong dataset contribution**: SOREC is the first REC dataset targeting small objects in autonomous driving, with objects occupying only 0.05% of image area and expressions averaging 25.5 words (vs. 3.5 in RefCOCO), filling an important gap in the field.
- **Mimicking human visual search**: The progressive zooming strategy is intuitive and efficient, analogous to the human "scan-then-focus" visual search behavior.
- **Parameter efficiency**: The PIZA module introduces only 0.27M parameters and achieves substantial performance gains by flexibly injecting zooming-step information into various PEFT frameworks.
- **Reproducible dataset pipeline**: The semi-automated construction pipeline (SAM + GPT-4o + crowdsourcing) provides a replicable template for creating similar datasets in other domains.

## Limitations & Future Work

- Autoregressive zooming increases the number of inference steps (averaging 2–3 per sample), which may be insufficient for latency-critical applications.
- Validation is currently limited to GroundingDINO; transferability to other foundation models (e.g., GLIPv2, Florence) remains to be explored.
- Dataset expressions are generated by GPT-4o, which may result in limited linguistic diversity (18.45% contain minor errors).
- The reliability of the automatic stopping decision (EOS prediction) warrants further investigation.
- Integration with classical small-object detection techniques such as multi-scale feature extraction could be explored.

## Related Work & Insights

- This work reframes REC from a "single-step localization" problem to a "multi-step search" problem, providing a new paradigm for handling extreme-scale objects.
- The zooming-step embedding in PIZA draws inspiration from timestep embeddings in diffusion models, demonstrating an elegant cross-domain transfer of ideas.
- The SOREC construction pipeline (foundation model segmentation → GPT-generated descriptions → crowdsourced verification) offers a reference for automated training data production.
- The approach has direct implications for long-range object understanding and safety-critical planning in autonomous driving.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The progressive zooming localization concept is intuitive and novel; the dataset fills a critical gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple PEFT methods are compared with thorough ablations, though comparisons against broader REC baselines are limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The paper is well-structured with clearly articulated motivation and detailed dataset construction descriptions.
- **Value**: ⭐⭐⭐⭐ — Both the dataset and the method make important contributions to small-object understanding.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Counting Stacked Objects](counting_stacked_objects.md)
- [\[CVPR 2026\] SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors](../../CVPR2026/autonomous_driving/saber_spatially_consistent_3d_universal_adversarial_objects_for_bev_detectors.md)
- [\[ICCV 2025\] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)
- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](occupancy_learning_with_spatiotemporal_memory.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)

<!-- RELATED:END -->
