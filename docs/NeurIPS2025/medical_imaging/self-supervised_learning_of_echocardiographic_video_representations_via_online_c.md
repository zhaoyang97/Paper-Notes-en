---
title: >-
  [Paper Note] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation
description: >-
  [NeurIPS 2025][Medical Imaging][Self-supervised learning] This paper proposes DISCOVR, a self-supervised dual-branch framework that transfers fine-grained spatial semantics from an image encoder to the temporal represent…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Self-supervised learning"
  - "echocardiography"
  - "video representation learning"
  - "cross-modal distillation"
  - "semantic clustering"
date: 2026-05-08
content_hash: 8fb85b9f9360d2d7
---

# Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2506.11777](https://arxiv.org/abs/2506.11777)
**Code**: [GitHub](https://github.com/mdivyanshu97/DISCOVR)
**Area**: Medical Imaging
**Keywords**: Self-supervised learning, echocardiography, video representation learning, cross-modal distillation, semantic clustering

## TL;DR

This paper proposes DISCOVR, a self-supervised dual-branch framework that transfers fine-grained spatial semantics from an image encoder to the temporal representations of a video encoder via online semantic cluster distillation, achieving state-of-the-art performance across six cross-population cardiac ultrasound datasets on anomaly detection, classification, and segmentation tasks.

## Background & Motivation

Echocardiographic video understanding poses unique challenges, and existing self-supervised learning (SSL) methods perform poorly in this domain:

**High inter-frame similarity**: The cosine similarity between consecutive echocardiographic frames can reach 0.99 (using pretrained VideoMAE), making frame discrimination extremely difficult. Normal and abnormal hearts are visually nearly identical, yet subtle differences—such as biventricular systolic dysfunction and a dilated spherical left ventricle—are clinically critical.

**Limitations of existing SSL methods**:
- **Masked video modeling** (VideoMAE, MGMAE) focuses on reconstructing low signal-to-noise ultrasound pixels, making it difficult to capture high-level semantics.
- **Contrastive learning** (MoCo, SimCLR) struggles to construct effective positive/negative pairs under high inter-frame similarity.
- **Clustering methods** (SIGMA) rely on aggressive data augmentation, which may destroy clinically relevant anatomical details.

**Lack of domain-specific pretrained models**: Unlike natural images or videos, the echocardiography domain lacks large-scale pretrained backbones.

The core insight of DISCOVR is that echocardiographic analysis requires simultaneous modeling of **temporal dynamics** (e.g., periodic changes in cardiac wall motion and valve function) and **fine-grained spatial semantics** (e.g., atrial septal thickness, endocardial borders), whereas existing methods typically address only one of these aspects.

## Method

### Overall Architecture

DISCOVR is a dual-branch self-supervised framework: the video branch captures global cardiac motion dynamics via masked self-distillation, while the image branch learns fine-grained spatial semantics via masked image self-distillation. The two branches are connected by a Semantic Cluster Distillation (SCD) loss, which transfers evolving anatomical knowledge from the image encoder to the video encoder.

### Key Designs

1. **Video Self-Distillation**: A student-teacher architecture is adopted, where the input video is tokenized into 3D spatiotemporal tube tokens. The teacher encoder $E_{\theta_t}$ processes the full video to produce a global CLS representation $z_t$, while the student encoder $E_{\theta_s}$ processes multiple masked variants $v_{\mathcal{M}_m}$ to produce $z_s^{(m)}$. Teacher parameters are updated via EMA: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$. Alignment is achieved via temperature-scaled softmax and cross-entropy loss: $\mathcal{L}_{\text{ssl}}^{vid} = \frac{1}{M}\sum_{m=1}^M H(P_t, P_s^{(m)})$. **Design motivation**: The student learns to recover complete cardiac motion representations from incomplete video observations.

2. **Masked Image Self-Distillation**: A parallel image encoder $\mathcal{I}_\theta$ processes each video frame independently. The teacher receives complete frames while the student receives $N$ masked variants, trained via an analogous self-distillation loss: $\mathcal{L}_{\text{ssl}}^{img} = \frac{1}{N}\sum_{i=1}^N H(P_t, P_s^{(i)})$. **Design motivation**: The encoder learns spatially grounded representations encoding fine-grained clinical concepts such as fetal cardiac valves, ventricular anatomy, and atrial septal contours.

3. **Semantic Cluster Distillation (SCD)**: This is the core innovation of DISCOVR. Token-level features $\hat{\mathbf{z}}_v$ reconstructed by the video decoder and spatial features $\hat{\mathbf{z}}_i$ produced by the image encoder (with stop-gradient) are projected onto a shared set of learnable prototypes $P \in \mathbb{R}^{K \times D}$. Soft cluster assignments are generated via the Sinkhorn-Knopp algorithm and aligned using symmetric cross-entropy: $\mathcal{L}_{\text{SCD}} = \text{CE}(\mathbf{s}_v, \text{stopgrad}(\mathbf{q}_i)) + \text{CE}(\mathbf{s}_i, \text{stopgrad}(\mathbf{q}_v))$. Gradients propagate only through the video model and prototype matrix; the image encoder is updated solely via its own self-distillation loss. **Design motivation**: Spatial semantic clusters discovered by the image encoder are anchored to the token representations of the video encoder, enriching temporal features with fine-grained anatomical detail.

### Loss & Training

- **Total loss**: $\mathcal{L} = \mathcal{L}_{\text{ssl}}^{vid} + \mathcal{L}_{\text{ssl}}^{img} + \mathcal{L}_{\text{SCD}}$
- Input configuration: 64-frame video clips sampled at stride 3; spatiotemporal tube embedding $2 \times 16 \times 16$; masking ratio of 90%
- ViT-Base backbone; teacher and student use separate projection heads
- Trained exclusively on normal videos, treating pathology as deviation from normal cardiac dynamics
- **No pretrained models, labels, or aggressive data augmentation required**

## Key Experimental Results

### Main Results

Zero-shot anomaly detection (kNN classifier, evaluated on six datasets):

| Dataset | Population | DISCOVR F1 | Best Baseline F1 | Baseline | Gain |
|---|---|---|---|---|---|
| EchoNet-Dynamic | Adult | **61.45** | 57.56 | MVD | +6.8% |
| RVENET | Pediatric/Adult | **53.88** | 52.18 | MNAD | +3.3% |
| EchoPediatric-LVH | Pediatric | **54.63** | 51.31 | C2FPL | +6.5% |
| FetalEcho 1 | Fetal | **61.79** | 60.64 | MGMAE | +1.9% |
| FetalEcho 2 | Fetal | **56.69** | 56.09 | MGMAE | +1.1% |

Linear probe classification:

| Dataset | DISCOVR F1 | SIGMA F1 | VideoMAE F1 |
|---|---|---|---|
| EchoNet-Dynamic | **77.63** | 75.50 | 70.85 |
| FetalEcho 2 | **63.59** | 55.81 | 51.60 |
| RVENET | **62.65** | 58.98 | 59.70 |

### Ablation Study

| Configuration | Balanced Acc. | F1 | Notes |
|---|---|---|---|
| $\mathcal{L}_{\text{ssl}}^{vid}$ only | 52.27 | 48.23 | Video self-distillation only; lacks spatial semantics |
| $\mathcal{L}_{\text{ssl}}^{vid} + \mathcal{L}_{\text{SCD}}$ (full) | **63.20** | **61.45** | SCD yields +13.22% F1 gain |
| ViT-Small | 59.44 | 57.52 | Smaller backbone still performs reasonably |
| ViT-Base | **63.20** | **61.45** | Larger model yields clear improvement |
| 50% masking ratio | 55.60 | 52.98 | Low masking insufficient for robust representation learning |
| 90% masking ratio | **63.20** | **61.45** | High masking promotes semantic learning |
| 16 frames | 57.89 | 55.68 | Short clips insufficient to cover cardiac cycle |
| 64 frames | **63.20** | **61.45** | Long clips are critical for echocardiographic video |

### Key Findings

- **Segmentation**: On the CAMUS dataset with a frozen backbone and a simple linear head, DISCOVR achieves a Dice score of 0.844, surpassing dedicated segmentation architectures such as UNet (0.816) and DeepLabV3 (0.819).
- **LVEF prediction**: Linear probe MAE of 7.79; fine-tuning with 3 blocks achieves 6.32, outperforming several end-to-end supervised baselines (MC3: 6.59, EchoNet-Dynamic: 7.35).
- **Computational overhead**: Training requires only a marginal increase in GPU memory (10.5 GB vs. 9.0–9.5 GB); inference is identical to all compared methods.

## Highlights & Insights

- **Task-agnostic universal representations**: A single self-supervised pretrained model simultaneously excels at anomaly detection, classification, segmentation, and functional assessment, demonstrating strong transferability.
- **Cross-population generalization**: DISCOVR is the first cardiac ultrasound SSL method to systematically evaluate across fetal, pediatric, and adult populations.
- **Elegant design of the SCD loss**: Cross-modal knowledge transfer is realized through prototype clustering, avoiding the difficulties of direct feature alignment.
- **Importance of long-sequence modeling**: 64-frame clips (covering approximately two cardiac cycles) offer a significant advantage over short clips, providing important guidance for the echocardiographic video domain.

## Limitations & Future Work

- The video anomaly detection baselines compared (MNAD, MemAE, C2FPL) are primarily designed for natural scenes; comparison with medical-specific anomaly detection methods is insufficient.
- A fixed 90% masking ratio is used; adaptive masking strategies may be better suited to the high redundancy characteristic of echocardiographic video.
- The image and video encoders share the same architecture; exploring heterogeneous architectural designs may yield additional gains.
- Validation on other ultrasound modalities (e.g., Doppler, contrast-enhanced ultrasound) is lacking.

## Related Work & Insights

- **Relation to DINO/iBOT**: DISCOVR extends image self-distillation to a dual-modality setting (image + video) and enables cross-modal connection via SCD.
- **Distinction from SIGMA**: SIGMA uses fixed clustering targets, whereas DISCOVR's image encoder evolves online to provide dynamic semantic guidance.
- **Implications for medical video SSL**: The paradigm of decoupling spatial and temporal learning and recombining them via cluster distillation is generalizable to other medical video modalities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-branch + SCD distillation design is elegant, though each component builds on prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, four tasks, cross-population evaluation, complete ablations, and open-sourced code.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated with rich visualizations (particularly the comparisons in Fig. 1 and Fig. 3).
- **Value**: ⭐⭐⭐⭐⭐ Provides a powerful general-purpose pretraining solution for echocardiographic analysis with broad clinical application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](gflownets_for_learning_better_drug-drug_interaction_representations.md)

</div>

<!-- RELATED:END -->
