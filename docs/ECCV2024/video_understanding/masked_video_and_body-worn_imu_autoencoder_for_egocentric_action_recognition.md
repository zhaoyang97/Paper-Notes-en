---
title: >-
  [Paper Note] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition
description: >-
  [ECCV 2024][Video Understanding][Egocentric Action Recognition] This paper proposes EVI-MAE, the first multimodal representation learning method that jointly models egocentric video and body-worn IMUs. Through self-supervised MAE pre-training, it learns cross-modal video-IMU alignment and utilizes a graph neural network to model cooperative movement relationships among multiple IMU devices. It achieves state-of-the-art (SOTA) performance on egocentric action recognition with…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Egocentric Action Recognition"
  - "Inertial Measurement Unit (IMU)"
  - "Multimodal Masked Autoencoder"
  - "Graph Neural Network"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: 79c2b2b89d4fc182
---

# Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition

**Conference**: ECCV 2024  
**arXiv**: [2407.06628](https://arxiv.org/abs/2407.06628)  
**Code**: None  
**Area**: Video Understanding / Action Recognition  
**Keywords**: Egocentric Action Recognition, Inertial Measurement Unit (IMU), Multimodal Masked Autoencoder, Graph Neural Network, Self-Supervised Learning

## TL;DR

This paper proposes EVI-MAE, the first multimodal representation learning method that jointly models egocentric video and body-worn IMUs. Through self-supervised MAE pre-training, it learns cross-modal video-IMU alignment and utilizes a graph neural network to model cooperative movement relationships among multiple IMU devices. It achieves state-of-the-art (SOTA) performance on egocentric action recognition with outstanding robustness.

## Background & Motivation

Egocentric action recognition is a crucial research direction for understanding human behavior. Compared to visual signals, body-worn IMU sensors can precisely capture motion signals while being robust to changing illumination and occlusions. However, both modalities have distinct limitations.

**Limitations of Prior Work**:

**Underutilized IMU Data**: Most egocentric video datasets only feature camera IMUs (head-mounted), which provide motion cues that can already be inferred from the visual stream, failing to capture limb movements. Datasets with body-worn IMUs are exceptionally scarce (the largest contains only 9 hours of labeled video).

**Difficulty in Modeling Multi-Device Relationships**: A single IMU only registers the motion of one body part, whereas action recognition requires a holistic understanding of multi-limb movements. Existing IMU-based methods (such as DeepConvLSTM and LIMU-BERT) neglect the collaborative relationships among multiple devices.

**Scarcity of Labeled Data**: Synchronized video and body-worn IMU labeled data are extremely rare, limiting the performance of fully supervised multimodal frameworks.

**Key Insight**:
- Egocentric video and body-worn IMUs share inherent correspondences: upper-limb movements are reflected in the hands captured on camera, while global movements correspond to lower-limb IMU signals.
- Utilizing unstructured and unlabeled data for self-supervised pre-training can mitigate the scarcity of annotations.
- Multiple IMU devices represent the relative motions of body joints, making them naturally suited for graph structural modeling.

## Method

### Overall Architecture

EVI-MAE consists of two stages: (1) **Self-Supervised MAE Pre-training**: A dual-branch structure where the multimodal pixel reconstruction branch learns cross-modal video-IMU alignment, and the IMU feature reconstruction branch learns multi-device collaborative relationships. (2) **Fine-tuning**: The decoders are discarded, and only the encoders plus a linear classifier are used for action classification.

Input: $T=2$ seconds of synchronized video $D_v \in \mathbb{R}^{T \times S_v \times H \times W \times 3}$ and $N_{imu}=4$ IMU signals $D_{raw} \in \mathbb{R}^{N_{imu} \times T \times S_{imu} \times 3}$.

### Key Designs

1. **IMU Preprocessing and Graph Modeling**: Addresses collaborative relationships among multiple IMUs.

    - After resampling and normalization, the raw acceleration signals from the IMUs are transformed into spectrograms $D_{spec} \in \mathbb{R}^{T_{imu} \times M_{imu}}$ via STFT, which are then split into $16 \times 16$ patches.
    - Features extracted from the spectrogram patches of each IMU by the IMU encoder are used to construct an IMU feature graph $\mathcal{G} = (\mathcal{V}, A, f_d)$.
    - Nodes $\mathcal{V}$ correspond to the $N_{imu}$ IMU devices, and the adjacency matrix $A$ is modeled as fully connected (as actions typically require multi-limb coordination).
    - **Design Motivation**: Inspired by skeleton-based action recognition (e.g., ST-GCN), IMU devices are treated similarly to skeleton joints. Graph structures can effectively capture the relationships between the motions of different body parts.

2. **Multimodal Pixel Reconstruction Branch**: Learns cross-modal video-IMU alignment.

    - Tube masking (masking ratio $R_v = 90\%$) is applied to videos, while unstructured random masking (masking ratio $R_{imu} = 75\%$) is applied to IMU spectrograms.
    - Visible patches are processed by the video encoder (ViT + joint space-time attention) and the IMU encoder, respectively, and are then fused by a unified encoder.
    - The decoders insert learnable mask tokens at the masked positions to reconstruct the original video frames and IMU spectrograms: $O_p = \mathcal{D}_{uni}([e_{imu}, e_v] + [m'_{imu}, m'_v] + [p'_{imu}, p'_v])$.
    - **Design Motivation**: By reconstructing the masked parts of one modality from the visible parts of another, the model is forced to learn the inherent correlations between video and IMU streams.

3. **IMU Feature Reconstruction Branch**: Node-level self-supervised learning of multi-device relationships.

    - A fraction $R_g$ of node features in the IMU feature graph is replaced with mask tokens, generating a corrupted graph $f_{dc}$.
    - A Graph Isomorphism Network (GIN) encoder-decoder is used to reconstruct the original graph: $f_g = \text{GraphEnc}(A, f_{dc}),\ \hat{\mathcal{G}} = \text{GraphDec}(A, f_g)$.
    - **Design Motivation**: By inferring the features of masked devices from the remaining ones, the model learns the collaborative movement patterns across multiple IMUs.

### Loss & Training

The total pre-training loss consists of three terms: $L = \alpha L_{mse} + \beta L_{cos} + \gamma L_{con}$

- **$L_{mse}$**: Mean Squared Error (MSE) loss for pixel reconstruction (video frames and IMU spectrograms).
- **$L_{cos}$**: Cosine similarity loss for IMU feature graph reconstruction: $L_{cos} = \frac{1}{|\mathcal{V}_c|} \sum_{\nu_n \in \mathcal{V}_c} (1 - \frac{(\hat{f}_d^{\nu_n})^T f_d^{\nu_n}}{\|\hat{f}_d^{\nu_n}\| \cdot \|f_d^{\nu_n}\|})$.
- **$L_{con}$**: Cross-modal contrastive loss between video and IMU (InfoNCE-style).
- Hyperparameters: $\alpha=1, \beta=10, \gamma=0.01, \tau=0.05$

## Key Experimental Results

### Main Results

Comparison with SOTA methods on the CMU-MMAC and WEAR datasets:

| Method | Modality | Pre-training | CMU-MMAC (mAP) | WEAR (Acc) |
|------|------|--------|----------------|------------|
| DeepConvLSTM | IMU | ✗ | 7.46 | 74.37 |
| LIMU-BERT | IMU | ✓ | 15.30 | 79.60 |
| **Ours (IMU only)** | IMU | ✓ | **31.68** | **86.53** |
| VideoMAE | Video | ✓ | 84.63 | 88.47 |
| **Ours (Video only)** | Video | ✓ | **85.07** | **89.78** |
| AV-MAE | IMU-Video | ✓ | 84.75 | 91.60 |
| **Ours (IMU-Video)** | IMU-Video | ✓ | **87.96** | **92.78** |

Key Gain: In the IMU-only modality, Ours outperforms LIMU-BERT by **16.38%** mAP on CMU-MMAC.

### Ablation Study

Component ablation study (CMU-MMAC / WEAR):

| Modality | Pre-training | IMU Graph | CMU-MMAC | WEAR |
|------|--------|--------|----------|------|
| IMU | ✗ | ✗ | 24.87 | 79.66 |
| IMU | ✗ | ✓ | 27.36 | 83.55 |
| IMU | ✓ | ✗ | 28.96 | 85.29 |
| IMU | ✓ | ✓ | **31.68** | **86.53** |
| IMU-Video | ✗ | ✗ | 71.53 | 90.93 |
| IMU-Video | ✓ | ✓ | **87.96** | **92.78** |

Robustness to missing IMU devices (masking 2 out of 4 IMUs):

| Method | Normal (CMU) | Missing (CMU) | Drop |
|------|-----------|-----------|----------|
| ADCL | 8.59 | 5.25 | ↓38% |
| LIMU-BERT | 15.30 | 11.40 | ↓25% |
| **Ours (IMU)** | 31.68 | 26.47 | **↓16%** |
| **Ours (IMU-Video)** | 87.96 | 85.86 | **↓3%** |

### Key Findings

- **Graph Modeling is Highly Effective**: Regardless of whether pre-training is applied, incorporating the IMU feature graph consistently boosts performance (+2.49% without pre-training, +2.72% with pre-training on CMU-MMAC).
- **High Masking Ratios Suit IMU Data**: Masking ratios between 75% and 90% yield the best performance, as human motion signals exhibit periodic redundancy.
- **Unstructured Random Masking Slightly Outperforms Structured Masking** (e.g., separate time/frequency masking).
- **Cross-Dataset Generalization**: Adapting pre-trained representations from WEAR to CMU-MMAC via fine-tuning outperforms training from scratch across all modalities.
- **Robustness to Low-Light Conditions**: When video quality degrades, the multimodal model automatically shifts its reliance toward the IMU modality.

## Highlights & Insights

- The first multimodal MAE pre-training method that jointly models egocentric videos and body-worn IMUs.
- An elegant integration of the spatial distribution of IMU devices to construct graph structures, transferring concepts from skeleton-based action recognition to raw sensor signals.
- Meticulously designed experiments: The evaluation does not just analyze standard accuracy, but also thoroughly assesses realistic challenges, such as missing IMU data, video degradation, and cross-dataset generalization.
- The massive performance leap in the IMU-only modality (from 15.30 to 31.68) demonstrates the significant gains that self-supervised pre-training brings to small-scale datasets.

## Limitations & Future Work

- Currently, only 4 IMUs are used with a fully connected graph structure; incorporating more IMUs would require designing more detailed graph topologies.
- The IMU-only accuracy on the CMU-MMAC dataset remains relatively low (31.68%) because IMUs cannot perceive the object/environment interactions involved in the actions.
- The current GIN encoder-decoder setup is relatively simple; future work could explore more powerful graph learning techniques (e.g., GATs, Graph Transformers).
- The framework can be scaled to support joint learning with additional modalities, such as audio and eye-tracking.

## Related Work & Insights

- Compared to VideoMAE, EVI-MAE introduces a complementary IMU modality and graph structure, achieving superior performance in both standard and challenging test scenarios.
- While the efficacy of the MAE framework has been validated in diverse modalities like video (VideoMAE), audio-video (AV-MAE), and point clouds (Point-MAE), this work extends it to the IMU-video combination.
- The design of the IMU feature graph can be generalized to other multi-sensor fusion applications (such as multi-LiDAR setups in autonomous driving).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first to jointly pre-train body-worn IMUs and videos with MAE. Modeling IMU relationships with graph networks is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive, spanning multiple datasets, detailed ablation studies, three realistic challenge scenarios, and cross-dataset generalization.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, although some notations could be further simplified.
- **Value**: ⭐⭐⭐⭐ — Offers an effective multimodal learning paradigm for action recognition using wearable devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Text-Guided Video Masked Autoencoder](text-guided_video_masked_autoencoder.md)
- [\[ECCV 2024\] Data Collection-Free Masked Video Modeling](data_collection-free_masked_video_modeling.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[ECCV 2024\] Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)

</div>

<!-- RELATED:END -->
