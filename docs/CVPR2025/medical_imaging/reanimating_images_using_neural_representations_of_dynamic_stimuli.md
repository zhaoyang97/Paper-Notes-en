---
title: >-
  [Paper Note] Reanimating Images using Neural Representations of Dynamic Stimuli
description: >-
  [CVPR 2025][Medical Imaging][fMRI brain activity decoding] BrainNRDS framework is proposed to decouple static image representations from motion generation. By leveraging fMRI brain activity to decode optical flow information and combining it with motion-conditioned diffusion models, the model generates videos from an initial frame. Additionally, video encoders (VideoMAE) are found to outperform image encoders in predicting brain activity.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "fMRI brain activity decoding"
  - "optical flow prediction"
  - "video diffusion models"
  - "dynamic visual stimuli"
  - "neural representations"
date: 2026-05-08
content_hash: 14d0ea444efeda9f
---

# Reanimating Images using Neural Representations of Dynamic Stimuli

**Conference**: CVPR 2025  
**arXiv**: [2406.02659](https://arxiv.org/abs/2406.02659)  
**Code**: [Project Page](https://brain-nrds.github.io/)  
**Area**: Image Generation  
**Keywords**: fMRI brain activity decoding, optical flow prediction, video diffusion models, dynamic visual stimuli, neural representations

## TL;DR

BrainNRDS framework is proposed to decouple static image representations from motion generation. By leveraging fMRI brain activity to decode optical flow information and combining it with motion-conditioned diffusion models, the model generates videos from an initial frame. Additionally, video encoders (VideoMAE) are found to outperform image encoders in predicting brain activity.

## Background & Motivation

Although computer vision has achieved breakthroughs in static image recognition, it still lags behind humans in understanding complex dynamic motion. For embodied agents operating in motion-rich environments, understanding dynamic scenes is critical. The human brain has evolved highly efficient mechanisms to process spatial and temporal information simultaneously—for instance, when watching a video of walking pedestrians, humans can not only recognize visual features but also infer motion patterns, intentions, and relationships among scene elements.

Existing brain decoding studies mostly focus on static image reconstruction (e.g., using fMRI + Stable Diffusion), whereas research on decoding dynamic visual stimuli remains sparse. Current video decoding methods (e.g., MindVideo) model static and dynamic features jointly, which lacks interpretability and struggles to decode detailed motion information. Although directly using image-conditioned video diffusion models (e.g., SVD) can generate plausible motions, they fail to align with the actual watched motion, as the models merely "hallucinate" plausible optical flows.

The **Key Insight** of BrainNRDS is to **explicitly decouple static image representations and motion representations**. By modeling motion as optical flow, the framework predicts frame-by-frame optical flow conditioned on fMRI brain activity, and then animates the initial frame using a motion-conditioned diffusion model (DragNUWA) based on the decoded motion. This decoupled design not only improves motion decoding accuracy, but more importantly, provides strong interpretability: predicted optical flow can be directly compared quantitatively with ground-truth optical flow and mapped to specific motion-processing regions in the brain.

## Method

### Overall Architecture

The pipeline of BrainNRDS consists of three components: (1) decoding optical flow from fMRI brain activity (motion prediction); (2) reconstructing video using the decoded optical flow via the motion-conditioned diffusion model DragNUWA; and (3) analyzing which visual features best predict voxel-level brain activity using encoding models. The input consists of fMRI voxel data $B_i \in \mathbb{R}^n$ and initial frame image features, while the output is the predicted optical flow field $O_i \in \mathbb{R}^{(T-1) \times H \times W \times 2}$.

### Key Designs

1. **Motion Prediction Module**:
    - **Function**: Decodes visual motion information from fMRI brain activity to predict a quantized optical flow field.
    - **Mechanism**: First, RAFT is used to extract optical flow from ground-truth videos as training labels, which are quantized using k-means into a codebook of $k=40$ clusters. The model $M_\theta$ takes fMRI voxels $B_i$ (across a temporal window of $[i-2, i-1, i, i+1, i+2]$ comprising 5 TRs) and initial frame features $G(\mathcal{I}_{i,1})$ extracted by DINOv2 as inputs. The fMRI data is processed by an MLP and spatially broadcasted before being concatenated with image features, then passed through three residual $1\times1$ convolutional blocks (with dropout) and global average pooling, finally classifying each spatial patch into the codebook. During inference, continuous optical flow is reconstructed through a weighted sum of all classes. Training employs cross-entropy loss.
    - **Design Motivation**: Redefining optical flow prediction as a classification problem (instead of regression), inspired by prior findings where classification-based optical flow prediction outperforms regression; conditioning on the initial frame image features allows the model to focus on extracting dynamic information from fMRI data.

2. **Appearance-Motion Disentanglement**:
    - **Function**: Architecturally separates static image representations from dynamic motion representations to improve interpretability.
    - **Mechanism**: Unlike methods like MindVideo that jointly decode static and dynamic features, BrainNRDS processes the initial frame features (from a frozen DINOv2) and the fMRI signals separately at the input stage. The model "locks" appearance information by conditioning on the initial frame, forcing the fMRI pathway to focus purely on learning motion information. FlowSAM is used to mask salient objects to focus evaluations on critical motion regions.
    - **Design Motivation**: Joint modeling makes it difficult to isolate the individual contributions of static and dynamic features. Decoupling allows for a direct quantitative comparison between predicted and ground-truth optical flow, enabling rigorous motion decoding evaluation.

3. **Video Reanimation & Brain Encoding**:
    - **Function**: Visualizes decoded optical flow as videos and identifies brain regions sensitive to dynamic features.
    - **Mechanism**: Predicted optical flow and the initial frame are fed into a pre-trained DragNUWA model to generate videos. Simultaneously, features are extracted from various visual encoders (VideoMAE, CLIP, DINOv2, VC-1, etc.) to train Ridge regression models to predict voxel-wise fMRI responses. Comparing the prediction performance of different encoders helps identify brain regions highly selective for dynamic features.
    - **Design Motivation**: Optical flow itself is difficult to evaluate visually, so utilizing a diffusion model for visualization makes assessments more intuitive. Brain encoding analysis provides inverse validation—confirming that fMRI indeed contains rich dynamic information.

### Loss & Training

- **Cross-entropy loss**: Treats optical flow prediction as a classification problem, classifying each spatial patch into one of 40 optical flow codebook classes.
- **Data preprocessing**: fMRI data is normalized to zero-mean and unit variance per session, and responses from repeated viewings are averaged.
- **Temporal processing**: Optical flow is downsampled to 3 frames (evenly spaced within a 2-second TR) and spatially downsampled to $32 \times 32$.
- **fMRI Window**: Leveraging hemodynamic response characteristics, data from 2 TRs before and after the current TR are concatenated (5 TRs in total).
- **Encoding models**: Voxel-wise encoding predictions are implemented using Ridge regression.

## Key Experimental Results

### Main Results

Optical flow decoding End Point Error (EPE↓, lower is better):

| Method | S1 | S2 | S3 | Characteristics |
|------|-----|-----|-----|------|
| SVD (Best, 10 samples) | 1.192 | 1.192 | 1.192 | No brain data, initial frame only |
| MindVideo (Best, 100 samples) | — | — | — | Joint decoding |
| **BrainNRDS (Ours)** | **0.543** | **0.572** | **0.634** | Brain data + GT initial frame |

Video generation quality (using initial frames generated by MindVideo):

| Method | VideoMAE CosSim↑ | CLIP CosSim↑ | Pixel SSIM↑ |
|------|-----------------|-------------|-------------|
| MindVideo | 0.742±0.006 | 0.879±0.004 | 0.171±0.02 |
| **Ours end-to-end** | **0.769±0.006** | **0.896±0.003** | **0.214±0.01** |

### Ablation Study

| Encoding Model Class | Best Model | S1 Pearson r | S2 Pearson r | S3 Pearson r |
|-------------|---------|-------------|-------------|-------------|
| Video Self-Supervised | VideoMAE Large | 0.285 | 0.314 | 0.324 |
| Embodied AI Self-Supervised | VC-1 | 0.260 | 0.290 | 0.294 |
| Image Semantic | CLIP ConvNeXt | 0.219 | 0.263 | 0.272 |

### Key Findings

- Models using brain data significantly outperform SVD without brain data in EPE ($p \ll 0.001$), demonstrating that fMRI indeed contains motion information that cannot be obtained solely from static images.
- Video encoders (VideoMAE Large) consistently outperform image encoders in predicting fMRI responses, indicating that fMRI contains rich dynamic information.
- Somatosensory cortical areas (5m, 5mv, 23c, etc.) in the brain are predicted significantly better by VideoMAE, suggesting these regions integrate visual motion and somatosensory information.

## Highlights & Insights

- **The core concept of motion-appearance decoupling is highly valuable**: Separating motion (explicitly represented as optical flow) from appearance not only improves decoding accuracy, but also provides scientific interpretability. It allows for the precise localization of brain regions encoding motion information, offering a valuable tool for understanding the neural mechanisms of dynamic visual processing.
- **The three scenarios of brain-data disambiguation are compelling**: The paper demonstrates three types of ambiguities (action ambiguity, e.g., the flight of an eagle gliding vs. flapping; camera motion ambiguity, e.g., the real motion direction of a space shuttle; static object motion ambiguity, e.g., the camera panning direction relative to the Eiffel Tower). Brain data successfully resolves these ambiguities, unlike purely image-based models.
- **Brain region difference analysis between VideoMAE and CLIP** reveals additional motion and action representation information captured by video models, providing neuroscientific guidance for selecting visual encoders.

## Limitations & Future Work

- Only evaluated on a single dataset (Dynamic Natural Vision), which may limit generalizability.
- Subject-specific training, leaving cross-subject alignment methods unexplored.
- Low temporal resolution of fMRI (2-second TR) limits the capacity to decode fine-grained motions.
- Low spatial resolution of optical flow ($32 \times 32$) limits the reconstruction of fine-grained motions.

## Related Work & Insights

- **vs. MindVideo**: MindVideo decodes appearance and motion jointly, which lacks interpretability; BrainNRDS achieves more precise motion prediction and better video generation quality via decoupled modeling.
- **vs. Stable Video Diffusion**: SVD only conditions on the initial frame to generate video and cannot align with the actual watched motion; brain data provides critical motion disambiguation information.
- **vs. NeuroClips**: NeuroClips uses blurred video as a motion proxy, retaining rough scene composition; BrainNRDS explicitly encodes motion direction and magnitude via optical flow, which is more directly relevant to neural motion processing studies.

## Rating

- Novelty: ⭐⭐⭐⭐ First to explicitly decouple appearance and motion in brain activity decoding, using optical flow as a bridge for motion representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative (EPE, CosSim, SSIM) + qualitative visualization + brain encoding analysis + multi-model comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, well-articulated motivation, and polished illustrations.
- Value: ⭐⭐⭐⭐ Provides a new framework and discoveries for the intersection of neuroscience and computer vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NOIR: Neural Operator Mapping for Implicit Representations](noir_neural_operator_mapping_for_implicit_representations.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)
- [\[ICML 2025\] Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images](../../ICML2025/medical_imaging/context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images.md)
- [\[CVPR 2025\] T-FAKE: Synthesizing Thermal Images for Facial Landmarking](t-fake_synthesizing_thermal_images_for_facial_landmarking.md)
- [\[CVPR 2026\] Decoding 3D Perception via BrainSSD: Synergistic Fusion of EEG Representations from Static and Dynamic Visual Streams](../../CVPR2026/medical_imaging/decoding_3d_perception_via_brainssd_synergistic_fusion_of_eeg_representations_fr.md)

</div>

<!-- RELATED:END -->
