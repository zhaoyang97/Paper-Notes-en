---
title: >-
  [Paper Note] egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks
description: >-
  [ICCV 2025][Video Understanding][egocentric vision] This paper introduces egoPPG as a new egocentric vision task, proposes PulseFormer to estimate heart rate (MAE=7.67 bpm) from the eye-tracking cameras of unmodified ego…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "egocentric vision"
  - "heart rate estimation"
  - "rPPG"
  - "eye tracking"
  - "physiological sensing"
date: 2026-05-08
content_hash: d7559f84f2fc34b1
---

# egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks

**Conference**: ICCV 2025
**arXiv**: [2502.20879](https://arxiv.org/abs/2502.20879)
**Code**: [https://siplab.org/projects/egoPPG](https://siplab.org/projects/egoPPG)
**Area**: Video Understanding
**Keywords**: egocentric vision, heart rate estimation, rPPG, eye tracking, physiological sensing

## TL;DR
This paper introduces egoPPG as a new egocentric vision task, proposes PulseFormer to estimate heart rate (MAE=7.67 bpm) from the eye-tracking cameras of unmodified egocentric head-mounted devices, and demonstrates that heart rate estimation improves skill assessment accuracy on EgoExo4D by 14.1%.

## Background & Motivation
Egocentric vision systems (e.g., MR/AR glasses) aim to understand the user's spatial environment and behavior, including movement, activity, and interaction. → However, existing systems overlook a critical dimension: physiological state detection, such as heart rate (HR), which reflects attention, cognitive performance, emotion, stress, and fatigue. → Although Meta's Project Aria 2 introduces contact-based HR sensors, the vast majority of existing devices and recorded datasets (e.g., EgoExo4D) lack this capability. → Core Idea: Leveraging the eye-tracking cameras already present on egocentric head-mounted devices, the method extracts photoplethysmography (PPG) signals from subtle light intensity changes in periocular skin, combined with IMU motion data, to achieve contactless HR estimation.

## Method

### Overall Architecture
PulseFormer takes eye-tracking video (consecutive frame-difference normalization) and IMU data as input, processes them through a 3D CNN backbone (PhysNet) equipped with Motion-aware Temporal Attention (MITA) and Spatial Attention (SA) modules, and outputs a BVP (Blood Volume Pulse) signal. HR is subsequently computed via Butterworth filtering and peak detection. The input consists of $T=128$ frames (4.3 seconds), with resolution downsampled to $48\times128$.

### Key Designs
1. **Motion-aware Temporal Attention (MITA) Module**:

    - Function: Uses IMU data to assign motion-intensity-based temporal weights to each video frame.
    - Mechanism: A ResNet18 with a linear layer extracts image embeddings $\mathbf{F_e} \in \mathbb{R}^{T \times D}$ from video frames; two-layer 1D convolutions extract IMU embeddings $\mathbf{I_e} \in \mathbb{R}^{T \times D}$. Cross-attention is computed as: $\mathbf{A} = \text{softmax}(\frac{\mathbf{QK}^\top}{\sqrt{D}})\mathbf{V}$, where IMU embeddings serve as Q and image embeddings serve as K and V. The output temporal attention $\mathbf{T} \in \mathbb{R}^{T \times 1 \times 1 \times 1}$ is multiplied element-wise with the input per frame.
    - Design Motivation: Egocentric devices suffer from severe motion artifacts during user movement. MITA enables the model to down-weight frames with intense motion (e.g., MAE during dancing drops from 10.54 to 7.85 bpm).

2. **Spatial Attention (SA) Module**:

    - Function: Enables the network to automatically focus on high-SNR periocular skin regions while suppressing the low-SNR influence of the eyeball area.
    - Mechanism: Spatial attention modules are inserted before each pooling layer. Average-pooled and max-pooled feature maps are passed through a $7\times7$ convolution to generate the spatial attention map: $\mathbf{M_s}(\mathbf{F}) = \sigma(f^{7 \times 7}([\mathbf{F_{avg}}; \mathbf{F_{max}}]))$.
    - Design Motivation: Although the bulbar conjunctiva (white of the eye) contains abundant vasculature, eye movement and blinking introduce substantial noise. Periocular skin exhibits less motion and a vascular distribution better suited for PPG extraction. The learned spatial attention maps confirm this finding—the model automatically excludes the eyeball region.

3. **Data Augmentation Strategy**:

    - Function: Addresses variability in the visibility of the eye region across users wearing the device.
    - Mechanism: Random rotation (±20°), random horizontal cropping, and horizontal and vertical flipping.
    - Design Motivation: Device placement varies across individuals, and the camera may capture only the upper, lower, or obliquely oriented portion of the eye.

### Loss & Training
MSE loss is applied on the normalized consecutive differences of the PPG signal as labels. Five-fold cross-validation is performed with participant-level splits, ensuring strict separation of training, validation, and test sets. Batch size=4, 100 epochs, learning rate=0.0009. Training across all folds takes approximately 20 hours on an RTX 4090. The model has only ~12M parameters, with inference speeds of 2.9k fps (RTX 4090) or 180 fps (CPU).

## Key Experimental Results

### Main Results

| Method | MAE↓ | RMSE↓ | MAPE↓ | r↑ |
|--------|------|-------|-------|-----|
| DeepPhys | 28.26 | 31.97 | 36.68 | 0.08 |
| PhysNet | 12.09 | 15.43 | 15.14 | 0.66 |
| PhysFormer | 10.71 | 13.97 | 12.69 | 0.72 |
| FactorizePhys (SOTA) | 10.07 | 13.43 | 12.36 | 0.67 |
| PulseFormer w/o SA | 10.49 | 13.62 | 12.83 | 0.73 |
| PulseFormer w/o MITA | 8.82 | 12.03 | 10.82 | 0.81 |
| **PulseFormer (Ours)** | **7.67** | **10.69** | **9.45** | **0.85** |
| Gain over runner-up | -2.40 | -2.74 | -2.91 | +0.13 |

Downstream Task — EgoExo4D Skill Assessment (Top-1 Accuracy):

| Configuration | Basketball | Cooking | Dancing | Overall |
|---------------|-----------|---------|---------|---------|
| Ego only | 45.45 | 20.00 | 43.44 | 39.69 |
| **Ego + HR** | 47.47 | **40.00** | **53.27** | **45.29** |
| Ego + Exo | 49.49 | 25.00 | 50.82 | 39.00 |
| **Ego + Exo + HR** | **50.50** | **40.00** | **59.84** | **43.94** |

### Ablation Study

| Configuration | MAE↓ | Notes |
|---------------|------|-------|
| Baseline signal processing (skin) | 12.40 | Manually defined skin region + signal processing |
| Baseline signal processing (eye) | 14.60 | Manually defined eye region + signal processing |
| PulseFormer w/o SA | 10.49 | Removing spatial attention increases MAE by 2.82 |
| PulseFormer w/o MITA | 8.82 | Removing motion attention increases MAE by 1.15 |
| **PulseFormer** | **7.67** | Full model |

Per-activity Analysis (MAE):

| Activity | Mean HR | Motion Intensity | PulseFormer | w/o MITA |
|----------|---------|-----------------|-------------|----------|
| Watching video | 71.5 | 0 | 5.52 | 5.97 |
| Office work | 75.7 | 0.45 | 7.50 | 8.22 |
| Kitchen | 85.3 | 0.54 | 7.22 | 8.89 |
| Dancing | 89.1 | 1.00 | 7.85 | 10.54 |
| Cycling | 113.1 | 0.77 | 12.91 | 14.62 |

Frame Rate Impact:

| Frame Rate | MAE↓ | r↑ |
|------------|------|-----|
| 30 fps (original) | 7.67 | 0.85 |
| 10 fps (downsampled) | 11.13 | 0.70 |
| 10→30 fps (interpolated upsampling) | 10.18 | 0.77 |

### Key Findings
- Periocular skin regions yield significantly better PPG signal quality than the eyeball region (signal processing baseline: 12.40 vs. 14.60 MAE).
- MITA is most effective during dancing activities (MAE reduced from 10.54 to 7.85), demonstrating that IMU motion information is critical for suppressing motion artifacts.
- HR information provides the largest gains for skill assessment in cooking (20%→40%) and dancing (43%→53%).
- A frame rate of 10 fps substantially degrades performance, though linear interpolation upsampling partially recovers it.

## Highlights & Insights
- Pioneering introduction of the egoPPG task, bringing physiological sensing into egocentric vision systems and expanding the scope of the field.
- High practicality: requires no hardware modification, operating directly on existing eye-tracking cameras.
- The egoPPG-DB dataset (13 hours, 25 participants) encompasses diverse daily activities and HR variations, with ground-truth validation via both PPG and ECG.
- The downstream task validation is compelling: HR information demonstrably improves performance on vision tasks.

## Limitations & Future Work
- The dataset is relatively limited in scale (25 participants), and demographic diversity warrants expansion.
- MAE remains high at elevated heart rates (e.g., 12.91 bpm during cycling at 113 bpm), suggesting the need for improved modeling of high-HR regimes.
- Downstream benefits are currently validated only on EgoExo4D skill assessment; additional tasks (e.g., emotion recognition, attention detection) remain unexplored.
- Performance degradation at 10 fps limits applicability to existing low-frame-rate datasets.

## Related Work & Insights
- Extending rPPG techniques from facial video to eye-tracking video represents an innovative application transfer.
- The multimodal approach of fusing IMU with video information is generalizable to other vital sign estimation tasks.
- egoPPG provides a physiological state augmentation pathway for large-scale existing datasets (EgoExo4D, Nymeria), constituting a valuable data enrichment paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces a fully novel egoPPG task definition; estimating heart rate from eye-tracking cameras is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ End-to-end validation spanning dataset, method, and downstream tasks, with detailed ablation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, well-defined contributions, and rigorous experimental design.
- Value: ⭐⭐⭐⭐ Establishes a new task, releases an open dataset, and validates downstream utility, with significant impact on the egocentric vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](../../NeurIPS2025/video_understanding/egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)

</div>

<!-- RELATED:END -->
