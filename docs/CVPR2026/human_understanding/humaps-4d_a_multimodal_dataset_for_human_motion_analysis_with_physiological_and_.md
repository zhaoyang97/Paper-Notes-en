---
title: >-
  [Paper Note] HUMAPS-4D: A Multimodal Dataset for HUman Motion Analysis with Physiological and Semantic informations
description: >-
  [CVPR 2026][Human Understanding][Human Motion Analysis] HUMAPS-4D is a large-scale human motion dataset that synchronizes optical motion capture, multi-view RGB, IMU, instrumented pressure insoles, surface electromyography (sEMG), anthropometry, and three-layer semantic annotations under a unified protocol (32 participants × 30 actions × 10 repetitions × 14 hours = 5.76 million frames). Its goal is to establish a rigorous benchmark for inferring full-body 3D poses/actions fro…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Human Motion Analysis"
  - "Plantar Pressure"
  - "Wearable Sensing"
  - "Privacy Protection"
  - "Multimodal Dataset"
date: 2026-05-08
content_hash: 78080e0ab7996f18
---

# HUMAPS-4D: A Multimodal Dataset for HUman Motion Analysis with Physiological and Semantic informations

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Dabrowski_HUMAPS-4D_A_Multimodal_Dataset_for_HUman_Motion_Analysis_with_Physiological_CVPR_2026_paper.html)  
**Code**: Dataset Homepage https://humaps4d.wp.imt.fr/ (Controlled access)  
**Area**: Human Understanding / Multimodal Datasets  
**Keywords**: Human Motion Analysis, Plantar Pressure, Wearable Sensing, Privacy Protection, Multimodal Dataset

## TL;DR
HUMAPS-4D is a large-scale human motion dataset that synchronizes optical motion capture, multi-view RGB, IMU, instrumented pressure insoles, surface electromyography (sEMG), anthropometry, and three-layer semantic annotations under a unified protocol (32 participants × 30 actions × 10 repetitions × 14 hours = 5.76 million frames). Its goal is to establish a rigorous benchmark for inferring full-body 3D poses/actions from physiological signals like plantar pressure without relying on cameras.

## Background & Motivation
**Background**: Human motion understanding is currently driven almost entirely by video data. Visual methods are "data-centric"—leveraging large-scale, multi-view videos with text and action segments to learn general representations. In contrast, biomechanical methods are "individual/pathology-centric"—utilizing motion capture (MoCap), plantar pressure, and sEMG for precise measurements on small cohorts. These two domains have developed independently with little overlap.

**Limitations of Prior Work**: ① Visual sensors invade privacy and are cumbersome/expensive to deploy, making them unsuitable for daily use. ② Lightweight wearables like smartwatches/wristbands provide only coarse signals, incapable of inferring full-body 3D motion. ③ Existing "pressure/pose" datasets (SolePoser, P2P-Insoles, MMVP, etc.) are small-scale, feature few participants and limited actions, and often lack RGB data, semantic annotations, or anthropometry; some even use estimated poses rather than ground truth. **Crucially, no single dataset has unified visual, textual, biomechanical, and anthropometric signals under one standard protocol**, preventing rigorous cross-evaluation of privacy-friendly directions like "plantar pressure to pose."

**Key Challenge**: The biomechanics field has long been stuck in a trade-off between "measurement precision" and "data scale/diversity"—high precision often limits the number of participants/actions, while scale often requires video at the expense of privacy and physiological signals.

**Goal**: To create a dataset with sufficient scale, sensor diversity, and semantic depth to transform the task of "analyzing full-body 3D motion using physiological signals with minimal or no camera reliance" into a standardized, reproducible task with baselines and evaluation protocols.

**Key Insight**: The authors bet on **instrumented insoles (plantar pressure sensing)**—they naturally integrate into daily life (regardless of age, occupation, or attire), have high user acceptance, and provide high-resolution foot-ground interaction dynamics that serve as "physical constraints" for 3D pose models.

**Core Idea**: To **pair** low-level biomechanical signals (plantar pressure, sEMG, IMU) with high-level semantic descriptions (clinical motion assessments, atomic action descriptions, temporal action segments) within the same time-synchronized data. This supports generative/inference models driven by both physical and semantic constraints while eliminating dependence on identifiable visual data.

## Method
As a dataset/benchmark paper, the "Method" refers to the data acquisition system, annotation hierarchy, and benchmark tasks rather than a single network architecture.

### Overall Architecture
Acquisition: All devices are wired to a central computer running Qualisys software. At the start of each recording, the insole software sends a digital trigger to the sEMG and Qualisys systems to **simultaneously** start all devices; another trigger stops them, ensuring hardware-level synchronization. Participants undergo camera calibration, insole/sEMG calibration, and personalized 3D body model generation based on their anthropometry before recording, followed by manual quality inspection.

Six Modalities (all time-synchronized and aligned to a metric reference frame with gravity):
- **MoCap**: 11 Miqus M3 cameras (3 RGB+IR, 8 IR), 120 Hz, calibration accuracy 0.22 mm, providing 42 marker 3D joints + 26 inferred joints and quaternions.
- **RGB**: 3 synchronized cameras, 120 Hz, 720p, including intrinsic/extrinsic parameters for 3D scene reconstruction and pose projection.
- **sEMG**: Delsys Trigno wireless system, 16 electrodes per person at 1259 Hz, with each electrode embedded with a 148 Hz IMU.
- **IPS (Plantar Pressure)**: Moticon OpenGo instrumented insoles, 16 pressure sensors + 1 IMU per foot, 100 Hz, including pressure, acceleration, angular velocity, total force, and center of pressure (CoP).
- **Anthropometry**: Age, gender, height, weight, limb segment lengths, foot length, etc.
- **3-Layer Semantic Annotations**: (See below).

Scale: 32 healthy participants (18–42 years old), 30 actions per session (postural / locomotion / dynamic / interaction), approximately 2'40" per session, 10 sessions with randomized action sequences, totaling 320 instances, 14 hours, and 5.76 million frames.

### Key Designs

**1. Multimodal Unified Acquisition with Hardware Synchronization**

To fill the gap where no dataset aligns four types of signals, the authors use a digital trigger to synchronize insoles, sEMG, and Qualisys at the moment of capture. All modalities are aligned to a metric coordinate system with a gravity vector. To ensure comparability across participants/conditions, three types of signals are explicitly normalized: 3D joints are pelvis-centered and scaled by anthropometry, $\mathbf{p}_i^{\text{norm}} = \mathbf{S}_{\text{participant}}(\mathbf{p}_i - \mathbf{p}_{\text{pelvis}})$, where $\mathbf{S}_{\text{participant}}$ is a diagonal scaling matrix; sEMG is normalized to reference contractions $\text{EMG}^{\text{norm}}_{j,m}(t) = \text{EMG}_{j,m}(t) / \overline{\text{EMG}}^{\text{ref}}_{j,m}$ to eliminate electrode/skin variations; and plantar pressure is normalized by peak step value or body weight $P^{\text{norm}}_{j,s}(t) = P_{j,s}(t)/\max_t P_{j,s}(t)$ or $P_{j,s}(t)/W_j$. This "synchronization + individual normalization" enables comparative joint learning across modalities and participants.

**2. Three-layer Paired Semantic Annotations**

Low-level physiological signals do not inherently carry high-level meaning. The authors pair each recording with time-aligned (indexed to timestamps and frames) natural language corpora: ① **Clinical Motion Assessment Narratives**—high-level judgments of posture, coordination, compensatory strategies, and balance from a physiotherapist's perspective; ② **Atomic Action Descriptions**—short, coarse-grained text for retrieval/captioning; ③ **Temporal Action Segmentation**—frame-level labels of start/end boundaries for structured event sequences. This allows generative models to produce expert-style assessment reports directly from physiological signals.

**3. Privacy-preserving Benchmark Tasks**

The value of the dataset is realized through two core tasks, emphasizing that "inference uses only insoles, while training can leverage other modalities for constraints":

- **Insole-based Action Recognition**: Given a plantar pressure sequence $\mathbf{P}_{1:T}$ from a single insole, identify one of 30 action classes per frame, learning the mapping $f:\mathbf{P}_{1:T}\to\{a_1,\dots,a_{30}\}^T$. Evaluated using a 3-second sliding window and a leave-one-subject-out (LOSO) protocol.
- **Plantar Pressure 3D Pose Inference**: Given pressure $\mathbf{P}_{1:T}$ and auxiliary vectors $\mathbf{A}_{1:T}$ (acceleration, force, CoP), infer 3D coordinates for 16 joints (excluding arms) $f:\mathbf{P}_{1:T},\mathbf{A}_{1:T}\to\mathbf{J}_{1:T}$. The baseline uses a pose encoder during training with MPJPE/MPJAE losses and a "Feature Loss" to constrain encoder consistency, while only pressure is used during validation.

The authors propose a custom metric, **Inconsistency (IS)**, to measure the temporal drift of inferred joints relative to ground truth: $$IS = \frac{1}{n_J}\sum_{i=0}^{n_J\cdot 3 - 1}\sigma_{1:T}(\bar{J}_{i_{1:T}} - J_{i_{1:T}})$$ where $\sigma$ is the standard deviation over the time series. Higher values indicate a lack of coherence in the inferred motion.

## Key Experimental Results

### Dataset Comparison
HUMAPS-4D leads existing datasets in scale, action diversity, modality completeness, and semantic/anthropometric labeling.

| Dataset | Subjects | Actions | Frames | sEMG | Multi-RGB | Semantics | Anthropometry |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SP-Sport (SolePoser) | 28 | 4 | 606 k | No | No | No | No |
| P2P-Insole | 4 | 5 | 14 k | No | No | No | No |
| MMVP | 16 | 6 | 44 k | No | 1 | No | No |
| SIAT-LLMD | 40 | 16 | — | 16 | No | No | ✓ |
| **HUMAPS-4D** | **32** | **30** | **5.76 M** | **16** | **3** | **3 Layers** | **✓** |

### Benchmark Results
Action recognition (LOSO, Accuracy↑) shows that the insole-only modality (82.71%) is weaker than MoCap (89.45%), but their fusion performs best (90.33%). Dynamic actions (jumping) and interactions (upper-limb dominant) are most difficult for pure plantar pressure. Pose inference (4-fold) shows significant error, confirming the inherent difficulty of reconstructing 3D poses from foot signals alone.

| Task | Metric | Insole | MoCap | Insole+MoCap |
| :--- | :--- | :--- | :--- | :--- |
| Action Recog. (Overall) | Acc↑ | 82.71% | 89.45% | **90.33%** |
| Action Recog. (Dynamic) | Acc↑ | 74.36% | 92.05% | 87.57% |
| Action Recog. (Interact.)| Acc↑ | 73.84% | 84.83% | 85.71% |
| Pose Inf. (Overall) | MPJPE↓(cm) | 31.1 | — | — |
| Pose Inf. (Overall) | Inconsist.↓(cm)| 7.5 | — | — |
| Pose Inf. (Overall) | MPJAE↓(°) | 13.3 | — | — |

### Key Findings
- **Multimodal Training for Sparse Inference**: Using MoCap/RGB/sEMG as constraints during training improves top-tier performance for insole-only inference, proving that non-invasive vision-free motion analysis is viable.
- **Action Type Sensitivity**: Dynamic actions (where pressure is absent during flight) and interactions (where the feet don't sense upper-body movement) are the hardest for pressure sensors, but improve significantly with MoCap.
- **Pose Inference as an Open Challenge**: Even with MoCap guidance, MPJPE remains at 31.1 cm, which is too high for many applications. This highlights the fundamental limitations of reconstructing full poses from sparse foot signals.

## Highlights & Insights
- **"Pairing" is the true differentiator**: Unlike datasets that offer either vision or biomechanics, HUMAPS-4D pairs low-level physiological signals with high-level clinical descriptions, enabling "language-conditioned + physically-constrained" generation.
- **Privacy-First Philosophy**: Video faces are blurred, access is controlled, and inference relies solely on insoles—transforming privacy into a first-order design constraint rather than a post-hoc patch.
- **Transferable Inconsistency Metric**: Measuring temporal variance of inferred joints relative to ground truth is a useful metric for any task involving sparse/weak signal reconstruction of temporal structures.
- **Engineering Value of Hardware Sync**: Aligning six types of sensors with differing frequencies (100–1259 Hz) into a unified metric system provides valuable engineering experience for multimodal collection.

## Limitations & Future Work
- The cohort is limited to healthy 18–42 year olds, lacking data from elderly or clinical populations.
- Pose inference baseline error is high (31 cm) and excludes arms, indicating the limits of pure foot signals. This is presented as an open challenge for the community.
- Data collection occurred in a controlled laboratory environment; a gap remains compared to free-living outdoor scenarios.

## Related Work & Insights
- **vs SolePoser / P2P-Insoles**: These were the first to use insole pairs for 3D joint regression, but their data is either not public or lacks semantic context. HUMAPS-4D standardizes these tasks and adds sEMG and anthropometry.
- **vs IMU-based (DIP / Sparse Inertial Poser)**: IMUs face drift and user interference; this work shifts toward insoles as a more "seamless" daily-wear alternative.
- **vs Smart Carpets (Intelligent Carpet)**: Carpets are limited by environment; insoles are mobile and wearable, offering greater generalization potential.

## Rating
- Novelty: ⭐⭐⭐⭐ First to unify vision/text/biomechanics/anthropometry under one protocol with layered semantics.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive tasks and protocols, though pose results are still preliminary.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and normalization details; some benchmark implementation details are relegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a rare, large-scale foundation for privacy-friendly, non-visual motion analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RoMo: A Large-Scale, Richly Organized Dataset and Semantic Taxonomy for Human Motion Generation](romo_a_large-scale_richly_organized_dataset_and_semantic_taxonomy_for_human_moti.md)
- [\[CVPR 2026\] M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction](m4human_a_large-scale_multimodal_mmwave_radar_benchmark_for_human_mesh_reconstru.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](../../ICCV2025/human_understanding/humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[CVPR 2026\] Real-Time Multimodal Fingertip Contact Detection via Depth and Motion Fusion for Vision-Based Human-Computer Interaction](real-time_multimodal_fingertip_contact_detection_via_depth_and_motion_fusion_for.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)

</div>

<!-- RELATED:END -->
