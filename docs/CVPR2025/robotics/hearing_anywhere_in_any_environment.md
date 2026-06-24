---
title: >-
  [Paper Note] Hearing Anywhere in Any Environment
description: >-
  [CVPR 2025][Robotics][Cross-Room RIR Prediction] Proposes xRIR, a unified model for Room Impulse Response (RIR) prediction that generalizes across rooms. It combines a geometric feature extractor using panoramic depth maps with an acoustic encoder leveraging a few-shot reference RIRs. Supported by the newly constructed AcousticRooms dataset (260 rooms, 300k+ RIRs), it significantly outperforms baseline methods in both seen/unseen simulated environments and real-world environm…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Cross-Room RIR Prediction"
  - "Spatial Acoustics"
  - "Geometric Feature Extraction"
  - "Reference RIR Encoding"
  - "Sim-to-Real Transfer"
date: 2026-05-08
content_hash: 0f4930d245ee89f0
---

# Hearing Anywhere in Any Environment

**Conference**: CVPR 2025  
**arXiv**: [2504.10746](https://arxiv.org/abs/2504.10746)  
**Code**: None  
**Area**: Audio & Speech  
**Keywords**: Cross-Room RIR Prediction, Spatial Acoustics, Geometric Feature Extraction, Reference RIR Encoding, Sim-to-Real Transfer

## TL;DR

Proposes xRIR, a unified model for Room Impulse Response (RIR) prediction that generalizes across rooms. It combines a geometric feature extractor using panoramic depth maps with an acoustic encoder leveraging a few-shot reference RIRs. Supported by the newly constructed AcousticRooms dataset (260 rooms, 300k+ RIRs), it significantly outperforms baseline methods in both seen/unseen simulated environments and real-world environments.

## Background & Motivation

**Background**: Room impulse responses (RIRs) characterize the sound reflection, absorption, and scattering processes from a source to a receiver, serving as a key element for immersive acoustic experiences in mixed reality. Traditional approaches require dense sampling of hundreds of RIR measurements within a room. Recently, implicit neural network-based methods can "compress" dense RIR measurements into a single model, but they are limited to the single room used during training.

**Limitations of Prior Work**: Existing deep learning approaches for RIR prediction (e.g., INRAS, NAFs) are designed for single-room settings. They require retraining with dense RIR data for every new room, failing to generalize to unseen environments with novel geometries and materials. This severely limits their utility in practical applications like VR/AR where rapid adaptation to diverse environments is required.

**Key Challenge**: Room geometries and wall materials vary dramatically across different rooms, and these two factors jointly determine the unique acoustic characteristics of a room. Having a single model handle massive variations in both geometry and materials simultaneously is highly challenging.

**Goal**: (1) How to extract room geometry from easily accessible visual representations; (2) how to quickly capture the acoustic properties of room materials using a few reference RIR measurements; and (3) a large-scale, high-fidelity, multi-room RIR dataset is needed to support cross-room generalization pretraining.

**Key Insight**: Geometry and materials are complementary information sources—panoramic depth maps encode the geometric structure, while a few reference RIRs implicitly encode material properties (via energy decay and reverberation patterns). Fusing both allows approximating the complete acoustic information without explicitly modeling material coefficients.

**Core Idea**: Encode room geometry using panoramic depth maps and acoustic properties using a few reference RIRs, then predict the target RIR via a fusion and weighting module to achieve single-model, cross-room generalization.

## Method

### Overall Architecture

xRIR consists of three components: (1) a geometric feature extractor, which includes a direct-path module (encoding the line-of-sight source-to-receiver features) and a reflection module (modeling the sound reflection paths via walls using panoramic depth maps); (2) a reference RIR encoder, which extracts acoustic features from the log-magnitude spectrograms of $K$ reference RIRs using ResNet-18; and (3) a fusion and weighting module, which fuses the geometric and acoustic features and then uses an attention mechanism and a time-aligned weighting matrix to combine the reference RIR spectrograms to generate the target RIR.

### Key Designs

1. **Reflection module based on panoramic depth maps**:

    - **Function**: Encodes the path features of sound reflection from the source to the wall and then to the receiver.
    - **Mechanism**: (a) Convert the panoramic depth map at the receiver to a 3D coordinate map $I_{coord}$ using an equirectangular projection, where each pixel represents a wall boundary point; (b) for each source, compute the difference map with all boundary points $I_{s,rf} = P_{rel,s} - I_{coord}$ (likewise, compute the difference maps for the receiver and reference sources); these difference maps encode the reflection path information of source $\rightarrow$ boundary point $\rightarrow$ receiver; (c) use a Vision Transformer (6 layers, 8 heads, 512 dimensions) to process the patch features of the difference maps, modeling the spatial dependencies among patches, and finally project them into compact geometric feature vectors.
    - **Design Motivation**: Unlike INRAS, which uses a fixed set of bounce points (failing to unify across rooms), panoramic depth maps offer a room-agnostic geometric representation. The global attention of ViT naturally models the spatial relationships between diverse multipath reflections.

2. **Reference RIR encoder**:

    - **Function**: Captures room material-related acoustic features from a small number of reference RIR measurements.
    - **Mechanism**: For $K$ reference RIRs, compute their respective STFT log-magnitude spectrograms $\mathbf{S}_{ref,k} = \log(\|\text{STFT}(A_{ref,k})\|)$, extract features using ResNet-18, and use the final mean-pooled features $f_a^{(k)}$ to represent the acoustic characteristics of each reference RIR.
    - **Design Motivation**: Material properties are difficult to observe directly, but the energy decay patterns and reverberation characteristics of RIRs implicitly encode material information. A few reference RIRs are used as "acoustic samples" to capture these properties.

3. **Fusion and time-aligned weighting module**:

    - **Function**: Integrates geometric and acoustic features to generate a time-aligned weighted combination of reference RIRs.
    - **Mechanism**: (a) Concatenate the geometric features (direct path + reflection path + receiver reflection) and acoustic features of each reference source into a joint representation $\mathbf{h}_{ref}^{(k)}$; (b) concatenate the geometric features of the target source as $\mathbf{h}_t$, and compute the attention between the target and each reference: $\mathbf{Z} = \text{softmax}(\mathbf{H}_{ref} \cdot \mathbf{h}_t^T / \sqrt{C}) \odot \mathbf{H}_{ref}$; (c) introduce a time base vector $\mathbf{T}_b$ (sinusoidally encoded time steps) and generate a time-aligned weight matrix $\mathbf{W} = \mathbf{Z} \cdot \mathbf{T}_b^T$ via outer product, allowing different weights for different time steps—this is particularly important for RIRs, where early reflections and late reverberations have vastly different characteristics; (d) finally predict $\mathbf{S}_{pred} = \sum_k \mathbf{W}_k \odot \mathbf{S}_{ref,k}$.
    - **Design Motivation**: Different reference RIRs contribute differently to the target RIR across different time periods—spatially close references are more informative for early reflections, whereas late reverberation relies heavily on global room properties.

### Loss & Training

The total loss is the sum of the spectral L1 loss and the energy decay loss: $\mathcal{L}_{total} = \mathcal{L}_{STFT} + \lambda \mathcal{L}_{ED}$, where $\mathcal{L}_{STFT} = \|\exp(\mathbf{S}_{pred}) - \exp(\mathbf{S}_{gt})\|_1$, and $\mathcal{L}_{ED} = \|\text{EDC}(\mathbf{S}_{pred}) - \text{EDC}(\mathbf{S}_{gt})\|_1$. During inference, the predicted spectrogram is converted back to a waveform using the Griffin-Lim algorithm.

## Key Experimental Results

### Main Results (AcousticRooms Cross-Room Prediction)

| Method | Seen EDT↓ | Seen C50↓ | Seen T60↓ | Unseen EDT↓ | Unseen C50↓ | Unseen T60↓ |
|------|-----------|-----------|-----------|-------------|-------------|-------------|
| Few-shot RIR (K=8) | 0.174 | 4.451 | 32.71% | 0.187 | 4.470 | 21.15% |
| Nearest Neighbor (K=8) | 0.064 | 1.717 | 8.94% | 0.090 | 2.667 | 11.64% |
| **xRIR (K=8)** | **0.038** | **0.940** | **8.13%** | **0.055** | **1.457** | **10.53%** |

### Sim-to-Real Transfer (4 Real Rooms)

| Method | Classroom EDT↓ | Corridor EDT↓ | Complex Room EDT↓ |
|------|-----------|-----------|---------------|
| Diff-RIR (K=12) | 0.113 | 0.160 | 0.115 |
| Nearest Neighbor (K=8) | 0.108 | 0.068 | 0.091 |
| **xRIR (K=8)** | **0.093** | **0.062** | **0.077** |

### Key Findings

- xRIR achieves a C50 error of only 0.940 dB in seen environments, which is 45% lower than Nearest Neighbor, indicating that the model learns acoustic principles beyond simple spatial interpolation.
- On unseen environments, xRIR maintains strong performance, with an EDT error of 0.055s, far lower than the 0.187s of Few-shot RIR, verifying its generalization capability across rooms.
- Increasing the number of reference RIRs from $K=1$ to $K=8$ brings continuous improvements, though considerable performance is already achieved at $K=4$.
- Successful sim-to-real transfer: xRIR pretrained on simulation data outperforms Diff-RIR (which was trained specifically on real rooms, using 12 references, whereas xRIR only uses 8).
- On the T60 metric, xRIR is sometimes inferior to Nearest Neighbor, because T60 is a global metric sensitive to late waveforms with a low signal-to-noise ratio (SNR), and learned methods are affected by SNR discrepancies in sim-to-real transfer.

## Highlights & Insights

- **Reference RIRs as material "probes"**: Instead of explicitly modeling material properties, a few RIR measurements are used to implicitly capture the acoustic characteristics of the room. This is a clever workaround, as material properties are notoriously hard to measure directly, but their effects are naturally reflected in RIRs.
- **Time-aligned weighting matrix**: Instead of applying a global weight to the reference RIRs, weights are applied independently at each time step. This aligns with acoustic intuition: early reflections (dependent on geometry) and late reverberations (dependent on materials + geometry) require different reference weights.
- **AcousticRooms Dataset**: Employs the advanced DG (discontinuous Galerkin) method to simulate RIRs, yielding higher fidelity than the PFFDTD method in the GWA dataset. The randomized assignment of 332 types of materials ensures acoustic diversity.

## Limitations & Future Work

- Currently, only single-channel (omnidirectional) RIRs are predicted, without addressing spatial audio (e.g., binaural HRTF), which is crucial for VR immersion.
- It relies on panoramic depth maps as input, which are more costly to acquire than standard RGB images.
- Performance on the T60 metric is suboptimal, which might suggest a need for specialized late-reverberation modeling (e.g., introducing diffusion models to generate waveforms directly rather than spectral weighting).
- Using Griffin-Lim to reconstruct phase during inference introduces artifacts; exploring better phase estimation or direct waveform generation methods is a promising direction.
- While the 260 room categories in AcousticRooms are diverse, they are still limited. Scaling up to more varied building types could further improve generalization.

## Related Work & Insights

- **vs INRAS/NAFs**: These methods train separate implicit neural networks for each room, requiring dense RIR data and failing to generalize across rooms. xRIR handles all rooms with a single unified model, requiring only a few reference RIRs.
- **vs Few-Shot RIR**: Few-Shot RIR also attempts cross-room generalization, but its UNet decoder is sub-optimal for high-fidelity RIR reconstruction. Additionally, its original design uses binaural echoes rather than separate source-receiver position RIRs, leading to poor performance on AcousticRooms.
- **vs Diff-RIR**: Diff-RIR uses a differentiable rendering framework to learn material coefficients, but it requires separate training for each room and is computationally expensive. xRIR serves as a pretrained single model that can adapt to new rooms with minimal fine-tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ Cross-room RIR generalization is a novel problem, and the proposed combination of panoramic depth maps + reference RIRs is reasonable and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive baselines including simulated seen/unseen + real-world sim-to-real environments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, detailed methodology, and high-quality illustrations.
- Value: ⭐⭐⭐⭐ The AcousticRooms dataset and the xRIR framework make practical contributions to the spatial acoustics community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DRAWER: Digital Reconstruction and Articulation with Environment Realism](drawer_digital_reconstruction_and_articulation_with_environment_realism.md)
- [\[ICLR 2026\] WorldGym: World Model as an Environment for Policy Evaluation](../../ICLR2026/robotics/worldgym_world_model_as_an_environment_for_policy_evaluation.md)
- [\[ECCV 2024\] See and Think: Embodied Agent in Virtual Environment](../../ECCV2024/robotics/see_and_think_embodied_agent_in_virtual_environment.md)
- [\[ICLR 2026\] Capturing Visual Environment Structure Correlates with Control Performance](../../ICLR2026/robotics/capturing_visual_environment_structure_correlates_with_control_performance.md)
- [\[CVPR 2026\] GraspALL: Adaptive Structural Compensation from Illumination Variation for Robotic Garment Grasping in Any Low-Light Conditions](../../CVPR2026/robotics/graspall_adaptive_structural_compensation_from_illumination_variation_for_roboti.md)

</div>

<!-- RELATED:END -->
