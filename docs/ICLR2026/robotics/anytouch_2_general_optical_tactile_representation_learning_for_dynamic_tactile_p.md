---
title: >-
  [Paper Note] AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception
description: >-
  [ICLR 2026][Robotics][Tactile representation learning] AnyTouch 2 proposes a Tactile Dynamic Pyramid framework, constructs the ToucHD hierarchical dataset comprising 2,426,174 contact samples (covering atomic actions, real-world manipulation, and tactile-force paired data), and designs a unified tactile representation learning framework that operates across three levels of dynamic perception—pixel-level, semantic-level, and physical-level. The approach comprehensively outperforms existing methods on four tasks: static attribute recognition, dynamic physical prediction, and real-world manipulation.
tags:
  - ICLR 2026
  - Robotics
  - Tactile representation learning
  - dynamic perception
  - optical tactile sensors
  - force sensing
  - tactile dataset
date: 2026-05-08
content_hash: 08522f90889c2f75
---

# AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception

**Conference**: ICLR 2026
**arXiv**: [2602.09617](https://arxiv.org/abs/2602.09617)
**Code**: [https://github.com/GeWu-Lab/AnyTouch2](https://github.com/GeWu-Lab/AnyTouch2)
**Area**: Tactile Perception / Robotics
**Keywords**: Tactile representation learning, dynamic perception, optical tactile sensors, force sensing, tactile dataset

## TL;DR
AnyTouch 2 proposes a Tactile Dynamic Pyramid framework, constructs the ToucHD hierarchical dataset comprising 2,426,174 contact samples (covering atomic actions, real-world manipulation, and tactile-force paired data), and designs a unified tactile representation learning framework that operates across three levels of dynamic perception—pixel-level, semantic-level, and physical-level. The approach comprehensively outperforms existing methods on four tasks: static attribute recognition, dynamic physical prediction, and real-world manipulation.

## Background & Motivation
Contact-intensive manipulation in the real world requires robots to perceive temporal tactile feedback, capture subtle surface deformations, and reason about object properties and force dynamics. Optical tactile sensors can provide such rich information; however, existing tactile datasets and models suffer from severe limitations: (1) data predominantly focuses on object-level attributes (e.g., material), neglecting fine-grained temporal tactile dynamics during physical interaction; (2) existing pre-trained models based on image self-supervision or multimodal alignment struggle to capture fine-grained deformation and force-sensing dynamics. The root cause lies in the absence of a systematic dynamic tactile perception paradigm—lacking both a hierarchical framework to guide data collection and a corresponding model design. The core idea of this paper is to establish a Tactile Dynamic Pyramid that systematically advances dynamic tactile perception along both data and model dimensions.

## Method

### Overall Architecture
The core of AnyTouch 2 is a hierarchical design philosophy. The Tactile Dynamic Pyramid stratifies tactile data into five tiers of dynamic perception complexity: T5 (pressing only) → T4 (random actions) → T3 (specific actions) → T2 (manipulation data) → T1 (force data). Correspondingly, the ToucHD dataset covers the three higher-order tiers T3–T1, while the AnyTouch 2 framework progressively builds dynamic perception capabilities from pixel-level → semantic-level → physical-level. The input consists of 4 consecutive tactile frames (after background subtraction), and the output is a unified tactile representation supporting multiple downstream tasks.

### Key Designs

1. **Pixel-Level Dynamic Details**: A Video Masked Autoencoder (VideoMAE) is employed to learn diverse deformation patterns from consecutive frames across multiple optical sensors. The input undergoes background frame subtraction to yield a normalized input $\mathbf{T} \in \mathbb{R}^{N \times H \times W \times 3}$. The video is partitioned into 3D spatiotemporal tokens, tube masking (ratio $\rho=0.75$) is applied, and reconstruction is performed via a frame decoder. The key innovation is the additional introduction of **Frame-difference Reconstruction**: frame differences $D_n = T_n - T_1$ are computed, and a dedicated frame-difference decoder is jointly trained to reconstruct these differences. The total pixel-level loss is $\mathcal{L}_{Pixel} = \mathcal{L}_{rec}^{ori} + \mathcal{L}_{rec}^{dif}$. Frame-difference reconstruction forces the model to attend to subtle inter-frame local changes, which is critical for capturing the highly localized and minute deformations in tactile signals.

2. **Semantic-Level Tactile Features**: Three parallel objectives are used to build semantic understanding. **(a) Multimodal Alignment**: Following the CLIP paradigm, tactile features are aligned with visual and linguistic features: $\mathcal{L}_{Align} = \frac{\alpha_{TV}}{2}(\mathcal{L}_{T\to V} + \mathcal{L}_{V\to T}) + \frac{\alpha_{TL}}{2}(\mathcal{L}_{T\to L} + \mathcal{L}_{L\to T})$. **(b) Cross-Sensor Matching**: Positive and negative sample matching is performed on tactile signals from the same object captured by different sensors, promoting the learning of sensor-agnostic object-level features: $\mathcal{L}_{obj} = -\log\sigma(sim(\mathbf{T}, \mathbf{T}_{obj}^+)) - \log(1 - \sigma(sim(\mathbf{T}, \mathbf{T}_{obj}^-)))$. **(c) Action Matching** (newly introduced): Tactile videos in ToucHD are grouped by 8 atomic action categories (pressing, lifting, sliding in 4 directions, rotating in 2 directions), and the model is trained to pull same-action instances closer and push different-action instances apart, yielding $\mathcal{L}_{act}$. This explicitly injects action-level semantic information into the representation.

3. **Physical-Level Dynamic Properties**: Large-scale tactile-force paired data from ToucHD is leveraged to train a force prediction task. Given a tactile video $\mathbf{T}$, the model predicts the 3D contact force per frame $\mathbf{F} \in \mathbb{R}^{(N-1) \times 3}$. **Delta-force Prediction** is additionally introduced, where $\Delta\mathbf{F}_n = F_n - F_{n-1}$ focuses on temporal force variations rather than static values. The total force loss is $\mathcal{L}_{Force} = \frac{1}{3(N-1)} \|\hat{\mathbf{F}} - \mathbf{F}\|_1 + \frac{1}{3(N-1)} \|\Delta\hat{\mathbf{F}} - \Delta\mathbf{F}\|_1$. This bridges high-level semantic understanding with low-level physical properties, equipping the model with comprehensive representations spanning all pyramid tiers.

4. **ToucHD Dataset**: Contains 2,426,174 contact samples. **(a) Simulated Atomic Action Data (Sim, T3)**: The IMPM simulator is used with 5 optical sensor types performing 4 atomic action categories (sliding, rotating) on 1,043 objects; after rotation augmentation, 8 action categories are obtained, yielding 1,118,896 frames. **(b) Real-World Manipulation Data (Mani, T2)**: A FastUMI gripper is retrofitted with multiple tactile sensors, and 46 manipulation tasks are designed (including capping a pen, inserting a USB, kneading clay, and stacking blocks), yielding 584,842 frames with synchronized video. **(c) Tactile-Force Paired Data (Force, T1)**: Five sensor types, 71 indenters, and robotic arm control are used; multi-directional sliding is performed while a 6-axis force sensor records 3D forces, yielding 722,436 tactile-force pairs.

### Loss & Training
A curriculum task scheduling strategy is adopted: pixel-level reconstruction is trained from scratch with the highest weight, while higher-level tasks are progressively introduced after specific epochs with linearly increasing weights:
$$\mathcal{L}_{total} = \mathcal{L}_{Pixel} + \lambda_{Align}^i \mathcal{L}_{Align} + \lambda_{Match}^i \mathcal{L}_{Match} + \lambda_{Force}^i \mathcal{L}_{Force}$$
Specifically, matching and force prediction tasks are introduced at epoch 20, and alignment at epoch 30. Maximum weights are $\lambda_{Align}^{max}=1.0$, $\lambda_{Match}^{max}=0.02$, $\lambda_{Force}^{max}=0.1$. The model is built on an OpenCLIP-Base encoder and trained for 40 epochs on 4×H100 GPUs.

## Key Experimental Results

### Main Results
**Offline Benchmarks (Object Bench + Sparsh Bench + ToucHD Bench):**

| Task | Sensor | AnyTouch 2 | AnyTouch 1 | MAE(Sparsh) | VJEPA(Sparsh) | Note |
|------|--------|------|----------|------|------|------|
| TAG Material Classification | GS | 76.97% | 71.10% | 67.06% | 66.57% | Acc↑ |
| Cloth Textile Classification | GS | 42.31% | 39.73% | 35.38% | 35.96% | Acc↑ |
| Slip Detection | DG | 86.66 F1 | 81.20 | 82.44 | 83.90 | F1↑ |
| Force Prediction (ToucHD) | DG | 624.26 | 1540.76 | 783.64* | 1232.65 | RMSE(mN)↓ |
| Force Prediction (ToucHD) | Mini | 202.14 | 652.61 | 257.95* | 331.12 | RMSE(mN)↓ |

(*indicates use of ToucHD augmented data)

**Real-World Manipulation Tasks (4 tasks × 20 trials):**

| Task | Pyramid Tier | AnyTouch 2 (DG) | AnyTouch 2 (Mini) | MAE(S)† (DG) | AnyTouch 1 (DG) |
|------|-----------|------|------|------|------|
| Tactile Grasping | T5 | 0.75 | 0.80 | 0.65 | 0.70 |
| Whiteboard Erasing | T4&3 | 0.85 | 0.80 | 0.70 | 0.55 |
| USB Insertion | T2 | 0.30 | 0.25 | 0.20 | 0.10 |
| Chip Moving | T1 | - | 0.85 | - | 0.60 |

### Ablation Study

| Configuration | TAG Acc | ToucHD Force(DG) | ToucHD Force(Mini) | Note |
|------|---------|------|------|------|
| Full AnyTouch 2 | 76.97 | 624.26 | 202.14 | All modules |
| − Frame-diff Reconstruction | 76.19 | 687.13↓ | 225.18↓ | Pixel-level dynamic foundation degrades |
| − Action Matching | 76.56 | 640.15 | 215.83 | Slip detection degrades |
| − Force Prediction | 75.17 | 777.41↓ | 283.59↓ | Force-related tasks degrade significantly |
| − Multimodal Alignment | 71.70↓ | 594.15↑ | 196.10↑ | Static degrades but dynamic improves (interesting) |
| − Full ToucHD | 68.58↓ | 1365.60↓ | 519.55↓ | All tasks degrade comprehensively |

### Key Findings
- Removing multimodal alignment unexpectedly improves dynamic task performance, because coarse-grained text labels pull together same-object samples recorded under different force levels, impairing fine-grained force perception—reflecting a trade-off between static and dynamic perception.
- Removing the ToucHD dataset causes comprehensive degradation across all tasks, validating the irreplaceability of higher-order tier data.
- 4-frame input consistently outperforms 2-frame input; denser dynamic information benefits tactile perception.
- GelSight Mini's clear deformation imaging favors fine-grained attribute tasks, while DIGIT's 30 Hz high sampling rate offers advantages in higher-order manipulation tasks—demonstrating sensor complementarity.
- Replacing the gel pad results in only marginal performance degradation, demonstrating the generalization capability of the sensor-agnostic representation.

## Highlights & Insights
- **Tactile Dynamic Pyramid**: A clear hierarchical framework is proposed that systematically defines tiers of tactile perception capability, providing a unified conceptual paradigm for the field.
- **Data and Model Co-driven Design**: Beyond constructing a large-scale hierarchical dataset, a multi-level learning architecture aligned with the data hierarchy is designed; the two components synergistically reinforce each other.
- **Intriguing Alignment Paradox**: The finding that multimodal alignment improves static understanding but degrades dynamic perception profoundly reveals the limitations of CLIP-style training on fine-grained physical tasks.
- **46 Manipulation Task Design**: ToucHD (Mani) covers an exceptionally rich range of practical manipulation scenarios (from kneading clay to rotating a Rubik's cube), providing a valuable resource for the tactile research community.
- **Physical Significance of Force Prediction**: By explicitly predicting contact forces and their increments, tactile representations are grounded in quantifiable physical quantities, transcending purely semantic understanding.

## Limitations & Future Work
- Data from the DM-Tac W and GelStereo BioTip sensors in ToucHD are not utilized.
- Force data collection is constrained by the simplified indenter-plus-sensor setup; tactile-force data during manipulation of everyday objects is absent.
- Multi-sensor paired manipulation data is used only for alignment, without a dedicated architecture for cross-sensor collaboration.
- The approach is limited to optical tactile sensors and does not extend to array-based tactile sensors.
- Real-world manipulation tasks employ UMI with human hands rather than dual UMI, potentially introducing visual modality bias.

## Related Work & Insights
- **AnyTouch 1**: The predecessor focuses on cross-sensor static feature learning; this work comprehensively introduces the dynamic dimension on that foundation.
- **Sparsh (Meta)**: A tactile self-supervised model based on MAE/VJEPA, but lacking higher-order tier data and force-sensing capability.
- **FeelAnyForce**: A pioneering tactile-force paired dataset, but covering only pressing interactions without complex dynamics such as sliding.
- **Insight**: The hierarchical design philosophy (data tiers → capability tiers → task tiers) can be adapted for pre-training in other perceptual modalities.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)
- [\[NeurIPS 2025\] Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain](../../NeurIPS2025/robotics/task-optimized_convolutional_recurrent_networks_align_with_tactile_processing_in.md)
- [\[ICLR 2026\] RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation](robointer_a_holistic_intermediate_representation_suite_towards_robotic_manipulat.md)
- [\[CVPR 2026\] STRNet: Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation](../../CVPR2026/robotics/strnet_visual_navigation_with_spatio-temporal_representation_through_dynamic_gra.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)

</div>

<!-- RELATED:END -->
