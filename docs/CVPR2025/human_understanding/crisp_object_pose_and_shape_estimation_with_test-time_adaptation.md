---
title: >-
  [Paper Note] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation
description: >-
  [Human Understanding] Proposes CRISP, a category-agnostic object pose and shape estimation pipeline. The core innovations are an optimization-based corrector utilizing an active shape model and a correct-and-certify self-training strategy, which can adaptively bridge large domain gaps at test time.
tags:
  - "Human Understanding"
date: 2026-05-08
content_hash: b3c722e02ae70dff
---

# CRISP: Object Pose and Shape Estimation with Test-Time Adaptation

| Property | Value |
|------|------|
| Conference | CVPR 2025 |
| arXiv | [2412.01052](https://arxiv.org/abs/2412.01052) |
| Code | [Project Page](https://web.mit.edu/sparklab/research/crisp_object_pose_shape/) |
| Area | Human Understanding / 3D Object Perception |
| Keywords | pose estimation, shape reconstruction, SDF, test-time adaptation, self-training, corrector |

## TL;DR

Proposes CRISP, a category-agnostic object pose and shape estimation pipeline. The core innovations are an optimization-based corrector utilizing an active shape model and a correct-and-certify self-training strategy, which can adaptively bridge large domain gaps at test time.

## Background & Motivation

### Background

Object pose and shape estimation is a key capability for applications such as augmented reality, robotics, and autonomous space docking. Existing methods are roughly classified into: instance-level (requiring CAD models), category-level (requiring category priors), and category-agnostic methods that have emerged recently. Conditional diffusion models show potential in 3D reconstruction, but their inference speed is too slow.

### Limitations of Prior Work

1. **Lack of generalizability**: Category-level methods require specific category priors, making it difficult to generalize to novel categories.
2. **Severe domain gap**: A large discrepancy exists between the distribution of training data (especially synthetic data) and real-world test environments, leading to unusable estimates or even hazards in safety-critical scenarios.
3. **Unstable self-training**: Existing self-supervised methods (such as Chamfer loss) require synthetic data to stabilize training during self-training, making them unsuitable for real-world deployment.

### Goal

1. Design a **category-agnostic** pose and shape estimation pipeline.
2. Provide an **optimization-based corrector** to fix neural network estimation errors.
3. Achieve **synthetic-data-free** test-time self-training to bridge the sim-to-real domain gap.

### Key Insight & Core Idea

By approximating the shape decoder as an active shape model (a convex combination of known shapes), the shape correction problem is formulated as a constrained linear least-squares problem for efficient solving. Subsequently, a correct-and-certify paradigm is employed to generate pseudo-labels for self-training.

## Method

### Overall Architecture

The CRISP pipeline consists of three levels: (1) encoder-decoder shape estimation (DINOv2 backbone + FiLM-conditioned SDF decoder); (2) a DPT network that estimates pose-normalized coordinates (PNC) for pose estimation; (3) an optimization corrector + observable correctness certificates + self-training.

### Key Design 1: FiLM-conditioned Shape Estimation

- **Function**: Estimates the implicit shape representation (SDF) of an object from a single RGB-D image.
- **Mechanism**: The encoder extracts image features using a pre-trained DINOv2 ViT and regresses a latent shape code $\mathbf{h}$ via an MLP; the decoder generates the SDF using a sinusoidal-activated MLP conditioned with FiLM.
- **Design Motivation**: FiLM conditioning yields better implicit field reconstruction performance than concatenation conditioning, and does not require category labels. The category-agnostic design allows the pipeline to be trained on arbitrary CAD model collections.
- **Pose Estimation**: A DPT network directly regresses the pose-normalized coordinates (PNC) for each pixel, and the SE(3) pose is subsequently solved using Arun's method. Note that scale normalization is omitted to prevent PNC degradation during self-training.

### Key Design 2: Optimization Corrector and Active Shape Decoder

- **Function**: Corrects the pose and shape estimation errors from the network.
- **Mechanism**: Formulates a bi-level optimization problem where gradient descent is first used to correct PNC (with the shape code fixed), followed by projected gradient descent to correct the shape code (projected onto the simplex $S_K$ of training shapes). A key innovation is the construction of an active shape decoder: $f_a(\mathbf{z}|\mathbf{c}) = c_0 d_0 f_d(\mathbf{z}|\mathbf{h}) + \sum_{k=1}^{K} c_k d_k f_d(\mathbf{z}|\mathbf{h}_k)$, which reformulates shape correction as a constrained linear least-squares problem: $\min_{\mathbf{c} \geq 0, \mathbf{1}^T\mathbf{c}=1} \|\mathbf{F}(\mathbf{Z})\mathbf{D}\mathbf{c}\|^2$.
- **Design Motivation**: A trained shape decoder behaves well when interpolating in the latent space, but reconstructs implausible shapes during extrapolation (as verified via visualization). Therefore, the shape code is constrained within the convex hull of known shapes, and the linear structure is utilized to achieve efficient solving using interior-point methods.
- **Multi-view Extension**: Aggregates PNC correction results from multiple views for more accurate shape estimation.

### Key Design 3: Correct-and-Certify Self-Training (CRISP-ST)

- **Function**: Performs test-time domain adaptation without using synthetic data.
- **Mechanism**: A three-step workflow: (1) Correct the network output using the corrector; (2) Verify the quality of corrected results using an observable correctness certificate $\text{oc}(\hat{\mathbf{Z}}, \hat{\mathbf{h}}) = \mathbb{I}\{[|f_d(\hat{\mathbf{z}}_i|\hat{\mathbf{h}})|]_p < \epsilon\}$; (3) Use corrected results that pass the quality check as pseudo-labels to train the network.
- **Design Motivation**: Standard self-training is prone to collapsing due to erroneous pseudo-labels. The correctness certificate filters out unreliable estimates by verifying the geometric consistency between depth points and the implicit shape. As training progresses, more estimates pass the verification, establishing a virtuous cycle.
- **Loss Function**: $L_h = \|\hat{\mathbf{h}} - \mathbf{h}\|^2$, $L_z = \sum_i \|\hat{\mathbf{z}}_i - \mathbf{z}_i\|^2$ with the shape decoder frozen.

## Key Experimental Results

### YCBV Dataset

**Shape Estimation ($e_{shape}$ ↓)**:

| Method | Mean | Median | AUC@3cm | AUC@5cm |
|------|------|--------|---------|---------|
| Shap-E | 0.099 | 0.052 | 0.05 | 0.17 |
| CRISP-Syn | 0.045 | 0.032 | 0.18 | 0.35 |
| CRISP-Syn-ST (LSQ) | **0.037** | **0.024** | **0.25** | **0.43** |
| CRISP-Real | 0.026 | 0.016 | 0.40 | 0.58 |

**Pose Estimation (ADD-S)**: After self-training, CRISP-Syn-ST significantly narrows the gap with CRISP-Real.

### Key Findings

- Self-training (only 5 epochs) significantly reduces the sim-to-real gap: $e_{shape}$ decreases from 0.045 to 0.037.
- The LSQ solver generally performs better than the BCD solver (benefiting from the high efficiency and accuracy of solving linear least-squares).
- Including the network-estimated $\mathbf{h}$ as a "basis" in the active shape decoder is crucial; discarding it results in a significant performance drop.
- Supports diverse scenarios including SPE3R (satellites) and NOCS (household items), validating the category-agnostic generalization capability.

### Ablation Study

- Single-view vs. Multi-view corrector: Multi-view correction significantly improves shape estimation quality.
- Shape degeneracy detection: The minimum eigenvalue of $\mathbf{F}(\mathbf{Z})^T\mathbf{F}(\mathbf{Z})$ can serve as an indicator for shape identifiability.

## Highlights & Insights

1. **Elegant theory of Active Shape Decoder**: Formulates non-convex optimization as constrained linear least-squares, offering a closed-form solution with both theoretical guarantees and practical efficiency.
2. **Complete correct-and-certify paradigm**: A three-step closed loop of correction $\rightarrow$ certification $\rightarrow$ self-training, eliminating the need for synthetic data aid.
3. **Discovery of the convex hull behavior in shape decoders**: The observation that extrapolation is unreliable while interpolation is reliable inspired the projection-to-simplex strategy.
4. **Category-agnostic design**: Free from the need for category labels, truly scaling to arbitrary objects.

## Limitations & Future Work

1. The corrector relies on a good initialization; if the neural network estimates are excessively poor, the corrector may converge to local optima.
2. The number of bases $K$ in the active shape decoder is limited by the number of CAD models in the training set, potentially restricting expressivity for highly diverse test objects.
3. Self-training requires a certain number of observations to stabilize, which may limit its effectiveness in extremely few-shot scenarios.
4. The additional computational overhead of the corrector during real-time inference needs to be considered.

## Related Work & Insights

- **NOCS** (CVPR 2019): Pioneering category-level pose and shape estimation. The PNC in CRISP is similar but omits scale normalization.
- **RePoNet**: Utilizes differentiable rendering for training. CRISP employs the correct-and-certify scheme as an alternative.
- **Correct-and-certify by Talak et al.**: CRISP-ST extends this paradigm to simultaneous pose and shape estimation.
- **Insights**: The test-time adaptation concept can be generalized to other 3D perception tasks (e.g., hand pose, human pose estimation). The key lies in properly designing correctors and quality certificates.

## Rating

⭐⭐⭐⭐ — Solid work in both theory and system design. The linearization idea of the active shape model is elegant, and the self-training strategy is highly practical. The insight regarding the convex hull behavior of the shape decoder is particularly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [\[CVPR 2025\] CryptoFace: End-to-End Encrypted Face Recognition](cryptoface_end-to-end_encrypted_face_recognition.md)
- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](../../ICCV2025/human_understanding/semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](../../ICCV2025/human_understanding/sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)

</div>

<!-- RELATED:END -->
