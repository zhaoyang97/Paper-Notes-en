---
title: >-
  [Paper Note] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures
description: >-
  [Human Understanding] TeHOR leverages text descriptions as semantic guidance and jointly optimizes the geometry and texture of 3D humans and objects via Score Distillation Sampling from pretrained diffusion models. This approach eliminates the reliance on contact information required by conventional methods, enabling accurate and semantically consistent 3D reconstruction of both contact and non-contact interactions.
tags:
  - Human Understanding
date: 2026-05-08
content_hash: fb57e7715ed823f8
---

# TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures

## Basic Information

- **Conference**: CVPR 2026
- **arXiv**: [2602.19679](https://arxiv.org/abs/2602.19679)
- **Code**: [Project Page](https://hygenie1228.github.io/TeHOR/)
- **Area**: 3D Vision / Human-Object Reconstruction
- **Keywords**: 3D Human-Object Reconstruction, Text-Guided Optimization, Score Distillation Sampling, 3D Gaussian Splatting, Human-Object Interaction

## TL;DR

TeHOR leverages text descriptions as semantic guidance and jointly optimizes the geometry and texture of 3D humans and objects via Score Distillation Sampling from pretrained diffusion models. This approach eliminates the reliance on contact information required by conventional methods, enabling accurate and semantically consistent 3D reconstruction of both contact and non-contact interactions.

## Background & Motivation

Jointly reconstructing 3D humans and objects from a single image is a key task for understanding human behavior, with broad applications in robotics, AR/VR, and digital content creation. Existing methods suffer from two fundamental limitations:

**Over-reliance on contact information**: Existing methods (e.g., PHOSA, CONTHO, InteractVLM) primarily use human-object contact regions as the core cue for interaction reasoning, enforcing geometric proximity in contact areas through iterative fitting. However, a large proportion of real-world interactions are non-contact in nature (e.g., gazing at or pointing toward an object), rendering contact information entirely ineffective. Even when contact exists, erroneous contact predictions directly lead to reconstruction failure.

**Neglect of global appearance context**: The fitting process in existing methods is driven mainly by local geometric proximity, ignoring the global interaction context provided by appearance cues (color, shading, etc.) of the human and object. This results in globally implausible outputs, such as incorrect object orientation or misaligned human gaze direction.

## Method

### Overall Architecture

TeHOR adopts a two-stage framework: a **Reconstruction Stage** (initialization) and an **HOI Optimization Stage** (joint refinement).

| Stage | Objective | Key Techniques |
|-------|-----------|----------------|
| Reconstruction Stage | Obtain initial 3D human/object/background and text prompts | GPT-4 text generation, LHM human reconstruction, InstantMesh object reconstruction |
| HOI Optimization Stage | Jointly optimize geometry and texture (200 iterations) | SDS appearance loss, contact loss, collision loss |

**Stage 1: Reconstruction Stage**

- **Text Generation**: GPT-4 extracts two types of text prompts from the input image — $P_{\text{holistic}}$ (global interaction description, e.g., "a person riding a bicycle on grass") and $P_{\text{contact}}$ (contacting body parts, e.g., "right hand, left hand").
- **Human Reconstruction**: SmartEraser removes the object → SAM segments the human → LHM generates initial 3D Gaussian attributes $\phi_h$ (40,000 anchor points uniformly sampled on the SMPL-X surface) → Multi-HMR estimates SMPL-X pose $\theta$ and shape $\beta$.
- **Object Reconstruction**: SmartEraser + SAM isolate the object → InstantMesh reconstructs a 3D mesh (Zero123++ first generates 6-view images, then a tri-plane network reconstructs the mesh) → converted to 3D Gaussian attributes $\phi_o$ → ZoeDepth depth alignment estimates object pose $(R, t, s)$.
- **Background Reconstruction**: SmartEraser removes the human and object to obtain a 2D background image used for constructing realistic front-view and novel-view renderings.

### 3D Representation

The human and object are represented by 3D Gaussian sets $\Phi_h$ and $\Phi_o$, respectively:

- **Human Gaussians**: Parameterized as Gaussian attributes $\phi_h$ + SMPL-X pose $\theta$ + shape $\beta$. $\phi_h$ is defined in canonical pose, with each Gaussian anchored to a surface point on the SMPL-X mesh and animated via Linear Blend Skinning (LBS). Hand and face regions follow the original SMPL-X skinning weights; other regions use averaged weights from neighboring vertices.
- **Object Gaussians**: Parameterized as Gaussian attributes $\phi_o$ + rotation $R$ + translation $t$ + scale $s$, defined in canonical space and mapped to final positions via affine transformation.

Advantages of 3D Gaussians over traditional meshes: (1) Gaussians better model high-fidelity visual appearance, providing richer signals for the appearance loss; (2) the flexible topology-free structure allows more effective optimization of human-object spatial relationships.

### Key Designs

The total loss consists of four terms:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{appr}} + \mathcal{L}_{\text{contact}} + \mathcal{L}_{\text{collision}}$$

**1) Reconstruction Loss $\mathcal{L}_{\text{recon}}$**: MSE between the front-view rendering and the input image, including RGB reconstruction error and the discrepancy between human/object silhouettes and segmentation masks, ensuring the reconstruction is consistent with the input image at the observed viewpoint.

**2) Appearance Loss $\mathcal{L}_{\text{appr}}$ (Core Contribution)**: Based on the Score Distillation Sampling (SDS) strategy, this loss leverages the visual priors of pretrained StableDiffusion-v2.1 to align novel-view renderings with the semantics of $P_{\text{holistic}}$:

$$\nabla_{\Phi}\mathcal{L}_{\text{appr}} = \mathbb{E}\left[w_t\left(\hat{\epsilon}_t(\mathbf{x}_t; P_{\text{holistic}}) - \epsilon_t\right)\frac{\partial \mathbf{x}_t}{\partial \Phi}\right]$$

where $t$ is the noise level, $\mathbf{x}_t$ is the noise-augmented rendered image, and $w_t$ is a weighting factor. This loss minimizes the discrepancy between the diffusion model's predicted noise $\hat{\epsilon}_t(\cdot)$ and the true noise $\epsilon_t$, driving the rendered results of the 3D Gaussians toward a plausible appearance distribution conditioned on the text.

Key implementation details:

- Viewpoints are uniformly sampled in spherical coordinates $(r, \upsilon, \psi)$: full-body views with $r \in [1.0, 2.5]$, $\upsilon \in [-30°, 30°]$, $\psi \in [-180°, 180°]$; upper-body zoomed views centered at the SMPL-X spine with $r \in [0.7, 1.5]$.
- Classifier-free guidance (CFG) scale of 15.0; noise timestep randomly sampled within $[0.02, 0.98]$.
- Gradient clipping with maximum norm 1.0.

This design offers two key advantages: (a) text descriptions transcend contact information and can reason about non-contact interactions (e.g., catching a frisbee, gazing at an object); (b) pixel-level dense gradients provide fine-grained spatial supervision, far surpassing the single global vector encoding of CLIP.

**3) Contact Loss $\mathcal{L}_{\text{contact}}$**: Based on $P_{\text{contact}}$, the Gaussian center point set $V_{h,c}$ corresponding to the contacting body parts is identified, and the distance to the nearest object point $V_o$ is minimized:

$$\mathcal{L}_{\text{contact}} = \frac{1}{|V_{h,c}|}\sum_{v_h \in V_{h,c}} d(v_h, V_o) \cdot \mathbb{1}[d(v_h, V_o) < \tau]$$

The threshold $\tau = 10$ cm ensures local physical plausibility. Gradients are computed only for points within the threshold, preventing unrelated distant points from being incorrectly attracted.

**4) Collision Loss $\mathcal{L}_{\text{collision}}$**: Penalizes interpenetration between the human and object by computing the proportion of human vertices lying inside the object mesh, ensuring physical plausibility.

### Gaussians-to-Mesh Conversion

After optimization, the 3D Gaussians must be converted to meshes for evaluation (to enable fair comparison with existing mesh-based methods). Since Gaussians may deviate from the underlying base mesh, inconsistencies can arise in contact regions. The solution identifies contact regions where the human-object Gaussian distance is less than 5 cm, selects the corresponding mesh vertices, and minimizes their inter-distance to zero, achieving contact-consistent conversion.

## Key Experimental Results

### Datasets and Metrics

- **Open3DHOI**: An open-vocabulary in-the-wild 3D HOI dataset with 2.5K+ images and 133 object categories (evaluation only).
- **BEHAVE**: An indoor 3D HOI dataset with 8 subjects × 20 objects and 4.5K test images.
- **Metrics**: $\text{CD}_{\text{human}}$ / $\text{CD}_{\text{object}}$ (Chamfer Distance, cm↓), Contact (F1↑), Collision (interpenetration rate↓).

### Main Results: Comparison with State of the Art (Tab. 4)

| Method | CD↓_human (O3D) | CD↓_obj (O3D) | Contact↑ (O3D) | Coll.↓ (O3D) | CD↓_human (BH) | CD↓_obj (BH) | Contact↑ (BH) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PHOSA | 5.342 | 49.180 | 0.243 | 0.044 | 5.758 | 46.003 | 0.257 |
| LEMON+PICO | 5.948 | 25.889 | 0.335 | 0.078 | 6.159 | 22.585 | 0.082 |
| InteractVLM | 5.252 | 24.238 | 0.392 | 0.054 | 5.770 | 19.197 | 0.379 |
| HOI-Gaussian | 5.111 | 19.363 | 0.348 | 0.070 | 5.748 | 21.774 | 0.371 |
| **TeHOR** | **4.941** | **16.701** | **0.412** | **0.047** | **5.615** | **17.339** | **0.412** |

TeHOR outperforms all prior state-of-the-art methods across all metrics. On Open3DHOI, object CD improves from 19.363 to 16.701 (↓13.7%) and Contact F1 improves from 0.392 to 0.412.

### Non-Contact Scenario Evaluation (Tab. 5)

| Method | CD↓_human | CD↓_object | Collision↓ |
|--------|:---:|:---:|:---:|
| PHOSA | 5.401 | 65.537 | 0.028 |
| InteractVLM | 5.390 | 46.819 | 0.011 |
| HOI-Gaussian | 5.244 | 25.374 | 0.037 |
| **TeHOR** | **4.958** | **17.546** | **0.005** |

The advantage is even more pronounced in non-contact scenarios, with object CD improving from 25.374 to 17.546 (↓30.8%), validating the critical role of text-semantic guidance.

### Ablation Study

**Effect of Text-Guided Optimization (Tab. 1)**:

| Setting | CD↓_human | CD↓_obj | Contact↑ | Collision↓ |
|---------|:---:|:---:|:---:|:---:|
| Before optimization | 5.252 | 31.268 | 0.305 | 0.040 |
| Optimization (w/o text) | 5.028 | 20.348 | 0.374 | 0.052 |
| **Optimization (full)** | **4.941** | **16.701** | **0.412** | **0.047** |

**Loss Function Configuration Ablation (Tab. 2)**:

| $\mathcal{L}_{\text{appr}}$ | $\mathcal{L}_{\text{contact}}$ | CD↓_obj | Contact↑ |
|:---:|:---:|:---:|:---:|
| ✗ | ✓ | 22.094 | 0.330 |
| ✓ | ✗ | 19.849 | 0.374 |
| CLIP substitute | ✓ | 18.504 | 0.366 |
| **✓ (SDS)** | **✓** | **16.701** | **0.412** |

Key finding: The SDS appearance loss substantially outperforms CLIP loss — CLIP encodes to a single 1D vector and cannot model dense spatial relationships, whereas SDS provides pixel-level dense gradients.

**Rendering Component Ablation (Tab. 3)**: Replacing 3D Gaussians with meshes degrades CD_obj to 25.162; removing the 2D background degrades CD_obj to 18.196, demonstrating that the complete scene context is essential for the diffusion prior.

## Highlights & Insights

- **Breaking the contact-dependency paradigm**: TeHOR is the first to incorporate text descriptions into joint 3D human-object reconstruction, enabling reasoning about non-contact interactions (gazing, pointing, catching a frisbee, etc.).
- **SDS appearance optimization**: By exploiting the visual priors of pretrained diffusion models, multi-view SDS achieves fine-grained semantic alignment, which ablation experiments confirm is far superior to CLIP.
- **First joint texture reconstruction**: The paper claims to be the first framework to simultaneously reconstruct complete 3D textures for both humans and objects, directly enabling the generation of immersive digital assets.
- **Thorough experimental design**: Separate evaluations on general and non-contact scenarios, along with five groups of ablation experiments, sufficiently validate the contribution of each component.

## Limitations & Future Work

- The pipeline depends on multiple external models (GPT-4, StableDiffusion, LHM, InstantMesh), resulting in a long dependency chain and high inference cost.
- Approximately 134 seconds per sample on a single RTX 8000 GPU; the 200-step optimization makes real-time application infeasible.
- The appearance loss provides primarily global guidance, with insufficient supervision over local details (small accessories, subtle surface deformations).
- Quantitative evaluation of texture quality is absent due to the lack of 3D HOI datasets with joint geometry and texture annotations.

## Rating

⭐⭐⭐⭐ — The paper clearly identifies the fundamental limitations of existing methods (contact dependency and neglect of global appearance) and proposes a novel and effective text-guided SDS optimization approach. It achieves comprehensive state-of-the-art performance in both general and non-contact scenarios, with a systematic ablation study design. Points are deducted mainly for optimization efficiency and the long dependency chain on multiple external models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](../../ICCV2025/human_understanding/sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](../../ICCV2025/human_understanding/semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)

</div>

<!-- RELATED:END -->
