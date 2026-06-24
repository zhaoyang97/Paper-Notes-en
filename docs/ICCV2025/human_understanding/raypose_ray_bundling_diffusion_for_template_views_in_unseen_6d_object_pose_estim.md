---
title: >-
  [Paper Note] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation
description: >-
  [Human Understanding] This work reformulates unseen 6D object pose estimation as a ray alignment problem, proposes an object-centric ray parameterization scheme, and employs a diffusion transformer to infer the 6D pose of a query image from multiple template images with known poses.
tags:
  - "Human Understanding"
date: 2026-05-08
content_hash: 4491990c39a9a49f
---

# RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2510.18521](https://arxiv.org/abs/2510.18521)
- **Code**: [Project Page](https://demianhj.github.io/projects/RayPose)
- **Area**: Human Understanding
- **Keywords**: 6D object pose estimation, unseen objects, diffusion model, ray parameterization, multi-view inference

## TL;DR

This work reformulates unseen 6D object pose estimation as a ray alignment problem, proposes an object-centric ray parameterization scheme, and employs a diffusion transformer to infer the 6D pose of a query image from multiple template images with known poses.

## Background & Motivation

6D object pose estimation is critical for applications such as robotic grasping, AR/VR, and autonomous driving. Existing methods face two core challenges:

**Limitations of template-matching approaches**: Conventional methods first retrieve the closest matching template and then perform alignment; however, retrieval failures directly lead to inaccurate poses. This sequential pipeline decomposes the problem into template retrieval → correspondence estimation → pose prediction → optional refinement, causing errors to accumulate across stages.

**Insufficient constraints from single-view methods**: Multi-view geometry provides essential constraints for 3D perception, yet existing monocular methods cannot leverage such constraints. Although template images inherently constitute posed multi-view observations, prior methods use them only for classification and matching rather than geometric reasoning.

**Gap in applicability of diffusion models**: Methods such as RayDiffusion have demonstrated strong generalization in camera pose estimation, but camera poses are defined in a large-scale world coordinate system whereas object poses reside in a compact object-centric space, making direct transfer ineffective.

## Method

### Overall Architecture

RayPose takes a query image of an unseen object and a set of template images with known poses as input, and progressively denoises structured 2D pose maps via a diffusion transformer to predict the 6D object pose. The core components include a Query Encoder, a Template Encoder (with a Multiview Fuser), and a Diffusion Transformer Decoder.

### Rotation Parameterization — Object-Centric Ray Representation

In contrast to the camera-centric rays in RayDiffusion, this work proposes object-centric rays: the object center is treated as a virtual pinhole camera, and rays are cast from the object center toward the camera coordinate frame. The set of direction vectors is defined as:

$$\mathcal{M}_R = \{d_1, \ldots, d_n\}$$

Each direction vector is normalized to unit length and mapped onto a uniform 2D grid of size $p \times p \times 3$. This representation maps an arbitrary rotation matrix $R$ to a unique structured grid on the unit sphere. Rotation recovery is performed via SVD to solve for the optimal alignment:

$$R^* = \arg\min_{R \in \text{SO}(3)} \sum_{i=1}^n \|Rd_i^* - d_i\|^2$$

### Translation Parameterization — Dense Translation Offset Map

Scale-Invariant Translation Estimation (SITE) is extended to a patch-level dense translation map. Given object translation $t = [t_x, t_y, t_z]$ and camera intrinsics $K$, the object center is projected onto image coordinates $[o_x, o_y, 1]^T = Kt$, and a dense normalized translation offset map is constructed as:

$$\mathcal{M}_T = \left(\frac{u - o_x}{w}, \frac{v - o_y}{h}, \frac{t_z}{r_z}\right)$$

where $w, h$ denote the bounding box width and height, and $r_z$ is a scaling factor. This decoupled representation enables rotation and translation to be predicted independently.

### Multi-View Template-Conditioned Diffusion

**Template Encoder**: A frozen DINOv2 backbone extracts image features. A View Encoder employs three Fourier encoders to process rotation maps, translation maps, and 2D normalized bounding box coordinates. A Multiview Fuser aggregates multi-view information via self-attention.

**Diffusion Process**: The forward process adds Gaussian noise to pose maps: $\mathcal{M}_t = \sqrt{\alpha_t}\mathcal{M}_0 + \sqrt{1-\alpha_t}\epsilon$. The network is trained to predict the clean pose map rather than the noise:

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t,\epsilon}[\|\mathcal{M}_0 - \epsilon_\theta(\mathcal{M}_t, t, \mathcal{F}_C)\|_2^2]$$

**Coarse-to-Fine Strategy**: Coarse and fine predictors are trained using templates sampled from different distributions. The coarse predictor samples templates randomly, while the fine predictor samples within ±30° of the query pose. At inference, coarse prediction precedes fine prediction without any modification to the network architecture.

### Loss & Training

Rotation loss: $\mathcal{L}^R = \lambda_{\text{recon}}\mathcal{L}_{\text{recon}}^R + \lambda_{\cos}\mathcal{L}_{\cos}^R + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}^R$

where $\mathcal{L}_{\text{reg}}^R$ is a ray consistency loss that enforces geometric consistency among neighboring rays:

$$\mathcal{L}_{\text{reg}}^R = \frac{1}{|\mathcal{N}_r|}\sum_{(i,j)\in\mathcal{N}_r}(\alpha_{ij} - \alpha_{ij}^*)^2$$

The translation loss combines dense map reconstruction and 3D translation L1 supervision. The total loss is $\mathcal{L} = \lambda_{\text{rot}}\mathcal{L}^R + \lambda_{\text{trans}}\mathcal{L}^T$.

## Key Experimental Results

### Main Results on BOP Benchmark

| Method | Refinement | Multi-Hyp. | LM-O | T-LESS | TUD-L | IC-BIN | YCB-V | Mean |
|--------|-----------|------------|------|--------|-------|--------|-------|------|
| ZS6D | ✗ | ✗ | 29.8 | 21.0 | — | — | 32.4 | 27.7 |
| MegaPose | ✗ | ✗ | 22.9 | 17.7 | 25.8 | 15.2 | 28.1 | 21.9 |
| GenFlow | ✗ | ✗ | 25.0 | 21.5 | 30.0 | 16.8 | 27.7 | 24.2 |
| OSOP | ✗ | ✗ | 31.2 | — | — | — | 33.2 | 32.2 |

### Ablation Study

| Configuration | Description | Effect on Performance |
|---------------|-------------|----------------------|
| Camera-centric rays → Object-centric rays | Object-centric ray representation | Significant improvement in rotation accuracy |
| w/o angular consistency loss | Removing ray regularization | Degraded rotation accuracy |
| Coarse only → Coarse + Fine | Coarse-to-fine strategy | Further performance improvement |
| Dense translation map vs. SITE | Dense offset map | Higher translation accuracy |

### Key Findings

1. Object-centric ray representation is more suitable for object pose estimation than camera-centric Plücker coordinates, as it decouples the effect of camera intrinsics.
2. Multi-hypothesis sampling (initialized from different noise samples) effectively captures multimodal distributions, which is particularly beneficial for symmetric objects.
3. The coarse-to-fine strategy improves performance without modifying the network architecture, demonstrating the flexibility of the diffusion framework.
4. The ray consistency loss serves as geometric regularization that preserves the structural integrity of predicted ray maps.

## Highlights & Insights

1. **Distinctive problem reformulation**: Treating pose estimation as ray bundle alignment rather than template matching with correspondence estimation better exploits multi-view geometric constraints.
2. **Elegant parameterization design**: Object-centric rays and dense translation maps project 6D pose into structured 2D representations that naturally suit the pixel-level denoising paradigm of diffusion models.
3. **Flexible inference strategy**: Coarse-to-fine prediction requires only a change in template sampling distribution, enabling the same network to serve different accuracy requirements.
4. **Solid theoretical foundation**: SVD-based rotation recovery guarantees outputs lie on the SO(3) manifold.

## Limitations & Future Work

1. Dependence on CAD model-rendered templates limits applicability in fully model-free scenarios.
2. Multi-step denoising in the diffusion model results in slower inference speed.
3. Robustness under severe occlusion and truncation scenarios remains to be validated.

## Related Work & Insights

- **Template-based methods**: Render-and-compare strategies such as MegaPose, GigaPose, and OSOP.
- **Foundation model-based methods**: FoundPose (DINOv2), ZeroPose (ImageBind+SAM).
- **Diffusion-based pose estimation**: RayDiffusion, PoseDiffusion, DiffusionNOCS.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Ray bundle diffusion paradigm and object-centric parameterization constitute genuinely novel contributions.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rotation/translation parameterization is mathematically rigorous with well-designed losses.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset evaluation with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-formatted equations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](../../CVPR2025/human_understanding/crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)

</div>

<!-- RELATED:END -->
