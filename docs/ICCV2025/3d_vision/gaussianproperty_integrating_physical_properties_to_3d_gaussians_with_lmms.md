---
title: >-
  [Paper Note] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs
description: >-
  [3D Vision] GaussianProperty presents a training-free framework that assigns physical properties (density, elastic modulus, friction coefficient…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: fae91e32102a8461
---

# GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2412.11258](https://arxiv.org/abs/2412.11258)
- **Code**: [Project Page](https://Gaussian-Property.github.io)
- **Area**: 3D Vision
- **Keywords**: 3D Gaussian Splatting, physical property estimation, large multimodal models, robotic grasping, dynamic simulation

## TL;DR
GaussianProperty presents a training-free framework that assigns physical properties (density, elastic modulus, friction coefficient, etc.) to 3D Gaussians by leveraging SAM for segmentation and GPT-4V for recognition, via a global-local reasoning module and a multi-view voting strategy. The framework supports two downstream tasks: physics-based simulation and robotic grasping.

## Background & Motivation

Estimating physical properties from visual data is a fundamental task in computer vision and graphics, with critical applications in augmented reality, physics simulation, and robotic grasping. Nevertheless, this area remains underexplored due to the following challenges:

**Difficulty in acquiring annotated data**: Physical properties such as density and elastic modulus cannot be directly observed from visual surfaces, and ground-truth annotations are scarce.

**Inherent ambiguity in prediction**: Inferring intrinsic physical properties from limited observable surfaces is fundamentally ill-posed.

**Limitations of existing methods**: Prior work largely addresses specific material property types (e.g., mass or hardness), requires task-specific annotated data, and exhibits limited generalization.

The authors observe that humans can predict physical properties from visual cues by associating appearance with prior knowledge of known materials. Large multimodal models (LMMs) such as GPT-4V, having been trained on extensive prior knowledge, demonstrate human-like visual recognition capabilities that can be leveraged for physical property estimation.

## Method

### Overall Architecture

The GaussianProperty pipeline consists of three stages:
1. **Part-level segmentation**: SAM is applied to multi-view images to obtain part-level semantic segmentation.
2. **Physical property matching**: A global-local reasoning module and GPT-4V are used to identify materials and query their physical properties.
3. **2D-to-3D projection**: A multi-view voting strategy lifts 2D physical properties to 3D Gaussians.

### Key Design 1: Part-Level Segmentation

For each observed image $I \in \mathcal{I}^N$, a $32 \times 32$ point prompt grid is fed into SAM to obtain multi-granularity segmentation masks $\mathbf{M}$. Redundant masks are filtered using IoU scores, stability scores, and mask overlap ratios, yielding semantic part-level segmentation.

### Key Design 2: Global-Local Physical Property Reasoning

Querying physical properties directly from a global image is insufficient, as the model struggles to associate global context with local parts. A three-component reasoning strategy is therefore designed:

- **Material candidate library**: A library comprising 15 material families and 600+ specific materials is constructed to simplify material retrieval for the LMM.
- **Global-local combined reasoning**: The original global image, a segmentation map with red mask annotations, and a cropped local part image are provided simultaneously, enabling GPT-4V to associate each part with the overall object.
- **Progressive prompt guidance**: A step-wise prompting strategy is designed in which the LMM first describes the part's appearance, then identifies the material and outputs physical parameters including mass density $\rho$, Young's modulus $E$, and Poisson's ratio $P$.

### Key Design 3: Voting-Based 2D-to-3D Lifting

Each 3D Gaussian $\mathbf{s}$ is projected onto all visible 2D images via camera parameters:

$$u, v = \mathbf{K} \cdot [\mathbf{R}|\mathbf{t}] \cdot \mathbf{s}$$

Visibility is determined using Gaussian-estimated depth, and a frequency voting strategy is then applied to determine the final property assignment:

$$\hat{a} = \arg\max_a \sum_{i=1}^N \mathbb{I}(a_i = a)$$

where $N$ is the number of viewpoints and $a_i$ is the property observed from the $i$-th view.

### Downstream Application 1: Material-Aware Robotic Grasping

Based on predicted physical properties, the lower and upper bounds of grasping force are computed as:

$$F_{\min} = \sum_{i=1}^M \frac{1}{2}\rho(i)V(i)g\left(\frac{\cos\theta}{\mu(s)} - \sin\theta\right)$$

$$F_{\max} = \min\left[A\sigma_y(s), \frac{1}{2}AE(s)d(s)\kappa_{\max}(s)\right]$$

This ensures the grasping force is sufficient to lift the object without causing deformation.

### Downstream Application 2: Physics-Based Dynamic Simulation

Predicted physical properties (density, Young's modulus, Poisson's ratio, and material type) are directly assigned to 3D Gaussians, enabling physics-driven dynamic simulation via the Material Point Method (MPM) without manual annotation.

## Experiments

### Main Results: Material Segmentation

| Method | ABO Mean mIoU | MVImgNet Mean mIoU |
|--------|--------------|-------------------|
| NeRF2Physics | 25.59 | 4.02 |
| **Ours** | **55.83** | **34.83** |

GaussianProperty substantially outperforms NeRF2Physics across all material categories on both the ABO and MVImgNet datasets.

### Ablation Study: Global-Local Reasoning and Voting Strategy

| Global-Local Reasoning | Voting Strategy | Mean mIoU (%) |
|-----------------------|----------------|--------------|
| ✗ | ✓ | 22.17 |
| ✓ | ✗ | 51.28 |
| ✓ | ✓ | **55.83** |

Both key designs contribute significantly to final performance: global-local reasoning yields a 29-point improvement, and the voting strategy provides an additional 4.5-point gain.

### Robotic Grasping Experiments

| Method | PUR (%) | NDR (%) | SR (%) |
|--------|---------|---------|--------|
| MinGF (fixed minimum force) | Low | High | Low |
| MidGF (fixed medium force) | Medium | Medium | Medium |
| MaxGF (fixed maximum force) | High | Low | Low |
| **Ours (adaptive force)** | **High** | **High** | **Highest** |

In grasping experiments on 16 real-world objects, the adaptive grasping force strategy surpasses all fixed-force baselines in terms of success rate.

### Key Findings
- GPT-4V possesses the capability to infer material properties from visual input, but requires carefully designed prompting strategies.
- Combined global-local information is critical for accurate material recognition.
- Multi-view voting effectively mitigates incidental errors from single-view predictions.
- Physical property annotation enables seamless integration of 3DGS into simulation and robotic systems.

## Highlights & Insights

1. **Training-free framework**: Fully exploits the zero-shot capabilities of SAM and GPT-4V without any additional training.
2. **First exploration of LMMs for 3D physical property estimation**: Demonstrates the potential of vision foundation models in physical understanding.
3. **Practical applicability**: Material-aware grasping is validated on a real robotic platform.
4. **Extensible design**: The material candidate library and reasoning module can be flexibly adapted to different downstream tasks.

## Limitations & Future Work
- Reliance on GPT-4V API calls incurs non-trivial costs and is constrained by the model's knowledge coverage.
- Objects that share similar appearances but differ in material (e.g., metallic paint vs. actual metal) may be misclassified.
- The voting strategy assumes independence of property predictions across viewpoints, without modeling spatial continuity.
- Physical properties in dynamic simulation are assigned discretely, without modeling continuous material gradients.

## Related Work & Insights
- **NeRF2Physics**: Uses an LLM to propose material candidates and estimates physical properties via zero-shot kernel regression.
- **Make-it-Real**: Reasons about PBR material properties of objects for texture generation.
- **PhysGaussian**: Integrates Newtonian physics into 3DGS for dynamic simulation, but requires manual property assignment.
- **SAM / GPT-4V**: Provide accurate segmentation and multimodal understanding capabilities, respectively.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First training-free framework leveraging LMMs for 3D physical property estimation.
- **Practicality**: ⭐⭐⭐⭐ — Validated on real robotic grasping with broad application prospects.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across material segmentation, dynamic simulation, and robotic grasping.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clear motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness](dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)

</div>

<!-- RELATED:END -->
