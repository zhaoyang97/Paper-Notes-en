---
title: >-
  [Paper Note] Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids
description: >-
  [AAAI 2026][Scientific Computing][physics-informed dataset] This work introduces the Phys-Liquid dataset (97,200 physics-simulated images with 3D meshes)…
tags:
  - "AAAI 2026"
  - "Scientific Computing"
  - "physics-informed dataset"
  - "transparent liquid"
  - "3D reconstruction"
  - "liquid simulation"
  - "deformable objects"
date: 2026-05-08
content_hash: 14c05c33f70f3f08
---

# Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids

**Conference**: AAAI 2026
**arXiv**: [2511.11077](https://arxiv.org/abs/2511.11077)  
**Code**: [https://dualtransparency.github.io/Phys-Liquid/](https://dualtransparency.github.io/Phys-Liquid/)  
**Area**: 3D Vision / Scientific Computing
**Keywords**: physics-informed dataset, transparent liquid, 3D reconstruction, liquid simulation, deformable objects

## TL;DR

This work introduces the Phys-Liquid dataset (97,200 physics-simulated images with 3D meshes), which models dynamic deformation of liquids inside transparent containers based on the Navier-Stokes equations, and proposes a four-stage reconstruction pipeline (segmentation → multi-view mask generation → 3D reconstruction → scaling) to achieve high-accuracy liquid geometry and volume estimation in both simulated and real-world scenes.

## Background & Motivation

Autonomous laboratory robots performing liquid manipulation tasks (pipetting, aspiration, mixing) require accurate perception of dynamic liquid deformation induced by container motion. However, existing datasets exhibit significant limitations:

- Large-scale 3D datasets such as **Objaverse** predominantly contain rigid or opaque objects with no liquid content.
- Transparent object datasets such as **ClearGrasp** and **ClearPose** focus on 6D pose estimation while ignoring internal liquids.
- **DTLD** includes liquids but only in static states; **Narasimhan et al.** covers only single-view scenarios without deformation.

**Key Challenge**: The absence of a dynamic liquid simulation dataset with physically realistic deformation impedes the development of accurate liquid perception algorithms.

The paper addresses this gap by leveraging Blender and Mantaflow to simulate liquid deformation caused by container rotation via the Navier-Stokes equations, systematically constructing a large-scale dataset spanning multiple scenes, lighting conditions, viewpoints, and continuous temporal frames.

## Method

### Overall Architecture

This work comprises two components: (1) **Dataset construction**—generating 97,200 images and 8,100 liquid 3D meshes through physics-based simulation; and (2) **A four-stage validation pipeline**—reconstructing liquid 3D geometry from a single image and estimating real-world volumetric scale, serving as a benchmark for dataset utility.

### Key Designs

**1. Physics-Based Simulation via the Navier-Stokes Equations**

Liquid dynamics are governed by the momentum equation:

$$\frac{D\mathbf{u}}{Dt} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}$$

where $\mathbf{u}$ is the velocity field, $\rho$ is density, $p$ is the pressure field, $\nu$ is kinematic viscosity, and $\mathbf{g}$ denotes external forces. The incompressibility condition $\nabla \cdot \mathbf{u} = 0$ is imposed and solved using the Mantaflow solver within Blender.

Data generation covers: 20 common laboratory containers, 5 laboratory scenes, 8 lighting conditions, 5 liquid colors, 6 rotation modes (combinations along X/Y/Z axes, 0°–80°), 81 temporal frames, and 6 orthographic camera viewpoints (top, bottom, front, back, left, right). Each container is assigned 5 condition combinations, yielding 100 total configurations.

**2. Four-Stage Liquid Reconstruction Pipeline**

The overall pipeline is formalized as $S = F(I) = T(R(G(S(I))), s)$:

- **Stage 1: Liquid Segmentation**—YOLO-World detects liquid regions (treating "liquid" and "colored liquid" as positive classes while excluding "bottle"), and the resulting bounding boxes guide SAM2 for precise segmentation.
- **Stage 2: Multi-View Mask Generation**—The CRM diffusion model (fine-tuned on Phys-Liquid) generates liquid masks and canonical coordinate maps for 6 orthographic viewpoints from a single-view mask.
- **Stage 3: 3D Mesh Reconstruction**—Based on a triplane representation, a convolutional U-Net encodes multi-view masks into triplane features, which an MLP decodes into a textured 3D mesh.
- **Stage 4: Real-World Scale Estimation**—A multi-view ViT architecture regresses the scaling factor $s$, supervised with an L2 loss.

The scaling factor is computed as:

$$s = \sqrt[3]{\frac{S_{\text{PI},x}}{V_x} \cdot \frac{S_{\text{PI},y}}{V_y} \cdot \frac{S_{\text{PI},z}}{V_z}}$$

**3. Diffusion Model Fine-Tuning Strategy**

The 6 orthographic views per timestep in Phys-Liquid serve as fine-tuning data. Training is conducted on 2× RTX 6000 Ada 48 GB GPUs for 16 hours and 10k iterations. After fine-tuning, mean IoU improves from 74.38% to 90.05%.

### Loss & Training

- Diffusion model fine-tuning: standard denoising loss.
- Scaling model: L2 regression loss, trained on 1× RTX 6000 Ada GPU for 12 hours and 500 iterations.
- Dataset split: 9:1 train/test split along complete temporal sequences to prevent frame-level data leakage.

## Key Experimental Results

### Main Results

**Comparison with liquid-specific baselines** (Phys-Liquid test set):

| Method | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|--------|------|-----------------|------------|-------------|
| Eppel et al. | 0.0842 | 0.0412 | 0.1216 | 30.91 |
| **Ours (fine-tuned)** | **0.0192** | **0.0079** | **0.4748** | **75.38** |

RMSE is reduced by 77.2% and F-Score improves by 44.47 percentage points.

**Comparison with general-purpose reconstruction baselines** (50 test images):

| Method | Chamfer Distance | Volume IoU | F-Score (%) |
|--------|-----------------|------------|-------------|
| InstantMesh | 0.0189 | 0.2794 | 46.18 |
| TripoSR | 0.0275 | 0.2275 | 38.06 |
| Ours (w/o fine-tuning) | 0.0128 | 0.3246 | 58.19 |
| **Ours (w/ fine-tuning)** | **0.0085** | **0.6236** | **78.57** |

Fine-tuning raises Volume IoU from 0.3246 to 0.6236 and F-Score from 58.19% to 78.57%.

### Ablation Study

**Pipeline module ablation** (progressively replacing module outputs with simulation ground truth):

| Replaced Module | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|----------------|------|-----------------|------------|-------------|
| Full pipeline | 0.0192 | 0.0079 | 0.4748 | 75.38 |
| Replace segmentation | 0.0130 | 0.0075 | 0.5504 | 78.42 |
| Replace multi-view generation | 0.0105 | 0.0067 | 0.6532 | 81.36 |
| Replace mesh reconstruction | 0.0085 | 0.0058 | 0.7687 | 85.64 |
| Replace scaling | 0.0071 | 0.0042 | 0.7511 | 88.47 |

The mesh reconstruction and scaling modules exert the greatest influence on final performance.

**Effect of diffusion model fine-tuning**:

| Setting | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|---------|------|-----------------|------------|-------------|
| w/o fine-tuning | 0.0254 | 0.0139 | 0.2850 | 46.19 |
| w/ fine-tuning (test set) | 0.0192 | 0.0079 | 0.4748 | 75.38 |

Fine-tuning reduces RMSE by 24.4% and improves Volume IoU by 66.6%.

**Generalization to real-world scenes** (DTLD dataset):

| Dataset | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|---------|------|-----------------|------------|-------------|
| DTLD (real-world) | 0.0266 | 0.0172 | 0.3861 | 62.43 |
| Phys-Liquid (simulated) | 0.0192 | 0.0079 | 0.4748 | 75.38 |

Training exclusively on simulated data yields reasonable accuracy on real-world data.

### Key Findings

- Multi-view mask generation exhibits strong cross-view consistency, with per-view mean IoU ranging from 89.21% to 91.76%.
- Temporal consistency is high: RMSE variance across 100 temporal sequences is only 0.00038, with a standard deviation of 0.00644.
- Simulated and real liquid deformation angles align closely, validating the physical fidelity of the simulation.

## Highlights & Insights

- **Filling a data gap**: The first physics-simulated dataset capturing dynamic liquid deformation, extending the problem from 3D to 4D (spatiotemporal).
- **End-to-end validation**: The integrated dataset and reconstruction pipeline provides not only data but also a functional baseline method.
- **Extensibility**: The Blender-based framework can render additional modalities such as surface normals and refraction flow, offering significant tooling value.

## Limitations & Future Work

- Dataset scale is moderate (97,200 images), remaining relatively small compared to large-scale vision datasets.
- Only rotation-induced deformation is modeled; more complex operations such as pouring and mixing are not covered.
- All containers are standard laboratory vessels; irregular container geometries are not addressed.
- The scaling model relies on CAD model information to provide real-world size references.

## Related Work & Insights

- **vs. DTLD**: DTLD contains 27,458 multi-view images but only static liquids; Phys-Liquid provides 97,200 images with dynamic deformation and temporal variation.
- **vs. TransProteus**: TransProteus employs Mantaflow for liquid simulation but does not model deformation induced by container rotation.
- **vs. InstantMesh/TripoSR**: General-purpose 3D reconstruction methods fail to capture fine-grained liquid deformation characteristics; the proposed fine-tuned model substantially outperforms these baselines.

## Rating

- Novelty: ⭐⭐⭐⭐ First physics-simulated dataset capturing dynamic liquid deformation, filling a clear gap in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation (reconstruction accuracy, generalization, temporal consistency, module ablation) with well-designed protocols.
- Writing Quality: ⭐⭐⭐⭐ Richly illustrated with clear pipeline descriptions.
- Value: ⭐⭐⭐ Focused on laboratory liquid manipulation scenarios; application scope is narrow but highly significant within the target domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](../../NeurIPS2025/scientific_computing/neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/scientific_computing/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/scientific_computing/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)

</div>

<!-- RELATED:END -->
