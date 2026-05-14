---
title: >-
  [Paper Note] Fast 3D Surrogate Modeling for Data Center Thermal Management
description: >-
  [AAAI 2026][3D Vision][Surrogate modeling] This paper develops a vision-based 3D surrogate modeling framework for data centers. Server workloads, fan speeds…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Surrogate modeling"
  - "data center thermal management"
  - "3D voxelization"
  - "Fourier neural operator"
  - "temperature prediction"
date: 2026-05-08
content_hash: e4696bda26ee888b
---

# Fast 3D Surrogate Modeling for Data Center Thermal Management

**Conference**: AAAI 2026
**arXiv**: [2511.11722](https://arxiv.org/abs/2511.11722)
**Code**: None
**Area**: 3D Vision / Scientific Computing
**Keywords**: Surrogate modeling, data center thermal management, 3D voxelization, Fourier neural operator, temperature prediction

## TL;DR

This paper develops a vision-based 3D surrogate modeling framework for data centers. Server workloads, fan speeds, and air-conditioning temperature setpoints are encoded as 3D voxel representations, and architectures including 3D CNN U-Net, 3D Fourier Neural Operator, and 3D Vision Transformer are employed for real-time temperature field prediction. The proposed framework achieves inference speeds up to 20,000× faster than traditional CFD solvers while enabling a 7% reduction in energy consumption.

## Background & Motivation

**Background**: Data centers are among the leading contributors to global energy consumption and carbon emissions. Effective thermal management not only prevents hardware failures due to overheating but also substantially reduces cooling energy costs. Accurate 3D temperature field modeling is essential for optimizing cooling strategies and workload distribution.

**Limitations of Prior Work**: Traditional computational fluid dynamics (CFD) solvers, while highly accurate, are computationally prohibitive—a single simulation may require hundreds of milliseconds to several hours, and demands expert-level effort to construct meshes and boundary conditions. This renders CFD infeasible for real-time control scenarios. Data centers must rapidly adjust cooling strategies in response to dynamically changing workloads, yet CFD is far too slow to meet this demand.

**Key Challenge**: A fundamental tension exists between accuracy and speed—precise 3D temperature field modeling requires solving complex coupled thermal-flow equations (accurate but slow), whereas real-time control demands millisecond-level responses (fast, but typically relying on oversimplified and inaccurate models).

**Goal**: (1) Construct a surrogate model capable of rapidly predicting 3D temperature fields in data centers; (2) maintain accuracy comparable to CFD solvers; (3) generalize across diverse data center configurations.

**Key Insight**: The authors reformulate temperature field prediction as a "vision" problem—encoding the physical layout and operational parameters of a data center as a 3D voxel map, and leveraging well-established 3D architectures from computer vision for end-to-end learning.

**Core Idea**: A 3D voxelization representation uniformly encodes both the geometric structure and operational state of a data center, enabling real-time inference at CFD-level accuracy through modern deep learning architectures.

## Method

### Overall Architecture

The input is a 3D voxelized representation of the data center, encoding server locations, workloads, fan speeds, and HVAC temperature setpoints. The output is the corresponding 3D temperature heat map. The framework evaluates multiple architectures: 3D CNN U-Net variants, 3D Fourier Neural Operator (FNO), and 3D Vision Transformer.

### Key Designs

1. **3D Voxelization Representation**:

   - **Function**: Encodes the complex physical environment of a data center into a regularized input suitable for deep learning.
   - **Mechanism**: The data center space is discretized into a uniform 3D voxel grid. Each voxel contains multi-channel information: server occupancy, server workload level, nearby fan speed, and HVAC setpoint temperature. This representation preserves spatial relationships while unifying heterogeneous physical quantities into a consistent data format.
   - **Design Motivation**: CFD relies on unstructured meshes that are incompatible with standard deep learning architectures. Voxelization transforms the problem into a standard 3D image-to-image regression task, enabling direct application of 3D CNN/Transformer architectures.

2. **Multi-Architecture Evaluation (3D U-Net, FNO, ViT)**:

   - **Function**: Systematically compares different architectures in terms of the accuracy–efficiency trade-off for temperature field prediction.
   - **Mechanism**: (a) The 3D CNN U-Net employs an encoder–decoder structure with skip connections to capture local thermal flow patterns at multiple scales; (b) the 3D Fourier Neural Operator (FNO) learns integral kernel operators in the frequency domain, making it naturally suited for PDE-solving tasks; (c) the 3D Vision Transformer leverages self-attention to capture long-range thermal interactions, such as the influence of distant air conditioning units on local temperatures.
   - **Design Motivation**: The physical characteristics of temperature fields span multiple scales—local heat source effects favor CNNs, while long-range airflow propagation is better captured by Transformers or FNOs. A systematic comparison facilitates identifying the most suitable architecture for this problem.

3. **Cross-Configuration Generalization Design**:

   - **Function**: Enables a single model to generalize across data centers with different layouts.
   - **Mechanism**: Training data encompass a wide range of data center configurations (varying server layouts and cooling system setups), allowing the model to learn generalizable physical laws of heat conduction and convection. The unified voxelization input format allows different configurations to be expressed as distinct voxel encodings.
   - **Design Motivation**: Training a separate model for each data center would incur prohibitive deployment costs. A model that learns general physical laws can adapt to new configurations in a zero-shot or few-shot manner.

### Loss & Training

An MSE loss function is used to measure the discrepancy between the predicted temperature field and the CFD ground truth. Training data are generated in batch by a CFD simulator across diverse operating conditions. Data augmentation involves random variations in workload and cooling parameters to increase training diversity.

## Key Experimental Results

### Main Results

| Method | Inference Time | Accuracy | Speedup | Notes |
|--------|---------------|----------|---------|-------|
| CFD Solver | Hundreds of ms to hours | Baseline | 1× | Conventional method |
| Surrogate Model (best) | Milliseconds | High-fidelity | 20,000× | Meets real-time control requirements |
| 3D U-Net | Fast | Good | High | Strong local feature capture |
| 3D FNO | Fast | Good | High | Frequency-domain PDE learning |
| 3D ViT | Fast | Good | High | Strong long-range interaction modeling |

Real-time temperature prediction enables prediction-based cooling control and workload redistribution, achieving approximately 7% reduction in energy consumption and carbon footprint.

### Ablation Study

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| Full input | Best | Workload + fan speed + HVAC fully encoded |
| Without workload | Degraded | Missing critical heat source information |
| Without fan speed | Degraded | Airflow driving factor neglected |
| Single-configuration training | Poor generalization | Multi-configuration training is critical for generalization |

### Key Findings

- A 20,000× speedup enables real-time thermal management, shifting the paradigm from "offline planning" to "online control."
- Multiple architectures achieve acceptable accuracy, each with distinct strengths: U-Net is most sensitive to local hotspot detection, FNO produces the smoothest global temperature distribution predictions, and ViT best models long-range thermal influences.
- The 7% energy saving carries significant economic and environmental implications—given the enormous global energy consumption of data centers, a 7% reduction corresponds to a substantial absolute value.

## Highlights & Insights

- **The "vision" reformulation of a physical simulation problem** is an elegant insight—3D voxelization allows mature computer vision architectures to be directly applied to scientific computing problems.
- **The 20,000× speedup** has profound practical impact, enabling real-time control scenarios that were previously infeasible.
- **Direct industrial value**: a 7% energy reduction multiplied by the massive global energy consumption of data centers translates into significant economic and environmental benefits.

## Limitations & Future Work

- The accuracy of the surrogate model is bounded by the quality and diversity of the training data (CFD simulations).
- A trade-off exists between voxelization resolution and model complexity—higher resolution improves accuracy but increases computational cost.
- Transient thermal processes are not considered—the current model likely performs steady-state prediction, and temporal modeling would be required for transient temperature variations under dynamic workload changes.
- The surrogate model could be integrated with reinforcement learning as an environment simulator for training cooling control policies.

## Related Work & Insights

- **vs. Traditional CFD**: CFD achieves high accuracy but is too slow; the surrogate model trades a minor accuracy loss for orders-of-magnitude speedup.
- **vs. Simplified Physical Models (e.g., thermal resistance networks)**: Simplified models are fast but insufficiently accurate; the deep learning surrogate achieves a better balance between accuracy and speed.
- **vs. 2D Surrogate Models**: Prior work largely relies on 2D planar approximations; the 3D modeling in this paper more faithfully captures vertical temperature gradients and airflow patterns.

## Rating

- Novelty: ⭐⭐⭐ Applying 3D vision architectures to thermal management is a sound engineering innovation, though the core techniques (U-Net/FNO/ViT) are pre-existing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-architecture comparisons, cross-configuration generalization, and real-world energy saving measurements.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated and experimental design is thorough.
- Value: ⭐⭐⭐⭐ High practical industrial value with direct economic and environmental significance in energy reduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](../../CVPR2026/3d_vision/fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)
- [\[ICCV 2025\] CF³: Compact and Fast 3D Feature Fields](../../ICCV2025/3d_vision/cf3_compact_and_fast_3d_feature_fields.md)
- [\[CVPR 2026\] Action-guided Generation of 3D Functionality Segmentation Data](../../CVPR2026/3d_vision/action-guided_generation_of_3d_functionality_segmentation_data.md)
- [\[AAAI 2026\] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)
- [\[CVPR 2026\] Lifting Unlabeled Internet-level Data for 3D Scene Understanding](../../CVPR2026/3d_vision/lifting_unlabeled_internet-level_data_for_3d_scene_understanding.md)

</div>

<!-- RELATED:END -->
