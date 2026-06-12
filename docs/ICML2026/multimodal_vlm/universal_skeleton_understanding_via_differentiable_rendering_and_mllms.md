---
title: >-
  [Paper Note] General Skeleton Understanding: Differentiable Rendering and MLLMs
description: >-
  [ICML 2026][Multimodal VLM][Skeleton Understanding] By rendering skeleton sequences into images, MLLMs can understand various formats of skeleton data—achieving general skeleton understanding and resolving cross-modal an…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Skeleton Understanding"
  - "Differentiable Rendering"
  - "Multi-modal Large Language Models"
  - "Action Recognition"
  - "Format Agnosticism"
date: 2026-05-08
content_hash: c7f73604741728d4
---

# General Skeleton Understanding: Differentiable Rendering and MLLMs

**Conference**: ICML 2026  
**arXiv**: [2603.18003](https://arxiv.org/abs/2603.18003)  
**Code**: https://github.com/wangzy01/SkeletonLLM  
**Area**: Multimodal VLM / 3D Vision / Human Understanding  
**Keywords**: Skeleton Understanding, Differentiable Rendering, Multi-modal Large Language Models, Action Recognition, Format Agnosticism

## TL;DR
By rendering skeleton sequences into images, MLLMs can understand various formats of skeleton data—achieving general skeleton understanding and resolving cross-modal and format heterogeneity issues.

## Background & Motivation

**Background**: MLLMs exhibit strong performance in vision-language tasks but can only process visual modalities such as images and videos, failing to directly understand structured non-visual data like skeletons. Simultaneously, skeleton data suffers from severe format fragmentation—Kinect v2 consists of 25 joints, MoCap has 22 SMPL joints, and 2D pose estimation utilizes 17 COCO joints.

**Limitations of Prior Work**: Traditional methods fall into two categories: feature-text alignment (e.g., CLIP alignment, which compresses skeleton encoder outputs into a single vector for text alignment, creating a representation bottleneck) and LLM discretization (e.g., MotionGPT, which uses VQ-VAE to quantize motion into a codebook, where the quantization is lossy and the codebook is highly dependent on formatting). Neither approach fully activates the visual understanding capabilities of MLLMs.

**Key Challenge**: There is a modality mismatch between skeletons and MLLMs—skeletons are structured coordinates, whereas MLLMs natively understand images. Furthermore, cross-format generalization requires that the model architecture not be tied to a specific skeleton topology.

**Goal**: Design a unified framework that enables a single model to process any skeleton format, supporting multiple tasks such as recognition, description, and question answering.

**Key Insight**: Instead of compressing skeletons or quantizing them into discrete symbols, it is more effective to "translate" skeletons into the native visual modality of MLLMs, thereby directly leveraging their visual understanding capabilities.

**Core Idea**: Design DrAction, a differentiable and format-agnostic skeleton renderer, to map arbitrary skeleton sequences to images. This allows gradients to flow back from the MLLM to the renderer, ensuring the rendering is optimized for downstream tasks.

## Method

### Overall Architecture
The SkeletonLLM pipeline consists of three stages: "Render-Reason-Respond." Given an input skeleton sequence $\mathbf{S}=\{\mathbf{p}_t\}_{t=1}^T$, DrAction renders it into an image sequence $\mathbf{V}=\{\mathbf{I}_t\}_{t=1}^{T'}$. Visual tokens are then extracted by the MLLM's vision encoder to facilitate linguistic reasoning. The entire process is end-to-end differentiable.

### Key Designs

1. **3D Gaussian Primitive Representation and Kinematic Transformation**:

    - **Function**: Represents the human body using K deformable 3D Gaussian primitives instead of meshes, where $K = J + J \times \text{number of edges} \times 10$. Gaussians are defined in a canonical pose space.
    - **Mechanism**: Joint motions are converted into Gaussian transformations via Linear Blend Skinning (LBS). For each joint $i$, a rigid body transformation $\mathbf{T}_i \in \mathrm{SE}(3)$ is calculated. After blending rotations $\tilde{\mathbf{R}}_k = \sum_i w_{k,i} \mathbf{R}_i$, they are projected back to $\mathrm{SO}(3)$ using SVD polar decomposition.
    - **Design Motivation**: LBS is a standard computer graphics technique. Format agnosticism is achieved by dynamically reading the number of joints $J$ and edges from the input skeleton. For formats without orientation data, setting $\mathbf{R}_i=\mathbf{I}_3$ simplifies the process to translation only. Gaussian representation natively supports differentiable rendering.

2. **Neural Feature Modulator (NFM) and Kinetic Vision**:

    - **Function**: Adaptively adjusts the color and opacity of each Gaussian based on local kinematics (position and velocity).
    - **Mechanism**: For Gaussian $k$, the position $p_k^t$ and velocity $v_k^t$ (derived via finite difference) of associated joints are aggregated. These are concatenated with base features and processed by a single-layer GRU for temporal modeling, outputting RGB and opacity residuals alongside a saliency gate. Finally, $\alpha_k = \sigma(\alpha_k^{\mathrm{base}} + \Delta\alpha_k) \cdot \sigma(g_k)$.
    - **Design Motivation**: Static appearance cannot distinguish between different movement phases of the same pose. Dynamic modulation allows the rendering to emphasize motion-salient regions.

3. **Four-Stage Collaborative Training Strategy**:

    - **Function**: Solves the "chicken and egg" problem of jointly optimizing a randomly initialized renderer with a pre-trained MLLM in a progressive manner.
    - **Mechanism**: ① Alignment Warm-up (freeze MLLM and optimize the renderer only); ② Discriminative Fine-tuning (binary classification of confusing action pairs); ③ Causal Reasoning Distillation (use a teacher model to generate step-by-step causal chains); ④ Recognition Refinement (freeze the matured renderer and update only the projection layers and LoRA).
    - **Design Motivation**: Progressing through visual recognizability, discriminative boundaries, causal understanding, and task refinement prevents initial gradient instability and meaningless rendering output.

## Key Experimental Results

### Main Results: Open-Vocabulary Action Recognition

| Dataset | Split | TDSM | MotionGPT | InternVL3-8B Baseline | SkeletonLLM | Gain |
|--------|------|------|-----------|------------------|-------------|------|
| NTU-60 | 55/5 | 86.49 | 29.88 | 76.08 | **87.37** | +0.88% |
| NTU-60 | 30/30 | 25.88 | 8.57 | 26.95 | **37.84** | +11.96% |
| NTU-120 | 60/60 | 27.21 | 5.15 | 25.12 | **34.94** | +7.73% |

### Cross-Format Transfer Accuracy

| Source Format | Target Format | TDSM | MotionGPT | SkeletonLLM |
|--------|---------|------|-----------|------------|
| Kinect v2 (NTU-60) | Kinect v1 (NW-UCLA) | 43.19 | 10.35 | **68.50** |
| MoCap (HumanML3D) | Kinect v2 (NTU-60) | 23.15 | 12.40 | **54.80** |

### Key Findings
- **Cruciality of DrAction differentiability**: Using the same InternVL3-8B backbone, a fixed renderer achieves 76.82%, while the differentiable DrAction achieves 87.48%.
- **Contribution of training stages**: Removing CR-Distill leads to a 3.2% drop, while removing Disc-FT leads to a 2.1% drop.
- **Extreme sparse scenarios**: In the 30/30 split challenge, SkeletonLLM shows a 41% relative improvement over InternVL3.

## Highlights & Insights
- **Elegant Modality Translation**: Rendering non-visual data into a visual format directly exploits the native strengths of MLLMs.
- **Universal Design for Format Agnosticism**: By dynamically reading Gaussian primitive counts and joint fusion weights from the input skeleton, the model achieves the first seamless cross-format transfer between Kinect, MoCap, and 2D positions.
- **Progressive Collaborative Training**: The 4-stage division effectively prevents initial gradient instability or rendering collapse.

## Limitations & Future Work
- The computational cost of rendering has not been analyzed in detail.
- Limited cross-dataset generalization: The paper lacks evaluation of generalization across entirely different data sources.
- Insufficient support for multi-person scenarios: While the framework is designed to support multi-person inputs, experimental performance in such scenarios was not reported.

## Related Work & Insights
- **vs. Feature-Text Alignment (PURLS/TDSM)**: Ours retains complete spatiotemporal information through rendering, whereas previous formats were still dependent on specific topologies.
- **vs. LLM Discretization (MotionGPT/MotionLLM)**: Ours is format-agnostic and incurs no information loss.
- **vs. Direct Encoding (SKI-LVLM)**: End-to-end optimization in this work allows MLLM gradients to directly guide the rendering process.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The modality translation paradigm is novel, and format-agnostic differentiable rendering is a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple datasets, formats, and tasks; cross-format transfer results are particularly compelling.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clear, though some mathematical derivations could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ Provides a general solution for skeleton-MLLM alignment with significant application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](../../ICCV2025/multimodal_vlm/orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](../../CVPR2026/multimodal_vlm/mmrad_multimodal_anomaly_detection.md)
- [\[ICML 2026\] FreeRet: MLLMs as Training-Free Retrievers](freeret_mllms_as_training-free_retrievers.md)
- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[ICML 2026\] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)

</div>

<!-- RELATED:END -->
