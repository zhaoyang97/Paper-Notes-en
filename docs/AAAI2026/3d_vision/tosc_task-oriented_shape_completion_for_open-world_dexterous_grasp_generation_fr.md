---
title: >-
  [Paper Note] TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds
description: >-
  [AAAI 2026][3D Vision][Task-Oriented Shape Completion] This work proposes a new task named Task-Oriented Shape Completion (TOSC), which only reconstructs task-relevant contact areas rather than the entire object. By generating candidates with pre-trained foundation models, filtering the optimal shape with a 3D Discriminative Autoencoder (DAE), and synthesizing dexterous grasps via a FlowGrasp flow-matching model, the proposed method yields performance gains of 16.17% in grasp…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Task-Oriented Shape Completion"
  - "Dexterous Grasping"
  - "Point Cloud Completion"
  - "Flow Matching"
  - "Foundation Models"
date: 2026-05-08
content_hash: ec1fffa1d9c9c329
---

# TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds

**Conference**: AAAI 2026  
**arXiv**: [2601.05499](https://arxiv.org/abs/2601.05499)  
**Code**: [github.com/SyKszzzzz/TOSC](https://github.com/SyKszzzzz/TOSC)  
**Area**: 3D Vision  
**Keywords**: Task-Oriented Shape Completion, Dexterous Grasping, Point Cloud Completion, Flow Matching, Foundation Models

## TL;DR

This work proposes a new task named Task-Oriented Shape Completion (TOSC), which only reconstructs task-relevant contact areas rather than the entire object. By generating candidates with pre-trained foundation models, filtering the optimal shape with a 3D Discriminative Autoencoder (DAE), and synthesizing dexterous grasps via a FlowGrasp flow-matching model, the proposed method yields performance gains of 16.17% in grasp displacement and 55.26% in Chamfer distance.

## Background & Motivation

### Problem Definition

Task-oriented dexterous grasping demands the generation of grasp poses that are both stable and suited to a prescribed downstream manipulation task. For instance, "using scissors to cut" requires placing fingers on the handle loops, whereas "passing scissors to someone else" requires holding the blade section.

### Limitations of Prior Work

**Dependence on Complete Geometry**: Existing methods (such as DexTOG and DexGraspVLA) assume the availability of a complete object point cloud as input. However, in real-world scenarios, occlusions, clutter, and sensor noise often restrict the input to severely incomplete partial point clouds.

**Limitations of General Shape Completion**: Decoupled methods that first perform complete shape reconstruction and then generate grasps suffer from a fundamental flaw. General completion models recover the entire object, but occlusion-induced ambiguities can lead to incorrect completions and propagate errors to crucial contact areas.

**Decoupled Completion and Manipulation Tasks**: Because general completion is oblivious to downstream task definitions, it cannot prioritize accuracy in task-relevant contact areas.

### Core Motivation

**Key Insight**: Shape completion meant for grasping should be **explicitly guided** by the downstream manipulation task. Reconstructing a perfect shape of the entire object is unnecessary; instead, insuring the geometric correctness of the **contact regions** associated with the grasping task is sufficient, while leaving negligible shape imperfections in irrelevant regions tolerable.

For example, to "drink with a mug," the handle geometry is critical, whereas a perfect reconstruction of the mug's bottom is secondary. This insight drives the concept of **Task-Oriented Shape Completion (TOSC)**.

## Method

### Overall Architecture

The framework consists of three stages:
1. **TOSC Candidate Generation**: Uses pre-trained foundation models (ControlNet and 3D generative models) to generate multiple task-oriented shape completion candidates.
2. **TOSC Selection & Reconstruction**: Employs a 3D Discriminative Autoencoder (DAE) to score candidate reliability and reconstruct the optimal shape.
3. **FlowGrasp**: Generates task-oriented dexterous grasps using a conditional flow matching model.

### Key Designs

#### 1. **TOSC Candidate Generation**: Leveraging the Zero-shot Capability of Foundation Models

**Function**: Generates multiple potential completed shapes from a partial point cloud input using ControlNet and 3D generative models.

**Mechanism**:
1. Render the input point cloud $P_{in}$ into a depth map $I_{depth}$ (using the HPR algorithm to choose the optimal perspective).
2. Input the depth map as a ControlNet condition and the object category as a text prompt to synthesize multiple RGB images.
3. Generate multiple RGB variations by varying the control strength $\lambda$—a large $\lambda$ enforces strict compliance with the depth map but may retain incomplete artifacts, while a small $\lambda$ allows more freedom but may deviate from the input.
4. Pass each RGB image through a 3D shape generation network (Hunyuan3D-DiT-v2-mini-Fast) to produce 3D meshes.
5. Use SAM segmentation and GPT-4o to identify task-relevant areas on both the generated 3D shapes and the input point cloud.
6. Align and fuse the generated shapes with the input point cloud using ICP and task-aware alignment.

Alignment optimization:
$$\underset{k,tr}{\text{argmin}}[\text{CD}(P_{in}, tr(kP_{gen})) + w_{task} \cdot \text{CD}(P_{in}^{task}, tr(kP_{gen}^{task}))]$$

**Design Motivation**: Leveraging the zero-shot capabilities of foundational models (ControlNet, Hunyuan3D, GPT-4o, SAM) allows the generation of reasonable shape completion candidates for open-world objects without requiring task-specific training data. Varying the control strengths establishes a diverse candidate pool.

#### 2. **3D Discriminative Autoencoder (DAE)**: Evaluating Credibility and Reconstructing Global Geometry

**Function**: Filters the most credible shape from the candidates and reconstructs its overall global geometry.

**Mechanism**:

**Training Data Generation**: Positive samples (credible shapes) are drawn from 6 datasets (72,524 objects). Negative samples (non-credible shapes) are constructed by removing task-relevant segments, adding noise, and corrupting local patches.

**Network Architecture**:
- Encoder $\varepsilon$: Composed of $N_{encoder}$ Transformer blocks that tokenize input point clouds (via FPS and KNN patching followed by PointNet encoding).
- Randomly mask a portion of tokens during training (the task-relevant segments of positive samples are preserved).
- Output a latent variable $l_{can}$, estimating the distribution $\mathcal{N}(\mu, \sigma)$.

**Credibility Evaluation**: Assess the credibility of the input shape by computing the KL divergence:
- Positive sample distributions are optimized toward $\mathcal{N}(0,1)$.
- Negative sample distributions are optimized toward $\mathcal{N}(1,1)$.
- Confidence score during inference:

$$s_{can} = \text{Sigmoid}[-\mathcal{D}_{KL}(d_{can}\|\mathcal{N}(0,1)) + \mathcal{D}_{KL}(d_{can}\|\mathcal{N}(1,1))]$$

**Design Motivation**: Generated shapes from foundation models can suffer from hallucinations (due to ControlNet rendering variations) or reconstruction errors (from 3D generators), demanding a discriminative scoring pipeline. Meanwhile, the masked autoencoder's reconstructive capacity helps rectify local geometric defects from a global perspective.

#### 3. **FlowGrasp**: Constraint-Aware Conditional Flow Matching Model

**Function**: Generates dexterous grasps satisfying geometric and semantic constraints from the reconstructed 3D shape.

**Mechanism**: Applies a single-step gradient correction directly to the predicted velocity within the standard flow matching framework to implicitly enforce constraints:

$$u_t^*(x_t) = u_t(x_t) - \alpha(t) \nabla\left(\sum_i w_{con}^i g_i(x_t)\right)$$

where $g_i$ denotes geometric or semantic constraints, and $\alpha(t)$ is a time-decaying factor.

During training, the network directly regresses the modified velocity target:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{x_0,x_1,t,l_{con}} \|v_t^\theta(x_t|l_{con}) - u_t^*(x_t)\|^2$$

The conditioning vector $l_{con}$ is formed by concatenating PointNet++ features (extracted from the 3D shape) and CLIP embeddings (describing the task text).

**Design Motivation**: Traditional methods enforce constraints using weighted penalty terms in losses or through inference-time gradient tuning; the former compromises the probabilistic foundation of generative models, while the latter incurs heavy latency. FlowGrasp incorporates constraints directly into the training targets, eliminating extra loss parameters and inference overhead.

### Loss & Training

Total loss for the 3D DAE: $L = L_{pos}^{KL} + L_{neg}^{KL} + L_{recon} + L_{mask}$

- $L_{pos}^{KL}$: Pulls positive sample distributions toward $\mathcal{N}(0,1)$
- $L_{neg}^{KL}$: Drives negative sample distributions toward $\mathcal{N}(1,1)$
- $L_{recon}$: Chamfer distance reconstruction loss
- $L_{mask}$: Mean Squared Error (MSE) measuring feature consistency across masking operations

FlowGrasp: Standard CFM loss + single-step gradient correction.

Training details: DAE trained for 300 epochs, FlowGrasp trained for 350 epochs, on a single RTX 4090 GPU.

## Key Experimental Results

### Main Results

**Task-Oriented Dexterous Grasping (OakInk-PartialPC Dataset)**:

| Method | Penetration Volume ↓ | Penetration Depth ↓ | Grasp Displacement Mean ↓ | Contact Ratio ↑ | P-FID ↓ | LLM Score ↑ |
|------|----------|----------|-------------|--------|--------|---------|
| GraspCVAE | 16.84 | 0.141 | 3.92 | 94.74% | 39.03 | 55.0 |
| SceneDiffuser | 6.52 | 0.090 | 3.81 | 95.62% | 29.38 | 61.7 |
| DexGYSGrasp | 7.16 | 0.096 | 3.76 | 97.20% | 25.98 | 68.3 |
| **Ours** | 6.87 | 0.090 | **3.11** | **98.30%** | **21.60** | **88.3** |

**Task-Oriented Shape Completion**:

| Method | CD-&ell;₂&times;10&⁻\*;⁴ ↓ | F-Score@1 ↑ | DCD ↓ |
|------|-------------|-----------|------|
| PointAttn | 4.58 | 0.512 | 0.698 |
| SVDFormer | 3.71 | 0.643 | 0.603 |
| SymmCompletion | 3.94 | 0.618 | 0.611 |
| **Ours** | **1.66** | **0.860** | **0.488** |

### Ablation Study

| Configuration | Grasp Displacement Mean ↓ | P-FID ↓ | LLM Score ↑ | SC ↑ | PP ↑ | IS ↑ |
|------|-------------|--------|---------|-----|-----|-----|
| w/o TCG (Candidate Gen.) | 3.50 | 22.34 | 71.7 | 2.63 | 0.81 | 1.72 |
| w/o TSR (Select. & Recon.) | 3.35 | 22.96 | 75.0 | 2.36 | 2.63 | 2.72 |
| w/o TOSC (Gen. Comp. Subst.) | 3.51 | 23.83 | 66.7 | 2.18 | 1.36 | 2.18 |
| w/o token masking | 3.21 | 22.53 | 78.3 | 3.09 | 3.18 | 3.36 |
| w/o gradient guidance | 3.43 | 24.01 | 83.3 | 3.72 | 2.18 | 3.63 |
| **Full method** | **3.11** | **21.60** | **88.3** | **4.38** | **3.84** | **3.80** |

### Key Findings

1. **TOSC Outperforms General Completion**: The CD falls from 3.71 to 1.66 (a 55.26% reduction), proving the superior suitability of the task-oriented paradigm over overall shape completion.
2. **Improved Grasp Stability**: Grasp displacement drops from 3.76 to 3.11 (a 16.17% improvement).
3. **Strong Zero-shot Generalization**: Generates high-quality grasps across 9 unseen categories and novel text instructions.
4. **Significant Lead in LLM Evaluation**: Outscores the closest baseline by a wide margin (88.3 vs. 68.3), demonstrating that the generated grasps exhibit higher semantic agreement with instructions.
5. **Necessity of All Components**: TCG provides necessary zero-shot capability, TSR eliminates model hallucinations, and gradient guidance assures physical/semantic constraint satisfaction.

## Highlights & Insights

1. **Innovative Problem Definition**: Reframing "completing the entire object" to "completing task-relevant contact areas" represents an insightful and practical paradigm shift.
2. **Deft Pipeline Orchestration**: The pipeline of ControlNet $\rightarrow$ Hunyuan3D $\rightarrow$ SAM $\rightarrow$ GPT-4o elegantly leverages the complementary modalities of several foundational models.
3. **Hybrid Discriminative-Generative Design**: The 3D DAE successfully bridges discriminative filtering (the credibility score) with generative modeling (shape reconstruction).
4. **Efficient Constraint Injection in FlowGrasp**: Incorporating constraints directly into the training targets bypasses the need for costly iterative optimization during grasp generation.
5. **Perceptual Human Study**: Incorporates solid user evaluations (measuring SC, PP, and IS dimensions) to inspect grasp reliability thoroughly.

## Limitations & Future Work

1. **Strong Foundational Dependency**: The performance of ControlNet and Hunyuan3D directly governs candidate quality; updates to these backends can dramatically shift behavior.
2. **Complex Pipeline**: Transitioning from point cloud inputs to final grasps demands several stages (rendering $\rightarrow$ RGB generation $\rightarrow$ 3D reconstruction $\rightarrow$ segmentation $\rightarrow$ alignment $\rightarrow$ DAE $\rightarrow$ FlowGrasp), which can introduce significant computational latency.
3. **reliance on GPT-4o**: Using proprietary APIs for region selection poses additional latency and financial constraints.
4. **Synthetic DAE Defect Modeling**: Creating negative samples through handcrafted corruptions might introduce a domain gap relative to actual errors generated by foundation models.
5. **No Physical Demonstration**: Experiments remain confined to simulation environments without real-robot validation.

## Related Work & Insights

- Unlike DexGraspVLA, which trains large vision-language-action models on extensive supervised datasets, TOSC relies on zero-shot foundational features to achieve comparable results with lower data dependency.
- This work augments Point-MAE's masked autoencoder design with discriminative classification capabilities.
- Leveraging multiple control strengths under ControlNet provides an intuitive solution to handle geometry ambiguities caused by partial observations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The conversion to TOSC is highly convincing and FlowGrasp introduces an elegant design for incorporating physical constraints.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive metrics are reported across physical, semantic, and human dimensions; however, physical robot experiments are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — The methods and formulations are presented clearly with helpful illustrations.
- **Value**: ⭐⭐⭐⭐⭐ — Presents a paradigm-shifting approach for handling dexterous grasping under partial observations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DynaTok: Token-Based 4D Reconstruction from Partial Point Clouds](../../ICML2026/3d_vision/dynatok_token-based_4d_reconstruction_from_partial_point_clouds.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning](open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)

</div>

<!-- RELATED:END -->
