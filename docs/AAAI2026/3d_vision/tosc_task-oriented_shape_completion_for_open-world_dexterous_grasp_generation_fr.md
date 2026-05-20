---
title: >-
  [Paper Note] TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds
description: >-
  [AAAI 2026][3D Vision][Task-oriented shape completion] This paper introduces Task-Oriented Shape Completion (TOSC), a novel task that completes only the contact regions relevant to a manipulation task—rather than the ent…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Task-oriented shape completion"
  - "dexterous grasping"
  - "point cloud completion"
  - "flow matching"
  - "foundation models"
date: 2026-05-08
content_hash: b6d4da7c8fce0177
---

# TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds

**Conference**: AAAI 2026
**arXiv**: [2601.05499](https://arxiv.org/abs/2601.05499)  
**Code**: [github.com/SyKszzzzz/TOSC](https://github.com/SyKszzzzz/TOSC)  
**Area**: 3D Vision
**Keywords**: Task-oriented shape completion, dexterous grasping, point cloud completion, flow matching, foundation models

## TL;DR

This paper introduces Task-Oriented Shape Completion (TOSC), a novel task that completes only the contact regions relevant to a manipulation task—rather than the entire object—by leveraging pretrained foundation models to generate candidate shapes, a 3D Discriminative Autoencoder (DAE) to select the optimal shape, and a FlowGrasp flow-matching model to synthesize dexterous grasps. The approach achieves improvements of 16.17% in grasp displacement and 55.26% in Chamfer Distance over prior methods.

## Background & Motivation

### Problem Definition

Task-oriented dexterous grasping requires generating grasp poses that are both stable and appropriate for a specific downstream manipulation task. For example, "cutting with scissors" requires fingers to be placed on the handle, whereas "handing scissors to someone" requires gripping the blade side.

### Limitations of Prior Work

**Dependence on complete geometry**: Existing methods (DexTOG, DexGraspVLA, etc.) assume fully observed object point clouds as input, whereas real-world scenarios frequently yield severely incomplete partial point clouds due to occlusion, clutter, and sensor noise.

**Limitations of generic shape completion**: The decoupled pipeline of completing first and grasping second suffers from a fundamental issue—generic completion methods recover the entire object shape, but occlusion-induced ambiguity can lead to incorrect completions or propagate errors from task-irrelevant regions into the contact area.

**Decoupling of completion and task**: Generic completion is unaware of the downstream task and therefore cannot prioritize geometric accuracy in task-relevant contact regions.

### Core Motivation

**Key insight**: Shape completion for grasping should be **explicitly guided** by the downstream manipulation task. Rather than perfectly reconstructing the full object, it suffices to ensure geometric fidelity in the **contact region** relevant to the grasp task, while tolerating imperfections in irrelevant regions.

For instance, to "drink from a cup," what matters is accurately completing the geometry of the handle; whether the bottom of the cup is perfectly reconstructed is inconsequential. This is the central idea behind **Task-Oriented Shape Completion (TOSC)**.

## Method

### Overall Architecture

The pipeline comprises three stages:
1. **TOSC Candidate Generation**: Pretrained foundation models (ControlNet + 3D generative model) produce multiple task-oriented shape completion candidates.
2. **TOSC Selection and Recovery**: A 3D Discriminative Autoencoder (DAE) evaluates candidate credibility and recovers the optimal shape.
3. **FlowGrasp**: A conditional flow-matching model generates task-oriented dexterous grasps.

### Key Designs

#### 1. **TOSC Candidate Generation**: Zero-Shot Capability via Foundation Models

**Function**: Starting from a partial point cloud, leverages ControlNet and a 3D generative model to produce multiple plausible completed shapes.

**Mechanism**:
1. Render the input point cloud $P_{in}$ as a depth map $I_{depth}$ (using the HPR algorithm to select the optimal viewpoint).
2. Use the depth map as the ControlNet condition with the object category as the text prompt to synthesize multiple RGB images.
3. Generate multiple RGB variants by varying the control strength $\lambda$—large $\lambda$ strictly adheres to the depth map but may preserve incompleteness artifacts, while small $\lambda$ allows more freedom but may deviate from the input.
4. Use a 3D shape generation network (Hunyuan3D-DiT-v2-mini-Fast) to reconstruct a 3D mesh from each RGB image.
5. Apply SAM segmentation and GPT-4o detection to identify task-relevant regions on both the generated 3D shape and the input point cloud.
6. Fuse the generated shape with the input point cloud via ICP and task-aware alignment.

Alignment optimization:
$$\underset{k,tr}{\text{argmin}}[\text{CD}(P_{in}, tr(kP_{gen})) + w_{task} \cdot \text{CD}(P_{in}^{task}, tr(kP_{gen}^{task}))]$$

**Design Motivation**: By exploiting the zero-shot capabilities of foundation models (ControlNet, Hunyuan3D, GPT-4o, SAM), the method generates plausible shape completion candidates for open-world objects without any task-specific training data. Varying control strengths cover a diverse range of completion hypotheses.

#### 2. **3D Discriminative Autoencoder (DAE)**: Credibility Assessment and Global Geometry Recovery

**Function**: Selects the most credible candidate from the generated set and repairs its geometry from a global perspective.

**Mechanism**:

**Training data construction**: Positive samples (credible shapes) are drawn from 6 datasets (72,524 objects); negative samples (non-credible shapes) are synthesized by removing task-relevant segments, adding noise, and perturbing local patches.

**Network architecture**:
- Encoder $\varepsilon$: $N_{encoder}$ Transformer blocks that tokenize the point cloud (FPS + KNN patch grouping → PointNet encoding).
- During training, a random subset of tokens is masked (task-relevant tokens are not masked for positive samples).
- Outputs a latent variable $l_{can}$ with estimated distribution $\mathcal{N}(\mu, \sigma)$.

**Credibility scoring**: KL divergence is used to assess whether an input shape is credible:
- Positive sample distributions are optimized toward $\mathcal{N}(0,1)$.
- Negative sample distributions are optimized toward $\mathcal{N}(1,1)$.
- Credibility score at inference:

$$s_{can} = \text{Sigmoid}[-\mathcal{D}_{KL}(d_{can}\|\mathcal{N}(0,1)) + \mathcal{D}_{KL}(d_{can}\|\mathcal{N}(1,1))]$$

**Design Motivation**: Shapes generated by foundation models may contain hallucinations (inaccurate RGB images from ControlNet) or geometric errors (imprecise reconstruction by the 3D generative model), necessitating a discriminative mechanism to filter for the most credible candidate. Additionally, the masked autoencoder's reconstruction capability enables global correction of local defects.

#### 3. **FlowGrasp**: Constraint-Aware Conditional Flow Matching

**Function**: Generates dexterous grasps satisfying geometric and semantic constraints from the recovered 3D shape.

**Mechanism**: Within the standard flow-matching framework, the predicted velocity undergoes a single-step gradient correction to implicitly enforce constraints:

$$u_t^*(x_t) = u_t(x_t) - \alpha(t) \nabla\left(\sum_i w_{con}^i g_i(x_t)\right)$$

where $g_i$ encodes geometric or semantic constraints and $\alpha(t)$ is a time-decaying factor.

During training, the model directly regresses toward the corrected velocity target:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{x_0,x_1,t,l_{con}} \|v_t^\theta(x_t|l_{con}) - u_t^*(x_t)\|^2$$

The conditioning vector $l_{con}$ is formed by concatenating PointNet++ features (3D shape) and CLIP embeddings (task language description).

**Design Motivation**: Conventional approaches enforce constraints via weighted penalty losses or inference-time gradient adjustments. The former compromises the likelihood interpretation of the probabilistic model; the latter introduces inference overhead. FlowGrasp incorporates constraints directly into the training objective, requiring neither additional loss terms nor extra inference cost.

### Loss & Training

3D DAE total loss: $L = L_{pos}^{KL} + L_{neg}^{KL} + L_{recon} + L_{mask}$

- $L_{pos}^{KL}$: Positive sample distribution → $\mathcal{N}(0,1)$
- $L_{neg}^{KL}$: Negative sample distribution → $\mathcal{N}(1,1)$
- $L_{recon}$: Chamfer Distance reconstruction loss
- $L_{mask}$: Feature consistency before and after masking (MSE)

FlowGrasp: Standard CFM loss with single-step gradient correction.

Training: DAE for 300 epochs, FlowGrasp for 350 epochs, on a single RTX 4090.

## Key Experimental Results

### Main Results

**Task-Oriented Dexterous Grasping (OakInk-PartialPC dataset)**:

| Method | Penetration Volume↓ | Penetration Depth↓ | Grasp Displacement Mean↓ | Contact Ratio↑ | P-FID↓ | LLM Score↑ |
|--------|--------------------|--------------------|--------------------------|----------------|--------|------------|
| GraspCVAE | 16.84 | 0.141 | 3.92 | 94.74% | 39.03 | 55.0 |
| SceneDiffuser | 6.52 | 0.090 | 3.81 | 95.62% | 29.38 | 61.7 |
| DexGYSGrasp | 7.16 | 0.096 | 3.76 | 97.20% | 25.98 | 68.3 |
| **Ours** | 6.87 | 0.090 | **3.11** | **98.30%** | **21.60** | **88.3** |

**Task-Oriented Shape Completion**:

| Method | CD-ℓ₂×10⁻⁴↓ | F-Score@1↑ | DCD↓ |
|--------|-------------|-----------|------|
| PointAttn | 4.58 | 0.512 | 0.698 |
| SVDFormer | 3.71 | 0.643 | 0.603 |
| SymmCompletion | 3.94 | 0.618 | 0.611 |
| **Ours** | **1.66** | **0.860** | **0.488** |

### Ablation Study

| Configuration | Grasp Displacement Mean↓ | P-FID↓ | LLM Score↑ | SC↑ | PP↑ | IS↑ |
|---------------|--------------------------|--------|------------|-----|-----|-----|
| w/o TCG (candidate generation) | 3.50 | 22.34 | 71.7 | 2.63 | 0.81 | 1.72 |
| w/o TSR (selection & recovery) | 3.35 | 22.96 | 75.0 | 2.36 | 2.63 | 2.72 |
| w/o TOSC (generic completion) | 3.51 | 23.83 | 66.7 | 2.18 | 1.36 | 2.18 |
| w/o token masking | 3.21 | 22.53 | 78.3 | 3.09 | 3.18 | 3.36 |
| w/o gradient guidance | 3.43 | 24.01 | 83.3 | 3.72 | 2.18 | 3.63 |
| **Full method** | **3.11** | **21.60** | **88.3** | **4.38** | **3.84** | **3.80** |

### Key Findings

1. **TOSC substantially outperforms generic completion**: CD decreases from 3.71 to 1.66 (↓55.26%), confirming the core value of task-oriented completion.
2. **Significant improvement in grasp stability**: Grasp Displacement decreases from 3.76 to 3.11 (↓16.17%).
3. **Strong zero-shot generalization**: The method maintains state-of-the-art performance on 9 unseen object categories and novel language instructions.
4. **LLM scoring shows a large margin**: 88.3 vs. 68.3 (second best), demonstrating substantially superior semantic consistency in the generated grasps.
5. **All components are indispensable**: TCG provides zero-shot capability, TSR corrects hallucinations, and gradient guidance ensures constraint satisfaction.

## Highlights & Insights

1. **Novel problem formulation**: TOSC reframes "completing the entire object" as "completing task-relevant contact regions," representing a compelling paradigm shift.
2. **Chained foundation model composition**: The pipeline ControlNet → Hunyuan3D → SAM → GPT-4o effectively exploits the complementary capabilities of multiple foundation models.
3. **Combined discriminative and generative design**: The 3D DAE simultaneously performs discrimination (credibility scoring) and generation (shape recovery).
4. **Constraint integration in FlowGrasp**: Embedding constraints into the training objective rather than the inference process avoids costly inference-time gradient optimization.
5. **Human perceptual evaluation**: A user study incorporating three dimensions (SC, PP, IS) provides a comprehensive assessment of grasp quality.

## Limitations & Future Work

1. **Strong dependence on foundation models**: The quality of ControlNet and Hunyuan3D directly affects candidate generation; future model updates may substantially alter performance.
2. **Long pipeline**: The process from point cloud to final grasp involves rendering → RGB generation → 3D reconstruction → segmentation → alignment → DAE → FlowGrasp, potentially introducing significant inference latency.
3. **Reliance on GPT-4o**: GPT-4o is used for task-relevant region detection, incurring additional API costs and latency.
4. **Limited diversity of DAE training negatives**: Negative samples are synthesized by artificially corrupting shapes, and their distribution may not fully reflect the actual failure modes of foundation models.
5. **No physical robot validation**: All experiments are conducted in simulation.

## Related Work & Insights

- **Comparison with DexGraspVLA**: The latter trains a vision-language-action model on large-scale supervised data, whereas TOSC exploits the zero-shot capabilities of foundation models, requiring significantly less data.
- The masked autoencoder paradigm from Point-MAE is extended here with an additional discriminative capability.
- The multi-control-strength variant strategy from ControlNet provides a practical approach for handling shape ambiguity.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The TOSC task formulation is novel and convincing; the constraint integration in FlowGrasp is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation metrics are comprehensive (physical + semantic + human), but real-robot experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — The method description is clear and well-illustrated.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a paradigm-level solution for dexterous grasping under partial observation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning](open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](../../ICCV2025/3d_vision/revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)

</div>

<!-- RELATED:END -->
