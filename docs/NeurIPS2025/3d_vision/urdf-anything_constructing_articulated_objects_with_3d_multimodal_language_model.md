---
title: >-
  [Paper Note] URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model
description: >-
  [NeurIPS 2025][3D Vision][Articulated Object Reconstruction] This paper proposes URDF-Anything, the first end-to-end articulated object reconstruction framework based on a 3D Multimodal Large Language Model (MLLM). By introducing a [SEG] token mechanism, the framework jointly predicts geometric segmentation and kinematic parameters, achieving state-of-the-art performance in segmentation accuracy (mIoU +17%), parameter error (−29%), and physical executability (surpassing basel…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Articulated Object Reconstruction"
  - "URDF"
  - "3D Multimodal Large Language Model"
  - "Digital Twin"
  - "Robot Simulation"
date: 2026-05-08
content_hash: b351833a2f08233c
---

# URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model

**Conference**: NeurIPS 2025
**arXiv**: [2511.00940](https://arxiv.org/abs/2511.00940)  
**Code**: [Project Page](https://lzvsdy.github.io/URDF-Anything/)  
**Area**: 3D Vision
**Keywords**: Articulated Object Reconstruction, URDF, 3D Multimodal Large Language Model, Digital Twin, Robot Simulation

## TL;DR
This paper proposes URDF-Anything, the first end-to-end articulated object reconstruction framework based on a 3D Multimodal Large Language Model (MLLM). By introducing a [SEG] token mechanism, the framework jointly predicts geometric segmentation and kinematic parameters, achieving state-of-the-art performance in segmentation accuracy (mIoU +17%), parameter error (−29%), and physical executability (surpassing baselines by 50%).

## Background & Motivation
- **Background**: Constructing high-fidelity digital twins of articulated objects (e.g., doors, drawers, scissors) is critical for robot simulation training and embodied AI world model construction. These objects are characterized by multiple rigid links connected via joints, requiring complex reasoning over both geometry and kinematic parameters (type, origin, axis, limits).
- **Limitations of Prior Work**: Existing methods have notable shortcomings: (1) Real2Code abstracts parts as oriented bounding boxes (OBBs), losing critical geometric detail, and uses an LLM to predict kinematic parameters with limited accuracy; (2) Articulate-Anything relies on mesh asset libraries and iterative VLM refinement, resulting in a fragile pipeline constrained by the asset library; (3) URDFormer relies on hard-coded discrete classification to assign kinematic parameters and retrieve meshes, limiting reconstruction fidelity. A shared problem across these methods is their **multi-stage pipeline design** — segmentation, parameter prediction, and mesh generation are handled independently, causing error accumulation across stages.
- **Key Challenge**: The fundamental limitation is the decoupled treatment of geometry and kinematics, which prevents joint optimization and leads to error propagation.
- **Goal**: This paper fundamentally reframes the problem by leveraging a 3D MLLM to process point clouds end-to-end, **jointly** predicting geometric segmentation and kinematic parameters. 3D MLLMs are naturally suited for this task — they handle multimodal inputs, encode large-scale 3D shape priors, and directly understand spatial relationships to output precise coordinates.
- **Core Idea**: A [SEG] token mechanism is introduced, enabling the model to simultaneously output a symbolic representation of the kinematic structure and geometric segmentation signals during autoregressive generation, achieving fundamental consistency between the two.

## Method

### Overall Architecture
URDF-Anything consists of three stages: (1) **Input Representation** — dense 3D point clouds are generated from single-view or multi-view RGB images; (2) **Multimodal Articulation Parsing** — a 3D MLLM jointly predicts part segmentation and kinematic parameters; (3) **Mesh Conversion** — segmented point clouds are converted to meshes and assembled into a URDF file.

### Key Designs
1. **Input Representation**:

    - Multi-view input: DUSt3R is used to generate dense 3D point clouds $P_{obj} \in \mathbb{R}^{N \times 6}$ from multi-view RGB images.
    - Single-view input: A diffusion model first generates consistent multi-view images, followed by LGM for 3D geometry reconstruction.
    - **Design Motivation**: Accommodates different input conditions (monocular/multi-view) while producing a unified holistic point cloud representation.

2. **3D MLLM-Based Articulation Parsing and [SEG] Token Mechanism**:

    - ShapeLLM serves as the backbone, combining a point cloud encoder and an LLM.
    - The point cloud $P_{obj}$ is encoded into features $F_{pc} \in \mathbb{R}^{M \times d_{pc}}$ via the 3D encoder; text instructions are embedded as $F_{txt}$ via the LLM token embedding layer.
    - The MLLM autoregressively outputs: $Y_{output} = \text{MLLM}(F_{pc}, F_{txt})$
    - **Core Innovation**: The vocabulary is extended with a [SEG] token. Each link description is associated with a [SEG] token (e.g., `"link_0": "base_cabinet[SEG]"`), tightly coupling symbolic output with geometric segmentation.
    - **Design Motivation**: Standard MLLMs cannot perform per-point prediction. Inspired by LISA, the [SEG] token enables the MLLM to tag geometric regions for segmentation while generating the kinematic structure.

3. **From [SEG] Token to Geometric Segmentation**:

    - For each generated [SEG] token, its final hidden state $h_{seg}$ is fused with the preceding category token state $h_{category}$: $h_{combined} = [h_{category}; h_{seg}]$
    - The fused representation is projected via an MLP into a query $H_{query}$, which computes per-point scores over point cloud features $F'_{pc}$ via cross-attention: $y_{mask} = \text{CrossAttn}(Q=H_{query}, K=F'_{pc}, V=F'_{pc})$
    - Sigmoid activation followed by thresholding yields a binary segmentation mask for each part.
    - **Design Motivation**: Cross-attention enables efficient interaction between [SEG] token hidden states and point cloud features, leveraging the semantic understanding of the MLLM while preserving fine-grained geometric segmentation.

### Loss & Training
- The total loss is a weighted sum of language modeling loss and segmentation loss: $L = \lambda_{text}L_{text} + \lambda_{seg}\sum_{i=1}^{N}L_{i,seg}$
- The segmentation loss combines BCE and Dice: $L_{seg} = \lambda_{bce}\text{BCE}(\hat{M}, M_{gt}) + \lambda_{dice}\text{DICE}(\hat{M}, M_{gt})$
- LoRA (rank=8) is used for efficient fine-tuning of ShapeLLM-7B with the AdamW optimizer at a learning rate of 0.0003.
- Training completes in 2.5 hours on a single A800 (80GB) GPU.

## Key Experimental Results

### Main Results

| Task / Metric | URDF-Anything | Articulate-Anything | Real2Code Oracle | URDFormer Oracle |
|---|---|---|---|---|
| Segmentation mIoU (ALL) | **0.63** | — | — | — |
| Segmentation Count Acc (ALL) | **0.97** | — | — | — |
| Joint Type Error ↓ | **0.008** | 0.025 | 0.537 | 0.556 |
| Joint Axis Error ↓ | **0.132** | 0.145 | 1.006 | 0.374 |
| Joint Origin Error ↓ | **0.164** | 0.207 | 0.294 | 0.581 |
| Physical Executability (ALL) | **78%** | 52% | 41% | 24% |
| Physical Executability (OOD) | **71%** | 44% | 23% | 15% |

### Ablation Study

| Configuration | Type Error ↓ | Axis Error ↓ | Origin Error ↓ | mIoU | Count Acc |
|---|---|---|---|---|---|
| OBB Input | 0.42 | 0.70 | 0.47 | — | — |
| Point Cloud Only (no text) | 0.34 | 0.29 | 0.26 | — | — |
| Qwen2.5-VL-7B+ft (image input) | 0.38 | 0.81 | 0.18 | — | — |
| Kinematics Prediction Only | 0.009 | 0.138 | 0.175 | — | — |
| Segmentation Only | — | — | — | 0.61 | 0.89 |
| **Full Model (point cloud + text)** | **0.008** | **0.132** | **0.164** | **0.63** | **0.97** |

### Key Findings
- 2D image MLLMs (even when fine-tuned) fail to reason about precise 3D kinematic parameters, confirming the necessity of 3D point cloud input.
- Joint prediction outperforms decoupled prediction: the segmentation task provides geometric regularization for kinematics, while kinematic reasoning provides structural priors for segmentation — the two tasks mutually reinforce each other.
- Physical executability on OOD objects improves from 44% (baseline) to 71%, demonstrating strong generalization.
- The segmentation-only model yields lower mIoU (0.61) and Count Acc (0.89) compared to the joint model (0.63/0.97), indicating that the kinematic task compels the model to learn more coherent structural representations.

## Highlights & Insights
- This is the first work to apply a 3D MLLM to end-to-end URDF reconstruction of articulated objects, establishing a new paradigm. The [SEG] token mechanism is particularly elegant — it naturally embeds geometric segmentation signals within the autoregressive text generation stream, achieving seamless unification of symbolic and geometric outputs.
- The ablation study on joint vs. decoupled prediction provides a profound insight: this is not merely an engineering choice, but reflects a fundamental reciprocal relationship between the two tasks. Kinematic structure constraints improve segmentation accuracy, while geometric segmentation in turn regularizes kinematic parameters.
- The comparison between 2D image MLLMs and 3D point cloud MLLMs is highly convincing — even a fine-tuned Qwen2.5-VL-7B exhibits substantially higher axis error than the 3D approach, demonstrating the irreplaceability of explicit 3D geometry.
- Training efficiency is notable: only 2.5 hours on a single A800 GPU, with a lightweight LoRA rank=8 fine-tuning strategy that makes the method easy to reproduce.

## Limitations & Future Work
- Certain URDF attributes (e.g., mass, inertia tensors) cannot be generated, due to limitations in training data and the base model.
- The pipeline is not fully end-to-end: it still relies on an external point cloud-to-mesh conversion module to generate link geometry.
- Numerical parameter accuracy is limited by the tokenization-based generation approach, where continuous values are discretized into token sequences of finite precision.
- Evaluation is conducted solely on the PartNet-Mobility dataset; validation on complex real-world scenarios (e.g., occlusion, noisy point clouds) is insufficient.
- Future work could explore integrating implicit 3D representations (e.g., NeRF/3DGS) in place of explicit meshes to improve geometric quality.
- Multi-object scenes involving articulated object detection and segmentation remain unaddressed.

## Related Work & Insights
- **vs. Real2Code**: Real2Code uses coarse OBB representations and an LLM, losing geometric detail; URDF-Anything reasons end-to-end directly from point clouds.
- **vs. Articulate-Anything**: Relies on a mesh asset library with iterative refinement, resulting in a fragile pipeline constrained by the asset library; URDF-Anything requires no external asset library.
- **vs. URDFormer**: Uses hard-coded discrete classification for parameter assignment and mesh retrieval, yielding low fidelity; URDF-Anything directly regresses continuous parameters via MLLM.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First work to apply a 3D MLLM to end-to-end URDF reconstruction; the [SEG] token mechanism is clever and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation design is thoughtful, but evaluation is limited to a single dataset (PartNet-Mobility only).
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and motivation is well-articulated, though some details require consulting the appendix.
- **Value**: ⭐⭐⭐⭐⭐ Has direct application value for robot simulation and embodied AI; the end-to-end paradigm has strong potential for generalization to broader 3D structural reasoning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Part-X-MLLM: Part-aware 3D Multimodal Large Language Model](../../ICLR2026/3d_vision/part-x-mllm_part-aware_3d_multimodal_large_language_model.md)
- [\[CVPR 2026\] LAM: Language Articulated Object Modelers](../../CVPR2026/3d_vision/lam_language_articulated_object_modelers.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](../../CVPR2025/3d_vision/riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](../../CVPR2025/3d_vision/iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](../../CVPR2025/3d_vision/perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)

</div>

<!-- RELATED:END -->
