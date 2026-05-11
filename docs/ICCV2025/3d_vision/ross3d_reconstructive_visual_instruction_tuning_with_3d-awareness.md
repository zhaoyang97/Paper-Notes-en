---
title: >-
  [Paper Note] Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness
description: >-
  [ICCV 2025][3D Vision][3D scene understanding] Ross3D introduces 3D-aware visual reconstruction pretraining tasks—cross-view reconstruction and global BEV reconstruction—into the training pipeline of 2D large multimodal models (LMMs). Without modifying the input representation, it significantly improves 3D scene understanding through output-level supervision signals, achieving state-of-the-art performance on five benchmarks: SQA3D, ScanQA, Scan2Cap, ScanRefer, and Multi3DRefer.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D scene understanding
  - large multimodal models
  - visual reconstruction supervision
  - cross-view reconstruction
  - bird's-eye-view reconstruction
date: 2026-05-08
content_hash: 9c0a93c221c1005a
---

# Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness

**Conference**: ICCV 2025
**arXiv**: [2504.01901](https://arxiv.org/abs/2504.01901)
**Code**: [https://haochen-wang409.github.io/ross3d](https://haochen-wang409.github.io/ross3d) (Project Page)
**Area**: 3D Vision
**Keywords**: 3D scene understanding, large multimodal models, visual reconstruction supervision, cross-view reconstruction, bird's-eye-view reconstruction

## TL;DR

Ross3D introduces 3D-aware visual reconstruction pretraining tasks—cross-view reconstruction and global BEV reconstruction—into the training pipeline of 2D large multimodal models (LMMs). Without modifying the input representation, it significantly improves 3D scene understanding through output-level supervision signals, achieving state-of-the-art performance on five benchmarks: SQA3D, ScanQA, Scan2Cap, ScanRefer, and Multi3DRefer.

## Background & Motivation

3D scene understanding is a core capability for embodied intelligence, requiring comprehensive modeling of spatial relationships and scene layouts. Recent advances in 2D LMMs have achieved remarkable success in image and video understanding, motivating researchers to transfer these models to 3D scene understanding. However, two fundamental contradictions arise:

**First contradiction**: A severe scarcity of 3D vision-language data. Unlike the 2D domain with its massive image-text pairs, the 3D domain lacks large-scale, high-quality scene-text annotations and powerful 3D pretrained encoders analogous to CLIP, making the direct path from 3D point clouds to language models ineffective.

**Second contradiction**: Existing methods focus exclusively on input-level 3D representation design, an approach that is inherently limited. Specifically, prior methods fall into three categories: (a) fusing 3D point cloud features with 2D image features (e.g., ChatScene, LEO); (b) aggregating 2D features in 3D voxel space (e.g., LLaVA-3D); (c) treating multi-view images as video sequences (e.g., Video-3D-LLM). However, because LMMs carry an inherent inductive bias toward 2D data, input-level modifications alone cannot enable genuine 3D spatial understanding.

**Key Insight**: Since input-level modifications are insufficient, the solution lies in output-level supervision. Ross3D proposes a novel perspective—introducing 3D-aware visual reconstruction supervision into the training pipeline by designing 3D-relevant pretext tasks that compel the model to learn 3D spatial relationships. The key insight is that rather than changing the input, 3D-aware reconstruction objectives are imposed on the visual output tokens, effectively "injecting" 3D understanding into the model's feature representations.

## Method

### Overall Architecture

Ross3D is built upon LLaVA-Video-7B and comprises three core components: a video encoder $\mathcal{E}_\phi$, a large language model $\mathcal{P}_\theta$, and a denoising network $\mathcal{J}_\pi$. Unlike conventional methods that supervise only text outputs $\bm{x}_{i>N}$, Ross3D additionally applies 3D-aware visual reconstruction supervision to the visual outputs $\bm{x}_{i\leq N}$.

The core training objective is:

$$\mathcal{L}_{3D}(\bm{x}, \bm{I}; \Theta) = \mathcal{D}(\mathcal{J}_\pi(\bm{x}_{i\leq N}), \mathcal{F} \circ \mathcal{T}_o(\bm{I}))$$

where $\mathcal{T}_i$ and $\mathcal{T}_o$ denote the input and output transformation functions, respectively. The design of these transformations is key to injecting 3D awareness—when both $\mathcal{T}_i$ and $\mathcal{T}_o$ are identity mappings, the objective degenerates to ordinary 2D reconstruction.

### Key Designs

1. **Cross-View Reconstruction**:

    - **Function**: Randomly masks a subset of views and requires the model to infer the content of the masked views from the remaining ones.
    - **Mechanism**: Given multi-view images $\bm{I} \in \mathbb{R}^{M\times H\times W\times 3}$, a view-level binary mask $\bm{M} \in \{0,1\}^M$ is generated with a mask ratio $\gamma=25\%$. Features of masked views are replaced by a learnable mask token $\bm{m}$, and the reconstruction target is the VAE latent tokens of the masked views.
    - **Key formula**: $\mathcal{L}_{3D}^{cross} = \frac{1}{\gamma M}\sum_{j=1}^{M}(1-\bm{M}_j)\cdot\mathcal{D}(\mathcal{J}_\pi \circ \mathcal{P}_\theta(\bm{v}), \mathcal{F}(\bm{I}_j))$
    - **Design Motivation**: Cross-view reconstruction requires the model to identify overlapping information across views to recover masked content, forcing it to learn fine-grained inter-view spatial relationships. This is particularly critical for tasks requiring precise cross-view alignment, such as 3D visual grounding.
    - **Important detail**: To avoid train-test inconsistency, this objective is applied only every $\Delta t=4$ steps with a small mask ratio (25%).

2. **Global-View Reconstruction**:

    - **Function**: Aggregates information from all available views to reconstruct the bird's-eye-view (BEV) image of the entire scene.
    - **Mechanism**: 3D reconstruction techniques are applied using ego-centric video, extrinsic matrices, and intrinsic matrices to generate 3D meshes and point clouds, which are then rendered into a BEV image from above as the reconstruction target.
    - **Key formula**: $\mathcal{L}_{3D}^{global}(\bm{x}, \bm{I}; \Theta) = \mathcal{D}(J_\pi \circ \mathcal{P}_\theta(\bm{v}), \mathcal{F}(\bm{I}_{BEV}))$
    - **Design Motivation**: The BEV image encodes the global layout of the entire scene; reconstructing it requires the model to integrate all viewpoints and understand the complete scene context. This is critical for tasks requiring global understanding, such as 3D question answering.
    - **Important detail**: Since BEV images rendered from sparse point clouds contain black void regions, these blank areas are skipped during reconstruction.

3. **Denoiser $\mathcal{J}_\pi$**:

    - **Function**: Based on a DiT backbone, it recovers clean target tokens from noisy latent tokens.
    - **Mechanism**: A continuous VAE from FLUX serves as the teacher tokenizer. Learnable queries $\bm{q}$ extract a condition $\bm{c}$ from the LMM visual outputs $\bm{x}_{i\leq N}$ and the timestep $t$, followed by denoising under a diffusion framework.
    - **Design Motivation**: Denoising is preferred over direct regression because direct regression suffers from severe spatial redundancy in visual signals, making it unable to produce effective supervision signals.

### Loss & Training

The total training objective combines the text cross-entropy loss, cross-view reconstruction loss, global-view reconstruction loss, and grounding loss:

$$\mathcal{L} = \mathcal{L}_{text} + \mathcal{L}_{3D}^{cross} + \mathcal{L}_{3D}^{global} + \mathcal{L}_{grounding}$$

Training details: fine-tuned from LLaVA-Video-7B using AdamW optimizer, global batch size 256, peak learning rate 1e-5, with the visual encoder frozen. Each scene uses 32 frames at resolution 384×384. BEV images are rendered at 432×432. Training is conducted for one epoch on 8×A100-80G GPUs.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Ross3D | Video-3D-LLM (Prev. SOTA) | Gain |
|-----------|--------|--------|--------------------------|------|
| SQA3D | EM | **63.0** | 58.6 | +4.4 |
| ScanQA | CIDEr | **107.0** | 102.1 | +4.9 |
| Scan2Cap | ROUGE@0.5 | **66.9** | 62.3 | +4.6 |
| ScanRefer | Acc@0.25 | **61.1** | 58.1 | +3.0 |
| Multi3DRefer | F1@0.25 | **59.6** | 58.0 | +1.6 |

### Ablation Study

| Configuration | SQA3D (EM) | ScanQA (CIDEr) | ScanRefer (Acc@0.25) | Note |
|---------------|-----------|---------------|---------------------|------|
| Video-3D-LLM Baseline | 58.6 | 102.1 | 58.1 | No visual reconstruction supervision |
| + Vanilla reconstruction | 58.8 (+0.2) | 103.5 (+1.4) | 58.2 (+0.1) | Marginal gain without 3D awareness |
| + Cross-view reconstruction | 60.0 (+1.4) | 103.6 (+1.5) | 60.3 (+2.1) | Notable gain on grounding |
| + Global-view reconstruction | 61.6 (+3.0) | 105.6 (+3.5) | 58.8 (+0.7) | Notable gain on QA |
| + Both (Ross3D) | **63.0 (+4.4)** | **107.0 (+4.9)** | **61.1 (+3.0)** | Mutual reinforcement between tasks |

### Key Findings

- Vanilla reconstruction without 3D awareness yields negligible improvement (only +0.2 on SQA3D), demonstrating that 3D awareness—not reconstruction signals per se—is the critical factor.
- Cross-view reconstruction is particularly effective on ScanRefer (+2.1), as the task demands fine-grained cross-view spatial alignment.
- Global-view reconstruction is particularly effective on SQA3D (+3.0), as question answering tasks require global scene understanding.
- The combination of both pretext tasks substantially outperforms either alone, indicating mutual reinforcement at different levels of understanding.
- In semi-supervised experiments, using only 50% text-annotated data combined with 50% vision-only data under Ross3D even surpasses the 100% text-supervised baseline on ScanQA (103.2 vs. 102.1 CIDEr), demonstrating the significant potential of leveraging unannotated 3D data.

## Highlights & Insights

- **A novel paradigm shift**: Transitioning from input-level modification to output-level supervision is an elegant change of perspective. Prior work focused on "how to construct better 3D inputs"; Ross3D asks "how to design better 3D learning objectives."
- **Semi-supervised learning potential**: Ross3D naturally supports learning from 3D visual data without text annotations, opening a pathway for leveraging large quantities of unlabeled 3D scans.
- **Plug-and-play design**: The denoising network is used only during training and introduces no additional overhead at inference.
- **Complementarity of the two pretext tasks**: Cross-view reconstruction captures local, fine-grained inter-view relationships, while global-view reconstruction captures macro-level scene layout—the two tasks are perfectly complementary.

## Limitations & Future Work

- The approach relies on depth maps to generate position-aware video representations; depth estimation quality directly affects the final performance.
- BEV images rendered from sparse point clouds suffer from void regions, which may limit the effectiveness of global-view reconstruction.
- Validation is currently limited to indoor scenes (ScanNet); generalization to larger-scale or outdoor environments remains unexplored.
- Design choices for the denoising network (e.g., using the FLUX VAE as the teacher) may not be optimal.

## Related Work & Insights

- **Reconstructive Visual Instruction Tuning (VITRON)**: The 2D predecessor of Ross3D, establishing the effectiveness of vision-centric supervision in 2D LMMs.
- **Video-3D-LLM**: The baseline model for Ross3D, which treats multi-view images as video sequences.
- **MAE/BEiT**: Cross-view reconstruction shares conceptual lineage with masked autoencoders, operating at the view level rather than the patch level.
- **Insight**: A similar "change the supervision, not the input" strategy could be explored for other 3D tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Output-level 3D-aware supervision is a highly original contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 benchmarks + detailed ablations + semi-supervised experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, polished figures and tables)
- Value: ⭐⭐⭐⭐⭐ (High practical value; significant potential for semi-supervised learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](../../CVPR2026/3d_vision/openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)
- [\[ICCV 2025\] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](../../NeurIPS2025/3d_vision/on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [\[ICCV 2025\] 4D Visual Pre-training for Robot Learning](4d_visual_pretraining_for_robot_learning.md)

</div>

<!-- RELATED:END -->
