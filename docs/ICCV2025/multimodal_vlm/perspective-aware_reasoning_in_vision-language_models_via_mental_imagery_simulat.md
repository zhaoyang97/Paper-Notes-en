---
title: >-
  [Paper Note] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation
description: >-
  [ICCV 2025][Multimodal VLM][perspective-aware reasoning] This paper proposes the Abstract Perspective Change (APC) framework, which leverages visual foundation models to construct an abstract scene representation and per…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "perspective-aware reasoning"
  - "mental imagery simulation"
  - "vision-language models"
  - "spatial reasoning"
  - "perspective transformation"
date: 2026-05-08
content_hash: 5566567a3f41c75d
---

# Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation

**Conference**: ICCV 2025
**arXiv**: [2504.17207](https://arxiv.org/abs/2504.17207)
**Code**: [https://github.com/KAIST-Visual-AI-Group/APC-VLM](https://github.com/KAIST-Visual-AI-Group/APC-VLM)
**Area**: Multimodal VLM
**Keywords**: perspective-aware reasoning, mental imagery simulation, vision-language models, spatial reasoning, perspective transformation

## TL;DR

This paper proposes the Abstract Perspective Change (APC) framework, which leverages visual foundation models to construct an abstract scene representation and perform perspective transformations, enabling VLMs to reason spatially from arbitrary viewpoints. APC substantially outperforms existing VLMs and fine-tuned models on both synthetic and real-image benchmarks.

## Background & Motivation

**Background**: Vision-language models (VLMs) have achieved notable progress on spatial reasoning tasks, including object spatial relation judgment and depth-assisted spatial question answering. The prevailing approach is to feed images and textual questions end-to-end into VLMs, or to enhance spatial reasoning capabilities through fine-tuning on spatial reasoning data.

**Limitations of Prior Work**: Recent studies reveal that existing VLMs perform poorly when required to reason from non-camera viewpoints (i.e., non-egocentric perspectives), exhibiting a severe "egocentric bias" — models default to answering from the photographer's viewpoint and cannot shift to the perspective of other people or positions within the scene. For instance, when asked "From the person on the other side of the table, is the cup to the left or right of the plate?", VLMs almost invariably answer from the camera's perspective, yielding incorrect responses.

**Key Challenge**: VLMs lack the capacity for "mental rotation" — humans can form an abstract representation of a scene in their minds and freely rotate it to switch perspectives, but VLMs' 2D visual encoders are inherently tied to the camera viewpoint of the input image and cannot perform such perspective transformations. This is a structural deficiency that cannot be resolved through simple data augmentation or fine-tuning.

**Goal**: (1) Construct a 3D abstract representation of the scene from the input image; (2) Transform the abstract representation to the target viewpoint; (3) Convey the transformed information to the VLM in an interpretable form.

**Key Insight**: The authors draw inspiration from the cognitive psychology concept of "mental imagery" — when humans switch perspectives, they do not reconstruct a complete 3D scene in their minds, but rather form a simplified abstract representation (e.g., retaining only the relative positions and orientations of objects) and rotate that representation. This "abstract-first, then transform" strategy is more efficient and robust than direct novel view synthesis.

**Core Idea**: Off-the-shelf visual foundation models (detection, segmentation, depth estimation, orientation estimation) are employed to extract a 3D scene abstraction from the image; coordinate transformations simulate perspective switching; and the results are fed back to the VLM as numerical or visual prompts, enabling spatial reasoning from arbitrary viewpoints.

## Method

### Overall Architecture

The APC framework takes a scene image and a spatial question requiring a specific reference viewpoint as input, and outputs the correct spatial relationship answer as seen from that viewpoint. The overall pipeline comprises three stages: (1) **Scene Abstraction** — constructing a 3D abstract scene representation using visual foundation models; (2) **Perspective Change** — transforming the 3D abstraction into the coordinate frame of the reference viewpoint; (3) **Perspective Prompting** — encoding the transformed information into prompts interpretable by the VLM.

### Key Designs

1. **Scene Abstraction**:

    - **Function**: Constructs a simplified scene representation containing 3D positions and orientations of objects from a single 2D image.
    - **Mechanism**: Grounding DINO is first applied for open-vocabulary detection to localize objects mentioned in the question; SAM then obtains fine-grained segmentation masks to determine the center pixel of each object; Depth Pro estimates the depth of each object for back-projection into 3D space; and Orient Anything estimates object orientation vectors. Each object is ultimately characterized by a 3D coordinate $(x, y, z)$ and an orientation $(\theta)$.
    - **Design Motivation**: Full 3D scene reconstruction (e.g., NeRF) is unnecessary; only object-level position and orientation are required. This aligns with the "rough but sufficient" abstraction strategy in human mental imagery, while avoiding the high computational cost and quality instability associated with 3D reconstruction.

2. **Perspective Change**:

    - **Function**: Transforms the scene abstraction from the camera coordinate frame to the allocentric reference viewpoint.
    - **Mechanism**: The reference viewpoint position and orientation are determined from the question (e.g., "from Alice's perspective" places Alice's position and facing direction as the new coordinate origin and forward axis). A rotation and translation transformation is then applied to the 3D coordinates of all objects: $\mathbf{p}' = R(\theta)(\mathbf{p} - \mathbf{t})$, yielding the relative positions of all objects as seen from the reference viewpoint.
    - **Design Motivation**: This step is the core of the framework — performing coordinate transformation at the 3D abstraction level is more precise and faster than re-rendering the entire image via novel view synthesis, and is unaffected by rendering quality.

3. **Perspective Prompting**:

    - **Function**: Feeds the transformed scene information to the VLM in an interpretable form to guide correct viewpoint-aligned responses.
    - **Mechanism**: Two modalities are provided — (a) **Numerical prompts**: the transformed 3D coordinates of each object are written directly into the prompt as text (e.g., "From the reference perspective, object A is at (1.2, -0.3, 0.5)"); (b) **Visual prompts**: colored blocks are placed at the 3D position of each object, and an abstract bird's-eye or frontal view is rendered from the reference perspective and provided to the VLM alongside a color-to-object mapping.
    - **Design Motivation**: The two prompting modalities suit different scenarios — numerical prompts are more effective for VLMs with strong spatial reasoning capabilities (e.g., GPT-4o), while visual prompts are more compatible with models excelling at visual understanding (e.g., Qwen2.5-VL).

### Loss & Training

APC is a training-free framework requiring no additional training or fine-tuning. All components (detection, segmentation, depth estimation, orientation estimation) use off-the-shelf pretrained models; coordinate transformation is a deterministic computation; and VLM inference uses the original pretrained weights with carefully designed prompts.

## Key Experimental Results

### Main Results

Perspective reasoning accuracy comparison on the synthetic benchmark (Spatial-Map) and real-image benchmark:

| Method | Spatial-Map Ego (%) | Spatial-Map Allo (%) | Real-Image Allo (%) | Type |
|---|---|---|---|---|
| GPT-4o (baseline) | 78.5 | 42.3 | 38.7 | Vanilla VLM |
| Qwen2.5-VL (baseline) | 72.1 | 35.8 | 33.2 | Vanilla VLM |
| Cambrian-1 (baseline) | 65.4 | 30.5 | 28.9 | Vanilla VLM |
| SpatialRGPT (fine-tuned) | 70.2 | 45.6 | 41.3 | Fine-tuned |
| NVS-based approach | 68.3 | 48.2 | 39.8 | NVS-assisted |
| **APC + GPT-4o** | **79.1** | **68.7** | **62.5** | Ours |
| **APC + Qwen2.5-VL** | **73.5** | **63.2** | **57.8** | Ours |

### Ablation Study

| Configuration | Allo Accuracy (%) | Note |
|---|---|---|
| Full APC (visual prompt) | 63.2 | Full model + visual prompt |
| Full APC (numerical prompt) | 68.7 | Full model + numerical prompt |
| w/o Depth (2D only) | 48.5 | Depth estimation removed; 2D coordinates only |
| w/o Orientation | 55.3 | Orientation estimation removed |
| w/o Scene Abstraction | 42.3 | Original image used directly (baseline) |
| Random perspective guess | 25.0 | Random guess (4-way classification) |

### Key Findings

- Depth estimation is the most critical module; its removal reduces allocentric accuracy from 68.7% to 48.5%, confirming that 3D positional information is fundamental to perspective transformation.
- Numerical prompts outperform visual prompts on GPT-4o (68.7% vs. 63.2%), though visual prompts may be more advantageous for models with stronger visual understanding capabilities.
- APC maintains stable accuracy even at large angular offsets $\theta$ between the camera and reference viewpoints, whereas baseline VLMs degrade sharply with increasing angle — demonstrating that the framework achieves genuine viewpoint decoupling.
- Performance gains on real images are slightly smaller than in synthetic environments, primarily limited by the accuracy of depth estimation and object detection in complex real-world scenes.

## Highlights & Insights

- **The "abstract-first, then transform" paradigm is elegant**: rather than solving perspective change at the pixel level (avoiding NVS quality issues), it performs simple coordinate transformations at the level of object-centric abstract representations. This closely mirrors human cognitive processes and incurs minimal computational cost.
- **Fully training-free modular design**: all components are plug-and-play and can benefit automatically from advances in visual foundation models without retraining.
- **The dual-modality prompting strategy is generalizable**: the idea of encoding 3D information simultaneously as both text and visual prompts for VLM consumption can be extended to any task requiring the injection of 3D spatial information into VLMs (e.g., robot navigation, embodied question answering).

## Limitations & Future Work

- The quality of scene abstraction is highly dependent on the accuracy of object detection and depth estimation, and may degrade in scenes with severe occlusion or depth ambiguity.
- The current approach handles only the position and orientation of rigid objects, and cannot accommodate non-rigid deformations or fine-grained spatial relations (e.g., "A is stacked on top of B" along the vertical axis).
- The framework assumes that the reference viewpoint can be unambiguously determined from the question (requiring the reference person's position and orientation), whereas this information is sometimes implicit in natural language.
- Future work could integrate 3D scene graphs or world models to construct richer scene abstractions supporting more complex spatial reasoning.

## Related Work & Insights

- **vs. SpatialRGPT**: SpatialRGPT improves spatial reasoning by fine-tuning VLMs on spatial reasoning data, but the fine-tuning data are predominantly egocentric, offering limited gains for allocentric reasoning. APC's advantage lies in requiring no training data and achieving truly viewpoint-agnostic reasoning through explicit coordinate transformation.
- **vs. Novel View Synthesis (NVS)**: NVS-based methods (e.g., Zero-1-to-3) attempt to synthesize new-viewpoint images before VLM inference, but generation quality is unstable and computation is expensive. APC bypasses pixel-level reconstruction and operates directly at the abstraction level, yielding greater efficiency and robustness.
- **vs. 3DSRBench/3D-PC**: These are evaluation benchmarks for perspective reasoning; APC demonstrates substantial improvements on these benchmarks and can serve as a strong baseline for future perspective reasoning research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The cognitive-science-inspired angle of mental imagery simulation is novel, though the individual modules are combinations of existing tools.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Both synthetic and real-world scenarios are evaluated, and ablation studies cover key components; however, evaluation on larger-scale benchmarks is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, the method description is intuitive, and the logical chain from cognitive science to the technical solution is coherent and fluent.
- **Value**: ⭐⭐⭐⭐ The paper identifies and addresses an important capability gap in VLMs; the training-free design offers strong practicality and has direct implications for embodied intelligence applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](global_and_local_entailment_learning_for_natural_world_imagery.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](../../NeurIPS2025/multimodal_vlm/mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)
- [\[CVPR 2026\] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](../../CVPR2026/multimodal_vlm/simpact_simulation-enabled_action_planning_using_vision-language_models.md)
- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)

</div>

<!-- RELATED:END -->
