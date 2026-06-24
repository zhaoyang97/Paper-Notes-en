---
title: >-
  [Paper Note] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models
description: >-
  [NeurIPS 2025][VLM Reasoning][Affordance] This paper introduces a fine-grained 3D embodied reasoning task—jointly predicting the spatial location, motion type, and motion axis of actionable elements—and proposes rendering 3D point clouds into panoramic views with projected affordance candidates, guided by a customized Chain-of-Thought (CoT) reasoning paradigm for MLLMs, achieving state-of-the-art performance with AP25 of 23.3%.
tags:
  - "NeurIPS 2025"
  - "VLM Reasoning"
  - "Affordance"
  - "3D Reasoning"
  - "Chain-of-Thought"
  - "MLLM"
  - "Motion Estimation"
date: 2026-05-08
content_hash: a778da03af5fb254
---

# AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.10017](https://arxiv.org/abs/2511.10017)  
**Code**: To be confirmed  
**Area**: 3D Vision / Embodied Intelligence / Multimodal VLM
**Keywords**: Affordance, 3D Reasoning, Chain-of-Thought, MLLM, Motion Estimation

## TL;DR
This paper introduces a fine-grained 3D embodied reasoning task—jointly predicting the spatial location, motion type, and motion axis of actionable elements—and proposes rendering 3D point clouds into panoramic views with projected affordance candidates, guided by a customized Chain-of-Thought (CoT) reasoning paradigm for MLLMs, achieving state-of-the-art performance with AP25 of 23.3%.

## Background & Motivation
**Background**: MLLMs have made progress in 3D scene understanding, but existing methods remain at the object-level recognition and localization stage.

**Limitations of Prior Work**:
   - Existing 3D understanding methods perform only object-level grounding, neglecting fine-grained affordance reasoning at the part level.
   - SceneFun3D introduces affordance grounding but treats grounding and motion estimation as separate tasks.
   - Video-based methods suffer from information redundancy, limited viewpoints, and slow processing.

**Key Challenge**: MLLMs natively support 2D inputs, yet affordance reasoning requires 3D spatial understanding and physical grounding.

**Goal**: Given a 3D scene and a language instruction, jointly predict the mask, motion type, and motion axis direction of affordance elements.

**Key Insight**: 3D-to-2D projection + active viewpoint selection + multi-step CoT reasoning.

**Core Idea**: Project 3D affordance candidates onto panoramic views, then use CoT to guide the MLLM to sequentially select a viewpoint, localize the target, and reason about motion.

## Method

### Overall Architecture
**Input**: A 3D point cloud scene and a natural language instruction (e.g., "unplug the Christmas tree lights"). (1) 360° panoramic rendering generates $N$ candidate views; (2) 3D instance segmentation extracts affordance elements and constructs geometric-semantic descriptors; (3) 3D information is projected onto 2D views with annotations; (4) CoT reasoning: active viewpoint selection → affordance localization → motion inference. **Output**: A triplet $\{$3D mask, motion type, motion axis direction$\}$ for each element.

### Key Designs

1. **Holistic Multimodal Representation**:

    - **Function**: Converts the 3D scene into a 2D enriched representation that MLLMs can process.
    - **Mechanism**: Performs a 360° horizontal scan centered at the scene origin to generate $N$ uniformly distributed panoramic views. For each 3D affordance element, geometric descriptors (centroid $C_j$ and extent $\Sigma_j$) and a semantic descriptor $S_j$ are extracted and projected onto the 2D views as annotated bounding boxes with IDs.
    - **Design Motivation**: Provides more complete visual coverage than video frames and avoids the problem of key anchors and targets appearing in different frames.

2. **Adaptive Labeling**:

    - **Function**: Resolves label overlap caused by 2D projection.
    - **Mechanism**: Multiple candidate anchor positions are predefined for each projected bounding box; positions are checked one by one for non-overlap, and the first suitable position is selected for label placement.
    - **Design Motivation**: Prevents label stacking that would hinder MLLM recognition.

3. **Customized CoT Reasoning Paradigm**:

    - **Function**: Three-step structured reasoning—observation → localization → motion inference.
    - **Step 1 — Active Viewpoint Selection**: The MLLM receives all annotated views and the instruction, then autonomously selects the most informative viewpoint (with optional detail zooming).
    - **Step 2 — Affordance Localization**: Based on the selected view and instruction, the MLLM identifies the ID of the target element.
    - **Step 3 — Motion Estimation**: Based on the instruction and localization result, the MLLM infers the motion type (rotation, translation, etc.) and motion axis direction (horizontal inward/outward, vertical, etc.).
    - **Design Motivation**: Decomposes complex reasoning into interpretable steps, each grounded in spatial input and task intent.

### Loss & Training
- The 3D instance segmentation module is trained with Dice loss and Cross-entropy loss.
- A coarse-to-fine curriculum learning strategy is adopted: the dilation radius of the ground-truth mask is progressively reduced as $\delta_t = \delta_0 \cdot \beta^{\lfloor t/\tau \rfloor}$.
- The MLLM component (Qwen2.5-VL-72B) is used directly without additional fine-tuning.

## Key Experimental Results

### Main Results
Affordance grounding and motion estimation on the SceneFun3D dataset.

| Method | Raw 2D Input | mIoU | AP25 | +T (Motion Type) | +TD (Type+Direction) |
|--------|-------------|------|------|------------------|----------------------|
| OpenMask3D | ✓ | - | 0.0 | - | - |
| LERF | ✓ | - | Low | - | - |
| Fun3DU | ✓ | - | Lower | - | - |
| **AffordBot** | ✗ | **14.0** | **23.3** | **18.3** | **10.8** |

### Ablation Study

| Configuration | AP25 | Note |
|---------------|------|------|
| w/o Active View Selection | Lower | No active viewpoint selection |
| w/o CoT (direct prediction) | Lower | No chain-of-thought reasoning |
| LLaVA-34B (MLLM replaced) | 20.0 | Weaker MLLM |
| GPT-4o | 28.9 | Stronger MLLM |
| GPT-o1 | **33.4** | Strongest reasoning MLLM |

### Key Findings
- Stronger MLLMs (GPT-o1 vs. Qwen2.5-VL) yield substantial gains (23.3 → 33.4 AP25), demonstrating that the framework scales with MLLM capability.
- Multi-target scenes (Multiple) outperform single-target scenes (Unique), as multiple elements provide richer contextual cues.
- Performance varies drastically across affordance types: *foot_push* achieves 100%, while *rotate* achieves only 2.5%, indicating that certain operation types remain highly challenging.

## Highlights & Insights
- **Video-free 3D-to-2D Bridging**: Panoramic rendering replaces video frames, providing complete scene coverage with zero redundancy—a practical approach for enabling MLLMs to process 3D scenes.
- **Active Viewpoint Selection**: Allowing the MLLM to autonomously determine "where to look" mirrors human exploratory behavior, improving reasoning focus and accuracy.
- **MLLM Scalability**: The framework requires no MLLM fine-tuning and directly benefits from stronger models (GPT-o1 yields +10 AP25)—a sound engineering design.
- **Unified Task Formulation**: Unifying affordance grounding and motion estimation into triplet prediction more closely reflects the demands of real-world robotic manipulation.

## Limitations & Future Work
- Overall performance remains modest (AP25 of only 23.3%); the joint metric +TD reaches only 10.8%, leaving substantial room for improvement before practical deployment.
- The pipeline depends on the upstream segmentation quality of Mask3D; segmentation failures propagate through the entire system.
- Discretization of motion axis directions sacrifices continuous directional precision.
- Validation on physical robots is absent; evaluation is conducted offline on SceneFun3D only.
- End-to-end training or fine-tuning of the MLLM could be explored to further boost performance.

## Related Work & Insights
- **vs. SceneFun3D**: SceneFun3D treats grounding and motion estimation as separate, instruction-agnostic tasks; AffordBot unifies them into an instruction-conditioned joint prediction task.
- **vs. Fun3DU**: Fun3DU relies on video frames and a VLM+SAM pipeline; AffordBot renders directly from point clouds, bypassing the video processing bottleneck.
- **vs. 3D-LLM / LEO**: These methods operate at the object level, whereas AffordBot descends to part-level affordance reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Unified task formulation with a CoT reasoning paradigm; the problem definition is clear and valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-MLLM comparisons, component-wise ablations, and per-category analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Well-designed figures and tables; method description is clear and coherent.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for fine-grained manipulation reasoning in embodied intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments](physvlm-avr_active_visual_reasoning_for_multimodal_large_language_models_in_phys.md)
- [\[ICLR 2026\] VisuRiddles: Fine-Grained Perception is a Primary Bottleneck for Multimodal Large Language Models in Abstract Visual Reasoning](../../ICLR2026/vlm_reasoning/visuriddles_fine-grained_perception_is_a_primary_bottleneck_for_multimodal_large.md)
- [\[NeurIPS 2025\] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](act_as_human_multimodal_large_language_model_data_annotation.md)
- [\[NeurIPS 2025\] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)

</div>

<!-- RELATED:END -->
