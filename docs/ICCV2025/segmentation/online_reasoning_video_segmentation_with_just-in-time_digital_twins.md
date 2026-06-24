---
title: >-
  [Paper Note] Online Reasoning Video Segmentation with Just-in-Time Digital Twins
description: >-
  [ICCV 2025][Segmentation][Reasoning Segmentation] This paper proposes a multi-agent framework based on the concept of "Just-in-Time Digital Twins" that decouples perception from reasoning. Without any LLM fine-tuning, the framework enables online video reasoning segmentation and comprehensively outperforms existing methods across semantic, spatial, and temporal reasoning tasks.
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Reasoning Segmentation"
  - "Digital Twin"
  - "Video Understanding"
  - "Multi-Agent Framework"
  - "Online Processing"
date: 2026-05-08
content_hash: c9f80eb077d52b76
---

# Online Reasoning Video Segmentation with Just-in-Time Digital Twins

**Conference**: ICCV 2025
**arXiv**: [2503.21056](https://arxiv.org/abs/2503.21056)  
**Code**: None  
**Area**: Video Segmentation / Reasoning Segmentation
**Keywords**: Reasoning Segmentation, Digital Twin, Video Understanding, Multi-Agent Framework, Online Processing

## TL;DR

This paper proposes a multi-agent framework based on the concept of "Just-in-Time Digital Twins" that decouples perception from reasoning. Without any LLM fine-tuning, the framework enables online video reasoning segmentation and comprehensively outperforms existing methods across semantic, spatial, and temporal reasoning tasks.

## Background & Motivation

Reasoning Segmentation (RS) aims to identify and segment objects of interest based on implicit textual queries—e.g., "segment the object used to hold hot beverages" rather than "coffee cup"—and is a core capability for embodied intelligence.

Three key limitations of existing RS methods:

**Limited reasoning capacity**: Relying on multimodal LLMs to jointly handle perception and reasoning leads to poor performance on queries requiring multi-step or complex spatial/temporal reasoning. LLMs must compress rich visual information into a limited number of tokens, losing fine-grained spatial and temporal details.

**High maintenance cost**: LLM fine-tuning is required, and as LLMs iterate rapidly, repeated re-tuning is necessary to avoid catastrophic forgetting.

**No support for online processing**: Existing methods are primarily designed for static images or offline videos and cannot handle real-time video streams.

## Method

### Overall Architecture

A two-stage pipeline: **Planning Phase** → **Execution Phase**

- Planning Phase: An LLM planner analyzes the implicit query, constructs an execution graph (DAG), and selects the necessary specialist vision models.
- Execution Phase: The video is processed online frame by frame; a digital twin is constructed and maintained, reasoning operations are executed, and segmentation masks are produced.

### Key Designs

1. **Query-Driven Specialist Vision Model Selection**:

    - The LLM planner analyzes the semantic, spatial, and temporal requirements of the query.
    - A structured prompt template is used to output a JSON configuration specifying the required models and their rationales.
    - For example, "segment the object that moved behind the dining table after the person sat down" → requires SAM-2 (segmentation) + DepthAnything-2 (spatial relations).
    - Core Idea: **activate specific models only when needed**, rather than always running all models, thereby reducing computational overhead.

2. **Just-in-Time Digital Twin Construction**:

    - For each frame $I^{(t)}$, a scene graph $G_s^{(t)} = (V_s^{(t)}, E_s^{(t)})$ is constructed.
    - Node attributes contain three-dimensional features: $\text{attr}(v_{i,s}^{(t)}) = [h_{\text{vis}}, h_{\text{spa}}, h_{\text{temp}}]$ (visual, spatial, temporal).
    - Edges represent inter-object relationships (e.g., "behind," "above," "moving towards").
    - **On-demand construction**: Unlike traditional digital twins that maintain a complete representation, only the information subset required by the query is generated and updated.
    - A sliding-window mechanism maintains temporal consistency: $SG^{(t)} = \{G_s^{(t)} | t-w \leq k \leq t\}$

3. **Reasoning Graph Construction and Execution**:

    - Reasoning is modeled as a DAG: $G = (V, E)$, where $V = V_p \cup V_s \cup V_r$.
    - $V_p$: perception nodes (specialist vision models), $V_s$: state nodes (maintaining the digital twin), $V_r$: reasoning nodes.
    - Two types of reasoning nodes:
        - **Semantic Reasoning**: handled by the base LLM (gpt-4o-mini), which formats the digital twin state as natural language context.
        - **Spatial/Temporal Reasoning**: handled by the LLM-coder (gpt-4o), which generates executable code to operate on the scene graph.
    - Example for evaluating a "behind" relation: $\text{Behind}(v_i, v_j) = (h_{\text{spa}}^i[z] > h_{\text{spa}}^j[z]) \wedge \text{Overlap}(v_i, v_j)$

### Loss & Training

This method **requires no training** and is entirely based on a combination of pretrained models:
- gpt-4o-mini serves as the planner and semantic reasoner.
- gpt-4o serves as the code generator.
- SAM-2 for segmentation, DepthAnything-2 for spatial relations, OWLv2 for object detection, DINOv2 for visual feature extraction.
- Temporal smoothing coefficient $\alpha = 0.8$, tracking function $\lambda = 0.5$, default window size $w = 6$.

## Key Experimental Results

### Main Results — Video Reasoning Segmentation

A newly constructed benchmark comprising 200 videos and 895 implicit queries covers three reasoning types (semantic/spatial/temporal) and three difficulty levels (L1/L2/L3).

| Method | Sem.-L1 | Sem.-L3 | Spa.-L1 | Spa.-L3 | Tem.-L1 | Tem.-L3 |
|--------|---------|---------|---------|---------|---------|---------|
| LISA-7B | 0.635 | 0.274 | 0.226 | 0.229 | 0.398 | 0.229 |
| LISA-13B | 0.669 | 0.301 | 0.258 | 0.234 | 0.237 | 0.177 |
| VISA | 0.563 | 0.432 | 0.521 | 0.411 | 0.354 | 0.218 |
| **Ours** | **0.865** | **0.810** | **0.789** | **0.741** | **0.721** | **0.690** |

The proposed method leads by a large margin across all categories and difficulty levels, with particularly pronounced advantages in spatial reasoning (+26.8% vs. VISA) and temporal reasoning (+47.2% vs. VISA).

### Ablation Study

| Model Selection | DT Update | Temporal Integration | Sem.-L1 | Spa.-L1 | Tem.-L1 |
|----------------|-----------|---------------------|---------|---------|---------|
| ✗ | ✓ | ✓ | 0.821 | 0.753 | 0.701 |
| ✓ | ✗ | ✓ | 0.831 | 0.721 | 0.675 |
| ✓ | ✓ | ✗ | 0.842 | 0.757 | 0.654 |
| ✓ | ✓ | ✓ | **0.865** | **0.789** | **0.721** |

LLM configuration ablation (semantic reasoning):

| Base LLM | LLM-coder | L1 | L2 | L3 |
|----------|-----------|-----|-----|-----|
| gpt4o-mini | gpt4o-mini | 0.832 | 0.804 | 0.801 |
| gpt4o-mini | gpt4o | 0.865 | 0.841 | 0.810 |
| gpt4o | gpt4o | 0.879 | 0.865 | 0.822 |

### Key Findings

- Existing methods (LISA-13B) suffer a sharp performance drop from L1 to L3 ($\mathcal{J}$: 0.669→0.301), whereas the proposed method remains stable (0.865→0.810), with less than 10% degradation across difficulty levels.
- The method also achieves state-of-the-art performance on the ReVOS benchmark (Overall $\mathcal{J}$: 0.748 vs. VISA 0.488).
- It likewise achieves SOTA on image reasoning segmentation (ReasonSeg) (long query gIoU: 69.5 vs. LISA-13B 63.2).
- Disabling digital twin updates has the greatest impact on temporal reasoning; disabling temporal integration also significantly affects temporal reasoning performance.

## Highlights & Insights

- **Perception–Reasoning Decoupling**: Avoids having the LLM directly process pixel-level visual information; specialist models are used to preserve fine-grained spatial and temporal details.
- **"Just-in-Time" Digital Twin Concept**: Scene representations are constructed on demand, balancing computational efficiency with information completeness.
- **Fine-Tuning-Free Design**: The modular architecture allows any LLM or vision model to be replaced with a better alternative at any time, minimizing maintenance cost.
- **Online Processing Capability**: Real-time frame-by-frame video stream processing makes the system suitable for practical deployment in embodied AI scenarios.
- **Code-Generation-Based Reasoning**: Spatial and temporal reasoning is converted into executable code, circumventing the limitations of LLMs in handling numerical computation.

## Limitations & Future Work

- Dependence on the GPT-4o API results in relatively high inference costs and latency.
- The robustness of the scene-graph-based digital twin representation under extreme conditions such as occlusion and rapid motion is not thoroughly discussed.
- The benchmark is of moderate scale (200 videos, 895 queries); validation at larger scale remains to be conducted.
- Errors in the planning phase may cascade and affect subsequent execution steps in an unrecoverable manner.
- The sliding window size is fixed at 6 frames, which may be insufficient for queries with very long temporal dependencies.

## Related Work & Insights

- LISA pioneered the embedding-as-mask paradigm, but its single-token design limits multi-step reasoning.
- VISA was the first to extend RS to the video domain, but frame sampling may cause critical temporal information to be missed.
- The digital twin concept is borrowed from the industrial/robotics domain and introduced into computer vision, representing a meaningful cross-domain transfer.
- Using LLMs as planners and reasoners rather than end-to-end perception models constitutes a more flexible and scalable system design paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The "Just-in-Time Digital Twin" concept is novel; the perception–reasoning-decoupled agent design is pioneering in the context of video RS.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The newly constructed benchmark covers three reasoning types and three difficulty levels; multi-dataset evaluation and detailed ablations are provided.
- **Writing Quality**: ⭐⭐⭐⭐ The presentation is clear, the formalization is complete, and the system design is well described.
- **Value**: ⭐⭐⭐⭐⭐ The work makes an important contribution to embodied AI and video understanding; its design philosophy is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation](veggie_instructional_editing_and_reasoning_video_concepts_with_grounded_generati.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[ICCV 2025\] Online Generic Event Boundary Detection](online_generic_event_boundary_detection.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[CVPR 2026\] VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation](../../CVPR2026/segmentation/virst_video-instructed_reasoning_assistant_for_spatiotemporal_segmentation.md)

</div>

<!-- RELATED:END -->
