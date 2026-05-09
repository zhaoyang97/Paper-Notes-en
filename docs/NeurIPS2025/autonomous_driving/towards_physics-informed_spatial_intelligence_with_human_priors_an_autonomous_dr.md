---
title: >-
  [Paper Note] Towards Physics-Informed Spatial Intelligence with Human Priors: An Autonomous Driving Perspective
description: >-
  [NeurIPS 2025][Autonomous Driving][Spatial Intelligence Grid] This paper proposes the Spatial Intelligence Grid (SIG)—a structured representation inspired by the perspective grids used by Renaissance painters—that explicitly encodes object layout, directional relationships, and distance relationships in driving scenes as a grid structure. The authors further construct the SIGBench benchmark, demonstrating that SIG enables more stable and comprehensive improvements in the spatial reasoning capabilities of MLLMs under few-shot in-context learning compared to conventional VQA-based approaches.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - Spatial Intelligence Grid
  - Visual Spatial Reasoning
  - Human Priors
  - Multimodal Large Language Models
  - Driving Scene Understanding
date: 2026-05-08
content_hash: 4e26bff2cb65508d
---

# Towards Physics-Informed Spatial Intelligence with Human Priors: An Autonomous Driving Perspective

**Conference**: NeurIPS 2025
**arXiv**: [2510.21160](https://arxiv.org/abs/2510.21160)
**Code**: [Project Page](https://guanlinwu123.github.io/sigbench)
**Area**: Autonomous Driving / Spatial Intelligence
**Keywords**: Spatial Intelligence Grid, Visual Spatial Reasoning, Human Priors, Multimodal Large Language Models, Driving Scene Understanding

## TL;DR

This paper proposes the Spatial Intelligence Grid (SIG)—a structured representation inspired by the perspective grids used by Renaissance painters—that explicitly encodes object layout, directional relationships, and distance relationships in driving scenes as a grid structure. The authors further construct the SIGBench benchmark, demonstrating that SIG enables more stable and comprehensive improvements in the spatial reasoning capabilities of MLLMs under few-shot in-context learning compared to conventional VQA-based approaches.

## Background & Motivation

Visual Spatial Intelligence (VSI) is one of the key capabilities of multimodal large language models; however, the dominant paradigm for evaluating and enhancing VSI is VQA (Visual Question Answering), where spatial questions are posed and answered in natural language. The VQA paradigm suffers from inherent limitations: (1) it entangles linguistic ability with spatial reasoning, preventing disentangled evaluation; (2) text-based answers may obscure underlying geometric understanding; and (3) language shortcuts allow models to answer correctly without genuinely understanding spatial relationships.

Inspired by the historical practice of Renaissance painters (e.g., Dürer, da Vinci) who decomposed 3D scenes using perspective grid systems, this paper asks: can a grid structure serve as a machine representation of spatial intelligence? Just as painters trained spatial perception through grids and could subsequently draw any object, a model that learns spatial relationships via grid representations should generalize accordingly.

- **Starting Point**: Shifting VSI from pure text representation to graph-structured representation.
- **Core Idea**: SIG—a 10×10 grid that maps objects in driving scenes (vehicles, traffic signs, traffic lights) to discrete spatial positions, from which a Spatial Relationship Graph (SRG) and Spatial Relationship Paragraph (SRP) are extracted, accompanied by specially designed graph similarity evaluation metrics.

## Method

### Overall Architecture

The SIG system comprises three layers: (1) **Representation Layer**—encodes driving scenes as a 10×10 spatial grid JSON (object positions) + SRG (directed graph of directional and distance relationships) + SRP (textualized spatial relationship descriptions); (2) **Evaluation Layer**—three dedicated graph/spatial evaluation metrics (MLSM, SRGS, SRD); (3) **Human Prior Layer**—integrates human attention into SIG via gaze tracking and homographic transformation.

### Key Designs

1. **Spatial Intelligence Grid (SIG) Representation**:

    - A 10×10 grid in which each traffic entity (ego vehicle, other vehicles, traffic signs/lights) occupies one or more grid cells.
    - Vehicles are annotated with "color + type + index" labels (e.g., *black car 1*), ordered from left to right in the image.
    - The SRG (a directed graph with edges encoding direction and grid distance) and SRP (textualized descriptions of orientation and distance) are derived from the SIG.
    - Output format is JSON, facilitating model processing and evaluation.

2. **Three-Tier Evaluation Metric Framework**:

    - **Multi-Level Spatial Matching (MLSM)**: Analogous to HOTA in object tracking, this metric pairs predicted and ground-truth objects via bipartite matching, computing precision/recall/F1/association accuracy at multiple distance thresholds $\alpha$. Vehicles support three matching levels (type; type + index; type + index + color). Complexity: $O(n^3)$.
    - **Spatial Relationship Graph Similarity (SRGS)**: Based on Graph Edit Distance (GED), this metric computes the node and edge edit costs (substitution, deletion, insertion) required to transform the predicted SRG into the ground-truth SRG. The weighted total edit distance is normalized to a $[0,1]$ similarity score. Complexity: $O(n^3)$.
    - **Semantic Relationship Distance (SRD)**: Directional relationships are measured via a shortest-arc distance on an 8-direction circular scale; distance relationships are measured on a 5-level linear scale. Computes MAE/MSE/Accuracy. Complexity: $O(n)$.

3. **SIG-Driven In-Context Learning (ICL)**:

    - SIG is used as the demonstration format in few-shot ICL: each example consists of an image (with annotated bounding boxes) + ground-truth SIG (JSON) + SRP.
    - The model learns the mapping from object positions in the image to grid positions in SIG.
    - Compared against conventional VQA-MC (multiple-choice VQA) ICL.

4. **Human Gaze-Augmented SIG (Human-Like SIG)**:

    - A homography matrix $H$ projects human gaze heatmaps from image space onto the SIG grid space.
    - Attention-weighted SRGS: graph edit costs are multiplied by gaze weights, imposing larger penalties for errors on highly attended objects.
    - Attention-weighted SRD: spatial relationship distances are multiplied by the average attention weight of the two relevant objects.
    - Gaze prediction task: given the attention maps of the preceding 5 frames, predict the human gaze distribution for the current frame.

5. **SIGBench Benchmark**:

    - 1,423 driving scene frames, each annotated with SIG, SRP, and human gaze heatmaps.
    - Includes grid-level VSI tasks (SIGC generation, SRPF fill-in-the-blank) and human-analogous VSI tasks (gaze prediction, attention-weighted SIGC/SRPF).
    - Curated from 6 autonomous driving datasets, covering normal driving to accident scenarios.

### Loss & Training

This work primarily constitutes an evaluation and ICL framework and does not involve model training. All SIG evaluation metrics (MLSM, SRGS, SRD) can be computed in sub-millisecond time, satisfying real-time requirements (the most complex scene with 22 objects requires <1 ms).

## Key Experimental Results

### Main Results

**Zero-shot SIG Generation (SIGBench)**:

| Model | MLSM F1↑ | SRGS S↑ | SRD Dir. Acc↑ | SRD Dist. Acc↑ |
|-------|----------|---------|--------------|---------------|
| Human | 0.938 | 0.897 | 0.753 | 0.760 |
| GPT-4o | 0.458 | **0.337** | 0.144 | **0.313** |
| Gemini-2.5-Pro | **0.507** | 0.232 | **0.254** | 0.398 |
| Claude-3.7-Sonnet | 0.450 | 0.299 | 0.092 | 0.420 |
| Qwen-VL-2.5-32B | 0.375 | 0.248 | 0.113 | 0.128 |

### Ablation Study

**3-shot ICL Comparison (SIGBench-tiny)**:

| Model | ICL Type | MLSM F1↑ | SRGS S↑ | SRD Dir. Acc↑ | SRD Dist. Acc↑ |
|-------|----------|----------|---------|--------------|---------------|
| GPT-4o | Zero-shot | 0.462 | 0.327 | 0.186 | 0.346 |
| GPT-4o | ICL-MC | 0.468 | 0.324 | 0.218 | 0.365 |
| GPT-4o | **ICL-SIG** | **0.479** | **0.337** | **0.220** | **0.436** |
| Gemini-2.5-Pro | Zero-shot | 0.496 | 0.224 | 0.295 | 0.439 |
| Gemini-2.5-Pro | ICL-MC | 0.524 | 0.185↓ | 0.325 | 0.384↓ |
| Gemini-2.5-Pro | **ICL-SIG** | **0.565** | **0.305** | **0.316** | **0.493** |

### Key Findings

- Even the most capable MLLMs exhibit a substantial gap relative to human VSI: the best model achieves an MLSM F1 of only 0.507 (vs. human 0.938).
- ICL-SIG outperforms the zero-shot baseline on **all** VSI metrics, whereas ICL-MC causes some metrics to degrade (e.g., Gemini's SRGS drops from 0.224 to 0.185), confirming that SIG is a superior format for learning spatial reasoning.
- SIG-based ICL yields more stable and comprehensive improvements—unlike MC-based ICL, which can occasionally hurt performance.
- Directional relationship understanding is substantially harder than distance understanding: human direction accuracy is 0.753, while the best model achieves only 0.254.
- Small objects and highly overlapping objects (high IoU) are the primary failure modes for current models.

## Highlights & Insights

- **Interdisciplinary Inspiration**: Drawing on the perspective grid techniques of Renaissance painters to propose a structured representation for machine spatial intelligence is a novel and elegant conceptual contribution.
- **Evaluation Free from Text Bias**: MLSM, SRGS, and SRD are entirely based on spatial structure comparison rather than text matching, genuinely assessing spatial understanding rather than linguistic ability.
- **ICL-SIG vs. ICL-MC Comparison**: The ablation experiments convincingly demonstrate that structured spatial representations are better suited than multiple-choice VQA for teaching spatial reasoning to models—VQA-based ICL can sometimes introduce linguistic bias and hurt performance.
- **Integration of Human Gaze**: Incorporating human gaze data aligns evaluation more closely with human attentional patterns in driving, reflecting a human-centered design philosophy.

## Limitations & Future Work

- The fixed 10×10 grid resolution is relatively coarse and may be insufficient for precise localization tasks (e.g., lane-level positioning).
- Validation is currently limited to autonomous driving scenes; generalization to other VSI settings such as indoor environments and robotics remains to be explored.
- SIG annotation requires manual effort (bounding boxes + attributes + grid positions), resulting in relatively high annotation costs.
- Only ICL (few-shot) is evaluated; using SIG as a training data format for fine-tuning has not been investigated.
- Evaluation accuracy is bounded by the quality of bounding box detection—detection errors are conflated with spatial reasoning errors.
- Temporal spatial reasoning at the video level is not considered; the current approach processes only individual frames.

## Related Work & Insights

- **vs. SRBench / VSR**: These benchmarks evaluate spatial relationships solely via text; SIGBench is the first to introduce graph-level structural evaluation and human gaze mechanisms.
- **vs. NuScenes-SpatialQA**: Also a driving-scene VSI benchmark, but remains in VQA format; SIGBench provides richer structured evaluation.
- **Implications for VSI Research**: SIG can serve as a general-purpose intermediate representation for spatial intelligence, useful not only for evaluation but also as a training signal—integrating SIG into MLLM training pipelines may be an effective avenue for improving VSI.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Drawing inspiration from Renaissance painters to propose the SIG representation demonstrates highly creative interdisciplinary thinking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-model evaluation, ICL comparisons, and human baselines are included, but fine-tuning experiments using SIG as training data are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is elegantly articulated and the methodology is clearly presented, though the abundance of formulas and notation makes certain sections dense.
- **Value**: ⭐⭐⭐⭐ — The SIG representation and evaluation metrics have the potential to become standard tools in spatial intelligence research, though broader empirical validation is needed before practical adoption.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](towards_predicting_any_human_trajectory_in_context.md)
- [\[NeurIPS 2025\] Predictive Preference Learning from Human Interventions](predictive_preference_learning_from_human_interventions.md)
- [\[NeurIPS 2025\] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving](futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)
- [\[NeurIPS 2025\] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)

<!-- RELATED:END -->
