---
title: >-
  [Paper Note] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics
description: >-
  [NeurIPS 2025][Multimodal VLM][Spatial Referring] This paper proposes **RoboRefer**, a 3D-aware reasoning VLM trained via a two-stage **SFT + RFT** strategy with a metric-sensitive process reward function. It achieves pr…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Spatial Referring"
  - "Vision-Language Models"
  - "Multi-Step Reasoning"
  - "Reinforcement Fine-Tuning"
  - "Robot Manipulation"
date: 2026-05-08
content_hash: 0c8ebcb246f086e9
---

# RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics

**Conference**: NeurIPS 2025
**arXiv**: [2506.04308](https://arxiv.org/abs/2506.04308)  
**Code**: [zhoues.github.io/RoboRefer](https://zhoues.github.io/RoboRefer)  
**Area**: Multimodal VLM
**Keywords**: Spatial Referring, Vision-Language Models, Multi-Step Reasoning, Reinforcement Fine-Tuning, Robot Manipulation
**arXiv**: [2506.04308](https://arxiv.org/abs/2506.04308)  
**Code**: None  
**Area**: Multimodal VLM

## TL;DR

This paper proposes **RoboRefer**, a 3D-aware reasoning VLM trained via a two-stage **SFT + RFT** strategy with a metric-sensitive process reward function. It achieves precise single-step spatial understanding and multi-step spatial reasoning on spatial referring tasks, surpassing Gemini-2.5-Pro by 17.4% on RefSpatial-Bench.

## Background & Motivation

**Background**: Spatial referring is a fundamental capability for embodied AI — robots must interpret spatial constraint instructions such as "place the cup between the keyboard and the pen holder, aligned with the cup's logo."

**Limitations of Prior Work**: Existing VLM approaches either require costly 3D reconstruction that introduces modality gaps, or treat depth as RGB input within a shared encoder, causing modality interference. Moreover, nearly all prior methods handle only single-step spatial understanding (e.g., "which is closer?") and cannot perform multi-step reasoning.

**Key Challenge**: (1) Existing datasets cover only 15 spatial relations and lack multi-step reasoning annotations; (2) SFT training tends to memorize answers rather than generalize reasoning; (3) No benchmark exists for evaluating multi-step spatial reasoning.

**Goal**: To construct a VLM with both precise 3D spatial perception and multi-step spatial reasoning capabilities, applicable to real-world robot manipulation and navigation.

**Key Insight**: A dedicated depth encoder avoids modality interference, while reinforcement fine-tuning produces generalizable reasoning strategies.

**Core Idea**: SFT learns spatial perception; RFT learns spatial reasoning — guided by a metric-sensitive process reward that supervises the precision of intermediate reasoning steps.

## Method

### Overall Architecture (Figure 2)

RoboRefer employs separate RGB and depth encoders to extract features independently, each projected into the LLM token space via dedicated projectors. Training proceeds in three stages: (1) depth alignment, (2) SFT for spatial understanding enhancement, and (3) RFT for multi-step reasoning enhancement.

### Key Design 1: Dedicated Depth Encoder

- **Function**: Provides a dedicated encoding pathway for the depth modality, eliminating interference with the RGB encoder.
- **Mechanism**: The depth encoder and depth projector are initialized from their RGB counterparts. During joint training, the RGB encoder remains unaffected by depth inputs, while the depth encoder is updated independently.
- **Design Motivation**: Shared-encoder approaches (e.g., SpatialRGPT) require more than twice the RGB data to compensate for modality interference and degrade the pretrained image encoder. Tab. 4 demonstrates that the dedicated encoder incurs negligible loss on general VQA.

### Key Design 2: Two-Stage SFT Training

1. **Depth Alignment**: Only the depth projector is trained, aligning the depth space to the text space using RGB-D annotations from RefSpatial.
2. **Spatial Understanding Enhancement**: All parameters are fine-tuned using both RGB and RGB-D inputs, compelling the image encoder to learn spatial understanding beyond depth cues. Training data includes single-step annotations and multi-step reasoning process data (serving as a "cold start" for RFT).

### Key Design 3: RFT with Metric-Sensitive Process Reward

- **Function**: Applies GRPO reinforcement learning on top of the SFT model to further improve the generalizability of multi-step reasoning.
- **Four Reward Functions**:
    - **Outcome Format Reward** $R_{OF}$: rewards correct output format
    - **Point L1 Reward** $R_P$: binary reward for whether the final predicted point falls within the target region
    - **Process Format Reward** $R_{PF}$: rewards correct intermediate step format "[Perception Type] [Target Object]:"
    - **Accuracy Reward** $R_{Acc}$: measures prediction error at each key step according to perception type (e.g., L1 distance for coordinates)
- **Total Reward**: $r_i = R_{OF}(a_i) + R_P(a_i) + \alpha R_{PF}(a_i) + \alpha R_{Acc}(a_i)$, where $\alpha = 0.25$
- **Key Detail**: The process reward is order-invariant, imposing no fixed ordering on the reasoning trajectory.
- **Design Motivation**: SFT tends to memorize answers, whereas RFT learns more generalizable reasoning strategies through exploration (sampling $N$ responses) and reward-driven supervision.

### Key Design 4: RefSpatial Dataset

- **Scale**: 2.5M samples, 20M QA pairs (2× the size of the previous largest dataset)
- **Three Data Sources**: 2D web images (spatial concepts + broad depth perception) → 3D embodied videos (fine-grained indoor spatial understanding) → simulation data (multi-step reasoning process annotations)
- **Covers 31 spatial relations** (vs. 15 in prior work), with up to 5-step reasoning chains
- **Fine-Grained Annotations**: Each object has hierarchical descriptions (e.g., "cup" → "the cup closest to the camera"), enabling unambiguous referring in cluttered scenes

## Key Experimental Results

### Main Results: Single-Step Spatial Understanding (Table 1)

| Method | Input | CV-Bench Avg | BLINK Avg | RoboSpatial | SAT |
|---|---|---|---|---|---|
| Gemini-2.5-Pro | RGB | 91.74 | 89.17 | 77.24 | 70.59 |
| SpatialRGPT-8B | RGB-D | 89.77 | 85.32 | 66.67 | 64.00 |
| **RoboRefer-8B-SFT** | **RGB-D** | **96.24** | **92.18** | **84.55** | **86.67** |

### Main Results: Multi-Step Spatial Referring (Table 2)

| Method | RefSpatial-Bench-L. | RefSpatial-Bench-P. | RefSpatial-Bench-U. |
|---|---|---|---|
| Gemini-2.5-Pro | 46.96 | 24.21 | 27.14 |
| Molmo-72B | 45.77 | 14.74 | 21.24 |
| RoboRefer-2B-SFT | 47.00 | 48.00 | 33.77 |
| **RoboRefer-2B-RFT** | **52.00** | **54.00** | **41.56** |

**On unseen spatial relation combinations (Unseen), RFT outperforms SFT by 9.1%.**

### 2D Referring Tasks (Table 3)

| Method | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|---|---|---|---|
| Qwen2.5-VL-72B (B.→P.) | 95.4 | 91.5 | 92.5 |
| **RoboRefer-8B-SFT** | **96.6** | **91.9** | **94.3** |

### Ablation Study (Table 7)

| Configuration | CV-Bench | BLINK |
|---|---|---|
| w/o 2D data | 84.17 | 74.48 |
| w/o 3D data | 81.83 | 74.61 |
| w/o simulation data | 83.96 | 75.10 |
| w/o depth encoder | 91.24 | 85.27 |
| **Full model** | **94.77** | **89.27** |

### Key Findings

- **Data recipe is critical**: All three data sources are indispensable; removing 2D data severely degrades outdoor scene performance (BLINK), while removing 3D data heavily impacts indoor scenes (CV-Bench).
- **The depth encoder yields larger gains in multi-step reasoning**: Errors in intermediate steps accumulate, amplifying the value of depth cues.
- **Process reward contributes approximately 5 percentage points of improvement** (Tab. 7, RFT section).
- **Real-robot validation (UR5 and G1 humanoid)**: The model successfully completes long-horizon dynamic tasks requiring multi-step spatial reasoning (Tab. 5/6).

## Highlights & Insights

1. **Progressive SFT + RFT training paradigm**: SFT provides a spatial-perception "cold start," while RFT unlocks generalizable reasoning capability, mitigating the memorization tendency of pure SFT.
2. **Metric-sensitive process reward is the key innovation**: Unlike outcome rewards that only inspect the final answer, the process reward precisely supervises the accuracy of intermediate predictions at each reasoning step.
3. **Point prediction instead of bounding boxes**: For robotics, single-point prediction is more natural — it directly maps to 3D coordinates, avoids occlusion issues, and unifies grasping, placement, and navigation targets.
4. **A 2B model outperforms Gemini-2.5-Pro by 17.4%**: On the specific capability of multi-step spatial reasoning, a specialized small model can substantially exceed a general-purpose large model.

## Limitations & Future Work

1. **2D point prediction only**: An additional depth-to-3D mapping step is required; directly predicting 3D points is a natural future direction.
2. **Limited human intent understanding**: Human instructions are often brief and ambiguous; improved intent inference capability is needed.
3. **RFT validated only on the 2B model**: Computational constraints precluded RFT experiments on the 8B model; scaling effects remain unknown.
4. **Limited quantitative spatial relations**: The model primarily handles qualitative relations (left/right/near/far); precise metric reasoning (e.g., "10 cm from the table edge") remains challenging.
5. **Simulation-to-real transfer**: Although real-world feasibility is demonstrated, the reasoning annotation patterns derived from simulation data may introduce a domain gap.

## Related Work & Insights

- **vs. RoboPoint**: RoboPoint relies only on basic spatial cues in images, supporting neither multi-step reasoning nor 3D depth perception.
- **vs. SpatialRGPT**: SpatialRGPT addresses simpler VQA tasks and requires external mask/detection tool inputs, whereas RoboRefer localizes targets directly from text instructions.
- **Implications for embodied AI**: Spatial referring can unify manipulation and navigation — a single VLM simultaneously provides target points for grasping and locomotion.

## Rating

⭐⭐⭐⭐⭐ (5/5)

A comprehensive contribution covering a new task formulation (multi-step spatial referring), a new dataset (RefSpatial, 20M QA pairs), a new training paradigm (SFT + RFT + process reward), a new benchmark (RefSpatial-Bench), and real-robot validation. The 2B model's substantial margin over Gemini-2.5-Pro is particularly impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)
- [\[NeurIPS 2025\] Sherlock: Self-Correcting Reasoning in Vision-Language Models](sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[NeurIPS 2025\] GoalLadder: Incremental Goal Discovery with Vision-Language Models](goalladder_incremental_goal_discovery_with_vision-language_models.md)

</div>

<!-- RELATED:END -->
