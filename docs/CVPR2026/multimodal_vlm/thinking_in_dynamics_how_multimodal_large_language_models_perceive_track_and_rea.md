---
title: >-
  [Paper Note] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World
description: >-
  [CVPR 2026][Multimodal VLM][4D dynamics] This paper proposes Dyn-Bench — a large-scale benchmark for dynamic understanding of the physical 4D world (1k videos, 7k VQA pairs…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "4D dynamics"
  - "Dyn-Bench benchmark"
  - "spatio-temporal reasoning"
  - "dynamic grounding"
  - "MLLM evaluation"
date: 2026-05-08
content_hash: 419e17a9cc6c6486
---

# Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World

**Conference**: CVPR 2026
**arXiv**: [2603.12746](https://arxiv.org/abs/2603.12746)  
**Code**: [https://dyn-bench.github.io/](https://dyn-bench.github.io/)  
**Area**: Multimodal VLM / Video Spatio-Temporal Reasoning
**Keywords**: 4D dynamics, Dyn-Bench benchmark, spatio-temporal reasoning, dynamic grounding, MLLM evaluation

## TL;DR
This paper proposes Dyn-Bench — a large-scale benchmark for dynamic understanding of the physical 4D world (1k videos, 7k VQA pairs, 3k dynamic grounding pairs) — that systematically evaluates the spatio-temporal reasoning capabilities of general, spatial-aware, and region-level MLLMs. The study finds that existing models fail to maintain consistency between reasoning and grounding simultaneously, and introduces two structured integration methods, Mask-Guided Fusion and ST-TCM, that significantly improve dynamic perception.

## Background & Motivation

### State of the Field
Humans inhabit a physical 4D world in which geometric structure and semantic content evolve over time. While current MLLMs demonstrate strong performance on static image understanding, their ability to comprehend dynamics in video — i.e., perceiving, tracking, and reasoning about spatio-temporal dynamics — has not been systematically evaluated.

### Limitations of Prior Work
1. No benchmark specifically evaluates MLLMs' spatio-temporal reasoning in **dynamic 4D scenes** — existing video QA datasets primarily focus on event description rather than spatial dynamics.
2. Existing models exhibit **inconsistency** between spatio-temporal reasoning and dynamic object grounding — a model may correctly answer "the ball moved to the left" yet fail to accurately localize the motion trajectory in the video.
3. Conventional prompting strategies (e.g., CoT, caption-based hints) yield limited improvements for dynamic reasoning.

### Root Cause
Success in static image understanding does not transfer directly to dynamic scenes — spatio-temporal dynamics involve complex reasoning over motion trajectories, object interactions, and physical causality, necessitating dedicated modeling.

### Core Idea
Dyn-Bench is constructed to evaluate MLLMs' dynamic understanding across multiple dimensions (linguistic reasoning + visual grounding), and two structured integration methods (Mask-Guided Fusion + ST-TCM) are proposed to enhance dynamic perception.

## Method

### Overall Architecture
The Dyn-Bench construction pipeline processes large-scale 2D (video) and 4D (point cloud sequence) data sources through a multi-stage filtering procedure to obtain a high-quality collection of dynamic scenes. Evaluation covers two primary tasks:
1. **Spatio-Temporal VQA**: answering spatio-temporal reasoning questions about dynamic events (7k pairs).
2. **Dynamic Object Grounding**: localizing objects participating in dynamic interactions within video frames (3k pairs).

### Key Designs

#### 1. Dyn-Bench Data Construction (Multi-stage Filtering Pipeline)
- **Function**: Filters genuinely dynamic interactive scenes from multi-source video/4D data including Ego4D and RealWorld-4D.
- **Mechanism**: Three-stage filtering — (a) motion detection to remove static scenes; (b) semantic diversity filtering to eliminate repetitive actions; (c) manual annotation quality control.
- **Design Motivation**: Existing video datasets contain substantial "pseudo-dynamic" content (e.g., camera motion with a static scene), requiring careful selection of authentic object dynamics.

#### 2. Mask-Guided Fusion (MGF)
- **Function**: Fuses segmentation masks with video frames to direct MLLMs' attention toward specific dynamic objects.
- **Mechanism**: Segmentation masks of objects (highlighting moving objects) are overlaid onto video frames and provided as an additional visual input channel.
- **Design Motivation**: MLLMs are prone to background distraction when processing full-frame video; mask guidance explicitly focuses attention on dynamic objects.
- **Effect**: MGF significantly improves grounding accuracy compared to unguided standard input.

#### 3. Spatio-Temporal Textual Cognitive Map (ST-TCM)
- **Function**: Converts spatio-temporal video dynamics into a structured cognitive map in text form as auxiliary input to the MLLM.
- **Mechanism**: ST-TCM comprises: (a) per-frame object position coordinates; (b) inter-frame motion trajectory descriptions; (c) changes in spatial relationships between objects. This information is concatenated into the prompt as structured text.
- **Design Motivation**: Transforms implicit visual dynamic information into a textual format that LLMs are adept at processing, thereby reducing the difficulty of cross-modal reasoning.

### Evaluation Protocol
- Five categories of MLLMs are evaluated: general-purpose (GPT-4o, Gemini), spatial-aware (SpatialVLM), region-level (RegionGPT), and others.
- Dual-dimension evaluation: VQA accuracy + Grounding IoU.
- Reasoning–grounding consistency is examined: cases where VQA answers are correct but grounding is incorrect are flagged as "inconsistent."

## Key Experimental Results

### Main Results: Comparison of MLLM Dynamic Understanding

| Model | VQA Acc (%) | Grounding IoU (%) | Consistency (%) |
|-------|-------------|-------------------|-----------------|
| GPT-4o | 62.3 | 28.5 | 31.2 |
| Gemini-2.0 | 58.7 | 25.1 | 28.9 |
| LLaVA-Video | 51.2 | 32.4 | 35.6 |
| + Mask-Guided Fusion | 55.8 | 41.7 | 43.2 |
| + ST-TCM | 59.1 | 38.5 | 44.8 |
| + MGF + ST-TCM | **61.3** | **44.2** | **48.5** |

### Ablation Study: Prompting Strategy Comparison

| Prompting Strategy | VQA Acc (%) | Grounding IoU (%) |
|--------------------|-------------|-------------------|
| Direct | 51.2 | 32.4 |
| Chain-of-Thought | 52.8 | 33.1 |
| Caption-based Hints | 53.1 | 34.0 |
| **Mask-Guided Fusion** | **55.8** | **41.7** |
| **ST-TCM** | **59.1** | **38.5** |

### Key Findings
- **Existing MLLMs cannot excel at both reasoning and grounding simultaneously** — GPT-4o achieves relatively high VQA accuracy (62.3%) but an extremely low grounding IoU (28.5%), revealing a severe inconsistency between what the model "says" and what it "points to."
- **Conventional prompting is nearly ineffective** — CoT and caption hints yield improvements of less than 2%, indicating that dynamic understanding cannot be resolved by simply adding reasoning steps.
- **Structured integration methods are effective** — MGF and ST-TCM inject dynamic information through visual and textual channels respectively, yielding significant gains.
- **Spatial-aware models do not guarantee dynamic understanding** — SpatialVLM performs strongly on static spatial reasoning but exhibits unstable performance in dynamic scenes.

## Highlights & Insights
- **A profound framing of "Thinking in Dynamics"** — examining MLLMs from the perspective of the physical 4D world, transcending the conventional video QA paradigm.
- **Reasoning–grounding consistency evaluation** — the first systematic quantification of the gap between MLLMs' "understanding" and "localization" capabilities.
- **Structured information injection substantially outperforms prompting** — suggesting that the bottleneck in dynamic understanding lies in "information acquisition" rather than "reasoning capability."
- **Multi-source construction strategy for Dyn-Bench** — combining 2D video and 4D point cloud data ensures the authenticity and diversity of dynamic scenes.

## Limitations & Future Work
- Dyn-Bench is relatively small in scale (1k videos), which may be insufficient for training specialized models.
- ST-TCM relies on pre-extracted object position and trajectory information, requiring external trackers/detectors.
- Closed-loop scenarios (e.g., dynamic reasoning in robotic manipulation) are not evaluated.
- Grounding evaluation relies solely on bounding box IoU, without considering finer-grained pixel-level or 3D spatial localization.

## Related Work & Insights
- **vs. VideoChat/Video-LLaMA**: These works focus on video dialogue but do not evaluate structured spatio-temporal reasoning.
- **vs. EgoPlan-Bench**: EgoPlan targets first-person perspective planning, whereas Dyn-Bench more broadly covers third-person dynamic scenes.
- **Insights**: The MGF and ST-TCM approaches can be generalized to autonomous driving scene understanding — textualizing sensor information as auxiliary input to MLLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first systematic evaluation of MLLMs from the perspective of physical 4D world dynamics; both the framing and methodology are pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model and multi-strategy comparisons are comprehensive, though the dataset scale is relatively limited.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is insightful and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Dyn-Bench fills a critical gap in dynamic evaluation of MLLMs; the reasoning–grounding consistency analysis is highly informative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation](mos_mixture_of_states_multimodal_generation.md)
- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching VLA Models](flowhijack_dynamics_aware_backdoor_attack_on_flow_matching_vla_models.md)
- [\[ICML 2026\] Vision Language Models Cannot Reason Physical Transformations](../../ICML2026/multimodal_vlm/vision_language_models_cannot_reason_about_physical_transformation.md)
- [\[ICML 2026\] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](../../ICML2026/multimodal_vlm/dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](aif_adaptive_information_flow_vlm.md)

</div>

<!-- RELATED:END -->
