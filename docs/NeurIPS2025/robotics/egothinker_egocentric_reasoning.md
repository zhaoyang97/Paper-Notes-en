---
title: >-
  [Paper Note] EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT
description: >-
  [NeurIPS 2025][Robotics][Egocentric video] This paper proposes EgoThinker, which constructs the large-scale egocentric video reasoning dataset EgoRe-5M (with causal CoT annotations and hand-object grounding labels) and adopts a two-stage training paradigm (SFT + GRPO reinforcement fine-tuning) to endow MLLMs with robust egocentric reasoning, hand-object grounding, and temporal localization capabilities, achieving state-of-the-art performance across multiple egocentric benchmarks.
tags:
  - NeurIPS 2025
  - Robotics
  - Egocentric video
  - chain-of-thought reasoning
  - hand-object interaction
  - reinforcement fine-tuning
  - spatio-temporal grounding
date: 2026-05-08
content_hash: 7702d59aee72b7f1
---

# EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT

**Conference**: NeurIPS 2025
**arXiv**: [2510.23569](https://arxiv.org/abs/2510.23569)
**Code**: [GitHub](https://github.com/InternRobotics/EgoThinker)
**Area**: Robotics
**Keywords**: Egocentric video, chain-of-thought reasoning, hand-object interaction, reinforcement fine-tuning, spatio-temporal grounding

## TL;DR
This paper proposes EgoThinker, which constructs the large-scale egocentric video reasoning dataset EgoRe-5M (with causal CoT annotations and hand-object grounding labels) and adopts a two-stage training paradigm (SFT + GRPO reinforcement fine-tuning) to endow MLLMs with robust egocentric reasoning, hand-object grounding, and temporal localization capabilities, achieving state-of-the-art performance across multiple egocentric benchmarks.

## Background & Motivation
1. **Background**: MLLMs excel at third-person visual reasoning but lack embodied cognitive understanding from an egocentric perspective.
2. **Limitations of Prior Work**: Existing egocentric datasets (e.g., Ego4D) lack explicit reasoning chains, temporal span annotations, and fine-grained hand-object grounding data.
3. **Key Challenge**: Egocentric reasoning requires inferring the unseen camera-wearer's intent and actions, rather than merely recognizing visible events.
4. **Goal**: Equip MLLMs with comprehensive capabilities for egocentric reasoning, precise hand-object grounding, and long-range temporal understanding.
5. **Key Insight**: Construct large-scale causal CoT-annotated data and adopt a two-stage training paradigm (SFT to establish foundations + RFT to reinforce grounding).
6. **Core Idea**: Mine egocentric data at scale from web videos such as HowTo100M, construct causal reasoning QA pairs, and apply GRPO to reinforce spatio-temporal grounding.

## Method

### Overall Architecture
EgoRe-5M dataset → SFT stage (causal CoT reasoning capability) → RFT stage (GRPO-based reinforcement of hand-object and temporal grounding).

### Key Designs
1. **EgoRe-5M Dataset**: Constructs 5 million QA pairs from 13M egocentric video clips, including causal CoT annotations over multi-minute segments and dense hand-object grounding labels. Egocentric videos are mined from HowTo100M (30M initial clips) via a multi-stage filtering pipeline, leveraging temporally aligned annotations from HTM-AA and Howto-Interlink7M.
2. **Two-Stage Training Paradigm**:
   - SFT stage: Establishes foundational egocentric understanding and reasoning on EgoRe-5M by learning from causal CoT annotations.
   - RFT stage: Applies GRPO on spatio-temporal grounding data to reinforce precise localization, using IoU and bounding box matching as rewards.
3. **Spatio-Temporal CoT Annotations**: Annotations encode complete causal chains (why an action is performed → how it is executed → what comes next), enabling the model to simulate human egocentric causal reasoning and planning.
4. **Hand-Object Interaction Data**: Dedicated dense hand-object interaction grounding data is constructed, annotating hand positions, grasped objects, and interaction types.

### Loss & Training
- SFT: Standard cross-entropy loss
- RFT: GRPO with IoU/bounding box matching rewards
- Backbone: InternVL series vision-language models
- Data diversity: Covers temporal spans ranging from a few seconds to several minutes

## Key Experimental Results

| Benchmark | EgoThinker | Prev. SOTA | Gain |
|-----------|-----------|------------|------|
| EgoSchema | Significant improvement | — | SOTA |
| Ego4D NLQ | SOTA | — | +Significant |
| Multiple egocentric QA | SOTA | — | — |

### Key Findings
- Causal CoT annotations are critical for complex egocentric reasoning.
- GRPO reinforcement training significantly improves spatio-temporal grounding precision.
- Mining egocentric data from web videos is a scalable and effective strategy.

### EgoRe-5M Dataset Composition

| Source | Initial Clips | After Filtering | QA Pairs |
|--------|--------------|-----------------|----------|
| HowTo100M | 30M | ~6M | ~3M |
| HTM-AA | 2M | ~1.5M | ~1M |
| Howto-Interlink7M | 7M | ~3M | ~1M |
| Total | 39M | ~10.5M | ~5M |

### Ablation Study

| Configuration | EgoSchema | Ego4D NLQ | Hand-Object Grounding |
|---------------|----------|-----------|-----------------------|
| SFT only | Good | Moderate | Poor |
| SFT + RFT | **SOTA** | **SOTA** | **Best** |
| RFT only (w/o SFT) | Poor | Poor | Moderate |

## Highlights & Insights
- The first egocentric MLLM capable of both reasoning and precise hand-object understanding simultaneously.
- The EgoRe-5M data construction pipeline is reusable for other egocentric tasks.
- The two-stage training paradigm (SFT → RFT) is validated as effective in the egocentric domain.

## Limitations & Future Work
- The data mining pipeline may introduce non-egocentric video noise, and filtering quality directly affects downstream performance.
- Evaluation is limited to existing egocentric benchmarks; generalization to real wearable device scenarios (e.g., AR glasses) remains to be verified.
- Integration with robotic manipulation tasks—an important application of egocentric understanding—has not been explored.
- CoT annotations rely on LLM generation, and quality may vary, particularly for complex causal chains.
- Hand-object grounding accuracy may degrade under severe occlusion.
- Inference latency and efficiency for real-time deployment have not been investigated, though wearable devices impose strict latency requirements.
- The GRPO reward design (IoU/bounding box matching) may not cover all grounding scenarios sufficiently.
- HowTo100M primarily consists of instructional videos, potentially under-representing everyday life scenarios.

## Related Work & Insights
- **vs. EgoVLP**: EgoVLP performs vision-language pre-training but lacks causal reasoning chains.
- **vs. VideoChat-R1**: VideoChat-R1 applies general-purpose video RL fine-tuning, whereas EgoThinker targets egocentric scenarios specifically.
- **vs. Ego-Plan-Bench**: Evaluates planning capability but does not optimize the model itself.
- **vs. InternVL**: A general-purpose VLM that lacks embodied understanding from an egocentric perspective.

### Additional Discussion
- The core innovation lies in transforming the problem from a single dimension to a multi-dimensional analysis framework, providing a more comprehensive understanding perspective.
- The experimental design covers diverse scenarios and baselines, with statistically significant results.
- The modular design of the method facilitates extension to related tasks and new datasets.
- Open-sourcing the code and data provides significant value for community reproduction and follow-up research.
- Compared to concurrent works, this paper demonstrates advantages in the depth of problem formulation and the comprehensiveness of experimental analysis.
- The paper follows a clear logical structure, forming a complete loop from problem definition to method design to experimental validation.
- The computational overhead of the method is reasonable, supporting practical deployability.
- Future work may consider fusion with additional modalities (e.g., audio, 3D point clouds).
- Validating the scalability of the method on larger data and models is an important future direction.
- Combining the method with end-to-end reinforcement learning optimization is a worthwhile direction to explore.
- Cross-domain transfer is worth investigating, as the generality of the method requires further validation.
- Lightweight variants of the method for edge computing and mobile deployment scenarios merit further study.

## Rating
- Novelty: ⭐⭐⭐⭐ Large-scale egocentric causal reasoning dataset construction is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated with SOTA results across multiple benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation with thorough description of data construction.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for wearable assistants and embodied AI.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[NeurIPS 2025\] GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation](gui-rise_structured_reasoning_and_history_summarization_for_gui_navigation.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)

<!-- RELATED:END -->
