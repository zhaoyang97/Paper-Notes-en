---
title: >-
  [Paper Note] BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis
description: >-
  [ICML 2026][Robotics][Embodied evaluation] BEAR decomposes embodied tasks into 14 atomic skills and constructs 4,469 interleaved image-video-text VQA pairs. By performing horizontal and vertical skill-level diagnosis on…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Embodied evaluation"
  - "MLLM diagnosis"
  - "Skill-level assessment"
  - "Tool-augmented Agent"
  - "Long-horizon tasks"
date: 2026-05-08
content_hash: c6bbeda07a990514
---

# BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis

**Conference**: ICML 2026  
**arXiv**: [2510.08759](https://arxiv.org/abs/2510.08759)  
**Code**: https://bear-official66.github.io/ (Available, project page + evaluation data)  
**Area**: Multimodal VLM / Embodied AI / Evaluation Benchmarks  
**Keywords**: Embodied evaluation, MLLM diagnosis, Skill-level assessment, Tool-augmented Agent, Long-horizon tasks

## TL;DR
BEAR decomposes embodied tasks into 14 atomic skills and constructs 4,469 interleaved image-video-text VQA pairs. By performing horizontal and vertical skill-level diagnosis on 20 MLLMs, it identifies perception (rather than reasoning) as the primary bottleneck. Based on these findings, BEAR-Agent is developed using external visual/spatial tools such as GroundingDINO, 3D scene graphs, and trajectory visualization. This approach yields a 17.5% relative improvement for GPT-5 on the benchmark and a 20.17% improvement in real robot grasping.

## Background & Motivation

**Background**: MLLMs are increasingly deployed as embodied agents in both simulated and real robotic environments, outputting actions from perception to planning. Most existing embodied evaluation benchmarks (e.g., EmbodiedBench, Embodied-Agent-Interface, ALFRED) use "overall task success rate" as the sole metric, focusing either on single sub-domains (pointing, spatial) or high-level modules (goal interpretation / subgoal decomposition).

**Limitations of Prior Work**: Task-level evaluations conflate "perceptual failures" and "planning failures" within a binary success label. Whether a model succeeds or fails, the underlying "where and why" remains unclear, offering little actionable insight for model improvement. While module-based partitioning provides stage-wise success rates, the boundaries remain too coarse to pinpoint specific "perception vs. reasoning" atomic capabilities.

**Key Challenge**: A natural misalignment exists between the evaluation granularity (task-level) and the granularity required for improvement (capability-level). Attributing failures to underlying atomic capabilities is essential to determine whether researchers should focus on perception or reasoning.

**Goal**: (1) Evaluate at the atomic skill level; (2) Provide interpretable attribution of failures to specific capabilities; (3) Directly translate diagnostic conclusions into practical improvement strategies.

**Key Insight**: The authors summarize five "main tracks" of embodied task execution from cognitive science and domestic activity trajectories in BEHAVIOR-1K/ALFRED: task planning, spatial reasoning, bounding box coarse localization, pointing fine interaction, and trajectory movement—connected by long-horizon sequences. Each step corresponds to an "atomic skill," covering the cognitive chain of human task execution while allowing for automatic verification in simulation episodes.

**Core Idea**: Reframe "embodied evaluation" from "task success rate" to "14 atomic skills $\times$ horizontal & vertical diagnosis." Diagnostic conclusions are then used to reverse-engineer improvements, such as augmenting MLLMs with external visual/spatial tools. These improvements are verified back on the benchmark at the skill level, creating a "diagnosis–improvement–re-diagnosis" closed loop.

## Method

### Overall Architecture
BEAR consists of three components: (1) An evaluation dataset of 14 skills across 6 categories, featuring 4,469 interleaved image-video-text VQA pairs from 13 data sources; (2) A hierarchical diagnostic framework including "horizontal long-horizon bottleneck identification + vertical independent skill assessment + cross-skill failure attribution"; (3) BEAR-Agent, which utilizes the MLLM as a dialogue controller to call a set of Python tools, feeding extra visual/spatial cues back into the prompt.

### Key Designs

1.  **Skill Taxonomy and Data Curation**:
    *   Function: Decomposes any embodied task into 14 atomic skills within 5 core categories plus long-horizon capabilities. These cover pointing (GEN/SPA/PRT granularities), bounding box (GEN/SPA/PRT), trajectory (gripper/hand/object), task planning (TPR/NAP), and spatial reasoning (LOC/PTH/DIR).
    *   Mechanism: Data is curated from 13 sources (BEHAVIOR-1K, ALFRED, Open-X, OpenImages, AI2-THOR, etc.) to target specific categories. For example, pointing uses OpenImages, trajectory uses Open-X, and long-horizon uses ALFRED/AI2-THOR (35 episodes manually segmented). Automated generation is followed by GPT-o3 semantic filtering and at least three rounds of review by 10 human annotators, retaining 2,563 multiple-choice and 1,906 open-ended questions.
    *   Design Motivation: Single data sources often cover only one capability and risk data leakage from model pre-training. Multi-source and multi-modal (image/video/interleaved) data forces models to follow real "perception-reasoning" paths rather than overfitting to a specific style.

2.  **Hierarchical Skill-level Diagnostic Framework**:
    *   Function: Provides explanations for which atomic capabilities repeatedly fail, augmenting simple success rates.
    *   Mechanism: Horizontally, the long-horizon category unfolds 35 episodes into chains of the 5 core skills (e.g., "put apple in sink" = planning → search → path planning → relative direction → visual perception → trajectory placement). Bottleneck skills are identified by hit rates for each step. Vertically, independent skill tests pinpoint failures to specific atomic abilities. Finally, "cross-skill failure attribution" analyzes which capabilities fail consistently across different contexts to expose shared bottlenecks.
    *   Design Motivation: Horizontal analysis identifies that spatial reasoning fails in long-horizon tasks; vertical analysis specifies whether path planning or relative direction is at fault; cross-skill attribution asks if all depth-related problems are problematic.

3.  **BEAR-Agent: Diagnosis-driven Tool-augmented Multimodal Agent**:
    *   Function: Translates the diagnostic conclusion that "perception is a bottleneck and spatial-temporal modeling is unstable" into actionable tools, allowing the MLLM to actively request 3D scene graphs, bboxes, or trajectory visualizations.
    *   Mechanism: Tools are implemented as modular Python functions: GroundingDINO for object localization, a 3D scene graph module for spatial relationships, and a trajectory visualization module to overlay actions on images. The MLLM calls tools based on uncertainty during multi-turn dialogue, and results are fed back into the next prompt. This design is plug-and-play for any conversational MLLM without altering weights.
    *   Design Motivation: Diagnosis showed that CoT and test-time compute scaling gains were generally $<10\%$, indicating the problem is "insufficient vision" rather than "insufficient thinking." Directly providing external visual/spatial evidence addresses the core issue.

### Loss & Training
BEAR is an evaluation and agent framework; it does not involve retraining models. BEAR-Agent requires no fine-tuning of the backbone and relies on in-context tool calling. The evaluation protocol follows VLMEvalKit defaults, using Merged (multiple frames in one image) or Sequential (frame-by-frame) inputs. Pointing/spatial/planning/long-horizon use success rate, bounding box uses IoU. Long-horizon success requires all steps in an episode to be correct.

## Key Experimental Results

### Main Results
20 representative MLLMs (including GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet, InternVL3, Qwen2.5-VL, etc.) were evaluated. A BEAR-mini set (40 questions per skill) was used for human baseline testing.

| Model | Type | Pointing-GEN | Spatial-LOC | Long-horizon | Avg Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Human | Human | 95.50 | 94.50 | 92.50 | 89.40 |
| GPT-5 | Closed | 70.00 | – | – | 52.2 |
| Gemini-2.5-Pro | Closed | 55.00 | – | – | – |
| Claude-4-Sonnet | Closed | 39.12 | 46.25 | – | – |
| InternVL3-8B | Open | 52.65 | 50.16 | 8.57 | 33.32 |
| Qwen2.5-VL-32B | Open | 27.35 | 47.23 | 20.00 | 28.33 |
| Random | Baseline | – | – | 25 | – |

Closed-source models averaged 39.2%, 13.4 points higher than open-source models. The strongest model, GPT-5, reached 52.2%, still 37 points behind the human level (89.40%).

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| GPT-5 baseline | 52.2% | Direct evaluation |
| GPT-5 + CoT prompt | Gain <10% | Chain-of-Thought typically provides only slight improvements |
| GPT-5 + test-time scaling | Gain <10% | Increasing compute at inference time shows minimal benefit |
| GPT-5 + BEAR-Agent | 61.3% (+9.12 abs, +17.5% rel) | Tool augmentation is the only path providing significant gains |
| Real robot grasping + BEAR-Agent | +20.17% | Verified on Cobot Magic platform for tabletop manipulation |

### Key Findings
*   Perception (pointing, bbox, trajectory) is the root cause of the vast majority of failures. Even "reasoning-heavy" tasks like task planning and spatial reasoning often fail at the perceptual layer.
*   Spatial-temporal modeling issues recur in cross-skill attribution, particularly in trajectory reasoning (hand/gripper/object), which CoT and compute scaling cannot resolve.
*   The gains from BEAR-Agent come from "supplementing cues" rather than "supplementing reasoning," closing the loop with the diagnostic conclusions.

## Highlights & Insights
*   By using "atomic skills $\times$ three-layer diagnosis," embodied evaluation is upgraded from 0/1 binary labels to an interpretable capability radar, providing an actionable paradigm for evaluation, attribution, and improvement.
*   The diagnostic conclusion ("perception is the bottleneck, CoT cannot save it") is counter-intuitive but supported by data, refuting the common assumption that more CoT will lead to success and suggesting that future work should focus on visual/spatial tools rather than reasoning compute.
*   The training-free "BEAR-Agent via tool assembly" path can be replicated for any evaluation of capability bottlenecks: identify weaknesses through fine-grained evaluation, deploy targeted tools, and quantify the improvement.

## Limitations & Future Work
*   While the 14 skills are broad, they represent the authors' summarized cognitive chain and lack support for dual-arm collaboration, force/tactile feedback, or social navigation. Real-robot verification remains limited to tabletop manipulation.
*   The toolset (GroundingDINO, 3D scene graphs) is relatively static and may not suffice for dynamic environments or online perception. Tool dispatching relies on the MLLM's internal judgment without an explicit tool planning loss.
*   Despite three rounds of human review, embodied video questions can be ambiguous; OOD generalization and prompt bias require further auditing.

## Related Work & Insights
*   **vs EmbodiedBench / Embodied-Agent-Interface**: These benchmarks perform task-level or module-level evaluation. BEAR advances the granularity to the atomic skill level and adds cross-skill failure attribution.
*   **vs Single-domain Benchmarks (pointing/spatial)**: These target one capability where models can "game" scores. BEAR uses 14 skills and long-horizon sequences to expose all weaknesses in a unified framework.
*   **vs OpenVLA / RT-2**: These focus on "how the model acts," while BEAR focuses on "where the model fails to understand the world." BEAR's diagnostic conclusions can directly guide data augmentation for VLA-style models.

## Rating
*   Novelty: ⭐⭐⭐⭐ Combines atomic skills, three-layer diagnosis, and tool-augmented agents into a closed loop.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20 models $\times$ 14 skills $\times$ simulation and real robot verification.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure with logical diagnostic layers.
*   Value: ⭐⭐⭐⭐⭐ Provides the first actionable diagnostic benchmark and improvement paradigm for embodied MLLMs.

## Related Papers

- [\[ICML 2026\] Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)
- [\[ICML 2026\] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)
- [\[CVPR 2026\] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](../../CVPR2026/robotics/hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [\[ICML 2026\] Position: Good Embodied Reward Models Need Bad Behavior Data](position_good_embodied_reward_models_need_bad_behavior_data.md)
- [\[ICML 2026\] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)
- [\[ICML 2026\] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)
- [\[ICML 2026\] Position: Good Embodied Reward Models Need Bad Behavior Data](position_good_embodied_reward_models_need_bad_behavior_data.md)
- [\[ICML 2026\] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)

</div>

<!-- RELATED:END -->
