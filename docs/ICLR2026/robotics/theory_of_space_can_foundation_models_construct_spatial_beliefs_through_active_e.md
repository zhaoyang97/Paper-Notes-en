---
title: >-
  [Paper Note] Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?
description: >-
  [ICLR 2026][Robotics][Theory of Space] This paper proposes the Theory of Space framework, which systematically evaluates the ability of foundation models to construct and revise spatial beliefs through active exploration…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Theory of Space"
  - "active exploration"
  - "spatial belief"
  - "cognitive map"
  - "partial observability"
  - "belief inertia"
date: 2026-05-08
content_hash: f7872eaec50f257e
---

# Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?

**Conference**: ICLR 2026
**arXiv**: [2602.07055](https://arxiv.org/abs/2602.07055)  
**Code**: [GitHub](https://github.com/mll-lab-nu/Theory-of-Space)  
**Area**: Spatial Intelligence / Embodied AI
**Keywords**: Theory of Space, active exploration, spatial belief, cognitive map, partial observability, belief inertia

## TL;DR

This paper proposes the Theory of Space framework, which systematically evaluates the ability of foundation models to construct and revise spatial beliefs through active exploration, cognitive map probing, and a False Belief paradigm across both text-based and visual environments. The study reveals critical failure modes in current state-of-the-art models, including active-passive performance gaps, inefficient exploration strategies, and deficient belief revision.

## Background & Motivation

**Background**: Multimodal foundation models (e.g., GPT-5.2, Gemini-3 Pro) have demonstrated strong performance on passive multimodal perception and reasoning tasks. Existing benchmarks in spatial intelligence fall into two categories: (1) **passive benchmarks** (spatial reasoning given complete observations), such as various VQA and spatial relation reasoning datasets; and (2) **task-driven benchmarks** (e.g., "find the red chair"), which assess goal-completion performance. However, the core capability of spatial intelligence—**actively and autonomously acquiring information under partial observability to construct a global spatial understanding**—has not been systematically studied.

**Limitations of Prior Work**: Existing benchmarks cannot answer the key question: "Does the model truly understand space?" Passive benchmarks circumvent the challenge of information acquisition—models need not decide "what to look at next." Task-driven benchmarks couple exploration and reasoning, making it impossible to diagnose specific failure causes. More critically, there is no method to directly "open up" a model's internal spatial representation to inspect the quality of its spatial beliefs; only indirect inference from final task performance is possible.

**Key Challenge**: Cognitive science research demonstrates that **active exploration** leads to significantly better spatial understanding than passively receiving the same information. However, it remains unknown whether foundation models possess this active exploration capability—whether they can autonomously decide "where to look" under uncertainty and integrate sequential local observations into a globally consistent spatial belief.

**Goal**: (1) Define and formalize "Theory of Space" as a distinct capability dimension; (2) construct a controlled evaluation benchmark that decouples exploration ability from reasoning ability; (3) directly probe the quality of models' internal spatial beliefs rather than relying solely on task outcomes; (4) test models' ability to revise spatial beliefs in dynamic environments, analogous to the False Belief paradigm in Theory of Mind.

**Key Insight**: By analogy with Theory of Mind (which assesses an agent's ability to model others' mental states), this paper proposes Theory of Space (which assesses an agent's ability to model unseen spatial structures). The central innovation is **Belief Probing**—requiring models to externalize their internal spatial beliefs as cognitive maps at every step of exploration, thereby directly measuring the quality of the spatial model rather than inferring it in a black-box manner from task performance.

**Core Idea**: Directly evaluate foundation models' ability to actively construct and dynamically revise spatial beliefs through cognitive map probing and a False Belief paradigm.

## Method

### Overall Architecture

Theory of Space evaluation consists of two phases: an **Exploration Phase** (the agent actively explores a partially observable environment through actions such as moving, rotating, and observing to construct spatial beliefs) and a **Reasoning Phase** (completing 9 spatial tasks based on the constructed beliefs). Environments are provided in parallel text-based and visual modalities (sharing identical layouts) to decouple perceptual failures from reasoning failures. Two evaluation settings are used: (1) active exploration—the model autonomously determines its exploration strategy; (2) passive understanding—the model receives standardized exploration logs generated by a scripted agent. Comparing active and passive performance quantifies the deficit in exploration capability.

### Key Designs

1. **Text-Visual Parallel Environments**:

    - **Function**: Provide a controlled experimental environment supporting the decoupling of perceptual errors from spatial reasoning errors.
    - **Mechanism**: Procedurally generated multi-room indoor scenes on an $N \times M$ grid, each containing $n$ objects (2D coordinates + orientation). The agent starts from a random position and executes four actions: Goto (move to a visible object), Rotate (90°/180°/270° rotation), Observe (perceive objects within a 90° field of view), and Query (retrieve absolute coordinates of visible objects). The **text world** provides symbolic direction-distance discretized descriptions (e.g., "chair is front-left and near"), isolating pure spatial reasoning ability. The **visual world** uses ThreeDWorld+Objaverse to render egocentric RGB images, requiring simultaneous visual perception and spatial reasoning. Both share the same layout, enabling direct comparison.
    - **Design Motivation**: Single-modality evaluation cannot distinguish between "failure to interpret images" and "poor spatial reasoning." The text world eliminates perceptual noise to expose pure reasoning bottlenecks, while the visual world tests end-to-end capability. Performance differences between the two precisely quantify the impact of perception on spatial understanding.

2. **Cognitive Map Probing**:

    - **Function**: Directly inspect the quality of models' internal spatial beliefs rather than inferring them solely from task performance.
    - **Mechanism**: At every step of exploration, the model is required to output its current cognitive map—an estimate of the positions and orientations of all objects. Four dimensions are assessed: (D2-1) **Correctness**—a composite score of position, direction, and orientation accuracy relative to ground truth; (D2-2) **Perception Quality**—accuracy of converting single-step observations into correct information; (D2-3) **Self-Tracking**—accuracy of the agent's model of its own position and orientation; (D2-4) **Stability and Local-Global Consistency**—whether known information degrades over time and whether the local relational graph contradicts the global map. Additionally, (D3) **Uncertainty Modeling** is evaluated—given sets of observed and unobserved locations, whether the model correctly identifies unobserved regions (F1 score).
    - **Design Motivation**: Behavioral success (e.g., finding a chair) does not directly indicate the quality of internal representations—a model may succeed by chance or correct recall while its overall spatial belief is highly inaccurate. Cognitive map probing shifts evaluation from "can the model answer correctly" to "how accurate is the internal model," enabling fine-grained fault diagnosis.

3. **False Belief Paradigm and Belief Inertia**:

    - **Function**: Evaluate the model's ability to revise spatial beliefs after dynamic changes in the environment.
    - **Mechanism**: Inspired by the classic False Belief test in Theory of Mind. After the agent completes initial exploration, the environment is covertly modified (objects are moved or rotated). The agent is then required to re-explore and identify the changes. Evaluation metrics include the F1 score for change detection. A key phenomenon is identified—**Belief Inertia**: models persist in their prior spatial beliefs even after directly observing a new configuration, failing to override outdated beliefs with new sensory evidence. This problem is particularly severe in visual models.
    - **Design Motivation**: The real world is dynamic—objects are moved and environments change. The ability to update beliefs after re-exploration is a core component of spatial intelligence. The False Belief paradigm provides a precisely controlled testing framework, and the discovery of Belief Inertia reveals a fundamental deficit in current models' spatial memory plasticity.

### Task Taxonomy

Nine spatial tasks are organized into two categories: **Route Belief**—egocentric, step-by-step reasoning tasks including direction inference, perspective-taking, and action prediction; and **Survey Belief**—allocentric, global-map reasoning tasks including global coordinate mapping, mental rotation, and location-perspective transformation. All tasks use open-ended questions rather than multiple choice to reduce the risk of knowledge leakage.

## Key Experimental Results

### Active Exploration Performance (Visual World)

| Model | Avg. Steps | Route Tasks Avg. | Survey Tasks Avg. | Overall Avg. |
|------|---------|--------------|---------------|--------|
| GPT-5.2 | 17.2 | 44.2 | 48.0 | **46.0** |
| Gemini-3 Pro | 13.6 | 52.1 | 62.8 | **57.3** |
| Claude-4.5 Sonnet | 19.6 | 24.9 | 34.2 | **29.6** |
| Qwen3-VL | 16.3 | 19.6 | 23.3 | **21.3** |
| Human | 9.8 | — | — | **96.4** |

### Active-Passive Gap (Passive vs. Active, Visual World Overall Avg.)

| Model | Passive | Active | Gap |
|------|------|------|------|
| GPT-5.2 | 57.1 | 46.0 | **-11.1** |
| Gemini-3 Pro | 60.5 | 57.3 | **-3.2** |
| Claude-4.5 Sonnet | 43.1 | 29.6 | **-13.5** |
| Qwen3-VL | 24.9 | 21.3 | **-3.6** |

### Key Findings

- **Universal Active-Passive Gap**: All models perform worse during active exploration than when passively receiving identical information. GPT-5.2 drops from 57.1 to 46.0, and Claude-4.5 Sonnet shows the largest decline (43.1 → 29.6). This indicates that exploration strategy itself is a bottleneck—models do not know "what to look at."
- **Highly Inefficient Exploration**: Rule-based agents (Scout/Strategist) achieve target coverage in approximately 9 steps, while foundation models require ≥14 steps without achieving better belief accuracy. Models exhibit high redundancy—repeatedly observing already-known regions while neglecting unexplored areas.
- **Perception as the Initial Bottleneck, Stability as the Persistent Bottleneck**: Cognitive map probing reveals that perceptual accuracy is the first barrier in the visual world (nearly 100% accurate in the text world); however, even with correct perception, global beliefs degrade over time due to **insufficient stability**—spatial information correctly acquired in earlier steps is subsequently "forgotten" or overwritten.
- **Belief Inertia Is Particularly Severe in Visual Models**: In False Belief tests, text-world models demonstrate some capacity for belief revision, but visual-world models are nearly incapable of overriding prior beliefs—even upon directly observing that an object has been displaced, the output cognitive map retains the old coordinates. This reveals a fundamental deficit in current models' spatial memory plasticity.
- **Large Text-Visual Gap**: GPT-5.2 achieves an average score of 72.0 in active exploration in the text world but only 46.0 in the visual world—a gap of 26 points. This quantifies the substantial drag that visual perception exerts on spatial understanding.

## Highlights & Insights

- **Conceptual Framework of Theory of Space**: By analogy with Theory of Mind, the paper proposes Theory of Space and defines "actively constructing spatial beliefs" as an independent capability dimension. The value of this conceptual framework extends beyond the specific experimental results—it provides a lasting paradigm for thinking about and evaluating spatial intelligence research.
- **Direct Evaluation via Belief Probing**: Rather than treating the model as a black box, the framework directly requires the model to externalize its internal beliefs at every step. This "opening the black box" evaluation methodology is transferable to the assessment of other cognitive capabilities (e.g., causal reasoning, temporal reasoning) and carries broad methodological implications.
- **Discovery of Belief Inertia**: This is an important empirical finding—models do not merely "fail to retain" new information; rather, **prior beliefs actively resist new evidence**. This parallels confirmation bias in cognitive science and reveals a fundamental limitation in the representational update mechanisms of current foundation models.
- **Text-Visual Parallel Design**: By running text-based and visual experiments on identical spatial layouts, the framework precisely separates "failure to perceive" from "failure to reason," providing a clear roadmap for diagnosis and improvement.

## Limitations & Future Work

- **Simplified Grid-World Environment**: The experimental environment consists of multi-room layouts on a 2D grid, with objects represented by discrete coordinates and cardinal directions. This is substantially simplified relative to real 3D environments, and the transferability of conclusions requires further validation.
- **Single-Agent Scenarios Only**: Multi-agent collaborative exploration and spatial belief sharing/alignment—core challenges in multi-robot systems—are not considered.
- **Poor Performance of Open-Source Models**: GLM-4.6V and Qwen3-VL score only 14–21 points in the visual world (compared to 96 for humans), suggesting that results may primarily reflect insufficient basic visual-spatial perception rather than bottlenecks in higher-level spatial reasoning.
- **Format Dependency in Cognitive Map Probing**: The quality of cognitive map outputs may be affected by output format (JSON/coordinate table), and some models may be underestimated due to poor format compliance.

## Related Work & Insights

- **vs. Passive Spatial Reasoning Benchmarks (SpartQA, etc.)**: Passive benchmarks completely bypass the information acquisition process. The core contribution of Theory of Space lies in incorporating active exploration—"deciding what to look at"—into the evaluation, revealing that passive reasoning ability cannot predict active exploration performance.
- **vs. Task-Driven Navigation Benchmarks (ALFRED, ObjectNav, etc.)**: These benchmarks assess task completion rates for specific objectives and cannot diagnose the quality of internal spatial representations. Theory of Space shifts evaluation from "doing the right thing" to "thinking correctly" through cognitive map probing.
- **vs. Theory of Mind Evaluation**: ToM tests modeling of others' mental states; ToS tests modeling of physical spatial structures. Both emphasize "reasoning about hidden states that are not directly observable," but ToS focuses on the physical world rather than social cognition.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Highly pioneering conceptual framework; both cognitive map probing and the discovery of Belief Inertia are original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six SOTA models, text + visual dual environments, 9 task types, 5 evaluation dimensions, human baseline
- Writing Quality: ⭐⭐⭐⭐ Framework is clear but the paper is lengthy; some sections could be condensed
- Value: ⭐⭐⭐⭐⭐ Defines a new evaluation paradigm for spatial intelligence research and identifies critical bottlenecks in foundation models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)
- [\[ICLR 2026\] Building Spatial World Models from Sparse Transitional Episodic Memories](building_spatial_world_models_from_sparse_transitional_episodic_memories.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/robotics/sapave_active_perception_manipulation_vla_roboti.md)
- [\[ICLR 2026\] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)

</div>

<!-- RELATED:END -->
