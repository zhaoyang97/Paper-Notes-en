---
title: >-
  [Paper Note] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes
description: >-
  [ICLR 2026][LLM Reasoning][3D reasoning] This paper proposes SceneCOT, the first framework to introduce Chain-of-Thought reasoning into 3D scene understanding. Through a four-stage reasoning pipeline (task recognition →…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "3D reasoning"
  - "chain-of-thought"
  - "grounded QA"
  - "3D-LLM"
  - "scene understanding"
date: 2026-05-08
content_hash: 5573c8b694b25a02
---

# SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes

**Conference**: ICLR 2026
**arXiv**: [2510.16714](https://arxiv.org/abs/2510.16714)
**Code**: Available (project page)
**Area**: LLM Reasoning
**Keywords**: 3D reasoning, chain-of-thought, grounded QA, 3D-LLM, scene understanding

## TL;DR
This paper proposes SceneCOT, the first framework to introduce Chain-of-Thought reasoning into 3D scene understanding. Through a four-stage reasoning pipeline (task recognition → region localization → entity grounding → grounded reasoning), intermediate reasoning steps are explicitly linked to visual grounding. SceneCOT achieves 34.7% Good Coherence on Beacon3D, surpassing the strongest baseline (20.4%) by over 70%.

## Background & Motivation

**Background**: 3D-LLMs have made progress on scene question answering, but responses often lack genuine grounding in the scene — models may produce plausible answers without actually attending to the relevant objects.

**Limitations of Prior Work**: Beacon3D evaluation reveals extremely low grounding-QA coherence (Good Coherence): LEO 1.6%, PQ3D 16.5%, Chat-Scene 19.5%. A large proportion of responses exhibit mismatches — correct grounding with wrong QA, or correct QA with wrong grounding — indicating a disconnect between the reasoning process and visual perception.

**Key Challenge**: 3D reasoning tasks are diverse and complex (counting, existence, attribute, spatial relation, navigation, etc.), each requiring different visual cues and reasoning strategies. A single end-to-end model struggles to handle all task types flexibly.

**Key Insight**: Transfer CoT reasoning from the text domain to 3D scenes, decomposing complex reasoning into interpretable steps, each explicitly linked to objects or regions in the scene.

**Core Idea**: A four-stage CoT reasoning pipeline encoded via special tokens (task → region → grounding → reasoning) that tightly couples language reasoning with 3D visual perception.

## Method

### Overall Architecture
Input: 3D scene point cloud + natural language question → Stage 1: Task type recognition `<think_type>` → Stage 2: Region localization `<think_rgn>` (filtering irrelevant objects via directional/clock-face reference frames) → Stage 3: Entity grounding `<think_grd>` + `[OBJ]` (invoking a dedicated grounding module) → Stage 4: Grounded reasoning `<think_task>` + task-specific outputs (probabilities/coordinates/image tokens) → `<think_sum>` summary → `<answer>` final response.

### Key Designs

1. **Task-Aware Routing**: Upon recognizing the task type, the framework automatically selects the appropriate reasoning path and output format — `<obj_prob>` for counting, `<obj_loc_prob>` for spatial reasoning, polar coordinates `<obj_loc_plr_prob>` for navigation, and image tokens `<highlight_obj>` for attribute reasoning.

2. **Region Localization**: Spatial space is discretized using directional cues (front/back/left/right) and a clock-face reference frame (1–12 o'clock directions, 30° increments), substantially narrowing the reasoning search space. A rule-based parser extracts directional information.

3. **Modular Expert Composition**: MLLM backbone (LLaVA-1.5 + LoRA) + fine-tuned PQ3D (3D grounding) + 2D VLM (attribute reasoning) + lightweight mask predictor. A symbolic engine handles region identification.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{\text{CoT}} + \mathcal{L}_{\text{ans}} + \mathcal{L}_{\text{ground}}$$

Training data: SceneCOT-185K (145.6K situational reasoning + 40K object reasoning); trained on 4×A100 for 5 epochs with LoRA fine-tuning.

## Key Experimental Results

### Main Results

| Method | MSQA Overall | Beacon3D Case | Beacon3D Obj. | Good Coherence |
|--------|-------------|--------------|--------------|----------------|
| GPT-4o | 52.3 | 57.1 | 20.2 | - |
| LEO | 54.8 | 43.2 | 7.8 | 1.6 |
| Chat-Scene† | 56.6 | 53.6 | 14.0 | 19.5 |
| **SceneCOT** | **55.6** | **58.9** | **23.2** | **34.7** |

### Ablation Study

| Configuration | Overall |
|---------------|---------|
| Full Model | 55.6 |
| w/o Task Type Recognition | ~45 (forced incorrect type) |
| w/o Region Localization | ~50 |
| w/o Grounding Loss | ~53 |
| Oracle (perfect grounding) | 78.1 |

### Key Findings
- **Good Coherence is the standout result**: 34.7% vs. 20.4% (SceneVerse) — the only method to achieve genuine grounding-QA alignment.
- **Counting tasks show the largest gain**: 47.9% vs. Chat-Scene† 37.4% (+10.5), enabled by counting objects via explicit grounding.
- **Oracle analysis** reveals grounding errors as the primary bottleneck — perfect grounding lifts overall performance from 55.6 to 78.1.
- Zero-shot generalization: competitive performance on SQA3D/ScanQA without fine-tuning (F1@50: 51.6/40.8).

## Highlights & Insights
- **Interpretable reasoning process**: Each stage of the four-step CoT is inspectable — whether the task type is correct, the region is accurate, and the grounding targets the right object. This was previously infeasible in 3D-LLMs.
- **Region localization as an attention mechanism**: The clock-face reference frame elegantly discretizes 3D space and substantially reduces candidate objects, analogous to regional attention in vision Transformers.

## Limitations & Future Work
- MSQA Overall does not surpass Chat-Scene† (55.6 vs. 56.6); performance on attribute tasks remains relatively weak (49.6).
- Reliance on an external grounding module (PQ3D) means grounding accuracy constitutes the performance ceiling — as confirmed by the oracle experiment.
- Training is conducted solely on ScanNet scenes; generalization to outdoor or large-scale environments remains unverified.
- The four-stage pipeline incurs non-trivial inference latency, making it unsuitable for real-time interaction.

## Related Work & Insights
- **vs. Chat-Scene**: Chat-Scene achieves marginally higher QA accuracy but only 19.5% Good Coherence — its answers are not always grounded in the correct objects.
- **vs. LEO**: LEO achieves only 1.6% GC, suggesting it answers almost entirely without grounding.
- **vs. GPT-4o**: GPT-4o performs competitively on Beacon3D (57.1) but lacks explicit 3D grounding capability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to introduce CoT into 3D scene reasoning, with a systematic and complete four-stage design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation, detailed ablations, oracle analysis, and zero-shot generalization.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the motivation for the CoT design is well-articulated.
- Value: ⭐⭐⭐⭐⭐ Defines a new paradigm for 3D reasoning; the Good Coherence metric deserves broader adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[ICLR 2026\] Continuous Chain of Thought Enables Parallel Exploration and Reasoning](continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)
- [\[CVPR 2026\] Beyond Geometry: Artistic Disparity Synthesis for Immersive 2D-to-3D](../../CVPR2026/llm_reasoning/beyond_geometry_artistic_disparity_synthesis_for_immersive_2d-to-3d.md)

</div>

<!-- RELATED:END -->
