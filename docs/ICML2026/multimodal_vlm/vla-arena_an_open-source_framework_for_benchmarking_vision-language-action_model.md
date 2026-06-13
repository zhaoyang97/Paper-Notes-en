---
title: >-
  [Paper Note] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models
description: >-
  [ICML 2026][Multimodal VLM][Vision-Language-Action Models] VLA-Arena proposes a structured VLA benchmark that systematically quantifies difficulty through three orthogonal dimensions: task structure, language commands…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Vision-Language-Action Models"
  - "Benchmark Evaluation"
  - "Robotic Manipulation"
  - "Generalization Performance"
  - "Safety Constraints"
date: 2026-05-08
content_hash: b3cfba08e70281d8
---

# VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models

**Conference**: ICML 2026  
**arXiv**: [2512.22539](https://arxiv.org/abs/2512.22539)  
**Code**: https://github.com/VLA-Arena/VLA-Arena  
**Area**: Robotics / Vision-Language-Action Models / Agent  
**Keywords**: Vision-Language-Action Models, Benchmark Evaluation, Robotic Manipulation, Generalization Performance, Safety Constraints

## TL;DR
VLA-Arena proposes a structured VLA benchmark that systematically quantifies difficulty through three orthogonal dimensions: task structure, language commands, and visual observations. Using 170 tasks, it reveals critical deficiencies in the generalization, visual perception, and safety of existing VLA models.

## Background & Motivation

**Background**: VLA models are evolving rapidly—from RT-1 and RT-2 to the recent $\pi_0$ and UniVLA, demonstrating cross-embodiment, cross-scene, and long-horizon manipulation capabilities. However, their capability boundaries and failure modes lack quantitative characterization.

**Limitations of Prior Work**: Existing robotics benchmarks (LIBERO, VLABench, RoboCasa) suffer from three major issues: (1) Task designs are oversimplified with single-level complexity definitions; (2) They focus either on noise robustness or task generalization, making it difficult to understand how models perform under multi-dimensional concurrent challenges; (3) Safety constraints are ignored, assuming operation in ideal environments.

**Key Challenge**: Does high performance on in-distribution tasks (L0) truly reflect a model's generalization capability? Does the model operate via robust multimodal understanding or fragile pattern matching?

**Goal**: To design a benchmark capable of systematically quantifying VLA capability boundaries, distinguishing true language-vision-action understanding from superficial pattern memorization, and assessing safety.

**Key Insight**: Control task difficulty simultaneously through three orthogonal dimensions—task structural complexity, semantic variation of language instructions, and systemic perturbations of visual observations.

**Core Idea**: Construct a "Three-Axis Difficulty Space" benchmark system. Through structured, controllable, and repeatable task design, VLA evaluation is upgraded from binary "pass/fail" judgments to fine-grained diagnostics of "where the capability frontier lies."

## Method

### Overall Architecture
Three-dimensional task design—(1) **Task Structure Dimension (T-axis)**: 170 tasks divided into 11 suites organized by four core dimensions (safety/distractors/generalization/long-horizon) + 3 difficulty levels (L0/L1/L2); (2) **Language Command Dimension (W-axis)**: W0-W4 levels achieved via WordNet semantic similarity word replacement; (3) **Visual Observation Dimension (V-axis)**: V0-V4 levels through cumulative visual perturbations (lighting → color → viewpoint → noise).

### Key Designs

1.  **Constrained BDDL (CBDDL)**:
    *   **Function**: Extends standard BDDL to natively support dynamic entities, perturbations, and safety predicates.
    *   **Mechanism**: Precisely defines constraints using 10 types of safety predicates (collision, torque limits, object falls, etc.). Introduces **Cumulative Cost (CC)**: $CC(\tau) = \sum_{t=0}^{L-1} c^{inst}(s_t, a_t) + \alpha \cdot c^{term}(s_L)$, where $\alpha=10$ weights end-state hazards.
    *   **Design Motivation**: Distinguishes "task completion with safety violations" from "truly safe success," which is critical for real-world deployment.

2.  **Principled Language Perturbation (WordNet-based)**:
    *   **Function**: Systematically rewrites language instructions by replacing semantically similar words to diagnose model understanding versus rote memorization.
    *   **Mechanism**: Identifies instruction semantic slots (action verbs, target objects, locations) and uses WordNet to find synonyms with a distance of 1 for replacement. Difficulty levels correspond to the number of replaced slots from W0 (original) to W4 (all 4 slots replaced).
    *   **Design Motivation**: Distinguishes true language understanding from surface-level keyword matching. Models performing well at W0 but collapsing at W4 indicate reliance on pattern matching.

3.  **Cumulative Hierarchical Visual Perturbations (V0-V4)**:
    *   **Function**: Layer-by-layer application of visual challenges to diagnose visual robustness.
    *   **Mechanism**: V0 Standard → V1 Lighting perturbation → V2 Object color randomization → V3 Viewpoint shift → V4 Gaussian noise. Each level accumulates on the previous one.
    *   **Design Motivation**: Addresses sim-to-real issues and neural network shortcuts; moves from mild natural variations to extreme perturbations to pinpoint failure boundaries.

## Key Experimental Results

### Main Results: Six Mainstream VLA Models

| Task Dimension | Model | L0-SR | L1-SR | L2-SR | L1+L2 Decline | Key Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Safety-StaticObstacles | $\pi_0$ | 1.0 | 0.7 | 0.3 | 70% | Even the strongest model collapses at L2 |
| Safety-StaticObstacles | OpenVLA | 0.6 | 0.6 | 0.0 | 100% | Unable to handle L2 multiple obstacles |
| Distractor | $\pi_0$ | 0.9 | 0.1 | 0.0 | 100% | Distractors significantly impact performance |
| Extrapolation-UnseenObjects | $\pi_0$ | 0.8 | 0.5 | 0.0 | 100% | Total failure on unseen objects |
| Long Horizon | $\pi_0$ | 0.9 | 0.0 | 0.0 | 100% | Incapable of composing skills |

### Ablation Study

| Task Type | Perturbation Dim | W0 | W2 | W4 | Trend | Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| StatePreservation | Language | 0.8 | 0.8 | 0.7 | Gentle | Relies on visual cues |
| StatePreservation | Visual | 0.8 | 0.7 | 0.1 | Steep drop | Collapse at V3+V4 |
| UnseenObjects | Language | 0.8 | 0.4 | 0.1 | Monotonic decr. | Requires true understanding |
| UnseenObjects | Visual | 0.8 | 0.5 | 0.0 | Steep drop | Visuals equally lethal |

### VLM vs VLA Grounding Gap

| Visual Level | Qwen3-VL-8B Localization Acc. | VLA Average Drop |
| :--- | :--- | :--- |
| V0 | 100% | - |
| V1 | 100% | 13.5% |
| V2 | 100% | 24.0% |
| V3 | 96.7% | 30.5% |
| V4 | 93.3% | 50.5% |

**Key Findings**: Reveals "catastrophic forgetting"—fine-tuning causes VLAs to abandon general visual concepts. While the VLM only drops 6.7% at V4, VLAs drop 50.5%, suggesting the fine-tuning process causes the model to relearn inversions for specific pixel distributions rather than retaining robust representations.

### Key Findings
- **Memorization outperforms generalization**: All models perform excellently at L0, but L1-L2 see universal drops of 50%-100%.
- **Rampant visual shortcuts**: $\pi_0$ performance on StatePreservation tasks drops from 0.8 to 0.2 by level V3.
- **Lack of semantic understanding**: Language perturbations lead to a monotonic decrease in UnseenObjects performance (0.8 → 0.1).
- **Sim-to-Real validation**: Performance degradation (60% → 3.3%) and safety violations (0/10 → 4/10) from L0 to L2 were largely reproduced on real Franka robots.

## Highlights & Insights
- **Power of Three-Axis Orthogonal Design**: Each dimension is independent yet orthogonal, allowing for the precise isolation of failure causes.
- **Innovation with CBDDL + CC Metrics**: Safety is directly integrated into the benchmark definition, with cumulative costs effectively penalizing instantaneous hazards and end-state failures.
- **WordNet-driven Principled Language Perturbation**: More natural and controllable compared to random rewriting or template transformations.
- **Rigorous Sim-to-Real Validation**: Verifies the transferability of simulation results on physical Franka robots.
- **Rank Reversal Phenomenon**: Different difficulty levels yielding different optimal models indicates that each level provides non-redundant insights.

## Limitations & Future Work
- VLA-Arena is based on MuJoCo simulation, with limited coverage of real-world scenarios.
- The dataset scale is insufficient to completely eliminate the data influence of fine-tuning.
- Keyword replacement is based on WordNet, with limited support for non-English languages.
- The "skill composition" in the long-horizon dimension is relatively simple (maximum of 3 sub-skills).
- Improvements: Multi-embodiment learning; dynamic environments; open-vocabulary tasks; online learning evaluation.

## Related Work & Insights
- **vs LIBERO/RoboCase**: Ours introduces a "Three-Axis Difficulty Space" and a safety dimension, significantly enhancing diagnostic capabilities.
- **vs VLABench**: VLABench emphasizes language diversity but task design is flat; VLA-Arena decomposes via orthogonal dimensions.
- **vs LIBERO-Pro/Plus**: Ours is more systematic, featuring hierarchical diagnostics not only for the visual dimension (V0-V4) but also for the language dimension (W0-W4).
- **Insight**: Benchmark design should upgrade from asking "what can it do" to "where are the boundaries"; safety should be a first-class citizen in benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Pioneers the combination of a "Three-Axis Orthogonal Difficulty Space" and "W/V Perturbation Diagnostic Probes."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  6 mainstream VLAs + 170 tasks + real Franka validation + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logic with well-designed figures and tables.
- Value: ⭐⭐⭐⭐⭐  Provides a structural impact on the VLA community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VLANeXt: A Recipe for Building Robust VLA Models](vlanext_recipes_for_building_strong_vla_models.md)
- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](../../ICCV2025/multimodal_vlm/coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](../../ACL2026/multimodal_vlm/geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)
- [\[CVPR 2026\] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](../../CVPR2026/multimodal_vlm/simpact_simulation-enabled_action_planning_using_vision-language_models.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](../../CVPR2026/multimodal_vlm/from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)

</div>

<!-- RELATED:END -->
