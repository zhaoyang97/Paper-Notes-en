---
title: >-
  [Paper Note] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
description: >-
  [CVPR 2026][Robotics][robotic manipulation failure] This paper proposes ViFailback, a framework that leverages visual symbols (arrows, crosshairs, labels, etc.) to efficiently annotate real-world robotic manipulation failures. The framework constructs a dataset of 58,128 VQA pairs and trains ViFailback-8B, a VLM capable of failure diagnosis and both visual and textual corrective guidance. When integrated with a VLA, it achieves a 22.2% improvement in task success rate.
tags:
  - CVPR 2026
  - Robotics
  - robotic manipulation failure
  - visual symbols
  - VLM failure diagnosis
  - VLA recovery
  - real-world dataset
date: 2026-05-08
content_hash: 22c2c39620f71459
---

# Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols

**Conference**: CVPR 2026
**arXiv**: [2512.02787](https://arxiv.org/abs/2512.02787)
**Code**: [Project Page](https://vifailback.github.io/)
**Area**: Robotics
**Keywords**: robotic manipulation failure, visual symbols, VLM failure diagnosis, VLA recovery, real-world dataset

## TL;DR
This paper proposes ViFailback, a framework that leverages visual symbols (arrows, crosshairs, labels, etc.) to efficiently annotate real-world robotic manipulation failures. The framework constructs a dataset of 58,128 VQA pairs and trains ViFailback-8B, a VLM capable of failure diagnosis and both visual and textual corrective guidance. When integrated with a VLA, it achieves a 22.2% improvement in task success rate.

## Background & Motivation
**Background**: VLA models perform well in robotic manipulation but inevitably fail in out-of-distribution (OOD) scenarios. VLMs have been applied to task planning and reasoning, yet their capabilities in **failure diagnosis and correction** remain limited.

**Limitations of Prior Work**: (1) Existing failure datasets are primarily generated in simulation via perturbation injection, constrained by the sim-to-real gap; (2) Annotating real-world failure data is extremely time-consuming, particularly for abstract categories (task planning errors, failure causes) that require extensive textual descriptions.

**Key Challenge**: Real-world failure data is valuable but costly to annotate; simulation data is cheap but insufficiently realistic. The central question is how to efficiently leverage real-world failure data.

**Goal**: To establish a low-cost framework for real-world robotic failure diagnosis and correction, encompassing an efficient annotation methodology, a dataset, and a trained VLM.

**Key Insight**: Visual symbols (e.g., color-coded arrows for motion direction, crosshairs for target positions) are drawn directly on video frames, combined with VLM-generated textual descriptions, substantially reducing annotation cost.

**Core Idea**: Visual symbols serve as an intermediate representation that enables rapid human annotation (via mouse drag) while providing structured corrective guidance signals for VLMs.

## Method

### Overall Architecture
Three stages: (1) **Data Collection** — teleoperation and VLA rollouts to collect 5,202 real-world trajectories; (2) **ViFailback Annotation** — visual symbol drawing combined with VLM-assisted text generation, yielding 58,128 VQA pairs; (3) **Training ViFailback-8B** (fine-tuned from Qwen3-VL-8B) — deployed as an external supervisor for VLA systems.

### Key Designs
1. **Seven-Category Visual Symbol System**:

    - **Motion symbols**: color-coded straight arrows (red = forward/backward, green = left/right, blue = up/down), semicircular arrows (rotation)
    - **Spatial relation symbols**: dual crosshairs with dashed line (alignment), single crosshair (target region)
    - **State symbols**: ON/OFF labels (gripper open/close), prohibition icon (stop), rewind icon (retreat)

   **Design Motivation**: Color-encoding 3D direction via straight arrows is the key innovation for representing 3D motion through 2D annotation; the seven categories cover all fundamental action primitives required for manipulation correction.

2. **Fine-Grained Task Definitions**: Failure diagnosis encompasses five subtasks (detection, keyframe localization, sub-task localization, type identification, and cause analysis); corrective guidance encompasses three subtasks (low-level textual, high-level textual, and visual symbol guidance), yielding 11 VQA task categories in total. **Design Motivation**: Fine-grained decomposition enables precise identification of which capability dimension a VLM underperforms on.

3. **ViFailback-Bench**: Divided into Lite (closed-ended questions testing diagnosis and low-level correction) and Hard (open-ended questions testing failure reasoning and high-level strategy), covering 500 trajectories × 22 tasks. **Design Motivation**: Closed-ended questions assess foundational capabilities, open-ended questions probe reasoning depth, and CoT-format tests evaluate end-to-end diagnosis and correction.

### Loss & Training
- LoRA (rank=32, α=64) fine-tuning of Qwen3-VL-8B, 1 epoch, lr=1e-5
- LLM backbone and adapter parameters unfrozen
- DeepSpeed ZeRO Stage 2, 4× NVIDIA Hopper GPUs
- Temperature=0, maximum generation length of 2048 tokens

## Key Experimental Results

### Main Results (ViFailback-Bench Overall Accuracy %)

| Model | Lite↑ | Hard↑ | Average↑ |
|------|-------|-------|---------|
| Qwen3-VL-8B (base) | 38.33 | 33.04 | 35.92 |
| GPT-4o | 48.21 | 40.00 | 44.47 |
| Gemini-2.5-Pro | 54.64 | 32.45 | 44.54 |
| RoboBrain2.0-32B | 49.92 | 29.22 | 40.50 |
| **ViFailback-8B (Ours)** | **Outperforms all** | **Outperforms all** | **Significant gain** |

### Ablation Study (Real-World Robot Experiments)

| Configuration | Success Rate Gain | Notes |
|------|----------|------|
| VLA alone | baseline | No external supervisor |
| **VLA + ViFailback-8B** | **+22.2%** | External supervisor intervenes to recover from failures |

### Key Findings
- ViFailback-8B substantially outperforms the base Qwen3-VL-8B across all 11 VQA tasks, validating dataset training effectiveness.
- Even top-tier closed-source models such as GPT-4o and Gemini-2.5-Pro perform poorly on robotic failure analysis, underscoring the necessity of domain-specific data and training.
- Visual symbol outputs serve not only as intuitive human-readable representations but also directly guide VLA action adjustment.
- Among the four failure types, gripper 6D-pose errors are most prevalent, with task planning errors also accounting for a significant proportion.

## Highlights & Insights
- **Visual symbols as intermediate representation** constitute the core innovation: annotation cost is reduced while structured action guidance is provided to the robot. The color-encoded 3D directional arrow design is particularly elegant.
- **Real-world data outperforms simulation data**: 5,202 real-world trajectories provide substantially greater value than large-scale simulation data.
- **Closed-loop from diagnosis to recovery**: The framework not only analyzes failures but also guides VLA recovery via visual symbols; the 22.2% success rate improvement demonstrates practical value.
- **ViFailback-Bench** fills the gap in robotic VLM evaluation along the "failure reasoning" dimension.

## Limitations & Future Work
- Data collection currently uses the ALOHA bimanual platform; generalization to other robotic platforms requires further validation.
- The 100 tasks, while diverse, may not sufficiently cover all manipulation skills.
- VLA instruction-following capability remains a bottleneck — even with correct corrective guidance, the VLA may fail to execute it precisely.
- Visual symbol drawing still requires human involvement, albeit significantly faster than pure textual annotation.
- Online learning — real-time policy updates from failures — remains unexplored.
- The coverage adequacy of the seven-category visual symbol system requires further validation.
- Real-time failure detection latency (video processing delay) is not sufficiently discussed.

## Related Work & Insights
- **Compared to YAY** (human-in-the-loop correction): ViFailback reduces the complexity of human involvement via visual symbols.
- **Compared to simulation failure datasets (AHA, RACER)**: Real-world data eliminates the sim-to-real gap.
- **Compared to Robo2VLM and ManipBench**: The latter evaluate "what to do / how to do it," whereas ViFailback evaluates "where it went wrong / why it went wrong."
- The visual symbol paradigm is generalizable to other human-robot interaction scenarios (e.g., teleoperation guidance).
- Trajectory-conditioned models such as TracVLA overlay 2D trajectories but cannot revise them; ViFailback enables real-time correction.

## Technical Details
- **Data**: 4,995 teleoperation + 207 $\pi_{0.5}$ rollout trajectories; 657 successes + 4,545 failures
- **Four failure types**: task planning / gripper 6D-pose / gripper state / human intervention
- **Annotation stages**: 1. Fill basic information via UI → 2. Select correction options and draw symbols → 3. VLM generates text + human verification
- **VLM assistance**: Qwen2.5-Max for sub-task decomposition; Qwen3-VL-235B for high-level description generation
- **Platform**: ALOHA bimanual robot, 100 tasks
- **Bench**: 500 trajectories × 22 tasks, 5 fully OOD
- **Evaluation**: Closed-ended accuracy + open-ended GPT-4o scoring across three dimensions (semantic correctness / completeness / equivalence)
- **LoRA training**: rank=32, α=64, 1 epoch, lr=1e-5, DeepSpeed ZeRO Stage 2
- **Color encoding**: red = forward/backward (X-axis), green = left/right (Y-axis), blue = up/down (Z-axis); 2D arrows encode 3D motion

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Visual symbol annotation framework + real-world failure dataset + diagnosis-correction closed loop; comprehensive contributions
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 16-model benchmark + real-robot recovery experiments
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear; task definitions are comprehensive
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical solution to the critical problem of learning from robotic failures

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](force_transferable_visual_jailbreaking_attacks_via_feature_over-reliance_correct.md)

</div>

<!-- RELATED:END -->
