---
title: >-
  [Paper Note] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
description: >-
  [CVPR 2026][Robotics][Failure Diagnosis] This paper proposes the ViFailback framework, which leverages explicit visual symbols (arrows, crosshairs, etc.) to efficiently annotate real-world robotic manipulation failure data. It constructs a large-scale dataset of 58,128 VQA pairs and fine-tunes ViFailback-8B, which, when combined with a VLA model in real-robot experiments, achieves failure recovery with an average success rate improvement of 22.2%.
tags:
  - CVPR 2026
  - Robotics
  - Failure Diagnosis
  - Vision-Language Models
  - Robotic Manipulation
  - Visual Symbols
  - VLA
date: 2026-05-08
content_hash: 4e0b9bc6518b8096
---

# Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols

**Conference**: CVPR 2026
**arXiv**: [2512.02787](https://arxiv.org/abs/2512.02787)
**Code**: [Project Page](https://github.com/)
**Area**: Robotics
**Keywords**: Failure Diagnosis, Vision-Language Models, Robotic Manipulation, Visual Symbols, VLA

## TL;DR

This paper proposes the ViFailback framework, which leverages explicit visual symbols (arrows, crosshairs, etc.) to efficiently annotate real-world robotic manipulation failure data. It constructs a large-scale dataset of 58,128 VQA pairs and fine-tunes ViFailback-8B, which, when combined with a VLA model in real-robot experiments, achieves failure recovery with an average success rate improvement of 22.2%.

## Background & Motivation

Vision-Language-Action (VLA) models have achieved remarkable progress in robotic manipulation, yet inevitably encounter out-of-distribution (OOD) scenarios during real-world deployment, leading to action failures. Existing approaches face several core challenges:

1. **Scarcity of failure data**: Most existing failure datasets are programmatically generated in simulation via injected perturbations, and the sim-to-real gap limits their transferability to real-world scenarios.
2. **Low annotation efficiency**: Annotating real-world failure data requires extensive manual textual descriptions, particularly for abstract categories such as task planning failures and failure causes.
3. **Limited feedback modality**: Corrective feedback in existing methods is predominantly text-based; however, current VLA models have limited instruction-following capability, making pure text guidance insufficient for effective failure recovery.

The paper's core insight is that large volumes of failure data are inevitably produced during teleoperation data collection or policy rollouts, and the key challenge lies in how to annotate and exploit such data efficiently.

## Method

### Overall Architecture

The ViFailback framework consists of three core components:
1. **Data annotation framework**: An efficient, semi-automatic annotation pipeline based on visual symbols.
2. **ViFailback dataset and benchmark**: 58,128 VQA pairs and the ViFailback-Bench evaluation benchmark.
3. **ViFailback-8B model**: A fine-tuned Qwen3-VL-8B that serves as an external supervisor during VLA execution for failure diagnosis and correction.

### Key Designs

1. **Visual symbol system (7 symbols across 3 categories)**:
   - **Motion symbols**: Colored straight arrows (red = forward/backward, green = left/right, blue = up/down for 3D spatial motion) and semicircular arrows (indicating end-effector rotation direction).
   - **Spatial relationship symbols**: Dual crosshairs (connected by dashed lines to indicate alignment between two targets) and single crosshairs (marking target objects or regions).
   - **State symbols**: ON/OFF labels (end-effector open/close state), a prohibition icon (end-effector should stop), and a rewind icon (revert to a prior state).
   - **Design Motivation**: Annotators only need to draw these symbols on video frames with a mouse, after which a VLM automatically generates the required textual annotations, substantially reducing annotation cost.

2. **Fine-grained task definition**: Failure analysis is decomposed into two components:
   - **Failure diagnosis** (5 items): failure detection, critical frame localization, subtask localization, failure type identification (4 categories: task planning, gripper pose, gripper state, and human intervention), and failure cause reasoning.
   - **Corrective action guidance** (3 items): low-level text guidance (specific motion directions), high-level text guidance (task plan restructuring), and visual guidance (overlaying visual symbols on critical frames).

3. **Three-stage annotation pipeline**:
   - Stage 1: Fill in basic semantic information (failure diagnosis annotations via UI sliders and buttons).
   - Stage 2: Based on selected critical frames, annotators choose corrective action categories and draw visual symbols.
   - Stage 3: Qwen3-VL-235B automatically generates high-level descriptions by integrating all annotation information and visual symbols, followed by human verification and correction.

4. **ViFailback-Bench**: Comprises 500 trajectories across 22 tasks.
   - **Lite version**: Closed-form VQA assessing core diagnostic capabilities and low-level correction given a provided critical frame.
   - **Hard version**: Open-form VQA requiring the model to first detect and localize failures, then output guidance in Chain-of-Thought format.

### Loss & Training

- Qwen3-VL-8B is fine-tuned using LoRA (rank = 32, α = 64).
- Training runs for 1 epoch with a learning rate of 1e-5.
- DeepSpeed ZeRO-2 stage is used for distributed training.
- Both the LLM backbone and adapter parameters are unfrozen.
- Hardware: 4 × NVIDIA Hopper GPUs.

## Key Experimental Results

### Main Results (ViFailback-Bench Evaluation)

| Model | Lite (%) | Hard (%) | Avg (%) |
|-------|---------|---------|---------|
| Gemini-2.5-Pro | 54.64 | 32.45 | 44.54 |
| GPT-4o | 48.21 | 40.00 | 44.47 |
| Qwen2.5-VL-72B | 50.61 | 36.56 | 44.21 |
| Qwen3-VL-32B | 47.79 | 35.23 | 42.07 |
| RoboBrain2.0-32B | 49.92 | 29.22 | 40.50 |
| **ViFailback-8B (Ours)** | **Best** | **Best** | **Best** |

ViFailback-8B significantly outperforms all open-source and closed-source models under both the Lite and Hard settings.

### Real-Robot Experiments

| Configuration | Avg. Success Rate Improvement |
|---------------|-------------------------------|
| Baseline VLA (without ViFailback-8B) | Baseline |
| VLA + ViFailback-8B supervision | **+22.2%** |

### Dataset Scale

| Metric | Value |
|--------|-------|
| Real trajectories | 5,202 |
| VQA pairs | 58,128 |
| Tasks covered | 100 |
| Failure types | 4 categories |
| Success / Failure trajectories | 657 / 4,545 |

### Key Findings

1. Even state-of-the-art closed-source models such as Gemini-2.5-Pro exhibit limited performance on robotic failure diagnosis and correction (44.54%), indicating that this domain requires specialized data and training.
2. Embodied VLMs (e.g., RoboBrain2.0, Cosmos-Reason1) do not outperform general-purpose VLMs on this benchmark, suggesting that embodied knowledge does not equate to failure understanding capability.
3. Visual symbols are effective not only as annotation aids but also as runtime correction signals for VLA models, proving more effective than purely textual instructions.

## Highlights & Insights

1. **Elegant visual symbol design**: 3D directions are encoded via color, and complex semantics are expressed through simple geometric symbols, simultaneously lowering the annotation barrier and enabling VLMs to learn their generation.
2. **Real-world data priority**: Rather than pursuing simulation-based generation, the paper directly collects real failure data from teleoperation and policy rollouts, yielding greater practical value.
3. **Closed-loop validation**: Beyond training a diagnostic model, the paper validates the VLA + external supervisor failure recovery paradigm on a real robot.
4. **Low annotation cost**: The combination of visual symbols and VLM assistance enables semi-automatic annotation, making large-scale real-world failure data construction feasible.

## Limitations & Future Work

1. The current visual symbol system covers only 4 failure categories; more complex failure modes (e.g., multi-step reasoning failures) may require an extended symbol set.
2. The supervision frequency of ViFailback-8B is fixed (queried every 6 action chunks); an adaptive triggering mechanism may be more efficient.
3. The dataset is primarily built on the ALOHA bimanual platform; generalization to other robot morphologies requires further validation.
4. Although the annotation pipeline is efficient, human involvement remains necessary; fully automated annotation is a direction for future work.

## Related Work & Insights

- **YAY (Yell at Your Robot)**: Improves corrective instructions via human-in-the-loop feedback, but suffers from poor scalability.
- **AHA / RACER / RoboFAC**: Fine-tune VLMs on failure data synthesized in simulation, limited by the sim-to-real gap.
- **TraceVLA / MOKA / RoVI**: Leverage visual prompts to guide robot policies, but focus on initial guidance rather than real-time correction.
- **Inspiration from this work**: Visual symbols can serve as a bridge language between VLMs and VLAs — human-readable and machine-parseable simultaneously.

## Rating

- Novelty: ⭐⭐⭐⭐ — Using visual symbols for failure annotation and correction is a novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — The pipeline is complete, spanning dataset construction, model evaluation, and real-robot validation.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear, though the symbol system section could be more concise.
- Value: ⭐⭐⭐⭐⭐ — Provides a complete framework and data foundation for robots to learn from failures.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)

<!-- RELATED:END -->
