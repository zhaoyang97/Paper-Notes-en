---
title: >-
  [Paper Note] RoboAgent: Chaining Basic Capabilities for Embodied Task Planning
description: >-
  [CVPR 2026][Reinforcement Learning][Embodied Task Planning] This paper proposes RoboAgent, a capability-driven embodied task planning framework that employs a single VLM to simultaneously serve as a scheduler and five basic capabilities (exploration guidance, object grounding, scene description, action decoding, experience summarization). Through three-stage training (SFT + DAgger + expert-guided RL), RoboAgent achieves state-of-the-art performance on EB-ALFRED and ALFWorld.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Embodied Task Planning
  - Capability Chaining
  - Vision-Language Model
  - Multi-stage Training
date: 2026-05-08
content_hash: 3d5f980b7f7ec434
---

# RoboAgent: Chaining Basic Capabilities for Embodied Task Planning

**Conference**: CVPR 2026  
**arXiv**: [2604.07774](https://arxiv.org/abs/2604.07774)  
**Code**: [https://github.com/woyut/RoboAgent_CVPR26](https://github.com/woyut/RoboAgent_CVPR26)  
**Area**: Reinforcement Learning  
**Keywords**: Embodied Task Planning, Capability Chaining, Vision-Language Model, Reinforcement Learning, Multi-stage Training

## TL;DR

This paper proposes RoboAgent, a capability-driven embodied task planning framework that employs a single VLM to simultaneously serve as a scheduler and five basic capabilities (exploration guidance, object grounding, scene description, action decoding, experience summarization). Through three-stage training (SFT + DAgger + expert-guided RL), RoboAgent achieves state-of-the-art performance on EB-ALFRED and ALFWorld.

## Background & Motivation

1. **State of the Field**: Embodied Task Planning (ETP) requires an agent to execute sequences of atomic actions in an environment based on visual observations and language instructions. While VLMs have demonstrated strong multimodal understanding, their performance remains limited in embodied planning scenarios involving multi-turn interaction, long-horizon reasoning, and extended context analysis.

2. **Limitations of Prior Work**: (1) Intermediate reasoning generated via direct CoT lacks standardized formats and direct supervision, making it difficult to ensure correctness and utility; (2) methods relying on closed-source models or external tools cannot be trained end-to-end; (3) standard RL struggles to learn effective policies in exploration scenarios with sparse rewards.

3. **Root Cause**: Complex planning implicitly involves multiple intermediate processes—intent understanding, commonsense reasoning, environment analysis, action modeling, and progress monitoring—yet existing methods conflate these into a single process, making fine-grained supervision over intermediate steps difficult.

4. **Paper Goals**: To decompose complex planning into a series of fundamental visual-language problems, enabling a single VLM to achieve controllable and transparent reasoning through explicit capability invocations.

5. **Starting Point**: Define a set of visual-language capabilities critical to embodied scenarios; a scheduler determines which capability to invoke and when; each capability maintains its own context and produces intermediate reasoning results or environment interactions.

6. **Core Idea**: A single VLM simultaneously acts as the scheduler and multiple capability modules, replacing "free-form CoT" with a "structured capability invocation chain," coupled with multi-stage training that leverages simulator-internal information.

## Method

### Overall Architecture

RoboAgent consists of a Scheduler and five Capabilities, all implemented with the same VLM (Qwen2.5-VL-3B). The scheduler receives the task instruction and historical context, and generates a sequence of [(capability name, query)] pairs. Each capability receives a query and an optional observation image, and outputs an action sequence or textual feedback to the scheduler. The entire process requires no external tools.

### Key Designs

1. **Design of the Five Basic Capabilities**:

    - Function: Decompose complex planning into concrete visual-language sub-problems.
    - Mechanism: (1) **Exploration Guidance (EG)**: Predicts the most likely exploration direction for a target object via commonsense reasoning; (2) **Object Grounding (OG)**: Open-vocabulary object detection to determine whether the target is in the current field of view; (3) **Scene Description (SD)**: Generates textual descriptions of target object states; (4) **Action Decoding (AD)**: Translates navigation/manipulation instructions into atomic action sequences; (5) **Experience Summarization (ES)**: Summarizes the outcomes of recent actions and analyzes failure causes. AD produces actions without textual feedback; the remaining four produce textual feedback without generating actions.
    - Design Motivation: Each capability corresponds to a fundamental visual-language task at which VLMs inherently excel (spatial reasoning, detection, description, instruction following, summarization). Structured invocation fully exploits the VLM's intrinsic capabilities while enabling fine-grained supervision over each capability.

2. **Three-Stage Training Pipeline**:

    - Function: Progressively improve the model from basic capability acquisition to complex reasoning.
    - Mechanism: **Stage 1 (SFT-Expert)**: Supervised fine-tuning on expert trajectories. Simulator-internal information (scene graphs, segmentation masks, environment messages) is used to construct training labels for each capability, yielding 640k samples. **Stage 2 (DAgger-SFT)**: The Stage 1 model is deployed to collect self-generated trajectories; semantic matching aligns the model's capability invocations with ground-truth, constructing corrective training labels (690k samples) with object description and action format augmentation. **Stage 3 (RFT)**: The scheduler is reinforcement fine-tuned using the Expert-guided Policy Optimization (EIPO) algorithm, with rewards based on successful capability-invocation completion of manipulation sub-plans; diverse interactive data (25k trajectories) is synthesized.
    - Design Motivation: SFT instills basic formats and skills; DAgger corrects distributional shift; RFT improves the scheduler's generalization across diverse scenarios.

3. **Expert-guided Policy Optimization (EIPO)**:

    - Function: More stable reinforcement learning training for the scheduler.
    - Mechanism: Unlike PPO/GRPO, which optimize policy return improvements, EIPO directly maximizes the expert's advantage function $A_{\pi^*}(s,a)$. Since the expert policy is deterministic, $A_{\pi^*}$ can be computed exactly without Monte Carlo estimation. A GRPO-style group-based mean is used as the baseline, assigning positive gradients to relatively better actions within the group and negative gradients to worse ones. The objective is $J(\pi) = \mathbb{E}_{s \sim D} \frac{1}{G} \sum_{i=1}^{G} [r(a^i, s) \hat{A}_{\pi^*}(s, a^i)]$.
    - Design Motivation: The optimality guarantee of the expert policy ensures $A_{\pi^*}(s,a) \leq 0$, naturally suppressing all suboptimal actions. Compared to GRPO, which uses episode-level returns, EIPO employs step-level advantage functions, yielding faster and more stable convergence.

### Loss & Training

Stages 1–2 use standard cross-entropy loss for SFT. Stage 3 uses the EIPO algorithm with learning rate 5e-6, batch size 512, and 120 policy update iterations. All training is conducted on 4× H800 GPUs. The base model is Qwen2.5-VL-3B; Stage 1 uses learning rate 1e-5, batch size 32, and 2 epochs.

## Key Experimental Results

### Main Results

| Benchmark | Method | Base Model | Avg. SR |
|-----------|--------|------------|---------|
| EB-ALFRED | Claude-3.7-Sonnet (zero-shot) | - | 67.7 |
| EB-ALFRED | WAP | Qwen2.5-VL-7B | 62.7 |
| EB-ALFRED | **RoboAgent** | **Qwen2.5-VL-3B** | **67.0** |
| ALFWorld (Visual) | SEEA-R1 | Qwen2.5-VL-7B | 36.0 |
| ALFWorld (Visual) | **RoboAgent** | **Qwen2.5-VL-3B** | **77.6** |
| ALFWorld (Text) | DynaMind | Qwen2.5-7B | 92.5/89.1 |
| ALFWorld (Text) | **RoboAgent** | **Qwen2.5-VL-3B** | **92.1/94.0** |

In the ALFWorld visual environment, RoboAgent achieves 77.6% SR, substantially outperforming all existing RL methods (the runner-up SEEA-R1 achieves only 36.0%), a margin of 41.6 percentage points.

### Ablation Study

| Training Configuration | ALFWorld SR | EB-ALFRED SR | Notes |
|------------------------|-------------|--------------|-------|
| SFT-expert | 44.8 | 62.0 | Expert trajectory SFT only |
| +DAgger (aug. gen.) | 73.1 | 64.3 | DAgger with model-generated data |
| +RFT (aug. exp.) | 74.6 | 65.7 | RFT with augmented expert data |
| +RFT (aug. syn.) | **77.6** | **67.0** | Full model with synthetic data RFT |

### Key Findings

- **DAgger stage contributes most**: ALFWorld SR increases from 44.8 to 73.1 (+28.3), indicating that corrective supervision via self-collected trajectories is critical for mitigating distributional shift.
- **EIPO converges faster than GRPO**: Under the same number of iterations, EIPO achieves higher ALFWorld SR than GRPO, validating the stability advantage of step-level advantage functions.
- **Cross-modal generalization**: The same visual model directly adapts to text-only environments, achieving 92.1/94.0 (seen/unseen), approaching methods specifically designed for text, demonstrating that the capability framework acquires modality-agnostic planning abilities.
- **OOD generalization**: RoboAgent outperforms other open-source transfer models on EB-Habitat (22.3) and LoTa-WAH (22.1), though a gap remains compared to closed-source GPT-4o (59.0).

## Highlights & Insights

- **The "capability as interface" design philosophy is highly instructive**: Unlike free-form CoT, structured capability invocation renders intermediate reasoning supervisable, diagnosable, and replaceable—an important paradigm for VLM agent design.
- **Leveraging simulator privileged information for training data construction**: Using scene graphs, segmentation masks, and other inference-time-inaccessible information to construct high-quality labels for each capability during training is a clever knowledge distillation strategy.
- **Single model, multiple roles**: The scheduler and all capabilities share a single 3B VLM without requiring external tools or multi-model collaboration, substantially reducing deployment complexity. This design is transferable to other multi-tool agent systems.
- **EIPO algorithm generalizability**: Exploiting the determinism of the expert policy to obtain precise advantage estimates makes EIPO applicable to any RL scenario with a reliable expert policy.

## Limitations & Future Work

- Training is conducted solely in AI2-THOR/ALFRED simulators; real-world generalization remains unvalidated.
- The five capabilities are predefined, precluding dynamic extension with new capabilities or task-adaptive selection.
- OOD results reveal a substantial cross-simulator generalization gap (approximately 37 percentage points behind GPT-4o zero-shot).
- The 3B model may be constrained in complex reasoning scenarios; larger models could yield further improvements.
- Future work could explore mechanisms for automatic capability discovery and composition, rather than relying on a fixed set of five.

## Related Work & Insights

- **vs. SEEA-R1**: SEEA-R1 achieves 36.0% ALFWorld SR with a 7B model + RL; RoboAgent achieves 77.6% with a 3B model, demonstrating that structured capability invocation is more effective than free-form CoT reasoning.
- **vs. Closed-source zero-shot (Claude/GPT-4o)**: RoboAgent (67.0) approaches Claude-3.7-Sonnet (67.7) on EB-ALFRED and substantially surpasses GPT-4o (24.0) on ALFWorld visual, showing that domain-fine-tuned small models can match large models.
- **vs. Progressive planning (MPO/DynaMind)**: These methods decompose tasks via sub-goals; RoboAgent decomposes the reasoning process via capabilities, the latter providing a more fine-grained supervision interface.

## Rating

- Novelty: ⭐⭐⭐⭐ The capability-driven planning framework is novel in design and EIPO offers theoretical contributions, though capability definitions remain largely hand-crafted.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks (2 in-domain + 2 OOD), three modalities (visual/text/OOD), complete stage-wise ablation, and EIPO vs. GRPO comparison.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear and the training pipeline diagrams are intuitive, though the derivation sections are dense.
- Value: ⭐⭐⭐⭐⭐ Provides a reproducible and extensible paradigm for VLM-based embodied agents; 77.6% ALFWorld SR represents the strongest reported result to date.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Embodied Navigation with Auxiliary Task of Action Description Prediction](../../ICCV2025/reinforcement_learning/embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [\[CVPR 2026\] RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closedloop_robotic_data_generation_via_seman.md)
- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](../../ICLR2026/reinforcement_learning/autotool_scaling_tool_use.md)

<!-- RELATED:END -->
