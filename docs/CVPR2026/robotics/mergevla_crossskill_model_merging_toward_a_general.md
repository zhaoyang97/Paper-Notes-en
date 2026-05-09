---
title: >-
  [Paper Note] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent
description: >-
  [CVPR 2026][Robotics][Vision-Language-Action] MergeVLA diagnoses two root causes of VLA model unmergeability—LoRA parameter conflicts and architectural incompatibility induced by self-attention in the action expert—and addresses them via sparsely activated task masks and a self-attention-free action expert architecture. This enables training-free merging of multiple single-task VLA experts, achieving 90.2% success on LIBERO and 90.0% on a real-robot SO101 platform.
tags:
  - CVPR 2026
  - Robotics
  - Vision-Language-Action
  - model merging
  - multi-task robotics
  - task mask
  - cross-skill generalization
date: 2026-05-08
content_hash: ed6febf504782268
---

# MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent

**Conference**: CVPR 2026
**arXiv**: [2511.18810](https://arxiv.org/abs/2511.18810)
**Code**: [Project Page](https://mergevla.github.io/)
**Area**: Robotic Manipulation / VLA Models / Model Merging
**Keywords**: Vision-Language-Action, model merging, multi-task robotics, task mask, cross-skill generalization

## TL;DR

MergeVLA diagnoses two root causes of VLA model unmergeability—LoRA parameter conflicts and architectural incompatibility induced by self-attention in the action expert—and addresses them via sparsely activated task masks and a self-attention-free action expert architecture. This enables training-free merging of multiple single-task VLA experts, achieving 90.2% success on LIBERO and 90.0% on a real-robot SO101 platform.

## Background & Motivation

**Background**: VLA (Vision-Language-Action) models achieve strong single-task performance by fine-tuning VLMs for robotic manipulation, but fail to generalize across multiple tasks. Model merging has proven effective in the LLM/VLM literature.

**Limitations of Prior Work**:

1. Directly merging VLA experts causes **success rates to drop to zero**—a failure mode never observed in LLM/VLM merging.
2. When merging four tasks, over 75% of LoRA parameters are "selfish" (retained by only one task), resulting in severe parameter conflicts.
3. Self-attention layers in the action expert accumulate strong task-specific dependencies during training, causing explosive growth in parameter distances across deep blocks and undermining modular composability.

**Key Challenge**: LoRA parameters diverge drastically across tasks, and self-attention in the action expert propagates task-specific information throughout all layers—their combined effect renders existing merging methods completely ineffective.

**Goal**: Design a VLA architecture that is "merging-friendly" by construction, enabling efficient merging of single-task experts into a generalist model.

**Key Insight**: Precisely diagnose the two root causes of failure before designing targeted solutions—task masks to resolve parameter conflicts, and self-attention removal to resolve architectural incompatibility.

**Core Idea**: Suppress conflicting LoRA parameters via sparsely activated task masks, and eliminate task-dependency propagation by removing self-attention from the action expert, making VLA models inherently mergeable.

## Method

### Overall Architecture

MergeVLA builds on the VLA-Adapter architecture (Qwen2.5-0.5B as the VLM backbone) with three key modifications: (1) task-specific binary masks applied to VLM LoRA parameters to resolve parameter conflicts; (2) removal of all self-attention layers from the action expert, retaining only cross-attention; (3) a training-free task router at inference time to identify the current task. Each task is fine-tuned independently; the merging stage is entirely training-free.

### Key Designs

1. **Task Mask (Resolving LoRA Parameter Conflicts)**

   - Function: Constructs a binary mask per task to selectively activate merged parameters consistent with that task and suppress conflicting ones.
   - Mechanism: For each parameter position, the method checks whether the task vector aligns directionally and significantly with the merged vector: $S_m = \mathbf{I}[|\tau_m| > \lambda \cdot |\tau_{\text{merge}} - \tau_m|]$, where $\lambda$ controls the tolerance threshold.
   - Practical Effect: When merging four tasks, over 75% of parameters are selfish; the mask retains beneficial parameters, suppresses conflicts, and encourages some parameters to revert to pre-trained weights, mitigating visual forgetting.
   - Design Motivation: Naive merging activates parameters irrelevant or contradictory to the current task; the mask enables selective parameter activation.

2. **Self-Attention-Free Action Expert (Resolving Architectural Incompatibility)**

   - Function: Redesigns the action expert architecture to be inherently mergeable.
   - Mechanism: (a) Removes all self-attention layers, retaining only cross-attention—forcing the expert to rely on the VLM's robust representations; (b) replaces tanh gating with sigmoid gating to prevent negative activations from suppressing VLM signals.
   - Shallow blocks are merged via weight averaging; the final layer (expert head) remains task-specific and is not merged.
   - Design Motivation: Self-attention accumulates task-specific bias during training from scratch and propagates it across layers. Its removal enforces reliance on pre-trained VLM features, which improves generalization (+13.4% on OOD).

3. **Test-Time Task Router (Training-Free Task Inference)**

   - Function: Automatically identifies the current task at inference time when task identity is unknown, selecting the corresponding mask and expert head.
   - Mechanism: For each candidate task $m$, construct a VLM variant using the corresponding task mask → extract hidden states → project onto the top-$k_r$ right singular vector subspace of the action expert's value matrix → compute activation strength → select the highest-scoring task via softmax.
   - Routing is performed once at $t=0$ and fixed thereafter.
   - Design Motivation: The value subspace directly encodes task-dependent information and is more stable and discriminative than query or key subspaces.

### Loss & Training

- Each task is trained independently with standard imitation learning (30k–50k steps, batch size 8, LoRA rank 32).
- The merging stage is entirely training-free: LoRA weights are merged via TIES/TA/WUDI; shallow action expert blocks are merged by weight averaging; task-specific expert heads are preserved.
- Hardware: Single NVIDIA A6000 Ada 48GB GPU.

## Key Experimental Results

### Main Results

| Method | Dataset | Avg. Success Rate (%) | Comparison |
|---|---|---|---|
| MergeVLA (TIES+Mask) | LIBERO (4 suites) | **90.2** | Single-task upper bound 96.7% (−6.5 pp) |
| MergeVLA | LIBERO-Plus (OOD) | **62.5** | VLA-Adapter 59.0% (single-task fine-tuning) |
| MergeVLA | RoboTwin (cross-embodiment) | **70.7** | Single-task upper bound 76.0% (−5.3 pp) |
| MergeVLA | SO101 Real Robot (3 tasks) | **90.0** | On par with single-task fine-tuning |

| Method | Params (B) | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| OpenVLA (TA merge) | 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| OpenVLA (TA+Mask) | 7 | 74.2 | 82.6 | 68.8 | 24.0 | 62.4 |
| VLA-Adapter (TA+Mask) | 0.68 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MergeVLA (TIES+Mask) | 0.70 | 94.8 | 94.6 | 91.8 | 79.4 | **90.2** |

### Ablation Study

| Configuration | LIBERO Avg | Notes |
|---|---|---|
| Mask only (no action expert modification) | 0.0% | Mask necessary but insufficient |
| Self-attention removal only (no mask) | 65.5% | Architecture modification effective but requires mask |
| Self-attention removal + Mask | **90.2%** | Both components are indispensable |
| $\lambda = 0.6$–$0.9$ | >70% | Optimal tolerance range |
| Routing via Value | **89.7%** | Most stable |
| Routing via Key | Significant drop | Zero success on certain tasks |
| Self-attention removal (LIBERO-Plus OOD) | +13.4% | This modification alone substantially improves generalization |

### Key Findings

- Task mask and self-attention removal are **both indispensable**: the former resolves VLM parameter conflicts, the latter resolves action expert incomposability.
- Removing self-attention alone improves OOD performance by 13.4%, identifying it as the primary bottleneck for generalization.
- Value subspace routing substantially outperforms query/key-based alternatives.
- On the real robot, the merged model matches single-task fine-tuning (90.0%), demonstrating practical viability.

## Highlights & Insights

1. The **diagnosis-driven research paradigm** is particularly elegant: root causes are experimentally identified with precision before targeted solutions are designed.
2. The architectural modifications are minimal yet highly effective—removing self-attention and replacing the gating function substantially improves generalization.
3. The test-time task router is entirely training-free, leveraging SVD of the value subspace for task discrimination.
4. On the real SO101 robot, the merged model matches single-task fine-tuning performance (90%), demonstrating strong practical value.

## Limitations & Future Work

1. Each task still requires a dedicated expert head and task mask; storage grows linearly with the number of tasks.
2. The VLM backbone is limited to Qwen2.5-0.5B; effectiveness on larger models (7B+) remains unverified.
3. Routing is performed only once at $t=0$, which may be insufficient for long-horizon tasks requiring mid-sequence skill switching.
4. Cross-embodiment experiments are conducted at a relatively small scale (3 robot types); scalability to large-scale heterogeneous merging remains to be validated.

## Related Work & Insights

- **vs. OpenVLA**: Direct merging fails completely (0%), as task conflicts in the LM body cannot be resolved by simple methods. MergeVLA circumvents this via task masks.
- **vs. VLA-Adapter**: Self-attention renders the action expert incomposable; even adding masks fails. MergeVLA eliminates the architectural obstacle by design.
- **vs. pi0/pi0.5**: Large-scale VLAs rely on joint training for multi-task capability at high cost. MergeVLA permits independent training followed by merging, offering greater flexibility.
- **Insight**: "Removing self-attention improves generalization" deserves broader attention—similar phenomena may exist in other modules trained from scratch, and extension to continual skill learning scenarios is worth exploring.

## Rating

- Novelty: ⭐⭐⭐⭐ — The diagnosis-plus-design paradigm is well-structured, though individual technical components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three simulation benchmarks, real-robot experiments, and extensive ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ — Narrative logic is clear, progressing systematically from diagnosis to solution.
- Value: ⭐⭐⭐⭐ — Addresses a critical obstacle in VLA merging with practical implications for multi-skill robot learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)

</div>

<!-- RELATED:END -->
