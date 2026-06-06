---
title: >-
  [Paper Note] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent
description: >-
  [CVPR 2026][Robotics][VLA model merging] This paper presents the first systematic diagnosis of two root causes underlying the non-mergeability of VLA models—LoRA selfish parameter conflicts and task coupling induced by s…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "VLA model merging"
  - "multi-skill robotics"
  - "sparse LoRA masking"
  - "action expert redesign"
  - "test-time task routing"
date: 2026-05-08
content_hash: 9c50233716e96840
---

# MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent

**Conference**: CVPR 2026
**arXiv**: [2511.18810](https://arxiv.org/abs/2511.18810)  
**Code**: None  
**Area**: Robotics / Embodied Intelligence
**Keywords**: VLA model merging, multi-skill robotics, sparse LoRA masking, action expert redesign, test-time task routing

## TL;DR

This paper presents the first systematic diagnosis of two root causes underlying the non-mergeability of VLA models—LoRA selfish parameter conflicts and task coupling induced by self-attention in action experts—and proposes MergeVLA. By combining task-mask sparse LoRA activation, self-attention-free action experts, and training-free test-time task routing, MergeVLA merges multiple single-skill VLA specialists into a unified generalist agent, achieving a 90.2% success rate on LIBERO and 90% on the real-robot SO101 platform.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models fine-tune large-scale VLMs on millions of robot demonstration trajectories and achieve strong performance in single-task or single-embodiment settings. However, a truly generalist real-world agent must support diverse skills, embodiments, and environments. A natural approach is to merge multiple independently fine-tuned VLA specialists into a unified policy.

**Limitations of Prior Work**: Model merging is well-established for LLMs and VLMs (e.g., Task Arithmetic, TIES, DARE), yet directly applying these methods to VLAs causes the merged success rate to collapse to **0%**—a failure mode never observed in LLM merging.

**Root Cause Diagnosis** (a core contribution of this paper):

**LoRA Selfish Parameter Problem**: After fine-tuning four LIBERO tasks with LoRA, more than 75% of parameters are "selfish"—retained exclusively by a single task's mask—indicating that different tasks push LoRA updates into highly disjoint directions. Naively averaging or sign-based merging activates irrelevant or conflicting parameters, corrupting the shared visual-language subspace.

**Action Expert Architecture Incompatibility**: Even when the VLM backbone is merged perfectly, simply averaging action expert weights still yields 0% success. The root cause lies in the action experts being trained from scratch with self-attention layers; self-attention allows task-specific information to accumulate across layers, causing deep-layer parameters to become highly task-specialized and non-recomposable.

**Key Insight**: Since the problem stems from architectures that are inherently non-mergeable, the solution is to design VLA architectures that are inherently mergeable from the ground up.

## Method

### Overall Architecture

MergeVLA consists of three complementary components:

1. **Task-Mask Sparse LoRA** (resolves LoRA parameter conflicts in the VLM backbone)
2. **Self-Attention-Free Action Expert** (resolves action expert architectural incompatibility)
3. **Training-Free Test-Time Task Routing** (resolves unknown task identity at inference time)

The backbone VLM is Qwen2.5-0.5B; the action expert is redesigned from the VLA-Adapter architecture, yielding a total of approximately 0.7B parameters.

### Key Design 1: Task-Mask Sparse LoRA

**Problem**: Merging $M$ tasks' LoRA updates $\tau_m = \Theta_m - \Theta_0$ into $\tau_{\text{merge}}$ introduces pervasive conflicts.

**Solution**: A binary mask $\mathbf{S}_m$ is constructed for each task $m$ to selectively activate the merged parameters beneficial to that task:

$$\Theta_{\text{merge}}^{(m)} = \Theta_0 + \mathbf{S}_m \odot \tau_{\text{merge}}$$

Masks are generated via a parameter-level consistency test:

$$\mathbf{S}_m = \mathbb{I}\left[|\tau_m| > \lambda |\tau_{\text{merge}} - \tau_m|\right]$$

Intuitively, only parameters whose task-specific update magnitude is large and directionally consistent with the merged update are retained; $\lambda$ controls sparsity. Empirically, $\lambda = 0.6$ performs best. A beneficial side effect is that some LoRA parameters revert to pre-trained weights, thereby preserving the original visual-language representations.

### Key Design 2: Self-Attention-Free Action Expert

**Problem**: The VLA-Adapter action expert consists of $L$ Transformer blocks (self-attention + cross-attention + FFN) trained from scratch. Self-attention propagates task dependencies across layers, causing inter-task parameter distances to grow explosively in deeper layers.

**Two Architectural Modifications**:

- **Remove self-attention**: Only the cross-attention pathway is retained, forcing the expert to rely on the robust shared features provided by the VLM rather than its own highly task-specialized representations learned from scratch.
- **Sigmoid replaces tanh gating**: The original tanh gate can produce negative values that suppress VLM signals; sigmoid ensures VLM information is always preserved and positively propagated.

**Layer-wise Merging Strategy**: Shallow block parameters exhibit small inter-task differences and can be directly averaged. Deep layers—typically the final block, termed the *expert head*—are highly specialized due to regression targets and are therefore not merged; each task retains its own dedicated expert head.

**Unexpected Benefit**: The self-attention-free design improves out-of-distribution (OOD) performance on LIBERO-Plus by **13.4%** over VLA-Adapter, demonstrating that leveraging the VLM's pre-trained robustness is more effective than learning task-specific representations from scratch.

### Key Design 3: Training-Free Test-Time Task Routing

When task identity is unknown at inference time, the router must automatically select the corresponding task mask and expert head from initial observations.

**Procedure**:

1. For each candidate task $m$, run the VLM with mask $\mathbf{S}_m$ to obtain hidden states.
2. Apply SVD to the value projection matrix of the merged action expert's $l$-th block and retain the top $k_r = 8$ right singular vectors to form the principal subspace.
3. Project each task's hidden state onto this subspace and compute a response score $r_m$.
4. Select the task with the highest softmax score and fix the corresponding mask and expert head for the entire episode.

**Design Choice**: Empirical results demonstrate that the value projection (V) subspace is more stable and discriminative than the key projection (K)—V encodes actual behavioral semantics, while K defines query similarity structure and is more prone to collapsing into task-specialized subspaces.

### Loss & Training

- Each task is fine-tuned independently (LoRA + action expert trained from scratch); 50 demonstrations per task; single NVIDIA A6000 (48 GB) GPU.
- The merging phase is entirely offline: merge LoRA → compute masks → average shallow action expert blocks → retain expert heads.
- The router requires no training; it is based purely on SVD parameter subspace analysis.
- Default hyperparameters: $l = L$, $k_r = 8$, $\lambda = 0.6$, $\alpha = 1$.

## Key Experimental Results

### Main Results: LIBERO Success Rate (%)

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| OpenVLA (independent fine-tuning) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| VLA-Adapter (independent fine-tuning) | 99.6 | 99.6 | 98.2 | 96.4 | 98.5 |
| MergeVLA (independent fine-tuning) | 98.0 | 98.6 | 95.0 | 95.0 | 96.7 |
| OpenVLA + TA (full merge) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| OpenVLA + TA + Mask | 74.2 | 82.6 | 68.8 | 24.0 | 62.4 |
| VLA-Adapter + TA + Mask | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MergeVLA + TIES + Mask** | **94.8** | **94.6** | **91.8** | **79.4** | **90.2** |
| MergeVLA + TA + Mask | 98.0 | 98.8 | 85.4 | 76.6 | 89.7 |

### OOD Robustness: LIBERO-Plus Success Rate (%)

| Method | BG | View | Instr. | Light | Layout | Robot State | Noise | Avg. |
|---|---|---|---|---|---|---|---|---|
| π₀ (independent fine-tuning) | 81.4 | 13.8 | 58.8 | 85.0 | 68.9 | 6.9 | 79.0 | 56.3 |
| VLA-Adapter (independent fine-tuning) | 76.6 | 36.4 | 73.8 | 71.0 | 70.2 | 37.4 | 57.2 | 59.0 |
| MergeVLA (independent fine-tuning) | 92.7 | 62.4 | 75.7 | 92.7 | 73.7 | 46.4 | 74.7 | 72.4 |
| **MergeVLA + TIES Merge** | **85.7** | **50.7** | **66.0** | **84.2** | **68.1** | **30.3** | **66.0** | **62.5** |

### Ablation Study: Routing Subspace Selection (LIBERO Success Rate %)

| Subspace | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| K only | 98.0 | 0.0 | 39.6 | 76.6 | 53.6 |
| K & V | 98.0 | 0.0 | 85.8 | 76.6 | 65.1 |
| **V only** | **98.0** | **98.8** | **85.4** | **76.6** | **89.7** |

### Real-Robot SO101 Success Rate (%)

| Method | Pick & Place | Push | Stack | Avg. |
|---|---|---|---|---|
| Independent fine-tuning | 90.0 | 85.0 | 95.0 | 90.0 |
| MergeVLA + TA | 70.0 | 70.0 | 60.0 | 66.7 |
| **MergeVLA + TIES** | **90.0** | **90.0** | **90.0** | **90.0** |

### Key Findings

- **Both root causes are necessary**: Applying masks alone without architectural changes (VLA-Adapter + TA + Mask) still yields 0%; modifying the architecture alone without masks likewise fails.
- **Removing self-attention yields unexpected OOD gains**: MergeVLA with independent fine-tuning already outperforms VLA-Adapter on LIBERO-Plus by 13.4%.
- **Merged ≈ independently fine-tuned**: TIES merging fully matches independent fine-tuning performance on the real robot (90% vs. 90%).
- **Cross-embodiment generalization**: On RoboTwin with 3 bimanual robot types × 3 tasks, TIES merging achieves 70.7%.
- **Routing accuracy**: Value projection subspace routing substantially outperforms key projection (89.7% vs. 53.6%).
- **Mask sparsity**: $\lambda \in [0.6, 0.9]$ is optimal; too small admits conflicting parameters, too large discards useful information.

## Highlights & Insights

1. **Systematic diagnosis of "VLA non-mergeability"**: This work is the first to identify LoRA selfish parameters (>75%) and self-attention task coupling as two independent root causes; the diagnostic contribution alone is significant.
2. **Architecture as mergeability**: Rather than designing better merging algorithms, MergeVLA eliminates non-mergeability at the architectural level—an elegant approach generalizable to other multimodal domains.
3. **Unexpected OOD benefit from removing self-attention**: A modification originally motivated by mergeability turns out to improve OOD generalization, suggesting that having action experts "trust the VLM" is more robust than learning independently from scratch.
4. **Training-free routing**: Using SVD principal components for task discrimination is both elegant and practical, requiring no additional training data or auxiliary networks.
5. **End-to-end validation from simulation to real robot**: Four-tier evaluation across LIBERO, LIBERO-Plus, RoboTwin, and real-robot SO101 covers cross-task, cross-environment, and cross-embodiment dimensions.

## Limitations & Future Work

1. **No online incremental merging**: Adding a new task requires recomputing masks and re-merging; plug-and-play extension is not supported.
2. **Limited VLM scale**: Only Qwen2.5-0.5B is evaluated; applicability to larger models (e.g., 7B+) remains unexplored.
3. **Linear growth in expert heads**: Each task retains a dedicated expert head, leading to parameter redundancy as the number of tasks scales.
4. **Potential limitations for long-horizon tasks**: Removing self-attention may constrain tasks requiring extended temporal reasoning; current evaluation covers only relatively short-horizon tabletop manipulation.
5. **Routing depends on initial observation**: Routing is performed solely from the $t=0$ observation; if the initial frame is not sufficiently discriminative, routing may fail.
6. **Future directions**: Online incremental mask updates; lightweight learnable routers to replace SVD; validation on large-scale VLMs; expert head compression or sharing.

## Related Work & Insights

- **vs. Task Arithmetic / TIES**: These methods are effective for LLMs/VLMs but fail completely for VLAs (0%); MergeVLA restores their applicability through architectural redesign.
- **vs. jointly trained multi-task VLAs** (OpenVLA, π₀): Joint training requires retraining on all data simultaneously, whereas MergeVLA merges weights entirely offline without access to the original training data.
- **vs. VLA-Adapter**: Self-attention in the action expert causes non-mergeability; MergeVLA replaces it with cross-attention and demonstrates superior performance.
- **vs. ReVLA**: ReVLA uses merging to address visual forgetting; MergeVLA uses merging to achieve multi-skill capability—the objectives are distinct.
- **Takeaway**: Model merging in the VLA domain is far from mature; architectural design has a decisive impact on downstream mergeability—a lesson with broad implications for the modular design of all multimodal models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic solution to VLA merging; diagnosis and design philosophy are both elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three simulation benchmarks + real robot + extensive ablations + OOD evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Diagnosis → solution → validation follows a rigorous logical arc with clear figures and tables.
- Value: ⭐⭐⭐⭐⭐ Provides a viable lightweight pathway for multi-skill scaling in embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)
- [\[ICLR 2026\] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](../../ICLR2026/robotics/from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[ICCV 2025\] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games](../../ICCV2025/robotics/combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)

</div>

<!-- RELATED:END -->
