---
title: >-
  [Paper Note] Find, Fix, Reason: Context Repair for Video Reasoning
description: >-
  [ICML 2026][Multimodal VLM][Video Reasoning] Addressing the dilemma in video reasoning where "on-policy RL plateaus at capability ceilings and off-policy distillation suffers from entropy collapse…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Video Reasoning"
  - "GRPO"
  - "Tool-Integrated Teacher"
  - "Context Repair"
  - "Spatio-Temporal Dependency"
date: 2026-05-08
content_hash: b323dcd3487f24cd
---

# Find, Fix, Reason: Context Repair for Video Reasoning

**Conference**: ICML 2026  
**arXiv**: [2604.16243](https://arxiv.org/abs/2604.16243)  
**Code**: Yes (FFR, anonymous link)  
**Area**: Multimodal VLM / Video Reasoning / Reinforcement Learning  
**Keywords**: Video Reasoning, GRPO, Tool-Integrated Teacher, Context Repair, Spatio-Temporal Dependency

## TL;DR
Addressing the dilemma in video reasoning where "on-policy RL plateaus at capability ceilings and off-policy distillation suffers from entropy collapse," this work introduces a frozen, tool-integrated teacher model. When student rollouts fail, the teacher inserts minimal "evidence patches" (key-frame intervals, error types), prompting the student to re-answer the same question. These repaired trajectories are then incorporated into GRPO optimization through a chosen-rollout mechanism.

## Background & Motivation

**Background**: GRPO-based video reasoning LMMs (Video-R1, VideoRFT, VideoChat-R1) generally follow a pure on-policy RLVR route, where rewards are binary signals (0/1) based on answer correctness. While successful in textual mathematical reasoning, this paradigm struggles in video tasks due to complex spatio-temporal dependencies and sparse reasoning templates; self-exploration often plateaus and oscillates at the model's own capability limit.

**Limitations of Prior Work**: Three existing "breakthrough" routes have significant drawbacks: (a) Hybrid policies (LUFFY, Replay) insert strong teacher trajectories into the buffer, which mitigates entropy collapse but remains wholesale imitation requiring careful regularization; (b) Tool-integrated reasoners (Pixel-Reasoner, Video-Thinker) allow small models to call tools for evidence across multiple rounds, but are limited by the small model's tool-calling accuracy, often leading to "self-doubt" loops; (c) SFT teacher distillation is simple but compromises the student's on-policy exploration capabilities.

**Key Challenge**: The trade-off between the student model's capability ceiling and the distribution drift introduced by teacher guidance—the more information provided by the teacher, the more the student becomes a clone; too little information prevents the student from breaking through its ceiling. The essence of the problem lies in the *granularity of intervention*: should we modify the reward, the trajectory, or the observation?

**Goal**: To identify an "observation-level" intervention that modifies neither the task, the reward, nor the policy, but only the "evidence" seen by the student. This preserves on-policy characteristics while guiding exploration toward causal directions.

**Key Insight**: The authors observe that large models (GLM-4.5V) significantly outperform 7B student models in instruction following and tool usage. They can reliably diagnose "which spatio-temporal dependency caused the student's error" and quickly locate evidence using simple tools (frame ranges, object regions).

**Core Idea**: A frozen tool-integrated teacher acts as a "diagnostician," outputting a minimal evidence patch $c_i$ (e.g., "re-examine frames 13-17, focus on the color of the lifted object") for a student's failed rollout, **without directly revealing the answer**. The student re-attempts the question given the original problem + patch, and the repaired trajectory is included in GRPO updates as a chosen rollout.

## Method

### Overall Architecture
Each video-question pair $x=(v,q)$ undergoes a two-stage process: ① The student samples $G$ first-pass rollouts $\{\tau_i\}$ using $\pi_{\theta_{old}}$, which are scored by a verifier; ② For failed rollouts $\tau_i$ ($z_i=0$), a frozen teacher $\mathcal{T}$ outputs an error type $e_i\in\{$temporal, spatial, attribute, counting, dynamics, logic$\}$ and an evidence patch $c_i$. The student re-samples to obtain a repaired rollout $\tau_i^*$ using $\pi_{\theta_{old}}(\cdot|x,c_i)$. Correct rollouts $\tau_i$ are directly retained. Finally, "chosen rollouts" $\hat\tau_i$ ($z_i=1$ uses $\tau_i$, otherwise $\tau_i^*$) are used to calculate the token-level importance ratio for GRPO updates.

### Key Designs

1.  **Teacher Negative Constraint Strategy**:
    *   **Function**: Enables the teacher to diagnose errors and provide guidance without directly providing answer options or the final state.
    *   **Mechanism**: The teacher receives a data package $\mathcal{S}_i=(x,y,\tau_i)$ (with GT) or $(x,\tau_i)$ (without GT) and outputs $(e_i,c_i)$ via a carefully designed negative prompt + format constraint. For example, in a counting task, the teacher is forbidden from saying "there are exactly 3 people in frame 15" and can only say "re-count within the [13,17] frame interval." In temporal tasks, only frame intervals and event descriptions are provided, prohibiting hints about sequential relationships. Manual verification of 200 interactions showed the leakage rate dropped from 39.5% (unconstrained) to 0%.
    *   **Design Motivation**: Since traditional metrics struggle to cover all "soft leakage" edge cases in video tasks, the work leverages the teacher's ICL capabilities to separate "diagnosis" from "answer" via prompt engineering—forcing the student to **re-observe** rather than blindly apply an answer.

2.  **Chosen Rollout and Robust Improved Reward (RIR)**:
    *   **Function**: Ensures that within the GRPO framework, only the "adopted rollout" is backpropagated, while group-normalized advantage converts "successfully repaired" rollouts into effective gradient directions.
    *   **Mechanism**: After defining the chosen rollout $\hat\tau_i$, the scalar score for each sample is $\tilde R_i=z_i(R(\tau_i)+R_{fmt}(\tau_i))+(1-z_i)(R(\tau_i^*)+R_{fmt}(\tau_i^*)-\kappa)$, where $\kappa\ge 0$ is a "patch tax" used to penalize reliance on teacher patches. Group normalization within $G$ samples yields $A_i=(\tilde R_i-\text{mean})/\text{std}$, and updates are performed using the token-level ratio $r_{i,t}(\theta)$ within the PPO-clip framework.
    *   **Design Motivation**: Treating the "repaired trajectory" as an on-policy sample of the same policy under a different observation avoids the instability of off-policy importance sampling. A patch tax of $\kappa=0.3$ is found to be optimal—too small causes student over-reliance on the teacher, while too large negates teacher guidance.

3.  **Error Classification-Driven Tool Use**:
    *   **Function**: The teacher selects corresponding tools based on the error type (temporal/spatial/attribute/counting/dynamics/logic); the output evidence comprises a textual error class + optional visual context (frame indices, region masks).
    *   **Mechanism**: Each error type maps to a set of tools: temporal outputs frame intervals, spatial outputs region coordinates, attribute outputs object feature descriptions, etc. These signals are assembled into $c_i$ and injected into the student's prompt. Ablation studies show that removing visual context leads to a 10.0-point drop on Video-Holmes, while removing GT reference leads to a 7.6-point drop, proving these signals are complementary and critical.
    *   **Design Motivation**: Different root causes of errors require different granularities of repair signals; a uniform format (text-only) or homogeneous approach (visual-only) results in information loss. Error classification makes the "where to look" instruction concrete.

### Loss & Training
The GRPO objective is $\mathcal{J}_{FFR}(\theta)=\tfrac{1}{\sum|\hat\tau_i|}\sum_i\sum_{t\in\hat\tau_i}\text{CLIP}(r_{i,t}(\theta),A_i,\epsilon)-\beta D_{KL}[\pi_\theta\|\pi_{ref}]$, calculating loss only for tokens in chosen rollouts. Training utilized 4,000 samples, 8 rollouts/sample, 1 epoch, lr=5e-6, 8×A100, with GLM-4.5V as the default teacher.

## Key Experimental Results

### Main Results
Evaluation was conducted on 4 video reasoning benchmarks (MMVU/VSI-Bench/VideoMMMU/Video-Holmes) + 4 general video understanding benchmarks (LongVideoBench/LVBench/MVBench/TempCompass) against 7B student models.

| Baseline/Method | MMVU | VSI-Bench | Video-Holmes | LVBench |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 75.4 | 34.0 | 42.0 | 48.9 |
| GLM-4.5V (Teacher) | 68.7 | - | - | 53.8 |
| Video-R1 | 63.8 | 35.8 | 36.5 | 35.3 |
| **+ FFR** | **68.5** | **38.9** | **52.3** | **38.1** |
| Relative Gain | +11.75% | +22.33% | +51.16% | +24.10% |
| VideoRFT | 68.5 | 36.8 | 40.0 | 33.9 |
| **+ FFR** | **70.1** | **38.6** | **48.0** | **37.8** |

Notably, the 7B student outperformed GPT-4o by 10 points on Video-Holmes (causal narrative reasoning).

### Ablation Study

| Configuration | MMVU | Video-Holmes |
| :--- | :--- | :--- |
| vanilla GRPO | 60.3 | 45.6 |
| SFT + T-GRPO (Video-R1) | 63.8 | 36.5 |
| FFR (no visual context) | 64.4 | 42.3 |
| FFR (no GT reference) | 63.7 | 44.7 |
| FFR Full | **68.5** | **52.3** |
| SFT-Teacher 32B | 63.9 | 43.3 |
| SFT-Teacher 235B | 67.4 | 47.1 |
| FFR (teacher=32B) | **67.9** | **47.8** |
| FFR (teacher=235B) | **68.2** | **51.6** |

### Key Findings
- FFR with a 32B teacher (51.2 avg) outperformed SFT with a 235B teacher (50.7), suggesting that "targeted intervention" is far more data-efficient than "wholesale distillation."
- The intervention ratio decreased from 26.3% in early training to 13.7% in later stages, while accuracy rose from 77.5% to 80.2%—indicating the student internalized the teacher's diagnostic capability rather than becoming dependent.
- Error distribution: misconception (41.2%) > spatial (32%) > temporal (26.8%); students primarily struggle with "understanding what the question is asking" rather than "inability to see the image," aligning perfectly with FFR's intervention strategy.

## Highlights & Insights
- **"Observation-level intervention" is a genuine innovation**: Previous works modified rewards (downstream), trajectories (output), or policies (parameters). FFR only modifies what the student sees—the most lightweight yet targeted intervention.
- **Counter-intuitive finding: Teacher "diagnosis $\neq$ answering"**: The teacher does not need to answer correctly; they only need to diagnose where the student erred. This means "diagnostic-only" distillation can allow a small model to outperform its teacher, as evidenced by Qwen3-VL-8B + FFR beating the 32B teacher by 6 points on Video-Holmes.
- **The "patch tax" $\kappa$**: Using a scalar penalty for samples that are only correct "with the patch" forces the student to rank "independent correct answers" higher than "repaired correct answers" in the advantage sorting, cleverly transforming the tension between imitation and exploration into a tunable hyperparameter.

## Limitations & Future Work
- The teacher must be called for every failed rollout (including image understanding + tool calls), leading to high computational overhead during training. Using GLM-4.5V represents a Pareto-optimal cost-accuracy trade-off but remains several times more expensive than pure RLVR.
- There is no discussion of teacher bias—if a teacher systematically misdiagnoses certain problems, the student will be led astray. Leakage prevention relies solely on prompt engineering without mathematical guarantees.
- Experiments only covered the Qwen2.5-VL and Qwen3-VL series; portability to other architectures like InternVL or Llava remains unverified.
- While the "declining intervention ratio" is interpreted as "capability internalization," it could also stem from reward or data distribution drift; no trajectory probing experiments were conducted to distinguish these clearly.

## Related Work & Insights
- **vs LUFFY/Replay (hybrid policy)**: These works mix off-policy teacher trajectories directly into the buffer, requiring complex mixed regularization. FFR intervenes only at the observation level, maintaining on-policy characteristics and leading across all 8 benchmarks.
- **vs Pixel-Reasoner/Video-Thinker (tool-use reasoner)**: These allow students to call tools themselves across multiple rounds, but small models suffer from unstable tool-calling and self-doubt. FFR externalizes tool use to a large teacher, leaving the student to focus on "observing evidence and reasoning," a more rational division of labor.
- **vs SFT-Teacher**: SFT is wholesale imitation. FFR intervenes only upon failure and provides "where to look" rather than "what to answer"—FFR significantly outperforms SFT given the same teacher, a crucial comparison.
- **Insight**: Observation-level intervention can be extended to any scenario where "small models lack capability but large models excel at locating evidence," such as medical image diagnosis, code debugging, or agent task planning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The choice of "observation-level intervention" is highly novel and, combined with zero-leakage prompt design and chosen rollouts, constitutes a complete new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks × 2 base models × multiple teachers × cross-scale/cross-architecture ablations + analysis of intervention dynamics and error distribution; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 clearly compares the four paradigms; the case study (Figure 3) is instructional; the math sections are somewhat dense.
- Value: ⭐⭐⭐⭐⭐ The 7B student outperforming GPT-4o is highly practical, and the method can be seamlessly integrated into any GRPO training pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoVR-R: Reason-Aware Composed Video Retrieval](../../CVPR2026/multimodal_vlm/covr-rreason-aware_composed_video_retrieval.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[ICML 2026\] Vision Language Models Cannot Reason Physical Transformations](vision_language_models_cannot_reason_about_physical_transformation.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](../../CVPR2026/multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)

</div>

<!-- RELATED:END -->
