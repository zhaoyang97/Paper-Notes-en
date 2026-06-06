---
title: >-
  [Paper Note] Find, Fix, Reason: Context Repair for Video Reasoning
description: >-
  [ICML 2026][Video Understanding][Video Reasoning] This work addresses the dilemma in video reasoning where "on-policy RL stagnates at a capability ceiling…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Video Reasoning"
  - "GRPO"
  - "Tool-Integrated Teacher"
  - "Context Repair"
  - "Spatiotemporal Dependency"
date: 2026-05-08
content_hash: a473d5e2c9ed6f36
---

# Find, Fix, Reason: Context Repair for Video Reasoning

**Conference**: ICML 2026  
**arXiv**: [2604.16243](https://arxiv.org/abs/2604.16243)  
**Code**: Available (FFR, anonymous link)  
**Area**: Multimodal VLM / Video Reasoning / Reinforcement Learning  
**Keywords**: Video Reasoning, GRPO, Tool-Integrated Teacher, Context Repair, Spatiotemporal Dependency

## TL;DR
This work addresses the dilemma in video reasoning where "on-policy RL stagnates at a capability ceiling, while off-policy distillation leads to entropy collapse." It introduces a frozen, tool-integrated large teacher model that inserts minimal "evidence patches" (key-frame intervals, error types) when the student fails during rollout. The student then re-answers the same question, and the repaired trajectory is incorporated into GRPO optimization via a chosen-rollout mechanism.

## Background & Motivation

**Background**: Video reasoning LMMs based on GRPO (Video-R1, VideoRFT, VideoChat-R1) generally follow a pure on-policy RLVR approach, where the reward is a binary 0/1 signal verifying answer correctness. While this paradigm has succeeded in textual mathematical reasoning, the complex spatiotemporal dependencies and lack of reasoning templates in video make self-exploration prone to hitting the model's own capability ceiling, resulting in stagnation.

**Limitations of Prior Work**: Three existing "breakthrough" approaches each have critical flaws: (a) Hybrid policy (LUFFY, Replay) mixes strong teacher trajectories into the buffer, alleviating entropy collapse but still relying on wholesale imitation and requiring careful regularization; (b) Tool-integrated reasoners (Pixel-Reasoner, Video-Thinker) let small models call tools for evidence retrieval, but their tool-use accuracy limits their upper bound, often causing "self-doubt" loops; (c) SFT teacher distillation is simple but causes the student to lose on-policy exploration capability.

**Key Challenge**: The trade-off between the student's capability ceiling and distribution shift induced by teacher guidance—the more the student sees, the more it becomes a copy of the teacher; too little, and it cannot break its own ceiling. The core issue is the *granularity of intervention*: should one modify the reward, the trajectory, or the observation?

**Goal**: To find an "observation-level" intervention that does not alter the task, reward, or policy, but only modifies the "evidence" the student sees, thus preserving on-policy characteristics while guiding exploration in a causal direction.

**Key Insight**: The authors observe that large models (GLM-4.5V) are much stronger than 7B students in instruction following and tool use, reliably diagnosing "which spatiotemporal dependency the student failed on" and quickly locating evidence via simple tools (frame range, object region).

**Core Idea**: The frozen, tool-integrated large teacher acts as a "diagnostician," outputting a minimal evidence patch $c_i$ (e.g., "re-examine frames 13-17, focus on the color of the lifted object") for failed student rollouts, but **without directly providing the answer**. The student then re-answers under the original question plus the patch, and the repaired trajectory is used as the chosen rollout for GRPO updates.

## Method

### Overall Architecture
Each video-question pair $x=(v,q)$ proceeds in two stages: (1) The student samples $G$ first-pass rollouts $\{\tau_i\}$ using $\pi_{\theta_{old}}$, which are scored by a verifier; (2) For failed $\tau_i$ ($z_i=0$), the frozen teacher $\mathcal{T}$ outputs an error type $e_i\in\{$temporal, spatial, attribute, counting, dynamics, logic$\}$ and evidence patch $c_i$. The student then resamples a repaired rollout $\tau_i^*$ using $\pi_{\theta_{old}}(\cdot|x,c_i)$. Correct $\tau_i$ are retained as is. The final "chosen rollout" $\hat\tau_i$ ($z_i=1$ uses $\tau_i$, otherwise $\tau_i^*$) is used to compute token-level importance ratios for GRPO updates.

### Key Designs

1. **Zero-Leakage Teacher Evidence Patch (Teacher Negative Constraint Strategy)**:

    - **Function**: Enables the teacher to diagnose errors and provide guidance, but never directly gives answer options or end states.
    - **Mechanism**: The teacher receives a data packet $\mathcal{S}_i=(x,y,\tau_i)$ (with GT) or $(x,\tau_i)$ (without GT), and outputs $(e_i,c_i)$ via carefully designed negative prompts and format constraints. For example, in counting tasks, the teacher cannot say "there are exactly 3 people in frame 15," but only "please recount in frames [13,17]"; for temporal tasks, only frame intervals and event descriptions are given, with no hints about order. Manual verification of 200 interactions reduced leakage from 39.5% (unconstrained) to 0%.
    - **Design Motivation**: Traditional metrics cannot cover all "soft leakage" edge cases in video tasks, so the teacher's ICL ability is leveraged, using prompt engineering to separate "diagnosis" from "answer"—forcing the student to **re-observe** rather than copy the answer.

2. **Chosen Rollout and Robust Improved Reward (RIR)**:

    - **Function**: In the GRPO framework, only the "adopted rollout" is backpropagated, while group-normalized advantage turns "successfully repaired" rollouts into effective gradient directions.
    - **Mechanism**: After defining the chosen rollout $\hat\tau_i$, each sample's scalar score is $\tilde R_i=z_i(R(\tau_i)+R_{fmt}(\tau_i))+(1-z_i)(R(\tau_i^*)+R_{fmt}(\tau_i^*)-\kappa)$, where $\kappa\ge 0$ is a patch tax penalizing reliance on teacher patches. Group normalization within $G$ samples yields $A_i=(\tilde R_i-\text{mean})/\text{std}$, and token-level ratio $r_{i,t}(\theta)$ is used for PPO clip updates.
    - **Design Motivation**: Treats "repaired trajectories" as on-policy samples under different observations for the same policy, avoiding instability from off-policy importance sampling; patch tax $\kappa=0.3$ is optimal—too small leads to over-reliance on the teacher, too large negates teacher guidance.

3. **Alternating Error-Type-Driven Tool Use**:

    - **Function**: The teacher selects corresponding tools based on error type (temporal/spatial/attribute/counting/dynamics/logic), outputting textual error class plus optional visual context (frame indices, region masks).
    - **Mechanism**: Each error type corresponds to a set of tools: temporal outputs frame intervals, spatial outputs region coordinates, attribute outputs object feature descriptions, etc. These signals are assembled into $c_i$ and injected into the student's prompt. Ablation shows: removing visual context drops Video-Holmes by 10.0 points; removing GT reference drops by 7.6 points, proving both signals are complementary and critical.
    - **Design Motivation**: Different error causes require different granularity of repair signals; unified format (only text) or homogeneous (only visual) loses information; error classification concretizes "where to look."

### Loss & Training
The GRPO objective is $\mathcal{J}_{FFR}(\theta)=\tfrac{1}{\sum|\hat\tau_i|}\sum_i\sum_{t\in\hat\tau_i}\text{CLIP}(r_{i,t}(\theta),A_i,\epsilon)-\beta D_{KL}[\pi_\theta\|\pi_{ref}]$, with loss computed only on tokens of the chosen rollout. Training uses 4000 samples, 8 rollouts/sample, 1 epoch, lr=5e-6, 8×A100, with GLM-4.5V as the default teacher.

## Key Experimental Results

### Main Results
On 4 video reasoning (MMVU/VSI-Bench/VideoMMMU/Video-Holmes) and 4 general video understanding (LongVideoBench/LVBench/MVBench/TempCompass) benchmarks, 7B student models are compared.

| Baseline/Method | MMVU | VSI-Bench | Video-Holmes | LVBench |
|--------|------|------|------|------|
| GPT-4o | 75.4 | 34.0 | 42.0 | 48.9 |
| GLM-4.5V (Teacher) | 68.7 | - | - | 53.8 |
| Video-R1 | 63.8 | 35.8 | 36.5 | 35.3 |
| **+ FFR** | **68.5** | **38.9** | **52.3** | **38.1** |
| Gain | +11.75% | +22.33% | +51.16% | +24.10% |
| VideoRFT | 68.5 | 36.8 | 40.0 | 33.9 |
| **+ FFR** | **70.1** | **38.6** | **48.0** | **37.8** |

Most notably, on Video-Holmes (causal narrative reasoning), the 7B student surpasses GPT-4o by 10 points.

### Ablation Study

| Configuration | MMVU | Video-Holmes |
|------|---------|------|
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
- FFR + 32B teacher (51.2 avg) already surpasses SFT + 235B teacher (50.7), indicating that "targeted intervention" is much more data-efficient than "full distillation."
- The intervention ratio drops from 26.3% early in training to 13.7% later, while accuracy rises from 77.5% to 80.2%—the student internalizes the teacher's diagnostic ability, rather than "cheating" by dependence.
- Error distribution: misconception 41.2% > spatial 32% > temporal 26.8%; students mainly err in "understanding the question" rather than "failing to see the image," aligning with FFR's intervention strategy.

## Highlights & Insights
- **"Observation-level intervention" is the true innovation**: Previous work either modified the reward (downstream), trajectory (output), or policy (parameters); FFR only changes what the student sees—this is the lightest yet most targeted intervention.
- **Counterintuitive finding: teacher "diagnosis ≠ answering"**: The teacher need not answer correctly, only diagnose where the student failed. This enables "diagnostic-only" distillation, allowing small students to surpass the teacher; e.g., Qwen3-VL-8B + FFR outperforms the 32B teacher by 6 points on Video-Holmes.
- **Patch tax $\kappa$ detail**: A scalar penalty for "correct only after patch" samples, forcing the student to rank "independent correct" above "teacher-assisted correct" in advantage sorting, cleverly turning the tension between imitation and exploration into a tunable hyperparameter.

## Limitations & Future Work
- The teacher must be called for every failed rollout (including image understanding and tool use), making training computationally expensive; using GLM-4.5V is a Pareto-optimal cost-accuracy trade-off, but still several times costlier than pure RLVR.
- No discussion of what happens if the teacher itself is biased—for example, systematic misdiagnosis in certain question types could mislead the student; leakage prevention relies solely on prompt engineering, with no mathematical guarantee.
- Experiments only cover Qwen2.5-VL and Qwen3-VL series; transferability to other architectures like InternVL, Llava, etc., is unverified.
- The decline in "intervention ratio" is interpreted as "capability internalization," but could also be due to reward or data distribution drift; no trajectory probing experiments to rigorously distinguish these factors.

## Related Work & Insights
- **vs LUFFY/Replay (hybrid policy)**: These directly mix off-policy teacher trajectories into the buffer, requiring careful regularization; FFR intervenes only at the observation level, preserving on-policy nature, and outperforms on all 8 benchmarks.
- **vs Pixel-Reasoner/Video-Thinker (tool-use reasoner)**: These let students call tools themselves, but small models' tool use is unstable and often self-doubt; FFR outsources "tool use" entirely to the large teacher, with the student only "observing evidence and reasoning," making the division of labor more rational.
- **vs SFT-Teacher**: SFT is wholesale imitation; FFR intervenes only on failures, and only provides "where to look" rather than "what to answer"—under the same teacher, FFR far outperforms SFT, a crucial comparison.
- **Insights**: Observation-level intervention can be generalized to any scenario where "small models lack capability on certain problems, large models excel at evidence localization," such as medical image diagnosis, code debugging, or agent task planning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The choice of "observation-level intervention" is highly innovative, and together with zero-leakage prompt design and chosen rollout, forms a complete new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks × 2 base models × multiple teachers × cross-scale/cross-architecture ablation + intervention dynamics analysis + error distribution analysis, very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 clearly contrasts the four paradigms, and the case study (Figure 3) is instructive; the formula section is somewhat dense.
- Value: ⭐⭐⭐⭐⭐ The result of a 7B student surpassing GPT-4o is highly practical, and the method can be seamlessly integrated into any GRPO training pipeline.

## Related Papers

- [\[CVPR 2026\] CoVR-R: Reason-Aware Composed Video Retrieval](../../CVPR2026/multimodal_vlm/covr-rreason-aware_composed_video_retrieval.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](../../CVPR2026/multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[ICLR 2026\] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](../../ICLR2026/multimodal_vlm/vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)
- [\[ICML 2025\] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](../../ICML2025/multimodal_vlm/bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](../../AAAI2026/video_understanding/reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)
- [\[ICLR 2026\] Video-KTR: Enhancing Video Reasoning via Key Token Attribution](../../ICLR2026/video_understanding/video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](../../CVPR2026/video_understanding/cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](../../CVPR2026/video_understanding/videoauto-r1_video_auto_reasoning_via_thinking_once_answering_twice.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](../../ACL2026/video_understanding/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)

</div>

<!-- RELATED:END -->
