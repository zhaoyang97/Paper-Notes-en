---
title: >-
  [Paper Note] OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning
description: >-
  [ICLR 2026][Robotics & Embodied AI][Paper Note] OneTwoVLA unifies fast action execution and slow language reasoning within a single VLA. The model adaptively triggers reasoning using `[BOR]` during critical moments and outputs actions directly via `[BOA]` otherwise, significantly outperforming non-reasoning VLAs and dual-system approaches in long-horizon manipulatio
tags:
  - ICLR 2026
  - Robotics & Embodied AI
date: 2026-05-08
content_hash: 14ef995c01ffc07a
---
# OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=tWMfhoP3as](https://openreview.net/forum?id=tWMfhoP3as)  
**Code**: Not yet released  
**Area**: Robotics / Embodied AI  
**Keywords**: Vision-Language-Action (VLA) Model, Adaptive Reasoning, Long-horizon Manipulation, Human-Robot Interaction, Visual Grounding

## TL;DR
OneTwoVLA unifies fast action execution and slow language reasoning within a single VLA. The model adaptively triggers reasoning using `[BOR]` during critical moments and outputs actions directly via `[BOA]` otherwise, significantly outperforming non-reasoning VLAs and dual-system approaches in long-horizon manipulation, error recovery, human-robot interaction, and open-vocabulary visual grounding.

## Background & Motivation

**Background**: General-purpose robot control has recently evolved primarily along the line of vision-language-action (VLA) models, where models process multi-view images and language instructions to directly output continuous actions or action tokens. Another line of research draws inspiration from human dual-system cognition, treating high-level VLM/LLMs as a slow "System Two" to plan or decompose tasks, which are then executed by a low-level VLA or policy acting as "System One."

**Limitations of Prior Work**: Pure VLAs execute actions rapidly but tend to forget current progress during long-horizon tasks and struggle with error handling, ambiguity, or dynamic human instruction changes. Dual-system approaches enable explicit reasoning but introduce two practical issues: high-level models often lack knowledge of the low-level policy's specific capabilities, leading to unexecutable sub-tasks, and the high latency of VLM reasoning causes execution delays or reliance on outdated guidance if invoked at a fixed frequency.

**Key Challenge**: Robots require both "fast" and "slow" capabilities. Fast action execution demands low latency, closed-loop control, and proximity to sensor feedback, while slow language reasoning requires scene understanding, plan maintenance, history tracking, and anomaly handling. Forcing these into two separate models incurs high coordination costs, while removing reasoning entirely sacrifices long-range generalization.

**Goal**: The authors aim to develop a single-model policy capable of continuous action execution like System One during normal operation, while outputting natural language reasoning like System Two upon completing sub-tasks, detecting errors, or encountering human intervention—with the model itself deciding when to switch.

**Key Insight**: A key observation is that reasoning does not need to occur at every frame. Most of the time, the robot only needs to continue executing actions based on the most recent reasoning. Semantic state changes—such as starting a new step or encountering a failure—are the only points requiring scene re-description, plan updates, and decision-making. Thus, rather than "reasoning every step" or "never reasoning," the VLA should learn *when* to reason adaptively.

**Core Idea**: A unified VLA models decision tokens, language reasoning, and action chunks simultaneously. It is trained on robot data with embodied reasoning annotations alongside synthetic vision-language data, allowing the model to adaptively switch between "Reason" and "Act" modes during execution.

## Method

### Overall Architecture

The input to OneTwoVLA includes current multi-camera observations $I_t^{1:n}$, a reference image from the last reasoning step $I_{ref}^{1:n}$, a language instruction $\ell$, the latest reasoning content $R$, and the robot's proprioceptive state $s_t$ used during action mode. At each timestep, the model first outputs a decision token: if it is `[BOR]`, it enters reasoning mode to generate new textual reasoning; if it is `[BOA]`, it enters acting mode to generate and execute an action chunk $A_t$ based on the most recent reasoning.

Instead of concatenating a VLM planner and a VLA controller, the system integrates "choosing when to think" and "acting" into a single closed loop within one model. Reasoning content is stored as a state and reused during action execution; it is only refreshed when the model determines the reasoning is expired, the task stage has changed, an action has failed, or human input shifts the goal.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
	A["Current Obs + Ref Image<br/>Instruction + Latest Reason"] --> B["Adaptive Reason/Act Switch"]
	B -->|[BOR]| C["Embodied Reasoning State<br/>Scene/Plan/History/Next Step"]
	C --> A
	B -->|[BOA]| D["Unified VLA Action Generation<br/>Output Action Chunk"]
	D --> E["Robot Execution<br/>Update Env State"]
	E --> A
	F["Reasoning-Augmented Data Construction"] --> B
	G["Synthetic Embodied VL Data<br/>Enhances Generalization"] --> B
```

### Key Designs

**1. Adaptive Reason/Act Switching: Treating Slow Thinking as a Learnable Control Decision**

Traditional dual-system robots often use a fixed frequency for high-level reasoning calls (e.g., re-planning every fixed interval). OneTwoVLA instead learns to predict a decision token. Given current images, reference images, instructions, and recent reasoning $R$, if the model determines a semantic update is needed, it outputs beginning of reasoning `[BOR]`, followed by autoregressive reasoning text generation. If the recent reasoning is still relevant, it outputs beginning of action `[BOA]` and enters the action expert.

The core of this design is not just the tokens themselves, but framing "reasoning expiration" as an intermediate decision learned by the policy. For tasks like cooking or mixing drinks, the robot only needs to re-reason when an ingredient step is finished, a grasp fails, or an instruction changes; at other times, repeated language generation only slows down execution. Time analysis shows that OneTwoVLA's total task time is comparable to a non-reasoning flat VLA, whereas dual-system "always reasoning" schemes are significantly slower due to API latency.

**2. Embodied Reasoning State: Language as Scene Understanding, Planning, and Memory**

The reasoning content generated is not free-form chain-of-thought but a structured embodied state containing four types of information: scene description, global high-level plan, history summary, and current next step. Scene descriptions focus on task-relevant object locations; the plan lists steps; history summary tracks completed actions; and the next step compresses long-term goals into current intent.

This structure addresses two common VLA failure modes in long-horizon tasks. First, the model does not rely solely on implicit visual memory for progress, as it can read states like "Syrup and orange juice added, now add vodka." Second, the action expert receives a refined intent grounded in the current scene rather than just raw user instructions. The model also utilizes a reference image $I_{ref}^{1:n}$ from the last reasoning step as a historical anchor to prevent confusion between similar visual states.

**3. Reasoning-Augmented Robot Data Construction: Segmenting Demonstrations into Reasoning and Action Intervals**

Most robot datasets only contain observation-action pairs without indicating when to pause for task re-interpretation. The authors segment human demonstrations into two types of intervals: reasoning intervals (at sub-task completion, error points, or interaction triggers) annotated with reasoning text, and acting intervals where the model learns to execute actions based on previous reasoning. During training, if $R$ is expired in a reasoning interval, the decision token is supervised as `[BOR]`; once updated, it becomes `[BOA]`. Acting intervals always supervise `[BOA]`.

Reasoning annotations use a two-stage automated pipeline. First, given a plan $P=(p_1,\dots,p_K)$, $N=32$ frames are sampled to let Gemini 2.5 identify reasoning intervals. Second, four fields (description $D_j$, plan $P$, history $H_j$, next step $X_j$) are generated at the midpoint of each interval. Human evaluation on a "Tomato-Egg" task indicated that 81.5% of interval labels and 83.3% of scene descriptions were reasonable.

**4. Synthetic Embodied Vision-Language Data: Using Action-Free VL Samples for Generalization**

Robotic demonstrations are expensive and rarely cover diverse objects, spatial relations, and user intents. OneTwoVLA leverages its unified architecture to co-train on vision-language data without action labels. The authors use Gemini 2.5 Pro to generate diverse tabletop scene descriptions, then synthesize images using FLUX.1-dev with added fisheye distortion or gripper overlays to mimic robot perspectives. 

This pipeline generated 16,000 samples: 6,000 focused on visual grounding (spatial relations, attributes, semantic features) and 10,000 on long-horizon planning and human-robot interaction. While lacking action labels, these samples activate common sense and visual semantics from pre-trained VLMs, which transfer to the control policy via co-training.

### Loss & Training

OneTwoVLA is built on $\pi_0$ as the base VLA instance. The vision-language portion handles autoregressive text generation, supervised by cross-entropy for decision and reasoning tokens. The continuous action distribution follows $\pi_0$'s action expert, trained with a flow matching loss for action chunks. Formally, reasoning mode generates $\hat{R}\sim\pi_\theta(\cdot|I_t^{1:n}, I_{ref}^{1:n}, \ell, R)$ and acting mode generates $A_t\sim\pi_\theta(\cdot|I_t^{1:n}, I_{ref}^{1:n}, \ell, R, s_t)$.

The training data is a mixture of standard robot demonstrations, reasoning-augmented demonstrations, and 16,000 action-free synthetic embodied VL samples.

## Key Experimental Results

### Main Results

Real-robot experiments evaluated four capabilities: long-horizon planning, error detection/recovery, natural interaction, and visual grounding. Long-horizon tasks include Tomato-Egg, Hotpot, and Cocktail.

| Task / Setting | Metric | Ours (OneTwoVLA) | Baseline ($\pi_0$ / Dual) | Gain |
|----------------|--------|------------------|--------------------------|-------|
| Long-horizon Avg | Success Rate | 87% | $\pi_0$: 57% | +30 pts |
| Long-horizon Avg | Success Rate | 87% | Dual-System: 63% | +24 pts |
| Tomato-Egg | Success Rate | 85% | $\pi_0$: 70%, Dual: 55% | +15 / +30 pts |
| Hotpot | Success Rate | 80% | $\pi_0$: 50%, Dual: 70% | +30 / +10 pts |
| Cocktail | Success Rate | 95% | $\pi_0$: 50%, Dual: 65% | +45 / +30 pts |

OneTwoVLA's advantage is stable across all long-horizon tasks. Typical $\pi_0$ failures involved forgetting progress (e.g., repeatedly picking beef in Hotpot), while dual-system failures often stemmed from interface mismatches or high latency.

### Ablation Study

The authors compared the non-reasoning $\pi_0$, OneTwoVLA (robot data only), and OneTwoVLA-VL (including synthetic data).

| Configuration | Key Metrics | Description |
|---------------|-------------|-------------|
| $\pi_0$ flat VLA | Long-horizon 57%; Open-World grounding 3% | Fast execution but lacks reasoning/history; weak at long-horizon and semantics. |
| Dual-System VLA | Long-horizon 63%; Interaction 65% | Has high-level reasoning, but high/low levels are disconnected; updates are slow. |
| OneTwoVLA (Ours) | Long-horizon 87%; Interaction 100% | Unified model + adaptive reasoning significantly improves recovery and interaction. |
| OneTwoVLA-VL (Ours) | Open-World grounding 73%; Generalization 72.5% | Synthetic VL data significantly boosts generalization to unseen tasks and objects. |

### Key Findings

- **Adaptive reasoning balances efficiency and capability.** OneTwoVLA only reasons at critical junctions, maintaining task completion times close to the non-reasoning $\pi_0$, whereas dual-system approaches are much slower.
- **Explicit reasoning aids action learning.** The action MSE on the validation set for OneTwoVLA is 62% lower than $\pi_0$, proving that "next step + history" makes action prediction easier.
- **Synthetic VL data drives open-world generalization.** Without VL co-training, the model's success rate in open-world grounding was 8%, which jumped to 73% with co-training, even for items like "Sprite" or "GoPro" not present in the robot data.
- **Unified models excel at error recovery and interaction.** OneTwoVLA achieved 8/10 in error recovery and 100% success in interaction for drink/hotpot tasks, outperforming both baselines.

## Highlights & Insights

- Explicitly modeling "when to reason" as a token decision is a clever design. Most embodied reasoning methods use fixed workflows; OneTwoVLA makes reasoning frequency a learnable function of state, aligning with real-world efficiency constraints.
- The unified model's advantage lies in shared capability boundaries. Reasoning and actions stem from the same model and state, reducing the gap where a planner suggests something a controller cannot execute.
- The four-field reasoning format (Scene, Plan, History, Next Step) targets exactly where VLAs typically struggle in long-horizon tasks and is simple enough for automated pipeline generation.
- Synthetic VL data acts as a way to "activate" pre-trained VLM common sense within the robot policy context rather than just during pre-training.

## Limitations & Future Work

- Currently, reasoning intervals depend on human-defined key steps and automated labels. Future work could use reinforcement learning (RL) to optimize reasoning strategies based on success rates and latency.
- Even with sparse reasoning, the robot pauses for 2-3 seconds. For dynamic tasks, asynchronous reasoning and action generation would be preferable.
- As the unified model scales, the inference cost of the VLM backbone may become a bottleneck, requiring distillation or faster decoding.
- While effective, synthetic VL data still contains noise (roughly 20% error rate observed in a small sample); better filtering would be needed for larger scales.

## Related Work & Insights

- **vs. Flat VLAs (OpenVLA, $\pi_0$):** Flat VLAs are fast but lack explicit progress tracking. OneTwoVLA gains long-horizon and semantic capabilities with minimal latency trade-offs.
- **vs. Dual-System (ViLa, Hi Robot):** Separation leads to interface and latency issues. OneTwoVLA's unified approach ensures reasoning directly informs action generation.
- **vs. $\pi_0.5$:** OneTwoVLA's reasoning is more comprehensive (covering scene, plan, and history) and is generated adaptively rather than at every step.
- **Insight for Future Work:** The `[BOR]/[BOA]` mechanism could be expanded to finer-grained modes, such as "short update," "full re-plan," "ask human," or "tool call," making the VLA loop a learnable real-time scheduler.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Unifying adaptive reasoning/action switching in a VLA effectively solves the latency and interface issues of dual systems.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive real-robot tests across planning, recovery, interaction, and grounding.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure with intuitive examples and well-explained data pipelines.
- **Value**: ⭐⭐⭐⭐⭐ The principle that "reasoning need not happen every step but must occur online" is highly valuable for future embodied AI.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UniVLA: Unified Vision-Language-Action Model](unified_vision-language-action_model.md)
- [\[ICLR 2026\] Vlaser: Vision-Language-Action Model with Synergistic Embodied Reasoning](vlaser_vision-language-action_model_with_synergistic_embodied_reasoning.md)
- [\[ICLR 2026\] HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model](hybridvla_collaborative_diffusion_and_autoregression_in_a_unified_vision-languag.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/robotics/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[ICLR 2026\] Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process](unified_diffusion_vla_vision-language-action_model_via_joint_discrete_denosing_d.md)

</div>

<!-- RELATED:END -->
