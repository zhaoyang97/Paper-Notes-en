---
title: >-
  [Paper Note] Learning GUI Grounding with Spatial Reasoning from Visual Feedback
description: >-
  [ICML 2026][Multimodal VLM][GUI grounding] This work reframes GUI grounding from "one-step coordinate prediction" into an interactive search of "moving a cursor on the screen to find the target." By employing a dense rew…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "GUI grounding"
  - "virtual cursor"
  - "visual feedback"
  - "GRPO"
  - "spatial reasoning"
date: 2026-05-08
content_hash: dcbd51e75ac0aa79
---

# Learning GUI Grounding with Spatial Reasoning from Visual Feedback

**Conference**: ICML 2026  
**arXiv**: [2509.21552](https://arxiv.org/abs/2509.21552)  
**Code**: None  
**Area**: LLM Agent / GUI Grounding / Multimodal VLM / Reinforcement Learning  
**Keywords**: GUI grounding, virtual cursor, visual feedback, GRPO, spatial reasoning

## TL;DR
This work reframes GUI grounding from "one-step coordinate prediction" into an interactive search of "moving a cursor on the screen to find the target." By employing a dense reward with trajectory penalties and GRPO training, the VLM learns to align numerical coordinates with screen positions using visual feedback from rendered cursors. Using only 8K samples, it improves performance on ScreenSpot-Pro from GTA1's 50.1% to 58.1%.

## Background & Motivation

**Background**: GUI agents are typically divided into a planner and a grounding component. The latter maps instructions like "click the login button" to pixel coordinates $(x,y)$. Most current grounding models (SeeClick, UGround, OS-Atlas, UI-TARS, GUI-Actor, and RL methods like GUI-R1, SE-GUI, GUI-G2, GTA1) treat this as **single-step coordinate regression**, predicting a pair of numbers directly from a screenshot.

**Limitations of Prior Work**: VLMs perform poorly at predicting precise numerical coordinates on high-resolution or complex layouts; 7B-scale SOTA models only reach approximately 50% on ScreenSpot-Pro. The authors attribute this to the **spatial semantic alignment** problem: models must map their semantic understanding of visual elements to discrete coordinate tokens through implicit mapping. During training, however, models only see their output numbers and **never see where those numbers actually land on the screen**.

**Key Challenge**: Training signals cover "outputs" but not "impact points." Without a visual feedback loop, the alignment between numerical predictions and screen pixels is inherently fragile, leading to failures in out-of-distribution or high-resolution scenarios.

**Goal**: To allow the model to see "where my previous click landed" during training to stabilize coordinate-pixel alignment, and to enable a human-like "look again before confirming" mechanism during inference.

**Key Insight**: Humans do not find objects on a screen in a single leap; they move the mouse, check the alignment, and adjust. The authors explicitly model this as an interactive environment: each step renders the current prediction as a **virtual cursor** on the screenshot. The model decides the next move or outputs STOP based on the target appearance, current cursor position, and their spatial relationship.

**Core Idea**: Rewrite GUI grounding from "one-step regression of $(x,y)$" to "multi-step search + rendered cursor visual feedback + trajectory-level RL." Optimize this policy using GRPO and implement cursor-centric focusing for high-resolution usability.

## Method

### Overall Architecture
Input: Screenshot $O$ ($W\times H$) + natural language instruction $I$. Output: Pixel coordinates $(x,y)$ within target box $B$.

Gui-Cursor decomposes this into an interactive episode of at most $T$ steps:

1. Initialization: At $t=0$, the cursor is placed at the screen center $(x_0,y_0)$ to obtain $O_0$.
2. Multi-step Interaction: At each step, the model generates $A_t = \langle\text{think}\rangle\ldots\langle/\text{think}\rangle\langle\text{answer}\rangle\ldots\langle/\text{answer}\rangle$ based on the history $(I, O_0, A_0, O_1, \ldots, A_{t-1}, O_t)$. The `<think>` block explicitly describes the target's appearance, the current cursor position, and their spatial relationship. The `<answer>` block provides either new absolute coordinates $(x_t,y_t)$ or `STOP`.
3. Feedback Rendering: If new coordinates are provided, the environment moves the cursor to that position in the next frame $O_{t+1}$ and feeds it back. If `STOP` is called or the maximum steps are reached, the episode ends, and the last cursor position is the final prediction.

Training utilizes GRPO with Qwen2.5-VL-7B and UI-TARS-1.5-7B as base models. The maximum training steps are 250, with up to 4 moves per episode. 12 trajectories are sampled per instruction with a batch size of 32 and a learning rate of $10^{-6}$. Data consists of an 8K sample subset of Aria-UI + OS-Atlas (compared to GTA1's 64K), with online filtering to remove "all-correct/all-wrong" samples.

### Key Designs

1. **Interactive Cursor Search + Visual Feedback Loop**:
    - **Function**: Transitions from single-step regression to multi-step interaction. Each step renders the current prediction as a visible cursor, allowing the model to observe its previous action.
    - **Mechanism**: The VLM controls a virtual cursor, using the think block for spatial reasoning and the answer block for coordinate output. Inference supports adaptive steps: simple samples are resolved in one step, while difficult samples (small icons) trigger multiple moves. Statistics show 99.5% of ScreenSpot-v2 samples use one step, while 9.4% of ScreenSpot-Pro samples use multiple steps, where the average target area (5024 px) is much smaller than single-step targets (31584 px).
    - **Design Motivation**: To incorporate the missing "visualize own output" link in training, forcing the model to align coordinates in pixel space rather than just token space.

2. **Trajectory-level Dense Rewards (Position Reward + 4 Trajectory Penalties)**:
    - **Function**: Rewards reaching the target center while penalizing pathological search behaviors.
    - **Mechanism**: The position reward uses the dense format from SE-GUI: $r_p = 1 + (1 - d_{\text{centre}}/d_{\max})^2$ if inside box $B$ (higher closer to center), and $r_p = 1 - d_{\text{edge}}$ if outside (decaying by distance to nearest edge). All distances are normalized. Four binary penalties are added: False Stop (STOPS outside $B$), False Move (was inside $B$ but moved out), False Direction (final position is further from $B$ than the first step), and Repeated Position (same coordinates appear $\geq 2$ times). Total reward $R_T = r_p - w_p(r_{\text{FD}} + r_{\text{FS}} + r_{\text{FM}} + r_{\text{RP}})$, plus a format reward, optimized via GRPO.
    - **Design Motivation**: Multi-step interaction easily degrades into "always STOP" or oscillating between points. Trajectory-level negative feedback is necessary to rectify search behavior. Ablations show the False Stop penalty is essential for the model to learn multi-step movement.

3. **Cursor-Centric Focusing (CCF) — Training-Inference Resolution Decoupling**:
    - **Function**: Reduces training compute via lower resolution while maintaining inference accuracy on high-res screens via cropping.
    - **Mechanism**: During training, all images are scaled to not exceed $P = 1920\times 1080$ to prevent memory overflow from history stacking. During inference, if the image exceeds $P$, an initial coarse prediction is made. The model then crops a $P$-sized region centered on this prediction, places the cursor in the center of the crop, and performs subsequent moves within this crop without retaining the original large image in history.
    - **Design Motivation**: The iterative pipeline faces a bottleneck where history length $\times$ image size increases linearly. Decoupling resolutions stabilizes training while serving high-res screens. This technique also improved GTA1's performance on ScreenSpot-Pro from 50.1% to 54.0%.

### Loss & Training
GRPO optimizes $R_T$ plus format rewards. 12 trajectories are sampled per instruction for group comparison. Online filtering removes instructions where all samples succeed or fail to ensure a learning signal. Max interaction steps: 4. Max training steps: 250 gradient steps.

## Key Experimental Results

### Main Results

| Dataset | Metric | GUI-Cursor (UI-TARS-1.5-7B) | Prev. SOTA (Equivalent 7B) | Gain |
|--------|------|---------|---------|------|
| ScreenSpot-Pro | Avg. Acc | **58.1** | GTA1-7B 50.1 | **+8.0** |
| ScreenSpot-v2 | Avg. Acc | **93.9** | GUI-G2-7B 93.3 | +0.6 |
| OSWorld-G | Avg. Acc | 65.6 | GTA1-7B 67.7 | -2.1 |
| UI-Vision | Avg. Acc | **27.3** | GTA1-7B 26.2 | +1.1 |
| OSWorld (online, w/ o3 planner, 50 steps) | Success Rate | **57.1** (Qwen2.5-VL-7B base) | GTA1-7B 53.1 (100 steps) | +4.0, half steps |
| SpatialMQA (OOD Spatial Reasoning) | Acc | **43.4** | Qwen2.5-VL-7B base 38.1 | +5.3 |

Highlights: Surpassed GTA1's 64K sample setting using only 8K training samples, an ~8$\times$ data efficiency improvement.

### Ablation Study

| Configuration | ScreenSpot-Pro | Description |
|------|---------|------|
| Full Gui-Cursor (w/ ccf) | 56.5 | Qwen2.5-VL-7B base |
| w/o ccf | 45.3 | -11.2; CCF is vital for high-res screens |
| w/o False Stop penalty | Significant drop | Multi-step rate 9.4% $\to$ 0.1%; collapses to single-step |
| w/o Repeated Position penalty | Unstable after 220 steps | Multi-step rate 9.4% $\to$ 16.6%; oscillates between points |
| w/o False Move / False Direction | All weaker than Full | Every penalty contributes to accuracy |
| w/o thinking tokens | Significant drop | Contradicts single-step findings from GUI-G2/GTA1 |
| ScreenSpot-v2 w/o ccf | 93.6 $\to$ 93.9 | Minimal CCF gain in simple scenarios |
| Applying CCF to GTA1 | 50.1 $\to$ 54.0 | CCF serves as a general enhancer |

### Key Findings
- **Visual feedback loops are effective**: Thinking + multi-step feedback only becomes powerful in interactive training. While "thinking" was largely useless in single-step RL (GTA1), removing it from Gui-Cursor causes a major performance drop, suggesting spatial reasoning requires seeing the action's result to be realized.
- **Zero-shot comparisons are convincing**: Testing unmodified Qwen2.5-VL-7B on "moving the cursor" saw accuracy drop from 88.8% (single-step) to 36.3% (multi-step absolute) or 1.3% (relative offsets). GPT-4o improved from 17.5% to 25.5%. This indicates current grounding scores rely on "guessing coordinates" rather than robust spatial reasoning.
- **Adaptive steps are intuitive**: 99.5% of ScreenSpot-v2 samples are one-step; 9.4% of ScreenSpot-Pro samples are multi-step. Multi-step targets are significantly smaller (5024 px vs 31584 px), showing the model learned to "look closer at small objects."
- **Spatial reasoning spillover**: In a cursor-in-box binary task, the base model showed severe center bias. Gui-Cursor maintained high F1 even at edges without specific training. Improvements in OOD benchmarks like SpatialMQA/SPHERE suggest the "move cursor" training captures generalized spatial reasoning.

## Highlights & Insights
- **Small reframing, large leverage**: Moving from "single-step regression" to "interactive search" simply fixes the "blind output" training signal. This task-level modification is more efficient than scaling data or architectures.
- **Trajectory rewards matter more than position rewards**: The four binary penalties specifically target pathological behaviors (early stop, backtracking, repetition, reverse direction). This provides a "lesson list" for future multi-step RL.
- **Resolution decoupling is a classic systemic trick**: CCF (downsampling for training, cropping for inference) allows for stable training and high-resolution performance. This is easily extensible to video or manipulation tasks.
- **8$\times$ data efficiency**: Beating GTA1 with 1/8 the data suggests optimizing task formulation is prioritized over data scaling.

## Limitations & Future Work
- **Ours' dependence**: Still relies on an external planner (o3); the model does not do high-level planning. CCF has negligible benefits in simple scenarios; multi-step's value is primarily evidenced in difficult tasks like ScreenSpot-Pro.
- **Observations**: (1) The "multi-step" is still shallow (1-2 steps), functioning more like "single-shot + verify" rather than a long-horizon search. (2) Trailing behind GTA1 by 2.1% on OSWorld-G indicates cursor feedback is not a universal solution for all tasks. (3) Lack of scaling experiments on 32B+ models. (4) Increased inference latency due to multiple rendering passes.
- **Future Directions**: Expanding feedback to region highlighting/thumbnail navigation; introducing "zoom-in" actions; joint training with a planner; transitioning to non-GUI visual localization (e.g., medical imaging).

## Related Work & Insights
- **vs GTA1**: Both use GRPO. GTA1 uses 64K samples and high-res training ($4096\times 2160$). Gui-Cursor uses 8K samples, low-res training ($1920\times 1080$), and CCF. It leads by 8% on ScreenSpot-Pro.
- **vs SE-GUI / GUI-G2**: These are single-step RL with simpler rewards. Gui-Cursor adopts the dense reward $r_p$ but adds trajectory penalties to support the multi-step paradigm.
- **vs iterative cropping methods (GUI-Spotlight, etc.)**: These also use cropping but remain single-step tasks repeated. Gui-Cursor integrates the cursor as a training signal within RL.
- **vs GUI-Actor**: GUI-Actor uses architecture changes and a two-stage approach; Gui-Cursor uses RL + visual feedback to train alignment directly without extra parameters.
- **Insight**: Visual feedback of actions is a cross-domain concept. Any agent task where the "landing point" of an action can be rendered can benefit from this paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ Reframing the task + trajectory reward combo is a clear innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducted across 4 grounding benchmarks, OSWorld, spatial reasoning benchmarks, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and well-documented formulas.
- Value: ⭐⭐⭐⭐⭐ Extremely reproducible for those wanting to train GUI agents with minimal data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICML 2026\] Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-.md)
- [\[ICML 2026\] ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)
- [\[CVPR 2026\] Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation](../../CVPR2026/multimodal_vlm/training_high-level_schedulers_with_execution-feedback_reinforcement_learning_fo.md)
- [\[ICML 2026\] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)

</div>

<!-- RELATED:END -->
