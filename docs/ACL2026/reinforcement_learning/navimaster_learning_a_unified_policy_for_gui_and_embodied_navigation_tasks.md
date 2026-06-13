---
title: >-
  [Paper Note] NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks
description: >-
  [ACL2026][Reinforcement Learning][Unified Navigation Policy] NaviMaster reformulates both GUI operations and embodied navigation into a unified MDP of "visual target localization + action execution." It trains a Qwen2.5-…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "Unified Navigation Policy"
  - "GUI Agent"
  - "Embodied Navigation"
  - "GRPO"
  - "Dense Reward"
date: 2026-05-08
content_hash: 79c54675365fd8cb
---

# NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks

**Conference**: ACL2026  
**arXiv**: [2508.02046](https://arxiv.org/abs/2508.02046)  
**Code**: https://iron-boyy.github.io/navimaster-page/  
**Area**: Reinforcement Learning / Multi-modal Agents / GUI Navigation / Embodied Navigation  
**Keywords**: Unified Navigation Policy, GUI Agent, Embodied Navigation, GRPO, Dense Reward

## TL;DR
NaviMaster reformulates both GUI operations and embodied navigation into a unified MDP of "visual target localization + action execution." It trains a Qwen2.5-VL-7B policy using GRPO on hybrid trajectories with distance-aware dense rewards, outperforming single-domain training and mainstream baselines in OOD GUI tasks, spatial affordance prediction, and ObjectNav.

## Background & Motivation
**Background**: Both GUI agents and embodied navigation agents leverage multi-modal large models to "perceive images, understand instructions, and plan the next action." GUI tasks focus on operations like clicking, scrolling, and inputting on mobile, web, or desktop interfaces; embodied tasks focus on actions like turning, moving forward, and stopping for robots or agents in 3D environments. While seemingly different, both require the model to decide the next step based on the current egocentric observation, history, and natural language instructions.

**Limitations of Prior Work**: Past training frameworks were largely separated. GUI models are typically trained via SFT or RFT on datasets like GUI-Odyssey, AITW, and OmniAct; embodied models learn spatial localization and navigation on Matterport, Habitat, or RoboPoint. This leads to two issues: first, maintaining two separate models prevents the reuse of cross-task visuospatial capabilities; second, models easily learn shortcut correlations within a single dataset, resulting in poor generalization to OOD benchmarks.

**Key Challenge**: The primary difference between GUI and embodied navigation lies not in the "need for navigation" but in the expression of the action space. The core of GUI action is an explicit coordinate (e.g., clicking a pixel), while embodied action is often implicit movement (e.g., forward, turn, stop). As long as action spaces are unaligned, hybrid training treats the tasks as a loose collection rather than a unified policy problem.

**Goal**: The authors aim to answer three questions: whether GUI and embodied navigation can be unified into a single trajectory representation; whether both can be optimized within the same RL framework; and whether the sparsity of binary rewards can be mitigated by providing effective learning signals for rollouts that miss the target but remain close.

**Key Insight**: From an MDP perspective, both tasks share the same structure: states are egocentric observations, actions are interactions, and next states are determined by current states and actions. Both requires accumulating historical information to form an implicit allocentric spatial understanding. Thus, the authors use "visual targets" to reformulate embodied movement as localization of a target point within the image.

**Core Idea**: Use explicit visual targets to rewrite the embodied `MOVEFORWARD` into `MOVETO(x, y)`. This allows GUI clicks and embodied movements to share a pixel-level grounding representation, enabling the training of a unified navigation policy via GRPO and distance-aware rewards on hybrid data.

## Method
NaviMaster does not simply concatenate GUI and robot data. It first transforms both into comparable trajectories and then trains them under a single RL objective. The process involves three steps: constructing visual-target trajectories ("observe, think, act"); treating each step as a GRPO sample to output actions; and using rewards based on format, action type, and coordinate distance to refine precision.

### Overall Architecture
The input consists of a long-horizon task trajectory with instruction $I$, observations $o_i$, and actions $a_i$. GUI trajectories naturally contain screenshots and actions (click/scroll/type). For embodied navigation, where original data often contains only 3D poses, the authors use A* to find path points and project the next path point onto the current egocentric image, generating a visual target similar to a GUI click coordinate.

During training, each sample includes instruction $I$, history $H_i=\{(t_0,a_0),...,(t_{i-1},a_{i-1})\}$, and current observation $o_i$. The model outputs a `<think>...</think><answer>...</answer>` format, where the answer is an executable JSON action. Depth maps for GUI images are set to zero matrices, while embodied images use depth maps to constrain spatial grounding, preventing points that are close in 2D but distant in depth from being treated as equivalent.

### Key Designs
1. **Visual-target trajectories align dual action spaces**:
    - **Function**: Represents both GUI operations and embodied movements as target point selection or discrete action selection within the current visual observation.
    - **Mechanism**: Actions are divided into three types: fixed semantic actions (e.g., `BACK` in GUI, `STOP` in embodied); viewpoint adjustment actions (e.g., `SCROLL` in GUI, `TURN` in embodied); and localization actions. GUI already uses `CLICK(x, y)`, while embodied `MOVEFORWARD` is rewritten as `MOVETO(x, y)`, with coordinates projected from the next 3D path point.
    - **Design Motivation**: Standard embodied navigation learns "motion control," while GUI clicks learn "pixel grounding." NaviMaster merges them so that embodied data provides training signals for spatial localization needed by GUI, while GUI data trains affordance localization for embodied tasks.

2. **Unified GRPO training with historical reasoning**:
    - **Function**: Performs reinforcement learning directly on hybrid samples to train a unified policy capable of cross-task generalization.
    - **Mechanism**: At each step, the model samples $G$ candidate responses. GRPO calculates relative advantages within the group. The reward $R(i,j)$ measures the quality of a candidate relative to the ground truth. The advantage is approximated as $Adv=(R(i,j)-mean(R))/std(R)$. Like R1-Zero, this avoids a separate SFT stage and compares sample quality directly.
    - **Design Motivation**: Single-domain RL tends to overfit to specific action biases (e.g., high `CLICK` ratio in GUI). Hybrid training exposes the policy to a diverse set of MDPs, forcing it to learn generalized object permanence, spatial relations, and affordance grounding.

3. **Distance-aware dense rewards for sparse grounding**:
    - **Function**: Provides partial reward for rollouts that are "close to the target," improving RL efficiency.
    - **Mechanism**: Total reward is a weighted sum of format, action type, and grounding rewards: $R=\lambda_1R_F+\lambda_2R_T+\lambda_3R_G$. Grounding reward decays based on pixel distance: $R_G=(1-d_j/\theta_d)[d_j<\theta_d,p_j<\theta_h]$. $d_j$ is pixel distance and $p_j$ is depth difference.
    - **Design Motivation**: Methods like UI-R1 often use binary rewards (0 if not in the target zone), which lack gradient signals for large image coordinate prediction. Dense rewards signal that "closer is better," leading to faster training convergence.

### Loss & Training
The base model is Qwen2.5-VL-7B training on EasyR1. Training runs for 3 epochs on 8 NVIDIA A800 GPUs with a global batch size of 128. The learning rate decays linearly from $1e-6$ to 0. KL coefficient is 0.01, `num_generations=5`, max prompt length 7000, and max response length 1024. Reward weights are $\lambda_1=0.1, \lambda_2=1, \lambda_3=1$.

Total training data is 20k samples (10k GUI from GUI-Odyssey, 10k embodied from Matterport 3D and RoboPoint). GUI sub-sampling maintains the original action distribution (CLICK, SCROLL, TYPE). Trajectories exceeding 7000 tokens are truncated from the oldest history steps.

## Key Experimental Results
Experiments cover GUI navigation, spatial affordance prediction, and embodied navigation. Metrics include grounding rate (GR), success rate (SR), and Success weighted by Path Length (SPL).

### Main Results
Subsets highlighting the core conclusions:

| Task / Dataset | Metric | NaviMaster | Strong Baseline | Gain / Observation |
|--------|------|------|----------|------|
| GUI AC-Low | SR | 69.46 | UI-Shift 73.38 / GUI-R1 66.52 | Slightly below UI-Shift; exceeds GUI-R1 in OOD |
| GUI AC-High | SR | 55.89 | UI-Shift 52.16 / GUI-R1 51.56 | Outperforms mainstream RL GUI agents in OOD |
| GUI AITW | SR | 59.72 | GUI-R1 55.31 / UI-Shift 54.38 | Strong generalization on mobile tasks |
| GUI Llamatouch | SR | 67.39 | UI-AGILE 66.10 / GUI-R1 61.27 | Highest success in advanced operations |
| Odyssey (in-domain) | SR | 48.35 | Ours w/o Embodied 46.38 | Embodied data improves GUI source domain |

| Task / Dataset | Metric | NaviMaster | Comparison | Conclusion |
|--------|------|------|----------|------|
| RoboReflT | SR | 77.34 | RoboPoint-13B 49.82 | Significant lead in object referring |
| Where2Place | SR | 52.97 | RoboPoint-13B 46.77 | Free-space referring benefits from hybrid grounding |
| RefSpatial | SR | 19.49 | Ours w/o GUI 18.19 / RoboPoint-13B 8.40 | GUI data aids spatial relation generalization |
| ObjectNav (unseen) | SR | 33.20 | Qwen2.5-VL-7B 27.23 | Success rate increased by 5.97 points |
| ObjectNav (unseen) | SPL | 12.60 | Qwen2.5-VL-7B 9.68 | Path efficiency improved |

### Ablation Study

| Configuration | AC-High SR | AC-Low SR | Where2Place SR | RefSpatial SR | Note |
|------|---------|---------|---------|---------|------|
| NaviMaster | 55.89 | 69.46 | 52.97 | 19.49 | Main setup (1:1 hybrid, dense reward) |
| hard / sparse reward | 54.07 | 68.39 | 44.01 | 14.28 | Sparse reward severely hurts embodied grounding |
| 7k samples | 52.66 | 70.41 | 41.07 | 18.19 | Trainable with less data, but affordance drops |
| GUI Only | 52.44 | 69.37 | 47.99 | 16.49 | Good for flow, but worse than hybrid |
| $\lambda=(0.1,1,2)$ | 54.98 | 69.19 | 47.06 | 22.14 | Higher grounding weight helps RefSpatial |

### Key Findings
- Hybrid training is the primary source of gain. NaviMaster outperforms "w/o GUI" or "w/o Embodied" versions, suggesting that GUI pixel grounding and embodied spatial affordance provide complementary supervision rather than interference.
- Reward densification is critical for embodied grounding. Hard rewards caused Where2Place to drop from 52.97 to 44.01, proving that binary rewards cannot effectively train large-scale coordinate searches.
- A 1:1 data ratio is optimal. Among various ratios (1:9, 3:7, 5:5), the 5:5 split yielded the highest average SR.
- Absolute thresholds are more stable than relative ones. A threshold of 200 pixels prevents reward hacking in high-resolution images.

## Highlights & Insights
- The ingenious reformulation of "Forward" into "Move towards a point in the image" aligns 3D navigation and 2D GUI clicks into a single grounding problem, offering stronger unification than simple action vocabulary concatenation.
- NaviMaster is a complete training paradigm encompassing trajectory construction, action spaces, RL rewards, and benchmarks, rather than a single model trick.
- Dense rewards make intra-group relative advantages in GRPO more informative. It ranks failed samples, telling the model that "being closer to the target is better."
- Hybrid training improves spatial reasoning. Results on a custom visual-relation benchmark show the model better understands relative relations (e.g., "the second button above X"), likely reinforced by embodied 3D data.

## Limitations & Future Work
- Current trajectories still treat GUI and Embodied as separate tasks; the model does not yet handle interleaved tasks where GUI and physical navigation occur in the same trajectory.
- Rewriting embodied actions depends on projection and depth quality. Occlusions or camera noise can lead to unstable training labels.
- While ObjectNav improved, SR (33.20) still lags behind specialized systems, highlighting a lack of long-term mapping, exploration strategies, and closed-loop control.
- Safety constraints, execution latency, and recovery in real systems (OS or physical robots) have not been fully validated.

## Related Work & Insights
- **vs OS-Atlas / UI-TARS**: While these focus on large-scale GUI SFT for 2D grounding, NaviMaster emphasizes cross-domain spatial generalization via hybrid RL.
- **vs UI-R1 / GUI-R1**: NaviMaster adopts the RFT approach but replaces binary coordinates with distance-aware dense rewards and incorporates embodied tasks.
- **vs RoboPoint / SpaceLLaVA**: Unlike these, which focus on affordance prediction, NaviMaster integrates grounding into long-horizon navigation and uses GUI data to bolster spatial relation reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Clear unified formulation of GUI and Embodied navigation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive ablations, though real-world verification is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, though some table analysis is brief.
- Value: ⭐⭐⭐⭐⭐ High potential for training general multi-modal navigation agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Embodied Navigation with Auxiliary Task of Action Description Prediction](../../ICCV2025/reinforcement_learning/embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [\[ACL 2026\] Targeted Exploration via Unified Entropy Control for Reinforcement Learning](targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)
- [\[ACL 2026\] KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks](kaser_knowledge-aligned_student_error_simulator_for_open-ended_coding_tasks.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[AAAI 2026\] InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](../../AAAI2026/reinforcement_learning/infigui-g1_advancing_gui_grounding_with_adaptive_exploration_policy_optimization.md)

</div>

<!-- RELATED:END -->
