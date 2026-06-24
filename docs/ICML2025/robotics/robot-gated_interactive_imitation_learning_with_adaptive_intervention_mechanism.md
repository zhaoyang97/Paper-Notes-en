---
title: >-
  [Paper Note] Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism
description: >-
  [ICML2025][Robotics][Interactive Imitation Learning] This paper proposes the Adaptive Intervention Mechanism (AIM), which learns a proxy Q-function to simulate human intervention decisions, allowing the robot to proactively request expert assistance. Compared to the uncertainty-based baseline Thrifty-DAgger, AIM reduces human takeover costs and improves learning efficiency by 40%.
tags:
  - "ICML2025"
  - "Robotics"
  - "Interactive Imitation Learning"
  - "Robot-Gated Intervention"
  - "Proxy Q-function"
  - "Adaptive Mechanism"
  - "Human-in-the-Loop"
date: 2026-05-08
content_hash: bfb18e28ab8163c1
---

# Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism

**Conference**: ICML2025  
**arXiv**: [2506.09176](https://arxiv.org/abs/2506.09176)  
**Code**: [metadriverse/AIM](https://github.com/metadriverse/AIM)  
**Area**: Imitation Learning / Interactive Imitation Learning  
**Keywords**: Interactive Imitation Learning, Robot-Gated Intervention, Proxy Q-function, Adaptive Mechanism, Human-in-the-Loop

## TL;DR

This paper proposes the Adaptive Intervention Mechanism (AIM), which learns a proxy Q-function to simulate human intervention decisions, allowing the robot to proactively request expert assistance. Compared to the uncertainty-based baseline Thrifty-DAgger, AIM reduces human takeover costs and improves learning efficiency by 40%.

## Background & Motivation

Interactive Imitation Learning (IIL) allows agents to receive online human corrective demonstrations during training, which can be categorized into two paradigms:

- **Human-gated IIL**: Humans monitor the entire process and proactively intervene (e.g., HG-DAgger, PVP), resulting in a high cognitive burden.
- **Robot-gated IIL**: Robots autonomously request assistance based on certain criteria (e.g., Ensemble-DAgger, Thrifty-DAgger), mitigating the human monitoring workload.

Key limitations of prior work in robot-gated methods:

**Misalignment between uncertainty estimation and human intervention intent**: Uncertainty based on action variance can be low in safety-critical states (false negatives) and high in states where the agent is already proficient (false positives).

**Fixed thresholds cannot adapt to policy evolution**: The intervention rate does not automatically decrease as the agent gradually learns the task.

**High computational overhead**: Training an ensemble of policy networks is required to calculate action variance.

Key Insight of AIM: To design an adaptive mechanism that can **simulate human intervention decisions** and **automatically reduce the intervention rate as the policy improves**.

## Method

### Mechanism

Train a **proxy Q-function** $Q_\theta^I(s, a_r)$ to approximate human intervention decisions:

- A higher $Q_\theta^I(s, a_r)$ value indicates a higher likelihood of human intervention in that state.
- When the agent's action $a_r$ deviates from the expert action $a_h$, the Q-value approaches +1.
- When the agent is aligned with the expert, the Q-value approaches −1, automatically reducing requests.

### Loss & Training

$$J^{\text{AIM}}(\theta) = \mathbb{E}_{(s,a_h)\sim\mathcal{B}_h}\left[|Q_\theta^I(s,a_h)+1|^2\right] + \mathbb{E}_{(s,a_h)\sim\mathcal{B}_h, a_r\sim\pi_r(s)}\left[f(a_r,a_h)\cdot|Q_\theta^I(s,a_r)-1|^2\right]$$

where $f(a_r, a_h) = \mathbb{I}[\|a_r - a_h\|^2 > \epsilon]$ determines whether the action discrepancy exceeds the threshold.

**Intuition**: The first term pulls the Q-values of expert actions toward −1 (no intervention required), while the second term pushes the Q-values toward +1 when the agent's action deviates (intervention required).

### Temporal Difference (TD) Loss

To generalize Q-values to states explored by the agent but not covered by the expert, a TD loss is introduced:

$$J^{\text{TD}}(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{B}_h\cup\mathcal{B}_r}\left[|Q_\theta(s,a) - \gamma\max_{a'}Q_{\hat{\theta}}(s',a')|^2\right]$$

Total loss: $J(\theta) = J^{\text{AIM}}(\theta) + J^{\text{TD}}(\theta)$

### Intervention Trigger and Termination

- **Switch-to-human**: Request expert assistance when $Q_\theta^I(s, a_r) > \beta$, where the threshold $\beta$ is the $(1-\delta)$-quantile of the Q-value distribution ($\delta=0.05$).
- **Continue-with-human**: After the expert intervenes, the request stops if $\|a_r - a_h\|^2 \leq \epsilon$.
- $\epsilon$ is set to the mean discrepancy between the current policy and the expert actions, which is adaptively updated during training.

### Algorithm Flow

1. The first $n=2$ trajectories are fully monitored by humans (human-gated warm-up).
2. Initialize $Q_\theta^I$ and thresholds $\beta$, $\epsilon$ using the collected data.
3. Robot-gated phase: The agent explores autonomously and only requests help when $Q_\theta^I > \beta$.
4. Continuously update the policy $\pi_r$, the Q-function $Q_\theta^I$, and thresholds.

## Key Experimental Results

### MetaDrive Autonomous Driving (Continuous Action Space, 2000 steps Expert Budget)

| Method | Robot-Gated | Expert Data (Intervention Rate) | Total Data | Success Rate | Return | Route Completion Rate |
|---|---|---|---|---|---|---|
| BC | — | 2K | 2K | 0.33±0.04 | 243.0±46.7 | 0.62±0.08 |
| HG-DAgger | ✗ | 0.9K (0.45) | 2K | 0.61±0.07 | 310.8±16.7 | 0.78±0.07 |
| PVP | ✗ | 0.4K (0.19) | 2K | 0.62±0.06 | 270.4±28.6 | 0.77±0.04 |
| Ensemble-DAgger | ✓ | 2K (0.55) | 3.6K | 0.60±0.09 | 267.4±9.9 | 0.54±0.10 |
| Thrifty-DAgger | ✓ | 2K (0.21) | 9.5K | 0.58±0.03 | 250.0±23.9 | 0.73±0.03 |
| **AIM (Ours)** | ✓ | **1.9K (0.24)** | 7.7K | **0.82±0.06** | **328.4±20.4** | **0.91±0.03** |
| Neural Expert | — | — | — | 0.84±0.05 | 336.5±17.1 | 0.93±0.01 |

**Key Findings**:

- The success rate of AIM (0.82) is close to that of the Neural Expert (0.84), outperforming all baselines.
- Compared to Thrifty-DAgger, the success rate increases by **41%** (0.58→0.82), and the route completion rate increases by **25%**.
- AIM achieves superior performance using less expert data (1.9K vs. 2K).
- AIM also outperforms all baselines in MiniGrid discrete action space tasks.

## Highlights & Insights

1. **Adaptive intervention rate**: The Q-function naturally decreases intervention requests as the policy improves, without requiring manual adjustment of decay schedules.
2. **Alignment with human intent**: Directly learns a proxy model of human intervention decisions, instead of relying on heuristic uncertainty estimation.
3. **Precise localization of safety-critical states**: AIM only requests assistance near traffic cones and roadblocks, whereas Thrifty-DAgger frequently requests assistance even on straight roads.
4. **Look-ahead capability of TD propagation**: Temporal difference propagates Q-values to unseen states, enabling the anticipation of future errors.
5. **Minimal warm-up**: Requires human monitoring for only the first 2 trajectories to initialize the robot-gated mode.

## Limitations & Future Work

1. **Using a neural expert instead of real humans in experiments**: Although this is standard practice, the gap between neural experts and real human interaction has not been fully validated.
2. **Limited task complexity**: Evaluations are conducted only in two relatively simple environments, MetaDrive and MiniGrid.
3. **High-dimensional visual observations are not covered**: The current setup uses a 259-dimensional sensor vector, and the performance under image input scenarios remains unknown.
4. **Q-function generalization**: It remains unclear whether the proxy Q-function can still reliably predict intervention needs when the environment distribution shifts significantly.
5. **Cold start from offline to online**: The warm-up phase still requires full human monitoring, which may not be ideal in extremely high-cost scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — Modeling human intervention decisions via a proxy Q-function is an elegant and original idea.
- Experimental Thoroughness: ⭐⭐⭐ — Covers both continuous and discrete scenarios, but lacks environmental diversity.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive illustrations, and rigorous mathematical derivations.
- Value: ⭐⭐⭐⭐ — Practical significance for reducing human-in-the-loop costs, with a notable 40% efficiency improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Action-Constrained Imitation Learning](action-constrained_imitation_learning.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[ICLR 2026\] DataMIL: Selecting Data for Robot Imitation Learning with Datamodels](../../ICLR2026/robotics/datamil_selecting_data_for_robot_imitation_learning_with_datamodels.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](../../NeurIPS2025/robotics/adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[CVPR 2026\] Beyond Success: Refining Elegant Robot Manipulation from Mixed-Quality Data via Just-in-Time Intervention](../../CVPR2026/robotics/beyond_success_refining_elegant_robot_manipulation_from_mixed-quality_data_via_j.md)

</div>

<!-- RELATED:END -->
