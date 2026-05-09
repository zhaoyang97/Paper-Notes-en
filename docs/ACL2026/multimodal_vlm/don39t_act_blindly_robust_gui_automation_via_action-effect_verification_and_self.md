---
title: >-
  [Paper Note] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction
description: >-
  [ACL 2026][Multimodal VLM][GUI automation] This paper proposes VeriGUI, a framework that incorporates a Thinking-Verification-Action-Expectation (TVAE) closed-loop reasoning mechanism and a two-stage training pipeline (Robust SFT + GRPO), enabling GUI agents to verify whether each action succeeds and self-correct upon failure. VeriGUI achieves substantial improvements over baselines at both 3B and 7B scales.
tags:
  - ACL 2026
  - Multimodal VLM
  - GUI automation
  - action verification
  - self-correction
  - GRPO reinforcement learning
  - robustness
date: 2026-05-08
content_hash: c31975c9bb4343a5
---

# Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction

**Conference**: ACL 2026
**arXiv**: [2604.05477](https://arxiv.org/abs/2604.05477)
**Code**: None
**Area**: Multimodal VLM / LLM Agent
**Keywords**: GUI automation, action verification, self-correction, GRPO reinforcement learning, robustness

## TL;DR
This paper proposes VeriGUI, a framework that incorporates a Thinking-Verification-Action-Expectation (TVAE) closed-loop reasoning mechanism and a two-stage training pipeline (Robust SFT + GRPO), enabling GUI agents to verify whether each action succeeds and self-correct upon failure. VeriGUI achieves substantial improvements over baselines at both 3B and 7B scales.

## Background & Motivation

**Background**: VLM-based GUI agents are capable of interpreting screenshots, understanding natural language instructions, and executing multi-step tasks. Models such as CogAgent, SeeClick, and UI-TARS have achieved rapid progress across multiple benchmarks. However, these agents implicitly assume that every action executes as intended.

**Limitations of Prior Work**: In real-world deployments, network latency, rendering delays, and system interruptions cause actions to fail. When failures occur, current agents continue to assume success and generate subsequent actions on an unchanged screen. Moreover, because failure scenarios are rarely encountered during training, agents tend to repeat the same ineffective action indefinitely, forming infinite execution loops. Empirical data indicate that execution timeouts caused by repeated ineffective actions account for 72.3% of all failures.

**Key Challenge**: Human users naturally verify whether expected changes have occurred after each interaction (e.g., whether a button is highlighted or a page has navigated), yet this verification-diagnosis-correction loop is entirely absent in current GUI agents. Online RL training incurs high interaction latency and infrastructure costs (requiring 64 parallel Android emulators), while offline datasets lack failure signals.

**Goal**: (1) Design a reasoning framework that explicitly models action-outcome verification and recovery; (2) Develop a training method that learns self-correction behavior without requiring online interaction.

**Key Insight**: This work exploits the idempotency of GUI errors — erroneous actions typically leave the screen state unchanged. This property allows online feedback to be simulated from offline data: an unchanged screen implies action failure.

**Core Idea**: Integrate verification and expected-effect prediction into the reasoning framework, and train the agent to perform honest self-monitoring by synthesizing failure trajectories and simulating online feedback via idempotency.

## Method

### Overall Architecture
VeriGUI comprises two core components: (1) the TVAE reasoning loop, which generates structured thinking (Think), a verification judgment (Verification), an action (Action), and an expected effect (Expectation) at each step, forming a temporally linked closed loop; and (2) a two-stage training pipeline, where Stage 1 uses Robust SFT on mixed success/failure trajectories to establish basic verification capability, and Stage 2 uses GRPO with an asymmetric verification reward to refine self-correction behavior.

### Key Designs

1. **TVAE Reasoning Loop**:

    - **Function**: Implements closed-loop verification at each interaction step — detecting failures, diagnosing causes, and executing recovery — to prevent blind error accumulation.
    - **Mechanism**: At each step $t$, four outputs are produced: Think $T_t$ (structured analysis using [Verify], [Recall], [Grounding], and [Action] tags, switching to [Diagnose] and [Recovery] during error correction); Verification $V_t$ (a binary judgment of SUCCESS or NO_CHANGE, comparing the current screen $S_t$ against the expected effect from the previous step $E_{t-1}$); Action $A_t$ (an executable JSON action); and Expected Effect $E_t$ (a prediction of screen changes following the action, serving as the verification target for the next step). Critically, TVAE forms a temporally linked loop rather than a linear chain: the expected effect at step $t$ becomes the verification hypothesis at step $t+1$.
    - **Design Motivation**: Humans implicitly verify whether GUI interactions have taken effect after each operation. Expected-effect prediction forces the agent to reason about the consequences of an action prior to execution, which both improves action quality and provides an explicit comparison reference for subsequent verification.

2. **Two-Stage Training Pipeline**:

    - **Function**: Teaches the agent to recognize failures and execute corrections without requiring online interaction.
    - **Mechanism**: Stage 1 (Robust SFT) constructs a mixed dataset — Type A consists of normal successful trajectories, and Type B consists of synthesized failure trajectories (pairing an unchanged screen $S_{t-1}$ with a history claiming that $A_{t-1}$ was executed), with structured chain-of-thought annotations generated by GPT-4o. Stage 2 (GRPO) simulates online feedback by exploiting the idempotency of GUI errors: for Type A inputs, $V_{\text{target}}=\text{SUCCESS}$; for Type B inputs, $V_{\text{target}}=\text{NO\_CHANGE}$. The model is rewarded only when its predicted verification judgment aligns with objective reality.
    - **Design Motivation**: Direct SFT may cause the model to overfit to an optimistic assumption that all actions succeed. The two-stage design first establishes a prior for verification behavior (Stage 1), then refines honest self-monitoring via reinforcement learning (Stage 2).

3. **Composite Reward Function and Asymmetric Verification Penalty**:

    - **Function**: Drives the agent to simultaneously optimize action correctness, effect prediction quality, and verification honesty.
    - **Mechanism**: The total reward is $R_t = R_{\text{act}} + \alpha \cdot R_{\text{eff}} + \beta \cdot R_{\text{ver}}$. The action reward is based on IoU matching against the ground-truth action; the effect reward uses BERTScore to measure the quality of effect descriptions when the action is correct; the verification reward applies an asymmetric penalty — a correct judgment yields +1.0, a false negative yields -0.5, and a false positive (hallucination) yields **-2.0**. The severe penalty for hallucination forces the agent to align its internal beliefs with visual reality.
    - **Design Motivation**: The asymmetric verification penalty is a core design choice — false positives (claiming success when the action has failed) are more dangerous than false negatives, as they lead to cascading errors. The -2.0 penalty magnitude ensures the agent prefers reporting NO_CHANGE over feigning success under uncertainty.

### Loss & Training
Stage 1: Standard cross-entropy loss, 2 epochs, learning rate $1 \times 10^{-5}$. Stage 2: GRPO objective, 15 epochs, learning rate $5 \times 10^{-6}$, group size $G=6$, KL regularization to constrain deviation from the reference policy, $\alpha=0.5$, $\beta=0.5$. Training is conducted on $8\times$ A100 GPUs.

## Key Experimental Results

### Main Results (AndroidControl-High)

| Model | TM | GR | SR | Sim-TSR | ASO↓ |
|------|-----|-----|-----|---------|------|
| Qwen2.5-VL-3B | 68.7 | 28.3 | 20.2 | 0 | — |
| UI-R1-3B | 69.0 | 27.3 | 19.1 | 0 | — |
| VeriGUI-3B | **72.2** | 32.4 | 24.8 | **16.7** | 1.25 |
| UI-TARS-7B | 72.3 | 35.2 | 30.8 | 14.1 | — |
| VeriGUI-7B | **74.2** | **36.8** | **33.1** | **23.5** | **1.09** |
| GPT-5.1 | 70.1 | 30.0 | 23.1 | — | — |

### Ablation Study (Robustness Benchmark)

| Model | Loop Rate↓ | Recovery Success Rate↑ |
|------|-----------|----------------------|
| Qwen2.5-VL-3B | High | Low |
| VeriGUI-3B | Significantly reduced | **51.1%** |
| VeriGUI-7B | Lowest | **52.5%** |

### Key Findings
- All 3B baselines achieve a Sim-TSR of zero under pseudo-online conditions (completely unable to complete tasks), while VeriGUI-3B reaches 16.7%.
- VeriGUI-3B surpasses several 7B baselines on Type Match, indicating that TVAE's structured reasoning actively improves action type selection.
- The gap between TSR and Sim-TSR directly quantifies the proportion of tasks completed through error recovery.
- VeriGUI-7B achieves an ASO of only 1.09, meaning it requires on average only 9% more steps than the optimal path.
- On the GUI Odyssey cross-distribution test, the verification-recovery mechanism demonstrates strong transferability.

## Highlights & Insights
- **Exploiting Idempotency to Simulate Online Feedback**: The observation that GUI errors are idempotent (unchanged screen = action failure) is a precise insight that enables training self-correction behavior without constructing complex online environments. This approach generalizes to other settings with similar "no change = failure" characteristics.
- **Asymmetric Verification Penalty**: Penalizing hallucinations (false reports of success) at four times the rate of false negatives ensures honest self-monitoring through the incentive structure. This reward design pattern is worth adopting in broader RL settings.
- **Dual Role of Expected-Effect Prediction**: The predicted expected effect serves both as a comparison reference for verification and as a mechanism that forces the agent to anticipate the consequences of an action before execution, indirectly improving action quality.

## Limitations & Future Work
- The TVAE framework increases per-step generation volume (Think + Verification + Expectation), potentially increasing inference latency.
- The idempotency assumption does not apply to all GUI failure modes (e.g., irreversible changes caused by erroneous actions).
- Synthesized failure trajectories may not cover all real-world failure types (e.g., partial loading, interrupted animations).
- Results on online MiniWoB++ and AndroidWorld are not reported in detail.

## Related Work & Insights
- **vs UI-TARS**: UI-TARS improves accuracy through large-scale pretraining and unified modeling but lacks verification and recovery capabilities. VeriGUI achieves comparable performance at a smaller scale through closed-loop verification.
- **vs DigiRL / DistRL**: These methods optimize task success rates via online RL but require substantial simulator infrastructure. VeriGUI achieves similar effects using offline data via idempotency.
- **vs LLM Self-Correction**: Self-correction work such as Madaan et al. assumes external feedback or oracle information, whereas VeriGUI's verification is based entirely on internal reasoning over visual evidence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of TVAE closed-loop verification, idempotency exploitation, and asymmetric penalty is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, robustness tests, and cross-distribution evaluation, though online results are insufficiently detailed.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and method descriptions are thorough, though notation is occasionally dense.
- Value: ⭐⭐⭐⭐⭐ Addresses a core pain point of GUI agents — blind execution and infinite loops — with high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](what39s_missing_in_screen-to-action_towards_a_ui-in-the-loop_paradigm_for_multim.md)
- [\[ICLR 2026\] Procedural Mistake Detection via Action Effect Modeling](../../ICLR2026/multimodal_vlm/procedural_mistake_detection_via_action_effect_modeling.md)
- [\[CVPR 2026\] Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation](../../CVPR2026/multimodal_vlm/training_high-level_schedulers_with_execution-feedback_reinforcement_learning_fo.md)
- [\[CVPR 2026\] Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](../../CVPR2026/multimodal_vlm/self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)
- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](../../CVPR2026/multimodal_vlm/see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)

<!-- RELATED:END -->
