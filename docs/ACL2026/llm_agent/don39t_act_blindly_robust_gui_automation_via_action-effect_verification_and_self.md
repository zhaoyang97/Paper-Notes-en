---
title: >-
  [Paper Note] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction
description: >-
  [ACL 2026][LLM Agent][GUI Automation] This paper proposes the VeriGUI framework, which utilizes a Thinking-Verification-Action-Expectation (TVAE) closed-loop reasoning mechanism and a two-stage training pipeline (Robust…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "GUI Automation"
  - "Action Verification"
  - "Self-Correction"
  - "GRPO Reinforcement Learning"
  - "Robustness"
date: 2026-05-08
content_hash: 162df67b5c510d1a
---

# Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction

**Conference**: ACL 2026  
**arXiv**: [2604.05477](https://arxiv.org/abs/2604.05477)  
**Code**: None  
**Area**: Multimodal VLM / LLM Agent  
**Keywords**: GUI Automation, Action Verification, Self-Correction, GRPO Reinforcement Learning, Robustness

## TL;DR
This paper proposes the VeriGUI framework, which utilizes a Thinking-Verification-Action-Expectation (TVAE) closed-loop reasoning mechanism and a two-stage training pipeline (Robust SFT + GRPO). This allows the GUI Agent to verify the success of each step and self-correct upon failure, significantly outperforming baselines at both 3B and 7B scales.

## Background & Motivation

**Background**: VLM-based GUI Agents are already capable of interpreting screenshots, understanding natural language instructions, and executing multi-step tasks. Models such as CogAgent, SeeClick, and UI-TARS have made rapid progress across multiple benchmarks. However, these agents implicitly assume that every action will be executed as intended.

**Limitations of Prior Work**: In practical deployment, network latency, rendering delays, and system interruptions can cause operation failures. When a failure occurs, current agents continue to assume the operation was successful, generating the next step based on an unchanged screen. Worse, since failure scenarios are rarely encountered during training, agents tend to repeat the exact same ineffective operation, forming an infinite execution loop. Empirical data shows that timeouts caused by repeating ineffective actions account for 72.3% of all failures.

**Key Challenge**: Human users naturally verify whether expected changes (e.g., button highlighting, page navigation) occur after each interaction. This verification-diagnosis-correction loop is entirely missing in current GUI Agents. Online RL training faces high interaction latency and infrastructure costs (requiring 64 parallel Android emulators), while offline datasets lack failure signals.

**Goal**: (1) Design a reasoning framework that explicitly models action result verification and recovery mechanisms; (2) Develop a training method to learn self-correction behavior without the need for online interaction.

**Key Insight**: Leverage the idempotency of GUI errors—ineffective actions typically do not change the screen state. This property enables the simulation of online feedback from offline data: if the screen does not change, the action has failed.

**Core Idea**: Incorporate verification and expected effect prediction steps into the reasoning framework. During training, learn "honest" self-monitoring by synthesizing failure trajectories and utilizing idempotency to simulate online feedback.

## Method

### Overall Architecture
VeriGUI consists of two core components: (1) The TVAE reasoning loop—generating structured Thinking ($T_t$), Verification ($V_t$), Action ($A_t$), and Expectation ($E_t$) at each step to form a time-linked closed loop; (2) A two-stage training pipeline—Stage 1 establishes basic verification capabilities via Robust SFT on mixed success/failure trajectories, and Stage 2 refines self-correction behavior via GRPO using asymmetric verification rewards.

### Key Designs

1. **TVAE Reasoning Loop**:

    - **Function**: Implements closed-loop verification at each interaction step—detecting failures, diagnosing causes, and executing recovery to prevent blind error accumulation.
    - **Mechanism**: Each step $t$ generates four outputs: Think $T_t$ (structured analysis using tags like [Verify], [Recall], [Grounding], [Action], switching to [Diagnose] and [Recovery] during error correction); Verification $V_t$ (a binary judgment of SUCCESS or NO_CHANGE, comparing the current screen $S_t$ with the previous expectation $E_{t-1}$); Action $A_t$ (executable JSON operation); and Expected Effect $E_t$ (predicting screen changes after the action, serving as the verification target for the next step). Crucially, TVAE is not a linear chain but a time-linked loop where the expected effect at step $t$ becomes the verification hypothesis for step $t+1$.
    - **Design Motivation**: Humans implicitly verify if an action has taken effect after every GUI interaction. Expected effect prediction forces the agent to contemplate the consequences of an action before execution, improving both action quality and providing a clear baseline for subsequent verification.

2. **Two-Stage Training Pipeline**:

    - **Function**: Teaches the agent to identify failures and perform corrections without requiring online interaction.
    - **Mechanism**: Stage 1 (Robust SFT) constructs a hybrid dataset—Type A consists of normal success trajectories, and Type B consists of synthetic failure trajectories (pairing an unchanged screen $S_{t-1}$ with a history claiming $A_{t-1}$ was executed), using GPT-4o to generate structured CoT labels. Stage 2 (GRPO) simulates online feedback using GUI idempotency: for Type A inputs, $V_{\text{target}}=\text{SUCCESS}$; for Type B inputs, $V_{\text{target}}=\text{NO\_CHANGE}$. The model is rewarded only when predicted verification matches objective reality.
    - **Design Motivation**: Direct SFT may lead the model to overfit the optimistic assumption that "all operations succeed." The two-stage design first establishes a prior for verification behavior (Stage 1) and then refines "honest" self-monitoring via reinforcement learning (Stage 2).

3. **Composite Reward Function and Asymmetric Verification Penalty**:

    - **Function**: Drives the agent to simultaneously optimize action correctness, effect prediction quality, and verification honesty.
    - **Mechanism**: The total reward is $R_t = R_{\text{act}} + \alpha \cdot R_{\text{eff}} + \beta \cdot R_{\text{ver}}$. Action rewards are based on IoU matching with ground truth; effect rewards measure description quality via BERTScore when the action is correct; verification rewards use asymmetric penalties—Correct judgment +1.0, False Negative -0.5, and False Positive (Hallucination) **-2.0**. Severe penalties for hallucinations force the agent to align internal beliefs with visual reality.
    - **Design Motivation**: The asymmetric penalty for verification is a core design—False Positives (claiming success when it failed) are more dangerous than False Negatives because they lead to error accumulation. The -2.0 penalty ensures the agent prefers reporting NO_CHANGE over faking success when uncertain.

### Loss & Training
Stage 1: Standard cross-entropy loss, 2 epochs, learning rate $1 \times 10^{-5}$. Stage 2: GRPO objective, 15 epochs, learning rate $5 \times 10^{-6}$, group size $G=6$, KL divergence constraint from the reference policy, $\alpha=0.5, \beta=0.5$. Trained using $8 \times$ A100 GPUs.

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

### Ablation Study (Robustness Benchmarks)

| Model | Loop Rate↓ | Recovery Success Rate↑ |
|------|-----------|----------------------|
| Qwen2.5-VL-3B | High | Low |
| VeriGUI-3B | Significantly Reduced | **51.1%** |
| VeriGUI-7B | Lowest | **52.5%** |

### Key Findings
- All 3B baselines had a Sim-TSR of zero under pseudo-online conditions (completely unable to complete tasks), while VeriGUI-3B reached 16.7%.
- VeriGUI-3B outperformed several 7B baselines in Type Match, indicating that TVAE’s structured reasoning actively improves action type selection.
- The gap between TSR and Sim-TSR directly quantifies the proportion of tasks completed through error recovery.
- VeriGUI-7B achieved an ASO of 1.09, meaning it requires only 9% more steps than the optimal path on average.
- The verification-recovery mechanism demonstrated good transferability in GUI Odyssey cross-distribution testing.

## Highlights & Insights
- **Simulating Online Feedback via Idempotency**: The observation of GUI error idempotency (unchanged screen = failed operation) is an elegant insight that allows training self-correction behaviors without building complex online environments. This idea can be extended to other environments with similar "no change = failure" characteristics.
- **Asymmetric Verification Penalty**: Applying a penalty for hallucinations (False Positives) that is four times greater than for False Negatives ensures "honest" self-monitoring through the incentive mechanism. This is a pattern worth emulating in RL reward design.
- **Dual Role of Expected Effect Prediction**: It serves both as a comparison baseline for verification and forces the agent to anticipate consequences before execution, indirectly improving action quality.

## Limitations & Future Work
- The TVAE framework increases the generation overhead per step (Think+Verification+Expectation), potentially increasing inference latency.
- The idempotency assumption does not apply to all GUI failure modes (e.g., mis-operations causing irreversible changes).
- Synthetic failure trajectories may not cover all real-world failure types (e.g., partial loading, interrupted animations).
- Detailed results for online MiniWoB++ and AndroidWorld were not fully reported.

## Related Work & Insights
- **vs UI-TARS**: UI-TARS improves accuracy through large-scale pre-training and unified modeling but lacks verification and recovery capabilities. VeriGUI achieves comparable performance at a smaller scale through closed-loop verification.
- **vs DigiRL / DistRL**: These optimize task success rates through online RL but require significant simulator infrastructure. VeriGUI achieves similar effects using offline data via idempotency.
- **vs LLM Self-Correction**: Self-correction work by Madaan et al. assumes external feedback or oracle information, whereas VeriGUI’s verification is based entirely on internal reasoning of visual evidence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of TVAE closed-loop verification, idempotency utilization, and asymmetric penalties is highly sophisticated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, robustness tests, and cross-distribution validation are provided, though online results are less detailed.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and method descriptions are detailed, though some sections are heavy on notation.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value as it addresses the core pain points of GUI Agents—blind execution and infinite loops.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](../../ICML2026/llm_agent/think_twice_before_you_act_enhancing_agent_behavioral_safety_with_thought_correc.md)
- [\[ICML 2026\] AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions](../../ICML2026/llm_agent/autorpa_efficient_gui_automation_through_llm-driven_code_synthesis_from_interact.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)
- [\[ACL 2026\] Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces](dont_click_that_teaching_web_agents_to_resist_deceptive_interfaces.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)

</div>

<!-- RELATED:END -->
