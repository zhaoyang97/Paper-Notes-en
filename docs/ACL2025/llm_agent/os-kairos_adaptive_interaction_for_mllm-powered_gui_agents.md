---
title: >-
  [Paper Note] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents
description: >-
  [ACL 2025][LLM Agent][GUI Agent] This paper proposes OS-Kairos, which automatically annotates step-by-step confidence scores via a collaborative probing framework and fine-tunes them into a base model. This enables the GUI Agent to predict confidence at each step, autonomously deciding to execute the action or request human intervention. In complex scenarios, the task success rate (TSR) is improved from 14.29% (OS-Atlas-Pro-7B) to 88.20%, along with an absolute improvement of…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "Confidence-driven Interaction"
  - "Human-AI Collaboration"
  - "Over-execution"
  - "Adaptive Interaction"
  - "Collaborative Probing Framework"
date: 2026-05-08
content_hash: 749c71ef942a0bb5
---

# OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents

**Conference**: ACL 2025  
**arXiv**: [2503.16465](https://arxiv.org/abs/2503.16465)  
**Code**: [https://github.com/Wuzheng02/OS-Kairos](https://github.com/Wuzheng02/OS-Kairos)  
**Area**: LLM Agent / GUI Automation  
**Keywords**: GUI Agent, Confidence-driven Interaction, Human-AI Collaboration, Over-execution, Adaptive Interaction, Collaborative Probing Framework

## TL;DR

This paper proposes OS-Kairos, which automatically annotates step-by-step confidence scores via a collaborative probing framework and fine-tunes them into a base model. This enables the GUI Agent to predict confidence at each step, autonomously deciding to execute the action or request human intervention. In complex scenarios, the task success rate (TSR) is improved from 14.29% (OS-Atlas-Pro-7B) to 88.20%, along with an absolute improvement of 24–87% on the AITZ and Meta-GUI benchmarks.

## Background & Motivation

**Background**: MLLM-driven GUI Agents (e.g., OS-Atlas, Auto-UI, AppAgent) are capable of screenshot analysis and action prediction in mobile/desktop environments, achieving promising performance on simple GUI tasks. Existing works primarily focus on enhancing grounding capabilities (SeeClick, OS-Atlas) and reasoning capabilities (Auto-UI, AITZ).

**Limitations of Prior Work — Over-execution**: Existing GUI Agents operate fully autonomously **without evaluating action confidence**. Consequently, they continue blind execution even under uncertainty, leading to irreversible errors. Pilot experiments show that Qwen2-VL-7B achieves 0% TSR in complex scenarios, while OS-Atlas-Pro-7B achieves only 17%.

**Three Typical Complex Scenarios**: (1) **Ambiguous instructions**—user instructions lack crucial information (e.g., shopping without specifying size, account logout scenarios); (2) **Unexpected interruptions**—model hallucinations and shortcut predictions deviate from the correct trajectory; (3) **Environmental hijack**—external interferences such as pop-up hijacking, network disconnection, or permission requests.

**Limitations of Existing Solutions**: Meta-GUI proposes conversational guidance, but requires manual annotation of each complex step, which severely limits scalability. Prompt-based interaction methods yield a 0% Human Success Rate (HSR) on OS-Atlas-Pro-7B, proving ineffective. Completely relying on step-by-step human intervention can improve TSR to 62%, but is impracticable.

**Key Insight**: Integrate confidence assessment capabilities directly into the base model, enabling the Agent to possess "self-awareness"—knowing when to act autonomously and when to pause and seek assistance.

**Core Idea**: A two-stage approach: first, use GPT-4o as a Critic to automatically annotate and probe the confidence score of each step for the Agent (collaborative probing). Then, jointly train both confidence scores and action predictions into the model (confidence-driven interaction). During deployment, a threshold is used to control interaction sensitivity.

## Method

### Overall Architecture

OS-Kairos's pipeline consists of three steps:
1. **Instruction Collection**: Collect complex instructions from public datasets and manual designs, which are further diversified using GPT-4 to cover Chinese and English, 12 applications, and 12 topics.
2. **Collaborative Probing Framework**: Automatically generate high-quality trajectories with confidence annotations on real mobile devices using an Agent-Critic collaboration paradigm.
3. **Confidence-driven Interaction**: Fine-tune the annotated data into the base model, allowing the model to simultaneously output both actions and confidence scores.

### Key Designs

#### Key Design 1: Collaborative Probing Framework

- **Function**: Automatically annotate the probing Agent (OS-Atlas-Pro-7B) with a confidence score of 1–5 at each interaction step while generating high-quality GUI trajectory data.
- **Mechanism**: Agent-Critic collaboration paradigm. The Agent (OS-Atlas-Pro-7B) predicts current actions, while the Critic (GPT-4o + layout parser model) rates and supervises the execution:
    - The Agent predicts the action for the current step $a_t^p$.
    - The Critic comprehensively evaluates the action based on screenshots, plan lists, and historical trajectories to output a score $\text{score}_t \in [1, 5]$.
    - If $\text{score}_t = 5$, indicating the Agent is correct, the Agent's action is executed.
    - If $\text{score}_t < 5$, the Critic provides a corrective action $a_t^c$ to correct and continue probing.
    - The Critic simultaneously monitors the plan progress to determine if the instruction is completed.
- **Design Motivation**: (1) Connecting to real physical mobile devices (rather than simulators) covers commercial applications (such as Xiaohongshu, which has protection mechanisms); (2) GPT-4o, as the strongest multimodal model, possesses reliable judgment capabilities; (3) Automated annotation via Agent-Critic collaboration avoids the high cost of manual step-by-step annotation.
- **Data Refinement**: Verify and refine trajectories to ensure consistency between actions and confidence scores. Steps with a score of 5 are concentrated on routine operations (opening apps, clicking search bars), whereas complex steps exhibit significantly lower scores.

#### Key Design 2: Confidence-driven Interaction

- **Function**: Integrate confidence evaluation capabilities into the GUI Agent, enabling it to output both current actions and confidence scores at each step, and adaptively decide whether to request human intervention based on confidence.
- **Mechanism**:
    - **Training**: Concatenate action predictions and confidence scores into a sequence, and train via standard next-token prediction: $\mathcal{L} = \sum_{i=1}^{N} \mathcal{P}_\theta((a_t || \text{score}_t)^i | P_p(s_t, \tau_i, h_{t-1}, (a_t || \text{score}_t)^{<i}))$
    - **Inference**: Introduce a threshold $\gamma$. When $\text{score}_t < \gamma$, human intervention is triggered; otherwise, autonomous execution proceeds.
- **Design Motivation**: (1) Concatenated sequence training is more stable than multi-task training and does not degrade original action prediction capabilities; (2) The threshold mechanism provides flexibility: minimum $\gamma$ = fully autonomous, maximum $\gamma$ = fully interactive, intermediate values = adaptive; (3) Confidence scores provide explainable decision support for human users.

### Loss & Training

- Base model: OS-Atlas-Pro-7B
- 8 epochs, learning rate 1e-5, train/test split of 80/20
- Default threshold $\gamma = 4$
- Confidence annotation uses GPT-4o as Critic, layout parsing uses ResNet18 (OCR detection) + ConvNextTiny (OCR recognition)

## Key Experimental Results

### Main Results: Zero-shot Comparison in Complex Scenarios

| Model | Type (%) | SR (%) | TSR (%) |
|------|----------|--------|---------|
| Qwen2-VL-7B | 43.19 | 18.94 | 0.00 |
| OS-Atlas-Pro-7B | 97.69 | 59.12 | 17.00 |
| GPT-4o (API) | 90.07 | 76.35 | 39.13 |
| Qwen-VL-MAX (API) | 92.21 | 46.89 | 29.81 |
| **OS-Kairos** | **99.88** | **95.90** | **88.20** |

### Cross-Benchmark Zero-shot Results

| Benchmark | OS-Atlas-Pro-7B SR/TSR | OS-Kairos SR/TSR |
|------|----------------------|-----------------|
| Complex Scenarios | 61.36 / 14.29 | 95.90 / 88.20 (+73.91) |
| AITZ | 58.32 / 11.15 | 87.54 / 24.51 (+24.51) |
| Meta-GUI | 84.27 / 57.29 | 96.36 / 87.71 (+87.29) |

### Dynamic Evaluation on Real Devices

| Model | Actual Steps | Relative Efficiency RE (%) | TSR (%) |
|------|---------|----------------|---------|
| GPT-4o | 302 | 75.83 | 36.00 |
| OS-Atlas-Pro-7B | 359 | 63.79 | 26.00 |
| OS-Kairos (GPT-4o Assisted) | 245 | 93.47 | 32.00 |
| OS-Kairos (Human Assisted) | 265 | 86.42 | **70.00** |

### Ablation Study

**Critic Model Ablation**:

| Critic Model | TSR (%) | HSR (%) | IP (%) | AP (%) |
|------------|---------|---------|--------|--------|
| GPT-4o | 87.71 | 86.87 | 70.75 | 96.44 |
| Qwen-VL-MAX | 85.71 | 57.63 | 61.50 | 91.55 |

**Data Scale Ablation** (Train/Test):

| Split Ratio | Type (%) | SR (%) | TSR (%) | HSR (%) |
|---------|----------|--------|---------|---------|
| 9:1 | 99.25 | 92.21 | 76.19 | 84.67 |
| 8:2 | 99.88 | 95.90 | 88.20 | 86.87 |
| 7:3 | 99.46 | 94.16 | 83.94 | 84.79 |
| 6:4 | 99.41 | 94.05 | 78.30 | 84.47 |

**Threshold Sensitivity**: At $\gamma=4$, a 37.28% intervention rate achieves 88.20% TSR; at $\gamma=2$, only a 19.01% intervention rate (averaging 0.81 human interventions per instruction) is required to achieve a 55.28% TSR, which is close to the fine-tuned model level.

### Key Findings

- **TSR from 14% $\rightarrow$ 88%**: Adaptive interaction brings a qualitative leap in complex scenarios. Low SR leads to exponential decay in TSR, whereas human intervention in critical steps unlocks global success.
- **Prompt-based interaction is ineffective**: When using prompts for interaction on OS-Atlas-Pro-7B, the HSR is 0% and the TSR is only 9.94%. This demonstrates that confidence capabilities must be internalized through fine-tuning.
- **Effective cross-model generalization**: Knowledge distillation of confidence data to Qwen2-VL-2B still achieves 85.09% TSR, indicating backward compatibility of the annotated data.
- **Minimal data is sufficient**: An 8:2 split achieves the optimal performance, indicating that confidence integration does not require a large volume of data.
- **Efficiency close to human level**: OS-Kairos achieves a relative efficiency (RE) of 86–93%, significantly outperforming baseline models (57–75%).

## Highlights & Insights

- **"Knowing when to pause and ask" is more important than "knowing how to do everything"**: This is a critical capability for transitioning from demo agents to practical deployment. Under complex scenarios, blindly autonomous TSR is only 0–17%, whereas knowing when to seek help reaches 88%.
- **Confidence as a natural byproduct of sequence prediction**: Concatenating confidence scores seamlessly to the end of action sequences and training via standard NTP is straightforward without hurting original action prediction capabilities. This serves as a highly practical design pattern.
- **Agent-Critic collaborative annotation paradigm**: Distilling knowledge from a strong model to supervise a weak model for automatic annotation of datasets addresses the core issue of **where ground truth confidence comes from**, offering strong generalizability.
- **Thresholds offer a continuous human-machine control spectrum**: Smooth adjustment from fully autonomous to fully interactive can be flexibly configured depending on safety requirements in different application scenarios.
- **Taxonomy of three over-execution scenarios**: Ambiguous instructions, unexpected interruptions, and environmental hijacking suggest a clear framework for GUI Agent safety research.

## Limitations & Future Work

1. **Critic model dependency on GPT-4o**: The validation of confidence scores highly depends on the Critic's capabilities. Replacing GPT-4o with Qwen-VL-MAX decreases the HSR from 86.87% to 57.63%. Future research can explore self-supervised or RL methods to capture confidence signals.
2. **Limited scenario coverage**: It only covers three typical complex scenarios (12 apps, 12 topics), while real-world long-tail scenarios are far more diverse.
3. **Idealized human intervention assumptions**: Evaluations use ground truth or GPT-4o simulated human feedback, which might be inaccurate or delayed in real-world scenarios.
4. **Validation strictly on mobile devices**: The paper only evaluates Android environments and lacks generalization studies on web and desktop platforms.
5. **Confidence calibration issue**: No calibration analysis was performed to verify whether scores reflect actual probabilities (e.g., whether `score=3` represents a 60% success rate), which may result in under/over-confidence.
6. **Potential for dynamic thresholds**: Since the threshold $\gamma$ is fixed throughout execution, investigating dynamically adjusted environments depending on history/task complexity is a promising angle.

## Related Work & Insights

- **GUI Agent Trajectory**: AppAgent (prompt-based) $\rightarrow$ Auto-UI (reasoning-enhanced) $\rightarrow$ OS-Atlas (grounding-enhanced) $\rightarrow$ OS-Kairos (adaptive interaction) defines a clear evolutionary path for capabilities.
- **Ability Probing Direction**: From static benchmark evaluations (AITW, AndroidControl) to dynamic confidence probing represents a significant paradigm shift.
- **Inspirations for general Agent systems**: Confidence-driven interaction is not limited to GUI Agents. Any Human-AI collaboration scenario (code generation, autonomous driving, medical diagnostics) can adopt the "high confidence: act autonomously, low confidence: seek help" guideline.
- **Connection to RLHF**: Critic models play a role akin to reward models without utilizing complex RL strategies, relying on simple but highly effective SFT.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Confidence-driven adaptive GUI interaction is a natural yet overlooked angle. The Agent-Critic collaborative annotation paradigm holds generic value, though the core training mechanism (SFT + threshold inference) is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The paper covers 3 datasets, real-device dynamic evaluation, and 6 thorough ablation studies (Critic models, data scale, model size, thresholds, interaction paradigms, and prompt-based comparisons).
- **Writing Quality**: ⭐⭐⭐⭐ The taxonomy of three over-execution scenarios is clear, the pilot experiments are convincing, and the overall logic is fluent, though some mathematical formulated explanations are slightly redundant.
- **Value**: ⭐⭐⭐⭐⭐ It directly targets the core bottleneck of deploying GUI Agents (safety). The performance lift (88% TSR vs 14%) is highly persuasive, and the flexibility of threshold controls is beneficial for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)
- [\[ACL 2025\] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)
- [\[ICCV 2025\] UIPro: Unleashing Superior Interaction Capability for GUI Agents](../../ICCV2025/llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)
- [\[ICML 2025\] Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](../../ICML2025/llm_agent/aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)

</div>

<!-- RELATED:END -->
