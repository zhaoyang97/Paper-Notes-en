---
title: >-
  [Paper Note] Attacking Vision-Language Computer Agents via Pop-ups
description: >-
  [ACL 2025][Multimodal VLM][adversarial attack] This work systematically designs a set of adversarial pop-up attack methods to attack vision-language model-based computer agents. The proposed methods achieve an average attack success rate of $86\%$ on OSWorld and VisualWebArena, decreasing the task success rate by $47\%$ while showing that basic defense mechanisms are almost ineffective.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "adversarial attack"
  - "VLM agent"
  - "pop-up attack"
  - "computer use safety"
  - "agent robustness"
date: 2026-05-08
content_hash: 715c9f2f1160b4d3
---

# Attacking Vision-Language Computer Agents via Pop-ups

**Conference**: ACL 2025  
**arXiv**: [2411.02391](https://arxiv.org/abs/2411.02391)  
**Code**: [https://github.com/SALT-NLP/PopupAttack](https://github.com/SALT-NLP/PopupAttack)  
**Area**: Multimodal VLM  
**Keywords**: adversarial attack, VLM agent, pop-up attack, computer use safety, agent robustness

## TL;DR
This work systematically designs a set of adversarial pop-up attack methods to attack vision-language model-based computer agents. The proposed methods achieve an average attack success rate of $86\%$ on OSWorld and VisualWebArena, decreasing the task success rate by $47\%$ while showing that basic defense mechanisms are almost ineffective.

## Background & Motivation

**Background**:
   - VLM-driven autonomous agents have demonstrated great potential in daily computer tasks (such as web browsing and desktop software operations).
   - Agent frameworks based on screenshots and Set-of-Mark (SoM) are becoming mainstream directions.
   - Companies like Anthropic have launched computer-use products, making visual inputs increasingly important in agentic applications.

**Limitations of Prior Work**:
   - Existing attack methods are mostly based on textual injection (e.g., inserting invisible instructions in HTML) or gradient-optimized image perturbations. The former is not applicable to screenshot-based agents, while the latter is non-transferable to closed-source models.
   - There is a lack of systematic research on visual-level risks faced by VLM agents.
   - Agents perform operations on behalf of users without supervision, making the consequences of attacks potentially severe (such as installing malware or redirecting to fraudulent websites).

**Key Challenge**:
   - Human users can easily recognize and ignore malicious pop-ups and advertisements, but VLM agents are misled by them.
   - Since the environment functions normally for human users, agents should theoretically be able to complete tasks, but the reality is precisely the opposite.

**Goal**
   - Reveal the vulnerability of VLM agents under visual adversarial attacks.
   - Systematically define the design space of pop-up attacks and quantify their attack effectiveness.
   - Evaluate the efficacy of existing basic defense strategies.

**Key Insight**:
   - Designing adversarial pop-ups that are recognizable to humans but irresistible to agents, spanning a four-dimensional design space.
   - Implementing attack experiments in real agent evaluation environments.

**Core Idea**:
   - Carefully crafted pop-up attacks can easily mislead state-of-the-art VLM agents, whereas human users can easily identify and ignore them, exposing significant risks for safe agent deployment.

## Method

### Overall Architecture
The attacker inserts carefully designed adversarial pop-ups into the agent's observation space (screenshots and a11y trees). The goal is to make the agent click on the pop-up instead of performing the normal task. The attacks correspond to various real-world scenarios: malvertising, XSS injection, and clickable images in phishing emails.

### Key Designs

1. **Attention Hook**:

    - **Function**: Attracts the agent's attention using several keywords.
    - **Mechanism**: By default, it uses a summarized version of the user query as a hook, making the pop-up highly relevant to the task.
    - **Design Motivation**: Agents prioritize elements that semantically match the current task target.
    - **Ablation Study Results**: Using the user query summary yields the highest ASR ($93.3\%$), followed by "Virus Warning" ($90.0\%$), while using screen-based inferred intent drops significantly to $53.9\%$—demonstrating that **knowing the user query is key to attack success**.

2. **Instruction**:

    - **Function**: Specifies the action the attacker wants the agent to execute.
    - **Mechanism**: Divided into variants such as Click Tag, Click Coordinate, and Click Here.
    - **Design Motivation**: Different agent frameworks use different action spaces (coordinates vs. tags).
    - **Ablation Study Results**: Click Tag and Click Coordinate perform best ($>90\%$ ASR), Click Here drops to $72.8\%$, and Click Random is the worst ($13.7\%$).

3. **Information Banner**:

    - **Function**: Adds contextual information to imply the purpose of the pop-up.
    - **Mechanism**: Uses the "OK" button by default to mimic normal operations.
    - **Ablation Study Results**: Even when labeled as "ADVERTISEMENT", the ASR remains high at $66.5\% \sim 77.9\%$—showing that simple advertisement labels cannot defend effectively.

4. **ALT Descriptor**:

    - **Function**: Adds adversarial text descriptions to the pop-up in the a11y tree.
    - **Mechanism**: Provides additional textual misleading information for SoM agents that utilize the a11y tree.
    - **Ablation Study Results**: With an empty ALT descriptor, the ASR of OSWorld-SoM drops from $91.8\%$ to $68.1\%$, and VisualWebArena-SoM drops from $62.1\%$ to $42.9\%$—indicating that ALT text is an important factor in attacking SoM agents.

### Loss & Training
- As a study of attack methods, no training is required.
- Attack Assumption: The attacker possesses complete information (user query, pop-up position, agent framework type).
- Evaluation Metrics:
    - **ASR (Attack Success Rate)**: The frequency of the agent clicking the pop-up (lower is better).
    - **SR (Success Rate)**: The task completion rate under attack (higher is better).
    - **OSR (Original Success Rate)**: The original task completion rate without attacks.

## Key Experimental Results

### Main Results
- **OSWorld-Screen (GPT-4-Turbo)**: ASR = $93.3\%$, SR is only $2.0\%$ (Original OSR was $18.0\%$).
- **OSWorld-SoM (GPT-4-Turbo)**: ASR = $91.8\%$, SR is only $8.0\%$ (Original OSR was $52.0\%$).
- **VisualWebArena-SoM (GPT-4o)**: ASR = $62.1\%$, SR = $45.8\%$ (Original OSR was $63.9\%$).
- **The average attack success rate is around $86\%$, with task success rate dropping by $47\%$ on average.**
- **Claude 3.5 Sonnet reaches $100\%$ ASR on OSWorld-Screen**—all tasks were successfully attacked.
- **Among all attacked agents, $92.7\%$ (OSWorld) / $73.1\%$ (VisualWebArena) of the actions involved clicking the pop-up.**

### Ablation Study & Key Findings
- **Blank Pop-up**: ASR drops sharply to $2.4\%$ (Screen) / $3.7\%$ (SoM), proving that pop-up content is the key to attack success.
- **Shrinking the Pop-up**: ASR drops slightly to $87.4\%$ (Screen) / $90.1\%$ (SoM), indicating visual size has little impact.
- **System Prompt Defense**: Generic defense only slightly reduces (actually increases!) ASR from $93.3\%$ to $95.9\%$; pop-up-specific defense drops it to $52.0\%$.
- **Step-level Prompt Defense**: ASR drops significantly to $5.9\%$, but introduces a $32\%$ TASR (the rate of agents terminating the task due to pop-ups), and attack variants can increase TASR to $44.0\%$.
- **Attacks significantly prolong task steps**: More tasks only stop when reaching the maximum step limit.

## Highlights & Insights

- **Simple yet highly effective attack**: Without requiring complex gradient optimization or model internal access, a few carefully designed images can paralyze SOTA agents.
- **Profound revelation of human-AI difference**: Pop-ups that humans easily ignore pose a fatal threat to agents, highlighting the shortcomings of VLMs in "common-sense judgment".
- **Difficulty in defense**: Even when labeling advertisements or reminding with system prompts, the effects are unsatisfactory; step-level defenses are effective but incur high rejection rates.
- **User query is key to attack**: This implies a deep risk—if an attacker can obtain the user's intent, they can target them precisely.

## Limitations & Future Work

- The threat model assumes the attacker possesses complete information (user query, agent framework), which might be more challenging in reality.
- Only a limited number of VLMs were tested as backbones; the robustness of other models remains unknown.
- No effective defense scheme was proposed, other than validating the failure of basic defenses.
- Attack scenarios are limited to the pop-up format; other visual attack forms (e.g., interface fine-tuning, fake button mimicry) are not covered.
- Only tested on two agent evaluation environments; risk assessment for more scenarios (e.g., mobile devices) is missing.
- Lacks constructive guidance on how to build robust agents.

## Related Work & Insights

- **Comparison with Wu et al. (2024)**: They used learnable noise to attack VLMs to output adversarial captions, which requires thousands of optimization steps and is difficult to transfer to closed-source models.
- **Comparison with Liao et al. (2024)**: They injected invisible malicious instructions in web pages, but such attacks will fail as agents pivot to being screenshot-based.
- **Comparison with Ma et al. (2024)**: They studied faithfulness without malicious distractors, whereas this paper investigates malicious attacks.
- **Insights**: Agent safety is a prerequisite for large-scale agent deployment, requiring the construction of defense mechanisms from multiple dimensions such as visual understanding, instruction following, and task separation.

## Rating

- **Novelty**: ⭐⭐⭐⭐
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐
- **Writing Quality**: ⭐⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents](../../AAAI2026/multimodal_vlm/are_we_done_yet_a_vision-based_judge_for_autonomous_task_completion_of_computer_.md)
- [\[ACL 2025\] MMINA: Benchmarking Multihop Multimodal Internet Agents](mmina_benchmarking_multihop_multimodal_internet_agents.md)
- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)
- [\[ICML 2025\] Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning](../../ICML2025/multimodal_vlm/towards_efficient_online_tuning_of_vlm_agents_via_counterfactual_soft_reinforcem.md)

</div>

<!-- RELATED:END -->
