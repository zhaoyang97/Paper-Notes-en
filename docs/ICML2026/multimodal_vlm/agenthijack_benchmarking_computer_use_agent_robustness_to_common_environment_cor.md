---
title: >-
  [Paper Note] AgentHijack: Benchmarking Computer Use Agent Robustness to Common Environment Corruptions
description: >-
  [ICML2026][Multimodal VLM][computer-use agent] This paper introduces AgentHijack, which evaluates the robustness of computer-use agents against 9 types of configurable daily environment corruptions. By further utilizing…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "computer-use agent"
  - "GUI Agent"
  - "environment corruption"
  - "robustness benchmarking"
  - "DA-GRPO"
date: 2026-05-08
content_hash: 990b1901c4f90aea
---

# AgentHijack: Benchmarking Computer Use Agent Robustness to Common Environment Corruptions

**Conference**: ICML2026  
**arXiv**: [2605.25707](https://arxiv.org/abs/2605.25707)  
**Code**: https://AgentHijack.github.io  
**Area**: LLM Agent / Multimodal Robustness  
**Keywords**: computer-use agent, GUI Agent, environment corruption, robustness benchmarking, DA-GRPO  

## TL;DR
This paper introduces AgentHijack, which evaluates the robustness of computer-use agents against 9 types of configurable daily environment corruptions. By further utilizing DA-GRPO to enhance grounding and introducing an "onlooker" for behavior summarization and environment checking, the average success rate of UI-TARS-1.5-7B is improved from 18.74% to 22.89%.

## Background & Motivation

**Background**: Computer-use agents driven by Multimodal Large Language Models (MLLMs) can already complete complex workflows such as office tasks, system operations, and browser navigation in virtual machines, web pages, and mobile environments. Benchmarks like OSWorld, WebArena, and AndroidWorld primarily focus on task success rates in clean environments, evaluating whether agents can understand screenshots, plan steps, and execute clicks or inputs.

**Limitations of Prior Work**: Real-world desktop environments are rarely "clean." Pop-ups, resolution changes, subtitle overlays, other application windows, accidental touches, window minimization, network disconnections, and login verifications often interrupt the execution flow. Existing agents exhibit localization drift, incorrect attribution, and meaningless repetitive attempts under these common corruptions. However, current robustness benchmarks either focus on adversarial attacks or use QA formats, lacking real executable GUI environments and configurable corruptions.

**Key Challenge**: The deployment risks of computer-use agents stem from daily environmental uncertainties rather than just malicious attacks. Nevertheless, mainstream evaluations still assume ideal environmental states, leaving the failure modes that models are most likely to expose in the real world systematically unmeasured.

**Goal**: The authors aim to construct a benchmark capable of injecting common corruptions into virtual machine environments like OSWorld to systematically evaluate agent corruption robustness and propose an agent framework to mitigate the impact of these corruptions.

**Key Insight**: Instead of labeling these disturbances as adversarial robustness, the paper defines them as "corruption robustness." Corruptions do not change the user task itself and have no direct malicious intent; instead, they alter the observation space, state transitions, or environment states, causing the agent's closed-loop execution to deviate from the original plan.

**Core Idea**: Expose weaknesses in agent grounding, memory, and environment checking using configurable real GUI corruptions, and then improve robustness using a "grounding-enhanced action generator + onlooker."

## Method

### Overall Architecture

AgentHijack models the computer-use agent as a POMDP. At each step, the agent outputs an action $a_t$ based on the screenshot observation $o_t$, and the environment transitions to a new state $s_{t+1}$. After reaching the maximum steps or outputting Done/Fail, the final state is judged against the objective using a task reward. While standard benchmarks evaluate the average success rate in clean environments, AgentHijack evaluates success under observations, states, or transitions altered by a corruption function $\mathcal{C}_i$.

The benchmark is built upon OSWorld, comprising 3,321 tasks with 9 types of common corruptions injected, providing YAML configurations to adjust position, intensity, and content. Corruptions are divided into three groups: visual disruptors (pop-ups, resolution changes, screen marks, subtitles, multi-app windows), unexpected operations (accidental touch, app minimization), and environment errors (network errors, login verification).

On the methodology side, the AgentHijack-Agent consists of two roles. The **action generator** is the GUI agent executing the task, based on UI-TARS-1.5-7B and trained via DA-GRPO across different corrupted environments to enhance grounding. The **onlooker** acts as an environmental observer that checks for initial anomalies before execution and continuously summarizes environmental changes during execution, compressing historical screenshots and actions into behavioral descriptions to help the action generator determine if a change was self-induced or an external disturbance.

### Key Designs

1.  **Nine Categories of Configurable Environment Corruptions**:
    *   **Function**: Systematically injects non-ideal scenarios from real desktop usage into OSWorld tasks.
    *   **Mechanism**: Pop-ups find coverable areas and draw induced text; resolution changes resize screenshots directly; marks and subtitles draw visual interference in random or specified regions; multi-apps launch irrelevant apps at initialization; accidental touch clicks random buttons at specified steps; app minimization triggers Win+D; network error blocks external traffic via firewall rules; verification locks the screen with Win+L.
    *   **Design Motivation**: These are not security attacks but frequent disturbances in daily computer use. Configurable parameters allow researchers to vary intensity, content, and timing, preventing the benchmark from testing only fixed failures.

2.  **DA-GRPO for Grounding Enhancement**:
    *   **Function**: Enables the action generator to learn more robust localization and operation strategies across different corrupted environments.
    *   **Mechanism**: For the same task, multiple responses are rolled out in random corruption environments. The reward is composed of task success and format rewards: $r_i=r_i^{success}+r_i^{format}$. If an entire group of rollouts fails, one is replaced with a historical success trajectory from a replay buffer to ensure positive signals for training when advantages would otherwise be zero. The goal is to explicitly include trajectories under different corruptions in GRPO's group relative advantage estimation.
    *   **Design Motivation**: Standard SFT requires massive trajectories and lacks self-correction. Standard GRPO trained in a single environment fails to learn cross-corruption robustness. DA-GRPO exposes grounding training to diverse visual and state changes via data-augmented environmental disturbances.

3.  **Onlooker Behavior Summarization and Environment Checking**:
    *   **Function**: Adds an auxiliary perspective for continuous environment monitoring.
    *   **Mechanism**: Before execution, the onlooker uses an external error knowledge base to determine if the initial environment has network errors or login prompts, requesting re-initialization if necessary. During execution, it records each environmental change and generates a short description $d_t$, transforming the historical context from raw screenshot sequences to screenshot plus behavioral summary sequences.
    *   **Design Motivation**: GUI agents often misattribute external disturbances to their own actions (e.g., getting distracted by a menu opened by an accidental touch). The onlooker's summary helps the model recover the original task context and intercept non-executable environments early.

### Loss & Training

AgentHijack-Agent uses UI-TARS-1.5-7B as the base model, trained on 128 AgentHijack tasks for 15 epochs. Training utilizes VERL with a batch size of 1, rollout number of 4, learning rate of $1\times10^{-6}$, and gradient accumulation of 4, with KL loss disabled to encourage exploration. The default onlooker also uses a fine-tuned UI-TARS-1.5-7B. Screenshots are $1920\times1080$, and the maximum execution steps are 10.

The evaluation covers 9 representative agents, including GLM-4.5V, Llama-3.2-90B-Vision-Instruct, Qwen2.5-VL-72B-Instruct, GPT-4o, Claude-3.7-Sonnet, Gemini-2.5-Pro, UI-TARS-7B-DPO, UI-TARS-72B-DPO, and UI-TARS-1.5-7B. All models use temperature 0.6, top-p 0.9, and a maximum output of 1,500 tokens, with historical context containing up to 15 GUI screenshots.

## Key Experimental Results

### Main Results

| Agent | Clean | Pop ups | Resolution | Accidental Touch | Network Error | Verification | Average |
|-------|-------|---------|------------|------------------|---------------|--------------|---------|
| Qwen2.5-VL-72B-Instruct | 10.99% | 1.86% | 6.38% | 7.48% | 7.48% | 6.63% | 7.47% |
| GPT-4o | 5.38% | 1.44% | 4.82% | 3.12% | 4.24% | 3.25% | 3.69% |
| Gemini-2.5-Pro | 8.11% | 5.20% | 6.98% | 4.61% | 7.02% | 7.81% | 5.82% |
| UI-TARS-72B-DPO | 22.38% | 15.51% | 14.32% | 14.44% | 19.76% | 9.42% | 16.96% |
| UI-TARS-1.5-7B | 24.21% | 10.28% | 11.69% | 22.54% | 22.02% | 10.48% | 18.74% |
| AgentHijack-Agent | 27.80% | 21.51% | 12.53% | 24.37% | 23.09% | 20.15% | 22.89% |

### Ablation Study

| Analysis Item | Setting | Key Metric | Description |
|------|------|---------|------|
| Relative comparison | AgentHijack-Agent vs UI-TARS-1.5-7B | Average +4.15, Clean +3.59, Pop ups +11.23, Verification +9.67 | Onlooker helps most with pop-ups and verification |
| Corruption Intensity | Resolution scaling, marks count, touch frequency | SR decreases as intensity grows, but Ours > Base | DA-GRPO generalizes beyond default intensities |
| Corruption Content | Different pop-up/subtitle text, mark shapes/colors | Performance fluctuates but gains remain stable | Method learns anti-interference strategies, not patterns |
| Corruption Position | Subtitle position, early/middle/late timing | Consistent gains across space and time | Onlooker's summary mitigates timing-based confusion |
| Module Necessity | Remove RL or remove Onlooker | Both lead to significant performance drops | Grounding and onlooker views are complementary |

### Key Findings
- Even large-scale general MLLMs perform poorly in real GUI execution. GPT-4o averages 3.69% and Gemini-2.5-Pro 5.82%, indicating that GUI agent capability cannot be directly extrapolated from VLM QA performance.
- The UI-TARS series is stronger in clean environments but drops significantly under corruption. UI-TARS-1.5-7B drops from 24.21% (clean) to 18.74% (average), with verification and pop-ups being the most vulnerable categories.
- AgentHijack-Agent shows positive gains across all corruption types, especially for pop-ups and verification, proving that environmental checking and behavior summarization reduce meaningless clicks and blind attempts in non-executable environments.
- Improvement in resolution is minimal (+0.84 points), suggesting that coordinate/scaling-based grounding remains a hard problem for GUI agents that DA-GRPO and onlookers alone cannot fully solve.

## Highlights & Insights
- Distinguishing "daily corruption" from "security attacks" is highly valuable. Real-world deployment failures often result from a pop-up or a login page rather than malicious intent; these issues deserve an independent evaluation axis.
- The 9 categories of corruption cover observation space, state transitions, and initial environments, which is more comprehensive than simple visual occlusion. Accidental touch and app minimization specifically test for agent misattribution.
- The onlooker design is simple yet effective, acting as an "environment logger." For long GUI trajectories, compressing history into environmental change summaries is more aligned with human debugging methods than stacking more screenshots.
- The details of DA-GRPO with a replay buffer are practical. Successful GUI trajectories are sparse; if a batch of rollouts all fail, group relative advantage lacks learning signals. Caching successful trajectories ensures the model sees imitable recovery paths.

## Limitations & Future Work
- AgentHijack is based on OSWorld and virtual machine desktops; corruptions in mobile terminals, browsers, remote desktops, and enterprise software may manifest differently.
- Overall success rates remain low, with AgentHijack-Agent averaging only 22.89%. This indicates the framework mitigates the problem but is far from deployable robustness.
- The onlooker increases inference costs and relies on the same fine-tuned model; if the onlooker errs, it might pass incorrect summaries to the action generator.
- The 9 corruption types are pre-defined; while configurable, they do not cover all real-world desktop anomalies. Future work could introduce automatically mined corruptions or real system event streams.
- DA-GRPO training used only 128 tasks, a relatively limited scale. Future research could explore larger corruption curricula and decoupled optimization for visual grounding, coordinate calibration, and state recovery.

## Related Work & Insights
- **vs OSWorld / WebArena / AndroidWorld**: These evaluate task completion in clean environments; AgentHijack injects anomalies into real interaction environments to evaluate deployment robustness.
- **vs Agent-SafetyBench / SafeArena**: Safe-evals focus on high-risk task tendencies; AgentHijack focuses on non-malicious daily corruptions that do not change task goals.
- **vs GUI-Robust / Env. Distractions**: These also look at disturbances but often use QA formats; AgentHijack measures closed-loop failure in executable VMs.
- **vs UI-TARS**: While UI-TARS is a strong baseline, AgentHijack shows its vulnerability to pop-ups/verification; this work builds on UI-TARS using DA-GRPO and onlookers.
- **vs Standard GRPO Training**: Standard GRPO rollouts usually occur in a single environment; DA-GRPO treats corruption as environmental data augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The definition of corruption robustness and the executable benchmark are highly relevant; the framework is intuitive but addresses key failure modes.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 9 corruptions, 9 agents, and ablations on intensity/content/position; however, the training task scale is small, and some ablation figures are primary in the appendix.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and examples; benchmark construction is specific; some formulas and charts are slightly scattered.
- Value: ⭐⭐⭐⭐⭐ Extremely valuable for real-world GUI agent deployment, reminding researchers to look beyond clean task success and test recovery under environmental anomalies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](../../CVPR2026/multimodal_vlm/adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[AAAI 2026\] "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents](../../AAAI2026/multimodal_vlm/are_we_done_yet_a_vision-based_judge_for_autonomous_task_completion_of_computer_.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](../../AAAI2026/multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](../../NeurIPS2025/multimodal_vlm/mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[ICML 2026\] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)

</div>

<!-- RELATED:END -->
