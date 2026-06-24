---
title: >-
  [Paper Note] Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions
description: >-
  [ACL 2025][LLM Agent][GUI agent] This paper presents the first systematic study on the vulnerability of multimodal GUI agents to environmental distractions (e.g., pop-up ads, recommended content). In natural, non-adversarial scenarios, even the most advanced MLLMs (including GPT-4o) exhibit a 20-40% probability of being distracted by irrelevant environmental content, leading them to execute actions that deviate from the user's objectives.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "GUI agent"
  - "environmental distraction"
  - "faithfulness"
  - "multimodal agent"
  - "robustness"
date: 2026-05-08
content_hash: 5416cf935cfb2b85
---

# Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions

**Conference**: ACL 2025  
**arXiv**: [2408.02544](https://arxiv.org/abs/2408.02544)  
**Code**: [https://github.com/xbmxb/EnvDistraction](https://github.com/xbmxb/EnvDistraction)  
**Area**: LLM Agent / GUI Agent / Security  
**Keywords**: GUI agent, environmental distraction, faithfulness, multimodal agent, robustness

## TL;DR
This paper presents the first systematic study on the vulnerability of multimodal GUI agents to environmental distractions (e.g., pop-up ads, recommended content). In natural, non-adversarial scenarios, even the most advanced MLLMs (including GPT-4o) exhibit a 20-40% probability of being distracted by irrelevant environmental content, leading them to execute actions that deviate from the user's objectives.

## Background & Motivation

**Background**: GUI agents leverage MLLMs to perceive screenshots and predict actions (e.g., clicks, inputs), serving as a crucial application scenario for LLM agents. Current research primarily focuses on improving action prediction accuracy.

**Limitations of Prior Work**:
   - Prior studies assume a "clean" environment, overlooking the substantial amount of distracting information (ads, pop-ups, recommended content, etc.) in real-world GUI environments.
   - Security research tends to focus on adversarial attacks and jailbreaking, while neglecting **non-malicious yet distracting** environmental content.
   - There is a lack of systematic evaluation regarding the agent's faithfulness against environmental distractions.

**Key Challenge**: GUI agents must perceive the entire screen to comprehend the environment, yet the screen inevitably contains enticing, irrelevant content that is completely separate from the user's goals.

**Goal**: To quantify the vulnerability of GUI agents to environmental distractions and analyze the key factors influencing this susceptibility.

**Key Insight**: A general setting is proposed: the user is benign, the agent is benign, and the environment is non-malicious but distracting—which is closer to real-world scenarios than adversarial attacks.

**Core Idea**: By injecting four categories of natural distractions (pop-ups, search recommendations, content recommendations, and chat messages) into GUI screens and evaluating 10 MLLMs, the authors demonstrate that even the strongest models struggle to remain faithful to the user's goals.

## Method

### Overall Architecture
Constructing a GUI environment with distractions $\to$ Defining three working modes (at different perception levels) $\to$ Evaluating 10 MLLMs $\to$ Categorizing actions into Gold (correct), Distracted, and Invalid $\to$ Analyzing the factors driving distractions.

### Key Designs

1. **Four Categories of Distraction Scenarios**:

    - **Pop-up**: Ad or promotional pop-ups overlaying the page.
    - **Search**: Sponsored or promoted content within search results.
    - **Recommendation**: Recommended content in information feeds.
    - **Chat**: Instant messaging or social media notifications.
    - **Design Motivation**: Covering the most prevalent distraction types encountered during real-world GUI usage.

2. **Three Working Modes**:

    - **HTML Mode**: The agent only reads HTML code (minimal perception).
    - **Screenshot Mode**: The agent views the screenshot (visual perception).
    - **Screenshot + Label Mode**: Element annotations (Set-of-Mark) are overlaid on the screenshot (maximal perception).
    - **Design Motivation**: Testing whether the impact of distractions varies across different levels of perception.

3. **Evaluation Framework**:

    - Each action is classified as Gold (correctly executing the user's target), Distracted (interacting with the distracting content), or Invalid (other invalid actions).
    - $$\text{Distraction Rate} = |a_{dist}| / (|a_{gold}| + |a_{dist}| + |a_{other}|)$$
    - **Design Motivation**: Distinguishing between being "distracted" and "general failures"—as the former is more critical since it can lead to uncontrollable behaviors.

## Key Experimental Results

### Main Results (Distraction Rates of 10 MLLMs)

| Model | Pop-up | Search | Recommend | Chat | Average Distraction Rate |
|------|--------|--------|-----------|------|----------|
| GPT-4o | ~20% | ~25% | ~30% | ~15% | ~22% |
| Claude 3.5 | ~25% | ~28% | ~35% | ~20% | ~27% |
| Qwen-VL | ~30% | ~35% | ~40% | ~25% | ~32% |
| Specialized GUI Agent | ~25% | ~30% | ~35% | ~20% | ~27% |

### Ablation Study: Impact of Different Perception Modes

| Perception Mode | Average Gold Rate | Average Distraction Rate |
|-----------------|-------------------|-------------------------|
| HTML | ~55% | ~15% |
| Screenshot | ~50% | ~28% |
| Screenshot + Label | ~52% | ~30% |

### Key Findings
- **All evaluated models are vulnerable to environmental distractions**: This includes state-of-the-art models like GPT-4o and specialized GUI agents, without exception.
- **Visual perception counterintuitively increases the distraction rate**: The Screenshot mode exhibits higher distraction rates than the HTML mode, as visual distractions (e.g., colorful buttons and pop-ups) are more enticing than distractions hidden within HTML code.
- **Recommendation and Search pose the most severe distractions**: Since these distractions share some semantic relevance with the user's goals, they are highly confusing to the agents.
- **Enhancing perception fails to resolve the issue**: Set-of-Mark annotations provide the agent with more details, but also expose it to more distracting elements.
- **Adversarial environment injection is feasible**: Attackers can potentially manipulate the agent's behavior remotely by meticulously crafting distracting content.

## Highlights & Insights
- **"Environmental faithfulness" represents an overlooked critical dimension for GUI agents**: While current research heavily optimizes task success rate, executing actions safely is impossible if an agent cannot even resist ad distractions. This concept can be adapted for the safety evaluation of any interactive agent.
- **The counterintuitive finding that "stronger visual perception leads to higher susceptibility to distraction"**: This suggests that the visual capability of multimodal agents is a double-edged sword; improvements in perception must be coupled with enhanced judgment.
- **The importance of distinguishing "benign-but-risky" scenarios from "adversarial attacks"**: Most security research inspects extreme attack vectors, whereas actual deployment environments are far more likely to present natural distractions.

## Limitations & Future Work
- **Evaluations are based on simulated datasets**: Distractions in real-world GUI environments are more complex.
- **Action evaluation is limited to single steps**: The cumulative effects of distractions across multi-step execution trajectories remain unstudied.
- **Defensive approaches (such as custom prompting) show limited efficacy**: More fundamental solutions are required.
- **Fine-tuned GUI agents under distraction conditions were not evaluated**: Injecting distractions directly into training data might provide a viable defense.

## Related Work & Insights
- **vs ToolEmu (Ruan et al., 2024)**: ToolEmu concentrates on security risks arising from malicious inputs, whereas this paper highlights faithfulness risks from a natural environment—making them highly complementary.
- **vs WebArena (Zhou et al., 2024)**: WebArena evaluates models in a "clean" environment, whereas this research advocates for integrating environmental distractions into current benchmarks.
- **vs R2D2**: R2D2 utilizes a replay buffer to enhance navigation, yet omits consideration of on-screen environmental distractions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study on the vulnerability of GUI agents to environmental distractions, filling a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extremely comprehensive, evaluating 10 models across 4 distraction types and 3 perception modes.
- Writing Quality: ⭐⭐⭐⭐ Demarcates the problem definition clearly, distinguishing it from existing security research.
- Value: ⭐⭐⭐⭐⭐ Holds critical cautionary significance for the safe deployment of GUI agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Adaptation for LLM Agents via Environment Interaction](../../ICLR2026/llm_agent/test-time_adaptation_for_llm_agents_via_environment_interaction.md)
- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](../../ICLR2026/llm_agent/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[ICML 2026\] MacArena: Benchmarking Computer Use Agents on an Online macOS Environment](../../ICML2026/llm_agent/macarena_benchmarking_computer_use_agents_on_an_online_macos_environment.md)
- [\[ACL 2025\] Browsing Like Human: A Multimodal Web Agent with Experiential Fast-and-Slow Thinking](browsing_like_human_a_multimodal_web_agent_with_experiential_fast-and-slow_think.md)

</div>

<!-- RELATED:END -->
