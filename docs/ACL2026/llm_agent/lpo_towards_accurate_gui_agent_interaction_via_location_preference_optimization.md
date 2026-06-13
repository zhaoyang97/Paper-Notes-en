---
title: >-
  [Paper Note] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization
description: >-
  [ACL 2026][LLM Agent][GUI interaction] This paper proposes Location Preference Optimization (LPO), which optimizes the spatial localization accuracy of GUI agents through entropy-based window rewards and physical distanc…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "GUI interaction"
  - "Location Preference Optimization"
  - "reinforcement learning"
  - "information entropy"
  - "GRPO"
date: 2026-05-08
content_hash: 63e6af8c030ff408
---

# LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization

**Conference**: ACL 2026 Findings  
**arXiv**: [2506.09373](https://arxiv.org/abs/2506.09373)  
**Code**: [GitHub](https://github.com/jqtangust/LPO)  
**Area**: GUI Agent  
**Keywords**: GUI interaction, Location Preference Optimization, reinforcement learning, information entropy, GRPO

## TL;DR

This paper proposes Location Preference Optimization (LPO), which optimizes the spatial localization accuracy of GUI agents through entropy-based window rewards and physical distance-based dynamic location rewards combined with the GRPO framework, achieving SOTA results in both offline and online evaluations.

## Background & Motivation

**Background**: Autonomous GUI agents automate graphical user interface operations via natural language and are becoming a significant direction for AI applications. Most GUI agents rely on Supervised Fine-Tuning (SFT) and have achieved initial success in interaction behavior prediction.

**Limitations of Prior Work**: SFT methods face severe challenges in **spatial localization** due to their limited capacity in perceiving and interpreting location data. Although some methods attempt to enhance UI action decision accuracy with Reinforcement Learning (RL), existing RL strategies lack mechanisms for **precisely evaluating interaction location accuracy**: UI-TARS uses text-level exact matching; UI-R1 and InfiGUI-R1 use bounding box IoU; GUI-R1 relies on fixed boundaries. These methods only provide coarse-grained spatial evaluation.

**Key Challenge**: The core of GUI interaction lies in precise coordinate positioning, but existing reward functions fail to capture the **continuous distance relationship** of locations—a prediction close to the target but outside the bounding box receives the same zero reward as one far away.

**Goal**: To design a location-aware preference optimization method to empower GUI agents with more precise spatial interaction capabilities. **Key Insight**: Utilizing information entropy to guide the direction of regional exploration and using physical distance to construct continuous reward signals. **Core Idea**: Users tend to interact in high-information-density areas, and predictions closer to the target should receive higher rewards.

## Method

### Overall Architecture

LPO performs preference optimization on SFT-pretrained GUI agents. GUI interaction is modeled as an MDP where the state $s_t \in \mathbb{R}^{C \times H \times W}$ is a screenshot, and action $a_t = (\mathcal{A}_t \times \mathcal{E}_t)$ consists of the interaction type and coordinates. The reward is the product of a window information density reward $r_w$ and a dynamic location reward $r_d$, optimizing the policy via the GRPO framework.

### Key Designs

1.  **Window Information Density Reward $r_w$**:

    - **Function**: Guides the agent to focus on information-rich areas (e.g., buttons, text boxes) rather than blank space.
    - **Mechanism**: Divides the screenshot into $K = M \times N$ windows, calculates the pixel grayscale entropy $\mathcal{H}_{i,j} = -\sum_{b=1}^{B} p_b(\mathbf{W}_{i,j}) \log_2 p_b(\mathbf{W}_{i,j})$ for each window, maps the interaction coordinates to the corresponding window, and rewards the normalized entropy $r_w = \mathcal{H}_{i^*,j^*} / (\max_{i,j} \mathcal{H}_{i,j} + \epsilon)$.
    - **Design Motivation**: Functional elements (buttons, input boxes) cluster in high-information-density areas; window partitioning aligns with visual tokenizer patches to ensure consistent visual perceptual granularity.

2.  **Dynamic Location Reward $r_d$**:

    - **Function**: Provides continuous, fine-grained feedback on location accuracy based on physical distance.
    - **Mechanism**: Computes the Euclidean distance between predicted coordinates $(x^{*k}, y^{*k})$ and target coordinates $(x^k, y^k)$, linearly mapping it to a reward $r_k = \max(0, 1 - \frac{\sqrt{(x^k - x^{*k})^2 + (y^k - y^{*k})^2}}{d_{\max}})$. It aggregates into $r_d = \frac{1}{K}\sum_{k=1}^{K} r_k$ only when action types match.
    - **Design Motivation**: Overcomes the limitations of fixed bounding box judgments, allowing predictions closer to the target to receive higher rewards and providing smoother optimization signals for gradients.

3.  **Location Preference Optimization (LPO)**:

    - **Function**: Uses location rewards with the GRPO framework for policy optimization via intra-group relative advantages.
    - **Mechanism**: Samples a group of actions $\{a_g\}_{g=1}^{G}$ for each state, calculates the composite reward $r^{(g)} = r_w^{(g)} \cdot r_d^{(g)}$, determines intra-group normalized advantages $A^{(g)}$, and updates the policy using the PPO-clip objective function with KL regularization.
    - **Design Motivation**: GRPO enables broader GUI spatial exploration; intra-group relative comparisons effectively distinguish the quality of different position predictions.

### Loss & Training

The SFT phase uses multiple internal datasets to train basic interaction capabilities. The RL phase uses preference data from datasets such as Mind2Web, AITZ, and OmniAct. Learning rate is $1 \times 10^{-6}$, lower clip range $\epsilon_1 = 0.2$, upper clip range $\epsilon_2 = 0.28$, and KL coefficient $\beta = 1 \times 10^{-4}$. The base model is Ovis2 8B. Training took approximately 300 H100 GPU hours.

## Key Experimental Results

### Main Results

| Benchmark | Metric | LPO | GUI-R1 | InfiGUI-R1 | UI-R1 | Base SFT |
|------|------|-----|--------|------------|-------|----------|
| Mind2Web Cross-Task | Step SR | **49.5** | 46.6 | 35.8 | 24.9 | 38.2 |
| Mind2Web Cross-Task | Ele.Acc | **64.3** | 62.5 | 62.6 | 59.5 | 60.3 |
| VisualWebBench | Average | **79.5** | 78.8 | 78.5 | 78.7 | 78.7 |
| ScreenSpot V2 | Average | **90.5** | 88.7 | 89.5 | 88.2 | 89.5 |
| WebVoyager | Overall | **57.6** | 37.5 | 54.1 | 47.3 | 48.0 |

### Ablation Study

| Configuration | Step SR (Cross-Task) | Ele.Acc | Description |
|------|---------------------|---------|------|
| LPO (Full) | **49.5** | **64.3** | Full Model |
| w/o $r_d$ | 42.3 | 56.7 | Without dynamic location reward, element accuracy drops significantly |
| w/o $r_w$ | 46.4 | 62.7 | Without window info-density reward, overall accuracy decreases |

### Key Findings
- LPO achieves SOTA on both offline benchmarks (Mind2Web, VisualWebBench, ScreenSpot V2) and online evaluation (WebVoyager).
- The dynamic location reward $r_d$ has the greatest impact on element positioning accuracy (Ele.Acc), which drops by 7.6% when removed.
- The window information density reward $r_w$ is more important for decision accuracy, as removing it leads to a 3.1% decrease in Step SR.
- Existing baseline methods (UI-R1, GUI-R1) show local advantages on certain websites, but their overall consistency is significantly lower than LPO.

## Highlights & Insights
- Entropy-driven window reward is a simple but effective prior—functional regions indeed have higher information density, and this can be transferred to other visual interaction tasks.
- Replacing discrete bounding box judgments with continuous distance rewards is a natural and elegant improvement that eliminates the influence of arbitrary human thresholds.
- The multiplicative combination of the two rewards forces the agent to optimize both "looking at the right area" and "clicking the right spot," balancing macro and micro perspectives.
- The GRPO-based exploration mechanism is well-suited for GUI scenarios involving large spaces and sparse rewards.
- Validation through online evaluation (WebVoyager) strengthens the evidence for the method's practical application value.

## Limitations & Future Work
- High dependency on large-scale grounding datasets with precise annotations; the cost of data collection and labeling limits practical scalability.
- Training requires approximately 300 GPU hours of computational resources, limiting real-time application and usage by smaller teams.
- Window partitioning depends on the visual tokenizer's patch scheme, and generalization across different base models needs further verification.
- Information entropy rewards may not be robust enough for certain special interfaces (e.g., a few high-contrast elements on an all-white background).
- Future work could explore self-supervised location rewards that do not require ground-truth coordinates, as well as joint optimization with multi-step planning.

## Related Work & Insights
- **vs UI-TARS**: UI-TARS uses DPO, requiring manually constructed positive and negative sample pairs; LPO is based on GRPO's automated exploration, reducing human dependency.
- **vs GUI-R1**: GUI-R1 uses fixed position boundaries as rewards; LPO's continuous distance reward is more precise.
- **vs InfiGUI-R1**: InfiGUI-R1 uses bounding box IoU; LPO directly uses coordinate distance, providing finer granularity.

## Rating
- Novelty: ⭐⭐⭐⭐ Information entropy window rewards and dynamic distance rewards are meaningful innovations for GUI RL reward design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 offline benchmarks + 1 online benchmark, with fair comparisons against 4 RL baselines and clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The motivation diagram (Figure 1) intuitively demonstrates the limitations of existing methods, and the methodological derivation is clear.
- Value: ⭐⭐⭐⭐ Provides a practical and effective RL training strategy for precise interaction in GUI agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DEPO: Dual-Efficiency Preference Optimization for LLM Agents](../../AAAI2026/llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ICML 2026\] Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](../../ICML2026/llm_agent/video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

</div>

<!-- RELATED:END -->
