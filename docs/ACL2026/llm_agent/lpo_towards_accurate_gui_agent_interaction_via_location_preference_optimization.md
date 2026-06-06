---
title: >-
  [Paper Note] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization
description: >-
  [ACL 2026][LLM Agent][GUI Interaction] This paper proposes Location Preference Optimization (LPO), which combines entropy-based window rewards and distance-based dynamic location rewards within the GRPO framework to impr…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "GUI Interaction"
  - "Location Preference Optimization"
  - "Reinforcement Learning"
  - "Information Entropy"
  - "GRPO"
date: 2026-05-08
content_hash: 91add2dfee78bb3b
---

# LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization

**Conference**: ACL 2026
**arXiv**: [2506.09373](https://arxiv.org/abs/2506.09373)  
**Code**: [GitHub](https://github.com/jqtangust/LPO)  
**Area**: GUI Agents
**Keywords**: GUI Interaction, Location Preference Optimization, Reinforcement Learning, Information Entropy, GRPO

## TL;DR

This paper proposes Location Preference Optimization (LPO), which combines entropy-based window rewards and distance-based dynamic location rewards within the GRPO framework to improve the spatial grounding accuracy of GUI agents, achieving state-of-the-art performance on both offline and online benchmarks.

## Background & Motivation

**Background**: Autonomous GUI agents automate graphical user interface operations using natural language as an intermediary and are becoming an important direction in AI applications. Most GUI agents rely on supervised fine-tuning (SFT) and have achieved preliminary success in predicting interactive behaviors.

**Limitations of Prior Work**: SFT-based methods face significant challenges in **spatial grounding**, as their ability to perceive and interpret positional data is limited. While some approaches attempt to enhance UI action decision accuracy via reinforcement learning (RL), existing RL strategies lack mechanisms to **precisely evaluate the accuracy of interaction locations**: UI-TARS relies on text-level exact matching; UI-R1 and InfiGUI-R1 use bounding box IoU; GUI-R1 depends on fixed positional boundaries. These methods provide only coarse-grained spatial evaluation.

**Key Challenge**: The essence of GUI interaction lies in precise coordinate localization, yet existing reward functions cannot capture the **continuous distance relationship** of locations — predictions that are close to the target but outside the bounding box receive the same zero reward as those far from the target.

**Goal**: To design a location-aware preference optimization method that endows GUI agents with more precise spatial interaction capabilities. **Key Insight**: Leverage information entropy to guide regional exploration, and use physical distance to construct a continuous reward signal. **Core Idea**: Users tend to interact in regions of high information density, and predictions closer to the target should receive higher rewards.

## Method

### Overall Architecture

LPO performs preference optimization on top of an SFT-pretrained GUI agent. GUI interaction is modeled as an MDP, where the state $s_t \in \mathbb{R}^{C \times H \times W}$ is a screenshot of the interface and the action $a_t = (\mathcal{A}_t \times \mathcal{E}_t)$ comprises the interaction type and coordinates. The reward is the product of a window information density reward $r_w$ and a dynamic location reward $r_d$, and the policy is optimized within the GRPO framework.

### Key Designs

1. **Window Information Density Reward $r_w$**:

    - **Function**: Guides the agent to focus on information-rich regions of the interface (e.g., buttons, text fields) rather than blank areas.
    - **Mechanism**: The interface screenshot is divided into $K = M \times N$ windows; the pixel grayscale information entropy of each window is computed as $\mathcal{H}_{i,j} = -\sum_{b=1}^{B} p_b(\mathbf{W}_{i,j}) \log_2 p_b(\mathbf{W}_{i,j})$; the interaction coordinates are mapped to their corresponding window; and the reward is the normalized entropy $r_w = \mathcal{H}_{i^*,j^*} / (\max_{i,j} \mathcal{H}_{i,j} + \epsilon)$.
    - **Design Motivation**: Functional elements (buttons, input fields) cluster in high information density regions; the window partitioning aligns with the visual tokenizer's patch scheme, ensuring consistency in visual perception granularity.

2. **Dynamic Location Reward $r_d$**:

    - **Function**: Provides continuous and fine-grained feedback on positional accuracy based on physical distance.
    - **Mechanism**: The Euclidean distance between the predicted coordinate $(x^{*k}, y^{*k})$ and the target coordinate $(x^k, y^k)$ is computed and linearly mapped to a reward $r_k = \max(0, 1 - \frac{\sqrt{(x^k - x^{*k})^2 + (y^k - y^{*k})^2}}{d_{\max}})$; rewards are aggregated only when the action type matches: $r_d = \frac{1}{K}\sum_{k=1}^{K} r_k$.
    - **Design Motivation**: Overcomes the limitations of fixed bounding box judgment, allows predictions closer to the target to receive higher rewards, and provides a smoother optimization signal.

3. **Location Preference Optimization (LPO)**:

    - **Function**: Leverages location rewards within the GRPO framework to construct within-group relative advantages for policy optimization.
    - **Mechanism**: A group of actions $\{a_g\}_{g=1}^{G}$ is sampled for each state; the combined reward $r^{(g)} = r_w^{(g)} \cdot r_d^{(g)}$ is computed; within-group normalized advantages $A^{(g)}$ are calculated; and the policy is updated using a PPO-clip objective with KL regularization.
    - **Design Motivation**: GRPO supports broader spatial exploration of the GUI space, and within-group relative comparison effectively distinguishes the quality of different positional predictions.

### Loss & Training

The SFT stage uses multiple in-house datasets to train basic interaction capabilities. The RL stage uses preference data from MMind2Web, AITZ, OmniAct, and other datasets. The learning rate is $1 \times 10^{-6}$, lower clip range $\epsilon_1 = 0.2$, upper clip range $\epsilon_2 = 0.28$, and KL coefficient $\beta = 1 \times 10^{-4}$. The base model is Ovis2 8B. Training requires approximately 300 H100 GPU hours.

## Key Experimental Results

### Main Results

| Benchmark | Metric | LPO | GUI-R1 | InfiGUI-R1 | UI-R1 | Base SFT |
|-----------|--------|-----|--------|------------|-------|----------|
| Mind2Web Cross-Task | Step SR | **49.5** | 46.6 | 35.8 | 24.9 | 38.2 |
| Mind2Web Cross-Task | Ele.Acc | **64.3** | 62.5 | 62.6 | 59.5 | 60.3 |
| VisualWebBench | Average | **79.5** | 78.8 | 78.5 | 78.7 | 78.7 |
| ScreenSpot V2 | Average | **90.5** | 88.7 | 89.5 | 88.2 | 89.5 |
| WebVoyager | Overall | **57.6** | 37.5 | 54.1 | 47.3 | 48.0 |

### Ablation Study

| Configuration | Step SR (Cross-Task) | Ele.Acc | Note |
|---------------|---------------------|---------|------|
| LPO (Full) | **49.5** | **64.3** | Full model |
| w/o $r_d$ | 42.3 | 56.7 | Removing the dynamic location reward leads to a significant drop in element accuracy |
| w/o $r_w$ | 46.4 | 62.7 | Removing the window information density reward reduces overall accuracy |

### Key Findings
- LPO achieves state-of-the-art performance on both offline benchmarks (Mind2Web, VisualWebBench, ScreenSpot V2) and the online evaluation (WebVoyager).
- The dynamic location reward $r_d$ has the largest impact on element grounding accuracy (Ele.Acc), with a 7.6% drop upon removal.
- The window information density reward $r_w$ is more critical for decision accuracy, with a 3.1% drop in Step SR upon removal.
- Existing baselines (UI-R1, GUI-R1) show local advantages on certain websites but are far less consistent than LPO overall.

## Highlights & Insights
- The entropy-driven window reward is a simple yet effective prior — functional regions do exhibit higher information density, and this principle is transferable to other visual interaction tasks.
- Replacing discrete bounding box judgment with continuous distance reward is a natural and elegant improvement that eliminates the influence of manually defined thresholds.
- The multiplicative combination of the two rewards enables the agent to simultaneously optimize "attending to the correct region" and "clicking the precise location," balancing macro- and micro-level spatial accuracy.
- The GRPO-based exploration mechanism is well-suited to GUI environments characterized by large action spaces and sparse rewards.
- Validation on the online evaluation (WebVoyager) strengthens the practical applicability of the proposed method.

## Limitations & Future Work
- The method is highly dependent on large-scale grounding datasets with precise annotations; the high cost of data collection and labeling limits broader adoption.
- Training requires approximately 300 GPU hours of computation, constraining real-time application and use by small research teams.
- Window partitioning relies on the visual tokenizer's patch scheme, and generalizability to different base models remains to be verified.
- The information entropy reward may be insufficiently robust for certain special interfaces (e.g., a small number of high-contrast elements on a fully white background).
- Future work may explore self-supervised location rewards that do not require ground-truth coordinates, as well as joint optimization with multi-step planning.

## Related Work & Insights
- **vs. UI-TARS**: UI-TARS employs DPO, which requires manual construction of positive-negative sample pairs; LPO uses GRPO for automatic exploration, reducing human effort.
- **vs. GUI-R1**: GUI-R1 uses fixed positional boundaries as rewards, whereas LPO's continuous distance reward provides greater precision.
- **vs. InfiGUI-R1**: InfiGUI-R1 uses bounding box IoU, while LPO directly uses coordinate distance, offering finer granularity.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The entropy-based window reward and dynamic distance reward represent meaningful innovations in GUI RL reward design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 3 offline benchmarks and 1 online benchmark, with fair comparison against 4 RL baselines and clear ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation figure (Figure 1) intuitively illustrates the limitations of prior methods, and the method derivation is clear.
- **Value**: ⭐⭐⭐⭐ Provides a practical and effective RL training strategy for precise GUI agent interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DEPO: Dual-Efficiency Preference Optimization for LLM Agents](../../AAAI2026/llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ICML 2026\] Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](../../ICML2026/llm_agent/video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[ICCV 2025\] UIPro: Unleashing Superior Interaction Capability for GUI Agents](../../ICCV2025/llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)

</div>

<!-- RELATED:END -->
