---
title: >-
  [Paper Note] M³HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality
description: >-
  [ICML 2025][LLM Alignment][Multi-agent Reinforcement Learning] The M³HF framework is proposed to integrate multi-phase, mixed-quality natural language human feedback during multi-agent reinforcement learning. It leverages LLMs to parse feedback, and updates the reward function through predefined templates and adaptive weights, significantly improving multi-agent coordination performance.
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "Multi-agent Reinforcement Learning"
  - "Human Feedback"
  - "Reward Design"
  - "LLM Parsing"
  - "Adaptive Weights"
  - "Mixed Quality Feedback"
date: 2026-05-08
content_hash: e9d14b17c3b3b324
---

# M³HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality

**Conference**: ICML 2025

**arXiv**: [2503.02077](https://arxiv.org/abs/2503.02077)

**Code**: Not released

**Area**: LLM Alignment

**Keywords**: Multi-agent Reinforcement Learning, Human Feedback, Reward Design, LLM Parsing, Adaptive Weights, Mixed Quality Feedback

## TL;DR

The M³HF framework is proposed to integrate multi-phase, mixed-quality natural language human feedback during multi-agent reinforcement learning. It leverages LLMs to parse feedback, and updates the reward function through predefined templates and adaptive weights, significantly improving multi-agent coordination performance.

## Background & Motivation

Reward function design in multi-agent reinforcement learning (MARL) remains a core challenge. In complex cooperative environments, hand-crafted reward functions often lead to sub-optimal or misaligned behaviors. Existing approaches face the following issues:

**Sparse Reward Dilemma**: Final reward signals for cooperative tasks (e.g., cooking, football) are sparse, making it difficult for agents to learn effective coordination through trial and error.

**Reward Engineering Challenges**: Designing dense rewards by hand requires extensive domain knowledge, and improper designs can easily lead to reward hacking.

**Limitations of Single-Phase Feedback**: Traditional RLHF methods collect feedback only pre- or post-training, which fails to dynamically adapt to the agents' actual behaviors.

**Inconsistent Feedback Quality**: The expertise of different human evaluators varies significantly, requiring the system to handle feedback of mixed quality.

The core insight of M³HF is that by strategically pausing training multiple times, collecting human feedback, and adaptively integrating feedback of varying quality, the system can guide multi-agent collaborative policy learning more efficiently.

## Method

### Overall Architecture

M³HF extends the standard Markov Game into a **Multi-phase Human Feedback Markov Game (MHF-MG)**, which consists of the following pipeline:

1. **Training Phase**: Agents undergo standard training for a certain period.
2. **Evaluation Pause**: Training is strategically paused to display videos of agent behaviors to human evaluators.
3. **Feedback Collection**: Human evaluators provide natural language feedback (e.g., "put the ingredients on the central table").
4. **LLM Parsing**: LLMs are utilized to parse natural language feedback into structured reward signals.
5. **Reward Update**: Reward functions are generated from predefined templates and integrated using adaptive weights.
6. **Resumed Training**: Training resumes with the updated reward functions.

### Key Designs: LLM Feedback Parsing

Human feedback is parsed by an LLM into a structured format containing:

- **Target Agent**: Which agent(s) the feedback is directed toward.
- **Reward Type**: Matches a predefined reward template (e.g., distance-based, action-based).
- **Parameter Settings**: Specific parameter values within the templates.

Predefined reward templates include:
- **Distance-based**: Rewards agents for moving closer to/further from specific locations.
- **Action-based**: Rewards execution of specific action sequences.
- **Coordination-based**: Rewards cooperative behaviors among multiple agents.

### Key Designs: Adaptive Weight Adjustment

At the $k$-th phase, the combined reward function is formulated as:

$$\hat{R}_k = \sum_{m=0}^{k} w_m^k R_m$$

where $R_0$ is the original environment reward, and $R_m$ ($m > 0$) is the reward generated from human feedback. The weight update mechanism includes:

1. **Weight Decay**: Gradually reduces the influence of historical feedback over time, steering the system to focus on the latest feedback.
2. **Performance-based Adjustment**: If performance degrades after introducing new feedback ($\Delta r_k < 0$), the weight of that feedback is automatically clipped to zero.
3. **Incremental Validation**: Evaluates the actual utility of new reward functions through rollouts, keeping only beneficial feedback.

### Theoretical Guarantee

**Proposition 4.2 (Robustness to Flawed Feedback)**: After any $K$ rounds of feedback:

$$J_{\text{ori}}(\pi_K) - J_{\text{ori}}(\pi_0) \geq \sum_{j=1}^{n(K)} \Delta r_{i_j} - \delta$$

where $\delta$ is a bounded positive constant, and $i_j$ is the index of the $j$-th valid feedback. This indicates that the algorithm benefits from every high-quality feedback, whereas performance degradation is at most bounded by the latest erroneous feedback.

## Key Experimental Results

### Main Results: Overcooked Environment

| Method | Layout A (Simple) | Layout B (Complex) | Layout C (Most Complex) |
|------|-----------------|-----------------|-------------------|
| IPPO | 19.2 ± 4.5 | 23.1 ± 2.7 | 27.4 ± 4.9 |
| MAPPO | Lower than IPPO | Lower than IPPO | Lower than IPPO |
| Mac-based | Moderate | Moderate | Moderate |
| **M³HF** | **164.8 ± 1.2** | **Significantly outperforms baselines** | **Significantly outperforms baselines** |

M³HF consistently and significantly outperforms the baselines across all layouts and recipe settings, yielding up to a 50% performance improvement in complex tasks.

### Ablation Study

| Experiment | Finding |
|------|------|
| Mixed-quality feedback | Even with incorrect feedback, M³HF performs close to the oracle and does not degrade significantly. |
| Deliberately misleading feedback | The adaptive weight mechanism effectively suppresses the negative impacts of misleading feedback. |
| Removing LLM parsing | Performance drops, validating the necessity of the LLM parsing component. |
| Fixed weights vs. adaptive weights | Adaptive weights significantly outperform the fixed weight strategy. |
| Replacing humans with VLMs | Gemini-1.5-Pro can generate human-like feedback, but still falls short on specificity compared to humans. |
| Google Football 5v5 | M³HF continues to outperform standard MARL baselines in more complex environments. |

### Training Efficiency

- **Feedback Rounds**: Only 2-5 rounds of human feedback are required.
- **Training Acceleration**: 3×-6× acceleration (15k episodes vs. 80k-100k episodes for traditional methods).
- **Human Labor Cost**: Approximately 3-5 minutes per feedback round, totaling under 25 minutes.
- **Time Saved**: Saves 10-15 hours of training time.

## Highlights & Insights

1. **Human-AI Collaboration Paradigm**: Differing from conventional offline preference collection in RLHF, M³HF achieves online human guidance during training, tightly coupling feedback with policy behaviors.
2. **Handling Mixed Quality**: The adaptive weight mechanism elegantly addresses the challenge of integrating feedback of varying quality, enabling non-experts to contribute effectively.
3. **LLM as a Bridge**: Utilizing LLMs to translate natural language feedback into structured rewards lowers the barrier for humans to provide feedback.
4. **High Practicality**: Only a few rounds of human feedback (2-5 rounds) are needed to drastically improve performance, resulting in very low practical deployment costs.
5. **Exploring VLM Alternatives**: Feasibility of replacing humans with VLMs is initially validated, paving the way for complete automation.

## Limitations & Future Work

1. **Dependency on Domain Knowledge for Reward Templates**: Predefined templates must be manually crafted for new environments.
2. **Limited Environmental Evaluation**: Primarily validated on Overcooked; although extended to Google Research Football, generalization to more complex environments remains to be studied.
3. **Strong Theoretical Assumptions**: Assumptions such as Gaussian noise and ergodicity may not always hold in realistic deployment scenarios.
4. **Inadequate VLM Feedback Precision**: Feedbacks currently generated by VLMs are often overly generic (e.g., "improve coordination") and lack concrete guidance.
5. **Lack of Direct Comparison with Other Human-Feedback MARL Methods**: Direct comparisons with preference-based reinforcement learning approaches like PbMARL are missing.

## Related Work & Insights

- **RLHF for LLM (Ouyang et al.)**: M³HF generalizes RLHF methodologies to multi-agent and multi-phase scenarios.
- **LLM-Driven Reward Design (Motif, etc.)**: The innovation of M³HF lies in multi-phase and multi-agent feedback allocation.
- **MARL (MAPPO/IPPO)**: M³HF serves as an orthogonal enhancement to these foundational algorithms.
- **Insight**: Multi-phase feedback collection strategies can be generalized to single-agent LLM alignment, suggesting that collecting feedback at training midpoints might be more efficient than doing so only pre- or post-training.

## Rating

- Novelty: ⭐⭐⭐⭐ — Integrating multi-phase, multi-agent human feedback presents a novel problem formulation.
- Experimental Thoroughness: ⭐⭐⭐ — Sufficiently ablated but limited in environment diversity (primarily focused on Overcooked).
- Value: ⭐⭐⭐⭐ — Substantial performance improvements can be achieved with minimal feedback, leading to low deployment costs.
- Recommendation Index: ⭐⭐⭐⭐ — Recommended reading for researchers focusing on the intersection of MARL and RLHF.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](../../NeurIPS2025/llm_alignment/strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[ACL 2025\] Curiosity-Driven Reinforcement Learning from Human Feedback](../../ACL2025/llm_alignment/curiosity_driven_rlhf.md)
- [\[ACL 2025\] Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](../../ACL2025/llm_alignment/debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)
- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](../../ACL2025/llm_alignment/expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)
- [\[ICML 2025\] AMPO: Active Multi-Preference Optimization for Self-play Preference Selection](ampo_active_multi-preference_optimization_for_self-play_preference_selection.md)

</div>

<!-- RELATED:END -->
