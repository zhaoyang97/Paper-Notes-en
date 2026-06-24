---
title: >-
  [Paper Note] Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning
description: >-
  [ICML 2025][Multimodal VLM][VLM Agent] This paper proposes Counterfactual Soft Reinforcement Learning (CoSo), which leverages counterfactual reasoning to evaluate the causal impact of each token on the final action. By incorporating causally weighted entropy regularization to concentrate exploration on key tokens, CoSo addresses the text action space explosion in online RL fine-tuning for VLM agents. Experimental results demonstrate performance gains of 12.3%, 9.3%…
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "VLM Agent"
  - "Counterfactual Reasoning"
  - "Soft RL"
  - "Exploration Efficiency"
  - "causal inference"
date: 2026-05-08
content_hash: 24720d919d6b5366
---

# Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning

**Conference**: ICML 2025  
**arXiv**: [2505.03792](https://arxiv.org/abs/2505.03792)  
**Code**: [github.com/langfengQ/CoSo](https://github.com/langfengQ/CoSo)  
**Area**: Multimodal VLM  
**Keywords**: VLM Agent, Counterfactual Reasoning, Soft RL, Exploration Efficiency, causal inference

## TL;DR

This paper proposes Counterfactual Soft Reinforcement Learning (CoSo), which leverages counterfactual reasoning to evaluate the causal impact of each token on the final action. By incorporating causally weighted entropy regularization to concentrate exploration on key tokens, CoSo addresses the text action space explosion in online RL fine-tuning for VLM agents. Experimental results demonstrate performance gains of 12.3%, 9.3%, and 16.7% on Android control, card reasoning games, and embodied AI tasks, respectively.

## Background & Motivation

VLMs are widely employed as decision-making agents in scenarios such as device control, gaming, and robotics. Online RL fine-tuning enables VLM agents to iteratively interact with environments and optimize multi-step objectives, but it faces two primary challenges:

**Explosion of Exploration Space**: While traditional RL action spaces are small (e.g., $|\mathcal{A}|=6$), the textual action space of a VLM scales up to $|V|^n = 32100^{100}$.

**Non-End-to-End Action Generation**: The output of a VLM must be converted into an executable action via a parsing function $f^{\text{parse}}$. Consequently, a vast number of tokens (e.g., chain-of-thought steps, fixed formatting) exert zero influence on the final action.

Key Insight: **Only a small fraction of "action-critical" tokens (< 10%) truly determine the final action**. Standard entropy regularization imposes uniform uncertainty across all tokens, leading to inefficient or even detrimental exploration.

## Method

### Overall Architecture

CoSo consists of three phases:
1. **Rollout**: The VLM agent interacts with the environment to collect data.
2. **Counterfactual Reasoning**: The causal weight of each token is computed.
3. **CoSo Update**: The VLM is optimized using causally weighted entropy regularization.

### Token-to-Action Causal Analysis

A Structural Causal Model (SCM) is used to characterize the relationship between token $y^i$ and action $a$:

$$\mathcal{B}_{y \to a}^i = |\mathbb{P}(a|y, \epsilon_a) - \mathbb{P}(a|y^{-i} \cup y^i_{\text{null}}, \epsilon_a)|$$

A "counterfactual intervention" is performed on each token by replacing it with a null value and observing the change in the probability of the parsed action. A larger deviation implies higher causal importance.

The SCM is instantiated using a lightweight BERT model (0.01B parameters).

### Causally Weighted Entropy Regularization

Standard soft RL:
$$\mathcal{H}(\pi(\cdot|s)) = \sum_{i=1}^n \mathcal{H}(y^i|y^{1:i-1})$$

CoSo modifies this to:
$$\mathcal{H}^{\mathcal{B}}(\pi(\cdot|s)) = \sum_{i=1}^n \mathcal{B}_{y \to a}^i \cdot \mathcal{H}(y^i|y^{1:i-1})$$

Key tokens (with high causal weights) receive more exploration, while exploration for low-impact tokens is suppressed.

### Theoretical Guarantees

- **Lemma 4.2 (Convergence of Policy Evaluation)**: The Bellman backup operator $\mathcal{T}^{\mathcal{B}}$ in CoSo converges to a fixed point.
- **Lemma 4.3 (Policy Improvement)**: $Q^{\tilde{\pi}}(s,a) \geq Q^{\pi}(s,a)$.
- **Proposition 4.4 (Policy Iteration)**: Repeated application of policy evaluation and policy improvement converges to the optimal policy.

### Implementation

- Offline Phase: SFT initializes the VLM to learn task formats and valid outputs.
- Online Phase: RL + CoSo fine-tuning (can be based on AWR or PPO).

## Key Experimental Results

### Android-in-the-Wild

| Method | General Train | General Test | Web Train | Web Test | Average |
|------|------|------|------|------|------|
| GPT-4V + AppAgent | 13.5 | 17.7 | 12.5 | 8.3 | 13.0 |
| DigiRL | 64.6 | 62.7 | 68.1 | 64.2 | 64.9 |
| **CoSo** | **72.9** | **71.3** | **77.0** | **70.5** | **72.9** |

### Gym Cards (Card Reasoning)

| Method | NL | EZP | P24 | BJ | Average |
|------|------|------|------|------|------|
| RL4VLM | 88.4 | 50.0 | 2.5 | 39.3 | 45.1 |
| **CoSo** | **100.0** | **50.0** | **5.8** | **41.5** | **49.3** |

### ALFWorld (Embodied AI)

CoSo achieves 26.5%, representing a 16.7% improvement over RL4VLM (22.7%).

### Ablation Study

- RL (w/o entropy): Baseline.
- RL + $\mathcal{H}$ (uniform entropy): Showed only marginal improvement.
- RL + $\mathcal{H}^{\mathcal{B}}$ (CoSo): **Significantly accelerated convergence speed and enhanced final performance.**

Key tokens account for less than 10% of total tokens, whereas over 80% of tokens have causal weights below 0.2.

## Highlights & Insights

1. **Precise localization of key tokens**: Counterfactual reasoning effectively shrinks the exploration space by approximately $32100^{90}$ times.
2. **General framework**: Naturally compatible with various RL objectives such as AWR and PPO.
3. **Theoretical completeness**: Guarantees for convergence and policy improvement are consistent with standard soft RL.
4. **Compelling real-world case**: In the Android environment, CoSo successfully samples the "Home" recovery action after erroneous operations, whereas standard RL repeatedly targets unclickable buttons.

## Limitations & Future Work

- The maximum utterance length in experiments did not exceed 300 tokens; thus, the effectiveness of ultra-long CoTs remains unverified.
- The BERT model acting as the SCM needs to be updated synchronously with the VLM.
- Counterfactual reasoning introduces additional computational overhead per epoch.

## Related Work & Insights

- VLM Agents (DigiRL, RL4VLM, AutoUI)
- RL Exploration (intrinsic motivation, entropy regularization)
- RLHF (PPO, DeepSeek-R1)

## Rating

⭐⭐⭐⭐⭐ — Well-defined problem formulation (addressing exploration explosion in VLM text-action spaces), with an elegant methodology (integrating counterfactual causality with weighted entropy), rigorous theoretical proofs, and consistent significant improvements across three task categories.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-Based VLM Agent](../../ICCV2025/multimodal_vlm/gtr_guided_thought_reinforcement_prevents_thought_collapse_in_rl-based_vlm_agent.md)
- [\[CVPR 2025\] DocVLM: Make Your VLM an Efficient Reader](../../CVPR2025/multimodal_vlm/docvlm_make_your_vlm_an_efficient_reader.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](../../NeurIPS2025/multimodal_vlm/advancing_compositional_awareness_in_clip_with_efficient_fin.md)

</div>

<!-- RELATED:END -->
