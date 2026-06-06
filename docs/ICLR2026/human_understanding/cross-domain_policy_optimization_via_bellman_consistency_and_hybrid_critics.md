---
title: >-
  [Paper Note] Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics
description: >-
  [ICLR 2026][Human Understanding][Cross-Domain Reinforcement Learning] This paper proposes the Q Avatar framework, which quantifies the transferability of source-domain models via cross-domain Bellman consistency and comb…
tags:
  - "ICLR 2026"
  - "Human Understanding"
  - "Cross-Domain Reinforcement Learning"
  - "Bellman Consistency"
  - "Hybrid Critic"
  - "Q-Function Transfer"
  - "Negative Transfer Prevention"
date: 2026-05-08
content_hash: f5d3956dd7dc82ea
---

# Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics

**Conference**: ICLR 2026
**arXiv**: [2603.12087](https://arxiv.org/abs/2603.12087)  
**Code**: [https://rl-bandits-lab.github.io/Cross-Domain-RL/](https://rl-bandits-lab.github.io/Cross-Domain-RL/)  
**Area**: Human Understanding
**Keywords**: Cross-Domain Reinforcement Learning, Bellman Consistency, Hybrid Critic, Q-Function Transfer, Negative Transfer Prevention

## TL;DR

This paper proposes the Q Avatar framework, which quantifies the transferability of source-domain models via cross-domain Bellman consistency and combines source- and target-domain Q-functions through an adaptive, hyperparameter-free weighting function. The framework enables reliable knowledge transfer in cross-domain RL with mismatched state-action spaces, guaranteeing no negative transfer regardless of source model quality or domain similarity.

## Background & Motivation

### Problem Background
Cross-domain reinforcement learning (CDRL) aims to leverage data collected in a source domain to improve learning efficiency in a target domain. In practical scenarios—such as transfer between robots with different morphologies—the source and target domains often have **different state and action spaces**, making direct transfer infeasible.

### Two Fundamental Challenges

**State-Action Space Mismatch**: Source and target domains may have state and action representations of different dimensionalities, requiring complex inter-domain mappings.

**Unknown Transferability**: The effectiveness of transferring a source-domain model is difficult to assess in advance, and CDRL is prone to **negative transfer**—where performance after transfer is worse than learning from scratch.

### Limitations of Prior Work
- Manually designed latent-space mapping methods (Ammar & Taylor, 2012) lack flexibility.
- Learned inter-domain mapping methods (Zhang et al., 2021; Gui et al., 2023) rely on dynamics alignment but offer no performance guarantees and ignore the transferability issue.
- All existing methods assume sufficient domain similarity and do not address negative transfer prevention.

## Method

### Overall Architecture

Q Avatar consists of three core components:
1. **Cross-Domain Bellman Consistency**: Quantifies the transferability of the source-domain Q-function.
2. **Hybrid Critic**: Adaptively combines source- and target-domain Q-functions via learned weights.
3. **Normalizing Flow Inter-Domain Mapping**: Learns cross-domain correspondences for states and actions.

### Key Designs

1. **Cross-Domain Bellman Consistency**

   A cross-domain Bellman error is defined to measure the applicability of the source-domain Q-function in the target domain:

   $\epsilon_{\text{cd}}(s,a;\phi,\psi,Q_{\text{src}},\pi) = |Q_{\text{src}}(\phi(s),\psi(a)) - r_{\text{tar}}(s,a) - \gamma \mathbb{E}_{s',a'}[Q_{\text{src}}(\phi(s'),\psi(a'))]|$

   where $\phi: \mathcal{S}_{\text{tar}} \to \mathcal{S}_{\text{src}}$ and $\psi: \mathcal{A}_{\text{tar}} \to \mathcal{A}_{\text{src}}$ are inter-domain mappings. A small error indicates that the source-domain Q-function satisfies the Bellman equation in the target domain after mapping, implying high transferability.

2. **Q Avatar Hybrid Critic**

   At each policy update step, a weighted combination is used:

   $Q^{(t)}_{\text{avatar}} = (1-\alpha(t)) Q^{(t)}_{\text{tar}} + \alpha(t) Q_{\text{src}}(\phi^{(t)}, \psi^{(t)})$

   The weight $\alpha(t)$ is **adaptive and hyperparameter-free**, determined by the ratio of the cross-domain Bellman error to the TD error:

   $\alpha(t) = \frac{1/\|\epsilon_{\text{cd}}\|_{d^{\pi^{(t)}}}}{1/\|\epsilon_{\text{td}}^{(t)}\|_{d^{\pi^{(t)}}} + 1/\|\epsilon_{\text{cd}}\|_{d^{\pi^{(t)}}}}$

   When the source-domain Q-function has a small Bellman error (high transferability), $\alpha$ is large and source-domain knowledge is weighted more heavily; otherwise, $\alpha$ is small and the agent relies primarily on target-domain learning. This guarantees **no negative transfer regardless of source model quality**.

3. **Normalizing Flow Inter-Domain Mapping**

   Normalizing flow models are employed to learn $\phi$ and $\psi$ by minimizing the cross-domain Bellman loss. The invertibility of flow models ensures training stability. This design demonstrates the compatibility of the Q Avatar framework with existing inter-domain mapping methods.

### Convergence Guarantee

**Theorem** (informal): The average suboptimality gap of Q Avatar is upper-bounded by:
$$O\left(\frac{\log|\mathcal{A}|}{\sqrt{T}(1-\gamma)}\right) + C \cdot \min\{\|\epsilon_{\text{td}}^{(t)}\|, \|\epsilon_{\text{cd}}\|\}$$

That is, Q Avatar automatically selects the smaller of the TD error and the cross-domain Bellman error, ensuring effective utilization under any source model quality.

## Key Experimental Results

### Main Results

Evaluation environments span locomotion control, robotic arm manipulation, and goal-conditioned navigation:

| Environment | Threshold | Q Avatar | SAC (from scratch) | Ratio |
|---|---|---|---|---|
| HalfCheetah | 6000 | 126K steps | 176K steps | 0.71 |
| Ant | 1600 | 206K steps | 346K steps | 0.59 |
| Door Opening | 90 | 48K steps | 98K steps | 0.49 |
| Table Wiping | 45 | 72K steps | 98K steps | 0.73 |
| Navigation | 20 | 218K steps | 490K steps | **0.44** |

In the best case, Q Avatar reaches the performance threshold using only 44% of the environment steps required by SAC.

### Comparison with Baselines
Q Avatar outperforms CMD, CAT-SAC, CAT-PPO, and FT (fine-tuning) on all tasks, with a notably superior IQM aggregate metric.

### Ablation Study

| Configuration | Key Metric | Observation |
|---|---|---|
| Strong positive transfer (symmetric Ant) | High $\alpha(t)$ | Source-domain knowledge effectively utilized |
| Strong negative transfer (reversed-goal Ant) | Low $\alpha(t)$ | Automatic negative transfer prevention |
| Low-quality source model (return 1000 vs. 7000) | Gradually decreasing $\alpha(t)$ | Adaptive reduction of source dependence |
| Unrelated domain transfer (Hopper → Table Wiping) | No negative transfer | Reliability guarantee |
| Non-stationary environment (noisy reward + actions) | Positive transfer preserved | Robustness |
| $N_\alpha$ sensitivity test | Slight sensitivity | 300/1000/3000 all viable |

### Key Findings
- $\alpha(t)$ accurately reflects transferability: high under positive transfer, low under negative transfer.
- Even when source and target domains are completely unrelated (Hopper vs. Table Wiping), Q Avatar does not incur negative transfer.
- Multi-source transfer is supported with automatic weight allocation.
- The framework is effective on image-based DMC tasks as well.

## Highlights & Insights

1. **Theory-Driven Framework Design**: The formal definition of transferability is grounded in Bellman consistency, with theory and algorithm design tightly coupled.
2. **Hyperparameter-Free Adaptive Weighting**: $\alpha(t)$ is determined entirely by the Bellman error ratio, requiring no manual tuning—a key factor for practical usability.
3. **Negative Transfer Guarantee**: Regardless of source model quality or domain discrepancy, performance is guaranteed to be no worse than pure target-domain learning—a property that existing CDRL methods generally lack.
4. **"Avatar" Metaphor**: The naming draws an analogy to the film in which humans remotely control engineered bodies to adapt to an alien environment, elegantly conveying the algorithmic intuition.

## Limitations & Future Work

- The tabular analysis assumes finite state-action spaces and an exploratory initial distribution, which diverges from practical continuous control settings.
- The quality of the normalizing flow mapping directly affects the accuracy of cross-domain Bellman error estimation.
- Experiments use SAC-pretrained source models exclusively; effectiveness with models trained by other algorithms (e.g., PPO) remains unvalidated.
- Scalability to high-dimensional complex tasks (e.g., dexterous hand manipulation) has yet to be verified.
- The current framework addresses only single-source-to-single-target and multi-source-to-single-target settings; multi-target scenarios are not considered.

## Related Work & Insights

- **CMD** (Gui et al., 2023): Learns inter-domain mappings via dynamics cycle-consistency, but offers no performance guarantees.
- **CAT** (You et al., 2022): Learns mappings via encoder-decoder architectures, but is limited to parameter-level transfer.
- **DARC** (Eysenbach et al., 2021): A reward augmentation approach, but assumes identical state-action spaces.
- **Task Vectors** (Wang et al., 2020): Uses dual Q-functions for Q-learning updates, but is similarly restricted to identical spaces.
- **Insight**: The notion of Bellman consistency as a transferability measure is generalizable to imitation learning, offline RL, and related settings.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of cross-domain Bellman consistency and adaptive hybrid critics is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across diverse environments, positive/negative transfer scenarios, multi-source transfer, image-based tasks, and sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ — Theory is clearly presented, experiments are thorough, and the "Avatar" naming is elegant.
- Value: ⭐⭐⭐⭐⭐ — Addresses the core negative transfer challenge in CDRL with theoretical guarantees and strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CarGait: Cross-Attention based Re-ranking for Gait Recognition](../../ICCV2025/human_understanding/cargait_cross_attention_based_re_ranking_for_gait_recognition.md)
- [\[CVPR 2026\] WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](../../CVPR2026/human_understanding/wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)
- [\[ACL 2026\] Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production](../../ACL2026/human_understanding/hybrid_autoregressive-diffusion_model_for_real-time_sign_language_production.md)
- [\[CVPR 2026\] SVC 2026: The Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge](../../CVPR2026/human_understanding/svc_2026_the_second_multimodal_deception_detection_challenge_and_the_first_domai.md)
- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](../../ICCV2025/human_understanding/lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)

</div>

<!-- RELATED:END -->
