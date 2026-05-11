---
title: >-
  [Paper Note] AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification
description: >-
  [ICLR 2026][Reinforcement Learning][Unsupervised skill learning] This paper proposes AMPED, a framework that applies gradient surgery (PCGrad) during skill pretraining to balance gradient conflicts between exploration (e…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Unsupervised skill learning"
  - "gradient surgery"
  - "exploration-diversity trade-off"
  - "skill selector"
  - "multi-objective RL"
date: 2026-05-08
content_hash: b149a650fbad3354
---

# AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification

**Conference**: ICLR 2026  
**arXiv**: [2506.05980](https://arxiv.org/abs/2506.05980)  
**Code**: [https://github.com/Cho-Geonwoo/amped](https://github.com/Cho-Geonwoo/amped)  
**Area**: Reinforcement Learning / Skill Discovery  
**Keywords**: Unsupervised skill learning, gradient surgery, exploration-diversity trade-off, skill selector, multi-objective RL  

## TL;DR
This paper proposes AMPED, a framework that applies gradient surgery (PCGrad) during skill pretraining to balance gradient conflicts between exploration (entropy + RND) and skill diversity (AnInfoNCE), and employs a SAC-based skill selector during fine-tuning to adaptively choose the optimal skill. AMPED outperforms SBRL baselines including DIAYN, CeSD, and CIC on Maze and URLB benchmarks.

## Background & Motivation

**Background**: Skill-based reinforcement learning (SBRL) enables rapid adaptation by pretraining skill-conditioned policies, but effective skill learning requires simultaneously maximizing exploration (covering more states) and skill diversity (distinguishability between skills).

**Limitations of Prior Work**: These two objectives are inherently conflicting — MI-driven diversity objectives lead to premature specialization (limiting exploration), while entropy-driven exploration sacrifices skill distinguishability. Existing methods (CeSD, ComSD) directly sum reward signals without systematically addressing gradient conflicts.

**Key Challenge**: Exploration gradients drive the agent to visit broader regions, while diversity gradients drive different skills apart — these update directions can be diametrically opposed, and naive summation leads to inefficient learning.

**Goal**: Systematically resolve gradient conflicts between exploration and diversity within a multi-objective RL framework, and fully exploit learned skill diversity during fine-tuning.

**Key Insight**: Treating exploration and diversity as two separate optimization objectives, applying PCGrad gradient surgery to remove conflicting gradient components, while introducing a learned skill selector to replace random selection.

**Core Idea**: Gradient surgery to resolve gradient conflicts during pretraining + a learned skill selector to exploit diversity during fine-tuning.

## Method

### Overall Architecture
AMPED operates in two phases: (1) **Pretraining** — the agent is conditioned on randomly sampled skills $z$ and trained jointly with exploration rewards (entropy + RND) and diversity rewards (AnInfoNCE), with PCGrad balancing the gradients of both objectives; (2) **Fine-tuning** — a SAC-based skill selector adaptively selects the optimal pretrained skill based on downstream task feedback, with the agent further optimized under extrinsic rewards.

### Key Designs

1. **Exploration Reward Design (Entropy + RND)**

    - Function: Combines particle-estimated state entropy and RND novelty as the exploration reward.
    - Mechanism: $r_{\text{exploration}} = \alpha \cdot r_{\text{entropy}} + \beta \cdot r_{\text{rnd}}$. Entropy is estimated via $k$-nearest neighbor distances (following CIC), and RND is based on prediction error $\|f_\theta(s) - f_{\text{target}}(s)\|^2$.
    - Design Motivation: Entropy estimation is reliable with small buffers but has high complexity ($O(n\log n)$) for large buffers; RND has $O(n)$ complexity but is unstable early in training — the two are complementary.

2. **Diversity Reward (AnInfoNCE)**

    - Function: Uses anisotropic InfoNCE contrastive learning to encourage different skills to produce distinguishable trajectories.
    - Mechanism: Standard InfoNCE encourages alignment between skill embeddings and corresponding trajectory embeddings; AnInfoNCE further accounts for the directionality of embeddings, enhancing repulsion between skills.
    - Design Motivation: MI-based skill diversity objectives are standard in SBRL; AnInfoNCE provides stronger discriminative capacity.

3. **Gradient Surgery (PCGrad-based)**

    - Function: When exploration and diversity gradients conflict, removes the interfering component of one gradient along the direction of the other.
    - Mechanism: Compute $g_{\text{explore}}$ and $g_{\text{diverse}}$; if $g_1 \cdot g_2 < 0$ (conflict), project $g_1$ onto the orthogonal complement of $g_2$: $g_1' = g_1 - \frac{g_1 \cdot g_2}{\|g_2\|^2} g_2$.
    - Design Motivation: Gradient conflict is the core bottleneck in jointly optimizing exploration and diversity — naive summation may cause the two objectives to cancel each other out. PCGrad ensures the modified gradient does not interfere with the other objective.

4. **SAC-based Skill Selector**

    - Function: Learns to select the pretrained skill best suited for the downstream task during fine-tuning.
    - Mechanism: Skill selection is modeled as a reinforcement learning problem — the selector observes the current state, outputs skill $z$, and receives extrinsic reward from the downstream task.
    - Design Motivation: Existing SBRL methods (DIAYN, CeSD) select skills randomly during fine-tuning, wasting the diversity obtained during pretraining. A learned selector can fully exploit this diversity — Theorem 1 formally shows that greater diversity leads to fewer fine-tuning samples.

### Theoretical Guarantee
**Theorem 1**: Given $\delta$ (minimum TV distance between skills) and $\varepsilon$ (TV distance between the best skill and the target policy), a greedy skill selector requires $n = O\!\left(\frac{1}{\Delta^2}(S\log 2 + \log H)\right)$ samples to identify the optimal skill with high probability, where $\Delta = \delta - 2\varepsilon$. Greater skill diversity (larger $\delta$) reduces sample requirements.

## Key Experimental Results

### Maze Environment
AMPED simultaneously achieves high state coverage and skill separation — CeSD achieves good coverage but with mixed skills, while BeCL produces strong skill separation but with notable gaps in coverage.

### URLB Benchmark (Statistically Significant Improvements)
AMPED achieves the highest return on multiple URLB tasks, outperforming baselines including DIAYN, CIC, CeSD, BeCL, ComSD, RND, and APT.

### Ablation Study
- Removing gradient surgery → significant performance drop.
- Removing RND → insufficient exploration in complex environments.
- Removing AnInfoNCE → poor skill discriminability.
- Removing the skill selector (replaced with random selection) → reduced fine-tuning efficiency.
- Each component contributes positively.

### Key Findings
- Gradient surgery is the most critical component, directly addressing the core conflict between exploration and diversity.
- The learned skill selector vs. random selection shows a substantial difference during fine-tuning — diversity only has value when it is actively exploited.
- The theoretical prediction of Theorem 1 is consistent with experimental results: greater skill diversity leads to faster fine-tuning convergence.

## Highlights & Insights
- **Formalizing the exploration-diversity conflict as a multi-objective gradient conflict** is the central insight — prior work recognized that jointly optimizing the two objectives is difficult, but did not use gradient analysis to identify the underlying cause.
- **Theorem 1** provides a formal proof of the "diversity → sample efficiency" relationship, addressing the fundamental question of why skill diversity should be pursued.
- The overall pipeline is logically coherent: balance the two objectives during pretraining → exploit the learned diversity during fine-tuning, forming a closed loop.

## Limitations & Future Work
- The pairwise projection in PCGrad has limited scalability as the number of objectives grows.
- Evaluation is restricted to continuous control tasks (MuJoCo/Maze); discrete action spaces and high-dimensional observation spaces remain untested.
- The skill selector itself must be trained from scratch, increasing fine-tuning cost.
- $\alpha$ and $\beta$ (exploration reward weights) still require manual tuning.
- No comparison with hierarchical RL methods is conducted.

## Related Work & Insights
- **vs. CeSD**: CeSD directly sums exploration and diversity rewards without handling gradient conflicts; AMPED systematically resolves this via gradient surgery.
- **vs. DIAYN**: DIAYN is purely MI-driven with insufficient exploration; AMPED explicitly incorporates an exploration objective.
- **vs. CIC**: CIC uses contrastive learning for exploration but does not address skill diversity; AMPED addresses both.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of gradient surgery applied to SBRL and a learned skill selector is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Maze + URLB + comprehensive ablations + theoretical validation; fairly complete.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear; Figures 1–3 are intuitive and compelling.
- Value: ⭐⭐⭐⭐ Provides a systematic solution to the exploration-diversity balance in SBRL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] unsupervised learning of efficient exploration pre-training adaptive policies vi](unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[ICLR 2026\] RLP: Reinforcement as a Pretraining Objective](rlp_reinforcement_as_a_pretraining_objective.md)

</div>

<!-- RELATED:END -->
