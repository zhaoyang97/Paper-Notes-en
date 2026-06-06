---
title: >-
  [Paper Note] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation
description: >-
  [ICLR 2026][Reinforcement Learning][behavior-targeted attack] This paper studies a novel threat in RL—behavior-targeted attacks (where an adversary manipulates observations to steer the victim toward executing a specific…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "behavior-targeted attack"
  - "adversarial robustness"
  - "imitation learning attack"
  - "temporal discounting defense"
  - "policy smoothing"
date: 2026-05-08
content_hash: 07d263db999dc2a3
---

# Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation

**Conference**: ICLR 2026
**arXiv**: [2406.03862](https://arxiv.org/abs/2406.03862)  
**Code**: None  
**Area**: AI Safety / Reinforcement Learning
**Keywords**: behavior-targeted attack, adversarial robustness, imitation learning attack, temporal discounting defense, policy smoothing

## TL;DR
This paper studies a novel threat in RL—behavior-targeted attacks (where an adversary manipulates observations to steer the victim toward executing a specific target policy)—and proposes BIA, a black-box attack method, along with TDRT, a temporally discounted robust training defense. TDRT achieves robustness against such attacks while outperforming the existing defense SA-PPO on original task performance by 28.2%.

## Background & Motivation

**Background**: Existing adversarial attack research in RL primarily focuses on reward-minimization attacks—making the victim perform as poorly as possible. Defense methods such as ATLA and SA-PPO are also designed with this threat model in mind.

**Limitations of Prior Work**: A more dangerous attack paradigm exists—behavior-targeted attacks—where the adversary does not cause the victim to fail outright, but instead steers it toward executing specific behaviors (e.g., rerouting an autonomous vehicle to a particular store). Existing behavior-targeted attacks (PA-AD, Targeted PGD) require white-box access to the victim policy, which is impractical in realistic settings. Moreover, no defense method has been specifically designed against this attack class.

**Key Challenge**: How can behavior-targeted attacks be carried out without accessing the internals of the victim policy? How can a defense be designed that resists behavior-targeted attacks without excessively sacrificing original task performance?

**Key Insight**: Behavior-targeted attacks are reformulated as a cumulative reward maximization problem within an MDP (Theorem 5.1), such that the victim policy is naturally embedded in the environment dynamics, eliminating the need for white-box access.

**Core Idea**: On the attack side, MDP reformulation converts white-box requirements into a black-box problem. On the defense side, temporally discounted robust training prioritizes the protection of early-step decisions.

## Method

### Overall Architecture
The framework consists of two components: the attack (BIA) and the defense (TDRT). The attacker constructs an auxiliary MDP and trains an observation-manipulation policy using standard imitation learning algorithms (GAIL/ILfO). The defender incorporates temporally discounted worst-case KL divergence regularization during training.

### Key Designs

1. **Behavior Imitation Attack (BIA)**:

    - **Function**: Executes behavior-targeted attacks without white-box access to the victim policy.
    - **Mechanism**: The adversary learns a policy $\nu: s \mapsto \hat{s}$ (mapping true states to perturbed states) such that the composed policy $\pi \circ \nu(a|s) = \sum_{\hat{s}} \nu(\hat{s}|s)\pi(a|\hat{s})$ matches the target policy $\pi_{\text{tgt}}$. Theorem 5.1 establishes the key equivalence: $\arg\min_\nu \mathcal{D}(\pi \circ \nu, \pi_{\text{tgt}})$ can be reformulated as cumulative reward maximization in a constructed MDP $\hat{M}$, where the victim policy is embedded in $\hat{M}$'s transition dynamics—thus requiring no white-box access.
    - **Design Motivation**: GAIL (black-box, requires demonstrations of target behavior) or ILfO (no-box, requires only observation trajectories of target states) can be directly applied to implement the attack. Effective attacks can be achieved with as few as 4–20 target behavior demonstrations.
    - **Distinction from White-Box Attacks**: Methods such as PA-AD require computing gradients through the victim policy, whereas BIA performs standard RL/IL within the constructed MDP.

2. **Temporally Discounted Robust Training (TDRT)**:

    - **Function**: Trains policies that are robust against behavior-targeted attacks while preserving original task performance.
    - **Mechanism**: Theorem 6.1 establishes that the adversary's gain is upper-bounded by $\sum_{t=0}^{\infty} \frac{\gamma^t}{1-\gamma} \mathbb{E}_{s \sim d_\pi^t}[D_{\text{KL}}(\pi(\cdot|s) \| \pi \circ \nu(\cdot|s))]$. Two key insights follow: (a) reducing the policy's sensitivity to state perturbations improves robustness; (b) **earlier timesteps matter more than later ones** (due to the $\gamma^t$ weighting). The TDRT objective is: $J_{\text{def}}(\pi) = -J_{\text{RL}}(\pi) + \lambda \max_\nu \sum_{s_t \in B} \gamma^t D_{\text{KL}}(\pi(\cdot|s_t) \| \pi \circ \nu(\cdot|s_t))$.
    - **Design Motivation**: SA-PPO (uniform policy smoothing) achieves comparable robustness but at a severe task performance cost (−28.2%). TDRT focuses the temporal discount on early decisions, preserving more task capability at equivalent robustness levels.
    - **Distinction from Adversarial Training**: ATLA/PA-ATLA simulate reward-minimization attacks during training and are ineffective against behavior-targeted attacks due to the mismatched threat model.

### Loss & Training
- **Attack training**: Standard GAIL/ILfO within the constructed MDP $\hat{M}$
- **Defense training**: PPO objective combined with temporally discounted worst-case KL divergence regularization

## Key Experimental Results

### Main Results
Evaluated on 10 task pairs in Meta-World; attack reward ↑ indicates a more successful attack.

| Attack Method | Requirement | Typical Attack Reward | Notes |
|--------------|-------------|----------------------|-------|
| Random | None | 947 | Weak perturbation |
| PA-AD | White-box | 4255 | Requires policy gradients |
| BIA-ILfD | Black-box (20 demos) | 3962 | Near white-box performance |
| BIA-ILfO | No-box | ~3900 | Approaches ILfD in deterministic environments |

Defense results (best attack reward ↓ = more robust):

| Defense Method | Typical Attack Reward ↓ | Original Task Performance |
|---------------|------------------------|--------------------------|
| No defense | 1556 | Baseline |
| ATLA-PPO | 1158 | Moderate |
| SA-PPO | 403 | Poor (−28.2%) |
| **TDRT-PPO** | **378** | **Good (near baseline)** |

### Ablation Study

| Configuration | Key Finding |
|--------------|-------------|
| Temporal discounting vs. uniform smoothing | TDRT achieves comparable robustness with 28.2% higher task performance |
| Number of demonstrations | As few as 4 demonstrations suffice for effective attacks |
| Adversarial training methods (ATLA) | Ineffective against behavior-targeted attacks due to mismatched threat model |
| Attack difficulty | Attacks become harder when victim and target behavior distributions diverge significantly (e.g., window-open, door-lock) |

### Key Findings
- BIA achieves near-white-box attack performance using only 20 demonstrations, demonstrating that behavior-targeted attacks constitute a practical and dangerous real-world threat.
- Adversarial training (ATLA) is nearly ineffective against behavior-targeted attacks, as it simulates reward-minimization attacks rather than behavior manipulation during training.
- The temporal discounting in TDRT is the critical differentiating factor: SA-PPO's uniform smoothing achieves similar robustness at the cost of 28.2% task performance degradation, whereas TDRT preserves task capability by focusing on early timesteps.
- Attack effectiveness degrades when the behavioral distributions of the victim and target diverge substantially.

## Highlights & Insights
- **The MDP reformulation (Theorem 5.1)** is particularly elegant: the key to converting white-box requirements into a black-box setting is embedding the victim policy into the environment dynamics—the adversary no longer needs to differentiate through the policy and instead performs standard RL within the newly constructed MDP. This idea is transferable to other security scenarios requiring white-box-to-black-box conversion.
- **The insight that "early decisions matter more than later ones"** has broad applicability: in sequential decision-making, early errors propagate and amplify over time. This motivates prioritizing the protection of decision quality at early states in any RL robust training framework.
- The attack and defense are studied as a unified framework: the theoretical analysis of the attack (Theorem 5.1) directly informs the defense design (Theorem 6.1), forming a complete and coherent closed loop.

## Limitations & Future Work
- Attack effectiveness is limited in high-dimensional observation spaces (e.g., image-based inputs).
- TDRT provides empirical robustness rather than certified robustness (no certified guarantee).
- Attacks become difficult when the behavioral distributions of the victim and target diverge substantially—suggesting that certain scenarios may not require defense.
- The defense relies on the assumption that the adversary operates under a KL divergence constraint.

## Related Work & Insights
- **vs. PA-AD (Zhang et al.)**: PA-AD requires white-box access to the victim policy; BIA achieves black-box/no-box attacks via MDP reformulation, with only ~7% degradation in attack effectiveness.
- **vs. SA-PPO**: SA-PPO applies uniform smoothing across all timesteps; TDRT uses temporal discounting to focus on early steps—achieving comparable robustness with 28.2% higher task performance.
- **vs. ATLA / adversarial training**: Adversarial training assumes a reward-minimizing attacker and is ineffective against behavior manipulation attacks—exposing the critical problem of **defense–threat model mismatch**.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the MDP reformulation for behavior-targeted attacks and the temporally discounted defense are entirely novel concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10 task pairs in Meta-World with diverse attack/defense comparisons, but lacks experiments on high-dimensional observations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from attack → theory → defense is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Reveals a neglected yet dangerous attack paradigm in RL and provides an effective defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](../../NeurIPS2025/reinforcement_learning/robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)
- [\[ICLR 2026\] Understanding and Improving Hyperbolic Deep Reinforcement Learning](understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)
- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)

</div>

<!-- RELATED:END -->
