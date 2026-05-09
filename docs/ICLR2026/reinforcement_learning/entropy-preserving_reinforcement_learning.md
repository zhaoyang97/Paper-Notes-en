---
title: >-
  [Paper Note] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)
description: >-
  [ICLR 2026][Reinforcement Learning][Entropy preservation] This paper identifies the theoretical root cause of systematic policy entropy collapse in policy gradient RL algorithms for LLM post-training — namely, the positive correlation between advantage functions and log-probabilities — and proposes two complementary solutions: REPO (decorrelating the advantage function) and ADAPO (adaptive asymmetric clipping), achieving state-of-the-art performance on interactive tool-use tasks.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Entropy preservation
  - policy gradient
  - LLM post-training
  - GRPO
  - exploration
date: 2026-05-08
content_hash: eb7a2da2cd19b847
---

# Entropy-Preserving Reinforcement Learning (REPO / ADAPO)

**Conference**: ICLR 2026
**arXiv**: [2603.11682](https://arxiv.org/abs/2603.11682)
**Code**: None
**Area**: Reinforcement Learning / LLM Training
**Keywords**: Entropy preservation, policy gradient, LLM post-training, GRPO, exploration

## TL;DR
This paper identifies the theoretical root cause of systematic policy entropy collapse in policy gradient RL algorithms for LLM post-training — namely, the positive correlation between advantage functions and log-probabilities — and proposes two complementary solutions: REPO (decorrelating the advantage function) and ADAPO (adaptive asymmetric clipping), achieving state-of-the-art performance on interactive tool-use tasks.

## Background & Motivation

**State of the Field**: Policy gradient algorithms such as GRPO, PPO, and RLOO are widely adopted for RL post-training to enhance LLM reasoning capabilities. DAPO introduces asymmetric clipping to implicitly preserve entropy, while GSPO employs sequence-level clipping.

**Limitations of Prior Work**: Policy gradient updates systematically collapse policy entropy — models concentrate probability mass on already high-probability correct solutions while neglecting other equally correct but lower-probability alternatives. Consequences include improved pass@1 but degraded pass@k, loss of exploration capacity, and failure to fine-tune on new tasks (sequential learning failure).

**Root Cause**: When a model is already "calibrated" with respect to reward (high-probability actions receive high rewards), policy gradient updates naturally sharpen the distribution and reduce entropy — an intrinsic property of policy gradients rather than an incidental bug.

**Starting Point**: Theoretically characterizing the per-step entropy change reveals that it is proportional to the correlation between the advantage function and log-probabilities — breaking this correlation suffices to preserve entropy.

**Core Idea**: Modifying the advantage function by subtracting a term proportional to log-probability eliminates the correlation responsible for entropy collapse while preserving the policy improvement direction.

## Method

### Overall Architecture
Analyze entropy dynamics of policy gradient updates → identify the drivers of entropy change → design two control mechanisms (REPO modifies the advantage function; ADAPO adapts clipping bounds) → adaptive controllers maintain a target entropy level.

### Key Designs

1. **Theoretical Analysis (Theorem 1)**:

    - Function: Precisely characterizes the entropy change after each policy gradient update step.
    - Mechanism: $\Delta\mathcal{H} \propto -\mathbb{E}_{a \sim \pi}[A(\mathbf{s},a) \cdot L(\mathbf{s},a) \cdot \pi(a|\mathbf{s})]$, where $L$ denotes mean-centered log-probabilities. When the advantage $A$ is positively correlated with $L$ (high-probability actions receive high rewards), entropy decreases.
    - Design Motivation: Provides a precise intervention target — breaking the correlation between $A$ and $L$.

2. **REPO-R (Rescale Variant)**:

    - Function: Modifies the advantage function to amplify rewards for high-probability correct actions and reduce penalties for low-probability incorrect ones.
    - Mechanism: $A_{\text{REPO}}(s,a) = A(s,a) - \beta \cdot L(s,a)$. The practical variant REPO-R sets $\beta = \zeta \cdot |A|$: for positive advantages, $A^+ = A(1 - \zeta\log\pi)$ (rare correct actions are amplified); for negative advantages, $A^- = A(1 + \zeta\log\pi)$ (rare incorrect actions are penalized less).
    - Adaptive Controller: If current entropy < initial entropy, $\zeta$ is doubled; if > initial entropy, $\zeta$ is halved. This supports bidirectional regulation.

3. **ADAPO (Adaptive Asymmetric Clipping)**:

    - Function: Dynamically adjusts the asymmetric clipping parameter $\epsilon_{\text{high}}$ in DAPO.
    - Mechanism: Theorem 2 proves that PPO's clipping constrains entropy change within the range $[(1-\epsilon_{\text{low}})\mathcal{H}, (1+\epsilon_{\text{high}})\mathcal{H}]$. If entropy is too low, $\epsilon_{\text{high}}$ is increased (allowing more entropy growth); if too high, it is decreased.
    - Design Motivation: DAPO's fixed asymmetric clipping leads to uncontrolled entropy growth (+298%) in certain settings, necessitating bidirectional regulation.

4. **BF16 Precision Issue (Theorem 3)**:

    - Function: Identifies and corrects a hidden bias in BF16 computation.
    - Mechanism: Computing the importance ratio $r = \pi_\theta / \pi_{\text{old}}$ in BF16 introduces an upward multiplicative bias, equivalent to an asymmetric clipping effect in the entropy-reducing direction — contrary to DAPO's design intent. Fix: compute log-probabilities in full precision.

### Loss & Training
REPO can be layered on top of any policy gradient algorithm (GRPO, RLOO, DAPO) with zero additional memory overhead and is compatible with Cut Cross-Entropy.

## Key Experimental Results

### Main Results
AppWorld (interactive tool use) — Qwen-3-32B:

| Algorithm | Test Normal↑ | Test Challenge↑ | Entropy Change |
|-----------|-------------|----------------|----------------|
| GRPO | 0.67 | 0.46 | −57% |
| DAPO | 0.73 | 0.52 | +298% (uncontrolled) |
| RLOO (FP16 fix) | **0.79** | **0.71** | −36% |
| ADAPO | 0.78 | 0.58 | +102% |
| REPO-R | 0.73 | 0.54 | +7% |

AIME 2024/2025 (mathematical reasoning) — Qwen-3-8B: differences are smaller (0.43–0.47), as the baseline model is already highly optimized.

### Ablation Study

| Finding | Description |
|---------|-------------|
| BF16→FP16 fix | Qualitatively alters DAPO behavior (entropy collapse → entropy growth) |
| Cumulative entropy positively correlates with final performance | "The journey matters, not just the destination" — models that maintain higher entropy during training achieve better final performance |
| Sequential learning ability | Models trained with GRPO (entropy collapse) catastrophically fail on new tasks; REPO/DAPO models successfully transfer |
| Explicit entropy bonus vs. REPO | Explicit $\beta\mathcal{H}$ requires logit materialization (incompatible with CCE); REPO is more efficient |

### Key Findings
- Strictly on-policy RLOO (after precision fix) proves to be the strongest baseline, raising the question of whether entropy collapse is primarily introduced by off-policy training.
- DAPO exhibits uncontrolled entropy growth (+298%) on large models; ADAPO resolves this via bidirectional regulation.
- Entropy preservation yields large gains on exploration-intensive tasks (AppWorld) but limited gains on already well-optimized tasks (AIME).

## Highlights & Insights
- **Theory-driven practical improvement**: Theorem 1 precisely characterizes the mechanism of entropy collapse (A–L correlation), and REPO directly intervenes on this correlation — a paradigmatic example of theory guiding practice.
- **Discovery of the BF16 bias** is significant: a precision issue affecting fewer than 0.1% of tokens can qualitatively alter training dynamics, serving as a cautionary finding for all RL training conducted in BF16.
- **Sequential learning evaluation** offers a novel assessment perspective: beyond final single-task performance, it examines whether models retain the capacity to continue learning new tasks.

## Limitations & Future Work
- Limited improvement on AIME indicates diminishing returns in already well-optimized domains.
- The adaptive controller relies on heuristic doubling/halving without convergence guarantees.
- All experiments employ LoRA fine-tuning; whether results generalize to full-parameter fine-tuning remains unknown.
- The first-order Taylor approximation and the assumed orthogonality between score functions may not hold in deep Transformers.

## Related Work & Insights
- **vs. GRPO**: GRPO suffers the most severe entropy collapse (−64% at 8B, −57% at 32B); REPO can be directly applied on top of GRPO to address this.
- **vs. DAPO**: DAPO's fixed asymmetric clipping lacks flexibility and may become unstable; ADAPO's adaptive regulation is more robust.
- **vs. explicit entropy bonus**: REPO achieves an equivalent effect via REINFORCE estimation with zero additional memory overhead.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Complete chain from theory to practice: root cause analysis → intervention design → precision bug fix → experimental validation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Strong results on AppWorld; limited improvement on AIME; non-LoRA experiments absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations closely aligned with experimental findings.
- Value: ⭐⭐⭐⭐⭐ — Provides both theoretical foundations and practical tools for entropy management in LLM RL training.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[NeurIPS 2025\] Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/convergence_theorems_for_entropy-regularized_and_distributional_reinforcement_le.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)
- [\[ICLR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)

<!-- RELATED:END -->
