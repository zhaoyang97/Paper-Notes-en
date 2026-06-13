---
title: >-
  [Paper Note] Safe In-Context Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Safe RL] This paper introduces "safety constraints" into in-context reinforcement learning (ICRL) for the first time, proposing SCARED. During the pre-training phase…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Safe RL"
  - "In-context RL"
  - "CMDP"
  - "exact penalty"
  - "cost-to-go"
date: 2026-05-08
content_hash: a229618942b2c9c9
---

# Safe In-Context Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2509.25582](https://arxiv.org/abs/2509.25582)  
**Code**: Not yet released  
**Area**: Reinforcement Learning / Safe RL / In-context Learning  
**Keywords**: Safe RL, In-context RL, CMDP, exact penalty, cost-to-go

## TL;DR
This paper introduces "safety constraints" into in-context reinforcement learning (ICRL) for the first time, proposing SCARED. During the pre-training phase, it uses an **exact-penalty Lagrangian with a single multiplier and a positive hinge function** to enable the Transformer policy to perform CMDP adaptation during test time **without updating any parameters**, relying solely on cost-to-go context. Results on OOD grid, MuJoCo, and Velocity benchmarks show monotonically increasing rewards and decreasing costs, allowing smooth transitions between conservative and aggressive behaviors based on a user-provided budget $\delta$.

## Background & Motivation
**Background**: In-context RL (ICRL) is a new paradigm adapted from GPT-style sequence models in recent years. It treats interaction trajectories from multiple tasks as a long context for Transformer/SSM pre-training. During testing on new tasks, it performs **only forward passes without backpropagation**, relying on the growing history to let the network "implicitly run an RL algorithm in the forward pass." Representative works like AD, DPT, AMAGO, and Headless-AD have demonstrated impressive zero-shot adaptation in DarkRoom and MuJoCo.

**Limitations of Prior Work**: All existing ICRL works focus on reward maximization and **completely ignore cost/safety**. However, the truly attractive scenarios for ICRL (embodied AI, robotics, autonomous driving) require satisfying hard safety constraints—not just post-deployment safety, but **safety throughout the entire test-time process of exploration and learning**, which is more challenging than standard RL.

**Key Challenge**: Existing approaches lack ready-made solutions: (1) Standard ICRL has no cost channel; (2) Safe meta-RL (MAML+penalty, SafeMeta) relies on gradient updates during testing, losing the "pure forward adaptation" selling point of ICRL, and can only use history for parameter fine-tuning without capturing fine-grained cost signals per episode; (3) Directly applying standard dual methods for CMDP, assigning a Lagrange multiplier $\lambda_k$ to each episode, requires fixing the number of test episodes $K$ during pre-training, and $\lambda_k$ updates in later stages are much less frequent than early ones, leading to unstable optimization.

**Goal**: (i) Formalize the safe ICRL problem: satisfying cost constraints for each test episode within the CMDP framework without parameter updates; (ii) Design a **stable, single-multiplier** dual algorithm; (iii) Construct true OOD safety benchmarks (not chessboard-style interpolation) to prove algorithmic extrapolation.

**Key Insight**: (1) Feed the cost-to-go $G_{c,t}(\tau)\!=\!\sum_{i=t+1}^{T} C_i$ as an **explicit context input** to the policy, allowing it to automatically adjust its aggressiveness/conservatism by conditioning on this scalar during test time—an extension of RTG/CTG decision-transformer tricks to safe RL. (2) Collapse "per-episode penalties" into a "penalty on the worst episode," using a hinge-positive surrogate $L_\Sigma$ and **exact penalty** theory to guarantee that when $\lambda \ge \|\lambda^\star\|_\infty$, the fixed point matches the original problem's optimal solution.

**Core Idea**: Incorporate CMDP safety constraints into ICRL pre-training using **exact-penalty dual + single-multiplier hinge penalty + CTG-conditioned Transformer**, enabling the agent to slide between different safety budgets by simply changing the input CTG value (without modifying weights).

## Method

### Overall Architecture
During the pre-training phase, SCARED follows the reinforcement pretraining route (optimizing $\pi_\theta$ at each step with standard online RL loss, similar to AMAGO/DDPG-style ICRL), adding three components:
1. **CMDP-based Environment Sampling**: Each source MDP includes a cost function $c$. In addition to states and history $H_t^k$, the policy input includes a scalar cost-to-go $G_{c,t}(\tau_k)$, initialized to the budget $\delta$ at the start of each episode and decreasing as actual cost is incurred.
2. **Actor-Critic Dual Critics**: A reward Q-function $Q_{\theta_v}$ and a cost Q-function $Q_{\theta_c}^c$ are trained via TD-targets. The actor maximizes $Q_{\theta_v}$ and receives a penalty from the cost Q-function when an episode exceeds the budget.
3. **Single-multiplier Exact-penalty Dual Iteration**: Use $L_\Sigma(\pi, \lambda) = \mathbb{E}_\pi[\sum_k G(\tau_k)] - \lambda \sum_k [g_k(\pi)]_+$ (where $g_k(\pi) = \mathbb{E}_\pi[G_c(\tau_k)] - \delta$) as the surrogate, iterating $\pi_{t+1} \in \arg\max L_\Sigma(\pi, \lambda_t)$ and $\lambda_{t+1} = [\lambda_t + \eta \max_k g_k(\pi_{t+1})]_+$.

During testing, the model is placed in a new CMDP (OOD goal/obstacle distributions). Given a user-specified budget $\delta$, the policy initializes CTG to $\delta$ at the start of each episode and runs **without any gradient updates**. The Transformer performs "in-context safe RL" through the sequence of (state, action, reward, cost, CTG) within its context.

### Key Designs

1. **Cost-to-go as a Controllable Context Scalar**:
    - **Function**: Explicitly adds the remaining budget $G_{c,t}(\tau_k)$ to the policy input, turning the "user budget $\delta$" into a knob that can be adjusted during testing, avoiding the need to guess the feasible region as seen with RTG/CTG pairs.
    - **Mechanism**: The policy is $\pi_\theta(\cdot|S_t^k, H_t^k, G_{c,t}(\tau_k))$. Initial $G_{c,0} = \delta$. During pre-training, $\delta$ is uniformly sampled (SafeDarkRoom $[1, 10]$, SafeDarkMujoco $[10, 50]$, SafeVelocity $[0, 5]$), teaching the network to produce actions with varying levels of conservatism based on CTG.
    - **Design Motivation**: Unlike SafeAD, which requires both RTG and CTG to control tradeoffs (with risks of infeasibility if RTG is misaligned), the single CTG knob is naturally monotonic—high CTG encourages risky high rewards, while low CTG enforces caution. This simplifies "user-policy negotiation" into a single scalar.

2. **Single-multiplier + Hinge Surrogate Lagrangian**:
    - **Function**: Uses one Lagrange multiplier $\lambda$ to manage all episode cost constraints simultaneously without "penalizing" episodes that already satisfy constraints.
    - **Mechanism**: Standard CMDP duals $\max_\pi \min_\lambda L(\pi, \lambda) = \mathbb{E}_\pi[\sum_k G(\tau_k)] - \sum_k \lambda_k(\mathbb{E}_\pi[G_c(\tau_k)] - \delta)$ require $K$ multipliers and suffer from uneven updates. Ours collapses this to $L_\Sigma(\pi, \lambda) = \mathbb{E}_\pi[\sum_k G(\tau_k)] - \lambda \sum_k [g_k(\pi)]_+$, only penalizing **budget-exceeding** episodes ($[x]_+ = \max(x, 0)$), with $\lambda$ tracking the "worst episode" via $\lambda \leftarrow [\lambda + \eta \max_k g_k(\pi)]_+$. Theorem 1 proves that when $\lambda \ge \|\lambda^\star\|_\infty$, the fixed point is **identical** to the original CMDP optimal feasible policy set, hence the name SCARED (Safe Contextual Adaptive Reinforcement via Exact-penalty Dual).
    - **Design Motivation**: (1) Single multiplier avoids fixing $K$ during pre-training; (2) Hinge prevents over-penalization; (3) Exact-penalty theory ensures neutrality compared to quadratic/squared penalties, strictly aligning the dual optimum with the original problem.

3. **Truly Extrapolating OOD Safety Benchmarks (Center $\to$ Edge Drift)**:
    - **Function**: Constructs an **analytically OOD** evaluation protocol (KL $\to \infty$, TV $\to 1$), upgrading SafeDarkRoom/SafeDarkMujoco from "chessboard interpolation" to "training on center, testing on edges."
    - **Mechanism**: In training, obstacles/goals at $(i,j)$ are sampled with $P_{\text{train}}((i,j)) \propto e^{-\alpha d((i,j),c)}$ ($c$ is map center). In testing, $P_{\text{test}}((i,j)) \propto e^{+\alpha d((i,j),c)}$, shifting toward edges. Proposition 1 shows that as $\alpha \to \infty$, $d_{TV} \to 1$ and $D_{KL} \to \infty$, meaning supports barely overlap. Combined with SafeVelocity unseen intervals, this covers both structural OOD and unseen ID generalization.
    - **Design Motivation**: Previous OOD in DarkRoom variants merely swapped grid targets, which is essentially interpolation. This distance-based exponential weighting forces support separation, ensuring that "consistent cost reduction during testing" truly indicates in-context safe adaptation.

### Loss & Training
- The foundation RL loss follows Grigsby et al. (2024a) DDPG-style ICRL; reward and cost critics are trained with TD-targets.
- The top-level actor gradient corresponds to $L_\Sigma$: $\nabla_\theta \mathbb{E}_\pi[\sum_k G(\tau_k)] - \lambda \nabla_\theta \sum_k [g_k(\pi)]_+$, with the cost critic estimating $g_k$.
- $\lambda$ is updated slowly via $\lambda_{t+1} = [\lambda_t + \eta \max_k g_k(\pi_{t+1})]_+$, with a small $\eta$.
- During pre-training, $\delta$ is sampled uniformly, and CTG is used as an additional input; context is encoded by a long Transformer across full episode histories.

## Key Experimental Results

### Main Results: Adaptation across 5 safe environments

| Environment | Type | SCARED return ↑ | SCARED cost ↓ | Safe AD | SafeMeta | MAML+penalty |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SafeDarkRoom (9×9) | OOD Grid | Monotonic to ~0.6 | Monotonic to ~1 | Slow rise, weak cost reduction | Cost doesn't drop | Failed |
| SafetyPoint | OOD Control | Up to ~0.6 | Down to ~2 | Return rises, cost **does not** | No drop | Failed |
| SafetyCar | OOD Control | Up to ~0.8–1.0 | Decreases | Adaptation fails | No drop | Failed |
| SafetyHalfCheetah | Unseen ID | ~200 | Constantly low | Return drops, cost rises | High reward, cost stays high | Failed |
| SafetyAnt | Unseen ID | ~200 | Constantly low | Return drops, cost rises | High reward, cost stays high | Failed |

> The x-axis is episode index $k$ ($k=0..50$). No parameter updates are performed (except for Meta-RL). SCARED is the only method where reward monotonically $\uparrow$ while cost monotonically $\downarrow$.

### Ablation Study / Key Findings

| Configuration | Behavior | Description |
| :--- | :--- | :--- |
| Full SCARED | reward↑ cost↓, cost ≤ $\delta$ | Single-multiplier + exact penalty stable convergence |
| Multi-multiplier | Occasional oscillations | Proves necessity of single multiplier (Fig 5c) |
| Safe AD (noise variant) | Fails OOD adaptation | Lacks behavioral diversity (Appendix D) |
| Change CTG value | $\delta \uparrow \to$ aggressive, $\delta \downarrow \to$ conservative | Proves pure forward trade-off capability |

### Key Findings
- **Safe AD works in simple grids but fails in complex control (SafeDarkMujoco)**: Cost doesn't drop even when rewards rise, suggesting that conditioning on cost has a ceiling compared to embedding it in the RL objective.
- **SafeMeta / MAML+penalty cannot reduce cost even with test-time gradients**: Demonstrates that cross-episode in-context history is essential for online safety control; parameter adaptation is insufficient.
- **SCARED treats CTG as a knob**: With fixed weights, changing $\delta$ from 1 to 10 results in significantly different, monotonic behaviors, making it deployment-friendly.

## Highlights & Insights
- **First to bring safety to the ICRL paradigm**: While previous ICRL papers assumed reward-only setups, this work fills the cost channel and proves that "single-multiplier hinge" is theoretically an exact penalty, which is more elegant than squared penalties or per-episode multipliers.
- **CTG as the sole knob**: Compared to SafeAD's need to pair RTG/CTG (where mismatch leads to infeasibility), SCARED reduces this to a 1D scalar, a significant engineering simplification for real-world deployment.
- **Center $\to$ Edge OOD Protocol**: This evaluation design is a valuable contribution for future ICRL safety/generalization research to prove true extrapolation rather than interpolation.
- **Transferable Design**: (i) The single-multiplier exact-penalty dual can be applied to any actor-critic safe RL as a replacement for PD-Lagrangian; (ii) The idea of using constraint budgets as controllable inputs can extend to multi-constraint, preference-conditioned, or anytime fairness scenarios.

## Limitations & Future Work
- **No hard safety guarantee during testing**: The method ensures costs decrease over episodes statistically; it is not "anytime safe." How to enforce constraints once CTG is exhausted remains an open problem.
- **High Pre-training Cost**: Online reinforcement pretraining across many source CMDPs using a Transformer + dual critics is computationally expensive compared to offline distillation routes like AD.
- **CMDP assumes a single scalar cost**: Real-world robots often face multiple hazards. Extending the single multiplier to multiple costs requires re-deriving the exact penalty form.
- **Low-dimensional Benchmarks**: Experiments use 9×9 grids and simplified MuJoCo scenes. Sim2real deployment will require visual inputs and longer horizons.
- **No Public Code**: Reproducibility is limited without official code, which may hinder adoption compared to open-sourced alternatives like AMAGO.

## Related Work & Insights
- **vs AD / AMAGO / Headless-AD**: These focus on reward-only ICRL. Ours adds a cost channel and CTG input. Our advantage is solving safety; the disadvantage is the need for online RL pre-training.
- **vs SafeAD**: SafeAD uses behavioral cloning on PPO-Lagrangian trajectories with RTG/CTG knobs. SCARED uses online pre-training and a single CTG knob. SCARED manages to reduce cost in complex control where SafeAD fails.
- **vs SafeMeta / MAML-with-penalty**: They rely on test-time gradient updates; SCARED is pure forward adaptation. SCARED is more deployment-friendly and outperforms gradient methods in partially observable scenes.
- **vs Standard PD-Lagrangian**: Classic methods maintain one multiplier per constraint and use squared penalties. Ours uses a single-multiplier + hinge exact-penalty form, providing theoretical proof of equivalence between fixed points and primal optima (Theorem 1).
- **Insight**: Treating "user safety budget" as an explicit scalar input is an undervalued interface. When combined with RL pre-training, it allows the model to compress complex safe policies into a single inference pass—an ideal SDK-style safety interface for embodied agents.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to formalize and solve safe ICRL with a clean theoretical framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 environments + OOD/Unseen ID, though lacks real robots or visual complexity.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, clean derivations for Theorem 1 and OOD metrics.
- **Value**: ⭐⭐⭐⭐⭐ Completes a missing piece for ICRL deployment in safety-critical tasks; the single-knob CTG is very practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Safe Reinforcement Learning with Preference-Based Constraint Inference](safe_reinforcement_learning_with_preference-based_constraint_inference.md)
- [\[ICLR 2026\] Scalable In-Context Q-Learning](../../ICLR2026/reinforcement_learning/scalable_in-context_q-learning.md)
- [\[ICML 2026\] Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](safety_generalization_under_distribution_shift_in_safe_reinforcement_learning_a_.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](../../ICLR2026/reinforcement_learning/longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[NeurIPS 2025\] Towards Provable Emergence of In-Context Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/towards_provable_emergence_of_in-context_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
