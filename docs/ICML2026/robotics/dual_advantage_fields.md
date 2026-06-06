---
title: >-
  [Paper Note] Dual Advantage Fields
description: >-
  [ICML 2026 Workshop on Decision Making][Robotics][Offline GCRL] The paper observes that in the bilinear goal-conditioned value model $V_\theta(s,g)=\psi_\theta(s)^\top\phi_\theta(g)$…
tags:
  - "ICML 2026 Workshop on Decision Making"
  - "Robotics"
  - "Offline GCRL"
  - "Dual Goal Representation"
  - "Bilinear Value"
  - "Advantage Weighted Regression"
  - "Policy Extraction"
date: 2026-05-08
content_hash: 4e5e57cf404c244a
---

# Dual Advantage Fields

**Conference**: ICML 2026 Workshop on Decision Making  
**arXiv**: [2606.04188](https://arxiv.org/abs/2606.04188)  
**Code**: Not disclosed (ICML 2026 Workshop paper)  
**Area**: Reinforcement Learning / Offline Goal-Conditioned RL  
**Keywords**: Offline GCRL, Dual Goal Representation, Bilinear Value, Advantage Weighted Regression, Policy Extraction  

## TL;DR
The paper observes that in the bilinear goal-conditioned value model $V_\theta(s,g)=\psi_\theta(s)^\top\phi_\theta(g)$, the goal embedding $\phi_\theta(g)$ is precisely the gradient direction of the value field with respect to the state embedding. Consequently, an "action-feature displacement predictor" $u_\xi(s,a)\approx\gamma\psi(s')-\psi(s)$ is utilized to compute a local advantage score via an inner product with the goal embedding. This learning-free Q-network approach significantly improves the RLiable aggregate metrics across long-range navigation, manipulation, and puzzle tasks in OGBench.

## Background & Motivation

**Background**: Offline Goal-Conditioned Reinforcement Learning (GCRL) must simultaneously address two challenges: (1) **Long-range reachability**—inferring multi-step connectivity between states from a fixed dataset to "stitch" different trajectory segments; (2) **Local action selection**—choosing the action at the current state that best facilitates reaching the goal. Recent dual goal representation methods (e.g., Park et al. 2024) effectively solve long-range reachability using bilinear potential functions $V_\theta(s,g)=\psi_\theta(s)^\top\phi_\theta(g)$, which encode temporal structures and generalize to unseen $(s,g)$ pairs.

**Limitations of Prior Work**: The value surface $V_\theta(s,g)$ only indicates "how good the current state is for the goal" but **does not specify which action is better than another**. Two different actions $a_1, a_2$ starting from the same $s$ share the same $V_\theta(s,g)$, yet only one truly moves the agent towards the goal—this represents a mismatch between **global value vs. local advantage**. Existing solutions either train an additional goal-conditioned Q-network (computationally expensive and decoupled from the value representation) or use hierarchical sub-goals (HIQL). The latter performs well in long-range navigation but struggles in manipulation tasks requiring "anti-intuitive" local control, such as pre-grasping before moving to a target.

**Key Challenge**: The objective is to "retain the global stitching capability of dual representations while obtaining a local action comparison signal without an extra Q-network." If the "local advantage direction" can be directly extracted from the pre-trained goal embedding $\phi_\theta(g)$, a lightweight actor-free mechanism could replace a separate Q-network.

**Goal**: (1) Pair the global value field of dual representations with a newly proposed "local advantage field"; (2) Design an actor-free policy extraction objective; (3) Validate its consistency across the full OGBench suite (locomotion + manipulation + puzzle) compared to hierarchical and quasimetric approaches.

**Key Insight**: Under bilinear parameterization, the gradient of the value field with respect to the state embedding has a concise closed-form: $\nabla_\psi V_\theta(s,g)=\phi_\theta(g)$. Therefore, **the goal embedding itself is "the direction of steepest value increase in the representation space."** The quality of an action can be measured by the inner product of its induced displacement $\Delta\psi$ in the $\psi$-space and the goal direction $\phi(g)$, transforming policy improvement into a **geometric alignment problem** in the representation space.

**Core Idea**: Learn an action-effect model $u_\xi(s,a)\approx\gamma\psi(s')-\psi(s)$ and define the Dual Advantage Field score as $z_\theta(s,a,g)=u_\xi(s,a)^\top\phi_\theta(g)$. Adding the reward term yields the model-induced Bellman advantage. Policy extraction is then completed using Advantage Weighted Regression (AWR), eliminating the need for a goal-conditioned Q-network throughout the process.

## Method

### Overall Architecture
DAF decomposes offline GCRL into three model components: (1) A bilinear critic network $V_\theta(s,g)=\psi_\theta(s)^\top\phi_\theta(g)$ (trained with IQL expectile loss); (2) An action-displacement predictor $u_\xi(s,a)$ (trained with an sg-stop regression loss); (3) A policy $\pi_\omega(a|s,c)$ (trained using AWR weighted by DAF scores). During training, these three components are updated in parallel within a single batch, alongside twin critics $Q_\theta^{(1)}, Q_\theta^{(2)}$ for pessimistic estimation and target networks for Bellman backup stability. At inference, $\pi_\omega$ is sampled directly **without online planning or maxQ operations**.

The core observation is Proposition 3.1: The gradient of the bilinear $V$ with respect to $\psi$ equals $\phi(g)$, so the value difference for any transition $V(s',g)-V(s,g)=\phi(g)^\top(\psi(s')-\psi(s))$ reduces to an inner product.

### Key Designs

1.  **DAF Score: Expressing Bellman Advantage as "Displacement · Gradient"**:
    - **Function**: Calculates a scalar score $\hat{A}_\theta(s,a,s',g)=r(s,a,g)+\phi_\theta(g)^\top(\gamma\psi_\theta(s')-\psi_\theta(s))$ for each offline sample $(s,a,s',g)$ as a local advantage signal.
    - **Mechanism**: Under the bilinear model, $\gamma V_\theta(s',g)-V_\theta(s,g)$ is expanded directly into $\phi_\theta(g)^\top(\gamma\psi_\theta(s')-\psi_\theta(s))$ (Eq. 7), representing the GCRL advantage as the inner product of the "action-induced feature displacement" and the "goal direction." In the realizable case (where $V^\pi=\psi^\top\phi$ holds exactly), this is strictly equal to the true Bellman advantage $A^\pi(s,a,g)$ (Corollary 3.2 + Appendix F.1), making AWR-based optimization a standard policy improvement step.
    - **Design Motivation**: Traditional methods either train Q-networks (expensive, decoupled from $V$, prone to bootstrap error accumulation) or use $V$ differences (still requiring $s'$, sensitive to stochastic environments). DAF learns the "how action changes features" term independently as $u_\xi(s,a)$ and synthesizes the advantage via a closed-form inner product, **reusing the geometry of the dual critic** instead of redundant learning.

2.  **Action-effect Model $u_\xi(s,a)$: Shifting $s'$ Dependency to Training Time**:
    - **Function**: Predicts the discounted displacement induced by an action in the representation space: $u_\xi(s,a)\approx\mathbb{E}_{s'\sim p(\cdot|s,a)}[\gamma\psi_\theta(s')-\psi_\theta(s)]$.
    - **Mechanism**: The training objective is $\mathcal{L}_{\mathrm{ae}}=\mathbb{E}[\|u_\xi(s,a)-\mathrm{sg}(\gamma\psi_\theta(s')-\psi_\theta(s))\|_2^2]$, where $\mathrm{sg}$ denotes stop-gradient, ensuring $u_\xi$ tracks fixed target feature dynamics without interfering with the training of $\psi$. During inference/policy extraction, $u_\xi(s,a)$ is used directly, removing the need for an environment model to provide $s'$.
    - **Design Motivation**: Using a sample $s'$ directly to calculate $\hat{A}$ in stochastic environments leads to high variance since only one sample per $(s,a)$ exists. Learning a regression model $u_\xi$ is equivalent to an **implicit expectation** over $s'$, which reduces variance and enables online generation of advantage scores without $s'$. This also decouples "action-state" transitions from "value-goal" dynamics, allowing each to use appropriate inductive biases (e.g., MLP / Transformer).

3.  **Actor-free Coupling + AWR Policy Extraction**:
    - **Function**: Converts the DAF score $z_\theta(s,a,g)=u_\xi(s,a)^\top\phi_\theta(g)$ into softmax weights $w_\theta=\min\{\exp(\alpha z_\theta),W_{\max}\}$ and trains $\pi_\omega$ to minimize $-\mathbb{E}[w_\theta\log\pi_\omega(a|s,c)]$, equivalent to advantage-weighted behavior cloning with temperature $\alpha$.
    - **Mechanism**: Since $z_\theta$ does not depend on $\omega$, the policy loss reduces to a simple weighted BC—weighting actions in the dataset that align the "local $\psi$ displacement with the goal direction." It also incorporates Perrin-Gilbert actor-free coupling (Appendix E) to bind $V_\theta$ and $z_\theta$ via a consistency loss, avoiding fragile dependency on $\max_a Q$; this is a stability technique for continuous control.
    - **Design Motivation**: Training goal-conditioned Q-networks requires addressing overestimation in $\max_a Q$ (especially in offline settings). DAF bypasses this entirely by using the dual representation for advantage scoring and AWR for BC-style policy improvement. Empirically, this "Q-free single V + local advantage" structure is more stable than HIQL’s hierarchical approach on OGBench.

### Loss & Training
Total loss: (1) Bellman/expectile loss for $V_\theta=\psi^\top\phi$ + twin $Q^{(j)}$; (2) $\mathcal{L}_{\mathrm{ae}}$ for $u_\xi$; (3) AWR loss for $\pi_\omega$; (4) Actor-free coupling loss linking $V_\theta$ and $z_\theta$. Hyperparameters include temperature $\alpha$, clipping $W_{\max}$, and IQL expectile $\tau>0.5$.

## Key Experimental Results

### Main Results
Full OGBench suite, state-based, comparing HIQL, OTA, MQE, CRL (including DUAL variants), GCIQL, and GCIVL (including DUAL variants). Success rates are $[0,1]$, numbers represent mean ± standard deviation across multiple seeds, with the highest values within the 95% interval bolded.

| Environment | Dataset | Dim | DAF | HIQL | OTA | MQE | CRL | CRL DUAL | GCIQL |
|---|---|---|---|---|---|---|---|---|---|
| humanoidmaze | navigate | medium | **0.93±0.03** | 0.91±0.01 | **0.95±0.01** | 0.49±0.09 | 0.59±0.03 | 0.62±0.03 | 0.31±0.04 |
| humanoidmaze | navigate | large | 0.66±0.03 | 0.45±0.04 | **0.83±0.03** | 0.20±0.07 | 0.26±0.03 | 0.21±0.05 | 0.04±0.01 |
| humanoidmaze | stitch | medium | **0.90±0.04** | 0.86±0.03 | **0.92±0.01** | 0.62±0.09 | 0.53±0.03 | 0.57±0.01 | 0.15±0.03 |
| humanoidmaze | stitch | large | **0.48±0.06** | 0.32±0.04 | 0.43±0.04 | 0.18±0.03 | 0.11±0.02 | 0.06±0.03 | 0.02±0.00 |
| antmaze | navigate | teleport | 0.51±0.08 | 0.46±0.03 | 0.53±0.03 | 0.49±0.04 | **0.60±0.01** | 0.57±0.04 | — |

DAF outperforms OTA by 5 percentage points and HIQL by 16 percentage points on the hardest setting, `humanoidmaze-stitch-large`, which requires stitching short trajectories. DAF matches or slightly trails OTA on `navigate` tasks (OTA is specialized for high-level subgoals), despite DAF not using a hierarchical structure.

### Visualization of local counter-intuitive behavior (cube-single manipulation task)

| Method | High-level direction around cube (pre-grasp) | Behavioral Meaning |
|---|---|---|
| OTA | Points to final placement position | Tries to push directly, but gripper hasn't grasped → Fail |
| DAF | Points to the cube itself | Grasps first, then considers placement → Success |

By decoding high-level latent outputs $h_m(\tilde{s}_i,g)$ to the XY plane using linear probes, DAF is shown to generate a vector field "pointing to the cube" (pre-grasp behavior), whereas OTA generates a field "pointing to the goal." Essentially, DAF learns that **local sub-goals $\neq$ final goal** geometry.

### Key Findings
- **DAF's Gain is maximal on stitch datasets**: The `navigate` dataset consists of noisy expert trajectories where global value is sufficient. The `stitch` dataset requires stitching short segments, necessitating the "ability to pick actions that truly move the state forward locally"—this is DAF's core design strength.
- **Superiority in manipulation tasks comes from local geometry**: In tasks like `cube-single` requiring pre-grasping, hierarchical methods (OTA/HIQL) often point in the wrong high-level direction. DAF's local gradient of $\phi(g)$ naturally produces pre-grasp behavior without explicit sub-goal decoding.
- **DUAL variants are not necessarily better**: Simply applying dual goal representations to existing methods yields limited improvements. The key is **using the geometry of the dual representation to generate action advantages**, rather than just as a state representation.
- **Clear scaling trends**: On `humanoidmaze-large`, the gap between DAF and HIQL widens from +3% (medium) to +21%, indicating that local advantage is increasingly critical for long-range tasks.

## Highlights & Insights
- **"Goal embedding = Value gradient direction" is an elegant geometric observation**: While $\nabla_\psi V_\theta=\phi_\theta(g)$ is simple algebra under bilinear parameterization, interpreting it as a "local improvement direction" paired with an action-effect model inner product provides a minimal policy extraction mechanism. Any method using bilinear/inner-product value functions (Contrastive RL, Successor Features, ICVF) can adapt this approach.
- **Actor-free policy extraction avoids maxQ difficulties**: In offline continuous control, $\max_a Q$ is a significant source of overestimation. DAF bypasses this with "dual value field + weighted BC" without losing policy improvement guarantees (strictly equal to Bellman advantage in the realizable case). This is insightful for general offline RL.
- **Transferability to the Successor Measure framework**: DAF's $u_\xi(s,a)$ is structurally identical to Successor Features. By treating SF's $\psi$ as state features and reward weights as $\phi(g)$, DAF is equivalent to "using SF geometry for action ranking," allowing it to leverage GPI/GPE tools.
- **Quantifying "Local vs. Global" requirements**: Using vector field comparisons to visualize concepts like "local subgoal $\neq$ final goal" provides a **geometric diagnostic** more convincing than success rates alone. This should be a standard visualization for manipulation GCRL.

## Limitations & Future Work
- **Strong Assumption: Realizable bilinear value**: Theoretical guarantees rely on $V^\pi(s,g)=\psi(s)^\top\phi(g)$ holding exactly. In practice, the impact of approximation errors on the policy improvement step lacks quantitative analysis.
- **Potential degradation of action-effect models in large action spaces**: $u_\xi(s,a)$ is a regression model. in environments with high action dimensions or multi-modal transitions (e.g., sim2real), point regression may underfit. Conditioning on flows or diffusion models could be a solution.
- **Lack of image-based task comparison**: All experiments are state-based. DAF's performance on pixel observations (like OGBench-Visual) is unverified, where learning $\psi(s)$ itself is more challenging.
- **Lack of online fine-tuning experiments**: A key selling point of DAF is its "actor-free, high scalability" nature, yet only the pure offline setting is shown. Its stability during offline-to-online transitions remains to be tested.

## Related Work & Insights
- **vs. HIQL (Park et al. 2023)**: HIQL uses hierarchical policies and a single V, relying on sub-goal decomposition; DAF uses a single level with a local advantage field, requiring no sub-goal decoding. DAF leads significantly in `stitch` and manipulation tasks.
- **vs. OTA / MQE / Quasimetric RL**: OTA uses option-aware abstractions; quasimetric methods use dual norms. DAF's advantage lies in not needing task-specific architectural modules for long-range or manipulation tasks.
- **vs. Dual Goal Representation (Original Park et al. 2024)**: The original work only uses dual representation for value estimation; DAF **repurposes the same gradient information** for policy extraction, effectively "leveraging geometry twice from one representation learning step."
- **vs. Successor Features**: $u_\xi(s,a)$ is structurally equivalent to SF, but uses a single-step difference $\gamma\psi(s')-\psi(s)$ rather than full SF accumulation, making it simpler. SF + GPI is a potential extension for DAF.
- **vs. Dayan & Singh (1996) Comparative Policy Improvement**: DAF is a modern instantiation of the "using relative advantage to improve policy" idea within the dual representation framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The geometric observation is clever, though it is a concise combination of bilinear gradients and AWR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers OGBench, RLiable metrics, and counter-intuitive cases, though lacks pixel tasks and online fine-tuning.
- Writing Quality: ⭐⭐⭐⭐ Concise formulas; pre-grasp visualization is very clear; propositions/corollaries are well-organized.
- Value: ⭐⭐⭐⭐ Provides a lightweight, Q-free policy extraction paradigm for the offline GCRL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)
- [\[ICML 2026\] Dual Quaternion SE(3) Synchronization with Recovery Guarantees](dual_quaternion_se3_synchronization_with_recovery_guarantees.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[ACL 2026\] Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System](../../ACL2026/robotics/libra-vla_achieving_learning_equilibrium_via_asynchronous_coarse-to-fine_dual-sy.md)
- [\[CVPR 2026\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](../../CVPR2026/robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)

</div>

<!-- RELATED:END -->
