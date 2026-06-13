---
title: >-
  [Paper Note] Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning
description: >-
  [ICML2026][Reinforcement Learning][Offline goal-conditioned reinforcement learning] This paper proposes CTA (Compositional Transduction with latent Analogies)…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Offline goal-conditioned reinforcement learning"
  - "compositional generalization"
  - "analogical transduction"
  - "temporal distance difference fields"
  - "bilinear transduction"
date: 2026-05-08
content_hash: 4409ae61b110454a
---

# Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning

**Conference**: ICML2026  
**arXiv**: [2605.20609](https://arxiv.org/abs/2605.20609)  
**Code**: https://rllab-snu.github.io/projects/CTA/  
**Area**: robotics  
**Keywords**: Offline goal-conditioned reinforcement learning, compositional generalization, analogical transduction, temporal distance difference fields, bilinear transduction  

## TL;DR

This paper proposes CTA (Compositional Transduction with latent Analogies), which factorizes goal-reaching tasks into "task-intrinsic analogies" and "task-extrinsic context." By utilizing temporal distance difference fields as analogy representations and combining them with bilinear transduction to extrapolate to unseen analogy-context combinations, CTA outperforms the strongest baseline by approximately 42% on OGBench manipulation environments.

## Background & Motivation

**Background**: Offline goal-conditioned reinforcement learning (offline GCRL) aims to train a general goal-reaching agent from reward-free offline data. Existing methods primarily achieve compositional generalization through **trajectory stitching**, which connects temporally adjacent segments to synthesize new goal-reaching behaviors.

**Limitations of Prior Work**: Trajectory stitching can only combine behavior segments that are temporally adjacent but fails to address a complementary compositional requirement: reusing the same **task-relevant behavioral transformation under different task-irrelevant contexts**. For instance, an agent might learn to open a drawer while a window is open, but it fails to transfer this behavior when the window is closed, as the specific combination of these contexts was never co-present in the training data.

**Key Challenge**: Offline data is limited and cannot cover all "task $\times$ context" combinations. Existing methods lack a mechanism to decouple and recombine **task-intrinsic transformations** (e.g., opening a drawer) from **task-extrinsic contexts** (e.g., window status), leading to generalization failure on unseen combinations.

**Goal**: (1) Define "analogy" and provide a learnable representation; (2) Solve the out-of-distribution (OOC) extrapolation problem for unseen analogy-context combinations.

**Key Insight**: The authors observe that the quasimetric space induced by the optimal temporal distance $d^*(s,g)$ is invariant to task-extrinsic contexts. Furthermore, the **temporal distance difference field** of a state-goal pair, $\alpha(s,g)(x) = d^*(x,g) - d^*(x,s)$, encodes the task-intrinsic displacement and is sufficient to support optimal goal-reaching.

**Core Idea**: Use the temporal distance difference field as the representation for task-intrinsic analogies. Decouple analogies and contexts into low-rank factors through bilinear transduction to achieve reliable extrapolation to unseen combinations (OOC).

## Method

### Overall Architecture

CTA consists of two stages: **analogy extraction** and **analogy transduction**. First, a pair of encoders $\phi, \varphi$ is learned to approximate the optimal temporal distance via $d^*(s,g) = \phi(s)^\top \varphi(g)$, resulting in the **dual analogy** $\alpha^\vee(s,g) = \varphi(g) - \varphi(s)$, a finite-dimensional instantiation of the temporal distance difference field. Subsequently, in the analogy transduction stage, the dual analogy is used as a displacement signal. The value function and hierarchical policy are parameterized via bilinear transduction, enabling the agent to extrapolate to unseen analogy-context combinations. During inference, the high-level policy generates $k$-step sub-goal analogies, while the low-level policy executes primitive actions.

### Key Designs

1.  **Temporal Distance Difference Field and Dual Analogy Representation**:
    - **Function**: Provides an analogy representation that is invariant to task-extrinsic contexts and sufficient for task-intrinsic displacement.
    - **Mechanism**: For a state-goal pair $(s,g)$, the temporal distance difference field is defined as $\alpha(s,g)(x) = d^*(x,g) - d^*(x,s)$. It aggregates the difference in difficulty between reaching $g$ and reaching $s$ across all probe states $x$, forming a "signature" of the task-intrinsic displacement. In practice, with $d^*$ parameterized as the inner product $\phi(s)^\top \varphi(g)$, the field simplifies to the dual analogy $\alpha^\vee(s,g) = \varphi(g) - \varphi(s)$, a $d$-dimensional vector independent of probe states $x$.
    - **Design Motivation**: A single scalar $d^*(s,g)$ can be degenerate (different tasks mapping to the same value), whereas the difference field eliminates degeneracy by using relative comparisons across the entire state space. Unlike bisimulation-based analogies, this method is constructed based on optimal temporal distance and is not affected by policy quality fluctuations in suboptimal data, making it more suitable for offline scenarios.

2.  **Bilinear Transduction Parameterization**:
    - **Function**: Enables the value function and policy to extrapolate to unseen analogy-context combinations (OOC generalization).
    - **Mechanism**: The value function is parameterized as $V(s,g) = \Omega_1(s) \cdot \Omega_2(\alpha^\vee(s,g))$, where $\Omega_1$ and $\Omega_2$ encode the current state (anchor) and the analogy (displacement) respectively into a $b$-dimensional low-rank bottleneck space ($b \ll d$). The policy mean also adopts a bilinear form, e.g., the high-level policy $\mu_h(s, \alpha^\vee(s,g)) = \omega_{h1}(s) \cdot \omega_{h2}(\alpha^\vee(s,g))$. Low-rank constraints allow the network to learn independently on the marginal distributions of anchors and displacements, enabling natural extrapolation to unseen combinations via the tensor product during inference.
    - **Design Motivation**: Standard MLP fitting of $V(s, \alpha^\vee)$ couples anchors and displacements, failing to generalize to combinations not co-occurring during training. Bilinear transduction provides theoretically guaranteed OOC error bounds.

3.  **Hierarchical Analogical Transduction Policy**:
    - **Function**: Decomposes long-range analogy transduction into short-range sub-tasks to improve data efficiency and transduction stability.
    - **Mechanism**: The high-level policy $\pi_h$ is conditioned on the current state and the final goal analogy, outputting a $k$-step sub-goal analogy $\alpha^\vee(s_t, s_{t+k})$. The low-level policy $\pi_\ell$ is conditioned on the current state and the sub-goal analogy, outputting the primitive action $a_t$. Both layers are trained using Advantage Weighted Regression (AWR).
    - **Design Motivation**: Long-range analogies are sparse in offline data, making direct long-range transduction unreliable. Decomposing into $k$-step sub-tasks significantly increases the amount of reusable short-range analogies and prevents the low-level policy from querying analogies outside the expected OOC range.

### Loss & Training

In the analogy extraction stage, $\phi, \varphi$ and the $Q$-function are trained using the IQL expectile loss (Eq. 9). In the analogy transduction stage, the value function $V$ is trained using an action-free IQL loss (Eq. 13). Both high-level and low-level policies are trained with Advantage Weighted Regression losses (Eqs. 14, 15), where temperature parameters $\beta_h, \beta_\ell$ control the weight of behavior cloning. Target networks are used to stabilize training.

## Key Experimental Results

### Main Results

Comparison with 11 baselines across 8 manipulation environments in OGBench (8 random seeds):

| Environment | GCBC | HIQL | GCIQL | GCIVL∨ | HIQL∨ | HIQL+α∨ | **CTA** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| scene-play | 5 | 38 | 51 | 72 | 87 | 80 | **90** |
| cube-single-play | 6 | 15 | 68 | 89 | 69 | 74 | **86** |
| cube-double-play | 1 | 6 | 40 | 60 | 38 | 30 | **50** |
| cube-triple-play | 1 | 3 | 3 | 2 | 18 | 11 | **17** |
| puzzle-3x3-play | 2 | 12 | 95 | 5 | 79 | 72 | **94** |
| puzzle-4x4-play | 0 | 7 | 26 | 23 | 16 | 50 | **84** |
| puzzle-4x5-play | 0 | 4 | 14 | 5 | 5 | 0 | **17** |
| puzzle-4x6-play | 0 | 3 | 12 | 2 | 2 | 0 | **12** |
| **Average** | 1.9 | 11.0 | 38.6 | 32.2 | 39.3 | 39.6 | **56.3** |

### OOC Extrapolation Case Study

Specific analogy-context combinations were intentionally removed from the training data in `scene` and `puzzle-4x4` to test the direct success rate:

| Environment | HIQL | GCIQL∨ | HIQL∨ | HIQL+α∨ | **CTA** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| scene | 19±10 (42±12) | 51±10 (63±11) | 45±11 (87±7) | 48±14 (86±6) | **73±9 (94±4)** |
| puzzle-4x4 | 37±11 (69±9) | 44±11 (55±12) | 35±17 (62±13) | 66±11 (95±4) | **80±8 (100±1)** |

> Values outside parentheses represent direct success rates (trajectories completing the task directly); values inside parentheses represent total success rates (including completion via detours).

### Key Findings

- **Maximum gain in Puzzle environments**: The state space grows exponentially, making compositional generalization critical. CTA improves average performance by approximately 40% across 4 puzzle environments, reaching 2.5x the performance of the strongest baseline on 4×4.
- **OOC extrapolation drives performance gains**: The performance of HIQL∨ and HIQL+α∨ is similar (39.3 vs 39.6), indicating that simply using the dual analogy representation is insufficient. Significant improvements are achieved only when CTA applies bilinear transduction for OOC extrapolation.
- **CTA has fewer parameters**: Despite using bilinear parameterization, CTA has approximately 20% fewer parameters than HIQL+α∨, ruling out model capacity as the source of gains.

## Highlights & Insights

- **Temporal distance difference field as an analogy representation** cleverly embeds distance differential embeddings from metric geometry into RL. By using relative comparisons across all probe states, it eliminates scalar degeneracy and naturally achieves context invariance. This approach can be transferred to any scenario requiring task semantic invariance across varying environments.
- **OOC guarantees of bilinear transduction** apply the OOC generalization theory of Netanyahu et al. (2023) to value function and policy parameterization in offline GCRL for the first time, achieving independent factor-level generalization with low-rank bottlenecks. This paradigm can be extended to other sequential decision-making problems requiring compositional generalization.
- **t-SNE visualization of analogies** intuitively verifies that learned dual analogies cluster according to task semantics (e.g., "opening drawer" vs. "closing drawer") and remain invariant to context factors such as windows or buttons.

## Limitations & Future Work

- **Strong Assumption 4.3**: Requires task-intrinsic components of all state-goal pairs within the same task block to remain consistent across different comparison endpoints, which may not hold in complex real-world environments.
- **Gap between dual analogy and theory**: In practice, $\varphi(g) - \varphi(s)$ is not guaranteed to be minimal or identifiable, leading to approximation errors relative to the theoretically invariant difference field.
- **Scope limitations**: CTA shows limited advantages in environments lacking clear task-context separation (e.g., mazes); the design is biased toward manipulation tasks.
- Future directions include more accurate approximation of distance difference fields and extending analogy transduction to continuous control and high-dimensional visual observation scenarios.

## Related Work & Insights

- **Offline GCRL**: HIQL (Hierarchical IQL), GCIQL, QRL (Quasimetric RL), and CRL (Contrastive RL) provide different schemes for temporal distance estimation. CTA introduces an analogy layer on top of these.
- **Analogy and Representation Learning**: Linear offset analogies in word2vec inspired the difference vector design in this work; goal-conditioned bisimulation is recent related work but relies on on-policy reward matching, which is unsuitable for offline settings.
- **Dual Goal Representation**: The dual representation $\varphi(g)$ in Park et al. (2026) shares encoder training methods with the dual analogy $\varphi(g) - \varphi(s)$ used here, but the former lacks analogical transduction capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning](latent_representation_alignment_for_offline_goal-conditioned_reinforcement_learn.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICML 2026\] Mind Dreamer: Untethering Imagination via Active Causal Intervention on Latent Manifolds](mind_dreamer_untethering_imagination_via_active_causal_intervention_on_latent_ma.md)
- [\[ICML 2026\] Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](beyond_the_proxy_trajectory-distilled_guidance_for_offline_gflownet_training.md)

</div>

<!-- RELATED:END -->
