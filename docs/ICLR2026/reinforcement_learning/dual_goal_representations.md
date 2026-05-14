---
title: >-
  [Paper Note] Dual Goal Representations
description: >-
  [ICLR 2026][Reinforcement Learning][goal-conditioned RL] This paper proposes *dual goal representations*, which encode a goal state via the set of optimal temporal distances from all states to that goal. The authors theo…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "goal-conditioned RL"
  - "dual goal representation"
  - "temporal distance"
  - "asymmetric inner product parameterization"
  - "OGBench"
date: 2026-05-08
content_hash: 7128ac00f8678d68
---

# Dual Goal Representations

**Conference**: ICLR 2026
**arXiv**: [2510.06714](https://arxiv.org/abs/2510.06714)
**Code**: None
**Area**: Reinforcement Learning / Goal-Conditioned RL
**Keywords**: goal-conditioned RL, dual goal representation, temporal distance, asymmetric inner product parameterization, OGBench

## TL;DR

This paper proposes *dual goal representations*, which encode a goal state via the set of optimal temporal distances from all states to that goal. The authors theoretically prove that this representation is sufficient for recovering the optimal policy and naturally filters exogenous noise. A practical learning algorithm based on asymmetric inner product parameterization is designed, and the resulting module consistently improves three mainstream offline GCRL methods across 20 OGBench tasks as a plug-and-play component.

## Background & Motivation

**Background**: Goal-conditioned reinforcement learning (GCRL) requires agents to learn to reach arbitrary goal states from arbitrary starting states. Existing methods typically feed raw state observations (e.g., pixel images, full joint-angle state vectors) directly into policy networks as goal inputs, or employ metric learning approaches (TCN, VIP, HILP, etc.) to learn a state embedding space in which L2 norm measures inter-state distances.

**Limitations of Prior Work**: Raw observations contain substantial information irrelevant to "how to reach the goal"—background textures, lighting variations, positions of irrelevant objects, and other exogenous noise. Although metric learning methods learn state embeddings, L2 norm parameterization imposes structural limitations: it is inherently symmetric ($\|f(s)-f(g)\| = \|f(g)-f(s)\|$), whereas true temporal distances are often asymmetric (e.g., dropping a block is much faster than picking it up). Furthermore, L2 norms are constrained by the triangle inequality, limiting their expressiveness and precluding arbitrary approximation of the true temporal distance function.

**Key Challenge**: Existing goal representations either suffer from information redundancy (raw observations contain noise) or are limited by the expressive capacity of their parameterization (L2 metrics are not universal approximators). A theoretically grounded yet practically viable framework for goal representation learning is lacking.

**Goal**: (1) Formally define what constitutes a "good goal representation" and establish its theoretical properties; (2) design a more expressive parameterization to learn such representations in practice; (3) make the representation learning module compatible with arbitrary existing GCRL algorithms.

**Key Insight**: The authors observe that if two goal states share identical distributions of optimal temporal distances from all other states in the environment, then the two goals are equivalent in terms of reachability—regardless of how their raw observations differ. This naturally motivates a "dual" perspective: rather than encoding the features of the goal itself, the goal is characterized indirectly through its relationship with all other states.

**Core Idea**: The temporal distances from all states to a goal collectively serve as the goal's representation. This representation is theoretically sufficient for recovering the optimal policy, invariant to exogenous noise, and practically learnable via asymmetric inner product parameterization.

## Method

### Overall Architecture

The method proceeds in two concurrently trained stages. **Stage 1 (Representation Learning)**: Using offline trajectory data, a parameterized distance function $d^*(s,g) \approx \psi(s)^\top \phi(g)$ is trained via goal-conditioned IQL (Implicit Q-Learning), where $\psi$ is the state encoder and $\phi$ is the goal encoder. After training, $\phi(g)$ is extracted as the goal representation. **Stage 2 (Policy Learning)**: $\phi(g)$ serves as a compact goal representation fed into any downstream GCRL algorithm (e.g., GCIVL, CRL, GCFBC), training a policy of the form $\pi(a|s, \phi(g))$. Both stages share the same dataset and update gradients jointly (not pre-train then fine-tune). For state-based tasks, training runs for 1M steps.

### Key Designs

1. **Dual Goal Representation (Theoretical Definition)**:

    - **Function**: Provides a compact goal representation that depends only on the environment dynamics and is independent of raw observations.
    - **Mechanism**: For goal $g$, the dual representation is defined as $\phi^\vee(g) = [d^*(s_1,g), d^*(s_2,g), \dots, d^*(s_K,g)]^\top$, i.e., a vector of optimal temporal distances from $K$ reference states to $g$. In finite-state MDPs, this corresponds to a column of the distance matrix. In continuous spaces, the parameterization $\psi(s)^\top\phi(g)$ implicitly encodes "distances from all states to $g$"—given $\phi(g)$, its inner product with any $\psi(s)$ recovers the corresponding distance.
    - **Design Motivation**: Traditional representations encode the intrinsic features of a state ("what is at this location"), whereas the dual representation encodes relational structure ("how far away is this location to reach"), which naturally retains only control-relevant information.

2. **Asymmetric Inner Product Parameterization**:

    - **Function**: Provides a universal approximator form for the distance function, replacing the L2 norm.
    - **Mechanism**: The distance function is modeled as $d^*(s,g) = \psi(s)^\top\phi(g)$, using two independent encoders $\psi$ and $\phi$ to map states and goals into $N$-dimensional vectors. The key property, established by METRA, is that this inner product form is a universal approximator—for sufficiently large $N$, it can approximate any continuous function $f(s,g)$ to arbitrary precision.
    - **Design Motivation**: The authors compare four parameterizations: (1) symmetric L2 $\|\phi(s)-\phi(g)\|$, (2) asymmetric L2 $\|\psi(s)-\phi(g)\|$, (3) symmetric inner product $\phi(s)^\top\phi(g)$, and (4) asymmetric inner product $\psi(s)^\top\phi(g)$. Only (4) is a universal approximator: L2 norms are inherently constrained by the triangle inequality and symmetry, while the symmetric inner product cannot model asymmetric distances ($d^*(s,g) \neq d^*(g,s)$). Formal proofs are provided showing that the first three fail to satisfy universality.

3. **IQL-Based Distance Function Learning**:

    - **Function**: Learns $\psi$ and $\phi$ from offline trajectory data.
    - **Mechanism**: The temporal distance $d^*(s,g)$ is equivalently reformulated as the optimal value function $V^*(s,g)$ (up to sign), and Implicit Q-Learning is used for approximation. IQL estimates the optimal value function from offline data via expectile regression, avoiding queries to unseen $(s,a)$ pairs. The network maps state $s$ through $\psi$ and goal $g$ through $\phi$ into $N$-dimensional spaces, with $\psi(s)^\top\phi(g)$ as the distance output.
    - **Design Motivation**: Theoretically, IQL recovers $V^*$ exactly as the expectile $\tau \to 1$ under sufficient data coverage, guaranteeing the theoretical correctness of the learned representation in the limit.

### Loss & Training

Training uses the standard IQL losses: expectile regression for the value function $V$, TD loss for the Q-function, and advantage-weighted regression for the policy. The distance function is parameterized in bilinear form as $\psi(s)^\top\phi(g)$. When $\phi(g)$ is passed to the downstream GCRL algorithm, gradients are stopped (stop-gradient), so the downstream policy $\pi(a|s,\phi(g))$ does not backpropagate into $\phi$. This design choice is motivated by the observation that policy learning requires richer information than the goal representation alone (e.g., in antmaze, the goal representation need only encode the x-y position, whereas the policy requires all joint angles). Allowing the downstream algorithm to learn its own value function and policy is more effective, as confirmed by the ablation in Table 5.

## Key Experimental Results

### Main Results

Experiments are conducted on 20 tasks from the OGBench benchmark suite, comprising 13 state-based tasks and 7 pixel-based tasks. The dual representation is combined with three downstream GCRL algorithms (GCIVL, CRL, GCFBC) and compared against baselines including raw observations (Original), TCN, VIP, and HILP.

| Method | Downstream Algorithm | State-based (13 tasks) Avg. | Pixel-based (7 tasks) Avg. | Notes |
|--------|---------------------|-----------------------------|---------------------------|-------|
| Original (no representation) | GCIVL / CRL / GCFBC | Baseline | Baseline (early fusion available) | No representation bottleneck |
| TCN | GCIVL / CRL / GCFBC | Moderate | Moderate | Symmetric L2 |
| VIP | GCIVL / CRL / GCFBC | Moderate | Moderate | Symmetric L2 |
| HILP | GCIVL / CRL / GCFBC | Good | Moderate | Symmetric L2 |
| **Dual (Ours)** | GCIVL / CRL / GCFBC | **Best or tied best** | **Best on most tasks** | Asymmetric inner product, universal |

On state-based tasks, the dual representation achieves best or tied-best performance on most tasks. On pixel-based tasks, it is also optimal on the majority of tasks; however, all representation learning methods perform poorly on visual puzzle tasks—because representation learning methods use late fusion (encoding $s$ and $g$ separately before combining), whereas the Original baseline can use early fusion (concatenating before encoding), which is required for the precise pixel-level alignment that puzzle tasks demand.

### Ablation Study: Parameterization Comparison

The authors compare four parameterization forms on 13 state-based tasks in terms of average performance and distance function accuracy (measured as MSE against a monolithic value function $V(s,g)$ trained independently, serving as an oracle):

| Parameterization | Universal | Distance Error (↓) | Avg. Performance (↑) | Notes |
|------------------|-----------|--------------------|----------------------|-------|
| (1) Symmetric L2 $\|\phi(s)-\phi(g)\|$ | ✗ | Moderate | Moderate | Cannot model asymmetric distances; triangle inequality constrained |
| (2) Asymmetric L2 $\|\psi(s)-\phi(g)\|$ | ✗ | Moderate | Moderate | Still constrained by norm structure |
| (3) Symmetric inner product $\phi(s)^\top\phi(g)$ | ✗ | Highest | Lowest | Forces symmetry $d(s,g)=d(g,s)$; worst overall |
| **(4) Asymmetric inner product $\psi(s)^\top\phi(g)$** | **✓** | **Lowest** | **Highest** | No structural constraint; most expressive |

A key finding is that the distance error ranking (3) > (1) ≈ (2) > (4) is fully consistent with the performance ranking, confirming the causal chain: more accurate distance estimation → better goal representation → better policy performance.

### Other Ablations and Analyses

| Experiment | Conclusion |
|------------|------------|
| Directly extracting policy from $\psi(s)^\top\phi(g)$ vs. training independent GCRL (Table 5) | Independent GCRL is significantly superior; the distance function suffices for learning the representation but is insufficiently precise for direct control. |
| Representation dimension $N \in \{32, 64, 256\}$ | Inner product form: increasing $N$ reduces error and improves performance; metric form: error saturates with increasing $N$ due to the triangle inequality constraint. |
| Noise robustness (Figure 4) | Adding Gaussian noise to goals at evaluation time causes significantly smaller performance degradation for dual representations compared to Original. |
| 2× training steps ablation | Original trained for 2M steps still underperforms Dual trained for 1M steps, indicating that the representational advantage cannot be recovered simply by increasing training budget. |

### Key Findings

- **Strong plug-and-play compatibility**: Dual representations consistently improve all three downstream algorithms (GCIVL, CRL, GCFBC), demonstrating the generality of the approach.
- **Asymmetry is critical**: Symmetric parameterizations (both L2 and inner product) are significantly inferior in performance and accuracy, consistent with the inherently asymmetric nature of temporal distances in real environments (e.g., directional traversal difficulty in antmaze).
- **Representation ≠ control**: Although $\phi(g)$ is an effective goal representation, directly extracting a policy from $\psi(s)^\top\phi(g)$ performs poorly—policy learning requires richer state information than goal representation alone provides.
- **Distinct scaling behavior**: Inner product parameterization continues to benefit from increasing $N$, while metric parameterizations exhibit a "ceiling effect" due to structural constraints.

## Highlights & Insights

- The **"define entities via relations" dual perspective** is elegant: rather than encoding what the goal is, the representation encodes "how far away is it from everywhere"—analogous to uniquely identifying nodes in a graph via distance vectors. The key insight is that this relational representation naturally retains only control-relevant information, filtering out exogenous noise unrelated to the dynamics.
- **The universality of asymmetric inner products vs. the non-universality of metrics** constitutes the paper's most technically substantial contribution. The authors not only invoke METRA's theoretical result to support the universality of inner products, but also construct counterexamples formally proving that L2 norms (including asymmetric variants) are not universal approximators, providing rigorous theoretical justification for the parameterization choice.
- The seemingly paradoxical conclusion that **"representations are sufficient to recover the optimal policy, yet insufficient for directly extracting it"** is deeply insightful: representational sufficiency is an information-theoretic property (information containment), whereas policy extraction additionally requires that the inner product form achieves sufficient function approximation accuracy for argmax operations—two qualitatively different requirements.

## Limitations & Future Work

- **Theory-practice gap**: The invariance and sufficiency theorems (Theorems 3.1 and 3.2) are proved for idealized infinite-dimensional dual representations; the theoretical guarantees do not strictly hold for the finite-dimensional $\phi(g)$ used in practice. The justification relies primarily on empirical validation, lacking finite-sample or finite-dimensional theoretical analysis.
- **Insufficient robustness validation**: The theory is grounded in the Ex-BCMP model (requiring disjoint observation supports for endogenous and exogenous states), yet experiments use only simple Gaussian noise. More compelling tests would employ structured distractors (e.g., moving irrelevant objects in the background, changing textures) rather than additive noise.
- **Early/late fusion asymmetry in pixel tasks**: All representation learning methods use late fusion, while the Original baseline may use early fusion—rendering the pixel-based comparison not entirely fair. The authors acknowledge that state-aware representations or information bottlenecks could alleviate this, but leave it to future work.
- **Offline setting only**: The entire framework depends on offline collected trajectory data; whether representations and policies can be effectively co-trained in online settings remains unclear.
- **No real-robot experiments**: Although OGBench is diverse (20 tasks), all experiments are conducted in simulation.

## Related Work & Insights

- **vs. TCN / VIP / HILP**: These methods similarly learn state/goal embeddings for GCRL but employ symmetric L2 norm parameterization, limiting their expressiveness. The key distinctions of the dual representation are: (1) only the goal encoder $\phi(g)$ is learned rather than a joint state representation; (2) a universal approximator inner product parameterization is used; and (3) invariance and sufficiency are theoretically guaranteed.
- **vs. Quasimetric RL (QRL)**: QRL also learns asymmetric distances but uses a quasimetric parameterization satisfying the triangle inequality. The dual representation sacrifices the triangle inequality in exchange for universal approximation; experiments demonstrate this trade-off is worthwhile.
- **vs. Contrastive RL (CRL)**: CRL trains value functions via contrastive learning objectives and can use its internal goal encoder as a representation. The dual representation can be layered on top of CRL for further improvement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of the dual perspective and universal approximator argument is novel, though the implementation-level distinction from existing metric learning methods primarily concerns the choice of parameterization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Twenty tasks, three downstream algorithms, and detailed ablations; however, noise testing is overly simplistic and real-robot experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ The connection between theory and practice is presented clearly, though multiple reviewers noted that Section 4's implementation details were insufficiently clear in the initial submission.
- **Value**: ⭐⭐⭐⭐ The plug-and-play module offers direct practical value to the GCRL community, and the theoretical analysis provides a guiding framework for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning](dvla-rl_dual-level_vision-language_alignment_with_reinforcement_learning_gating_.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](../../NeurIPS2025/reinforcement_learning/learning_from_demonstrations_via_capability-aware_goal_sampling.md)

</div>

<!-- RELATED:END -->
