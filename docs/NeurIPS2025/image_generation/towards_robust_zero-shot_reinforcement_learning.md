---
title: >-
  [Paper Note] Towards Robust Zero-Shot Reinforcement Learning
description: >-
  [NeurIPS 2025][Image Generation][Zero-shot reinforcement learning] This paper proposes BREEZE, a framework that systematically addresses out-of-distribution (OOD) extrapolation errors and insufficient expressivity in FB-based zero-shot RL through behavior-regularized representation guidance, task-conditioned diffusion policy extraction, and attention-enhanced representation modeling. BREEZE achieves state-of-the-art or near-state-of-the-art robust zero-shot generalization on ExORL and D4RL Kitchen benchmarks.
tags:
  - NeurIPS 2025
  - Image Generation
  - Zero-shot reinforcement learning
  - Forward-Backward representation
  - behavior regularization
  - diffusion policy
  - attention architecture
date: 2026-05-08
content_hash: 756cdd40d6d84c5c
---

# Towards Robust Zero-Shot Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.15382](https://arxiv.org/abs/2510.15382)
**Code**: [GitHub](https://github.com/Whiterrrrr/BREEZE)
**Area**: Image Generation
**Keywords**: Zero-shot reinforcement learning, Forward-Backward representation, behavior regularization, diffusion policy, attention architecture

## TL;DR

This paper proposes BREEZE, a framework that systematically addresses out-of-distribution (OOD) extrapolation errors and insufficient expressivity in FB-based zero-shot RL through behavior-regularized representation guidance, task-conditioned diffusion policy extraction, and attention-enhanced representation modeling. BREEZE achieves state-of-the-art or near-state-of-the-art robust zero-shot generalization on ExORL and D4RL Kitchen benchmarks.

## Background & Motivation

Zero-shot reinforcement learning aims to pretrain a universal policy on reward-free offline transition data, enabling zero-shot adaptation to arbitrary downstream tasks. The Forward-Backward (FB) representation is the dominant approach in this area, decomposing occupancy measures into a forward representation $F$ and a backward representation $B$ to approximate the successor measure: $M^{\pi_z}(s_0,a_0,ds_+) \approx F(s_0,a_0,z)^\top B(s_+)\rho(ds_+)$, from which $Q$-values for any task are computed as $Q_z(s,a) = F(s,a,z)^\top z$.

Through empirical investigation, the authors identify two core problems with existing FB methods:

**Severe bias in successor measure estimation**: $M^{\pi_z}$ should theoretically be non-negative (representing future state occupancy), yet the learned $F^\top B$ contains substantial invalid negative values and extreme magnitude mismatches. These inaccurate representations propagate into $Q$-value estimation, causing systematic prediction bias.

**Insufficient expressivity**: Learning successor measures and policies that cover all possible task vectors $z$ demands highly expressive models, yet existing FB methods are limited in both representation networks and policy capacity. Simply scaling up MLP size yields no improvement, indicating that the issue lies in the architectural design itself.

Furthermore, offline learning causes actions produced by $\pi_z(s_{t+1})$ to be potentially OOD, leading to extrapolation errors. CQL-style regularization (e.g., MCFB) provides partial relief but cannot fully resolve distribution shift.

## Method

### Overall Architecture

BREEZE (Behavior-REgularizEd Zero-shot RL with Expressivity Enhancement) operates simultaneously at three levels: (1) behavior regularization to stabilize representation learning; (2) a diffusion model to enhance policy expressivity; and (3) an attention-based architecture to improve representation modeling capacity.

### Key Designs

1. **Behavior-Regularized Representation Guidance**

   The core idea is to introduce a task-conditioned state value function $V_{\pi_z}(s,z)$ as a stable substitute for the unstable target $Q$ approximation:
    $V_{\pi_z}(s,z) := \max_{a \in A, \text{s.t.} \mu(a|s)>0} F(s,a,z)^\top z$

   This is solved via expectile regression:
    $\mathcal{L}_{V_{\pi_z}} = \mathbb{E}_{(s,a)\sim\mathcal{D}, z\sim\mathcal{Z}} \left[ L_2^\tau(F(s,a,z)^\top z - V_{\pi_z}(s,z)) \right]$

   where $L_2^\tau(u) = |\tau - \mathbb{I}(u<0)|u^2$ and $\tau > 0.5$. The unstable term $F(s_{t+1}, \pi_z(s_{t+1}), z)^\top z$ in the original FB loss is replaced by the well-constrained $V_{\pi_z}(s',z)$, yielding the corrected representation loss:
    $\mathcal{L}_{F\text{-reg}} = \mathbb{E}_{(s,a,s')\sim\mathcal{D}} \left[ (F(s,a,z)^\top z - B(s')^\top \mathbb{E}_D[BB^\top]^{-1}z - \gamma V_{\pi_z}(s',z))^2 \right]$

   **Design Motivation**: This avoids extrapolation errors from OOD actions while preserving the representational structure and optimality requirements.

2. **Task-Conditioned Diffusion Policy**

   Policy optimization is formulated as a KL-constrained behavior-regularized problem, with a closed-form solution:
    $\pi_z^*(s) \propto \mu(a|s) \exp(\alpha \cdot (F(s,a,z)^\top z - V_{\pi_z}(s,z)))$

   where temperature $\alpha$ controls conservatism. A diffusion model is used as the policy extractor, trained via a weighted regression objective:
    $\min_\theta \mathbb{E} \left[ \exp(\alpha \cdot (F(s,a,z)^\top z - V_{\pi_z}(s,z))) \| \epsilon - \epsilon_{\theta,z}(a_t, s, z, t) \|_2^2 \right]$

   At inference, rejection sampling is applied: $K$ candidate actions are generated and the one with the highest $Q$-value is selected. **Design Motivation**: Diffusion models effectively learn complex multimodal distributions, which is critical for arbitrary-task policy learning.

3. **Attention-Enhanced Representation Networks**

   - **Forward network $F$**: Two independent linear encoders separately encode (state, task) and (state, action) pairs, producing an embedding sequence of length 2, which is refined via a self-attention block enabling bidirectional feature interaction between task conditioning and behavioral patterns.
   - **Backward network $B$**: Stacked standard Transformer layers (with multi-head attention) serve as a global environment embedding, maintaining orthogonality and enforcing alignment with $F$.

   **Design Motivation**: Self-attention effectively captures complex relationships between task conditioning and dynamics; simply increasing MLP width proves ineffective.

### Loss & Training

The overall training objective comprises: (1) the original FB loss $\mathcal{L}_{FB}$ for successor measure learning; (2) the corrected forward loss $\mathcal{L}_{F\text{-reg}}$ for stable representation learning; (3) the expectile regression loss $\mathcal{L}_{V_{\pi_z}}$ for the state value function; and (4) the weighted regression loss for the diffusion policy. Default hyperparameters: $\tau=0.99$, $\alpha=0.05$.

## Key Experimental Results

### Main Results

IQM results on the ExORL benchmark (full dataset; 4 dataset types × 3 domains × 12 tasks = 48 tasks):

| Dataset | Domain | FB | MCFB | HILP | BREEZE | Max Gain |
|--------|------|------|------|------|--------|----------|
| RND | Walker | 661 | 659 | 665 | **693** | +4.2% |
| RND | Jaco | 32 | 41 | 52 | **84** | +61.5% |
| RND | Quadruped | 671 | 684 | 674 | **725** | +6.0% |
| APS | Jaco | 22 | 22 | 84 | **132** | +57.1% |
| PROTO | Quadruped | 222 | 219 | 216 | **389** | +75.2% |
| DIAYN | Jaco | 22 | 15 | 52 | **78** | +50.0% |

### Ablation Study

| Configuration | Walker-RND | Jaco-RND | Quadruped-RND | Note |
|------|-----------|----------|---------------|------|
| w/o FB Enhancement | 646 | 80 | 685 | Attention architecture removed |
| w/o Diffusion | 707 | 62 | 530 | Diffusion policy removed |
| BREEZE (Full) | **693** | **84** | **725** | Components are complementary |

In the low-data regime (100k transitions), BREEZE achieves 525 vs. FB's 264 on Walker-RND, demonstrating a substantial advantage.

### Key Findings

- BREEZE achieves optimal or near-optimal performance across almost all domains and datasets, with particularly large margins on manipulation tasks (Jaco) and low-quality datasets.
- Learning curves show that BREEZE converges faster with lower variance, indicating superior training stability.
- On long-horizon D4RL Kitchen tasks, the gap between BREEZE and vanilla FB is even more pronounced.
- Hyperparameter ablations show that $\tau$ yields a near-monotonic performance improvement, and $\alpha=0.05$ is optimal.

## Highlights & Insights

- The diagnostic analysis of FB methods is notably thorough: by visualizing the distributions of $M^{\pi_z}$ and $Q_z$, the fundamental failure modes of existing approaches are clearly revealed.
- Transferring IQL-style in-sample learning to zero-shot RL is a natural yet effective adaptation.
- The two-stage policy optimization combining diffusion policy generation with value-based selection elegantly balances conservatism and optimality.
- The attention architecture design—self-attention fusion for the forward network and stacked Transformers for the backward network—is validated through comprehensive ablations.

## Limitations & Future Work

- Evaluation is currently limited to state-based environments; validation on pixel-based (visual observation) settings is absent.
- Diffusion policy inference requires multi-step denoising plus rejection sampling, incurring non-trivial computational overhead.
- Whether approaches such as L2P or DPO could further improve performance remains an open direction.
- Training time is substantially increased compared to vanilla FB.

## Related Work & Insights

- The in-sample learning paradigm from IQL is successfully transferred to the zero-shot RL setting.
- Diffusion policies have demonstrated success in offline RL (e.g., Diffusion-QL); this work extends them to the more challenging zero-shot setting.
- The effectiveness of attention architectures in representation learning warrants broader exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of introducing behavior regularization into zero-shot RL is well-motivated and theoretically grounded; the diffusion policy and attention architecture represent reasonable but incremental innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 48 tasks, 4 dataset types, low-data regime, learning curves, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain from problem diagnosis to method design is clear and coherent.
- **Value**: ⭐⭐⭐⭐ Substantially advances the practical applicability of zero-shot RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Flex-Judge: Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators](flex-judge_text-only_reasoning_unleashes_zero-shot_multimodal_evaluators.md)
- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)

</div>

<!-- RELATED:END -->
