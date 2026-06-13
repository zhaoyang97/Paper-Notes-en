---
title: >-
  [Paper Note] DiffusionNFT: Online Diffusion Reinforcement with Forward Process
description: >-
  [ICLR 2026][Image Generation][online RL] DiffusionNFT proposes a fundamentally new online RL paradigm for diffusion models: rather than performing policy optimization over the reverse sampling process (as in GRPO)…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "online RL"
  - "forward process"
  - "negative-aware finetuning"
  - "flow matching"
  - "CFG-free"
date: 2026-05-08
content_hash: c9b8b9d402687d9e
---

# DiffusionNFT: Online Diffusion Reinforcement with Forward Process

**Conference**: ICLR 2026
**arXiv**: [2509.16117](https://arxiv.org/abs/2509.16117)  
**Code**: [https://research.nvidia.com/labs/dir/DiffusionNFT](https://research.nvidia.com/labs/dir/DiffusionNFT)  
**Area**: Diffusion Models / Reinforcement Learning Alignment
**Keywords**: online RL, forward process, negative-aware finetuning, flow matching, CFG-free

## TL;DR
DiffusionNFT proposes a fundamentally new online RL paradigm for diffusion models: rather than performing policy optimization over the reverse sampling process (as in GRPO), it performs contrastive training on positive and negative samples via a flow matching objective over the *forward* process, defining an implicit policy improvement direction. The method is 3–25× faster than FlowGRPO and requires no CFG.

## Background & Motivation

**Background**: FlowGRPO and DanceGRPO discretize the reverse sampling process into an MDP and apply GRPO with SDE samplers to achieve online RL alignment of diffusion models, yielding notable results.

**Limitations of Prior Work**: GRPO-based methods suffer from three fundamental limitations: (a) **forward inconsistency**—optimizing only the reverse process may cause the model to degenerate into a cascade of Gaussians; (b) **solver constraints**—only first-order SDE samplers are supported, precluding more efficient ODE or higher-order solvers; (c) **CFG complexity**—simultaneous optimization of conditional and unconditional models is inefficient and engineering-heavy.

**Key Challenge**: Reverse-process RL requires likelihood estimation, yet the exact likelihood of diffusion models is intractable. Discretization approximations introduce systematic bias.

**Goal**: Can RL be performed on the **forward process** (via a flow matching objective), entirely avoiding likelihood estimation, solver constraints, and CFG dependence?

**Key Insight**: A diffusion policy has a unique forward process but multiple reverse processes (corresponding to different solvers). Optimization over the forward process is more fundamental—it directly defines the policy improvement direction via positive/negative sample contrast, embedded within the supervised learning framework of flow matching.

**Core Idea**: RL signals are converted into a contrastive flow matching objective over the forward process using positive and negative samples. Implicit parameterization integrates reinforcement guidance directly into a single policy model.

## Method

### Overall Architecture

Each iteration proceeds as follows:
1. **Data Collection**: Sample $K$ images from the current model using any solver; score each image with a reward function.
2. **Positive/Negative Partitioning**: Each image is assigned to the positive set $\mathcal{D}^+$ with probability $r$ and to the negative set $\mathcal{D}^-$ with probability $1-r$.
3. **Policy Optimization**: Train simultaneously on a positive branch (flow matching on $\mathcal{D}^+$) and a negative branch (flow matching on $\mathcal{D}^-$) over the forward process, extracting the improvement direction via implicit parameterization.
4. **Only clean images need to be stored**—no trajectory storage is required.

### Key Designs

1. **Improvement Direction Theorem (Theorem 3.1)**:

    - **Function**: Proves that the difference directions among the positive, negative, and old policy velocity fields are proportional.
    - **Mechanism**: $\Delta := \alpha(\mathbf{x}_t)[\mathbf{v}^+(\mathbf{x}_t) - \mathbf{v}^{\text{old}}(\mathbf{x}_t)] = [1-\alpha(\mathbf{x}_t)][\mathbf{v}^{\text{old}}(\mathbf{x}_t) - \mathbf{v}^-(\mathbf{x}_t)]$, where $\alpha$ is a scalar related to the density ratio of the positive policy.
    - **Design Motivation**: Establishes the equivalence "moving away from negatives = moving toward positives," a form analogous to CFG but derived from RL principles.

2. **Policy Optimization Objective (Theorem 3.2)**:

    - **Function**: Designs a flow matching loss that jointly exploits positive and negative data.
    - **Mechanism**: $\mathcal{L}(\theta) = \mathbb{E}[r \|\mathbf{v}_\theta^+ - \mathbf{v}\|^2 + (1-r)\|\mathbf{v}_\theta^- - \mathbf{v}\|^2]$, where $\mathbf{v}_\theta^+ = (1-\beta)\mathbf{v}^{\text{old}} + \beta \mathbf{v}_\theta$ (implicit positive policy) and $\mathbf{v}_\theta^- = (1+\beta)\mathbf{v}^{\text{old}} - \beta \mathbf{v}_\theta$ (implicit negative policy).
    - **Design Motivation**: Through implicit parameterization, only a single model $\mathbf{v}_\theta$ is trained, yet it is equivalently pulled toward the positive policy and pushed away from the negative policy. The optimal solution $\mathbf{v}_{\theta^*} = \mathbf{v}^{\text{old}} + \frac{2}{\beta}\Delta$ automatically integrates reinforcement guidance into the policy.

3. **Forward Consistency**:

    - **Function**: Guarantees that the trained model still corresponds to a valid forward process.
    - **Mechanism**: DiffusionNFT employs the standard flow matching loss (forward process) rather than policy gradients over the reverse SDE.
    - **Design Motivation**: FlowGRPO's exclusive optimization of the reverse process may break forward–reverse consistency.

4. **CFG-free Training**:

    - **Function**: Operates without CFG; reinforcement guidance subsumes the role of CFG.
    - **Mechanism**: The term $\Delta$ in Theorem 3.1 is formally equivalent to guidance—RL automatically learns the guidance direction.
    - **Design Motivation**: Avoids the complexity of jointly training conditional and unconditional models as required by GRPO.

### Loss & Training

- Based on SD3.5-Medium with rectified flow parameterization.
- Each iteration samples $K$ images, partitioned into positives and negatives by reward.
- $\beta$ controls guidance strength (analogous to CFG scale).
- Supports joint training with multiple reward models.

## Key Experimental Results

### Main Results (SD3.5-Medium, Single-Reward Head-to-Head vs. FlowGRPO)

| Task | DiffusionNFT | FlowGRPO | Efficiency Gain |
|------|-------------|----------|----------------|
| GenEval | 0.98 (1k steps) | 0.95 (5k steps) | **25×** |
| PickScore | Higher | — | 3–5× |
| Aesthetic | Higher | — | 3× |
| OCR | Higher | — | 5× |

**Multi-Reward Joint Training (SD3.5-Medium → SD3.5-Medium-NFT)**:
- GenEval: 0.63 → **0.98** (w/o CFG)
- DPG-Bench: 81.65 → **92.82**
- T2I-CompBench: 0.54 → **0.75**
- HPS v2.1: 29.95 → **32.52**

### Ablation Study

| Configuration | Outcome |
|--------------|---------|
| Positive samples only (RFT) | Rapid collapse |
| DiffusionNFT (full) | Stable improvement |
| Larger $\beta$ | More aggressive but may overfit |
| Different solvers (ODE / higher-order) | Fully compatible, no performance drop |

### Key Findings
- **Negative samples are critical**: Training on positive samples alone (RFT) leads to mode collapse; incorporating negative samples stabilizes training.
- DiffusionNFT, entirely CFG-free, rapidly improves from a very low starting point (GenEval 0.24 w/o CFG) to 0.98, surpassing FlowGRPO + CFG at 0.95.
- Any solver (ODE / higher-order) can be used, and no trajectory storage is required, yielding significantly higher training efficiency.
- Generalization gains are observed on out-of-distribution rewards as well.

## Highlights & Insights
- The conceptual shift from **reverse-process RL to forward-process RL** is the central contribution. The forward process of a diffusion model is uniquely determined whereas the reverse process depends on solver choice; performing RL over the forward process is more fundamental and circumvents the difficulty of likelihood estimation.
- The **implicit parameterization** is particularly elegant: by defining $\mathbf{v}_\theta^+ = (1-\beta)\mathbf{v}^{\text{old}} + \beta \mathbf{v}_\theta$, training a single model is equivalent to simultaneously approaching positive and retreating from negative directions—far more efficient than explicitly training a guidance model.
- The **NFT vs. GRPO** analogy mirrors **DPO vs. PPO** in LLMs: RL is reformulated within a supervised learning framework, substantially simplifying engineering.

## Limitations & Future Work
- The choice of $\beta$ requires tuning; excessively large values lead to reward overfitting.
- Positive/negative partitioning based on sampling probability rather than hard thresholds may introduce noise.
- Evaluation is conducted solely on SD3.5-Medium; generalization to other architectures (SDXL / Flux / DiT) has not been verified.
- Theoretical analysis assumes infinite data and model capacity; approximation errors in practice are not quantified.
- Reward weighting in multi-reward joint training has not been systematically studied.

## Related Work & Insights
- **vs. FlowGRPO**: Fundamentally different—forward-process RL vs. reverse-process RL. DiffusionNFT is 3–25× faster, requires no SDE sampler, and eliminates CFG.
- **vs. DPO / DRaFT**: DiffusionNFT is an online (on-policy) method, avoiding the distributional shift inherent to offline approaches.
- **vs. LLM NFT (Chen et al., 2025c)**: Transfers the NFT paradigm from language models to diffusion models, adapting it to leverage the properties of flow matching.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Forward-process RL is an entirely new paradigm; implicit parameterization elegantly unifies positive and negative data training.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Head-to-head comparison with FlowGRPO is clear and rigorous; multi-reward joint training evaluation is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and clear; the analogy with CFG provides strong intuition; figures and tables are well designed.
- Value: ⭐⭐⭐⭐⭐ Addresses multiple fundamental problems in diffusion RL (solver constraints, CFG dependence, efficiency); has strong potential to become the new standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](../../CVPR2026/image_generation/oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)
- [\[AAAI 2026\] ORVIT: Near-Optimal Online Distributionally Robust Reinforcement Learning](../../AAAI2026/image_generation/orvit_near-optimal_online_distributionally_robust_reinforcement_learning.md)
- [\[ICLR 2026\] EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)
- [\[ICLR 2026\] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)

</div>

<!-- RELATED:END -->
