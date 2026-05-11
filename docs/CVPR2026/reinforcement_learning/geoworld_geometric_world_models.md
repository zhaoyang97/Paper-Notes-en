---
title: >-
  [Paper Note] GeoWorld: Geometric World Models
description: >-
  [CVPR 2026][Reinforcement Learning][Geometric World Models] GeoWorld maps the latent representations of predictive world models from Euclidean space onto a hyperbolic manifold…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "Geometric World Models"
  - "Hyperbolic Space"
  - "JEPA"
  - "Long-Horizon Planning"
date: 2026-05-08
content_hash: a7531000be8edbf8
---

# GeoWorld: Geometric World Models

**Conference**: CVPR 2026
**arXiv**: [2602.23058](https://arxiv.org/abs/2602.23058)
**Code**: [https://steve-zeyu-zhang.github.io/GeoWorld](https://steve-zeyu-zhang.github.io/GeoWorld)
**Area**: Reinforcement Learning
**Keywords**: Geometric World Models, Hyperbolic Space, JEPA, Reinforcement Learning, Long-Horizon Planning

## TL;DR
GeoWorld maps the latent representations of predictive world models from Euclidean space onto a hyperbolic manifold, preserving geometric structure and hierarchical relationships via Hyperbolic JEPA, and proposes Geometric Reinforcement Learning to optimize multi-step planning. The method achieves approximately 3% SR (T=3) and 2% SR (T=4) gains on CrossTask and COIN.

## Background & Motivation

**Background**: World models fall into two categories: generative and predictive. Generative world models (e.g., VideoWorld) explicitly generate pixels or visual tokens to predict the next state, but lack global awareness of trajectory structure and energy landscapes. Predictive world models (e.g., JEPA, V-JEPA 2) forgo pixel generation and instead learn an energy landscape in latent space to measure compatibility between the current and goal states, supporting multi-step hierarchical planning.

**Limitations of Prior Work**: Existing predictive world models face two critical issues: (1) *Geometric neglect*—latent representations are learned in Euclidean space, which cannot preserve the intrinsic geometric structure and hierarchical relationships among states, causing the learned energy landscape to fail at capturing meaningful geodesic distances; (2) *Multi-step degradation*—multi-step video data is scarce and costly, so models are primarily trained on single-step transitions; although conceptually capable of long-horizon planning, performance degrades rapidly as the horizon grows.

**Key Challenge**: The "flat" structure of Euclidean space cannot naturally encode the hierarchical relationships among real-world states (e.g., "slice vegetables" is a sub-step of "cook a meal"), and forcing long-range dependencies into Euclidean space leads to geometric drift.

**Goal**: (1) How can geometric and hierarchical structure be preserved in latent space? (2) How can multi-step planning stability be improved under limited training data?

**Key Insight**: The tree-like structure of hyperbolic space is naturally suited to encoding hierarchical relationships (distances grow exponentially with depth), and hyperbolic geodesics provide the most natural notion of "shortest path." Combining this with reinforcement learning to optimize the predictor's energy function steers trajectories along geodesics.

**Core Idea**: Lift the latent dynamics of JEPA from Euclidean space onto a hyperbolic manifold, and apply Geometric Reinforcement Learning to optimize geodesic consistency in multi-step planning.

## Method

### Overall Architecture
GeoWorld takes a current video observation and a goal video as input and outputs an action sequence to reach the goal. The system comprises three components: (1) *Hyperbolic JEPA (H-JEPA)*—a frozen encoder maps observations to Euclidean space, followed by an exponential map that projects them onto the Poincaré ball, after which a conditional predictor performs next-state prediction in hyperbolic space; (2) *Geometric Reinforcement Learning (GRL)*—multi-step planning is formulated as the minimization of hyperbolic energy, with triangle-inequality regularization enforcing geodesic consistency of the trajectory; (3) *Energy-Based Planning*—the Cross-Entropy Method (CEM) is used to search for the optimal action sequence that minimizes hyperbolic distance.

### Key Designs

1. **Hyperbolic JEPA (H-JEPA)**:

    - **Function**: Latent dynamics modeling that preserves geometric structure and hierarchical relationships on a hyperbolic manifold.
    - **Mechanism**: A pretrained frozen encoder $E_\theta$ encodes observation $x_t$ into a Euclidean embedding $s_t^x \in \mathbb{R}^n$, which is treated as a tangent vector at the origin and projected onto hyperbolic space via the Poincaré ball exponential map: $s_{t,\mathbb{H}}^x = \exp_0(s_t^x) = \tanh(\sqrt{c}\|s_t^x\|) \frac{s_t^x}{\sqrt{c}\|s_t^x\|}$, where the curvature $c$ is a learnable parameter. A conditional predictor $P_\phi$ receives the hyperbolic state sequence and action sequence to predict the next state. The training objective includes a teacher-forcing loss (single-step prediction accuracy) and a rollout loss (consistency of two-step recursive prediction), both defined over the hyperbolic geodesic distance $d_\mathbb{H}$.
    - **Design Motivation**: The volume of hyperbolic space grows exponentially with radius, making it naturally suited to embedding tree-like hierarchical structures. Geodesic distances between states more faithfully reflect the semantic hierarchical relationships among actions than Euclidean distances.

2. **Geometric Reinforcement Learning (GRL)**:

    - **Function**: Directly improves the predictor's multi-step planning capability through energy optimization.
    - **Mechanism**: The energy cost is defined as the hyperbolic distance between the predicted and target states, $c_t = d_\mathbb{H}(\hat{s}_{t+1,\mathbb{H}}^x, s_{t+1,\mathbb{H}}^x)$, and the reward is the negative energy cost. The path value function is the expected cumulative reward $V = \mathbb{E}[\sum \gamma^{t-1} r_t]$, and the optimal value function is equivalent to minimizing the cumulative hyperbolic distance. The key innovation is a triangle-inequality regularization term $\mathcal{L}_\Delta = \frac{1}{T-2}\sum[d_\mathbb{H}(\hat{s}_t, \hat{s}_{t+2}) - d_\mathbb{H}(\hat{s}_t, \hat{s}_{t+1}) - d_\mathbb{H}(\hat{s}_{t+1}, \hat{s}_{t+2})]_+$, which enforces geodesic properties on the predicted trajectory. The total loss is $\mathcal{L}_{\text{GRL}} = \text{cumulative geodesic distance} + \beta \mathcal{L}_\Delta$.
    - **Design Motivation**: Supervised learning alone is insufficient to improve multi-step planning, given the scarcity of multi-step video data. The RL objective directly targets final planning quality. The elegance of GRL lies in optimizing predictor parameters directly without requiring a separate policy network or reward model.

3. **Energy-Based Planning with CEM**:

    - **Function**: Searches for the optimal action sequence at inference time.
    - **Mechanism**: Given the current observation and goal, both are encoded into hyperbolic space, and the energy cost function $C = d_\mathbb{H}(P((\hat{a}_t); s_{1,\mathbb{H}}^x), s_{1+T,\mathbb{H}}^x)$ is defined. CEM is applied ($N=800$ samples, $K=80$ elites, $I=10$ iterations) to search for the action sequence that minimizes energy.
    - **Design Motivation**: CEM is an efficient zeroth-order optimization method well suited for searching in high-dimensional action spaces.

### Loss & Training
Training proceeds in two stages: (1) *Supervised fine-tuning stage*: $\mathcal{L}_{\text{SFT}} = \lambda \mathcal{L}_{\text{TF}} + (1-\lambda) \mathcal{L}_{\text{rollout}}$, with the AdamW optimizer, a warmup → constant → decay learning rate schedule, batch size 256, and approximately 94,500 iterations. (2) *GRL stage*: smaller learning rate and shorter schedule, batch size 128, approximately 25,000 iterations, $\gamma=0.99$, $\beta=0.1$. The predictor is a ~300M parameter Transformer (24 layers, 16 heads, 1024-dimensional). Training is conducted on 4 nodes with 32 H100 GPUs.

## Key Experimental Results

### Main Results — Procedural Planning (Image Input)

| Method | CrossTask T=3 SR | CrossTask T=4 SR | COIN T=3 SR | COIN T=4 SR |
|------|-----------------|-----------------|-------------|-------------|
| V-JEPA 2 ViT-g384 | 45.58 | 31.36 | 34.08 | 23.43 |
| **GeoWorld ViT-g384** | **47.47** | **31.48** | **34.85** | **27.79** |
| SCHEMA (LLM) | 38.93 | 24.50 | 32.09 | 22.02 |
| MTID (Generative) | 40.45 | 24.76 | 30.44 | 22.74 |

### Main Results — Visual Planning with Videos

| Method | CrossTask T=3 SR | CrossTask T=4 SR | COIN T=3 SR | COIN T=4 SR |
|------|-----------------|-----------------|-------------|-------------|
| V-JEPA 2 ViT-g384 | 50.16 | 35.01 | 42.74 | 31.63 |
| **GeoWorld ViT-g384** | **51.71** | **37.04** | **45.29** | **33.29** |
| GPT-5 | 50.03 | 30.20 | 43.84 | 32.64 |
| VideoWorld | 41.59 | 25.50 | 34.88 | 23.74 |

### Long-Horizon Planning (CrossTask)

| Method | T=3 | T=4 | T=5 | T=6 |
|------|-----|-----|-----|-----|
| V-JEPA 2 ViT-g384 | 50.16 | 35.01 | 23.17 | 16.88 |
| **GeoWorld ViT-g384** | **51.71** | **37.04** | **24.83** | **18.26** |

### Key Findings
- GeoWorld consistently outperforms V-JEPA 2 across all model scales (ViT-L/H/g/g384), indicating that the improvements are not scale-specific.
- As the planning horizon increases (T=3→T=6), GeoWorld's advantage over V-JEPA 2 gradually widens, achieving a 1.38% SR gain at T=6 (18.26 vs. 16.88), validating the enhanced long-horizon planning stability.
- The mIoU metric shows the most pronounced improvement (e.g., PP setting, CrossTask T=3: 86.55 vs. 69.42), suggesting that hyperbolic representations contribute most significantly to trajectory overlap.
- GeoWorld ViT-g384 surpasses GPT-5 and Gemini 2.5 Pro in the visual planning setting.

## Highlights & Insights
- Introducing hyperbolic geometry into world models is a natural and elegant design choice. The hierarchical structure of everyday activities (e.g., "make coffee" → "grind beans → boil water → brew") is inherently tree-like, and hyperbolic space is precisely the optimal space for embedding trees. This geometric inductive bias is more parameter-efficient than simply scaling up models.
- The design of GRL is particularly clever in using geodesic distance directly as the reward, without requiring a separate reward model. The triangle-inequality regularization enforces the geometric consistency that "traveling two steps should not exceed the sum of traveling each step individually"—a constraint that is difficult to impose in Euclidean space.
- The two-stage training paradigm (SFT → GRL) mirrors the SFT → RLHF pipeline in LLMs, but the objective function is entirely grounded in geometric principles rather than human preferences.

## Limitations & Future Work
- Validation is limited to CrossTask and COIN, both of which have relatively simple action spaces (everyday activities). Performance in more complex robotic manipulation or game environments remains unexplored.
- The curvature $c$ of the Poincaré ball is shared across all states; this may be insufficiently flexible, as different hierarchical subspaces may require different curvatures.
- The encoder is fully frozen; H-JEPA only trains the predictor. If the encoder's representations are not well-suited to hyperbolic space, the performance ceiling is limited accordingly.
- Absolute SR values remain low at long horizons (only 18.26% at T=6), indicating that long-horizon planning remains a fundamental challenge.

## Related Work & Insights
- **vs. V-JEPA 2**: Both are predictive world models; V-JEPA 2 learns energy landscapes in Euclidean space. GeoWorld achieves consistent improvements over the same backbone and data via hyperbolic mapping and GRL. Both use the same frozen encoder, enabling a fair comparison.
- **vs. VideoWorld**: A generative world model that requires pixel decoding; it performs substantially worse than GeoWorld in the planning setting.
- **vs. GPT-5/Gemini 2.5 Pro**: Strong LLM baselines are competitive in zero-shot settings, but GeoWorld ViT-g384 surpasses them in the visual planning setting, demonstrating the continued advantage of specialized geometric models.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of hyperbolic space, world models, and geometric RL is novel in the visual planning domain; both the theoretical motivation and technical implementation are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, multiple baselines, and multiple model scales are evaluated; however, only two datasets are used and ablation details are limited.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous, though the dense notation presents a non-trivial reading barrier.
- **Value**: ⭐⭐⭐⭐ The paper introduces a promising direction for incorporating geometric thinking into world models and surpasses GPT-5-level baselines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](../../ICLR2026/reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)
- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](../../NeurIPS2025/reinforcement_learning/foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](../../AAAI2026/reinforcement_learning/object-centric_world_models_for_causality-aware_reinforcement_learning.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)

</div>

<!-- RELATED:END -->
