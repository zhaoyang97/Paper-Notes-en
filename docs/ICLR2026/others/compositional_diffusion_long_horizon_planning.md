---
title: >-
  [Paper Note] Compositional Diffusion with Guided Search for Long-Horizon Planning
description: >-
  [ICLR 2026][compositional diffusion] This paper proposes CDGS (Compositional Diffusion with Guided Search), which embeds a population-based search mechanism—iterative resampling combined with likelihood-based pruning—int…
tags:
  - "ICLR 2026"
  - "compositional diffusion"
  - "long-horizon planning"
  - "mode averaging"
  - "guided search"
  - "inference-time compute"
date: 2026-05-08
content_hash: 13342ae37d9c7d8e
---

# Compositional Diffusion with Guided Search for Long-Horizon Planning

**Conference**: ICLR 2026
**arXiv**: [2601.00126](https://arxiv.org/abs/2601.00126)
**Code**: [cdgsearch.github.io](https://cdgsearch.github.io/)
**Area**: Other
**Keywords**: compositional diffusion, long-horizon planning, mode averaging, guided search, inference-time compute

## TL;DR

This paper proposes CDGS (Compositional Diffusion with Guided Search), which embeds a population-based search mechanism—iterative resampling combined with likelihood-based pruning—into the diffusion denoising process to address the mode averaging problem arising from the composition of multimodal local distributions. CDGS enables sampling of globally consistent long-horizon plans from short-horizon models without long-horizon training data.

## Background & Motivation

**Background**: Diffusion models have emerged as powerful tools for planning. Compositional approaches model long-horizon task distributions by combining local short-horizon generative models, with applications in multi-step robotic manipulation, panoramic image stitching, and long video generation.

**Limitations of Prior Work**: When local distributions are multimodal (e.g., a robot choosing among combinations of objects and actions), existing compositional sampling methods such as score averaging suffer from mode averaging—blending incompatible modes and producing plans that are neither locally feasible nor globally consistent.

**Key Challenge**: The search space for global planning grows exponentially with planning horizon, while existing inference-time scaling methods are designed for single-distribution sampling and cannot handle compositional reasoning over chains of distributions.

**Core Idea**: Search is embedded into the denoising process via (1) iterative resampling to promote long-range information propagation and construct globally consistent candidate plans, and (2) likelihood-based pruning to eliminate candidates containing locally inconsistent segments.

## Method

### Overall Architecture

CDGS decomposes the global plan $\tau = (x_1, \ldots, x_N)$ using a factor graph representation as a product of overlapping local distributions:

$$p(\tau) = \frac{\prod_{j=1}^M p(y_j)}{\prod_{i=1}^N p(x_i)^{d_i - 1}}$$

where $y_j$ denotes local factors corresponding to subsequences of adjacent variables. Under the diffusion framework, global sampling is realized via a composed score function:

$$\nabla \log p(\tau) = \sum_{j=1}^M \nabla \log p(y_j) + \sum_{i=1}^N (1 - d_i) \nabla \log p(x_i)$$

### Key Design 1: DDIM Inversion-Based Global Plan Ranking

DDIM inversion is used to approximately evaluate the likelihood of local segments, defining a smoothness metric:

$$g(y^{(0)}) = \sum_{i=1}^T \left\| \frac{\partial \epsilon_\theta(y^{(i-1)}, i)}{\partial i} \right\|_2$$

The global ranking objective is $J(\tau^{(0)}) = \prod_{m=1}^M \exp(-g(y_m^{(0)}))$, where high $g$ values indicate low-likelihood segments corresponding to inconsistent plans, which are subsequently pruned.

### Key Design 2: Iterative Resampling

At each denoising step, forward noising $\tau^{(t)} \sim p(\tau^{(t)} | \tau^{(t-1)})$ and denoising are alternately applied and repeated $U$ times. This process is analogous to belief propagation on a chain factor graph, enabling information from distal factors to gradually propagate through shared variables and thereby promoting global consistency.

### Loss & Training

- Local planning diffusion models (Diffuser) are trained on trajectories of approximately 4 seconds at 20 Hz.
- At inference time, models are composed into global plans of up to 10 seconds.
- Population size $B$ and the number of elites $K$ are adjustable, enabling adaptive inference-time computation.

## Key Experimental Results

### Main Results: OGBench Maze and Scene Tasks (Success Rate %)

| Environment | GCBC | HIQL | Diffuser | GSC | CD | **CDGS** |
|---|---|---|---|---|---|---|
| PointMaze-Giant | 0 | 0 | - | 29 | 68 | **82** |
| AntMaze-Giant | 0 | 2 | - | 20 | 65 | **84** |
| Scene-play (avg) | 5 | 38 | 6 | 8 | - | **51** |

### TAMP Mixed Planning Tasks (Success Rate)

| Task | Random CEM | STAP CEM | LLM-T2M | GSC (oracle) | **CDGS** |
|---|---|---|---|---|---|
| Hook Reach T1 | 0.14 | 0.66 | 0.0 | 0.78 | **0.64** |
| Rearrange Push T1 | 0.08 | 0.76 | 0.72 | 0.88 | **0.84** |
| Rearrange Memory T1 | 0.02 | 0.00 | 0.0 | 0.82 | **0.42** |

### Panoramic Image Generation (512×4608)

| Metric | Multi-Diffusion | Sync-Diffusion | **CDGS** |
|---|---|---|---|
| Intra-LPIPS↓ | 0.72 | **0.58** | **0.59** |
| Intra-Style-L↓ | 2.96 | **1.39** | **1.38** |
| Mean-CLIP-S↑ | 31.77 | 31.77 | **32.51** |

### Key Findings

- Without requiring long-horizon training data, CDGS matches inverse RL baselines and outperforms all generative baselines on OGBench.
- On TAMP tasks, feasible plans are discovered without task skeletons or PDDL; CDGS substantially outperforms prior-free methods on Rearrangement Memory.
- Inference-time computation scales favorably: increasing population size $B$ and resampling steps $U$ consistently improves success rates.

## Highlights & Insights

1. **Elegant problem formulation**: Long-horizon planning is unified as compositional sampling on a factor graph, applicable across domains (robotics / images / video).
2. **Training-free inference enhancement**: Without any additional training, inference-time search elevates naive compositional sampling to a level competitive with trained methods such as CompDiffuser.
3. **Adaptive inference computation**: Harder problems can be addressed by increasing $B$ and $U$, demonstrating the potential of inference-time scaling.
4. **DDIM inversion as a likelihood proxy**: Denoising trajectory curvature is leveraged to approximate sample likelihood, avoiding the high cost of exact likelihood computation.

## Limitations & Future Work

- A goal state must be specified in advance; the method cannot handle open-ended tasks with unknown goals.
- Planning horizon is fixed, though this can be partially mitigated by attempting multiple horizon lengths.
- Long-range dependencies are propagated only through score averaging and resampling; more advanced message-passing or attention mechanisms may improve efficiency.
- Inference overhead scales linearly with $B \times U$, which may be prohibitive in real-time applications.

## Related Work & Insights

- CDGS is complementary to compositional diffusion methods such as CompDiffuser and GSC: it specifically targets mode averaging without relying on additional training.
- The work echoes the inference-time scaling literature; embedding search into denoising is a natural extension of this paradigm to compositional generation.
- The factor graph + diffusion model framework is generalizable to other structured generation problems, such as fragment-based molecular design or protein folding.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of embedding search into the compositional diffusion denoising process is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across three domains (robotics / images / video).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear running examples and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Provides a general solution for long-horizon generation, though inference overhead may limit practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[AAAI 2026\] Controllable Financial Market Generation with Diffusion Guided Meta Agent](../../AAAI2026/others/controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)
- [\[ICLR 2026\] Hilbert-Guided Sparse Local Attention](hilbert-guided_sparse_local_attention.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](../../NeurIPS2025/others/polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)

</div>

<!-- RELATED:END -->
