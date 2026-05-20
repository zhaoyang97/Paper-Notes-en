---
title: >-
  [Paper Note] State-Covering Trajectory Stitching for Diffusion Planners
description: >-
  [NeurIPS 2025][Image Generation][Diffusion planner] This paper proposes SCoTS (State-Covering Trajectory Stitching), a reward-free trajectory augmentation framework that iteratively stitches short trajectory segments in…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion planner"
  - "trajectory stitching"
  - "state coverage"
  - "offline reinforcement learning"
  - "data augmentation"
date: 2026-05-08
content_hash: 95855bcb5568d585
---

# State-Covering Trajectory Stitching for Diffusion Planners

**Conference**: NeurIPS 2025
**arXiv**: [2506.00895](https://arxiv.org/abs/2506.00895)  
**Code**: [GitHub](https://github.com/leekwoon/scots/)  
**Area**: Diffusion Models / Trajectory Planning
**Keywords**: Diffusion planner, trajectory stitching, state coverage, offline reinforcement learning, data augmentation

## TL;DR

This paper proposes SCoTS (State-Covering Trajectory Stitching), a reward-free trajectory augmentation framework that iteratively stitches short trajectory segments in a temporal distance-preserving latent space to systematically expand state-space coverage, significantly improving the generalization of diffusion planners on long-horizon and out-of-distribution tasks.

## Background & Motivation

Diffusion models have demonstrated strong potential as trajectory generation tools in offline reinforcement learning—treating entire trajectories as high-dimensional samples for denoising generation, which naturally avoids the error accumulation of autoregressive models. However, the performance of diffusion planners is fundamentally constrained by the quality, diversity, and coverage of training data:

**Limited planning horizon**: Effective planning length is coupled with the maximum length of training trajectories, making it difficult to generate long-horizon plans far beyond the training distribution.

**Insufficient generalization**: When datasets predominantly contain specific motion patterns, planners struggle to synthesize solutions for new tasks requiring combinations of different behaviors.

**Expensive data collection**: Exhaustively collecting data across all scenarios is impractical.

Existing trajectory stitching methods rely on external rewards for segment selection and offer limited guarantees on the dynamic consistency and feasibility of stitched trajectories. This motivates the authors to design a **reward-free, state-coverage-driven** trajectory augmentation scheme.

## Method

### Overall Architecture

SCoTS follows a three-stage pipeline: (1) learning a temporal distance-preserving latent representation; (2) an iterative stitching strategy driven by directional exploration and novelty; and (3) diffusion model-based refinement at stitching points. The entire process starts from an offline dataset $\mathcal{D}$, produces an augmented dataset $\mathcal{D}_{\text{aug}}$, and trains a diffusion planner on the augmented data.

### Key Designs

1. **Temporal Distance-Preserving Embedding**

   The core objective is to map raw states into a latent space $\mathcal{Z}$ such that Euclidean distances approximate optimal temporal distances. A goal-conditioned value function is defined as:

    $V(\boldsymbol{s}, \boldsymbol{g}) \coloneqq -\|\phi(\boldsymbol{s}) - \phi(\boldsymbol{g})\|_2$

   Training uses an IQL-inspired temporal difference objective:

    $\mathcal{L}_\phi \coloneqq \mathbb{E}_{(\boldsymbol{s},\boldsymbol{a},\boldsymbol{s}',\boldsymbol{g})\sim\mathcal{D}}\left[\ell_\xi^2\left(-\mathbb{1}(\boldsymbol{s}\neq\boldsymbol{g}) - \gamma\|\bar{\phi}(\boldsymbol{s}')-\bar{\phi}(\boldsymbol{g})\|_2 + \|\phi(\boldsymbol{s})-\phi(\boldsymbol{g})\|_2\right)\right]$

   **Design Motivation**: Using raw state-space distances ignores dynamic reachability and leads to temporally incoherent stitching. Although the learned latent space is not a perfect metric, it is sufficiently reliable for the **local** retrieval of reachable candidate segments.

2. **Directional Exploration and Novelty-Driven Iterative Stitching**

   Each new trajectory samples a random initial segment and a fixed latent-space exploration direction $\boldsymbol{z}$ (a unit vector). At each stitching iteration:

    - Top-K nearest neighbors are retrieved as candidate segments.
    - A **directional progress score** $P_j = \langle \phi(\text{end}(\boldsymbol{\tau}_j)) - \phi(\boldsymbol{s}_{1,j}), \boldsymbol{z} \rangle$ is computed.
    - A **novelty score** $N_j$ is computed (based on a particle estimator measuring the entropy of endpoints relative to previously visited states).
    - Candidates are ranked by the combined score: $S_j = P_j + \beta N_j$.

   **Design Motivation**: Pure directional guidance yields limited coverage ($\beta=0$), while excessive novelty weight loses directional discrimination ($\beta=20$); $\beta=2$ achieves the optimal balance.

3. **Diffusion Model-Based Stitching Refinement (Diffusion Stitcher)**

   A conditional diffusion model $p_\theta^{\text{stitcher}}$ is trained, taking the terminal state of the current trajectory and the terminal state of the best candidate segment as boundary conditions to generate intermediate transition states:

    $\boldsymbol{\tau}' \sim p_\theta^{\text{stitcher}}(\cdot \mid \boldsymbol{s}_1 = \text{end}(\boldsymbol{\tau}_{\text{comp}}), \boldsymbol{s}_H = \text{end}(\boldsymbol{\tau}_{\text{best}}))$

   **Design Motivation**: Minor dynamic inconsistencies may exist at stitching points; the diffusion model smooths transitions and ensures dynamic feasibility.

### Loss & Training

- The embedding network is trained with an expectile regression loss.
- The diffusion stitcher uses the standard diffusion training objective $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{\tau}^i, i)\|^2$.
- Action sequences are inferred by an inverse dynamics model $\boldsymbol{a}_t = f_\psi(\boldsymbol{s}_t, \boldsymbol{s}_{t+1})$.
- Each dataset is upsampled to 5M samples.

## Key Experimental Results

### Main Results

Evaluated on the OGBench benchmark, covering Stitch and Explore datasets in PointMaze and AntMaze environments.

| Environment | Dataset Type | Scale | GCIQL | QRL | HIQL | GSC | CD | HD | SCoTS |
|---|---|---|---|---|---|---|---|---|---|
| PointMaze-Stitch | Giant | - | 0 | 50 | 0 | 29 | 68 | 0 | **100** |
| AntMaze-Stitch | Giant | - | 0 | 0 | 2 | 20 | 65 | 0 | **87** |
| AntMaze-Explore | Large | - | 0 | 0 | 4 | 21 | 27 | 13 | **98** |
| Average (all tasks) | - | - | 12.6 | 36.5 | 36.4 | 65.3 | 77.9 | 25.4 | **96.8** |

### Ablation Study / Offline GCRL Augmentation

| Algorithm | Data Source | PointMaze-Giant-Stitch | AntMaze-Giant-Stitch | AntMaze-Large-Explore |
|---|---|---|---|---|
| HIQL | Original | 0 | 2 | 4 |
| HIQL | SynthER | 0 | 0 | 12 |
| HIQL | **SCoTS** | **27** | **55** | **77** |
| CRL | Original | 0 | 0 | 0 |
| CRL | SynthER | 0 | 0 | 2 |
| CRL | **SCoTS** | **18** | **2** | **19** |

### Key Findings

1. SCoTS achieves near-optimal success rates across all tasks, with the most pronounced advantage in the largest-scale Giant environments.
2. The novelty weight $\beta=2$ is the optimal balance point, reconciling directional exploration and state coverage.
3. The diffusion refinement step significantly reduces Dynamic MSE at stitching points, ensuring dynamic consistency.
4. SCoTS-augmented data yields substantial improvements for conventional offline GCRL algorithms (GCIQL, CRL, HIQL), demonstrating that trajectory-level augmentation outperforms transition-level augmentation (SynthER).
5. SCoTS is insensitive to the horizon length of the low-level controller, and the generated subgoals exhibit high feasibility.

## Highlights & Insights

- A **reward-free** trajectory augmentation paradigm driven solely by state coverage, with strong generality.
- Using latent-space embeddings for *local* stitching retrieval rather than global metric learning elegantly circumvents the difficulty of learning a perfect metric.
- The dual scoring mechanism of directional exploration and novelty is an elegant design that achieves a balance between coverage and diversity.
- The end-to-end augmentation pipeline (embedding → stitching → refinement → inverse dynamics) forms a complete closed loop.

## Limitations & Future Work

- Validation is primarily conducted in Maze-type navigation environments; extension to high-dimensional continuous control or robotic manipulation tasks remains unexplored.
- Embedding quality depends on the distribution of offline data and may degrade under extremely sparse data conditions.
- Computational overhead is non-trivial, requiring pre-training of the embedding network, diffusion stitcher, and inverse dynamics model.
- Fixed exploration directions may be suboptimal in non-isotropic state spaces.

## Related Work & Insights

- Data augmentation for diffusion planners (Diffuser, HD) is a neglected yet critical direction.
- Temporal distance embeddings can be transferred to other scenarios requiring reachability estimation.
- The concept of trajectory-level data augmentation can be extended to imitation learning and world model training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ A reward-free trajectory augmentation framework; the directional + novelty dual scoring is a novel design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple environments, dataset types, baselines, and rich ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with intuitive figures.
- **Value**: ⭐⭐⭐⭐ Provides a systematic solution to the data bottleneck of diffusion planners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory](hierarchical_koopman_diffusion_fast_generation_with_interpretable_diffusion_traj.md)
- [\[ICCV 2025\] Long-Context State-Space Video World Models](../../ICCV2025/image_generation/long-context_state-space_video_world_models.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](../../ICCV2025/image_generation/learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)
- [\[ICCV 2025\] AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction](../../ICCV2025/image_generation/animegamer_infinite_anime_life_simulation_with_next_game_state_prediction.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](../../ICLR2026/image_generation/offline_reinforcement_learning_with_generative_trajectory_policies.md)

</div>

<!-- RELATED:END -->
