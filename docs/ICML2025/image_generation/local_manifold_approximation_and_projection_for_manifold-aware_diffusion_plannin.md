---
title: >-
  [Paper Note] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning
description: >-
  [ICML 2025][Image Generation][Diffusion Planning] Proposes LoMAP, a training-free correction method for diffusion planning. It projects guided samples onto a local low-rank subspace constructed from nearest neighbors in offline data at each reverse diffusion step to prevent the generation of infeasible trajectories, theoretically proving that the guidance error grows with dimensionality as $O(\sqrt{d})$.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Diffusion Planning"
  - "Manifold Off-drift"
  - "Low-Rank Projection"
  - "Offline RL"
  - "Trajectory Optimization"
date: 2026-05-08
content_hash: bba7bd65895e3950
---

# Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning

**Conference**: ICML 2025  
**arXiv**: [2506.00867](https://arxiv.org/abs/2506.00867)  
**Code**: [GitHub](https://github.com/leekwoon/lomap)  
**Area**: Diffusion Planning / Offline Reinforcement Learning  
**Keywords**: Diffusion Planning, Manifold Off-drift, Low-Rank Projection, Offline RL, Trajectory Optimization

## TL;DR
Proposes LoMAP, a training-free correction method for diffusion planning. It projects guided samples onto a local low-rank subspace constructed from nearest neighbors in offline data at each reverse diffusion step to prevent the generation of infeasible trajectories, theoretically proving that the guidance error grows with dimensionality as $O(\sqrt{d})$.

## Background & Motivation

**Background**: Diffusion models for trajectory planning (e.g., Diffuser) model the entire trajectory distribution to avoid error accumulation from step-by-step autoregressive generation, and can generate high-reward behaviors when combined with reward-guided sampling.  
**Limitations of Prior Work**: Reward-guided sampling suffers from a fundamental flaw: the MSE-trained guidance function $\mathcal{J}_\phi^{\text{MSE}}(\tau^i) = \mathbb{E}_{q(\tau^0|\tau^i)}[\mathcal{J}(\tau^0)]$ systematically underestimates the true guidance $\mathcal{J}_t(\tau^i) = \log\mathbb{E}_{q(\tau^0|\tau^i)}[e^{\mathcal{J}(\tau^0)}]$ (by Jensen's inequality). In high-dimensional, long-horizon tasks, the guidance error grows as $O(\sqrt{d})$, causing the sampled trajectories to drift away from the data manifold and generate infeasible paths.  
**Key Challenge**: Stronger reward guidance increases the likelihood of trajectories drifting from the manifold, whereas weaker guidance fails to achieve high rewards.  
**Goal**: Pull the generated trajectories back to the data manifold while maintaining reward guidance.  
**Key Insight**: Leveraging trajectories from offline datasets as the basis for local linear approximations of the manifold.  
**Core Idea**: Denoised estimation $\rightarrow$ retrieval of nearest neighbors $\rightarrow$ forward diffusion $\rightarrow$ PCA to obtain local subspace $\rightarrow$ projection.

## Method

### Overall Architecture
A LoMAP projection module is inserted after each reverse diffusion step of the standard Diffuser: first, reward-guided denoising is performed according to the normal procedure to obtain $\tau^{i-1}$, which is then projected onto a local low-rank subspace constructed from nearest neighbors in the offline data. The entire process requires no extra training.

### Key Designs

1. **Theoretical Lower Bound of Guidance Error (Proposition 3.2)**:
    - **Function**: Proves that the gap between MSE guidance and true guidance inevitably grows with dimensionality.
    - **Mechanism**: Uses Jensen's inequality to decompose the gap into the correlation between $\delta(\tau^0) = e^{\mathcal{J}(\tau^0)}/\mathbb{E}[e^{\mathcal{J}(\tau^0)}] - \mathcal{J}(\tau^0)$ and the forward noise $\epsilon$. In high dimensions, $\|\epsilon\|_2 \approx \sqrt{d}$, and when $\delta$ aligns with $\epsilon$, the lower bound of the guidance error is $c\sqrt{d}/\sqrt{1-\alpha_i}$.
    - **Design Motivation**: Theoretically demonstrates the necessity of correction—it is not due to insufficient model capacity, but rather an inherent limitation of the MSE objective itself.

2. **Local Manifold Approximation and Projection (Core of LoMAP)**:
    - **Function**: Constructs local linear approximations of the data manifold and projects onto them at each denoising step.
    - **Mechanism**: (1) Computes the denoised estimate $\hat{\tau}^{0|i-1}$ using Tweedie's formula; (2) Retrieves $k$ nearest-neighbor trajectories $\{\tau_{(n_j)}^0\}$ from the offline data; (3) Diffuses these neighbors forward to timestep $i{-}1$: $\tau_{(n_j)}^{i-1} = \sqrt{\alpha_{i-1}}\tau_{(n_j)}^0 + \sqrt{1-\alpha_{i-1}}\epsilon_{(n_j)}$; (4) Performs PCA on $\{\tau_{(n_j)}^{i-1}\}$ to obtain an orthogonal basis $U\in\mathbb{R}^{d\times r}$; (5) Projects $\tau^{i-1} \leftarrow UU^\top\tau^{i-1}$.
    - **Design Motivation**: Forward-diffused neighbors naturally lie near the manifold at timestep $i{-}1$ (under the low-dimensional manifold hypothesis), and PCA extracts the principal directions to filter out the drift components orthogonal to the manifold.

3. **Compatibility with Hierarchical Diffusion Planners (HD + LoMAP)**:
    - **Function**: Integrates LoMAP as a plug-and-play module into more complex planning architectures.
    - **Mechanism**: LoMAP only appends a projection step after the guidance update without modifying any preceding modules, allowing direct embedding into hierarchical planners such as Hierarchical Diffuser.
    - **Design Motivation**: Complex tasks like AntMaze require hierarchical decomposition (planning subgoals before low-level actions), and LoMAP can be applied independently at both levels.

### Loss & Training
- LoMAP itself is training-free, operating solely during inference.
- The underlying Diffuser training remains unchanged: MSE loss of the noise predictor $\epsilon_\theta$ is $\mathcal{L}(\theta) = \mathbb{E}_{i,\epsilon,\tau^0}[\|\epsilon - \epsilon_\theta(\tau^i)\|^2]$.
- The MSE loss of the guidance network $\mathcal{J}_\phi$ remains unchanged.
- The PCA variance retention ratio is $\lambda=0.99$, and the default number of neighbors is $k=5{-}10$.

## Key Experimental Results

### Main Results: Maze2D Single-Task Planning

| Environment | IQL | RGG | TAT | Diffuser | **Diffuser$^\mathcal{P}$** |
|------|-----|-----|-----|----------|---------------------------|
| U-Maze | 47.4 | 108.8 | 114.5 | 113.9 | **126.0±0.26** |
| Medium | 34.9 | 131.8 | 130.7 | 121.5 | **131.0±0.46** |
| Large | 58.6 | 135.4 | 133.4 | 123.0 | **151.9±2.66** |
| Average | 47.0 | 125.3 | 126.2 | 119.5 | **136.3** |

### Ablation Study: Infeasible Trajectory Ratio (Artifact Ratio)

| Sample Size | Diffuser | RGG | **Diffuser$^\mathcal{P}$** |
|---------|----------|-----|---------------------------|
| 100 (Medium) | ~15% | ~8% | **<1%** |
| 100 (Large) | ~30% | ~18% | **<3%** |
| 500 (Large) | ~50% | ~35% | **<5%** |

### Key Findings
- The improvement is most significant in Maze2D-Large (123.0 -> 151.9, +23.5%)—the more complex the environment, the more severe the manifold drift, and the larger the gain from LoMAP.
- The ratio of infeasible trajectories decreases from ~30% in Diffuser to <3% in the Large environment, verifying the effectiveness of manifold projection.
- While RGG reduces artifacts, it simultaneously decreases trajectory diversity (clustering into a few paths, as shown in Fig. 3); LoMAP maintains both high reliability and diversity.
- In multi-task (random goals) setups, IQL performance drops sharply (58.6 -> 24.8), whereas the diffusion planner + LoMAP maintains stability.
- The number of neighbors $k=5{-}10$ and the PCA variance retention ratio $\lambda=0.99$ demonstrate robust performance across most scenarios.
- Combination with the HD hierarchical planner can further boost performance on AntMaze.

## Highlights & Insights
- The proof of the $O(\sqrt{d})$ lower bound for guidance error is a key theoretical contribution. It demonstrates that the deviation of guidance is not due to poor model performance, but rather an inherent bias of the MSE objective in high dimensions (due to Jensen's inequality) which makes drift inevitable and increasingly severe as dimensionality grows.
- The "denoising -> retrieval -> forward diffusion -> PCA -> projection" workflow is simple and elegant. It incurs minimal computational overhead (since PCA is performed on a low-dimensional set of $k$ samples) and is entirely training-free, making it a true plug-and-play module.
- Guaranteeing that trajectories do not pass through walls or exceed boundaries in safety-critical scenarios (e.g., robotic planning) holds substantial practical value—a reduction in failure rate from >30% to <3% marks a quantitative leap.

## Limitations & Future Work
- Retrieving $k$ nearest neighbors and performing PCA at each step increases inference latency (though the overhead is relatively small compared to the multi-step denoising of Diffuser itself).
- The linear subspace assumption of PCA may be inadequate in highly non-linear regions of the manifold; kernel PCA or autoencoders could be considered.
- Effective subspaces cannot be constructed for regions outside the coverage of offline data—LoMAP does not assist in generalizing to unseen environments.
- The number of nearest neighbors $k$ and the PCA dimension $r$ require tuning.
- Integration with online RL or model-based methods has not yet been explored.

## Related Work & Insights
- **vs RGG (Lee et al., 2023b)**: RGG applies sample refinement using OOD detection metrics but relies on heavily tuned guidance steps and sacrifices diversity; LoMAP directly performs geometric corrections.
- **vs TAT (Feng et al., 2024)**: TAT also employs trajectory refinement strategies, but LoMAP's operation is more direct (projection vs. resampling).
- **vs MPGD (Chung et al., 2022)**: A similar manifold projection concept in the image domain. LoMAP adapts this to trajectory planning and constructs subspaces based on forward-diffused neighbors.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical lower bound of guidance error and the local low-rank projection idea are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-environment evaluation (Maze2D/Multi2D/AntMaze) + artifact analysis + visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ Tight logical flow from theoretical motivation to method design and experimental verification.
- Value: ⭐⭐⭐⭐⭐ Training-free and plug-and-play, holding direct practical value for the reliability of diffusion planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](../../NeurIPS2025/image_generation/what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[CVPR 2025\] Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI](../../CVPR2025/image_generation/derivative-free_diffusion_manifold-constrained_gradient_for_unified_xai.md)
- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](../../NeurIPS2025/image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ECCV 2024\] Scalable Group Choreography via Variational Phase Manifold Learning](../../ECCV2024/image_generation/scalable_group_choreography_via_variational_phase_manifold_learning.md)

</div>

<!-- RELATED:END -->
