---
title: >-
  [Paper Note] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache
description: >-
  [CVPR 2026][Image Generation][diffusion model acceleration] This paper formalizes diffusion model sampling acceleration as a global path planning problem. By constructing a Path-Aware Cost Tensor (PACT) and applying dynamic programming to select the optimal sequence of key timesteps, the method achieves training-free 4.87× acceleration while surpassing the full-step baseline in generation quality.
tags:
  - CVPR 2026
  - Image Generation
  - diffusion model acceleration
  - feature caching
  - dynamic programming
  - path planning
  - training-free
date: 2026-05-08
content_hash: a281e04a17808aff
---

# Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache

**Conference**: CVPR 2026
**arXiv**: [2602.22654](https://arxiv.org/abs/2602.22654)
**Code**: [https://github.com/argsss/DPCache](https://github.com/argsss/DPCache)
**Area**: Diffusion Models
**Keywords**: diffusion model acceleration, feature caching, dynamic programming, path planning, training-free

## TL;DR
This paper formalizes diffusion model sampling acceleration as a global path planning problem. By constructing a Path-Aware Cost Tensor (PACT) and applying dynamic programming to select the optimal sequence of key timesteps, the method achieves training-free 4.87× acceleration while surpassing the full-step baseline in generation quality.

## Background & Motivation

1. **Background**: Diffusion models, particularly DiT-based architectures, have achieved remarkable success in image and video generation. However, the substantial computational overhead of multi-step iterative sampling severely hinders practical deployment. Caching-based methods have attracted considerable attention as training-free acceleration solutions—their core idea is to reuse or predict intermediate features that are highly similar across adjacent timesteps.

2. **Limitations of Prior Work**: Existing caching methods suffer from two fundamental problems: (1) **Fixed scheduling strategies** (e.g., DeepCache) ignore local feature dynamics and introduce severe deviation at critical transition regions; (2) **Locally adaptive strategies** (e.g., TeaCache, SpeCa) make greedy, myopic decisions that may skip critical timesteps, leading to irreversible trajectory drift and error accumulation.

3. **Key Challenge**: The pivotal decision in caching acceleration—"at which timesteps to perform full computation, and at which to use cached predictions"—is intrinsically a global optimization problem. Yet all existing methods make this decision locally, completely ignoring the global structure of the denoising trajectory.

4. **Goal**: Design a globally optimal sampling schedule such that, given a fixed computational budget of $K$ steps, the selected sequence of key timesteps minimizes the total deviation along the entire denoising trajectory.

5. **Key Insight**: The authors observe that the shape of the denoising trajectory is largely independent of the generated content and is primarily determined by the diffusion model itself. This property enables precomputing the optimal schedule on a small set of calibration samples and applying it to arbitrary inputs.

6. **Core Idea**: Reformulate diffusion sampling acceleration as a path planning problem, capture the path-dependent nature of skip-step errors via a 3D Path-Aware Cost Tensor, and solve for the globally optimal schedule exactly using dynamic programming.

## Method

### Overall Architecture
DPCache proceeds in three stages: (1) **Calibration**: Run full $T$-step denoising on approximately 10 samples, collect and cache intermediate features from all layers, and construct the Path-Aware Cost Tensor (PACT); (2) **Optimal Schedule Selection**: Given a target step count $K < T$, apply dynamic programming to select from PACT the key timestep sequence $\mathcal{T}$ that minimizes total path cost; (3) **Inference**: Perform full forward passes only at timesteps in $\mathcal{T}$ and cache the resulting features; at all remaining timesteps, predict outputs from cached features using methods such as Taylor expansion.

### Key Designs

1. **Path-Aware Cost Tensor (PACT)**:

    - **Function**: Quantifies the path-dependent nature of skip-step errors.
    - **Mechanism**: Constructs a 3D tensor $\mathcal{C} \in \mathbb{R}^{(T+1) \times (T+1) \times (T+1)}$, where $\mathcal{C}[i,j,k]$ (with $i>j>k$) denotes the cumulative error incurred by skipping from timestep $j$ to $k$, conditioned on the previous key timestep being $i$: $\mathcal{C}[i,j,k] = \sum_{\tau=k}^{j-1} \|h_\tau^L - h_{pred,\tau}^L(i,j)\|_1$. The tensor is 3D rather than 2D because predicted features depend on the previously cached state (path dependency), which a 2D cost matrix cannot capture. The cumulative error formulation naturally penalizes large skips that appear locally optimal but are globally unstable, by accumulating prediction deviations across all intermediate steps.
    - **Design Motivation**: Addresses the failure of 2D cost matrices to account for path dependency, enabling schedule optimization to be grounded in true trajectory deviation.

2. **Dynamic Programming for Optimal Schedule Selection**:

    - **Function**: Exactly identifies the globally optimal $K$-step sampling sequence within an exponentially large search space.
    - **Mechanism**: Maintains a DP table $D[m,k]$ (minimum cumulative cost of reaching timestep $k$ using $m$ key steps) and a path table $P[m,k]$ for backtracking, with the recurrence $D[m,k] = \min_{j>k} D[m-1,j] + \mathcal{C}[P[m-1,j], j, k]$. The first $M=3$ timesteps are forced to be included to preserve critical early denoising dynamics. Time complexity is $O(KT^2)$ and space complexity is $O(KT)$; for the typical setting $K < T = 50$, the optimization overhead is negligible and incurred only once.
    - **Design Motivation**: Dynamic programming is the natural choice for this sequential decision problem, guaranteeing global optimality in polynomial time and far outperforming greedy or heuristic search.

3. **Content-Agnostic Calibration Strategy**:

    - **Function**: Ensures that the precomputed schedule generalizes to arbitrary inputs.
    - **Mechanism**: Exploits the content-independence of denoising trajectory shapes, requiring only approximately 10 random calibration samples. Experiments demonstrate that even a single sample yields nearly identical schedules and generation quality, and that switching to an entirely different prompt dataset (DrawBench → PartiPrompts) does not affect the result.
    - **Design Motivation**: Minimizes the calibration overhead of DPCache, requiring only a one-time precomputation for practical deployment.

### Loss & Training
DPCache is entirely training-free. The calibration stage requires only standard forward inference to collect features, with no gradient computation or parameter updates. The prediction step is compatible with any caching prediction method (e.g., Taylor expansion from TaylorSeer, Hermite polynomials from HiCache), with second-order Taylor prediction used by default.

## Key Experimental Results

### Main Results (FLUX.1-dev, DrawBench)

| Method | Speedup | ImageReward↑ | CLIP Score↑ | PSNR↑ | SSIM↑ |
|--------|---------|--------------|-------------|-------|-------|
| 50 steps (baseline) | 1.00× | 0.979 | 17.40 | - | - |
| DPCache (K=13) | 3.54× | **1.007** | **17.34** | **21.65** | **0.8106** |
| DPCache (K=9) | 4.87× | **0.958** | **17.33** | **18.77** | **0.7117** |
| TeaCache (K=13 equiv.) | 3.42× | 0.934 | 17.17 | 16.31 | 0.6812 |
| TaylorSeer (K=13 equiv.) | 3.51× | 0.939 | 17.31 | 16.95 | 0.6922 |
| SpeCa (K=13 equiv.) | 3.62× | 0.975 | 17.27 | 18.35 | 0.6773 |

### HunyuanVideo (VBench)

| Method | Speedup | VBench Score↑ | PSNR↑ | Memory (GB) |
|--------|---------|---------------|-------|-------------|
| 50 steps | 1.00× | 80.93 | - | 60.22 |
| DPCache (K=11) | 4.05× | **80.35** | **23.11** | 60.58 |
| DPCache (K=9) | 4.75× | **80.23** | **21.04** | 60.58 |
| TaylorSeer | 3.87× | 80.33 | 18.53 | 84.47 |
| SpeCa | 4.05× | 80.26 | 20.09 | 84.47 |

### Ablation Study (PACT, FLUX K=13)

| Cost Dimension | Cumulative Error | ImageReward↑ | PSNR↑ | SSIM↑ |
|---------------|-----------------|--------------|-------|-------|
| 2D | ✘ | 1.001 | 20.87 | 0.7881 |
| 2D | ✔ | 0.977 | 19.46 | 0.7605 |
| 3D | ✘ | 0.998 | 21.05 | 0.7952 |
| **3D** | **✔** | **1.007** | **21.65** | **0.8106** |

### Key Findings
- **Surpassing the full-step baseline**: On FLUX, DPCache at 3.54× speedup improves ImageReward over the baseline (+0.028), and at 4.87× speedup still substantially outperforms all competing methods (+0.031). This indicates that a globally optimal schedule effectively eliminates redundant steps from the original trajectory.
- **3D path dependency in PACT is critical**: The 3D + cumulative error configuration improves PSNR by 0.78 over the 2D variant, confirming the necessity of path-dependent modeling. Notably, 2D + cumulative error performs worst (0.977), since inaccurate error estimates in 2D are amplified by accumulation, misleading the schedule selection.
- **Calibration is remarkably robust**: A single calibration sample yields competitive results, and the derived schedules are identical across different datasets—confirming that denoising trajectory shape is indeed content-agnostic.
- **Significant memory advantage**: DPCache caches only the last-layer features, adding only 0.36 GB on HunyuanVideo, whereas TaylorSeer and SpeCa cache all layers, incurring an additional 24.25 GB.

## Highlights & Insights
- **Reformulating sampling acceleration as a path planning problem** is the paper's most significant contribution. This is not merely a change of framing; it reveals a qualitative leap from fixed/local scheduling to global scheduling—yielding PSNR gains of 2–5 dB and SSIM improvements exceeding 0.1. This path planning perspective is transferable to any acceleration scenario involving sequential decision-making.
- **Generalization from very few calibration samples** is highly practical—it demonstrates that the global structure of the denoising trajectory is an intrinsic property of the model rather than the data, providing a theoretical basis for one-time precomputation.
- The **3D cost tensor vs. 2D cost matrix** ablation is elegantly designed and clearly demonstrates that path dependency cannot be ignored.

## Limitations & Future Work
- The construction of PACT has complexity $O(T^3)$, which may become a bottleneck for models with large step counts ($T \gg 50$), even though it is computed only once.
- The method assumes that denoising trajectory shape is content-agnostic; whether this holds for out-of-distribution prompts warrants further investigation.
- Only the last-layer features are cached; applying distinct schedules at different layers (layer-adaptive scheduling) may yield further performance gains.
- Integration with training-based step reduction methods (e.g., DMD, consistency models) remains unexplored; the two approaches may be complementary.

## Related Work & Insights
- **vs. DeepCache**: DeepCache employs a fixed-interval caching strategy that ignores variability in timestep importance; DPCache automatically identifies critical timesteps through global optimization.
- **vs. TeaCache**: TeaCache's locally adaptive strategy improves upon fixed strategies but makes greedy decisions that may skip critical steps and cause irreversible drift; DPCache's global perspective fundamentally resolves this issue.
- **vs. TaylorSeer**: TaylorSeer proposes Taylor expansion-based feature prediction; DPCache directly reuses its prediction method while replacing the scheduling strategy, demonstrating that *when* to compute matters more than *how* to predict.

## Rating
- Novelty: ⭐⭐⭐⭐ — The path planning perspective is original; the 3D design of PACT has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three models (DiT/FLUX/HunyuanVideo), three tasks, comprehensive ablations, decisively outperforming all baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical structure, intuitive figures, rigorous algorithmic presentation.
- Value: ⭐⭐⭐⭐⭐ — A plug-and-play training-free acceleration framework balancing efficiency and quality with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[CVPR 2026\] TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration](tc-padé_trajectory-consistent_padé_approximation_for_diffusion_acceleration.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration](lesa_learnable_stage-aware_predictors_for_diffusion_model_acceleration.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)

<!-- RELATED:END -->
