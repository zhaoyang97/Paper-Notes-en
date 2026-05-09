---
title: >-
  [Paper Note] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching
description: >-
  [CVPR 2026][Video Generation][Feature Caching] DisCa is the first framework to unify learnable feature caching with step distillation in a compatible manner, replacing hand-crafted caching strategies with a lightweight neural predictor (<4% of model parameters). Combined with Restricted MeanFlow for stable large-scale video DiT distillation, DisCa achieves an 11.8× near-lossless speedup on HunyuanVideo.
tags:
  - CVPR 2026
  - Video Generation
  - Feature Caching
  - Step Distillation
  - MeanFlow
  - Learnable Predictor
  - HunyuanVideo
date: 2026-05-08
content_hash: 92b193175ec1ad0a
---

# DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching

**Conference**: CVPR 2026
**arXiv**: [2602.05449](https://arxiv.org/abs/2602.05449)
**Code**: Coming soon
**Area**: Video Generation / Diffusion Model Acceleration
**Keywords**: Feature Caching, Step Distillation, MeanFlow, Learnable Predictor, HunyuanVideo

## TL;DR

DisCa is the first framework to unify learnable feature caching with step distillation in a compatible manner, replacing hand-crafted caching strategies with a lightweight neural predictor (<4% of model parameters). Combined with Restricted MeanFlow for stable large-scale video DiT distillation, DisCa achieves an 11.8× near-lossless speedup on HunyuanVideo.

## Background & Motivation

**Background**: Video diffusion models (e.g., HunyuanVideo) have reached state-of-the-art generation quality, but inference is extremely slow — generating a 5-second 704×704 video with HunyuanVideo requires 1,155 seconds under 50-step CFG inference. Two main acceleration paradigms exist: **step distillation** (e.g., MeanFlow), which reduces the number of sampling steps, and **feature caching** (e.g., TaylorSeer, TeaCache), which skips redundant computations.

**Limitations of Prior Work**: For step distillation, MeanFlow performs well on image generation but fails when directly applied to large-scale video DiTs — numerical errors in JVP computation, compounded by its aggressive one-step generation objective, cause training divergence and severe artifacts, with semantic scores dropping by 17.1% at 10-step MeanFlow. For feature caching, conventional methods exploit inter-step feature similarity via reuse or Taylor expansion, but distilled sparse sampling trajectories drastically increase inter-step feature discrepancy, rendering naive hand-crafted strategies ineffective — TaylorSeer suffers a 13.3% semantic score drop at high speedup ratios.

**Key Challenge**: The two acceleration paradigms are inherently incompatible — the sparse trajectories produced by distillation violate the inter-step redundancy assumptions that caching methods rely upon, making naive combination worse than either approach alone.

**Goal**: To make step distillation and feature caching genuinely compatible and complementary, achieving extreme acceleration on large-scale video DiTs without sacrificing quality.

**Key Insight**: Replace hand-crafted caching formulas with learnable neural predictors to capture high-dimensional feature dynamics; simultaneously stabilize distillation by restricting the compression range of MeanFlow.

**Core Idea**: Although feature dynamics on distilled trajectories exceed the modeling capacity of hand-crafted methods such as Taylor expansion, lightweight neural networks can still accurately learn these high-dimensional evolution patterns.

## Method

### Overall Architecture

DisCa employs a three-stage cascaded acceleration pipeline: (1) **CFG distillation** merges dual-branch inference into a single branch (2× speedup); (2) **Restricted MeanFlow** step distillation further reduces sampling steps from 50 to 10 (~5× speedup); (3) **Learnable Feature Caching** uses a lightweight predictor to skip steps along the distilled sparse trajectory. The three stages combined yield an 11.8× speedup. During inference, a full DiT forward pass is performed every $N$ steps to initialize the cache, while the intermediate $N-1$ steps rely solely on the lightweight predictor.

### Key Designs

1. **Restricted MeanFlow (Conservative Step Distillation)**:

    - **Function**: Compresses 50-step sampling to 10–20 steps while maintaining stable generation quality.
    - **Mechanism**: The original MeanFlow samples the mean-velocity interval $\mathcal{I}=(t-r)$ over $[0,1]$, targeting one-step generation. Restricted MeanFlow introduces a restriction factor $\mathcal{R} \in (0,1)$ to constrain the interval to $\mathcal{I} \in [0, \mathcal{R}]$, directly eliminating training samples with excessively high compression ratios. $\mathcal{R}=0.2$ is found to be optimal.
    - **Design Motivation**: The high complexity of large-scale video DiTs amplifies numerical errors in MeanFlow's JVP computation, and large time intervals (high compression ratios) further accumulate these errors. Rather than aggressively learning global mean velocity, it is more effective to stably learn local mean velocity and achieve high-quality generation through multi-step chaining.

2. **Learnable Feature Caching**:

    - **Function**: Replaces full DiT computation with a predictor on the distilled sparse sampling trajectory to further accelerate inference.
    - **Mechanism**: A lightweight Predictor $\mathcal{P}$ consisting of only 2 DiT Blocks (<4% of model parameters) takes the cache $\mathcal{C}$ from the previous full computation and the current noisy input $x_{t'}$ to predict the mean-velocity output at the current step. Unlike TaylorSeer, which maintains multi-layer multi-order derivative caches (requiring an additional 33.5 GB VRAM), DisCa retains only a single cache tensor from the final layer (only +0.43 GB), replacing complex cache structures with learned capacity.
    - **Design Motivation**: Inter-step feature discrepancy after distillation is substantial, exceeding the modeling upper bound of hand-crafted methods such as Taylor expansion. Data-driven neural networks are naturally suited to capturing such high-dimensional nonlinear dynamics.

3. **Memory-Efficient Cache Design**:

    - **Function**: Substantially reduces cache memory overhead, making the method practical for high-resolution long-video scenarios.
    - **Mechanism**: Rather than maintaining multi-tensor caches per DiT layer (as in TaylorSeer), only the model's final output tensor is retained as the cache and passed to the Predictor.
    - **Design Motivation**: In practical distributed parallel settings (sequence parallel size=4), multi-layer caching requires cross-GPU cache synchronization, whose communication overhead offsets the computational savings. Single-tensor caching entirely avoids this issue.

### Loss & Training

The Predictor is trained in two stages using MSE followed by GAN training:

- **MSE Stage** (500 iterations): Minimizes the L2 distance between the predictor output and the ground-truth output of the full model:
  $$\mathcal{L}_\mathcal{P} = \mathbb{E}\|\mathcal{M}_{\theta_M}(x_{t'}, r', t') - \mathcal{P}_{\theta_p}(\mathcal{C}, x_{t'}, r', t')\|_2^2$$
- **GAN Stage** (1,000 iterations): Introduces a multi-scale spectrally normalized discriminator $\mathcal{D}$ with Hinge Loss for adversarial training, enforcing the predictor to preserve high-frequency details and visual fidelity. The full model itself serves as the feature extractor $\mathcal{F}$ for adversarial training in feature space.
- Hyperparameters: Predictor learning rate $10^{-4}$, discriminator learning rate $10^{-2}$, adversarial loss weight $\lambda=1.0$.

## Key Experimental Results

### Main Results

Experiments are conducted on HunyuanVideo generating 704×704 resolution, 129-frame, 5-second videos, evaluated with VBench.

**Restricted MeanFlow Comparison** (vs. original MeanFlow baseline):

| Method | Steps | Speedup | Semantic↑ | Quality↑ | Total↑ |
|------|------|--------|---------|---------|-------|
| Original 50-step | 50×2 | 1.0× | 73.5% | 81.5% | 79.9% |
| MeanFlow 20-step | 20 | 4.96× | 66.6% | 81.8% | 78.8% |
| Restricted MeanFlow (R=0.2) 20-step | 20 | 4.97× | **70.4% (+5.7%)** | 81.8% | 79.5% |
| MeanFlow 10-step | 10 | 9.68× | 60.9% | 80.6% | 76.7% |
| Restricted MeanFlow (R=0.2) 10-step | 10 | 9.68× | **68.2% (+12.0%)** | 81.3% | 78.7% |

**DisCa vs. Existing Acceleration Methods (Comprehensive Comparison)**:

| Method | Speedup | Peak VRAM | Semantic↑ | Quality↑ | Total↑ |
|------|--------|-----------|---------|---------|-------|
| Original 50-step | 1.0× | 99.23 GB | 73.5% | 81.5% | 79.9% |
| Δ-DiT (N=8) | 4.55× | 97.68 GB | 42.7% (-41.9%) | 70.9% | 65.2% |
| PAB (N=8) | 6.46× | 121.3 GB | 56.3% (-23.4%) | 76.1% | 72.1% |
| TeaCache (l=0.4) | 9.22× | 97.70 GB | 62.1% (-15.5%) | 78.7% | 75.4% |
| TaylorSeer (N=6) | 6.96× | 130.7 GB | 63.7% (-13.3%) | 79.9% | 76.7% |
| FORA (N=6) | 8.01× | 124.6 GB | 57.5% (-21.8%) | 76.4% | 72.6% |
| **DisCa (R=0.2, N=2)** | **7.56×** | **97.64 GB** | **70.8% (-3.7%)** | **81.9%** | **79.7%** |
| **DisCa (R=0.2, N=3)** | **8.84×** | **97.64 GB** | **70.3% (-4.4%)** | **81.8%** | **79.5%** |
| **DisCa (R=0.2, N=4)** | **11.8×** | **97.64 GB** | **69.3% (-5.7%)** | **81.1%** | **78.8%** |

### Ablation Study

| Restricted MeanFlow | Learnable Predictor | GAN Training | Semantic↑ | Quality↑ | Total↑ |
|:---:|:---:|:---:|---------|---------|-------|
| ✔ | ✔ | ✔ | 69.3% (+0.0%) | 81.1% (+0.0%) | 78.7% |
| ✘ | ✔ | ✔ | 65.2% (-5.9%) | 80.3% (-1.0%) | 77.3% |
| ✔ | ✘ | — | 67.3% (-2.9%) | 80.5% (-0.7%) | 77.9% |
| ✔ | ✔ | ✘ | 68.5% (-1.2%) | 81.0% (-0.1%) | 78.5% |

### Key Findings

- **Restricted MeanFlow is the cornerstone**: Training caching directly on the original MeanFlow without the restriction causes a 5.9% semantic score collapse and "completely unacceptable distortions" in generated results.
- **Learnable predictor vs. training-free caching**: Even with Restricted MeanFlow, training-free caching still loses 2.9% semantic score and 0.7% quality score — high-dimensional feature dynamics genuinely require learned modeling.
- **GAN training is indispensable**: Removing adversarial training reduces semantic score by 1.2%, demonstrating that the MSE + adversarial loss combination is critical for maintaining semantic fidelity.
- **Clear memory efficiency advantage**: DisCa requires only 97.64 GB (+0.43 GB), while TaylorSeer requires 130.7 GB (+33.5 GB) and FORA requires 124.6 GB (+27.4 GB).

## Highlights & Insights

- DisCa is the first work to demonstrate that step distillation and feature caching can be complementary rather than conflicting. The key insight is replacing hard dependence on inter-step redundancy with a learnable predictor, enabling effective acceleration even on the sparse sampling trajectories induced by distillation. This opens a new direction of "training-free + training-aware synergy" for diffusion model acceleration.
- The design of Restricted MeanFlow is remarkably simple — merely restricting the sampling range of time intervals during training — yet it delivers a 12.0% semantic score improvement in the 10-step setting. This reveals an important principle: for distilling large-scale complex models, abandoning extreme compression targets can yield a globally superior quality–speed trade-off.
- The single-tensor cache design not only reduces memory usage but also eliminates cross-GPU communication latency bottlenecks in distributed parallel settings, making DisCa the only method that simultaneously satisfies both memory and latency constraints in practical deployment scenarios.

## Limitations & Future Work

- Training the predictor and discriminator requires approximately 1,500 iterations, making the approach no longer fully training-free; retraining is necessary whenever the base model or resolution changes.
- Validation is limited to HunyuanVideo; transferability to other video DiTs (CogVideoX, Wan, etc.) remains unknown.
- The restriction factor $\mathcal{R}$ requires manual tuning ($\mathcal{R}=0.2$ being optimal in experiments), with no adaptive selection strategy proposed.

## Related Work & Insights

- **vs. TaylorSeer**: TaylorSeer predicts cached features via Taylor expansion, but suffers a large performance drop (-13.3% semantic score) on distilled sparse trajectories, with substantial memory overhead (+33.5 GB). DisCa addresses both the modeling capacity and memory bottlenecks via a learnable predictor.
- **vs. TeaCache**: TeaCache uses timestep embeddings for adaptive caching decisions but still loses 15.5% semantic score at high speedup ratios. DisCa loses only 5.7% at a higher speedup ratio (11.8× vs. 9.22×).
- **vs. MeanFlow**: The original MeanFlow is designed for one-step generation and is too aggressive for large-scale video models. Restricted MeanFlow achieves stable distillation through the minimal intervention of interval restriction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to propose a distillation-compatible learnable caching framework that unifies two major acceleration paradigms.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison against 6 methods on HunyuanVideo, with clear ablations and complete memory/latency analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with strong motivation; notation-heavy but derivations are complete.
- Value: ⭐⭐⭐⭐⭐ An 11.8× near-lossless speedup offers substantial practical value for real-world video generation deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)
- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[ICLR 2026\] PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](../../ICLR2026/video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior](dreamshot_storyboard_synthesis.md)

</div>

<!-- RELATED:END -->
