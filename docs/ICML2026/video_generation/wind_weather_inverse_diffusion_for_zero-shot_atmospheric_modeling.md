---
title: >-
  [Paper Note] WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling
description: >-
  [ICML2026][Video Generation][Meteorological foundation models] WIND models the global atmospheric sequence as an unconditional video diffusion prior. During inference, it formulates forecasting, downscaling…
tags:
  - "ICML2026"
  - "Video Generation"
  - "Meteorological foundation models"
  - "inverse problems"
  - "diffusion forcing"
  - "posterior sampling"
  - "physical constraints"
date: 2026-05-08
content_hash: 0e984e0ee9950f2f
---

# WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling

**Conference**: ICML2026  
**arXiv**: [2602.03924](https://arxiv.org/abs/2602.03924)  
**Code**: No public code available  
**Area**: Scientific Computing / Atmospheric Modeling / Diffusion Models  
**Keywords**: Meteorological foundation models, inverse problems, diffusion forcing, posterior sampling, physical constraints  

## TL;DR
WIND models the global atmospheric sequence as an unconditional video diffusion prior. During inference, it formulates forecasting, downscaling, sparse reconstruction, mass conservation, and warming scenarios as differentiable inverse problems, solving multiple meteorological and climate tasks zero-shot using a single frozen model.

## Background & Motivation
**Background**: AI weather forecasting has established an efficient alternative to traditional numerical weather prediction (NWP), with models like GraphCast and GenCast providing strong results for specific prediction tasks. Meanwhile, downstream demands in atmospheric science extend far beyond medium-range forecasting, including spatial downscaling, temporal downscaling, sparse observation completion, long-term climate scenarios, and physical conservation constraints.

**Limitations of Prior Work**: The current model ecosystem is fragmented. Models are often trained for a single task: forecasting models for prediction, downscaling models for resolution enhancement, and reconstruction models for observation completion. Switching tasks requires retraining or fine-tuning, which is costly and fails to guarantee a shared atmospheric physical prior across different tasks.

**Key Challenge**: Atmospheric systems require both strong probabilistic generative capabilities and the ability to be stably guided by external physical or observational constraints. Purely autoregressive models suffer from error accumulation during long rollouts; standard sequence diffusion models struggle with mixing the "clean" frames of the previous window with the "noisy" future frames; and conditional diffusion loses the unity of a foundation model if trained separately for each task.

**Goal**: The authors aim to train a single atmospheric generative prior that performs various weather/climate tasks during the inference phase solely through changes in the forward operator, without task-specific fine-tuning. In other words, the model learns "what a reasonable atmospheric sequence looks like" during training, and is told "what observations or physical conditions must be met this time" during inference.

**Key Insight**: Atmospheric data is treated as video, where variables are channels, timesteps are frames, and the global grid represents spatial dimensions. Training employs *diffusion forcing*, allowing each frame to have an independent noise level. Inference utilizes *moment matching posterior sampling* (MMPS) to estimate the likelihood gradient of observations, injecting arbitrary differentiable constraints into the reverse diffusion process.

**Core Idea**: Train an atmospheric video diffusion prior using diffusion forcing that can mix clean and noisy frames, then unify all downstream tasks as inverse problems defined by $Y = \mathcal{A}(X) + \eta$, with constraints imposed by MMPS during the sampling process.

## Method
The WIND approach resembles "learning an atmospheric world model first, then writing tasks as observation equations." The model itself does not know whether a specific task is called forecasting, downscaling, or sparse reconstruction; these differences are encapsulated in the operator $\mathcal{A}$ during inference.

### Overall Architecture
The training data comes from ERA5. The paper uses a 1.5-degree resolution, 70 atmospheric variables, and sequences of length 5 at 6-hour intervals. The backbone is a UViT, where inputs and outputs are atmospheric state sequences of shape $T \times C \times H \times W$. During training, noise levels are sampled independently for each frame, transforming clean atmospheric sequences into sequences with varying levels of corruption, from which the UViT recovers the clean sequence.

During inference, given a task observation $Y$ and a forward operator $\mathcal{A}$—for instance, $\mathcal{A}$ is average pooling in spatial downscaling, a temporal mean in temporal downscaling, a binary mask in sparse reconstruction, and a nonlinear global dry air mass calculation in mass conservation—each step of the reverse diffusion begins with a prior score from WIND, followed by a likelihood score from MMPS based on the difference between $\mathcal{A}(\hat X)$ and the target $Y$. The sample is updated by the sum of both.

### Key Designs
1. **Unifying Atmospheric Priors with Diffusion Forcing**:
    - **Function**: Enables the model to process clean context frames and noisy future frames simultaneously within the same sequence, supporting stable rollouts of arbitrary length.
    - **Mechanism**: While standard video diffusion typically assigns the same noise level to all frames, WIND independently samples $k^t$ for each timestep to generate $z^t = \alpha(k^t)x^t + \beta(k^t)\epsilon^t$. The model does not explicitly receive the noise levels and instead must infer the uncertainty of each frame from the input state.
    - **Design Motivation**: Autoregressive weather generation requires the last frame of the previous window to serve as the clean context for the next. If the model has never seen clean/noisy mixed states during training, out-of-distribution inputs occur; independent noise levels resolve this.

2. **Formulating Downstream Tasks as Differentiable Inverse Problems**:
    - **Function**: Avoids training separate conditional models for each weather task, allowing a frozen model to adapt via different operators.
    - **Mechanism**: During inference, $Y = \mathcal{A}(X) + \eta$ is uniformly set. Spatial downscaling uses $\mathcal{A}(X) = \mathrm{AvgPool}(X)$, temporal downscaling uses $\mathcal{A}(X) = \frac{1}{T}\sum_t x^t$, sparse reconstruction uses $\mathcal{A}(X) = M \odot X$, and physical conservation uses the nonlinear integral formula for dry air mass.
    - **Design Motivation**: Weather tasks often essentially involve "recovering a complete state satisfying atmospheric priors from partial observations." By compressing task differences into $\mathcal{A}$, the model achieves zero-shot transfer via a single set of priors and samplers.

3. **Inference with MMPS Guidance instead of Point-Estimate Constraints**:
    - **Function**: More stably incorporates observational/physical constraints during reverse diffusion, specifically avoiding strong misleading gradients during high-noise stages.
    - **Mechanism**: While standard diffusion posterior sampling often approximates $p(X|Z)$ as a Dirac delta at the current prediction, WIND uses MMPS to approximate $p(X|Z)$ as a Gaussian with covariance, using Tweedie covariance to estimate uncertainty. The prior dominates when noise is high, and likelihood guidance strengthens as noise decreases and predictions become more reliable.
    - **Design Motivation**: Atmospheric task constraints can be high-dimensional, low-dimensional, or nonlinear. Without considering the model's current uncertainty, observation gradients in early high-noise stages can easily distort samples and destroy the generative prior.

### Loss & Training
The training target is denoising score matching / clean sequence reconstruction. The model learns to recover atmospheric states from sequences with various noise level combinations. The paper uses a 5-frame window, 6-hour intervals, 70 variables, and a 1.5-degree ERA5 grid. The inference phase uses DDIM-like updates, adding the MMPS likelihood score for tasks requiring constraints. Meteorological forecasting, downscaling, and physical constraints are all completed during inference without task-specific fine-tuning.

## Key Experimental Results

### Main Results
The primary results demonstrate that the same model can work across various tasks. In medium-range forecasting, WIND's absolute CRPS on WeatherBench2 does not aim to surpass specialized models due to its coarser resolution, but it is more stable than autoregressive diffusion baselines. In downscaling and sparse reconstruction, WIND's advantages lie in spectral fidelity, physical consistency, and the lack of task-specific training.

| Task | Evaluation Setup | WIND Results | Baseline | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| 14-day Probabilistic Forecast | 24 initials from 2021, 10 member ensemble, CRPS/SSR | CRPS better than AR-UViT after several days; SSR approaches 1 | AR-UViT (Autoregressive Diffusion) | Diffusion forcing is more stable, avoiding overshoot in humidity/precipitation variables |
| WeatherBench2 24h T2m | CRPS ↓ | 0.286 | GenCast: 0.209, IFS ENS: 0.396 | Low-res general prior is weaker than specialized GenCast but better than IFS ENS |
| Spatial Downscaling | 6° to 1.5°, RMSE/PSD | Temp: 0.63, Geopotential: 45.17, MSLP: 42.68 | Specialized UViT/FNO models | RMSE often lower than UViT, high-frequency spectral detail better than FNO, no task training needed |
| Sparse Reconstruction (1%) | Only 1% observation points, RMSE | Temp: 0.65, Geopotential: 48.64, MSLP: 47.12 | UViT/Kriging | Better than specialized UViT for most variables; significantly less over-smoothing than Kriging |
| 4-year DAM Constrained Rollout | Dry air mass stability | Strictly maintains target DAM throughout | Unconstrained WIND | Physical constraints prevent mass drift after approx. 200 days |

### Ablation Study
| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| No DAM guidance | DAM drift after ~200 days in 4-year rollout | Purely data-driven generation deviates from physical conservation long-term |
| DAM guidance | Target DAM maintained throughout 4-year rollout | MMPS imposes hard physical constraints without retraining |
| Warming Free Run | Storm Bernd +2K/+14% humidity, only 50.3% precip signal retained | Model diffuses OOD thermal anomalies back to the training climate state |
| Warming Guided Run | Mean peak precipitation enhancement +13.9% | Close to the Clausius-Clapeyron expectation of ~+14% |
| Spatial Downscaling UViT | Lowest RMSE for most variables | Task-specific models prioritize pixel-wise error |
| WIND Spatial Downscaling | PSD closer to ERA5, Pearson consistency 0.96 | General prior preserves high-frequency and physical statistical structures better |

### Key Findings
- A single frozen model can cover multiple task categories by changing $\mathcal{A}$, proving that "meteorological foundation model + inverse problem inference" is more flexible than "one specialized model per task."
- While WIND does not always beat specialized UViT in RMSE, its spectra and distributions are closer to the ground truth ERA5, particularly reducing high-frequency smoothing issues common in deterministic models.
- Sparse reconstruction best highlights the value of a foundation prior: when input is only 1% of observations, specialized conditional models struggle to generalize, whereas WIND utilizes the global atmospheric prior to complete unobserved regions.
- Physical constraints are plug-and-play guidance during inference rather than soft regularization during training, making long-term mass conservation and warming scenarios controllable.

## Highlights & Insights
- The most elegant aspect of the paper is its unity: forecasting, downscaling, sparse reconstruction, mass conservation, and warming scenarios are not separate modules but different operators within the same posterior sampling framework.
- Diffusion forcing aligns perfectly with the needs of meteorological rollouts. It addresses a specific but critical problem in video diffusion: how to naturally accept a mixed noise state of "past knowns and future unknowns."
- The role of MMPS is not just to make the diffusion obey conditions, but to incorporate uncertainty into the guidance strength. For chaotic atmospheric dynamics, this is more rational than simple point-estimate DPS.
- This paper serves as a reminder for scientific machine learning not to focus solely on single-task SOTA. For climate scenarios, the ability to impose new physical constraints zero-shot may be more important than minor RMSE leads on a fixed benchmark.

## Limitations & Future Work
- WIND uses 1.5-degree ERA5; the authors acknowledge it is not intended to compete directly with 0.25-degree operational forecasting SOTA. Practical deployment would require higher resolution, more variables, and larger model scales.
- Many results are illustrated through plots and spectra; RMSE is not always superior to specialized models. Systematic quantitative evaluations for extreme events, local risks, and energy/moisture closures are still needed.
- MMPS guidance introduces additional inference costs, especially for constraints requiring conjugate gradient solvers. While the paper analyzes costs, optimization for multi-task, large ensembles, and long climate simulations is required.
- Warming experiments use simplified global thermal perturbations (+2K, +14% humidity), which is suitable for mechanism validation but still distant from realistic regional climate change scenarios.

## Related Work & Insights
- **vs. GenCast/GraphCast**: These models are optimized for medium-range forecasting and are stronger on WeatherBench2 absolute metrics; WIND’s advantage is unified inverse inference and zero-shot task transfer.
- **vs. full-sequence diffusion**: Full-sequence diffusion struggles to naturally continue from a clean context in long rollouts; WIND’s per-frame independent noise training is better suited for rolling generation.
- **vs. FNO/UViT downscaling**: Specialized models may have better RMSE but often produce smoothed predictions; WIND emphasizes physical realism in spectra and probabilistic distributions.
- **vs. Physics-Informed Neural Networks**: Conventional methods often write conservation laws into the loss or architecture; WIND chooses to impose constraints via operator guidance during inference, offering higher flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies diffusion forcing, MMPS, and atmospheric multi-task inverse problems naturally with high conceptual integrity.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers many tasks with long rollouts and OOD warming cases, though resolution and some metrics remain proof-of-concept.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology and helpful diagrams; however, results are scattered between the main text and appendix.
- Value: ⭐⭐⭐⭐☆ Highly inspiring for scientific foundation models and climate AI; currently more of a research framework, requiring high-resolution scaling for engineering deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seeing the Wind from a Falling Leaf](../../NeurIPS2025/video_generation/seeing_the_wind_from_a_falling_leaf.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](../../AAAI2026/video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)
- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[ICML 2026\] VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)
- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](../../ICLR2026/video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)

</div>

<!-- RELATED:END -->
