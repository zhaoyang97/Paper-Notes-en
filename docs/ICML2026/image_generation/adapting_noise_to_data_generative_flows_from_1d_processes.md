---
title: >-
  [Paper Note] Adapting Noise to Data: Generative Flows from Learned 1D Processes
description: >-
  [ICML 2026][Image Generation][flow matching] This work argues that the default Gaussian latent in flow/diffusion models is not always suitable for the data distribution. It proposes using learned 1D quantile functions to…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "flow matching"
  - "data-adaptive noise"
  - "quantile function"
  - "non-Gaussian prior"
  - "heavy-tailed generative modeling"
date: 2026-05-08
content_hash: d45de2a233332b0a
---

# Adapting Noise to Data: Generative Flows from Learned 1D Processes

**Conference**: ICML 2026  
**arXiv**: [2510.12636](https://arxiv.org/abs/2510.12636)  
**Code**: https://github.com/TUB-Angewandte-Mathematics/Adapting-Noise  
**Area**: Image Generation / Flow Matching  
**Keywords**: flow matching, data-adaptive noise, quantile function, non-Gaussian prior, heavy-tailed generative modeling  

## TL;DR
This work argues that the default Gaussian latent in flow/diffusion models is not always suitable for the data distribution. It proposes using learned 1D quantile functions to construct a data-adaptive product prior, jointly learning the noise and velocity field within a flow matching framework. This shortens the transport path and improves performance on heavy-tailed weather data and low-capacity image generation.

## Background & Motivation
**Background**: Flow matching, diffusion, and consistency-style models typically start from a simple latent/noise distribution and learn a velocity field or score to push the latent toward the data distribution. The default choice is almost always Gaussian due to easy sampling, mature theory, and independent dimensions.

**Limitations of Prior Work**: Gaussian latents may be unsuitable for data with heavy tails, compact support, or strong edge structures. For targets like heavy-tailed weather or Neal's funnel, a Gaussian starting point results in long transport paths, requiring the model to handle both marginal tail behavior and cross-dimensional dependencies via the velocity field. Existing heavy-tailed diffusion models manually select Student-t or alpha-stable noise, but tail parameters must be tuned and may not match the data marginals of each dimension.

**Key Challenge**: The latent needs to be simple enough for sampling and training, yet sufficiently close to the data's marginal structure to reduce the difficulty of flow learning. Learning a full high-dimensional prior might bake correlations into the latent, making it complex and unstable. Conversely, fixing it as Gaussian wastes model capacity on handling structures that could be explained by the marginal prior.

**Goal**: To learn a latent distribution that remains independent, sampleable, and lightweight, but allows the marginal distribution of each dimension to adapt to the data. This shifts cross-dimensional correlations to the velocity field while leaving marginal support/tails to the quantile prior.

**Key Insight**: One-dimensional distributions can be fully represented by quantile functions, and the Wasserstein-2 distance in 1D is equivalent to the $L_2$ distance between quantile functions. The authors parameterize the quantile of each dimension using rational quadratic splines, keeping the product latent simple while expressing heavy-tailed, compactly supported, and multimodal marginals.

**Core Idea**: Use 1D quantile functions to learn a data-adaptive noise $\mathbf{Q}_\phi(\mathbf{U})$, optimized jointly with the velocity field through Wasserstein alignment and flow matching loss.

## Method
The paper first establishes a general view: a high-dimensional noising process can be constructed from independent 1D processes. As long as each 1D process has an accessible velocity field, the conditional velocity for multidimensional flow matching can be constructed. The authors further formulate the 1D process as a quantile process to make the final latent distribution learnable.

### Overall Architecture
Traditional flow matching uses linear interpolation $X_t=(1-t)X_0+tX_1$, where $X_1$ is Gaussian noise. This paper replaces $X_1$ with $\mathbf{Q}_\phi(\mathbf{U})=(Q^1_\phi(U^1),\ldots,Q^d_\phi(U^d))$, where $U^i\sim\mathcal{U}(0,1)$. Each $Q^i_\phi$ is a monotonic 1D quantile function, ensuring a valid output distribution.

During training, the method computes a minibatch OT assignment between the data batch and the quantile latent batch. This coupling is used for two purposes: minimizing the Wasserstein alignment loss between the latent and the data, and training the velocity field using OT-coupled endpoints. After several training steps, the quantile is frozen, and only the velocity field is optimized further, resulting in almost no extra cost during inference.

The method also discusses more general 1D processes, such as Kac processes and MMD gradient flow, and how quantile interpolants can be connected to few-step/IMM methods. However, the main experiments focus on the learned static quantile prior.

### Key Designs
1. **Decomposition from 1D processes to high-dimensional product priors**:

	- **Function**: Introduces non-Gaussian latents without manually designing high-dimensional noise PDEs.
	- **Mechanism**: By keeping the dimensions of the multidimensional noise $\mathbf{N}_t=(N_t^1,\ldots,N_t^d)$ independent, each with a 1D velocity $v_t^i$, the high-dimensional velocity is simply the concatenation of components. Data correlation is not handled by the noise but learned by the velocity field.
	- **Design Motivation**: 1D processes like Kac or uniform/MMD may not be easily defined directly in high dimensions. Component-wise construction allows these 1D processes to be used for generative modeling in any dimension.

2. **Quantile function parameterization of the latent**:

	- **Function**: Allows the latent marginal of each dimension to automatically adapt to the scale, support, and tail of the data.
	- **Mechanism**: Uses rational quadratic splines to represent $Q^i_\phi$, with monotonicity constraints ensuring a valid quantile. Sampling involves drawing $U^i\sim\mathcal{U}(0,1)$ and passing it through $Q^i_\phi$.
	- **Design Motivation**: Quantile functions are universal representations for 1D distributions and naturally align with Wasserstein-2; they can learn different tail behaviors per dimension compared to manually tuning Student-t degrees of freedom.

3. **Joint training of Wasserstein alignment and FM**:

	- **Function**: Aligns learned noise with data marginals while ensuring the velocity field learns the transport from latent to data.
	- **Mechanism**: The objective is $\mathcal{L}(\theta,\phi)=\mathcal{L}_{CFM}(\theta,\phi)+\lambda\mathcal{L}_{AN}(\phi)-\beta\mathcal{R}(\phi)$. Here $\mathcal{L}_{AN}=W_2^2(\mu_0,\nu_\phi)$, and $\mathcal{R}$ is a log-det/entropy regularizer. The CFM loss uses the same minibatch OT coupling.
	- **Design Motivation**: Learning the latent solely via FM loss can lead to degeneracy; Wasserstein alignment provides a direct marginal matching signal, and entropy regularization prevents quantile collapse in high-dimensional small batches.

### Loss & Training
In practice, each batch samples data $\{\mathbf{x}_i\}$ and uniform latents $\{\mathbf{u}_j\}$, computes $\mathbf{y}_j=\mathbf{Q}_\phi(\mathbf{u}_j)$, and finds the assignment minimizing $\|\mathbf{x}_i-\mathbf{y}_j\|^2$. For matched endpoints, it interpolates $\mathbf{z}_j=(1-t_j)\mathbf{x}_{P(j)}+t_j\mathbf{y}_j$, with a velocity target of $\mathrm{sg}(\mathbf{y}_j-\mathbf{x}_{P(j)})$. Stop-gradient prevents the quantile from speculatively reducing FM loss by shrinking endpoint displacement.

The quantile parameter count is small: e.g., on CIFAR-10 with $d=3072$ and 32 spline bins, it is about 300k parameters, much smaller than the U-Net. The paper reports ~2.7% overhead during joint training and ~0.5% overhead after freezing the quantile.

## Key Experimental Results

### Main Results
The most convincing results come from HRRR-mini weather data. This data's total precipitation has strong heavy tails, and metrics focus on extreme event frequency, intensity, and tail distribution fitting.

| Metric | Gaussian baseline↓ | Student-t baseline↓ | Quantile (Ours)↓ | Interpretation |
|------|--------------------|---------------------|------------------|------|
| Extreme event frequency error | 0.9689 | 0.8859 | 0.7550 | Learned quantile better generates extreme precipitation events |
| Extreme event magnitude error | 0.2455 | 0.1482 | 0.0634 | Most significant improvement in extreme event intensity |
| Spectral distance | 3.1836 | 2.0719 | 1.1063 | Spatial spectrum is closer to real weather fields |
| Tail KS distance | 0.2067 | 0.1014 | 0.0393 | Tail distribution fit is superior to tuned Student-t |
| Kurtosis deviation | 4.930 | 2.890 | 1.588 | Reduced kurtosis deviation |
| Skewness deviation | 1.157 | 0.830 | 0.580 | Reduced skewness deviation |

For image generation, MNIST has strong marginal structures, and the learned latent significantly reduces FID with a low-capacity U-Net. For CIFAR-10, where spatial/channel correlations dominate, the product prior provides smaller improvements but remains competitive. Using a larger 55M parameter model, the quantile prior achieved an FID of 3.25 vs 3.37 for Gaussian.

### Ablation Study
On CIFAR-10, the authors scanned the entropy regularization strength $\beta$. Most settings outperformed the Gaussian baseline under 20-step and 100-step Euler sampling, indicating good stability of quantile learning, though excessive regularization leads to degradation.

| Configuration | FID @ 20 steps↓ | FID @ 100 steps↓ | Description |
|------|-----------------|------------------|------|
| Quantile, $\beta=0.2$ | 7.81 | 4.75 | Already better than baseline |
| Quantile, $\beta=0.3$ | 7.48 | 4.53 | Best at 20 steps |
| Quantile, $\beta=0.5$ | 7.66 | 4.49 | Near best at 100 steps |
| Quantile, $\beta=0.8$ | 7.77 | 4.42 | Best at 100 steps |
| Quantile, $\beta=1.0$ | 8.35 | 4.66 | Strong reg, 20-step degradation |
| Gaussian baseline | 8.42 | 4.63 | Default Gaussian starting point |

### Key Findings
- Learned quantiles are most valuable for heavy-tailed data. In HRRR, all tail-centric metrics are significantly better than Gaussian and Student-t, showing that per-dimension learned tails are more robust than manually tuned distributions.
- Since the product prior is not responsible for cross-dimensional correlations, the smaller gain on CIFAR-10 is expected; its main role is to relieve the burden of marginal distribution and support/tail modeling.
- In low-dimensional examples like checkerboard and funnel, the learned latent significantly shortens the transport path, leading to faster velocity field convergence.
- The regularizer $\beta$ is critical for stable training. Without proper entropy/log-det constraints, the quantile might over-contract or produce unstable gradients in high-dimensional small-batch settings.

## Highlights & Insights
- The paper transforms "noise distribution selection" from a manual hyperparameter into a learnable object while keeping the latent simple and sampleable, which is a practical compromise.
- The quantile function is an elegant entry point: it has high 1D expressivity, controllable monotonicity, and clear Wasserstein geometry, avoiding the complexity of learning a full high-dimensional prior.
- Using the same minibatch OT coupling for both alignment and OT-FM is efficient, reducing additional algorithmic components.
- Experiments on heavy-tailed scientific data demonstrate the method's value more effectively than image FID, as the limitations of Gaussian priors are amplified in extreme event modeling.

## Limitations & Future Work
- The learned latent is a product distribution and cannot directly represent correlations between dimensions. Gains are limited for data dominated by correlations, such as natural images.
- Quantile learning signals in high dimensions come from minibatch OT; this may be noisy with a fixed batch size, requiring regularization and freezing strategies for stability.
- The work has not yet systematically tested higher-dimensional, larger-resolution, or text-to-image conditional generation. Whether gains hold in large-scale diffusion systems remains to be verified.
- Future work could learn time-dependent quantile processes to optimize the entire path rather than just the endpoint prior, or explore conditional quantiles to modulate latent marginals with class or text conditions.

## Related Work & Insights
- **vs Gaussian diffusion/FM**: Standard Gaussian is simple but light-tailed, making it a poor match for heavy-tailed targets; this work uses learned quantiles to automatically adjust tail/support.
- **vs Student-t / alpha-stable noise**: Heavy-tailed noise requires manual selection of families and parameters; this work learns per-dimension quantiles directly from data, avoiding manual tuning of degrees of freedom.
- **vs normalizing-flow prior**: Full flow priors are more expressive but complex; this work deliberately restricts itself to product priors to keep the training lightweight while delegating correlations to the velocity field.
- **Insight**: For generative models, neither the path nor the prior should always default to Gaussian. Letting the latent capture simple marginal structures first, then having the main network model dependencies, might be a more efficient division of labor.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Comprehensive integration of data-adaptive noise via quantile functions into the FM and 1D process framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across synthetic, image, and weather data, though large-scale conditional generation is less explored.
- Writing Quality: ⭐⭐⭐⭐ Rich theoretical framework with a clear main thread, though some notations and the appendix make for a high entry barrier.
- Value: ⭐⭐⭐⭐⭐ Strong implications for flow matching, diffusion prior design, and heavy-tailed scientific generative modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2026\] Path-Coupled Bellman Flows for Distributional Reinforcement Learning](path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [\[ICML 2026\] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)

</div>

<!-- RELATED:END -->
