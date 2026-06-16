---
title: >-
  [Paper Note] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework
description: >-
  [CVPR 2026][Image Generation][Diffusion Model] This paper theoretically proves that the memorization problem in diffusion models stems from the sharpness of softmax weights in the empirical score function (where a single training sample dominates). It proposes two smoothing methods, Noise Unconditioning and Temperature Smoothing, to mitigate memorization and enhanc
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Model
date: 2026-05-08
content_hash: ea80e7412e6480e6
---
# Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework

**Conference**: CVPR 2026  
**arXiv**: [2601.19285](https://arxiv.org/abs/2601.19285)  
**Code**: [GitHub](https://github.com/xinyu-zhou/score-smoothing)  
**Area**: Image Generation  
**Keywords**: Diffusion Models, Memorization, Generalization, Score Function Smoothing, Temperature Scaling

## TL;DR
This paper theoretically proves that the memorization problem in diffusion models stems from the sharpness of softmax weights in the empirical score function (where a single training sample dominates). It proposes two smoothing methods, Noise Unconditioning and Temperature Smoothing, to mitigate memorization and enhance generalization while maintaining generation quality.

## Background & Motivation

1. **Background**: Diffusion models have become the mainstream framework for image generation, achieving high-quality results by gradually adding noise and learning the reverse process. Their core mechanism involves using neural networks to approximate the score function at different noise levels.
2. **Limitations of Prior Work**: Numerous studies have found that diffusion models "memorize" training data—generated samples can be identical to training samples. Theoretical analysis suggests that if a neural network perfectly learns the empirical score function, sampling degrades into copying training data, failing to produce novel samples.
3. **Key Challenge**: A fundamental contradiction exists between the perfect memorization predicted by theory and the generalization capability observed in practice. While finite neural network capacity and regularization are known to play a role, there is a lack of systematic theoretical explanation for "why neural networks can partially resolve memorization."
4. **Goal**: (1) Establish a theoretical framework to explain the root cause of memorization; (2) explain why neural networks achieve generalization; (3) propose methods to further enhance generalization based on the theory.
5. **Key Insight**: Decomposing the empirical score function into a weighted sum of Gaussian component scores reveals that the weights are essentially a softmax function. In high-dimensional space, each training sample geometrically corresponds to a "shell"; during sampling at low noise levels, the sampling point falls only within a single shell, leading to single-point dominance.
6. **Core Idea**: Neural networks achieve generalization by implicitly smoothing the score function weights, so that sampling is influenced by the local manifold (multiple neighbors) rather than a single point. This effect can be further enhanced through explicit smoothing methods.

## Method

### Overall Architecture
This paper addresses a seemingly paradoxical question: theoretically, if a neural network perfectly learns the empirical score function, diffusion sampling should degrade into "copying training samples," yet in reality, models generate new images—where does generalization come from? The authors'切入角度 is to dismantle the empirical score function: it is essentially a weighted sum of scores from Gaussian components corresponding to training samples, where the weights form a softmax. Thus, "memorization vs. generalization" is translated into a question of "how sharp the softmax weights are"—the sharper the weights (single sample exclusivity), the more sampling collapses to training points; the flatter the weights (multiple neighbors contributing together), the more sampling is constrained by the local manifold and achieves generalization. Following this line, the paper first uses high-dimensional geometry to explain why memorization occurs, then provides two methods to explicitly flatten the weights: Noise Unconditioning on the training side by removing noise conditioning, and Temperature Smoothing on the sampling side by introducing a temperature parameter; both switch adaptively based on noise levels.

### Key Designs

**1. High-dimensional Shell Geometry Analysis: Translating "Weight Sharpness" to "Thin Shell Overlap"**

To explain memorization, one must first see how weights appear in high-dimensional space. In a $d$-dimensional space, a Gaussian distribution centered at training sample $\mu_j$ does not concentrate its probability mass at the center, but rather on a thin shell with a radius of approximately $\sigma_i\sqrt{d}$. The weights in the empirical score function, $w_{ij}(x) = \text{Softmax}(f(x, \mu_j, \sigma_i))$, exhibit two sharp properties: first, $\sigma$-dominance, where for a fixed center and sampling position, there exists an optimal noise level $\sigma_j^*$ whose weight far exceeds others (with ratios up to $e^6 \approx 403$); second, $\mu$-dominance, where the weight of the nearest training sample grows exponentially relative to others as distances vary. The consequence of these properties is that at high noise levels, shells overlap and weights are relatively distributed; however, as sampling enters low-noise stages, shells shrink until they no longer intersect. A single training sample then monopolizes the score function, and the sampling trajectory is pulled towards it—this is the geometric root of memorization. This analysis provides a clear target for smoothing: to generalize, one must prevent single-point dominance in low-noise stages.

**2. Noise Unconditioning: Allowing Each Training Sample to Speak via Its Optimal Shell**

The first intervention targets the training side. Standard diffusion score functions $s_\theta(x, \sigma_i)$ use the noise level as an input condition. The problem is that sampling points may not fall on the optimal shells of most training samples; weights for samples whose "shells don't match" are suppressed and their contributions erased. The unconditioning approach removes this condition, letting the network learn only $s_\theta(x)$. This is equivalent to merging a series of distributions sliced by noise levels into a single Gaussian mixture $p_{\text{MN}}(x)$ with $M \times N$ terms and performing score matching on it—thus, sampling can be reinterpreted as gradient ascent on $\log p_{\text{MN}}(x)$. Training points remain optimal solutions, but because every sample can now participate via its own optimal shell, single-point dominance is postponed, the collapse time is delayed, and the generalization window is extended. Implementation is subtractive: the loss $\mathcal{L}_u$ is identical to standard diffusion, with the only difference being that noise is no longer fed to the network; since the noise level input is gone, preset step sizes become unreliable and are replaced by adaptive step sizes $\alpha \sigma_{n*}^2$.

**3. Temperature Smoothing: Controlling Weight Sharpness via Softmax Temperature**

The second intervention targets the sampling side, specifically the high-risk low-noise interval. Since sharpness comes from the softmax, a temperature is introduced: $w_j^*(x;T) = \frac{\exp(f/T_j^*)}{\sum_l \exp(f/T_l^*)}$. A threshold $\sigma_{\text{collapse}}$ is set, and temperature scaling is enabled only when $\sigma_i \leq \sigma_{\text{collapse}}$ (when shells are about to separate and collapse is imminent). By setting $T_i = \sigma_{\text{collapse}}/\sigma_i$—the smaller the noise, the higher the temperature and the flatter the smoothing. Increasing temperature lowers the dominance ratio $a$ and reduces the expansion factor $\gamma_{ex}$, redistributing monopolized weights to neighbors. To save costs, the score function is approximated using only top-K nearest neighbor samples; KNN must be performed in feature space rather than pixel space, as the lower local curvature of feature space allows neighbors to better fit the true manifold—a point validated by experiments where pixel-space FID collapsed to 50.81. Thus, temperature becomes a tunable knob providing a continuous trade-off between generalization and generation quality.

### Loss & Training
Two methods are adaptively spliced according to noise levels: when $\sigma_i > \sigma_{\text{collapse}}$ (shells still overlap), the Unconditioning loss $\mathcal{L}_u$ is used; when $\sigma_i \leq \sigma_{\text{collapse}}$ (shells about to separate), it switches to the Temperature loss $\mathcal{L}_T$. The overall framework follows VE-SDE, with the Unconditioning branch simply removing the time embedding layer while keeping the rest of the network structure unchanged.

## Key Experimental Results

### Main Results

| Dataset | Method | FID(G,Train) | FID(G,Test) | Description |
|--------|------|-------------|------------|------|
| CIFAR-10 | Conditioning (baseline) | 6.49 | 6.56 | Standard VE-SDE |
| CIFAR-10 | Unconditioning | 7.33 | 7.34 | FID rises slightly but generalization increases |
| CIFAR-10 | Temp T=7/σ, K=100 (feat) | 7.96 | 7.98 | Pixel-space KNN collapses (50.81) |
| CelebA | Conditioning | 7.25 | 7.81 | Standard VE-SDE |
| CelebA | Unconditioning | 7.07 | 7.34 | FID actually decreases |
| CelebA | Temp T=10/σ, K=100 (feat) | 8.40 | 8.19 | Feature KNN significantly superior to pixel KNN |

### Ablation Study

| Configuration | Expansion Factor $\gamma_{ex}$ | Description |
|------|-------------|------|
| Empirical Score Function (Conditioning) | ~10³ (at low noise) | Extremely sharp, memorization |
| NN-learned Score Function | ~1-2 | Implicit smoothing |
| Unconditioning (Empirical) | Medium | Smoother than Conditioning |
| Temperature T=10 | ~1 | Explicit smoothing is effective |
| Temperature T=100 | <1 | Near non-expansion |
| Temperature T=1000 | ≈1 | Extremely smooth |

### Key Findings
- **Feature-space KNN consistently outperforms pixel-space KNN**: On CIFAR-10 with T=7/σ and K=100, pixel-space FID collapsed to 50.81 while feature-space reached only 7.96, validating the theory that "lower curvature in local manifolds aids smoothing."
- **Unconditioning improves FID on CelebA** (7.07 vs 7.25), suggesting that smoothing does not necessarily harm quality and can have positive effects.
- **ODE samplers fail under Unconditioning** because the mismatch between preset noise levels and reality causes step size explosions; the stochastic term in SDE samplers provides a self-correction mechanism to maintain stability.

## Highlights & Insights
- The **shell geometry intuition** is elegant: it transforms abstract mathematical analysis of high-dimensional Gaussian mixtures into a geometric picture of "whether thin shells overlap," making the essence of memorization intuitive. This geometric thinking is transferable to other problems involving Gaussian mixtures.
- The **unified distribution perspective** is the biggest "aha" moment: Noise Unconditioning reinterprets the stepwise denoising of diffusion models as gradient ascent on a fixed objective function. This not only explains generalization but also opens possibilities for applying constraints (e.g., physical law constraints in video generation) via projected gradient methods.
- **Temperature Smoothing as a plug-and-play method** adds almost no extra cost and can be directly applied to existing diffusion frameworks, offering high engineering utility.

## Limitations & Future Work
- Experiments are primarily based on the VE-SDE framework and have not verified effects on more mainstream VP-SDE and Flow Matching frameworks.
- Temperature Smoothing requires KNN queries, incurring extra overhead on large-scale datasets.
- Selection of temperature parameters and $\sigma_{\text{collapse}}$ requires tuning; automated strategies are missing.
- Not yet extended to Latent Diffusion Models, which the authors identify as an important future direction.
- Future work could explore combining this framework with new paradigms like Consistency Models or Rectified Flow.

## Related Work & Insights
- **vs. Carlini et al. (2023) on memorization detection**: They prove the existence of memorization from an attack perspective; Ours explains the root cause and proposes mitigation from a theoretical perspective, making them complementary.
- **vs. Bonnaire et al. (2025) on implicit regularization**: That work studies how implicit regularization in training dynamics prevents memorization; Ours focuses on the structural analysis of the score function, providing a more direct geometric explanation and explicit intervention methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes a complete theoretical framework starting from the softmax structure of score function weights; unique and elegant perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical verification is sufficient, but experimental scale is limited (primarily on small datasets).
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations, clear geometric intuition, and smooth logical flow.
- Value: ⭐⭐⭐⭐ Provides deep theoretical insights into diffusion model generalization; practical methods are simple and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Smoothing the Score Function to Enhance Generalization in Diffusion Models](smoothing_the_score_function_to_enhance_generalization_in_diffusion_models.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] Visual Diffusion Models are Geometric Solvers](visual_diffusion_models_are_geometric_solvers.md)
- [\[ICML 2025\] Towards a Mechanistic Explanation of Diffusion Model Generalization](../../ICML2025/image_generation/towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)

</div>

<!-- RELATED:END -->
