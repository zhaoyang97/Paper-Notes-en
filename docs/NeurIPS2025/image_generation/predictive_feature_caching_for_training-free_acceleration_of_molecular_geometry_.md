---
title: >-
  [Paper Note] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation
description: >-
  [NeurIPS 2025][Image Generation][molecular geometry generation] This paper transfers predictive feature caching strategies from the image generation domain to molecular geometry generation, exploiting the temporal smoothness of hidden states along sampling trajectories to achieve training-free 2–3× inference acceleration, with up to 7× speedup when combined with other optimization techniques.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "molecular geometry generation"
  - "flow matching"
  - "feature caching"
  - "training-free acceleration"
  - "SE(3) equivariance"
date: 2026-05-08
content_hash: 071204c8c8a6bd1b
---

# Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.04646](https://arxiv.org/abs/2510.04646)  
**Code**: None  
**Area**: Molecular Generation / Inference Acceleration
**Keywords**: molecular geometry generation, flow matching, feature caching, training-free acceleration, SE(3) equivariance

## TL;DR

This paper transfers predictive feature caching strategies from the image generation domain to molecular geometry generation, exploiting the temporal smoothness of hidden states along sampling trajectories to achieve training-free 2–3× inference acceleration, with up to 7× speedup when combined with other optimization techniques.

## Background & Motivation

Flow matching models represent the current state of the art in molecular geometry generation, yet inference requires hundreds of neural network forward passes, imposing substantial computational cost. In drug discovery pipelines, where 500K to over 1M molecular candidates must be generated, the inference time of the generator becomes the primary bottleneck.

Existing acceleration methods—trajectory reparameterization, progressive distillation, and latent-space approaches—all require additional training, incurring extra data and compute overhead. This paper pursues a complementary direction: a **training-free** acceleration scheme inspired by feature caching techniques from image generation:

- Intermediate activations at adjacent time steps vary smoothly during ODE solving.
- These intermediate features can be cached and predicted, avoiding full forward passes.
- This direction has not been explored in the molecular domain.

## Method

### Overall Architecture

Molecular geometry generation is formulated via conditional flow matching (CFM), which learns a time-dependent vector field $v_\theta(x_t, t)$ to transport a noise distribution to the data distribution. Molecules are parameterized as $x = (c, a, b)$, representing coordinates, atom types, and bond orders, respectively. Sampling proceeds via Euler discretization:

$$x_{k+1} = x_k + \Delta t_k \, v_\theta(x_k, t_k)$$

The network backbone consists of $L$ blocks: $g_L \circ \cdots \circ g_1$. Since the ODE right-hand side is continuous and the network is continuous in $(x, t)$, intermediate activations vary smoothly over time. This paper exploits such smoothness by applying predictive caching only to the final block $L$.

### Key Designs

**TaylorSeer Caching**: Full forward passes are performed at designated checkpoint steps (every $D$ steps), and the following quantities are cached:

$$C(x_t) = \{F(x_t), \Delta F(x_t), \dots, \Delta^m F(x_t)\}$$

For intermediate steps within a window, an $m$-th order Taylor predictor is used:

$$F_{\text{pred},m}(x_{t+k}) = F(x_t) + \sum_{i=1}^{m} \frac{\Delta^i F(x_t)}{i! D^i} (-k)^i$$

Setting $m=0$ reduces to naive caching (direct reuse), $m=1$ yields linear prediction, and $m=2$ yields quadratic prediction.

**Adams–Bashforth (AB) Caching**: A $j$-step AB linear multistep recurrence is used for prediction:

$$F_{\text{AB}(j)}(x_{t+k}) = \sum_{i=1}^{j} (-1)^{i+1} \binom{j}{i} F(x_{t+k+i})$$

The most recent $j$ cached outputs are used to predict the current output.

**Equivariance Preservation**: The caching operation consists of linear combinations and finite differences of scalar time coefficients, which commute with the group action $G = E(3) \times S_N$. If the base density is $G$-invariant and the vector field is $G$-equivariant, the predicted evaluations preserve equivariance and the terminal density preserves invariance.

### Loss & Training

The proposed method is **entirely training-free**, applied directly to the pretrained SemlaFlow model. No modification of model weights is required; caching logic is introduced solely at inference time. The implementation ensures that the final step always performs a full computation regardless of the caching interval $D$, guaranteeing output quality.

## Key Experimental Results

### Main Results — GEOM-Drugs Dataset

| Steps | Method | Mol. Stability↑ | Validity (PRC)↑ | Energy/Atom↓ | Strain/Atom↓ | Throughput↑ |
|-------|--------|----------------|-----------------|--------------|--------------|-------------|
| 100 | Base | 0.98 | 0.88 | 2.38 | 1.50 | 11.4 |
| 51 | Base | 0.98 | 0.86 | 2.51 | 1.63 | 21.9 |
| 51 | Taylor $m=2$ | 0.98 | **0.86** | **2.25** | **1.46** | 22.1 |
| 51 | AB $j=3$ | 0.98 | **0.87** | **2.15** | **1.40** | 22.1 |
| 34 | Base | 0.97 | 0.85 | 2.62 | 1.78 | 32.2 |
| 34 | Taylor $m=2$ | 0.97 | 0.83 | **2.25** | **1.53** | 32.4 |
| 34 | AB $j=3$ | 0.97 | **0.85** | **2.25** | **1.51** | 32.1 |
| 26 | Base | 0.97 | 0.82 | 2.69 | 1.85 | 41.0 |
| 26 | AB $j=3$ | 0.96 | **0.82** | **2.30** | **1.60** | 41.2 |

### Combined Acceleration — Stacking with Orthogonal Optimizations

| Acceleration Combination | Inference Time (10K molecules) | Speedup |
|--------------------------|-------------------------------|---------|
| Base (100 steps) | ~14 min | 1× |
| Caching only (AB, $D=2$) | ~4.7 min | ~3× |
| Caching + torch.compile | ~3 min | ~4.5× |
| Caching + compile + TF32 | **~2 min** | **~7×** |

### Key Findings

1. **Quality-equivalent 2× speedup**: At 51 steps with caching ($D=2$), all quality metrics match or exceed the 100-step baseline while doubling throughput.
2. **Caching outperforms naive step reduction**: Directly reducing to 51 steps significantly degrades quality, whereas caching at the same step count maintains or improves quality (lower energy and strain).
3. **AB caching outperforms Taylor caching**: AB $j=3$ consistently outperforms Taylor $m=2$ across all configurations.
4. **Orthogonal stacking with compilation**: Caching speedups are fully composable with torch.compile and TF32 kernels, yielding a combined 7× acceleration.
5. **Full equivariance preservation**: Both theoretical proof and empirical validation confirm that caching does not violate SE(3) equivariance.

## Highlights & Insights

- **Successful cross-domain transfer**: Feature caching acceleration techniques from image/video diffusion models are successfully adapted to molecular SE(3)-equivariant architectures.
- **Plug-and-play without training**: The method directly improves the inference efficiency of existing pretrained models without fine-tuning or retraining.
- **Counterintuitive quality improvement**: Caching not only preserves but actually improves energy and strain metrics, possibly because the smoothing effect acts as a form of implicit regularization.
- **Strong practical impact**: Reducing the time to generate 10K molecules from 14 minutes to 2 minutes has direct implications for large-scale sampling in drug discovery.

## Limitations & Future Work

- Validation is primarily conducted on SemlaFlow; generalization to other molecular generation models (e.g., GeoDiff, MDM) remains to be confirmed.
- The choice of caching interval $D$ and predictor order lacks an adaptive mechanism (approaches analogous to TeaCache could be introduced).
- Evaluation is limited to GEOM-Drugs and QM9; assessment on larger-scale and more complex molecules warrants further investigation.
- Peak memory usage increases slightly, though the increment is modest.

## Related Work & Insights

- Feature caching has been extensively studied in image generation (DeepCache, FORA, TaylorSeer, AB-Cache, TeaCache); this paper represents the first transfer to the molecular domain.
- The proposed method is orthogonal and complementary to training-based acceleration approaches (trajectory reparameterization, distillation, latent-space methods).
- Insight: any model based on multi-step iterative generation, provided that intermediate features exhibit smoothness, is a candidate for predictive caching acceleration.

## Rating

- Novelty: ⭐⭐⭐⭐ (first transfer of feature caching to molecular generation)
- Technical Depth: ⭐⭐⭐⭐ (rigorous theoretical analysis of equivariance preservation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (systematic ablation and combination experiments)
- Practicality: ⭐⭐⭐⭐⭐ (training-free, plug-and-play, with substantial real-world speedup)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](../../CVPR2025/image_generation/dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[CVPR 2026\] ResCa: Residual Caching for Diffusion Transformers Acceleration](../../CVPR2026/image_generation/resca_residual_caching_for_diffusion_transformers_acceleration.md)
- [\[ICCV 2025\] MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics](../../ICCV2025/image_generation/mosaicdiff_training-free_structural_pruning_for_diffusion_model_acceleration_ref.md)

</div>

<!-- RELATED:END -->
