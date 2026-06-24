---
title: >-
  [Paper Note] HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration
description: >-
  [ICML 2025][Image Restoration][Diffusion Transformer] This work proposes the HarmoniCa framework, which addresses the misalignment between training and inference in existing learning-based feature caching methods through two core designs: Step-Wise Denoising Training (SDT) and Image Error Proxy-Guided Objective (IEPO). It achieves over a 40% reduction in latency (2.07× theoretical speedup) without compromising generation quality across 8 different models including PixArt-α.
tags:
  - "ICML 2025"
  - "Image Restoration"
  - "Diffusion Transformer"
  - "Feature Caching"
  - "Inference Acceleration"
  - "Training-Inference Alignment"
  - "Denoising Training"
date: 2026-05-08
content_hash: ef643ed775605301
---

# HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration

**Conference**: ICML 2025  
**arXiv**: [2410.01723](https://arxiv.org/abs/2410.01723)  
**Code**: [https://github.com/YSGuoUST/HarmoniCa](https://github.com/YSGuoUST/HarmoniCa)  
**Area**: Image Restoration  
**Keywords**: Diffusion Transformer, Feature Caching, Inference Acceleration, Training-Inference Alignment, Denoising Training

## TL;DR
This work proposes the HarmoniCa framework, which addresses the misalignment between training and inference in existing learning-based feature caching methods through two core designs: Step-Wise Denoising Training (SDT) and Image Error Proxy-Guided Objective (IEPO). It achieves over a 40% reduction in latency (2.07× theoretical speedup) without compromising generation quality across 8 different models including PixArt-α.

## Background & Motivation

**Background**: Diffusion Transformers (DiTs) demonstrate outstanding performance in image generation tasks, but suffer from extremely high inference costs. Each generation process requires multiple denoising steps, each of which involves a full forward pass through the entire Transformer, posing significant challenges for practical deployment.

**Limitations of Prior Work**: Feature caching is an effective acceleration strategy that stores and reuses redundant computed features between neighboring denoising steps, thereby skipping the computation of certain layers. Existing learning-based caching methods (such as Learning-to-Cache) train a lightweight router to adaptively decide which layers can be cached.

However, existing learning-based caching approaches suffer from two crucial misalignment issues:
   - **(a) Neglecting the influence of previous steps**: During training, each step is optimized independently. However, during inference, the caching decisions from previous steps alter the intermediate features, affecting the optimal caching strategy of the current step. This "step disconnection" between training and inference leads to suboptimal cache routing.
   - **(b) Objective misalignment**: The training process optimizes the alignment of predicted noise (i.e., $\epsilon$-prediction loss), whereas the ultimate goal is to generate high-quality images. Small discrepancies at the noise level can be magnified at the image level (or vice-versa), preventing the caching strategy from truly serving image quality.

**Goal**: How can the training process of the cache router be aligned with the actual inference process? How can the optimization objective be directly associated with the final image quality?

**Key Insight**: Core Observation: Training should simulate the step-by-step denoising process of inference (rather than training each step independently), and error signals from the image domain should be used to guide caching decisions.

**Core Idea**: Building a better feature cache by "harmonizing" training and inference—SDT allows training to observe the realistic intermediate states during inference, while IEPO shifts the optimization objective from the noise space to the image space.

## Method

### Overall Architecture
HarmoniCa is a learning-based feature caching framework applicable to various DiT architectures. The overall pipeline is as follows:
- **Input**: Pre-trained DiT model (frozen parameters) + noisy images
- **Learnable Components**: A lightweight cache router (which decides the caching/computation strategy for each layer)
- **Training Phase**: The router is trained via Step-Wise Denoising Training (SDT) using the Image Error Proxy-Guided Objective (IEPO) as the training target.
- **Inference Phase**: The router makes caching decisions for each layer at each denoising step—if a cache hit occurs, features from the previous step are reused; otherwise, standard computation is performed.
- **Output**: Accelerated high-quality generated images.

The core mechanism is to make the training process as close to the inference process as possible, allowing the router to "experience" cumulative error scenarios during training that it will encounter during inference.

### Key Designs

1. **Step-Wise Denoising Training (SDT)**:

    - **Function**: Mimics the complete step-wise denoising process during the training of the cache router, rather than training each timestep independently.
    - **Mechanism**: Traditional methods randomly sample a timestep $t$ during training and compute the caching loss for that step independently. In contrast, SDT lets the model denoise step-by-step from $t_T$ to $t_0$, where each step takes the actual output of the previous step (which contains approximation errors introduced by caching) as its input. This enables the router to perceive the caching effects of previous steps.
    - **Design Motivation**: During inference, the input at step $t$ is the output of step $t+1$. If step $t+1$ uses approximated features due to caching, the input for step $t$ already deviates from the ideal trajectory of "no caching". SDT trains the router to make sensible decisions under such deviations, bridging the step continuity gap between training and inference.
    - **Differences from Prior Work**: Methods like Learning-to-Cache train each step independently, assuming that the input at each step is a clean, error-free signal. SDT, however, exposes the training process to realistic scenarios with accumulated errors.

2. **Image Error Proxy-Guided Objective (IEPO)**:

    - **Function**: Approximates the final image error using an efficient proxy function to guide caching decisions instead of the traditional noise prediction loss.
    - **Mechanism**: Direct computation of image errors is too costly (requiring the execution of the entire denoising chain). IEPO designs a lightweight proxy to approximate errors in the image domain. Specifically, utilizing the known mathematical relationships between noise prediction and the final image during the denoising process (analytical formulas of DDPM/DDIM), errors in the $\epsilon$-space are mapped to the image space via scaling factors for evaluation. This allows the step-wise optimization objective $\mathcal{L}_{\text{IEPO}}$ to directly reflect the impact on final image quality.
    - **Design Motivation**: The error distributions in the noise space and image space differ—some layers show small errors in the noise space but have a significant impact in the image space (and vice versa). IEPO ensures that the router prioritizes caching layers that have the least impact on image quality, rather than those with the smallest error solely in the noise space.
    - **Additional Advantage**: IEPO is image-free (it does not require ground-truth images) and only depends on intermediate variables from the denoising process itself. Consequently, it reduces training time by approximately 25% compared to methods that require clean image supervision.

3. **Cache Router Design**:

    - **Function**: Decides whether to "compute" or "retrieve from cache" for each DiT layer at each denoising step.
    - **Mechanism**: The router is a lightweight network that takes feature statistics of the current step as input and outputs binary decisions for each layer (enabled by differentiable training via Gumbel-Softmax). Under a given computational budget (cache ratio), the router learns the optimal cache distribution strategy.
    - **Design Motivation**: Redundancy levels vary across different layers and denoising stages—some layers exhibit rapid changes in early steps (and should not be cached), while the same layers remain almost unchanged in later steps (safely cacheable). The router learns to adaptively discover these patterns.

### Loss & Training
- Freeze pre-trained DiT parameters, training only the lightweight router.
- Joint training combining SDT step-wise training and the IEPO objective function.
- Supports adjustable computational budgets (cache ratios), allowing a flexible trade-off between speed and quality.
- Image-free training: No ground-truth image dataset is required; training can be performed solely using random noise.
- High training efficiency: Reduces training time by approximately 25% compared to Learning-to-Cache.

## Key Experimental Results

### Main Results

Extensive validation is conducted across **8 models**, **4 samplers**, and resolutions ranging from $256 \times 256$ to $2K$.

| Model | Method | Speedup | Image Quality | Note |
|------|------|--------|----------|------|
| PixArt-α | No Cache (Baseline) | 1.0× | Baseline | Full computation |
| PixArt-α | Learning-to-Cache | ~1.5× | Slight degradation | Step discontinuity |
| PixArt-α | **HarmoniCa** | **2.07×** | **Comparable / Improved** | >40% latency reduction |
| DiT ($256 \times 256$) | No Cache | 1.0× | Baseline | — |
| DiT ($256 \times 256$) | **HarmoniCa** | ~1.8× | Lossless | Low-resolution validation |
| High-resolution model ($2K$) | No Cache | 1.0× | Baseline | — |
| High-resolution model ($2K$) | **HarmoniCa** | >1.5× | Lossless | High-resolution generalization |

### Ablation Study

| Configuration | Image Quality Change | Acceleration Efficiency | Note |
|------|-------------|----------|------|
| Full (SDT + IEPO) | Optimal | 2.07× | Full HarmoniCa |
| w/o SDT (Independent step training) | Obvious degradation | ~1.8× | Suboptimal routing due to lack of step continuity |
| w/o IEPO (Using noise loss) | Moderate degradation | ~1.9× | Objective misalignment, improper caching of some layers |
| w/o SDT & IEPO | Significant degradation | ~1.5× | Degenerates to Learning-to-Cache |
| Different cache ratios (20%-60%) | Smooth degradation | 1.3×-2.5× | Flexible speed-quality trade-off |

### Key Findings
- **SDT contributes more than IEPO**: Removing SDT leads to a more severe drop in quality than removing IEPO, indicating that the alignment of training and inference steps is the core bottleneck.
- **Image-free training is not only quality-preserving but also faster**: The proxy mechanism of IEPO avoids the pre-processing overhead of real images, saving 25% in training time compared to Learning-to-Cache.
- **High robustness across models, samplers, and resolutions**: Consistently effective across 8 models and 4 samplers, demonstrating strong generalizability.
- **More pronounced advantages at high cache ratios**: When the cache ratio is high (>50%), HarmoniCa's advantage over the baseline is magnified, as cache routing decisions become more critical, amplifying the alignment benefits of SDT+IEPO.

## Highlights & Insights
- **The concept of training-inference alignment** is highly generalizable: it applies not only to feature caching but also to any scenario with training-inference mismatch (such as exposure bias in autoregressive generation), where the step-wise training strategy of SDT can be adapted.
- **The proxy design of IEPO** is exceptionally clever: it avoids the high cost of directly computing image errors by exploiting the mathematical structure of diffusion models (the analytical relationship between $x_0$ and $\epsilon$) to construct a zero-cost lookup for image error approximation.
- **Image-free training** is a practical highlight: it reduces the barrier of data preparation and cuts down training time, making it highly deployment-friendly.
- **2.07× speedup with negligible quality loss** is highly outstanding in DiT acceleration literature, as most other methods suffer from severe quality degradation at this speedup ratio.

## Limitations & Future Work
- **Cache overhead is not discussed**: Storing intermediate features requires additional GPU memory, which could become a bottleneck when the batch size is large or the model has many layers.
- **Validation limited to image generation**: Although the framework is theoretically applicable to video DiTs (like Sora), the paper does not validate video generation scenarios.
- **Router generalizability**: The router is trained for specific models. Changing the DiT architecture requires retraining, indicating a lack of cross-model transferability.
- **Combination with other acceleration methods**: The combination with other acceleration techniques such as quantization, distillation, and token pruning remains unexplored.
- **Accuracy upper bound of the IEPO proxy**: The proxy function is ultimately an approximation; under extreme cache ratios (e.g., >70%), the proxy error may accumulate.

## Related Work & Insights
- **vs Learning-to-Cache**: Learning-to-Cache optimizes each step independently using noise loss. HarmoniCa simultaneously addresses both step continuity (SDT) and objective alignment (IEPO), outperforming it across the board.
- **vs AsymRnR**: AsymRnR is a training-free token pruning method, which is complementary to HarmoniCa's learning-based caching. The former reduces computation per layer, while the latter skips layers entirely, indicating great potential for combination.
- **vs Knowledge Distillation (e.g., Progressive Distillation)**: Distillation reduces the number of denoising steps, whereas caching reduces the computation per step. These are orthogonal acceleration directions that can be stacked.
- **Relationship with Consistency Models**: Consistency Models bypass multi-step denoising via single-step generation, whereas HarmoniCa preserves the multi-step framework but accelerates each step, making it more suitable for scenarios requiring high-quality or controllable generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The formulations of SDT and IEPO directly target the key bottlenecks of existing methods. The idea of aligning training and inference has broad inspirative value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation on 8 models × 4 samplers × multiple resolutions, with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with smooth logical flows deriving the motivations of SDT and IEPO.
- Value: ⭐⭐⭐⭐⭐ Achieving 2× speedup with negligible quality loss alongside efficient training directly drives the practical deployment of DiTs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICML 2025\] TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [\[ICML 2026\] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](../../ICML2026/image_restoration/dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)
- [\[ECCV 2024\] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](../../ECCV2024/image_restoration/efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](../../ICLR2026/image_restoration/breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)

</div>

<!-- RELATED:END -->
