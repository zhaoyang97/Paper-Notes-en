---
title: >-
  [Paper Note] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint
description: >-
  [ECCV 2024][Image Restoration][Adverse Weather Restoration] This paper proposes T3-DiffWeather, a diffusion-based all-in-one adverse weather restoration framework. It utilizes a prompt pool to allow the network to autonomously combine sub-prompts to construct instance-level weather-prompts for modeling diverse weather degradations. Concurrently, it leverages Depth-Anything features to constrain general prompts to model scene information. The method achieves state-of-the-art (…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Adverse Weather Restoration"
  - "Prompt Pool"
  - "Depth-Anything"
  - "diffusion model"
  - "All-in-One Restoration"
date: 2026-05-08
content_hash: 4a499f57576ddef1
---

# Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint

**Conference**: ECCV 2024  
**arXiv**: [2409.15739](https://arxiv.org/abs/2409.15739)  
**Code**: [https://github.com/Ephemeral182/T3-DiffWeather](https://github.com/Ephemeral182/T3-DiffWeather)  
**Area**: Other  
**Keywords**: Adverse Weather Restoration, Prompt Pool, Depth-Anything, diffusion model, All-in-One Restoration

## TL;DR

This paper proposes T3-DiffWeather, a diffusion-based all-in-one adverse weather restoration framework. It utilizes a prompt pool to allow the network to autonomously combine sub-prompts to construct instance-level weather-prompts for modeling diverse weather degradations. Concurrently, it leverages Depth-Anything features to constrain general prompts to model scene information. The method achieves state-of-the-art (SOTA) performance in only 2 sampling steps, with a computational cost of only 1/52 of WeatherDiffusion.

## Background & Motivation

Adverse weather image restoration (dehazing, deraining, desnowing, deraindropping) is an important direction in image restoration. Weather degradation in the real world is characterized by **unpredictability and compositionality**—a single scene may simultaneously suffer from multiple degradations such as haze, rain, and snow, with each degradation varying in severity and morphology.

**Limitations of Prior Work**:

**Early all-in-one methods** (TransWeather, All-in-One) use shared queries or NAS, failing to explicitly model the similarities and differences among different weather degradations, making it difficult to adaptively handle unseen weather combinations.

**WeatherDiffusion** first introduced diffusion to weather restoration and achieved SOTA, but directly using the degraded image as a condition provides insufficient information, and it requires 25 sampling steps, leading to slow inference.

**PromptIR** uses shared learnable prompts to adapt to different degradations, but the shared parameters cause interference among different degradations while ignoring instance-level degradation differences.

**Lack of scene modeling**: Existing methods focus solely on degradation understanding, ignoring the guiding role of scene information beneath the degraded image for reconstruction.

**Key Challenge**: How to flexibly model both the differences and commonalities of diverse weather degradations within a unified model, while fully utilizing scene prior information to guide diffusion denoising?

**Core Idea**: "Teaching Tailored to Talent"—utilizing a prompt pool to allow the network to autonomously select sub-prompts to construct weather-prompts tailored to specific degradations, and using general prompts constrained by Depth-Anything to provide scene conditions. These two types of prompts serve as conditions to guide the diffusion process in reconstructing degradation residuals.

## Method

### Overall Architecture

The pipeline of T3-DiffWeather: Input degraded image $\mathbf{y}$ → Calculate degradation residual $\mathbf{r}_d = \mathbf{x} - \mathbf{y}$ (during training) → Set the residual as the reconstruction target for diffusion → Use two types of prompts as conditions to inject into the latent layers of the diffusion network via cross-attention → Perform DDIM with 2 sampling steps to obtain the predicted residual $\mathbf{r}_d^{sample}$ → Restore the clean image as $\hat{\mathbf{x}} = \mathbf{r}_d^{sample} + \mathbf{y}$.

Key innovation: **Reconstructing the degradation residual instead of the clean image**. This is because t-SNE visualization shows that the degradation residual is more discriminative than the background, and the degradation itself is the main bottleneck for restoration.

### Key Designs

1. **Prompt Pool Construction of Weather-Prompts**:

    - Function: Design a prompt pool $\mathcal{P} = \{\mathcal{P}_s^i\}_{i=1}^N$ containing $N=20$ sub-prompts, where each sub-prompt $\mathcal{P}_s^i \in \mathbb{R}^{L_s \times D}$ ($L_s=64$ tokens). The network autonomously selects the top-$k$ ($k=5$) most relevant sub-prompts based on the input degraded image, concatenating them to construct the weather-prompt $\mathcal{P}_w$.
    - Mechanism: Each sub-prompt is paired with a learnable key $\mathcal{K}_s^i \in \mathbb{R}^{1 \times D}$. The degradation residual embedding $\mathcal{F}_e$ is processed by spatial mean pooling to obtain $\mathcal{F}_e^{mean} \in \mathbb{R}^{1 \times D}$. Its cosine similarity $\delta(\mathcal{K}_s^i, \mathcal{F}_e)$ with each key is computed, and the top-$k$ most similar sub-prompts are selected and concatenated: $\mathcal{P}_w = \bigcup_{i=1}^k \mathcal{K}_s^i$, where $\delta(\mathcal{K}_s^i, \mathcal{F}_e) \geq \delta(\mathcal{K}_s^{i+1}, \mathcal{F}_e)$.
    - Design Motivation: Different weather degradations share both **commonalities** (e.g., fog occlusion, reduced contrast) and **differences** (e.g., rain streaks vs. snow particles). The prompt pool allows the network to capture commonalities by sharing certain sub-prompts (e.g., rain and raindrops sharing similar activation frequencies for some sub-prompts) while modeling differences using independent sub-prompts.
    - t-SNE visualization validation: Weather-prompts constructed under different weather conditions not only maintain their respective clustering characteristics but also exhibit overlap across different weathers.

2. **Depth-Anything Constrained General Prompts**:

    - Function: Design a set of general prompts $\mathcal{P}_g \in \mathbb{R}^{L_g \times D}$ ($L_g=256$ tokens), which interact with the intermediate features $\mathcal{F}_d$ of Depth-Anything via cross-attention to obtain scene-aware $\mathcal{P}_{gd} = \text{softmax}(\frac{\mathcal{Q}_g \mathcal{K}_d^T}{\sqrt{\mathcal{D}}}) \mathcal{V}_d$.
    - Mechanism: t-SNE visualization shows that scene features of clean images share commonalities in the latent space (being distinctly different from degradation features). Depth-Anything remains robust in depth estimation even under extreme weather degradations. Its intermediate layer features possess **degradation invariance**, allowing them to reliably represent scene structures.
    - Design Motivation: The robustness of intermediate features from DINO, DINOv2, and Depth-Anything was compared, with Depth-Anything yielding the best results (as it was trained for depth estimation on large-scale datasets on top of DINOv2, gaining additional robustness against degradations).
    - Architecture selection: Depth-Anything ViT-S (only 115 MB) is used to achieve the best balance between performance and memory usage.

3. **Two-Stage Cross-Attention Injection**:

    - Function: Inject weather-prompts and general prompts into the latent features of the diffusion network in two separate stages via cross-attention.
    - Formulation: $\mathcal{F}_e' = \text{CA}(\mathcal{F}_e, \mathcal{P}_w)$, $\hat{\mathcal{F}_e} = \text{CA}(\mathcal{F}_e', \mathcal{P}_{gd})$
    - Design Motivation: Similar to the injection of text embeddings in Stable Diffusion, degradation information is processed first, followed by the integration of scene information, which is natural and efficient.

4. **Contrastive Prompt Loss**:

    - Function: Constrain the representations of both prompt types—the keys of weather-prompts should be far from the keys of general prompts (serving as mutual negatives), while the keys of general prompts should be close to Depth-Anything features (positives).
    - Formulation: $\mathcal{L}_{cp} = \frac{1}{b} \frac{1}{k} \sum_{j=1}^b \sum_{i=1}^k [\gamma(\mathcal{K}_{gd}, \mathcal{F}_d^{mean}) - \gamma(\mathcal{K}_s^i, \mathcal{K}_{gd})]$, where $\gamma(\cdot) = 1 - \delta(\cdot)$.
    - Design Motivation: The two types of prompts have fundamentally different design goals (degradation modeling vs. scene modeling) and naturally act as negative pairs for each other, eliminating the need to construct additional negative samples.

5. **Degradation Residual Reconstruction Objective**:

    - Function: Modify the reconstruction target of diffusion from the clean image $\mathbf{x}$ to the degradation residual $\mathbf{r}_d = \mathbf{x} - \mathbf{y}$.
    - Training objective: $\mathcal{L}_{res} = \mathbb{E}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}}(\mathbf{x}-\mathbf{y}) + \sqrt{1-\bar{\alpha}}\boldsymbol{\epsilon}, \mathbf{y}, \mathbf{c})\|_2^2$
    - Design Motivation: Residuals are easier to learn than full images because the information density in degraded regions is higher. Heatmap visualization confirms that the model indeed focuses on these degraded areas.

### Loss & Training

The total loss function consists of four items:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{res} + \lambda_2 \mathcal{L}_{cp} + \lambda_3 \|(\mathbf{r}_d^{sample} + \mathbf{y}) - \mathbf{x}\|_{psnr} + \lambda_4 \mathcal{L}_{cp}^{sample}$$

- $\mathcal{L}_{res}$: Noise estimation loss (residual reconstruction)
- $\mathcal{L}_{cp}$: Contrastive prompt loss (noise estimation stage)
- PSNR reconstruction loss: Explicit supervision on the sampled results
- $\mathcal{L}_{cp}^{sample}$: Contrastive prompt loss during the sampling stage
- All $\lambda$ values are set to 1

Training configuration: AllWeather dataset (18,069 images), Adam optimizer, initial learning rate of 1.5e-4, cosine annealing, 800K iterations, 2×A800 GPUs, DDIM sampling with 1000 timesteps during training / 2 steps during inference.

## Key Experimental Results

### Main Results

**Desnowing (Snow100K):**

| Method | Snow100K-S PSNR/SSIM | Snow100K-L PSNR/SSIM |
|------|---------------------|---------------------|
| Restormer | 35.03/0.9487 | 30.52/0.9092 |
| WeatherDiff64 | 35.83/0.9566 | 30.09/0.9041 |
| AWRCP | 36.92/0.9652 | 31.92/0.9341 |
| **T3-DiffWeather** | **37.51/0.9664** | **32.37/0.9355** |

**Deraining (Outdoor-Rain):**

| Method | PSNR | SSIM |
|------|------|------|
| Restormer | 29.97 | 0.9215 |
| WeatherDiff64 | 29.64 | 0.9312 |
| AWRCP | 31.39 | 0.9329 |
| **T3-DiffWeather** | **31.99** | **0.9365** |

**Deraindropping (RainDrop):**

| Method | PSNR | SSIM |
|------|------|------|
| UDR-S2Former | 32.64 | 0.9427 |
| AWRCP | 31.93 | 0.9314 |
| **T3-DiffWeather** | **32.66** | **0.9411** |

**Computational Efficiency Comparison (256×256 resolution):**

| Method | Parameters | GFLOPs |
|------|--------|--------|
| WeatherDiffusion | 113.68M | 248.4G × 25 steps |
| Refusion | 131.4M | 63.4G × 50 steps |
| **T3-DiffWeather** | **69.38M** | **59.82G × 2 steps** |

### Ablation Study

**Prompt Pool Ablation (Outdoor-Rain):**

| Configuration | PSNR | SSIM | Description |
|------|------|------|------|
| W/o prompt pool | 31.05 | 0.9325 | baseline |
| W/o matched keys | 31.72 | 0.9349 | W/o selection mechanism |
| PromptIR-style shared prompt | 31.38 | 0.9330 | Shared parameter interference |
| **Prompt pool (ours)** | **31.99** | **0.9365** | Autonomously selects the best |

**General Prompts Constraint Ablation:**

| Configuration | PSNR | SSIM | Description |
|------|------|------|------|
| W/o General Prompts | 31.52 | 0.9342 | Lacks scene information |
| W/o Depth-Anything constraint | 31.67 | 0.9349 | W/o explicit constraint |
| DINO constraint | 31.77 | 0.9357 | Insufficient robustness |
| DINOv2 constraint | 31.82 | 0.9359 | Suboptimal |
| **Depth-Anything constraint** | **31.99** | **0.9365** | Best robustness |

**Contrastive Prompt Loss Ablation:**

| Configuration | PSNR | SSIM |
|------|------|------|
| W/o CPL | 31.71 | 0.9350 |
| W/o Negative $\gamma$ | 31.81 | 0.9359 |
| W/o Positive $\gamma$ | 31.77 | 0.9358 |
| **Full CPL** | **31.99** | **0.9365** |

### Key Findings

- **Prompt pool contributes the most**: Compared to having no prompt pool, it boosts PSNR by 0.94 dB (31.05 $\to$ 31.99) and outperforms the shared prompt approach of PromptIR.
- **Depth-Anything > DINOv2 > DINO**: The choice of scene constraint is crucial, with Depth-Anything exhibiting the strongest degradation invariance.
- **2 sampling steps are sufficient**: Since the conditional information is highly expressive and the reconstruction target is the residual (which is simpler than restoring the full image), only 2 DDIM sampling steps are required.
- **Sensitivity of pool size and top-$k$**: A pool size of 20 and top-$k$ of 5 represent the optimal balance. Excessively large pools introduce redundancy, while over-inflated $k$ values lead to overfitting.
- **ViT-S is sufficient**: Comparing Depth-Anything ViT-S (115MB) vs. ViT-L (1314MB), the difference in PSNR is only 0.07 dB.

## Highlights & Insights

- **"Teaching Tailored to Talent" prompt design philosophy**: Weather-prompts and general prompts separately model degradation and context. With distinct responsibilities, they are prevented from being conflated via the contrastive loss.
- **Reconstructing residuals instead of clean images**: This seemingly simple shift enables diffusion with only 2 sampling steps, slashing computational complexity to 1/52 of SOTA, which is highly practical.
- **Leveraging the degradation-invariant features of Depth-Anything**: This is the first work to utilize intermediate features of a depth estimation model as scene constraints for image restoration tasks, presenting a highly novel perspective.
- **The visualization of sub-prompt activation frequencies** provides excellent interpretability—revealing which attributes are shared across different weathers (e.g., similar selection frequencies for rain and raindrops) and which are unique.
- **Transferable concept**: The mechanism of prompt pool + key matching can be generalized to other all-in-one models that handle multiple degradations or tasks.

## Limitations & Future Work

- **Requires paired data during training**: Reliance on clean-degraded image pairs limits training in unconstrained real-world scenarios.
- **Depth-Anything as an extra dependency**: Although the ViT-S variant is used, it still introduces extra model complexity and inference overhead.
- **Evaluated solely on weather degradations**: The method has not been extended to other types of image degradations (such as noise, blur, or compression artifacts).
- **Fixed prompt pool size**: A dynamically growing prompt pool remains unexplored.
- **Limited real-world evaluation**: The PSNR improvements on real-world datasets are relatively modest (~0.6 dB).

## Related Work & Insights

- **vs. WeatherDiffusion**: Both are diffusion-based, but WeatherDiffusion uses the degraded image directly as a condition, which lacks rich information and requires 25 sampling steps. Through rich prompt conditions and a residual reconstruction target, T3-DiffWeather outperforms it in only 2 steps, achieving a PSNR improvement of 1.68 dB on Snow100K-S.
- **vs. PromptIR**: PromptIR utilizes shared prompts to adapt to different degradations, failing to distinguish differences between them. The autonomous prompt selection mechanism in T3-DiffWeather's prompt pool is much more flexible, yielding a 0.61 dB higher PSNR.
- **vs. AWRCP**: AWRCP is the ICCV 2023 SOTA. T3-DiffWeather outperforms it across all benchmarks (Snow100K-S +0.59 dB, Outdoor-Rain +0.60 dB).
- **vs. TransWeather**: TransWeather uses a fixed weather-type query, which cannot adaptively compose. T3-DiffWeather's prompt pool offers a more advanced design philosophy.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of the prompt pool and Depth-Anything constraint is highly novel, and the residual reconstruction objective is elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on four synthetic benchmarks + two real-world datasets, with detailed ablations (prompt pool/general prompts/CPL/pool size/top-$k$/DA architecture) and comprehensive computational efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clean structure, rich visualizations (t-SNE, sub-prompt selection frequency, heatmaps), with well-written motivations and discussions.
- Value: ⭐⭐⭐⭐⭐ Achieving SOTA performances while consuming only 1/52 of previous SOTA's computational resources, significantly promoting the practical deployment of diffusion-based restoration. The prompt pool design philosophy can be widely generalized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Restoring Images in Adverse Weather Conditions via Histogram Transformer](restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](../../NeurIPS2025/image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ICCV 2025\] Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)](../../ICCV2025/image_restoration/robust_adverse_weather_removal_via_spectral-based_spatial_grouping.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)

</div>

<!-- RELATED:END -->
