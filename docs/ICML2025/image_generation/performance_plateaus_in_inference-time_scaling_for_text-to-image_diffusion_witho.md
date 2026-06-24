---
title: >-
  [Paper Note] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models
description: >-
  [ICML2025][Image Generation][inference-time scaling] This paper systematically investigates the effect of applying Best-of-N inference-time scaling to initial noise optimization algorithms for text-to-image diffusion models without relying on external models (VLMs/CLIP). The study reveals that performance rapidly hits a plateau, where a small number of optimization steps can closely approach the maximum achievable performance under this setting. Furthermore…
tags:
  - "ICML2025"
  - "Image Generation"
  - "inference-time scaling"
  - "text-to-image diffusion"
  - "initial noise optimization"
  - "Best-of-N"
  - "performance plateau"
  - "attention maps"
date: 2026-05-08
content_hash: 2ab5bf269bc3ba1b
---

# Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models

**Conference**: ICML2025  
**arXiv**: [2506.12633](https://arxiv.org/abs/2506.12633)  
**Authors**: Changhyun Choi, Sungha Kim, H. Jin Kim  
**Code**: [initno official](https://github.com/xiefan-guo/initno)  
**Area**: Image Generation  
**Keywords**: inference-time scaling, text-to-image diffusion, initial noise optimization, Best-of-N, performance plateau, attention maps

## TL;DR

This paper systematically investigates the effect of applying Best-of-N inference-time scaling to initial noise optimization algorithms for text-to-image diffusion models without relying on external models (VLMs/CLIP). The study reveals that performance rapidly hits a plateau, where a small number of optimization steps can closely approach the maximum achievable performance under this setting. Furthermore, the optimal algorithm varies across different backbone diffusion models.

## Background & Motivation

### Inference-Time Scaling: From LLMs to Diffusion Models
Inference-time scaling has demonstrated great potential in Large Language Models (LLMs)—enhancing generation quality by investing more inference computation without increasing model parameters (OpenAI 2024; Guo et al., 2025). This concept has naturally been introduced to the domain of diffusion models: pioneering work by Ma et al. (2025) showed that searching for better initial noise can significantly improve T2I task performance.

### Limitations of Prior Work
However, existing inference-time scaling methods **all rely on external models** (such as VLMs or CLIP) to evaluate the quality of the generated images. This introduces severe practical limitations:

- **High VRAM footprint**: Loading both the diffusion model and the evaluation model simultaneously requires high-end GPUs, making it difficult to support on consumer-grade graphics cards (e.g., 8-12GB VRAM).
- **High deployment cost**: This is highly impractical in personal desktop environments rather than large-scale laboratories.
- **Increased inference latency**: Each evaluation requires an additional forward pass.

### Alternative Paths Without External Models
Another path is to optimize the initial noise using only information from the diffusion model itself (such as attention maps), represented by methods like CONFORM, InitNO, and Self-Cross guidance. These methods do not require external models, but a crucial question remains unanswered:

> **When relying solely on the diffusion model itself to select good initial noise, can investing more computation (Best-of-N scaling) continuously improve performance?**

This work presents a systematic empirical study addressing this exact question.

## Method

### Overall Architecture: Best-of-N Initial Noise Selection

The core idea is straightforward:
1. Allocate a total budget of $N$ loss calculations.
2. Sample multiple candidate initial noises and evaluate the quality of each noise using the loss function of each algorithm.
3. Select the noise with the lowest loss value as the initial noise for the diffusion model.
4. Execute the standard denoising process using this noise to generate the final image.

If the loss function can accurately reflect the quality of the generated image, performance should increase monotonically as $N$ increases. This hypothesis is verified through large-scale experiments in this paper.

### Three Noise Optimization Algorithms Without External Models

**1. CONFORM (Meral et al., 2024)**
- Utilizes contrastive learning on the **cross-attention maps** of the T2I diffusion model.
- Uses an InfoNCE loss function to separate objects and attributes in the text prompt into positive and negative sample pairs.
- Evaluates each noise independently, where $N$ computations correspond to $N$ candidate noises.
- Advantage: Directly leverages the semantic structure of the prompt without iterative optimization.

**2. InitNO (Guo et al., 2024)**
- Core idea: Not all noises sampled from a standard normal distribution match a given prompt; "effective" and "ineffective" noises exist.
- The loss function consists of two parts:
    - $1 - \text{minmax\_cross}$: minmax_cross is the minimum of the maximum cross-attention weights for each target object, measuring whether all objects receive sufficient attention.
    - **Self-attention map overlap**: Measures the degree of overlap amongst the self-attention maps of the patches that correspond to the maximum cross-attention for each object (a larger overlap indicates more severe object confusion).
- Key characteristic: Performs **iterative optimization** (up to 10 steps) on a single noise; thus, $N$ computations yield only $N/10$ candidates.
- Includes an early stopping mechanism: If the loss falls below a predefined threshold, optimization stops immediately.

**3. Self-Cross Guidance (Qiu et al., 2024)**
- The loss is similar to InitNO, but the key difference is that **it does not focus only on the maximum cross-attention patches**.
- It uses **self-attention maps of all patches**, weighted by their respective cross-attention weights, to calculate the overlap between objects.
- Offers a more global measure of physical separation between objects.
- The authors suggest applying InitNO first, followed by Self-Cross guidance (two-stage optimization).
- Evaluates each noise independently, where $N$ computations correspond to $N$ candidates.

### Differences in Candidate Noise Quantity

| Algorithm | Computational Cost per Candidate | Number of Effective Candidates for $N$ Computations |
|------|------|------|
| CONFORM | 1 loss calculation | $N$ |
| InitNO | 10 loss calculations (iterative optimization) | $N/10$ |
| Self-Cross Guidance | 1 loss calculation | $N$ |

This difference necessitates unifying the "number of loss calculations" as the computational budget unit for a fair comparison.

## Key Experimental Results

### Experimental Setup
- **Diffusion Models**: Stable Diffusion 1.5 (SD1.5), SD2, and other backbone networks.
- **Datasets**: 4 test sets covering different compositional complexities.
    - `animal_animal`: Combinations of two animals.
    - `animal_object`: Combinations of an animal and an object.
    - `object_object`: Combinations of two objects.
    - `similar_subjects`: Similar subjects (the most challenging, highly prone to confusion).
- **Evaluation**: Alignment between the generated image and prompt (text-image consistency).
- **Scaling Range**: $N$ was systematically scaled from small to large to observe performance trends.

### Table 1: Performance Trends as N Scales on SD1.5 (Performance Plateau Phenomenon)

| Algorithm | N=10 | N=50 | N=100 | N=500 | Performance Trend |
|------|------|------|------|------|------|
| CONFORM | Near baseline | Near optimal | Plateau | No significant gain | Rapid saturation |
| InitNO | Significant gain | Near optimal | Plateau | No significant gain | Moderate saturation |
| Self-Cross | Near baseline | Near optimal | Plateau | No significant gain | Rapid saturation |

Core finding: **All algorithms hit a performance plateau at relatively small values of $N$**. Continued investment in computational budget does not yield meaningful performance improvements, which stands in stark contrast to default hyperparameter settings and intuitive expectations.

### Table 2: Comparison of Optimal Algorithms Across Different Backbone Models

| Backbone Model | Optimal Algorithm | Notes |
|------|------|------|
| SD1.5 (UNet) | Varies by dataset | CONFORM and InitNO each have scenarios where they excel |
| SD2 (UNet) | Different from SD1.5 | The optimal algorithm changes |
| Other Backbones | To be explored | SOTA algorithms change with the model |

**Key Finding**: SOTA algorithms for initial noise optimization without external models **shift depending on the underlying diffusion model**, indicating that no single algorithm is currently optimal across all models. This points out a direction for future research.

## Highlights & Insights

- **Clear and Compelling Empirical Contribution**: First to systematically demonstrate the existence of a **performance ceiling** (plateau) in inference-time scaling under the constraint of using no external models, shattering the intuitive assumption that "more computation = better performance."
- **High Practical Value**: Explicitly informs practitioners that under consumer-grade GPUs, a small number of optimization steps is sufficient to achieve optimal results, **eliminating the need to waste computational resources**. This provides direct guidance for deploying T2I models in resource-constrained environments.
- **Revealing the Limitations of Loss Functions**: The occurrence of the performance plateau implies that the loss functions of these algorithms **do not perfectly reflect the generated image quality**—if a loss function were a perfect proxy metric, performance would monotonically increase with $N$.
- **Model Dependency Discovery**: The divergence of the optimal algorithm across different diffusion models indicates that the information content and quality of attention maps vary by model, demanding tailored noise optimization strategies.
- **Fair Comparison Framework**: Unifying the budget unit as the "number of loss calculations" establishes a fair baseline for comparing algorithms with different levels of complexity (specifically, InitNO's iterative optimization vs. CONFORM's single evaluation).

## Limitations & Future Work

- **Only Three Algorithms Covered**: Currently, only CONFORM, InitNO, and Self-Cross guidance are evaluated; future algorithms without external models might break through the plateau.
- **Unexplored Loss Function Design Space**: The root cause of the plateau likely lies in the insufficient expressiveness of current interest/attention map-based loss functions. Seeking stronger self-supervised signals is a crucial direction.
- **Limited Backbones**: Experiments were primarily conducted on the SD1/SD2 series, leaving newer DiT-based architectures like SDXL, SD3, and Flux unexplored.
- **Singular Evaluation Metric**: Performance evaluation heavily relies on text-to-image alignment, failing to fully consider aesthetic quality, diversity, and other dimensions.
- **Gap with External Model Methods**: The performance plateau indicates an inherent ceiling for methods without external models. How to introduce stronger quality signals (such as lightweight evaluators) without increasing VRAM is a promising direction.
- **Insufficient Computation Overhead Analysis**: Although the budget was unified using the number of loss calculations, wall-clock time and actual memory usage under each algorithm were not reported in detail.

## Related Work

### Inference-Time Scaling
- **LLM Domain**: OpenAI (2024), Guo et al. (2025), Zhang et al. (2025), etc., demonstrated that increasing inference-time computation serves as an alternative to scaling up models.
- **Diffusion Domain**: Ma et al. (2025) first introduced inference-time scaling to T2I by searching for optimal initial noise using a VLM as an evaluator; Li et al. (2025) and Zhuo et al. (2025) further expanded on this.
- **Positioning of This Work**: Focuses on the constrained scenario of having **no external models**, revealing fundamental limitations of inference-time scaling in this setting.

### Initial Noise Optimization
- **CONFORM** (Meral et al., 2024): Performs contrastive learning based on cross-attention maps, using InfoNCE to constrain alignment between prompt and attention.
- **InitNO** (Guo et al., 2024): Introduces the concept of "effective/ineffective noise" and ensures semantic alignment through iterative optimization of initial noise.
- **Self-Cross Guidance** (Qiu et al., 2024): Integrates global self-attention with cross-attention to measure object separation more comprehensively than InitNO.

### Fundamentals of T2I Diffusion Models
- **Stable Diffusion** (Rombach et al., 2022): Latent diffusion models performing DDPM denoising in the latent space of an autoencoder.
- **DDPM** (Ho et al., 2020): Denoising Diffusion Probabilistic Models, the foundational framework of diffusion models.
- **Cross-Attention Mechanism**: Utilizes text encoder outputs as keys and UNet intermediate features as queries to form cross-modal alignment.

## Rating

- Novelty: ⭐⭐⭐ — The research question is important and timely, but the Best-of-N method itself is relatively straightforward; the core contribution is the empirical discovery rather than algorithmic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Highly systematic experiments covering multiple algorithms × datasets × backbones; however, backbone models and evaluation metrics could have been more diverse.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, experiments are logically designed, and findings are well-defined.
- Value: ⭐⭐⭐⭐ — Directly guides the practical deployment of T2I inference-time scaling, saving substantial trial-and-error costs for researchers in this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Scaling of Diffusion Models Through Classical Search](../../ICLR2026/image_generation/inference-time_scaling_of_diffusion_models_through_classical_search.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](../../NeurIPS2025/image_generation/inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[ICLR 2026\] Compositional Visual Planning via Inference-Time Diffusion Scaling](../../ICLR2026/image_generation/compositional_visual_planning_via_inference-time_diffusion_scaling.md)
- [\[CVPR 2025\] Scaling Down Text Encoders of Text-to-Image Diffusion Models](../../CVPR2025/image_generation/scaling_down_text_encoders_of_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
