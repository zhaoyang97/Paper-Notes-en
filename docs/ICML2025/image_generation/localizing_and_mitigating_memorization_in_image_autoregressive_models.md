---
title: >-
  [Paper Note] Localizing and Mitigating Memorization in Image Autoregressive Models
description: >-
  [ICML2025][Image Generation][Image Autoregressive Models] This work utilizes an improved UnitMem metric to localize memorized neurons in image autoregressive models (VAR/RAR). It reveals that memorization distribution patterns differ significantly across architectural designs, and presents a privacy mitigation solution. By scaling down the weights of highly memorized neurons, the method achieves a substantial reduction in the volume of extractable training data (from 672 to 1…
tags:
  - "ICML2025"
  - "Image Generation"
  - "Image Autoregressive Models"
  - "Memorization Localization"
  - "Privacy Protection"
  - "UnitMem"
  - "Data Extraction Attacks"
date: 2026-05-08
content_hash: d399ac45358bf70b
---

# Localizing and Mitigating Memorization in Image Autoregressive Models

**Conference**: ICML2025  
**arXiv**: [2509.00488](https://arxiv.org/abs/2509.00488)  
**Authors**: Aditya Kasliwal, Franziska Boenisch, Adam Dziedzic  
**Code**: Not publicly available  
**Area**: Image Generation  
**Keywords**: Image Autoregressive Models, Memorization Localization, Privacy Protection, UnitMem, Data Extraction Attacks  

## TL;DR

This work utilizes an improved UnitMem metric to localize memorized neurons in image autoregressive models (VAR/RAR). It reveals that memorization distribution patterns differ significantly across architectural designs, and presents a privacy mitigation solution. By scaling down the weights of highly memorized neurons, the method achieves a substantial reduction in the volume of extractable training data (from 672 to 110 images in VAR-d30) with a controllable impact on generation quality.

## Background & Motivation

### Problem Background
Image Autoregressive (IAR) models—such as Visual Autoregressive Modeling (VAR) and Randomized Autoregressive (RAR)—have achieved SOTA performance in image generation quality and speed, surpassing other frameworks like diffusion models. However, these models tend to memorize training data, posing severe privacy risks: memorized sensitive data can be maliciously extracted or unintentionally leaked.

### Limitations of Prior Work
- Memorization has been widely studied in **diffusion models** (Carlini et al., 2023; Somepalli et al., 2023), but research on memorization in IAR models is still in its early stages.
- Kowalczuk et al. (2025) demonstrated that IAR models exhibit a stronger tendency to memorize than diffusion models, but they lack an in-depth analysis of memorization **localization**.
- Existing memorization localization methods (specifically, UnitMem) were designed for vision encoders and have not been systematically applied to iterative generative architectures like IAR.
- There is a lack of end-to-end validation from memorization localization to practical privacy mitigation.

### Key Motivation
Understanding the **spatial distribution patterns** of memorization inside IAR models is a prerequisite for building practical privacy mitigation strategies. This paper aims to answer: (1) Which components in IAR models are responsible for storing training data? (2) How do memorization patterns differ across different architectures? (3) Can targeted interventions on highly memorized components reduce data extraction risks?

## Method

### Overall Architecture

The proposed method consists of three phases: **Memorization Measurement** $\rightarrow$ **Pattern Analysis** $\rightarrow$ **Intervention & Mitigation**.

1. **UnitMem Metric Adaptation**: Adapt the UnitMem metric, originally designed for vision encoders, to the iterative generation characteristics of IAR models.
2. **Memorization Localization Analysis**: Systematically localize memorized neurons in two architectures: VAR (hierarchical multi-scale) and RAR (token-by-token).
3. **Weight Scaling Intervention**: Perform weight scaling on highly memorized neurons to validate localization accuracy and achieve privacy mitigation.

### Key Designs 1: IAR Adaptation of the UnitMem Metric

The UnitMem metric quantifies the degree of memorization of a single neuron $u$:

$$\text{UnitMem}_{\mathcal{D}'}(u) = \frac{\mu_{max,u} - \mu_{-max,u}}{\mu_{max,u} + \mu_{-max,u}}$$

where $\mu_{max,u}$ is the maximum activation value of unit $u$ for a specific data point $x_k$, and $\mu_{-max,u}$ is the average activation value of $u$ across all other data points. A higher value indicates stronger "selectivity" of the neuron for a specific training sample, meaning higher memorization.

**GELU Adaptation**: The original UnitMem was designed for ReLU (non-negative activations), but VAR and RAR use the GELU activation function (which can produce negative values). This paper uses the **absolute value** of the activations to calculate $\mu_{max,u}$ and $\mu_{-max,u}$, ensuring that the magnitude of activation rather than the sign determines the memorization score.

**Teacher-Forced Inference**: IAR models iteratively use the same components during the generation process. To prevent error accumulation from affecting measurement accuracy, teacher-forcing is used when calculating UnitMem activations: the true preceding sequence is fed at each step instead of the model's own predictions.

**Analysis Focusing on the fc1 Layer**: Each block in VAR and RAR contains an attention layer and two fully connected layers (fc1, fc2). Since fc1 uses the GELU activation while fc2 has no activation function, the UnitMem analysis focuses on neurons in the fc1 layer.

### Key Designs 2: Architecture-Specific Memorization Localization

**VAR (Hierarchical Multi-Scale Architecture)**:
- VAR generates images hierarchically across 10 scales, with the same transformer blocks reused at each scale.
- This paper computes UnitMem in a two-dimensional manner: **scale + block position**, which reveals the distribution of memorization across scales and depths.
- **Finding**: At coarse scales (low resolution), memorization concentrates in the initial blocks, and gradually shifts to deeper layers as the scale becomes finer.

**RAR (Token-by-Token Autoregressive Architecture)**:
- RAR generates images token-by-token using randomly permuted token sequences and bidirectional attention.
- Memorization is analyzed at the block level.
- **Finding**: Memorization concentrates in the middle and late blocks.

### Key Designs 3: Weight Scaling Intervention

Interventions are validated on the identified neurons with high UnitMem scores:
- The weights of a target percentage of highly memorized training neurons are **scaled to half of their original values**.
- Evaluate post-intervention: (1) change in the number of extractable training images; (2) change in generation quality metrics such as FID.
- If the intervention effectively reduces extractions with a controllable impact on quality, the localization accuracy is validated.

## Key Experimental Results

### Experimental Setup
- Models: VAR-d16, VAR-d30 (minimum/maximum configurations), RAR-Base, RAR-XXL (minimum/maximum configurations)
- Training Data: ImageNet-1k
- UnitMem Calculation Subset: 1% of the ImageNet-1k training set (uniformly sampled per class); it is empirically verified that 1%, 5%, 10%, and 20% subsets yield similar memorization patterns.
- Each data point undergoes 10 forward passes with different data augmentations to obtain the average activation.

### Main Results: Mitigation Effect on Data Extraction

| Model | Extractable Images (Pre-intervention) | Extractable Images (Post-intervention) | Reduction Ratio | FID Impact |
|------|------------------|------------------|---------|--------|
| VAR-d16 | — | — | Significant Reduction | Controllable |
| VAR-d30 | 672 | 110 | **83.6%** | Limited Impact |
| RAR-Base | — | — | Significant Reduction | Controllable |
| RAR-XXL | 75 | 26 | **65.3%** | Limited Impact |

The extractable count for VAR-d30 drops sharply from 672 to 110 images (an 83.6% reduction), and RAR-XXL drops from 75 to 26 images (a 65.3% reduction), indicating that the localization method accurately identifies the critical neurons for memorization.

### Ablation Study: Comparison of Memorization Distribution Patterns

| Architecture Type | Generation Mechanism | Memorization Concentration Area | Memorization Shift with Depth |
|---------|---------|--------------|----------------|
| VAR (Hierarchical) | Scale-by-scale prediction | Coarse scales $\rightarrow$ initial blocks; Fine scales $\rightarrow$ deep blocks | Shifts to deeper layers as scale becomes finer |
| RAR (Token-by-token) | Randomly permuted token sequences | Middle and late blocks | Concentrated in the later processing stages |

This comparison reveals the decisive impact of architectural design on the distribution of memorization: the hierarchical architecture's memorization dynamically shifts with resolution, whereas standard autoregressive architecture's memorization remains stably concentrated in the later processing stages.

### Supplementary Experiments: Robustness to UnitMem Subset Size

| ImageNet-1k Subset Ratio | Memorization Pattern Consistency |
|--------------------|----------------|
| 1% | Baseline pattern |
| 5% | Consistent with 1% |
| 10% | Consistent with 1% |
| 20% | Consistent with 1% |

Verifies that using a 1% subset is sufficient to capture memorization patterns, significantly reducing computational overhead.

## Highlights & Insights

- **First systematic localization of IAR memorization**: Extends UnitMem from vision encoders to generative IAR models, revealing architecture-specific distribution patterns of memorization.
- **Discovery that architecture dictates memorization patterns**: The hierarchical design of VAR causes memorization to dynamically migrate with scale (coarse $\rightarrow$ initial, fine $\rightarrow$ deep layers), while the sequential design of RAR causes memorization to concentrate in later stages. This insight provides architectural perspectives for privacy-preserving model design.
- **Practical privacy mitigation solution**: Reduces 80%+ of extractable data purely via weight scaling (no retraining required) with a controllable loss in generation quality, showing high practical value.
- **Lightweight and efficient measurement scheme**: Localizes memorization with a single forward pass and 1% of data, demonstrating excellent scalability.
- **Simple and effective GELU adaptation**: Cleverly solves UnitMem's compatibility issue with non-ReLU activation functions by taking absolute values.

## Limitations & Future Work

- **Only covers two IAR architectures**: Only VAR and RAR are analyzed, without covering other IAR models (such as LlamaGen, etc.). The generalizability of the conclusions remains to be verified.
- **Simple intervention strategy**: Only uses a fixed 0.5 weight scaling, without exploring more granular intervention methods like adaptive scaling, pruning, or fine-tuning.
- **Validation limited to ImageNet-1k**: Lacks cross-validation on other datasets (such as LAION, COCO).
- **Lack of comparison with other privacy-preserving methods**: No systematic comparison with methods like Differential Privacy (DP-SGD) or Machine Unlearning.
- **Limited theoretical support for UnitMem**: The causal relationship between activation selectivity and memorization is not rigorously proven, and confounding factors may exist.
- **Single-dimension generative quality evaluation**: Processes mostly rely on FID without evaluating metrics for diversity (like IS, Precision/Recall) and semantic consistency.
- **No class-wise analysis of memorization**: Does not investigate whether certain classes are easier to memorize or how class imbalance affects memorization.

## Related Work & Insights

- **UnitMem (Wang et al., 2024a)**: The core tool of this work. The original paper proved that memorized units can exist distributively and vary with layer depth in vision encoders. This paper successfully extends it to generative models.
- **Kowalczuk et al. (2025)**: Proved that the memorization tendency of IAR models even exceeds that of diffusion models. This paper's data extraction experiments directly adopt their methodology.
- **Carlini et al. (2023)**: Pioneering work on data extraction attacks on diffusion models. This paper applies similar security auditing ideas to IAR models.
- **Maini et al. (2023)**: Investigated memorization localization in Large Language Models, which methodologically inspired this paper.
- **VAR (Tian et al., 2024)**: Representative work on visual autoregressive models. This paper finds that its multi-scale design introduces a unique dynamic migration of memorization.
- **RAR (Yu et al., 2024)**: Randomized autoregressive model. Under its bidirectional attention mechanism, memorization is more concentrated in the later processing stages.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic localization of IAR model memorization, revealing architecture-specific patterns with clear innovative contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Core experiments are reasonably designed and fully validated, but the coverage of models and datasets is relatively limited, and comparison with other baseline methods is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, fully motivated, and presents a complete logical chain of methodology and validation.
- Value: ⭐⭐⭐⭐ — Provides a practical, zero-cost privacy mitigation solution and architecture-level insights into memorization, holding direct significance for secure model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [\[ICLR 2026\] Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](../../ICLR2026/image_generation/detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)
- [\[CVPR 2025\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](../../CVPR2025/image_generation/mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [\[CVPR 2025\] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](../../CVPR2025/image_generation/enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)
- [\[ICML 2025\] Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots](hierarchical_masked_autoregressive_models_with_low-resolution_token_pivots.md)

</div>

<!-- RELATED:END -->
