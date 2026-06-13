---
title: >-
  [Paper Note] Diffusion-CAM: Faithful Visual Explanations for dMLLMs
description: >-
  [ACL 2026][Interpretability][Diffusion Multimodal Large Language Models] Proposes Diffusion-CAM, the first interpretability method designed specifically for diffusion-based multimodal large language models (dMLLMs). By e…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Diffusion Multimodal Large Language Models"
  - "Class Activation Mapping"
  - "Visual Explanation"
  - "Explainable AI"
  - "Parallel Generation"
date: 2026-05-08
content_hash: 9e98c1aa37785439
---

# Diffusion-CAM: Faithful Visual Explanations for dMLLMs

**Conference**: ACL 2026  
**arXiv**: [2604.11005](https://arxiv.org/abs/2604.11005)  
**Code**: [GitHub](https://github.com/ZzzzzZhhmm/Diffusion-CAM)  
**Area**: Image Restoration  
**Keywords**: Diffusion Multimodal Large Language Models, Class Activation Mapping, Visual Explanation, Explainable AI, Parallel Generation

## TL;DR
Proposes Diffusion-CAM, the first interpretability method designed specifically for diffusion-based multimodal large language models (dMLLMs). By extracting structurally valid intermediate representations from the denoising trajectory and employing four post-processing modules (Adaptive Kernel Denoising, Distribution-Aware Confidence Gating, Contextual Background Attenuation, and Single-Instance Causal Debiasing), it significantly outperforms autoregressive CAM baselines on COCO Caption and GranDf.

## Background & Motivation

**Background**: Multimodal LLMs are shifting from autoregressive architectures (LLaVA, Qwen-VL) to diffusion-based paradigms (LaViDa, LLaDA-V, MMaDA). Diffusion models generate entire sentences through parallel masked denoising, improving generation speed and global coherence.

**Limitations of Prior Work**: (1) Existing CAM methods (e.g., LLaVA-CAM, TAM) rely on the sequential, attention-rich nature of autoregressive models to track token generation—but dMLLMs lack explicit token-level attention weights and left-to-right causal structures; (2) Directly applying traditional CAM to dMLLMs produces diffuse, non-specific heatmaps; (3) The parallel denoising process of dMLLMs yields smooth, distributed activation patterns, fundamentally different from the local, sequential dependencies of autoregressive models.

**Key Challenge**: The architectural advantages of dMLLMs (parallel generation, global planning) are precisely the obstacles for traditional interpretability tools—the latter assume sequential dependency while the former are parallel.

**Goal**: Design the first visual explanation method adapted for diffusion-based multimodal models.

**Key Insight**: Identify "structurally valid" intermediate steps in the denoising trajectory—where image-conditioned spatial information is still preserved and can be linked to the final prediction via gradients.

**Core Idea**: Extract Gradient CAM from structurally valid steps of the denoising process combined with four diffusion-specific post-processing modules to solve issues like spatial noise, background diffusion, and redundant token correlation.

## Method

### Overall Architecture
Adapting CAM to dMLLM: (1) Register hooks in intermediate transformer blocks to extract features and gradients; (2) Dynamically locate the position boundaries of image tokens; (3) Backpropagate from final response scores to image region features at valid steps to generate the base CAM; (4) Refine with four post-processing modules.

### Key Designs

1. **Diffusion CAM Adaptation (Three-step Transformation)**:

    - **Function**: Makes CAM compatible with non-autoregressive denoising generation processes.
    - **Mechanism**: (1) **Model-Aware Feature Extraction**: Selects denoising steps satisfying the "feasibility condition"—where the hook hidden state sequence still contains the full image token span. (2) **Dynamic Image Span Localization**: Parses image token boundaries from info4cam metadata, extracts image features, and reshapes them into spatial feature maps. (3) **Grad-CAM Aggregation**: Averages gradients spatially to obtain channel weights, followed by a weighted sum and ReLU to obtain the base CAM.
    - **Design Motivation**: Instead of assuming fixed image token positions or specific denoising steps, a general feasibility criterion is used for adaptive selection.

2. **Adaptive Kernel Denoising Module**:

    - **Function**: Suppresses high-frequency architectural artifacts from Transformer self-attention.
    - **Mechanism**: Dynamically scales the filter kernel size $k_{\text{adaptive}}$, considering three factors: the number of denoising steps (larger kernel for more steps), spatial variance (larger kernel for high noise), and resolution (to ensure scale invariance). Employs rank-weighted Gaussian filtering—assigning weights according to activation value ranking rather than spatial distance.
    - **Design Motivation**: Fixed kernel sizes cannot adapt to the varying noise characteristics of different denoising steps and image contents.

3. **Distribution-Aware Confidence Gating + Contextual Background Attenuation + Single-Instance Causal Debiasing**:

    - **Function**: Respectively address high-variance activation artifacts, residual background signals, and abnormally high activation of repeated tokens.
    - **Mechanism**: Confidence gating adaptively determines thresholds based on global statistics to differentiate high/low confidence regions; background attenuation defines foreground/background separation boundaries using multi-scale statistical integration; causal debiasing clears redundant responses through repeated token detection and abnormally high activation masking.
    - **Design Motivation**: The multi-step denoising of diffusion models accumulates various noise sources, requiring specific modules to address them one by one.

## Key Experimental Results

### Main Results (COCO Caption + GranDf)

| Method | Localization Accuracy | Background Suppression | Visual Fidelity |
|------|---------|---------|---------|
| LLaVA-CAM | Baseline | Weak | Weak |
| Grad-CAM (Direct Apply) | Poor | Poor | Poor |
| **Diffusion-CAM** | **SOTA** | **SOTA** | **SOTA** |

### Ablation Study

| Module | Contribution |
|------|------|
| Adaptive Kernel Denoising | Suppresses high-frequency artifacts, improves heatmap smoothness |
| Confidence Gating | Distinguishes semantic regions from noise |
| Background Attenuation | Eliminates diffuse background responses |
| Causal Debiasing | Eliminates redundant activation caused by repeated tokens |
| **Combined Four Modules** | **Optimal, modules are complementary** |

### Key Findings
- **Directly applying autoregressive CAM to dMLLMs fails completely**—producing diffuse, uninterpretable heatmaps.
- **The four post-processing modules each solve a specific problem and are indispensable**.
- **The choice of denoising steps is crucial**: meaningful visual attribution can only be extracted at structurally valid steps.
- **Diffusion-CAM significantly outperforms all baselines in localization accuracy and visual fidelity**.

## Highlights & Insights
- **Reveals the fundamental challenge of dMLLM interpretability for the first time**: the conflict between parallel generation and sequential dependency. This issue will grow in importance as diffusion architectures gain popularity.
- **The concept of "structurally valid steps"** provides a general principle—in non-autoregressive models, attribution should be extracted from intermediate states that preserve input-conditioned spatial information.
- **The four-module design**, while engineering-oriented, is grounded in clear theoretical motivations (noise analysis).

## Limitations & Future Work
- Currently validated only on the LaViDa series; adaptability to other dMLLMs (e.g., LLaDA-V, MMaDA) remains to be confirmed.
- Hyperparameters for the four modules (e.g., $\delta_\sigma$, $\delta_\mu$) require adjustment based on the model.
- Gradient backpropagation paths may not be unique in parallel denoising; the causal validity of attribution needs deeper analysis.
- Computational overhead is higher than autoregressive CAM (requires storage of intermediate denoising states).
- Text-token level attribution was not explored (currently focused only on visual region attribution).

## Related Work & Insights
- **vs LLaVA-CAM**: Designed for autoregressive models; performs extremely poorly on dMLLMs. Diffusion-CAM is a necessary alternative.
- **vs DAAM (Tang et al.)**: DAAM performs attribution for text-to-image diffusion models, but the goals and methods differ from multimodal reasoning.
- **vs Attention Visualization**: dMLLMs lack explicit autoregressive attention weights, making attention-based methods inapplicable.

## Rating
- Novelty: ⭐⭐⭐⭐ First interpretability method for dMLLMs, though the core idea (Grad-CAM + post-processing) is established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks + ablation + comparison, though the dMLLM ecosystem is still small.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and well-organized four-module design.
- Value: ⭐⭐⭐⭐ Importance will grow as dMLLMs become more prevalent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[AAAI 2026\] Using Certifying Constraint Solvers for Generating Step-wise Explanations](../../AAAI2026/interpretability/using_certifying_constraint_solvers_for_generating_step-wise_explanations.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/interpretability/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)

</div>

<!-- RELATED:END -->
