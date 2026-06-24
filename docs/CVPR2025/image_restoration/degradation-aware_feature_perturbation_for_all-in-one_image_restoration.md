---
title: >-
  [Paper Note] Degradation-Aware Feature Perturbation for All-in-One Image Restoration
description: >-
  [CVPR 2025][Image Restoration][All-in-One Image Restoration] This paper proposes the DFPIR framework, which adapts the feature space between the encoder and decoder to fit a unified parameter space through two mechanisms: degradation type-guided channel shuffle perturbation and selective attention mask perturbation. It achieves state-of-the-art (SOTA) performance across five distinct tasks, including denoising, dehazing, deraining, deblurring, and low-light enhancement.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "All-in-One Image Restoration"
  - "Feature Perturbation"
  - "Channel Shuffle"
  - "Selective Attention"
  - "Degradation-Aware"
date: 2026-05-08
content_hash: a7ee8071a26463b5
---

# Degradation-Aware Feature Perturbation for All-in-One Image Restoration

**Conference**: CVPR 2025  
**arXiv**: [2505.12630](https://arxiv.org/abs/2505.12630)  
**Code**: [GitHub](https://github.com/TxpHome/DFPIR)  
**Area**: Image Restoration  
**Keywords**: All-in-One Image Restoration, Feature Perturbation, Channel Shuffle, Selective Attention, Degradation-Aware

## TL;DR
This paper proposes the DFPIR framework, which adapts the feature space between the encoder and decoder to fit a unified parameter space through two mechanisms: degradation type-guided channel shuffle perturbation and selective attention mask perturbation. It achieves state-of-the-art (SOTA) performance across five distinct tasks, including denoising, dehazing, deraining, deblurring, and low-light enhancement.

## Background & Motivation

**Background**: Single-task image restoration has made significant progress; however, training separate models for each degradation type is computationally expensive in real-world applications. All-in-One approaches attempt to handle multiple types of degradation using a single, unified model.

**Limitations of Prior Work**: Gradient conflicts arise among different degradation tasks when sharing parameters—for example, the optimization directions of denoising and dehazing can be contradictory. Existing methods generally fall into two categories: adjusting the parameter space (introducing extra degradation-specific parameters, which leads to high computational overhead) and adjusting the feature space (such as implicit prompts in PromptIR or textual instructions in InstructIR). However, the former increases complexity, while the latter struggles to effectively isolate inter-degradation interference.

**Key Challenge**: There is a need to preserve the inherent features of the image (which are universally beneficial to all tasks) while minimizing mutual interference between different degradation features. Existing prompt-based methods either ignore explicit degradation type information (e.g., PromptIR) or rely solely on channel attention modulation (e.g., InstructIR), making it difficult to fully isolate the impact of different degradations.

**Goal**: To design a lightweight mechanism that preserves inherent image features while modulating features according to the specific degradation type.

**Key Insight**: Modulate features through "perturbation" rather than "replacement"—channel shuffling alters feature arrangement while preserving details, and attention masking selectively filters feature information without completely discarding it.

**Core Idea**: Utilize degradation-type prompts to guide channel-wise shuffle for feature rearrangement, followed by selective attention masks to perturb the attention space. Working in tandem, these mechanisms adapt encoded features to fit a unified decoder.

## Method

### Overall Architecture
The proposed method is based on a 4-level encoder-decoder architecture of Restormer. The Degradation-Guided Perturbation Block (DGPB) is inserted into the skip connections between the encoder and decoder. Degradation type descriptions are encoded using a pre-trained CLIP text encoder to obtain prompt vectors, which guide the perturbation process.

### Key Designs

1. **Degradation-Guided Channel Perturbation Module (DGCPM)**:

    - **Function**: Rearrange feature channels based on degradation types.
    - **Mechanism**: First, the channel dimension is expanded by a factor of 2 into a high-dimensional space. An MLP maps the degradation prompt to a vector of the same length as the channel size, and the top-K indices are used to reorder the channels (channel shuffle). Finally, the features are halved back to the original channel dimension. Different degradation types prompt distinct channel arrangements due to their unique descriptions.
    - **Design Motivation**: Performing channel shuffling directly on the original channels yields excessively large perturbations, which hinders convergence. Thus, expanding to a higher dimension before shuffling and down-projecting is crucial. Channel shuffle preserves all information (since it is only rearranged), which is more gentle than the weighting and filtering of channel attention.

2. **Channel-Adapted Attention Perturbation Module (CAAPM)**:

    - **Function**: Enable interactions between shuffled features and original features, while applying selective perturbation in the attention space.
    - **Mechanism**: Utilizing the shuffled features as queries and the original features as keys/values, cross-attention is performed along the channel dimension. A top-K mask (with a perturbation factor of $\gamma = 0.9$, i.e., keeping 90%) is applied on the attention map to selectively screen out some attention weights before outputting through a Feed-Forward Network (FFN).
    - **Design Motivation**: Shuffled features contain degradation-related information but lack interaction with original features; cross-attention merges the two. Attention masking further filters out unnecessary information for each specific degradation, striking a balance between "interference isolation" and "inherent feature preservation".

3. **Degradation-Type Prompt System**:

    - **Function**: Provide degradation-type conditions for DGPB.
    - **Mechanism**: A pre-trained CLIP text encoder is utilized to encode degradation type descriptions (e.g., "denoising", "dehazing"). The resulting embedding vectors are mapped by the Degradation Guidance Module (DGM) to guide the index sequence for channel shuffling.
    - **Design Motivation**: CLIP text embeddings naturally contain semantic information about degradation types, and embedding vectors for different degradation types are sufficiently distinct.

### Loss & Training
Standard $L_1$ reconstruction loss is adopted. Degradation-type labels are required during training to obtain corresponding prompts.

## Key Experimental Results

### Main Results (3 Tasks: Dehazing + Deraining + Denoising)

| Method | Dehazing SOTS | Deraining Rain100L | Denoising CBSD68 ($\sigma=25$) | Average PSNR |
|------|---------|-------------|-----------------|---------|
| PromptIR | 30.58 | 36.37 | 31.31 | 32.06 |
| InstructIR | 30.22 | **37.98** | **31.52** | 32.43 |
| **DFPIR (Ours)** | **31.87** | 38.65 | 31.47 | **32.88** |

### Ablation Study (5-Task Setting)

| Method | Dehazing | Deraining | Denoising | Deblurring | Low-light | Average |
|------|------|------|------|--------|------|------|
| Restormer* | 24.09 | 34.81 | 31.49 | 27.22 | 20.41 | 27.60 |
| PromptIR | 30.58 | 36.37 | 31.31 | 29.40 | 23.15 | 30.16 |
| **DFPIR** | **31.87** | **38.65** | 31.47 | **29.86** | **23.38** | **31.05** |

### Key Findings
- Compared with InstructIR, the average PSNR increases by $0.45\text{ dB}$, with a particularly significant improvement in the dehazing task ($+1.65\text{ dB}$).
- t-SNE visualization reveals that DFPIR achieves tighter intra-task feature clustering compared to PromptIR, verifying that the perturbation strategy effectively isolates different degradations.
- The optimal perturbation factor is $\gamma = 0.9$; a value too small (discarding excessive information) or too large (insufficient perturbation) degrades performance.
- Performing channel shuffling in a higher-dimensional space is a necessary condition; shuffling directly within the original channel space leads to training instability.

## Highlights & Insights
- "Channel shuffling" as a means of feature modulation is novel—rearrangement that retains all information vs. standard attention-based weighted filtering is a clean and highly efficient concept.
- DGPB adds only a minimal number of parameters yet achieves significant performance gains, demonstrating that feature space "perturbation" is more suitable for multi-task parameter-sharing scenarios than "replacement".
- The t-SNE visualization is highly convincing, intuitively illustrating how perturbation makes the feature clusters of different degradations more distinct.

## Limitations & Future Work
- The degradation type must be known during inference to obtain the prompt, thus blind restoration of unknown degradations is not supported.
- Degradation types are discrete labels, which do not support continuous control of the degradation intensity.
- The five included degradation types are still limited; scalability to a broader range of degradations has not been verified.
- An degradation estimation network could be considered to automatically obtain prompts, enabling fully-blind restoration.

## Related Work & Insights
- **vs. PromptIR**: PromptIR utilizes implicit routing prompt learning on features while ignoring explicit degradation type information; in contrast, DFPIR uses CLIP text embeddings as explicit degradation conditions.
- **vs. InstructIR**: InstructIR modulates features using text instructions combined with channel attention, whereas DFPIR's channel shuffling and attention masking isolate interference more effectively.
- **vs. MedIR**: MedIR's hard routing strategy separates task-specific parameters, which may ignore inherent features; DFPIR's "soft perturbation" retains all information.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of utilizing channel shuffling for feature modulation is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluations under 3-task and 5-task settings, extensively compared against multiple baselines with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear diagrams and thorough motivation analysis.
- Value: ⭐⭐⭐⭐ A versatile and lightweight module for all-in-one image restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[CVPR 2026\] Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration](../../CVPR2026/image_restoration/degradation-consistent_test-time_adaptation_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [\[CVPR 2026\] Retrieve-to-Restore: Efficient All-in-One Image Restoration with a Retrieval-Based Degradation Bank](../../CVPR2026/image_restoration/retrieve-to-restore_efficient_all-in-one_image_restoration_with_a_retrieval-base.md)
- [\[ICLR 2026\] Rethinking Expressivity and Degradation-Awareness in Attention for All-in-One Blind Image Restoration](../../ICLR2026/image_restoration/rethinking_expressivity_and_degradation-awareness_in_attention_for_all-in-one_bl.md)

</div>

<!-- RELATED:END -->
