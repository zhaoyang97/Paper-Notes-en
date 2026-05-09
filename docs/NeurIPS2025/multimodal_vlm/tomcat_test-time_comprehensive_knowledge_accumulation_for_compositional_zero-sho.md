---
title: >-
  [Paper Note] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][Compositional Zero-Shot Learning] This paper proposes TOMCAT, which dynamically updates compositional prototypes by accumulating dual-modality (textual and visual) knowledge from unlabeled test data at test time, addressing label distribution shift and achieving state-of-the-art performance on four CZSL benchmarks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Compositional Zero-Shot Learning
  - Test-Time Adaptation
  - Knowledge Accumulation
  - Multimodal Prototypes
  - CLIP
date: 2026-05-08
content_hash: fdcbff4e45416d33
---

# TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.20162](https://arxiv.org/abs/2510.20162)
**Code**: [https://github.com/xud-yan/TOMCAT](https://github.com/xud-yan/TOMCAT)
**Area**: Multimodal VLM
**Keywords**: Compositional Zero-Shot Learning, Test-Time Adaptation, Knowledge Accumulation, Multimodal Prototypes, CLIP

## TL;DR
This paper proposes TOMCAT, which dynamically updates compositional prototypes by accumulating dual-modality (textual and visual) knowledge from unlabeled test data at test time, addressing label distribution shift and achieving state-of-the-art performance on four CZSL benchmarks.

## Background & Motivation

**State of the Field**: Compositional Zero-Shot Learning (CZSL) requires models to recognize unseen attribute-object compositions (e.g., "brown cheese") formed by recombining known attributes and objects. CLIP-based methods such as CSP, Troika, and CDS-CZSL have achieved notable progress by fine-tuning VLMs via prompt tuning.

**Limitations of Prior Work**: At test time, the label space includes compositions unseen during training, causing label distribution shift. Model parameters and class prototypes are frozen after training and cannot adapt to the new distribution using test data.

**Root Cause**: Training covers only a subset of compositions, yet inference requires discrimination over the full compositional space—a fundamental distribution shift problem.

**Starting Point**: Stream-wise knowledge accumulation from unlabeled test data at test time, updating prototypes in both textual and visual modalities simultaneously.

**Core Idea**: Design a learnable Knowledge Accumulation Module (KAM) to adjust prototype offsets, coupled with a priority queue that stores high-confidence historical image features to construct visual prototypes.

## Method

### Overall Architecture
During training, CLIP is fine-tuned via prompt tuning and adapters. During testing, the original model remains frozen. Upon receiving each test sample, the pipeline: (1) determines whether to update the priority queue; (2) updates textual and visual prototypes using KAM with adaptive weights; (3) makes predictions based on multimodal prototypes; (4) updates KAM parameters via backpropagation after prediction.

### Key Designs

1. **Knowledge Accumulation Module (KAM)**:

    - **Function**: Learns prototype offsets to adapt to label distribution shifts.
    - **Mechanism**: Maintains learnable parameters $\Delta\mathbf{t}, \Delta\mathbf{v} \in \mathbb{R}^{|C^{te}| \times d}$ for textual and visual prototypes respectively, initialized to zero. The updated prototype is: $\tilde{\mathbf{t}}_c = \frac{\mathbf{t}_c + w_c \Delta\mathbf{t}_c}{\|\mathbf{t}_c + w_c \Delta\mathbf{t}_c\|}$
    - **Design Motivation**: Rather than modifying original model parameters (which risks forgetting training knowledge), new knowledge is accumulated in a residual manner.

2. **Adaptive Update Weights**:

    - **Function**: Controls the magnitude of updates for each compositional prototype.
    - **Mechanism**: $w_c = \sigma(-\theta \cdot \cos(f^v, \mathbf{t}_c))$. Updates are suppressed when image-prototype similarity is high (likely seen compositions) and amplified when similarity is low (likely unseen compositions).
    - **Design Motivation**: Seen compositions already have well-calibrated prototypes requiring little adjustment, whereas unseen compositions need larger corrections.

3. **Dynamic Priority Queue + Visual Prototypes**:

    - **Function**: Constructs visual-modality prototypes from historical test images.
    - **Mechanism**: A priority queue of length $K=3$ is maintained per composition, storing image features with the lowest prediction entropy (highest confidence). The visual prototype is the mean of features in the queue.
    - **Design Motivation**: Textual prototypes encode label semantics while visual prototypes capture actual visual patterns—cross-modal complementarity improves discriminability.

### Loss & Training
Test-time optimization objective: $\mathcal{L}_{TOMCAT} = \mathcal{L}_{PE} + \lambda \mathcal{L}_{MCRL}$
- $\mathcal{L}_{PE}$: Prediction entropy minimization, encouraging more confident predictions.
- $\mathcal{L}_{MCRL}$: Multimodal Collaborative Representation Learning, aligning textual and visual prototypes via contrastive learning.

## Key Experimental Results

### Main Results — Closed-World Setting

| Method | UT-Zappos AUC | MIT-States AUC | C-GQA AUC |
|--------|--------------|----------------|-----------|
| CLIP | 5.0 | 11.0 | 1.4 |
| CSP (ICLR'23) | 33.0 | 19.4 | 6.2 |
| Troika (CVPR'24) | 38.9 | 21.0 | 7.6 |
| CDS-CZSL (CVPR'24) | 40.3 | 21.4 | 7.9 |
| ClusPro (ICLR'25) | 39.6 | 22.8 | 7.8 |
| **TOMCAT** | **42.6** | **24.3** | **8.6** |

### Ablation Study

| Configuration | UT-Zappos AUC | MIT-States AUC |
|---------------|--------------|----------------|
| Base model (w/o TOMCAT) | 40.3 | 21.4 |
| + Textual KAM | 41.4 | 22.8 |
| + Visual prototypes | 41.8 | 23.1 |
| + Adaptive weights | 42.1 | 23.6 |
| + MCRL loss (full) | **42.6** | **24.3** |

### Key Findings
- Each component contributes consistently: textual KAM is the most impactful (+1.1 AUC), followed by visual prototypes (+0.4).
- Advantages are more pronounced in the open-world setting, where label distribution shift is more severe.
- Inference latency increase is minimal—KAM updates occur after prediction and do not affect inference speed.
- A queue size of $K=3$ suffices; increasing to 5 yields negligible improvement.

## Highlights & Insights
- **First application of test-time adaptation to CZSL**—naturally resolving the train/test label space mismatch.
- **Intuitive design of adaptive update weights**: fewer updates for seen compositions, more for unseen ones, constituting an implicit difficulty-aware mechanism.
- **Multimodal prototype complementarity**: textual prototypes encode semantic knowledge while visual prototypes capture visual patterns; alignment of the two yields stronger representations.

## Limitations & Future Work
- Pseudo-labels in the priority queue may be noisy, particularly when few test samples have been observed early in the stream.
- KAM parameter count scales linearly with the number of compositions, potentially causing memory issues in large open-world settings (tens of thousands of compositions).
- The online streaming assumption may require adjustment for batch testing scenarios.
- Effectiveness on non-CLIP VLMs has not been verified.

## Related Work & Insights
- **vs. TDA (CVPR'24)**: TDA employs a key-value cache for training-free adaptation but does not consider multimodal alignment; TOMCAT performs explicit alignment.
- **vs. TPT (NeurIPS'22)**: TPT independently optimizes prompts for each test sample without cross-sample accumulation; TOMCAT accumulates knowledge across samples.
- **vs. DynaPrompt (ICLR'25)**: DynaPrompt adaptively selects prompts, whereas TOMCAT directly updates prototypes.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First combination of CZSL and test-time adaptation, with an elegant method design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, both open- and closed-world settings, complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, systematic methodological derivation.
- **Value**: ⭐⭐⭐⭐ Establishes a new paradigm for CZSL; the test-time adaptation approach is transferable to other settings.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](../../CVPR2026/multimodal_vlm/flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)

<!-- RELATED:END -->
