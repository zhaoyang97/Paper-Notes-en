---
title: >-
  [Paper Note] Improving Zero-Shot Generalization for CLIP with Variational Adapter
description: >-
  [ECCV 2024][Model Compression][CLIP] A Prompt-based Variational Adapter (PVA) is proposed to separate base and novel category samples in the latent space through a variational adapter. A divide-and-conquer strategy is adopted to process them separately, combined with residual connections to enhance the transfer capability of novel categories, achieving state-of-the-art performance on generalized zero-shot learning and cross-dataset transfer learning benchmarks.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "CLIP"
  - "Zero-Shot Generalization"
  - "Variational Adapter"
  - "Divide-and-Conquer Strategy"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 6663b3c9bb39bfc9
---

# Improving Zero-Shot Generalization for CLIP with Variational Adapter

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Model Compression  
**Keywords**: CLIP, Zero-Shot Generalization, Variational Adapter, Divide-and-Conquer Strategy, Vision-Language Models

## TL;DR

A Prompt-based Variational Adapter (PVA) is proposed to separate base and novel category samples in the latent space through a variational adapter. A divide-and-conquer strategy is adopted to process them separately, combined with residual connections to enhance the transfer capability of novel categories, achieving state-of-the-art performance on generalized zero-shot learning and cross-dataset transfer learning benchmarks.

## Background & Motivation

**Background**: Pre-trained vision-language models (VLMs) such as CLIP have demonstrated excellent zero-shot generalization capabilities. The current mainstream approach is to fine-tune CLIP for downstream tasks via prompt tuning or adapters, achieving high specialized performance on base categories while maintaining generalization to novel (unseen) categories.

**Limitations of Prior Work**: Most existing fine-tuning methods, while pursuing high performance on base categories, inevitably lead to "feature confusion" for novel categories. Specifically, the fine-tuned model confuses novel category samples with certain base categories, because the feature space is distorted during fine-tuning, blurring the semantic boundaries of novel categories. This base-novel performance trade-off has been a core challenge in zero-shot generalization.

**Key Challenge**: Fine-tuning specializes the model on base categories, but this specialization comes at the expense of generality for novel categories. The fundamental reason is that fine-tuning simultaneously processes predictions of both base and novel categories in a unified feature space, causing the model to incorrectly pull novel sample features toward the decision regions of base categories.

**Goal**: (1) How to identify and separate confused base and novel samples? (2) How to process the two groups of samples separately after separation to eliminate prediction bias? (3) How to enhance the transfer capability of novel categories while maintaining the performance on base categories?

**Key Insight**: The authors observe that the distribution differences between base and novel samples in the latent representation space can be modeled. Variational inference can learn a structured latent space where the distributions of the two types of samples can be explicitly distinguished. If the confused samples can be separated, a difficult mixed task can be decomposed into two independent and simpler sub-tasks.

**Core Idea**: Use a variational adapter to learn a structured latent space, separate novel samples from the confused space based on the similarity of latent features, and then apply a divide-and-conquer approach.

## Method

### Overall Architecture

PVA adds two lightweight variational adapters directly onto the frozen CLIP model—one for the visual modality and one for the textual modality. Each adapter contains learnable text tokens and a variational inference module. During the training phase, the adapters are trained on base category data to learn the mapping of input features to a shared, structured latent space. During the inference phase, the feature similarity metric in the latent space is utilized to divide the input samples into "base-like" and "novel-like" groups, which are then processed with different prediction strategies.

### Key Designs

1. **Bimodal Variational Adapters**:

    - **Function**: Align visual and textual features into a shared, structured latent space.
    - **Mechanism**: Each adapter consists of an encoder network and learnable text prompt tokens. The encoder maps the input features (visual or textual features from CLIP) to the mean $\mu$ and variance $\sigma^2$ of the latent space, and then samples the latent representation $z = \mu + \sigma \cdot \epsilon$ via the reparameterization trick. The latent representations of both modalities are aligned in the same space. Learnable prompt tokens are embedded into CLIP's text encoder to provide task-related semantic guidance for the adapter.
    - **Design Motivation**: Variational inference allows the model to learn probability distributions in the latent space rather than deterministic mappings. This probabilistic modeling naturally provides uncertainty estimation—the distribution of novel category samples in the latent space differs from that of base categories, allowing them to be detected and separated. The shared latent space ensures cross-modal alignment.

2. **Base-Novel Sample Separation Mechanism**:

    - **Function**: Inside the inference phase, divide input samples into base-like and novel-like groups.
    - **Mechanism**: After training, the similarity between the latent representation of each test sample and all base category latent representation prototypes is calculated. If the similarity is higher than a threshold, the sample is identified as base-like; otherwise, it is novel-like. Similarity is measured using cosine distance or KL divergence in the latent space. After separation, base-like samples are predicted using the fine-tuned classifier, and novel-like samples are predicted using the classifier that retains generalization capability.
    - **Design Motivation**: This "divide-and-conquer" strategy decomposes a difficult joint classification problem into two relatively simple sub-problems. It fully leverages the specialization of fine-tuning for base-like samples while preserving CLIP's original generalization ability for novel-like samples.

3. **Feature Enhancement via Residual Connections**:

    - **Function**: Improve the transfer performance on novel categories.
    - **Mechanism**: For novel-like samples, the latent features output by the adapter are fused with the original global features of CLIP via residual connections: $f_{out} = f_{adapter} + \lambda \cdot f_{CLIP}$, where $\lambda$ controls the strength of the residual connection. This ensures that predictions for novel categories incorporate both the structured information learned by the adapter and the powerful zero-shot representation pre-trained by CLIP.
    - **Design Motivation**: Relying solely on the adapter may discard the general semantic information learned during CLIP pre-training, while relying solely on the original CLIP features wastes downstream task knowledge learned by the adapter. Residual connections strike a balance between the two.

### Loss & Training

The training loss includes: (1) cross-entropy classification loss $L_{CE}$ for base category classification learning; (2) KL divergence regularization $L_{KL}$ to constrain the latent space distribution closely to the prior (standard normal distribution); and (3) cross-modal alignment loss to ensure consistency between visual and textual latent representations. The model is trained only on base category data, and novel category data is completely unseen during training.

## Key Experimental Results

### Main Results

| Benchmark Dataset | Metric | PVA | CoCoOp | MaPLe | Gain |
|-----------|------|-----|--------|-------|------|
| ImageNet (Base) | Acc | 77.5 | 75.98 | 76.66 | +0.84 |
| ImageNet (Novel) | Acc | 71.8 | 70.43 | 70.54 | +1.26 |
| ImageNet (HM) | Acc | 74.5 | 73.10 | 73.47 | +1.03 |
| 11 Dataset Average (Base) | Acc | 80.2 | 79.74 | 80.18 | +0.02 |
| 11 Dataset Average (Novel) | Acc | 74.6 | 71.69 | 72.46 | +2.14 |
| 11 Dataset Average (HM) | Acc | 77.3 | 75.83 | 76.18 | +1.12 |

### Ablation Study

| Configuration | Base Acc | Novel Acc | HM | Description |
|------|----------|-----------|-----|------|
| Full PVA | 77.5 | 71.8 | 74.5 | Full Model |
| w/o Variational Inference | 77.2 | 69.5 | 73.2 | Changed to deterministic mapping, Novel drops significantly |
| w/o Separation Mechanism | 77.8 | 68.9 | 73.1 | Processing all samples uniformly, Novel drops substantially |
| w/o Residual Connection | 76.8 | 70.1 | 73.3 | Novel drops slightly |

### Key Findings

- The separation mechanism is the most critical design. Removing the separation mechanism leads to a drop of about 2.9% in Novel Acc, indicating that the "divide-and-conquer" strategy is crucial for addressing the base-novel bias.
- Variational inference outperforms deterministic mapping, indicating that a probabilistic latent space significantly assists in detecting novel samples.
- PVA achieves more substantial improvements on novel categories compared to base categories (+2.14 vs +0.02), demonstrating that the method indeed resolves the feature confusion of novel categories.
- It also performs excellently in cross-dataset transfer (from ImageNet to 10 other datasets), demonstrating the generalization ability of the method.

## Highlights & Insights

- **Innovative application of the "divide-and-conquer" concept in zero-shot learning**: Explicitly separating confused base and novel samples and processing them individually is simple yet effective, and more direct than attempting to balance base-novel in an end-to-end manner. This strategy can be transferred to other scenarios involving the co-existence of known and unknown categories, such as open-set recognition and continual learning.
- **Variational inference as a novel detector**: Ingeniously utilizes the probabilistic modeling capability of variational inference to detect novel samples—having only observed base categories during training, novel samples naturally present different distribution characteristics in the latent space. This is more elegant than setting a hard threshold or employing an external OOD detection module.
- **Complementarity with prompt tuning**: The core of PVA is not learning better prompts, but learning a better inference strategy (separation + divide-and-conquer), which complements existing prompt tuning methods.

## Limitations & Future Work

- The choice of the separation threshold may require validation set tuning, which might be impractical in fully zero-shot scenarios.
- The method assumes sufficient distribution differences between novel and base category samples in the latent space, which may fail for highly similar base-novel category pairs.
- The two variational adapters increase model parameters and training complexity. Although the adapters themselves are lightweight, the sampling process of variational inference may affect inference speed.
- The evaluation is limited to classification tasks and has not been validated on downstream tasks such as detection or segmentation.

## Related Work & Insights

- **vs CoCoOp**: CoCoOp enhances generalization by conditioning learnable prompts, but still processes base and novel categories in a unified space. PVA's explicit separation strategy addresses the bias problem more directly.
- **vs PLOT**: PLOT uses optimal transport to align textual and visual prompts. PVA employs variational inference for alignment, gaining additional probabilistic separation capabilities.
- **vs Tip-Adapter**: Tip-Adapter constructs a cache model as an adapter. PVA's variational adapter learns a more structured representation space.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The "divide-and-conquer" strategy to solve the base-novel bias is highly novel, and the design of the variational adapter is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both generalized zero-shot learning and cross-dataset transfer scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem analysis, fully motivated methodology.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution to the base-novel trade-off in CLIP fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Logits DeConfusion with CLIP for Few-Shot Learning](../../CVPR2025/model_compression/logits_deconfusion_with_clip_for_few-shot_learning.md)
- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)
- [\[NeurIPS 2025\] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels](../../NeurIPS2025/model_compression/enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)
- [\[ICLR 2026\] Boomerang Distillation Enables Zero-Shot Model Size Interpolation](../../ICLR2026/model_compression/boomerang_distillation_enables_zero-shot_model_size_interpolation.md)
- [\[ACL 2025\] Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](../../ACL2025/model_compression/graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)

</div>

<!-- RELATED:END -->
