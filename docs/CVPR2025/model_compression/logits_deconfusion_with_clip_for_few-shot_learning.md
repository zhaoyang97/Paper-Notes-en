---
title: >-
  [Paper Note] Logits DeConfusion with CLIP for Few-Shot Learning
description: >-
  [CVPR 2025][Model Compression][few-shot learning] It is observed that CLIP suffers from severe inter-class confusion in logits under downstream tasks. To resolve this, this paper proposes Logits DeConfusion (LDC), which enhances feature representations through Multi-level Adapter Fusion (MAF) and integrates an Inter-Class Deconfusion (ICD) module to learn and eliminate confusion patterns via a residual architecture, achieving SOTA results across 11 benchmarks.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "few-shot learning"
  - "CLIP"
  - "logits deconfusion"
  - "inter-class confusion"
  - "adapter fusion"
  - "residual learning"
date: 2026-05-08
content_hash: bd31bf516e338dd6
---

# Logits DeConfusion with CLIP for Few-Shot Learning

**Conference**: CVPR 2025  
**arXiv**: [2504.12104](https://arxiv.org/abs/2504.12104)  
**Code**: [LiShuo1001/LDC](https://github.com/LiShuo1001/LDC)  
**Area**: Model Compression  
**Keywords**: few-shot learning, CLIP, logits deconfusion, inter-class confusion, adapter fusion, residual learning

## TL;DR

It is observed that CLIP suffers from severe inter-class confusion in logits under downstream tasks. To resolve this, this paper proposes Logits DeConfusion (LDC), which enhances feature representations through Multi-level Adapter Fusion (MAF) and integrates an Inter-Class Deconfusion (ICD) module to learn and eliminate confusion patterns via a residual architecture, achieving SOTA results across 11 benchmarks.

## Background & Motivation

**Background**: CLIP establishes powerful vision-language alignment via large-scale image-text contrastive learning, performing exceptionally well in zero-shot and few-shot learning. Subsequent methods such as CoOp and Tip-Adapter improve adaptability through prompt learning or adapters.

**Limitations of Prior Work**: Since CLIP's pre-training strategy is contrastive learning rather than directly optimizing classification boundaries, logits suffer from significant inter-class confusion in downstream tasks—making it difficult to accurately distinguish predicted values across different categories. This issue is particularly severe when class similarity is high or domain shift is large.

**Key Challenge**: The domain discrepancy between CLIP's pre-training data distribution and downstream tasks leads to blurry classification boundaries, and it is challenging to learn a reliable classifier under few-shot settings.

**Key Insight**: Instead of modifying CLIP's feature representations, this work directly models and eliminates confusion patterns at the logits level.

**Core Idea**: Inter-class confusion is treated as a learnable noise term $\Delta s$, which is eliminated through a residual structure: $\hat{s}_i = s_i^{ZS} - \Delta s(x_i)$.

## Method

### Overall Architecture

1. **ZS-CLIP**: Frozen CLIP generates zero-shot logits $s_i^{ZS}$.
2. **MAF (Multi-level Adapter Fusion)**: Extracts features from four levels of the image encoder, fuses them into an enhanced feature $z_i^e$, and then generates MAF logits $s_i^{MAF}$ via an MLP.
3. **ICD (Inter-Class Deconfusion)**: Leverages $z_i^e$ as a prior to learn the confusion patterns from $s_i^{ZS}$ and eliminates them via a residual connection, outputting ICD logits $s_i^{ICD}$.
4. **ALF (Adaptive Logits Fusion)**: Adaptively weights and fuses MAF and ICD logits via a learned weight $\alpha$ to obtain the final prediction.

### Key Designs

**1. Multi-level Adapter Fusion (MAF)**
- **Function**: Extracts features $f_i^1, f_i^2, f_i^3, f_i^4$ from four different levels of the CLIP image encoder, transforms them through independent adapters, and fuses them into a unified representation $z_i^e$.
- **Mechanism**: Provides two fusion mechanisms—Weighted Fusion (WF: weights 0.1:0.2:0.3:0.4) and Learnable Fusion (LF: concatenation followed by dimensionality reduction via an adapter). The fused features are processed by a frozen projector (attention pooling for ResNet, linear projection for ViT) to produce enhanced features.
- **Design Motivation**: Low-level features contain detailed information, while high-level features contain semantic information. Multi-level fusion provides a more comprehensive feature representation in few-shot scenarios, enhancing generalization capability.

**2. Inter-Class Deconfusion Module (ICD)**
- **Function**: Learns inter-class confusion patterns within logits and eliminates them with residuals via three cascaded adapters: $s_i^{ICD} = s_i^{ZS} + \mathcal{E}_{A_3}^{ICD}(\mathcal{E}_{A_1}^{ICD}(s_i^{ZS}) + \mathcal{E}_{A_2}^{ICD}(z_i^e))$.
- **Mechanism**: $\mathcal{E}_{A_1}$ extracts confusion clues from zero-shot logits, and $\mathcal{E}_{A_2}$ extracts confusion priors from enhanced visual features. The two paths are summed and forwarded to $\mathcal{E}_{A_3}$ for joint learning of confusion patterns, which is finally removed via a residual structure.
- **Design Motivation**: Experimental observations reveal that CLIP exhibits fixed inter-class confusion patterns for each category; visual features supply prior information about "what the class should be," directing the accurate mapping of confusion patterns.

**3. Adaptive Logits Fusion (ALF)**
- **Function**: Generates an adaptive weight $\alpha$ from the enhanced feature $z_i^e$ using an $\alpha$ Generator to fuse the two branches of logits: $s_i^{ALF} = \alpha \cdot s_i^{MAF} + (1-\alpha) \cdot s_i^{ICD}$.
- **Mechanism**: Dynamically adjusts the ratio between the visual feature logits and the deconfused logits for different samples.
- **Design Motivation**: MAF logits are based on pure visual features, whereas ICD logits are based on text-aligned deconfused logits; they are complementary, but the optimal weights vary per sample.

### Loss & Training

- Three-way cross-entropy loss: $\mathcal{L}_{CE} = \mathcal{L}_{CE}^{MAF} + \mathcal{L}_{CE}^{ICD} + \mathcal{L}_{CE}^{ALF}$
- Two-way similarity regularization: $\mathcal{L}_{Sim} = \|s_i^{MAF} - s_i^{ZS}\|_1 + \|s_i^{ICD} - s_i^{ZS}\|_1$ (to prevent over-deconfusion)
- Total loss: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Sim}$, with $\lambda = 1.0$
- AdamW optimizer, initial learning rate of 0.001, 50 epochs, batch size 64
- Input size of 224×224, random resized crop + horizontal flip

## Key Experimental Results

### Main Results—Average Accuracy on 11 Datasets

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp | 59.80 | 62.21 | 66.84 | 70.05 | 73.45 |
| Tip-Adapter-F | 64.55 | 66.79 | 69.76 | 72.59 | 75.69 |
| APE | 65.13 | 67.19 | 69.47 | 71.58 | 73.36 |
| Proto-CLIP-F | 61.84 | 65.96 | 68.29 | 73.13 | 76.18 |
| **LDC (Ours)** | **65.71** | **67.92** | **71.17** | **75.79** | **79.78** |
| Gain vs. runner-up | +0.58 | +0.73 | +1.41 | +2.66 | +3.60 |

### ImageNet Dataset

| Method | 1-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|
| Tip-Adapter-F | 61.32 | 62.52 | 64.00 | 65.51 |
| FAR | 60.80 | 62.40 | 64.30 | 66.39 |
| **LDC (Ours)** | 60.48 | 62.47 | **64.44** | **66.63** |

### Key Findings

1. **Widening gap as the number of shots increases**: The average gain on 11 datasets increases from +0.58% at 1-shot to +3.60% at 16-shot, showing that more samples enable the deconfusion module to learn more precise confusion patterns.
2. **At 1-shot on ImageNet, it is slightly inferior to APE but surpasses it after 8-shot**: This reflects that the deconfusion module requires a minimum number of samples to reliably estimate confusion patterns.
3. **Inter-class confusion is the primary bottleneck for CLIP FSL**: Experimental visualizations demonstrate that after eliminating confusion, the class discriminability of logits is significantly enhanced.
4. **Extremely efficient training**: The total training and testing time across 11 datasets in the 16-shot setting takes only 37 minutes on a single 4090D GPU.

## Highlights & Insights

- Precise problem formulation: Reproducible inter-class confusion patterns are identified from logits confusion matrices and elegantly resolved via residual learning.
- Multi-level feature fusion provides a more robust representation foundation for few-shot scenarios.
- The design of L1 similarity regularization to prevent over-deconfusion is well-thought-out.
- Adaptive fusion weights grant flexibility to the method across various samples.
- The overall method is lightweight and efficient, needing only to train a small number of adapter parameters.

## Limitations & Future Work

- The advantage is minor in extreme 1-shot scenarios, as the estimation of confusion patterns remains unstable.
- Main experiments are mostly conducted on the ResNet-50 backbone, leaving exploration of the ViT backbone insufficient.
- The robustness of preset values for $\lambda$ and fusion weights is not systematically analyzed.
- The scalability of ICD remains unanalyzed when the number of classes is extremely large (e.g., 1000+).
- Adaptability in open-vocabulary scenarios (newly emerging categories) is not explored.

## Related Work & Insights

- Tip-Adapter implements training-free adaptation with a cache model but does not address confusion; this work approaches the problem from a complementary perspective.
- CoOp/CoCoOp adapt to downstream tasks via prompt learning, which represents an orthogonal direction to logits-level deconfusion.
- Insight: Other vision-language models (e.g., BLIP, SigLIP) might also present similar logits confusion issues.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The identified problem is valuable, and the residual deconfusion framework is novel; however, the design of the core modules (Adapter + residual) is relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage with 11 datasets + OOD generalization + multi-shot settings.
- **Writing Quality**: ⭐⭐⭐ Technical descriptions are clear, but some expressions are redundant.
- **Value**: ⭐⭐⭐⭐ Unveils the key bottleneck in CLIP FSL and provides an effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/model_compression/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ECCV 2024\] Improving Zero-Shot Generalization for CLIP with Variational Adapter](../../ECCV2024/model_compression/improving_zero-shot_generalization_for_clip_with_variational_adapter.md)
- [\[CVPR 2025\] InsTaG: Learning Personalized 3D Talking Head from Few-Second Video](instag_learning_personalized_3d_talking_head_from_few-second_video.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/model_compression/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)

</div>

<!-- RELATED:END -->
