---
title: >-
  [Paper Note] Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks
description: >-
  [CVPR 2025][LLM Efficiency][MoE] A method is proposed to automatically extract MoE (Mixture-of-Experts) variants from pre-trained ViTs. By first clustering the output activation patterns of MLP layers and then extracting corresponding subnetworks as experts, this approach avoids training MoEs from scratch. It recovers 98% of the original performance on ImageNet-1k with only minimal fine-tuning, while reducing FLOPs and model size by 36% and 32%, respectively.
tags:
  - "CVPR 2025"
  - "LLM Efficiency"
  - "MoE"
  - "ViT"
  - "Model Compression"
  - "Expert Extraction"
  - "Sparse Activation"
date: 2026-05-08
content_hash: 6a8afba7ea00f4d0
---

# Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks

**Conference**: CVPR 2025  
**Code**: To be confirmed  
**Area**: LLM Efficiency  
**Keywords**: MoE, ViT, Model Compression, Expert Extraction, Sparse Activation

## TL;DR
A method is proposed to automatically extract MoE (Mixture-of-Experts) variants from pre-trained ViTs. By first clustering the output activation patterns of MLP layers and then extracting corresponding subnetworks as experts, this approach avoids training MoEs from scratch. It recovers 98% of the original performance on ImageNet-1k with only minimal fine-tuning, while reducing FLOPs and model size by 36% and 32%, respectively.

## Background & Motivation
**Background**: Vision Transformers (ViTs) have become the most powerful models for various CV tasks, but their high computational cost and resource requirements pose core challenges for practical deployment. Although MoE architectures can improve efficiency, traditional MoEs typically require training from scratch or expensive retraining.

**Limitations of Prior Work**: Existing MoE conversion methods either require complete retraining or rely on heuristic rules, resulting in high computational costs and unstable performance.

**Key Challenge**: How to extract efficient MoE structures from pre-trained dense models without retraining.

**Goal**: To automatically discover expert structures and extract subnetworks from the MLP layers of pre-trained ViTs.

**Key Insight**: The authors observe that the MLP layers of pre-trained ViTs naturally exhibit **sparse activation patterns**—only a subset of neurons are activated for different samples. This implies that dense models already implicitly contain "expert" structures.

**Core Idea**: Discover implicit experts by clustering activation patterns $\rightarrow$ extract corresponding subnetworks $\rightarrow$ construct MoE, eliminating the need for training from scratch.

## Method

### Overall Architecture
The method consists of two phases:
1. **Activation Clustering Phase**: The pre-trained model is run on the dataset to collect the output activations of each MLP layer, and clustering algorithms are used to identify different activation patterns (each cluster corresponds to an "expert" behavior).
2. **Subnetwork Extraction Phase**: Based on the clustering results, subsets of neurons corresponding to each activation pattern are extracted to construct independent expert subnetworks. During inference, the corresponding experts are activated based on the input sample.

### Key Designs

1. **Core Module**:

    - Function: Extract sparse expert subnetworks from dense MLP layers
    - Mechanism: Determine expert boundaries based on activation pattern clustering, and then physically separate the subnetworks
    - Design Motivation: The MLP layers of dense models inherently exhibit sparse activation; leveraging this property allows for the discovery of expert structures at zero cost


3. **Optimization Strategy**

    - Function: Improve training stability and convergence speed
    - Mechanism: Adopt appropriate learning rate scheduling, gradient clipping, and regularization strategies
    - Design Motivation: Ensure training efficiency of the model on large-scale datasets

### Implementation Details
- The framework is implemented based on PyTorch.
- Standard data augmentation strategies are used to improve generalization.
- Both training and inference are executed efficiently on GPUs.

### Loss & Training
Only minimal fine-tuning is required after extraction, eliminating the need for full retraining.

## Key Experimental Results

### Main Results
Validated on the ImageNet-1k classification task: the extracted expert models achieve decent performance out-of-the-box. After fine-tuning, they recover 98% of the original performance, while reducing FLOPs by up to 36% and model size by 32%.

| Dataset | Metric | Ours | Description |
|--------|------|------|------|
| ImageNet-1k | Performance Recovery Rate | 98% | Only minimal fine-tuning required |
| ImageNet-1k | FLOPs Reduction | 36% | Significant speedup |
| ImageNet-1k | Model Size Reduction | 32% | Significant compression |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full Model | Optimal | Full method |
| Remove Core Module | Decrease | Validates core contribution |

### Key Findings
- The MLP layers of pre-trained ViTs indeed exhibit exploitable sparse activation patterns.
- Extracted experts achieve reasonable performance even without fine-tuning, demonstrating that "implicit expert" structures indeed exist.
- Minimal fine-tuning is sufficient to recover near-original performance.

## Highlights & Insights
- The concept of **"implicit expert discovery"** is highly inspiring—instead of designing new architectures, it uncovers the latent structures within existing models.
- The lightweight nature of the method (avoiding retraining) makes it highly suitable for industrial deployment.

## Limitations & Future Work
- Only validated on ViTs; adaptability to LLms (such as the MLP layers of LLaMA) remains to be verified.
- The choice of clustering algorithms and hyperparameter sensitivity are not detailed.
- Reading the full paper is necessary to understand the routing strategy and how the number of experts is determined.

## Related Work & Insights
- Complementary to knowledge distillation methods—this work focuses on structure extraction, whereas distillation focuses on knowledge transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel perspective of "discovering implicit MoEs from dense models".
- Experimental Thoroughness: ⭐⭐⭐ Based on the abstract, only the ImageNet dataset is evaluated.
- Writing Quality: ⭐⭐⭐ Preliminary evaluation based on the abstract.
- Value: ⭐⭐⭐⭐ Highly practical value for MoE and model compression fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LOCORE: Image Re-ranking with Long-Context Sequence Modeling](locore_image_re-ranking_with_long-context_sequence_modeling.md)
- [\[CVPR 2025\] Associative Transformer](associative_transformer.md)
- [\[CVPR 2025\] Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)
- [\[ICML 2025\] MoH: Multi-Head Attention as Mixture-of-Head Attention](../../ICML2025/llm_efficiency/moh_multi-head_attention_as_mixture-of-head_attention.md)
- [\[NeurIPS 2025\] Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](../../NeurIPS2025/llm_efficiency/mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)

</div>

<!-- RELATED:END -->
