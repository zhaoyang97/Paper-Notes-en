---
title: >-
  [Paper Note] Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks
description: >-
  [CVPR 2025][LLM Efficiency][Mixture of Experts] A post-training method is proposed to extract MoE variants from pre-trained ViTs. By automatically discovering expert structures using HDBSCAN to cluster MLP hidden layer activation patterns, it reduces MACs by 36% and parameters by 32% on ImageNet-1k while preserving 98% of the original accuracy without retraining.
tags:
  - "CVPR 2025"
  - "LLM Efficiency"
  - "Mixture of Experts"
  - "ViT Compression"
  - "Post-Training Extraction"
  - "HDBSCAN Clustering"
  - "Sparse Activation"
date: 2026-05-08
content_hash: 02d30be01caf4907
---

# Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks

**Conference**: CVPR 2025  
**arXiv**: [2505.15414](https://arxiv.org/abs/2505.15414)  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: Mixture of Experts, ViT Compression, Post-Training Extraction, HDBSCAN Clustering, Sparse Activation

## TL;DR

A post-training method is proposed to extract MoE variants from pre-trained ViTs. By automatically discovering expert structures using HDBSCAN to cluster MLP hidden layer activation patterns, it reduces MACs by 36% and parameters by 32% on ImageNet-1k while preserving 98% of the original accuracy without retraining.

## Background & Motivation

**Background**: ViTs showcase excellent performance but have high computational demands. MoE can improve inference efficiency but requires training from scratch or expensive load balancing losses.

**Limitations of Prior Work**: (1) Conventional MoEs require training from scratch or training on large-scale datasets; (2) The number and size of experts must be selected manually; (3) Sparse activation discovery in language Transformers cannot be directly transferred to vision Transformers (spatial structure vs. sequence structure).

**Core Idea**: The MLP layers of pre-trained ViTs naturally possess sparse activation patterns, which can be discovered through clustering to extract corresponding subnetworks as experts.

## Method

### Key Designs

1. **Activation Clustering (Phase 1)**: Record MLP hidden layer activations, and apply HDBSCAN clustering (which automatically determines the number and shape of clusters) to each layer independently. Layers without clusters remain unchanged.

2. **Expert Extraction (Phase 2)**: Rank the importance of hidden neurons using intra-cluster variance, and extract subnetworks based on a cumulative variance percentage of $p\%$. Map back to the input space to compute the mean input vector for each cluster, which is used for routing.

3. **Inference Routing**: Computes the cosine similarity between a new token and the mean input vector of each expert, and routes the token to the most similar expert. Experts can overlap, and neurons not used by any expert are permanently removed.

### Loss & Training

The extraction process requires no training. Optional light fine-tuning can be used to restore accuracy. Routing overhead is negligible ($k \approx 10 \ll 3e$ hidden layer dimension).

## Key Experimental Results

### Main Results

| Model | MACs Reduction | Parameter Reduction | Accuracy Preserved |
|------|----------|---------|---------|
| DeiT-S | 29.0% | 20.1% | ~97% |
| DeiT-B | 36.0% | 32.0% | 98% |

### Key Findings
- Middle layers of ViT show a stronger pattern of specialization than shallow/deep layers.
- A small amount of fine-tuning (a few epochs) can recover most of the accuracy.
- There is significant overlap between experts, indicating that certain neurons participate in multiple functions.

### Hierarchical Specialization Analysis

| Layer Position | Number of Experts (HDBSCAN) | Activation Sparsity | Compression Potential |
|--------|----------------|----------|--------|
| Shallow Layers (1-3) | 2-3 | Low | Low |
| Middle Layers (4-8) | 5-8 | **High** | **High** |
| Deep Layers (9-12) | 3-4 | Medium | Medium |


- Middle layers of ViT show a stronger pattern of specialization than shallow/deep layers.
- A small amount of fine-tuning (a few epochs) can recover most of the accuracy.
- There is significant overlap between experts, indicating that certain neurons participate in multiple functions.

## Highlights & Insights

- Data-driven expert configuration avoids manual hyperparameter tuning.
- The automatic number of clusters identified by HDBSCAN is particularly suitable for this scenario.
- The method can utilize any existing pre-trained model without retraining.

## Limitations & Future Work

- The spherical clustering assumption may not always hold; complex activation patterns may require non-spherical clustering.
- Currently only validated on ImageNet classification; downstream tasks like detection and segmentation remain unexplored.
- Sparse activation patterns can vary across different architectures and tasks, which requires re-analysis each time.
- The hyperparameters of HDBSCAN (e.g., minimum cluster size) may affect performance, and a sensitivity analysis is lacking.
- Overlap between experts implies an upper bound on the compression rate, as some neurons participate in multiple functions and cannot be uniquely assigned to a single expert.
- Comparisons with structured pruning methods (e.g., SparseGPT, Wanda) are missing.
- The routing strategy is relatively simple (cosine similarity); more sophisticated routing might improve performance.
- Validated only on ViTs; expanding to MLP layers of LLMs requires handling a much larger activation space.

## Related Work & Insights
- **vs Switch Transformer/GShard**: Training MoE from scratch requires load balancing loss and large-scale data; MoEE extracts from pre-trained models without retraining.
- **vs Knowledge Distillation**: Knowledge distillation requires a teacher-student framework; MoEE directly extracts expert structures from model activation patterns.
- **vs Token Pruning (ToMe/EViT)**: Token pruning reduces token sequence length, while MoEE reduces the computational cost per token, and the two can be complementary.
- Writing Quality: 7/10

### Methodological Insights
- The core contribution of this work lies in introducing a new architecture to this field, revealing new technical possibilities.
- The experimental design covers a variety of baselines and scenarios, and the conclusions show statistical significance.
- The components of the method can be independently replaced, facilitating subsequent improvements and optimizations.
- It is highly compatible with the existing technical ecosystem, lowering the barrier to adoption.
- It provides an adjustable trade-off between computational efficiency and generation quality.
- The open-sourced code and model weights hold significant value for community reproduction.
- Driven by practical application needs, it fosters technological innovation with a clearly defined problem.
- Thorough comparative analysis with contemporary related work provides clear positioning.
- Future work can explore lighter variants to adapt to edge device deployment.
- Cross-modality and cross-task transferability are important directions for future validation.
- The integration with self-supervised learning and contrastive learning is worth exploring.
- Efficiency and cost optimization in large-scale deployment are critical for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Associative Transformer](associative_transformer.md)
- [\[CVPR 2025\] Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)
- [\[CVPR 2025\] LOCORE: Image Re-ranking with Long-Context Sequence Modeling](locore_image_re-ranking_with_long-context_sequence_modeling.md)
- [\[ACL 2025\] GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture](../../ACL2025/llm_efficiency/gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)
- [\[ICML 2025\] MoH: Multi-Head Attention as Mixture-of-Head Attention](../../ICML2025/llm_efficiency/moh_multi-head_attention_as_mixture-of-head_attention.md)

</div>

<!-- RELATED:END -->
