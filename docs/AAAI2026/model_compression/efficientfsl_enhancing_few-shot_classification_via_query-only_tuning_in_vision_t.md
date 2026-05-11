---
title: >-
  [Paper Note] EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers
description: >-
  [AAAI 2026][Model Compression][Few-Shot Learning] This paper proposes EfficientFSL, a query-only parameter-efficient fine-tuning framework for ViT-based few-shot classification. Through three components — the Forward Blo…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Few-Shot Learning"
  - "Parameter-Efficient Fine-Tuning"
  - "Vision Transformer"
  - "Query-Only Tuning"
  - "Prototypical Networks"
date: 2026-05-08
content_hash: 106de9e82dde5b03
---

# EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers

**Conference**: AAAI 2026
**arXiv**: [2601.08499](https://arxiv.org/abs/2601.08499)
**Code**: N/A
**Area**: Model Compression
**Keywords**: Few-Shot Learning, Parameter-Efficient Fine-Tuning, Vision Transformer, Query-Only Tuning, Prototypical Networks

## TL;DR
This paper proposes EfficientFSL, a query-only parameter-efficient fine-tuning framework for ViT-based few-shot classification. Through three components — the Forward Block (decoupled active/frozen sub-blocks), the Combine Block (adaptive multi-layer feature fusion), and the SQ Attention Block (support-query distribution alignment) — EfficientFSL achieves state-of-the-art performance on 4 in-domain and 6 cross-domain benchmarks using only 1.25M–2.48M trainable parameters.

## Background & Motivation

### State of the Field
Few-shot learning (FSL) requires models to recognize new categories from very few labeled examples. Recent ViT-based methods have achieved significant performance gains by replacing CNNs with large-scale pretrained visual models. The common paradigm involves pretraining on large-scale datasets followed by adaptation to few-shot tasks.

### Limitations of Prior Work

**High cost of full fine-tuning**: ViT-B contains 85.8M parameters; full fine-tuning requires storing a complete copy of parameters per task, incurring substantial GPU memory and training time overhead.

**Coupling issues in PETL methods**: Existing parameter-efficient methods (Adapter, LoRA, Prompt Tuning, etc.) adapt frozen backbones by inserting small modules — but these modules **modify the feature flow** and are **coupled** to the backbone weights, making them prone to overfitting under the data-scarce FSL setting.

**Support-query distribution shift**: In FSL, support and query images often differ in background, lighting, and viewpoint, causing prototypes computed from the support set to deviate from the distributional center of the query set.

### Root Cause
Large model + scarce data = high risk of overfitting. The fundamental challenge is how to leverage pretrained knowledge while achieving task adaptation with minimal parameters and avoiding overfitting.

### Starting Point

**Query-only paradigm**: The backbone is fully frozen without any modification to the feature flow. Instead, lightweight query modules are introduced to **selectively extract** task-relevant information from the backbone's intermediate representations. This is fundamentally different from conventional PETL methods that "modify the backbone" — EfficientFSL performs "side-channel querying" rather than "internal modification."

## Method

### Overall Architecture
EfficientFSL takes intermediate representations from each layer of a pretrained ViT as input and processes them through the following sequential components:
1. **Forward Block** (×n layers): each layer contains an Active Block (learning task-specific knowledge) + a Frozen Block (querying pretrained knowledge)
2. **Combine Block**: adaptive fusion of multi-layer features
3. **SQ Attention Block**: prototype adjustment to align with query set distribution
4. **PN Classifier**: prototypical network classification based on cosine similarity

### Key Designs

1. **Forward Block — Active Block**:

    - **Function**: A lightweight network trained from scratch to learn task-specific knowledge and generate adaptive queries.
    - **Core Structure**:
        - Input $H_{i-1}$ combined with a trainable prompt $P_i$ is projected through a bottleneck layer to produce $Z_i$
        - $Z_i$ is processed by a self-attention layer and MLP; all projection layers use a bottleneck structure (hidden dimension of only 8 or 48)
    - **Key Equations**:
    $Z_i = \text{Proj}(H_{i-1} + P_i)$
    $Z'_i = \xi \cdot \text{Att}(Q^A_i, K^A_i, V^A_i) + Z_i$
    $F_i = \zeta \cdot \text{MLP}(\text{LN}(Z'_i)) + Z'_i$
    - **Design Motivation**:
        - Prompt $P_i$ provides per-layer task guidance signals
        - Scaling factors $\xi, \zeta$ control the intensity of task knowledge injection
        - The bottleneck structure (attention head dimension of only 8) drastically reduces parameter count

2. **Forward Block — Frozen Block**:

    - **Function**: Reuses and freezes the pretrained ViT parameters to extract knowledge from the backbone in a query-only manner.
    - **Mechanism**:
        - **Query** comes from the Active Block output $F_i$ (task-specific)
        - **Key and Value** come from the $i$-th layer output $X_i$ of the pretrained ViT (general knowledge)
        - This achieves **decoupling**: task knowledge → queries; general knowledge → keys/values
    - **Key Equations**:
    $F^{att}_i = \text{Att}'(Q^F_i, K^F_i, V^F_i) + F_i$
    $F^{mlp}_i = \text{MLP}'(\text{LN}(F^{att}_i))$
    $H_i = F^{mlp}_i + F^{att}_i$
    - **Design Motivation**: Freezing the backbone weights fully preserves pretrained general knowledge and prevents overfitting on few-shot data; the query-only mode allows task-specific queries to selectively "retrieve" relevant information.

3. **Combine Block (Multi-Layer Feature Fusion)**:

    - **Function**: Adaptively fuses three types of features from all layers ($F^{att}_i, F^{mlp}_i, H_i$) into a unified representation.
    - **Mechanism**:
        - All features are projected into a unified space via a shared bottleneck MLP (alignment)
        - Conditioned on the final layer $H_n$, a Weight MLP generates adaptive fusion weights
    $$F^{agg} = \sum_{i=1}^{n}(w^{att}_i \cdot \hat{F}^{att}_i + w^{mlp}_i \cdot \hat{F}^{mlp}_i + w^H_i \cdot \hat{H}_i)$$
    - **Design Motivation**: Different tasks may require features at different granularities — shallow-layer texture information vs. deep-layer semantic information. Condition-based adaptive weights outperform simple averaging or fixed weighting.

4. **SQ Attention Block (Support-Query Alignment)**:

    - **Function**: Adjusts prototype positions to better approximate the distributional center of corresponding query samples.
    - **Core Equation**:
    $s^{att} = \alpha(s \cdot \text{Proj}(q)^T) \cdot q + (1-\alpha) \cdot s$
    - **Mechanism**:
        - A learnable projection is applied to query $q$ for class-aware alignment
        - Attention weights $s \cdot \text{Proj}(q)^T$ guide prototypes to shift toward associated query directions
        - Mixing coefficient $\alpha$ controls the adjustment magnitude
    - **Design Motivation**: In FSL, the support set is very small (1–5 shots), so computed prototypes may deviate from the true class center due to randomness. SQ Attention dynamically corrects this using query set information.

### Loss & Training
- Optimizer: AdamW, learning rate 0.0001, cosine schedule
- 5 training epochs, batch size 64
- Data augmentation: resize 256 → center crop 224
- All experiments conducted on a single NVIDIA V100 GPU

## Key Experimental Results

### Main Results

**In-domain few-shot classification** (ViT-B, ImageNet-21K pretrained):

| Dataset | Setting | EfficientFSL | FewVS (MM'24) | MetaFormer (ICML'24) | SemFew (CVPR'24) |
|--------|------|-------------|---------------|---------------------|-----------------|
| miniImageNet | 5w-1s | **98.34** | 86.80 | 84.78 | 78.94 |
| miniImageNet | 5w-5s | **99.12** | 90.32 | 91.39 | 86.49 |
| tieredImageNet | 5w-1s | **93.27** | 87.87 | 88.38 | 82.37 |
| FC100 | 5w-1s | **80.13** | 61.01 | 58.04 | 54.27 |
| FC100 | 5w-5s | **88.81** | 70.37 | 70.80 | 65.02 |

**Parameter count comparison**: EfficientFSL uses only 2.48M parameters, compared to 88.0M for SemFew and 21.7M for FewVS.

**Comparison with PETL methods** (ViT-B, FC100):

| Method | Accuracy (%) | Training Time (s/epoch) | Peak Memory (GB) | Inference Speed (img/s) |
|------|--------|------------------|-------------|----------------|
| Adapter | 67.48 | 74.19 | 1.09 | 134.21 |
| AdaptFormer | 68.83 | 76.87 | 1.19 | 129.98 |
| LoRA | 74.19 | 76.35 | 1.10 | 131.84 |
| **EfficientFSL** | **80.13** | **59.62** | **1.08** | **134.29** |

### Ablation Study

**Training module removal ablation** (ViT-B, FC100, 21K pretrained):

| Configuration | Params (M) | 1-shot | 5-shot | Notes |
|------|---------|--------|--------|------|
| w/o Proj layer | 1.58 | 51.15 | 68.95 | Severe performance drop; Proj is nearly indispensable |
| w/o Att & MLP | 1.05 | 72.15 | 87.81 | Params halved; slight degradation |
| w/o Combine Block | 2.37 | 75.55 | 88.60 | Multi-scale fusion is beneficial |
| **Full model** | **2.48** | **80.13** | **88.81** | Best performance |

**SQ Attention Block ablation**:

| Configuration | mini-1s | FC100-1s | FC100-5s |
|------|---------|----------|----------|
| w/o SQ Attention | 95.95 | 75.42 | 87.82 |
| w/o Proj(q) | 98.34 | 80.13 | 88.81 |
| **Full** | **98.49** | **80.20** | **89.53** |

**Feature fusion strategy comparison** (FC100):

| Fusion Method | 1-shot | 5-shot | Notes |
|----------|--------|--------|------|
| Simple average | 69.58 | 85.38 | Equal weight across all layers |
| Fixed weights | 69.97 | 85.49 | Learnable but input-agnostic |
| **Conditional weights** | **80.13** | **88.81** | Dynamically generated from final layer; significantly superior |

### Key Findings
1. **Remarkable parameter efficiency**: Using only ~3% of the parameters of full fine-tuning (2.48M vs. 85.8M), EfficientFSL substantially outperforms full fine-tuning on all benchmarks.
2. **Near-perfect performance on miniImageNet**: 5w-5s accuracy reaches 99.12%, leaving virtually no room for further improvement.
3. **Strong cross-domain generalization**: Outperforms existing SOTA on all 6 cross-domain datasets (CUB, Cars, Places, Plantae, EuroSAT, CropDiseases).
4. **Comprehensively superior efficiency**: Fastest training speed, lowest memory footprint, and fastest inference speed compared to Adapter/LoRA/AdaptFormer.
5. **Proj layer is critical**: Removing the projection layer causes a dramatic drop in 1-shot accuracy from 80.13% to 51.15%, underscoring the importance of high-quality feature mapping.
6. **Conditional weighting far surpasses simple fusion**: Adaptive weights improve absolute accuracy by ~10% over uniform averaging.

## Highlights & Insights
1. **Core value of the query-only paradigm**: By entirely avoiding modifications to the backbone's feature flow, EfficientFSL circumvents the coupled overfitting problem inherent in PETL methods — a gentler and safer knowledge extraction strategy.
2. **Decoupled design**: The division of labor between the Active Block (learning what to query) and the Frozen Block (preserving pretrained knowledge) is principled and well-motivated.
3. **Inspirational nature of SQ Attention**: Using query set information to retrospectively correct prototypes is well-grounded in the FSL setting, where prototypes estimated from $K$ samples are inherently biased.
4. **Convincing t-SNE visualizations**: Feature distributions evolve from scattered → coarse clustering → clear separation, with each added module producing visually evident improvement.
5. **Extreme parameter efficiency**: The combination of bottleneck structures and weight sharing compresses the parameter count to a minimum while maintaining high performance.

## Limitations & Future Work
1. The 99%+ accuracy on miniImageNet suggests the benchmark may be approaching saturation; more challenging few-shot benchmarks are needed to differentiate methods.
2. Validation is limited to classification tasks; extension to more complex few-shot settings such as detection and segmentation remains unexplored.
3. The SQ Attention Block introduces minimal additional parameters (0.07M), and its performance gain is relatively modest (~0.7%), yielding a limited cost-benefit ratio.
4. The method requires intermediate representations from all ViT layers as input, introducing additional computational and memory overhead at inference time.
5. The conditional weighting mechanism depends heavily on the final-layer features; degraded quality in the final layer could negatively affect global fusion.

## Related Work & Insights
- **FewTURE** (NeurIPS'22): An early ViT-based FSL method with 21.7M parameters but only ~68% accuracy.
- **MetaFormer-A** (ICML'24): Meta-learning combined with ViT, 24.5M parameters, 84.78% accuracy.
- **LoRA** (Hu et al. 2022): Low-rank adaptation achieving 74.19% accuracy on FSL, substantially outperformed by EfficientFSL.
- **Insights**: The query-only paradigm is generalizable to few-shot learning in other modalities (e.g., NLP few-shot with LLMs) and to efficient adaptation of multimodal models.

## Rating
- Novelty: ⭐⭐⭐⭐ (The query-only paradigm is novel, though individual components are relatively standard)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4+6 datasets, 3 backbone variants, complete ablations and visualizations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-organized figures and tables)
- Value: ⭐⭐⭐⭐ (Extremely high parameter efficiency, though FSL application scope is relatively narrow)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)
- [\[AAAI 2026\] QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching](quept_quantized_elastic_precision_transformers_with_one-shot_calibration_for_mul.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](../../ACL2026/model_compression/meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
