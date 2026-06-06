---
title: >-
  [Paper Note] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval
description: >-
  [ACL 2026][Multimodal VLM][Composed Image Retrieval] This paper proposes TEMA (Text-oriented Entity Mapping Architecture), the first CIR framework oriented towards multi-modification text. It enhances modified entity cov…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Composed Image Retrieval"
  - "Multi-modification Text"
  - "Entity Mapping"
  - "Fine-grained Retrieval"
  - "Vision-Language Pre-training"
date: 2026-05-08
content_hash: 210ac21870a45fc8
---

# TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval

**Conference**: ACL 2026  
**arXiv**: [2604.21806](https://arxiv.org/abs/2604.21806)  
**Code**: [https://github.com/lee-zixu/ACL26-TEMA/](https://github.com/lee-zixu/ACL26-TEMA/)  
**Area**: Image Retrieval / Multi-modal  
**Keywords**: Composed Image Retrieval, Multi-modification Text, Entity Mapping, Fine-grained Retrieval, Vision-Language Pre-training

## TL;DR

This paper proposes TEMA (Text-oriented Entity Mapping Architecture), the first CIR framework oriented towards multi-modification text. It enhances modified entity coverage via an MMT Parsing Assistant (PA), resolves clause-entity alignment issues through an Entity Mapping (EM) module, and establishes two multi-modification benchmark datasets, M-FashionIQ and M-CIRR, achieving state-of-the-art performance in both original and multi-modification scenarios.

## Background & Motivation

**Background**: Composed Image Retrieval (CIR) uses a "reference image + modification text" multi-modal query to retrieve target images. Existing methods have made significant progress in settings involving short modification text covering only a few salient changes.

**Limitations of Prior Work**: Existing CIR settings have two limitations highly relevant to real-world applications: (1) Insufficient entity coverage—when multiple entities require modification, training signals concentrate on salient regions, missing some entities (the proportion of explicit entity references in modification text is small); (2) Clause-entity misalignment—in practice, multiple modification clauses may constrain the same entity (e.g., modifying the hem, shoulder decoration, and belt of a dress simultaneously), or a single clause may constrain multiple entities of the same type (e.g., changing three Golden Retrievers to Huskies).

**Key Challenge**: Existing CIR models experience a sharp performance drop (a clear performance "cliff" in experiments) when facing multi-modification requirements. The root cause is the lack of multi-modification annotations during training, preventing the model from establishing "one-to-many" clause-entity correspondences.

**Goal**: (1) Construct multi-modification CIR benchmark datasets closer to real-world scenarios; (2) Design the first CIR framework adaptable to both simple and multi-modification scenarios.

**Key Insight**: Address the problem at both data and model levels—generate Multi-Modification Text (MMT) using MLLMs with human verification to build datasets, and design specialized modules to handle multi-entity coverage and clause aggregation.

**Core Idea**: Extract lists of entities to be modified via LLM-generated summaries, aggregate multiple modification clauses for the same entity into a unified representation using learnable queries, and align them with corresponding entities on the visual side.

## Method

### Overall Architecture

TEMA consists of two core components: (1) An MMT Parsing Assistant (PA), including an LLM text summarizer and a consistency detector, used during training to extract target entities and perform coverage checks (disabled during inference); (2) An MMT-oriented Entity Mapping (EM) module, which aggregates multiple MMT clauses for the same entity under summary guidance through text and visual entity mapping. BLIP serves as the underlying feature extraction backbone.

### Key Designs

1.  **MMT Parsing Assistant (PA)**:

    - **Function**: Enhance exposure and coverage of target entities during training.
    - **Mechanism**: Utilize an LLM (gpt-3.5-turbo) to generate a summary $t_s$ for each MMT, requiring the summary to include all target entities. A consistency detector (also an LLM) then verifies if the summary covers all target entities in the MMT without redundant ones, iterating if necessary. Summary features $\mathbf{E}_s = \Phi_\mathbb{T}(t_s)$ are extracted via a frozen BLIP text encoder.
    - **Design Motivation**: Entity information in MMT is sparse and scattered, making it easy for models to ignore. Explicitly extracting entity lists via summaries provides clear guidance for subsequent entity mapping. Disabling PA during inference avoids extra dependencies and latency.

2.  **MMT-oriented Entity Mapping (EM)**:

    - **Function**: Aggregate multiple modification clauses for the same entity to resolve one-to-many clause-entity relationships.
    - **Mechanism**: Introduce learnable queries $\mathbf{a}_q = \{a_1, ..., a_k\}$, which are input into a Transformer along with summary features $\mathbf{E}_s$ and MMT local features $\mathbf{E}_m^l$. Since the summary contains all target entities with minimal detail, learnable queries aggregate corresponding MMT clauses for the same entity under summary guidance: $\hat{\mathbf{a}}_q = \text{Transformer}([\mathbf{E}_s, \mathbf{E}_m^l, \mathbf{a}_q])$. Visual entity features $\hat{\mathbf{b}}_q$ in the reference image are aggregated similarly.
    - **Design Motivation**: Global features cannot distinguish modification requirements for different entities. Through the attention mechanism of learnable queries and Transformers, the model can adaptively aggregate "change shoulder decoration to lace" and "change hem to irregular" into the same entity channel.

3.  **Summary-guided Distillation + Orthogonal Regularization**:

    - **Function**: Ensure text tokens generated by EM retain complete entity information while maintaining differences between different entity channels.
    - **Mechanism**: The summary-guided distillation strategy aligns EM-generated text tokens closely with the target entities parsed by PA. Orthogonal regularization ensures different learnable query channels focus on different entities to avoid redundancy.
    - **Design Motivation**: Without distillation constraints, EM might lose entity information; without orthogonal constraints, multiple channels might collapse onto the same entity.

### Loss & Training

BLIP is used as the backbone with a frozen image encoder. AdamW optimizer (LR 2e-5), batch size 64, feature dimension 256, and $N=3$ learnable query channels. The loss function includes three parts: batch-based classification loss (contrastive learning), summary-guided distillation loss, and orthogonal regularization loss. The PA module is used only during training. All experiments were conducted on a single NVIDIA A40 48GB GPU.

## Key Experimental Results

### Main Results

**Performance on M-FashionIQ and M-CIRR Datasets (R@K %)**

| Method | M-FashionIQ Avg R@10 | M-FashionIQ Avg R@50 | M-CIRR Avg |
|------|---------------------|---------------------|------------|
| TIRG | 9.20 | 18.05 | 22.83 |
| BLIP4CIR | 40.99 | 62.44 | 70.92 |
| BLIP4CIR+Bi | 40.78 | 62.05 | 72.54 |
| Candidate | 47.38 | 66.71 | 72.75 |
| **TEMA (Ours)** | **50.59** | **72.09** | **75.76** |

### Ablation Study

| Config | M-FashionIQ R@10 | Δ | M-CIRR Avg | Δ |
|------|-----------------|---|------------|---|
| Full TEMA | 50.59 | - | 75.76 | - |
| w/o PA | 47.80 | -2.79 | 71.59 | -4.17 |
| w/o CD (Consistency Detection) | 49.14 | -1.45 | 73.87 | -1.89 |
| w/o EM | 45.41 | -5.18 | 70.99 | -4.77 |
| w/o EM_txt | 46.11 | -4.48 | 71.20 | -4.56 |
| w/o EM_img | 46.17 | -4.42 | 71.64 | -4.12 |
| w/o Summ (Distillation) | 49.40 | -1.19 | 74.16 | -1.60 |
| w/o Ortho (Orthogonal) | 49.38 | -1.21 | 75.02 | -0.74 |

### Key Findings

- The EM module contributes the most: dropping R@10 by 5.18 when removed, indicating that clause-entity alignment is the core bottleneck of multi-modification CIR.
- Entity mapping on both text and visual sides is equally important, with removal of either leading to a drop of approximately 4.5.
- The consistency detector in PA contributes significantly (1.45 drop when removed), showing that LLM summary hallucinations impact downstream performance.
- VLP methods with BLIP backbones significantly outperform traditional ResNet+LSTM architectures, highlighting the importance of pre-trained language understanding in multi-modification scenarios.
- TEMA also achieves optimal performance on original CIR datasets (FashionIQ, CIRR) without sacrificing performance in simple scenarios.

## Highlights & Insights

- Precise problem definition—formalizing two core challenges of multi-modification CIR (insufficient entity coverage and clause-entity misalignment) for the first time while providing data and model solutions.
- Practical design—using PA enhancement during training but disabling it during inference avoids LLM dependency during inference and maintains efficiency.
- Learnable queries as "entity proxies"—this design can be transferred to other multi-modal tasks requiring multi-entity aggregation.

## Limitations & Future Work

- Limited by the token length of the BLIP text encoder, CLIP backbones cannot be used, hindering fair comparison with more methods.
- Fixed number of learnable query channels $N=3$ might be inflexible for scenarios with varying entity counts.
- Dataset construction relies on MLLM generation, which may introduce systemic bias.
- Future work could explore dynamic channel allocation and end-to-end entity discovery mechanisms.

## Related Work & Insights

- **vs BLIP4CIR**: BLIP4CIR uses global feature combinations and cannot handle multi-entity scenarios; TEMA achieves fine-grained entity-level alignment through the EM module.
- **vs FineCIR**: FineCIR parses modification semantics but does not guarantee coverage of all target entities; TEMA explicitly ensures coverage via PA's consistency detection.
- **vs Cola/MagicLens**: These works focus on multi-object interference but do not resolve the aggregation of multi-modification clauses.

## Rating

- Novelty: ⭐⭐⭐⭐ First to propose the multi-modification CIR problem with a complete data+model solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four datasets, detailed ablation, and comparison with multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and intuitive architecture diagrams.
- Value: ⭐⭐⭐⭐ Fills the gap in multi-modification CIR; both datasets and methods have significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](../../CVPR2026/multimodal_vlm/recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[AAAI 2026\] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning](../../AAAI2026/multimodal_vlm/heterogeneous_uncertainty-guided_composed_image_retrieval_with_fine-grained_prob.md)
- [\[CVPR 2026\] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](../../CVPR2026/multimodal_vlm/text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[ACL 2026\] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](lami_augmenting_large_language_models_via_late_multi-image_fusion.md)

</div>

<!-- RELATED:END -->
