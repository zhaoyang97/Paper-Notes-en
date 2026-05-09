---
title: >-
  [Paper Note] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval
description: >-
  [ACL 2026][Multimodal VLM][Composed Image Retrieval] This paper proposes TEMA (Text-oriented Entity Mapping Architecture), the first CIR framework designed for multi-modification text (MMT). It enhances entity coverage via a Parsing Assistant (PA), resolves clause-entity misalignment via an Entity Mapping (EM) module, and introduces two multi-modification benchmarks—M-FashionIQ and M-CIRR—achieving state-of-the-art performance in both standard and multi-modification settings.
tags:
  - ACL 2026
  - Multimodal VLM
  - Composed Image Retrieval
  - Multi-Modification Text
  - Entity Mapping
  - Fine-Grained Retrieval
  - Vision-Language Pre-training
date: 2026-05-08
content_hash: 813c86e206371c42
---

# TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval

**Conference**: ACL 2026
**arXiv**: [2604.21806](https://arxiv.org/abs/2604.21806)
**Code**: [https://github.com/lee-zixu/ACL26-TEMA/](https://github.com/lee-zixu/ACL26-TEMA/)
**Area**: Image Retrieval / Multimodal
**Keywords**: Composed Image Retrieval, Multi-Modification Text, Entity Mapping, Fine-Grained Retrieval, Vision-Language Pre-training

## TL;DR

This paper proposes TEMA (Text-oriented Entity Mapping Architecture), the first CIR framework designed for multi-modification text (MMT). It enhances entity coverage via a Parsing Assistant (PA), resolves clause-entity misalignment via an Entity Mapping (EM) module, and introduces two multi-modification benchmarks—M-FashionIQ and M-CIRR—achieving state-of-the-art performance in both standard and multi-modification settings.

## Background & Motivation

**Background**: Composed Image Retrieval (CIR) uses multimodal queries—a reference image paired with modification text—to retrieve target images. Existing methods have achieved notable progress under settings with short modification texts covering only a few salient changes.

**Limitations of Prior Work**: Existing CIR settings have two limitations highly relevant to real-world applications: (1) *Insufficient entity coverage*—when multiple entities require modification, training signals concentrate on salient regions and miss certain entities (the proportion of modification texts that explicitly reference all entities to be modified is low); (2) *Clause-entity misalignment*—in practice, multiple modification clauses may constrain the same entity (e.g., simultaneously modifying the hem, shoulder trim, and belt of a skirt), or a single clause may constrain multiple entities of the same category (e.g., changing three golden retrievers to huskies).

**Key Challenge**: Existing CIR models suffer a sharp performance drop when facing multi-modification queries, as training data lacks multi-modification annotations and models cannot establish many-to-one clause-entity correspondences.

**Goal**: (1) Construct multi-modification CIR benchmark datasets closer to real-world scenarios; (2) Design the first CIR framework that handles both standard and multi-modification settings.

**Key Insight**: Address the problem at both the data and model levels—generate MMT via MLLMs with human verification to build datasets, while designing dedicated modules for multi-entity coverage and clause aggregation.

**Core Idea**: Extract lists of entities to be modified via LLM-generated summaries, aggregate multiple modification clauses targeting the same entity into a unified representation using learnable queries, and align these representations with corresponding visual entities.

## Method

### Overall Architecture

TEMA consists of two core components: (1) a **Parsing Assistant (PA)**, comprising an LLM-based text summarizer and a consistency detector that extracts entities to be modified and verifies entity coverage during training (disabled at inference); and (2) an **MMT-guided Entity Mapping (EM) module**, which aggregates multiple MMT clauses targeting the same entity through text- and vision-side entity mapping under summary guidance. BLIP serves as the feature extraction backbone.

### Key Designs

1. **Parsing Assistant (PA)**:

    - **Function**: Enhances exposure and coverage of entities to be modified during training.
    - **Mechanism**: An LLM (gpt-3.5-turbo) generates a summary $t_s$ for each MMT, explicitly required to include all entities to be modified. A consistency detector (also an LLM) then verifies whether the summary covers all such entities without introducing spurious ones; if not, the summary is iteratively refined. Summary features $\mathbf{E}_s = \Phi_\mathbb{T}(t_s)$ are extracted via the frozen BLIP text encoder.
    - **Design Motivation**: Entity information in MMT is sparse and dispersed, making it easy for models to overlook certain entities. Explicitly extracting entity lists via summaries provides clear guidance for subsequent entity mapping. PA is disabled at inference to avoid additional dependencies and latency.

2. **MMT-Guided Entity Mapping (EM) Module**:

    - **Function**: Aggregates multiple modification clauses targeting the same entity to resolve many-to-one clause-entity correspondences.
    - **Mechanism**: Learnable queries $\mathbf{a}_q = \{a_1, \ldots, a_k\}$ are introduced and fed into a Transformer alongside summary features $\mathbf{E}_s$ and MMT local features $\mathbf{E}_m^l$. Since the summary contains all entities to be modified with minimal detail, the learnable queries aggregate the corresponding MMT clauses for each entity under summary guidance: $\hat{\mathbf{a}}_q = \text{Transformer}([\mathbf{E}_s, \mathbf{E}_m^l, \mathbf{a}_q])$. On the visual side, analogous aggregation is applied to produce visual entity features $\hat{\mathbf{b}}_q$ from the reference image.
    - **Design Motivation**: Global features cannot distinguish modification requirements across different entities. Through learnable queries and Transformer attention, the model adaptively aggregates clauses such as "change shoulder trim to lace" and "change hem to irregular" into the same entity channel.

3. **Summary-Guided Distillation + Orthogonal Regularization**:

    - **Function**: Ensures that text tokens generated by EM retain complete entity information while different entity channels remain distinct.
    - **Mechanism**: A summary-guided distillation strategy aligns EM-generated text tokens closely with entities parsed by PA. Orthogonal regularization ensures different learnable query channels attend to different entities, preventing redundancy.
    - **Design Motivation**: Without distillation constraints, EM may lose partial entity information; without orthogonal constraints, multiple channels may collapse onto the same entity.

### Loss & Training

BLIP is used as the backbone with the image encoder frozen. The AdamW optimizer is used with a learning rate of 2e-5, batch size 64, feature dimension 256, and $N=3$ learnable query channels. The total loss comprises three terms: batch-based classification loss (contrastive learning), summary-guided distillation loss, and orthogonal regularization loss. The PA module is used only during training and disabled at inference. All experiments are conducted on a single NVIDIA A40 48GB GPU.

## Key Experimental Results

### Main Results

**Performance on M-FashionIQ and M-CIRR (R@K %)**

| Method | M-FashionIQ Avg R@10 | M-FashionIQ Avg R@50 | M-CIRR Avg |
|--------|---------------------|---------------------|------------|
| TIRG | 9.20 | 18.05 | 22.83 |
| BLIP4CIR | 40.99 | 62.44 | 70.92 |
| BLIP4CIR+Bi | 40.78 | 62.05 | 72.54 |
| Candidate | 47.38 | 66.71 | 72.75 |
| **TEMA (Ours)** | **50.59** | **72.09** | **75.76** |

### Ablation Study

| Configuration | M-FashionIQ R@10 | Δ | M-CIRR Avg | Δ |
|---------------|-----------------|---|------------|---|
| Full TEMA | 50.59 | - | 75.76 | - |
| w/o PA | 47.80 | -2.79 | 71.59 | -4.17 |
| w/o CD (consistency detection) | 49.14 | -1.45 | 73.87 | -1.89 |
| w/o EM | 45.41 | -5.18 | 70.99 | -4.77 |
| w/o EM_txt | 46.11 | -4.48 | 71.20 | -4.56 |
| w/o EM_img | 46.17 | -4.42 | 71.64 | -4.12 |
| w/o Summ (distillation) | 49.40 | -1.19 | 74.16 | -1.60 |
| w/o Ortho (orthogonal) | 49.38 | -1.21 | 75.02 | -0.74 |

### Key Findings

- The EM module contributes the most: removing it causes R@10 to drop by 5.18, indicating that clause-entity alignment is the core bottleneck in multi-modification CIR.
- Text-side and vision-side entity mapping are equally important; removing either leads to approximately 4.5-point degradation.
- The PA consistency detector contributes significantly (1.45-point drop upon removal), confirming that LLM summary hallucinations do affect downstream performance.
- VLP-based methods with BLIP backbones substantially outperform traditional ResNet+LSTM architectures, underscoring the importance of pre-trained language understanding in multi-modification settings.
- TEMA also achieves state-of-the-art performance on standard CIR benchmarks (FashionIQ, CIRR), with no performance sacrifice on simpler settings due to the multi-modification design.

## Highlights & Insights

- **Precise problem formulation**: The paper is the first to formally characterize the two core challenges of multi-modification CIR (insufficient entity coverage and clause-entity misalignment), providing corresponding data and model solutions.
- **Training-time augmentation with inference-time disabling**: The PA design is practically appealing—it avoids LLM dependency at inference while retaining the benefits of structured entity guidance during training.
- The learnable-query-as-entity-proxy design is transferable to other multimodal tasks requiring multi-entity aggregation.

## Limitations & Future Work

- The BLIP text encoder's token length constraint precludes the use of a CLIP backbone, limiting fair comparison with a broader range of methods.
- The number of learnable query channels $N$ is fixed at 3, which may be insufficiently flexible for scenarios with highly variable entity counts.
- Dataset construction relies on MLLM generation, which may introduce systematic biases.
- Future work may explore dynamic channel allocation and end-to-end entity discovery mechanisms.

## Related Work & Insights

- **vs. BLIP4CIR**: BLIP4CIR uses global feature composition and cannot handle multi-entity scenarios; TEMA achieves fine-grained entity-level alignment via the EM module.
- **vs. FineCIR**: FineCIR parses modification semantics but does not guarantee coverage of all entities to be modified; TEMA explicitly ensures entity coverage through PA's consistency detection.
- **vs. Cola/MagicLens**: These works address multi-object interference but do not solve the aggregation problem of multi-modification clauses.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to formulate the multi-modification CIR problem and provide a complete data + model solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, thorough ablations, and comparisons against multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem formulation and intuitive method diagrams.
- **Value**: ⭐⭐⭐⭐ — Fills a gap in multi-modification CIR; both the datasets and the method offer strong practical value.

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
