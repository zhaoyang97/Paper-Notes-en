---
title: >-
  [Paper Note] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts
description: >-
  [ICLR 2026][Visual Prompt Tuning] This paper proposes IVPT (Interpretable Visual Prompt Tuning), which associates abstract visual prompts with human-understandable semantic regions via cross-layer class-agnostic concept prototypes. IVPT is the first method to achieve interpretability for visual prompts while preserving the advantages of parameter-efficient fine-tuning, simultaneously improving explanation consistency (+8.4%) and classification accuracy on fine-grained benchmarks such as CUB-200.
tags:
  - ICLR 2026
  - Visual Prompt Tuning
  - Interpretability
  - Concept Prototypes
  - Cross-layer Fusion
  - Fine-grained Classification
date: 2026-05-08
content_hash: 47386aff1dc0f3c9
---

# Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts

**Conference**: ICLR 2026
**arXiv**: [2503.06084](https://arxiv.org/abs/2503.06084)
**Code**: [github.com/ThomasWangY/IVPT](https://github.com/ThomasWangY/IVPT)
**Area**: Interpretability
**Keywords**: Visual Prompt Tuning, Interpretability, Concept Prototypes, Cross-layer Fusion, Fine-grained Classification

## TL;DR
This paper proposes IVPT (Interpretable Visual Prompt Tuning), which associates abstract visual prompts with human-understandable semantic regions via cross-layer class-agnostic concept prototypes. IVPT is the first method to achieve interpretability for visual prompts while preserving the advantages of parameter-efficient fine-tuning, simultaneously improving explanation consistency (+8.4%) and classification accuracy on fine-grained benchmarks such as CUB-200.

## Background & Motivation

**Background**: Visual Prompt Tuning (VPT) has become a mainstream approach for adapting pre-trained vision models to downstream tasks by inserting a small number of learnable tokens at the Transformer input layer. Existing methods such as VPT-Deep, E2VPT, and Gated Prompt Tuning achieve strong performance, but the prompts remain black-box vectors.

**Limitations of Prior Work**: These prompts are unconstrained abstract embeddings that cannot provide human-understandable decision explanations. In safety-critical domains such as medical diagnosis and autonomous driving, the lack of interpretability severely limits the trustworthiness of AI systems. Existing interpretable methods (e.g., ProtoPNet, TesNet) focus solely on the last-layer features and cannot explain multi-layer prompts.

**Key Challenge**: VPT methods learn prompts across multiple Transformer layers, yet existing prototype-based methods can only explain single-layer features; moreover, prior methods learn class-specific prototypes and cannot analyze concepts shared across categories.

**Goal**
   - Associate abstract prompt embeddings with human-understandable visual concepts
   - Achieve cross-layer interpretability of prompts across multiple network layers
   - Learn class-agnostic shared concept prototypes

**Key Insight**: Each prompt is defined as an aggregated feature of a semantic region in the image (rather than an arbitrary vector), where the region is discovered by concept prototypes via an attention mechanism. Shallower layers use more prototypes to capture fine-grained features, while deeper layers use fewer prototypes to capture coarse-grained semantics.

**Core Idea**: Replace black-box prompt vectors with cross-layer class-agnostic concept prototypes. Each prompt is bound to an interpretable semantic region in the image through a concept region discovery mechanism followed by intra-region feature aggregation.

## Method

### Overall Architecture
Input: Pre-trained ViT + image
Intermediate: Cross-layer concept prototypes → Concept Region Discovery (CRD) → Intra-region Feature Aggregation (IFA) → Cross-layer prompt fusion
Output: Interpretable prompt embeddings → Classification head → Conditional class scores per concept → Averaged to produce the final prediction

### Key Designs

1. **Concept Region Discovery (CRD)**

    - Function: Associates each concept prototype $\mathbf{q}_k$ with a specific semantic region in the image.
    - Mechanism: Computes negative Euclidean distance attention between prototypes and patch embeddings, normalizes via Softmax, and adds a learnable spatial bias to obtain the concept attention map $\mathbf{A}$ for each prototype. Each patch is assigned to the concept with the highest attention, forming a region map $\mathbf{R}$.
    - Key formula: $a_{k,ij} = \frac{\exp(-\|\mathbf{e}_{ij} - \mathbf{q}_k\|^2)}{\sum_l \exp(-\|\mathbf{e}_{ij} - \mathbf{q}_l\|^2)} + b_{k,ij}$
    - Design Motivation: Class-agnostic prototypes capture semantic concepts shared across categories (e.g., "bird wing," "wheel"), revealing the model's learning of general visual concepts more effectively than class-specific prototypes.

2. **Intra-region Feature Aggregation (IFA)**

    - Function: Aggregates patch features within a concept region into the prompt embedding corresponding to that concept.
    - Mechanism: $\mathbf{p}_k = \frac{\sum_{i,j} \mathbf{z}_{k,ij}}{\sum_{i,j} r_{k,ij}}$, i.e., the prompt is the region-probability-weighted mean of patch features.
    - Design Motivation: The prompt is no longer an arbitrary vector but a "representative" of a semantic region, making it directly interpretable.

3. **Cross-layer Concept Prototypes and Fusion**

    - Function: Uses different numbers of prototypes at different Transformer layers—more in shallow layers (17) and fewer in deep layers (8)—and fuses fine-grained shallow-layer prompts into coarse-grained deep-layer prompts.
    - Mechanism: A learnable grouping layer (linear layer + Gumbel-Softmax) groups fine-grained prompts; intra-group means are passed through an MLP to produce deep-layer prompts. A concept region consistency loss $\mathcal{L}_{con}$ (KL divergence) ensures that the combination of fine-grained regions aligns with the coarse-grained regions.
    - Design Motivation: Simulates the "local → global" reasoning process of human visual cognition: shallow layers capture low-level attributes such as texture and color, while deep layers capture high-level semantics such as parts and wholes.

### Loss & Training
- Total loss: $\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{ps}\mathcal{L}_{ps} + \lambda_{con}\mathcal{L}_{con}$
- $\mathcal{L}_{cls}$: Classification cross-entropy (averaged over per-concept conditional scores)
- $\mathcal{L}_{ps}$: Part shaping loss (enforces non-overlapping regions, foreground coverage, connectivity, etc.)
- $\mathcal{L}_{con}$: Cross-layer region consistency loss (KL divergence)
- All $\lambda$ are set to 1; the backbone is frozen and only prompt-related parameters are trained.

## Key Experimental Results

### Main Results (CUB-200-2011, DinoV2-B backbone)

| Method | Consistency (Con.) | Stability (Sta.) | Accuracy (Acc.) |
|--------|--------------------|------------------|-----------------|
| ProtoPNet | 27.6 | 57.0 | 85.8 |
| Huang et al. | 68.6 | 71.4 | 89.9 |
| VPT-Deep | 14.6 | 39.5 | 89.1 |
| VPT-Deep (w/ Proto.) | 70.2 | 72.5 | 90.3 |
| **IVPT** | **75.3** | **75.9** | **90.8** |

IVPT achieves state-of-the-art results across all three dimensions (interpretability + accuracy).

### Ablation Study (DinoV2-B, CUB-200)

| Configuration | Con. | Sta. | Acc. |
|---------------|------|------|------|
| Baseline (last layer only, global attention) | 62.7 | 64.3 | 88.4 |
| + Spatial bias map | 63.5 | 66.7 | 88.7 |
| + Intra-region Feature Aggregation (IFA) | 65.4 | 68.3 | 89.8 |
| + Cross-layer prototypes | 70.4 | 70.9 | 90.5 |
| + Fine-to-coarse prompt fusion | **75.3** | **75.9** | **90.8** |

### Key Findings
- **Cross-layer structure contributes most**: Adding cross-layer prototypes and fusion raises consistency from 65.4 to 75.3 (+9.9), the largest gain among all components.
- **IFA contributes most to accuracy**: Adding IFA alone improves accuracy from 88.7 to 89.8 (+1.1%), indicating that region-conditioned features are more discriminative than global features.
- **Good generalization on PartImageNet and PASCAL-Part**: IVPT achieves consistency scores of 63.2 and 72.6 respectively, substantially surpassing ProtoPool and Huang et al.
- **Human evaluation**: Assessed by 20 participants, IVPT achieves 97.5% concept annotation accuracy, detail preservation 4.7/5, semantic abstraction 4.8/5, and transition naturalness 4.8/5.
- **Medical imaging applicability**: On the Gleason-2019 prostate cancer grading dataset, IVPT effectively identifies key grading features such as glandular lumens and cancerous acini.

## Highlights & Insights
- **First interpretable paradigm for VPT**: Transforming prompts from "black-box vectors" into "semantic representatives of image regions" is an elegant and practical design. Prior interpretability analyses for VPT relied on post-hoc methods (e.g., attention maps); IVPT instead integrates interpretability directly into the prompt construction process.
- **Advantages of class-agnostic prototypes**: Cross-category shared concepts (e.g., "wings" across different bird species, "tail fins" across different aircraft) not only improve explanation consistency but also uncover visually universal concepts across domains, which holds significant value for AI-assisted knowledge discovery.
- **Cross-layer fine-to-coarse fusion mimics human cognition**: Shallow layers capture texture and color, while deep layers capture parts and wholes. Learnable grouping establishes inter-layer relationships consistent with the human visual reasoning process from detail to gestalt.

## Limitations & Future Work
- Concept prototypes rely on in-domain learning; transfer to substantially different new domains requires retraining.
- On smaller backbones such as DinoV2-S, consistency is slightly lower than Huang et al. (−2.2%), suggesting that limited model capacity constrains the simultaneous maintenance of interpretability and accuracy.
- The number of prototypes per layer (17/14/11/8) is a manually set hyperparameter; automatically determining the optimal number could further improve performance.
- **Future directions**: Extending IVPT to text-visual multimodal prompts (e.g., CLIP) to leverage textual semantics for concept discovery.

## Related Work & Insights
- **vs. VPT-Deep (Jia et al., 2022)**: VPT-Deep achieves high accuracy but a consistency score of only 14.6; IVPT achieves a fivefold improvement in interpretability while also outperforming in accuracy (90.8 vs. 89.1).
- **vs. Huang et al. (2023)**: The strongest conventional prototype-based method; IVPT surpasses it by +6.7% consistency on CUB-200 with DinoV2-B, and IVPT is parameter-efficient (frozen backbone), whereas Huang et al. requires full model fine-tuning.
- **vs. Prompt-CAM (Chowdhury et al., 2025)**: Prompt-CAM learns class-specific prompts, cannot analyze cross-category shared concepts, and lacks cross-layer semantic structure.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to integrate interpretability into the VPT framework; the concept prototype → prompt design is entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across multiple backbones, datasets, ablations, and human evaluations, though validation is primarily on fine-grained classification with limited experiments on general classification settings.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, derivations are complete, and visualizations are rich.
- Value: ⭐⭐⭐⭐ Opens a new direction for VPT interpretability with practical significance for AI applications in safety-critical domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](../../ACL2026/interpretability/evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[ICLR 2026\] Layer by layer, module by module: Choose both for optimal OOD probing of ViT](layer_by_layer_module_by_module_choose_both_for_optimal_ood_probing_of_vit.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)

</div>

<!-- RELATED:END -->
