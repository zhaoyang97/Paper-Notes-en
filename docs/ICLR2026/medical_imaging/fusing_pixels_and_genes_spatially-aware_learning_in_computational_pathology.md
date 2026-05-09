---
title: >-
  [Paper Note] Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology
description: >-
  [ICLR 2026][Medical Imaging][Spatial Transcriptomics] This paper proposes the Stamp framework, which leverages spatial transcriptomics gene expression data as a supervisory signal. Through spatially-aware gene encoder pretraining and hierarchical multi-scale contrastive alignment, it enables joint representation learning of pathology images and spatial transcriptomics data, achieving state-of-the-art performance across 4 downstream tasks on 6 datasets.
tags:
  - ICLR 2026
  - Medical Imaging
  - Spatial Transcriptomics
  - Computational Pathology
  - Multimodal Pretraining
  - Gene Expression
  - Contrastive Learning
date: 2026-05-08
content_hash: 09f26bcac80bf287
---

# Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology

**Conference**: ICLR 2026
**arXiv**: [2602.13944](https://arxiv.org/abs/2602.13944)
**Code**: [https://github.com/Hanminghao/STAMP](https://github.com/Hanminghao/STAMP)
**Area**: Medical Imaging / Computational Pathology
**Keywords**: Spatial Transcriptomics, Computational Pathology, Multimodal Pretraining, Gene Expression, Contrastive Learning

## TL;DR

This paper proposes the Stamp framework, which leverages spatial transcriptomics gene expression data as a supervisory signal. Through spatially-aware gene encoder pretraining and hierarchical multi-scale contrastive alignment, it enables joint representation learning of pathology images and spatial transcriptomics data, achieving state-of-the-art performance across 4 downstream tasks on 6 datasets.

## Background & Motivation

**State of the Field**: Foundation models in computational pathology (CPATH) are evolving from unimodal (purely visual self-supervised pretraining) to multimodal approaches. Methods such as PLIP and CONCH align pathology images with natural language descriptions via image-text contrastive learning. TANGLE further incorporates bulk RNA-seq gene expression data to guide whole-slide image (WSI) representation learning.

**Limitations of Prior Work**: Natural language lacks molecular-level specificity and cannot provide deep pathological supervision. For instance, a textual description of "invasive ductal carcinoma" conveys no information about which gene pathways are activated. Although bulk RNA-seq provides molecular-level information, it averages gene expression across an entire tissue section, failing to capture intra-sample spatial heterogeneity (e.g., the substantial differences in gene expression between tumor cores and invasive fronts). Existing methods that incorporate spatial transcriptomics (ST) suffer from two key limitations: (1) overly simplistic encoding schemes (linear layers with a small number of genes) that require full-parameter fine-tuning of the visual backbone for each new dataset; and (2) neglect of the inherent spatial multi-scale structure of ST data.

**Root Cause**: ST data simultaneously encodes spatial positional information and gene expression, exhibiting strong spatial dependencies across spots. However, existing methods treat spots as independent samples and directly apply vision-language pretraining paradigms (treating each spot as an independent image-text pair), thereby squandering ST's most distinctive advantage—spatial context.

**Paper Goals**: (1) How to train a gene encoder that is aware of spatial structure? (2) How to achieve effective alignment between pathology images and genes given limited paired data? (3) How to capture multi-scale features relevant to pathological analysis?

**Starting Point**: The authors construct SpaVis-6M, the largest 10X Visium spatial transcriptomics dataset to date (5.75 million entries), and use it to pretrain a spatially-aware gene encoder. This encoder is then jointly trained with a pathology visual encoder via hierarchical multi-scale contrastive alignment. The two-stage strategy reduces dependence on paired data, requiring only 697K paired samples for alignment.

**Core Idea**: A gene encoder is pretrained via spatial neighborhood sampling and contextual gene reconstruction, then aligned with a visual encoder through cross-scale localization and hierarchical contrastive learning, enabling molecular supervision-driven pathology image representation learning.

## Method

### Overall Architecture

Stamp employs a two-stage pretraining pipeline. **Stage 1**: A spatially-aware gene encoder is pretrained on SpaVis-6M to learn gene co-expression patterns and spatial dependencies. **Stage 2**: On 697K pathology image–gene expression paired data (HEST dataset), the gene encoder is aligned with a visual encoder (UNI, ViT-L/16) via hierarchical multi-scale contrastive alignment. The outputs include gene embeddings (Stamp_G), visual embeddings (Stamp_V), and fused embeddings (Stamp_F).

### Key Designs

1. **Anomaly-Rank-Based Gene Tokenization**:

    - **Function**: Transforms high-dimensional, sparse gene expression data into a stable token sequence.
    - **Mechanism**: The average non-zero expression level of each gene across all samples is first computed, and each sample's gene expression is then divided by the corresponding mean for normalization. Crucially, rather than using the normalized values directly (which are susceptible to batch effects), the top $N=1500$ gene IDs are selected by ranking their normalized deviations in descending order to form the token sequence: $T_i = \{id(ep_i^0), id(ep_i^1), \ldots, id(ep_i^{N-1}) : ep_i^k \geq ep_i^{k+1}\}$. Genes with zero expression naturally rank last and are excluded.
    - **Design Motivation**: Rank-based tokenization is inherently robust to batch effects (ranks are more stable than absolute values) and naturally handles data sparsity—undetected genes never enter the token sequence.

2. **Spatially-Aware Pretraining (Dual IGR + CGR Loss)**:

    - **Function**: Enables the gene encoder to simultaneously learn intra-spot gene co-expression patterns and inter-spot spatial dependencies.
    - **Mechanism**: A neighborhood-centered sampling strategy constructs spatially coherent mini-batches (Algorithm 2)—starting from a random seed spot and iteratively incorporating neighboring spots by nearest-neighbor search. Two training objectives are employed: (a) **Intrinsic Gene Reconstruction (IGR)**: 15% of tokens are randomly masked, and the masked tokens are reconstructed from the unmasked tokens of the same spot, with loss $\mathcal{L}_{IGR} = -\frac{1}{|M|}\sum_{j \in M} \log P(t_{i,j} | x_{i,L-1})$; (b) **Contextual Gene Reconstruction (CGR)**: the aggregated features of neighboring spots $h_i = \frac{1}{|N(s_i)|}\sum_{k \in N(s_i)} x_{i,L-1}^k$ are used to predict the masked genes of the center spot. The gene encoder is a 12-layer Transformer.
    - **Design Motivation**: IGR captures intrinsic gene-gene expression associations (e.g., co-regulatory networks), while CGR is grounded in the biological prior that a spot's transcriptional state is highly correlated with its microenvironment, thereby compelling the model to encode tissue spatial structure.

3. **Hierarchical Multi-Scale Contrastive Alignment**:

    - **Function**: Aligns pathology images with gene expression while modeling cross-scale feature relationships.
    - **Mechanism**: The alignment stage incorporates four losses: (a) **Cross-Scale Patch Localization** $\mathcal{L}_{CSP}$: simulates pathologists' zoom-in/zoom-out workflow by treating a patch as a sub-region within a $3 \times 3$ regional grid; a "pretext token" is introduced to enable a shared visual encoder to process both patch and region inputs, with CE loss used to predict the patch's position within the region; (b) **Patch–Gene Contrastive Alignment** $\mathcal{L}_{P-S}$: standard symmetric InfoNCE loss; (c) **Region–Gene Contrastive Alignment** $\mathcal{L}_{R-S}$; (d) **Patch–Region Intra-Modal Alignment** $\mathcal{L}_{P-R}$: expands the visual encoder's receptive field while preventing the representation collapse associated with BERT-style methods. Total alignment loss: $\mathcal{L}_{Align} = \mathcal{L}_{CSP} + \mathcal{L}_{P-S} + \mathcal{L}_{R-S} + \mathcal{L}_{P-R}$.
    - **Design Motivation**: Directly applying vision-language pretraining ignores ST's spatial characteristics. Cross-scale localization establishes spatial relationships between patches and regions, while intra-modal alignment leverages multi-scale redundancy to enhance representation robustness.

### Loss & Training

Gene encoder pretraining loss: $\mathcal{L}_{Gene} = \mathcal{L}_{IGR} + \mathcal{L}_{CGR}$, trained for 1 epoch with $\mathcal{L}_{IGR}$ alone, followed by 1 epoch with CGR incorporated. Alignment pretraining loss: $\mathcal{L}_{Align} = \mathcal{L}_{CSP} + \mathcal{L}_{P-S} + \mathcal{L}_{R-S} + \mathcal{L}_{P-R}$, trained for 30 epochs. AdamW optimizer with learning rate $10^{-4}$, using 4 × A800 GPUs.

## Key Experimental Results

### Linear Probing and Unsupervised Clustering (DLPFC + HBC Datasets)

| Model | Pretraining Modality | DLPFC Bal.Acc | DLPFC ARI | HBC Bal.Acc | HBC ARI |
|------|-----------|-------------|-----------|-------------|---------|
| UNI | Vision | 0.544 | 0.144 | 0.859 | 0.499 |
| Hoptimus0 | Vision | 0.568 | 0.147 | 0.816 | 0.458 |
| CONCH | Vision + Language | 0.454 | 0.124 | 0.704 | 0.406 |
| mSTAR | Vision + Language + Gene | 0.540 | 0.159 | 0.869 | 0.505 |
| scGPT-Spatial | Gene | 0.558 | 0.215 | 0.610 | 0.208 |
| **Stamp_G** | Gene | **0.658** | **0.369** | 0.659 | 0.416 |
| **Stamp_V** | Vision + Gene | 0.624 | 0.246 | **0.872** | **0.526** |
| **Stamp_F** | Fusion | **0.721** | **0.342** | **0.899** | **0.590** |

On DLPFC, Stamp_F achieves 15.3% higher Bal.Acc and 13.3× higher ARI compared to the strongest unimodal visual model Hoptimus0.

### Gene Expression Prediction (PSC, HHK, HER2+ Datasets)

| Method | Trainable Parameters | PSC MSE↓ | PSC PCC-V↑ | HHK MSE↓ | HER2+ MSE↓ |
|------|-----------|----------|-----------|----------|-----------|
| STNet | 12.08M | 0.330 | 0.110 | 1.357 | 1.190 |
| EGN | 146.02M | 0.345 | 0.094 | 1.321 | 1.112 |
| Stamp (linear probing) | Few parameters | **Lowest** | **Highest** | **Lowest** | **Lowest** |

Stamp surpasses dedicated models requiring full-parameter training using only a frozen visual encoder with linear probing.

### Key Findings

- **Gene supervision substantially enhances visual representations**: On the DLPFC dataset, PLIP and CONCH fine-tuned with ST data show marked improvements across all clustering metrics (ARI from 0.128 to 0.174), confirming the value of molecular supervision.
- **Critical role of spatial context**: Under the same architecture, adding the CGR loss (which leverages neighborhood information) improves DLPFC ARI from 0.233 to 0.369 (+58%) over Stamp_G† trained with IGR alone, validating the necessity of spatially-aware pretraining.
- **Cross-platform generalization**: Despite being trained exclusively on 10X Visium data, Stamp achieves top performance on the HER2+ dataset acquired with a different sequencing platform, demonstrating strong generalizability.
- **Complementarity of fused embeddings**: Stamp_G and Stamp_V each excel on different datasets (gene modality stronger on DLPFC, visual modality stronger on HBC); their fusion (Stamp_F) achieves the best results on both.

## Highlights & Insights

- **Significant dataset contribution**: SpaVis-6M is the largest Visium spatial transcriptomics dataset to date, covering 35 organs, 1,982 sections, and 262 datasets/publications, representing an important community resource.
- **Elegance of gene tokenization**: Replacing normalized numerical values with rank-based ordering addresses batch effects and data sparsity simultaneously in a single design choice, and interfaces naturally with BERT's sequence paradigm.
- **Practical two-stage strategy**: The gene encoder is pretrained on 5.75 million unpaired samples, while the alignment stage requires only 700K paired samples, substantially reducing reliance on expensive paired data.
- **Pretext Token design**: A single learnable token enables the same visual encoder to switch between patch and region processing modes, avoiding the overhead of maintaining two separate encoders.

## Limitations & Future Work

- **Resolution constraints**: The 55 μm resolution of 10X Visium corresponds to multiple cells, precluding sub-cellular precision and potentially limiting the capture of single-cell-level heterogeneity.
- **Restricted to pathology images**: The framework focuses exclusively on H&E-stained sections and does not explore other imaging modalities such as IHC or fluorescence staining.
- **Limited depth of downstream task evaluation**: Although four task types are covered, the framework has not been validated on the most clinically relevant tasks such as clinical prognosis prediction or treatment response prediction.
- **Lack of comparison with newer visual backbones**: UNI (ViT-L/16) is used as the visual backbone, with no comparison against results obtained using more recent models such as Virchow2 or Hoptimus0 as the backbone.
- **Insufficient discussion of training costs**: The total compute required for 5.75M-sample gene pretraining combined with 700K-sample alignment training is not reported.

## Related Work & Insights

- **vs. TANGLE (Jaume et al., 2024)**: TANGLE uses bulk RNA-seq to guide WSI representation learning, capturing only patient-level information. Stamp uses spatial transcriptomics to preserve spatial heterogeneity and performs spot-level rather than WSI-level alignment.
- **vs. CONCH (Lu et al., 2024)**: CONCH aligns images with text (1.17M pairs), but text lacks molecular specificity. Stamp replaces text supervision with gene expression, improving ARI on DLPFC from 0.124 to 0.342.
- **vs. OmiCLIP (Chen et al., 2025)**: OmiCLIP converts gene features into text sentences before alignment, introducing indirection and information loss. Stamp aligns genes and images directly in embedding space, achieving greater efficiency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First large-scale spatial transcriptomics–pathology image multimodal pretraining framework; spatially-aware design is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, four task types, multiple evaluation metrics, comprehensive ablation studies and comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Framework is clearly presented with detailed descriptions of data and methods; notation and equations are somewhat dense.
- **Value**: ⭐⭐⭐⭐⭐ Both the dataset and the method make important contributions, potentially advancing computational pathology from image-text alignment toward a new paradigm of image-gene alignment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[CVPR 2026\] A protocol for evaluating robustness to H&E staining variation in computational pathology models](../../CVPR2026/medical_imaging/a_protocol_for_evaluating_robustness_to_he_stainin.md)
- [\[NeurIPS 2025\] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](../../NeurIPS2025/medical_imaging/revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](../../AAAI2026/medical_imaging/dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[CVPR 2026\] Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](../../CVPR2026/medical_imaging/better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)

<!-- RELATED:END -->
