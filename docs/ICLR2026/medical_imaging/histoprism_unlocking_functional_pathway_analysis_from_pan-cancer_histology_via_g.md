---
title: >-
  [Paper Note] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction
description: >-
  [ICLR 2026][Medical Imaging][Spatial Transcriptomics] This paper proposes HistoPrism, an efficient Transformer architecture that injects cancer-type conditioning via cross-attention to predict pan-cancer gene expression…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Spatial Transcriptomics"
  - "Gene Expression Prediction"
  - "Pan-Cancer"
  - "Pathway Analysis"
  - "Transformer"
date: 2026-05-08
content_hash: 25574ab1498e34e7
---

# HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction

**Conference**: ICLR 2026
**arXiv**: [2601.21560](https://arxiv.org/abs/2601.21560)
**Code**: [GitHub](https://github.com/susuhu/HistoPrism)
**Area**: Medical Imaging / Computational Pathology
**Keywords**: Spatial Transcriptomics, Gene Expression Prediction, Pan-Cancer, Pathway Analysis, Transformer

## TL;DR

This paper proposes HistoPrism, an efficient Transformer architecture that injects cancer-type conditioning via cross-attention to predict pan-cancer gene expression from H&E histology images. It further introduces the Gene Pathway Coherence (GPC) evaluation framework based on Hallmark/GO pathways, achieving substantial improvements over STPath at the pathway level—particularly on low-variance, biologically fundamental pathways.

## Background & Motivation

**Background**: Spatial transcriptomics (ST) integrates high-resolution imaging with transcriptomic profiling to map gene expression distributions in situ within tissue sections. However, ST is costly, labor-intensive, and difficult to scale. Since H&E-stained whole-slide images (WSIs) are routinely acquired in clinical settings, computationally inferring gene expression from H&E images has become an active research direction.

**Limitations of Prior Work**: (1) Early methods (BLEEP, GraphST, TRIPLEX) rely on complex multi-stage pipelines employing contrastive learning (where negative sample definition is non-trivial) or multi-resolution feature engineering (which incurs high computational overhead). (2) Generative approaches (STEM, STFlow) model one-to-many mappings but are validated only on single cancer types and are computationally intensive. (3) STPath adopts BERT-style masked gene modeling to learn pan-cancer predictions across 38k genes, yet assumes that inter-gene correlations remain stable across tissue types—an assumption that may fail under the high heterogeneity of a pan-cancer setting—and its large model size demands substantial training and inference resources.

**Key Challenge**: Existing evaluation standards focus exclusively on Pearson correlation of top-N highly variable genes (HVGs), neglecting biological coherence at the functional pathway level. A model can achieve high HVG scores while failing to recover biologically meaningful coordinated expression patterns, thereby limiting its clinical translational value.

**Goal**: (1) Design an efficient direct-mapping architecture to replace complex reconstruction-based methods. (2) Establish pathway-level evaluation criteria to measure the biological meaningfulness of predictions.

**Key Insight**: The authors argue that gene expression prediction is fundamentally a modality translation task (image → expression) rather than a reconstruction task, making direct mapping more appropriate than an autoencoder framework. Evaluation should shift from isolated gene-level variance to functional pathway-level coherence.

**Core Idea**: Inject cancer-type conditioning via cross-attention, capture inter-patch context with a Transformer encoder, directly regress gene expression via an MLP head, and assess biological fidelity using the pathway-level GPC benchmark.

## Method

### Overall Architecture

The input consists of patch-level features $\mathbf{x}_i \in \mathbb{R}^{D_{img}}$ extracted from H&E WSIs by a pathology foundation model (UNI PFM) along with a cancer-type one-hot encoding $\mathbf{c}$. These are passed through a cross-attention conditioning module, a Transformer encoder that models inter-patch context, and an MLP regression head to predict the $D_{gene}$-dimensional gene expression for each patch.

### Key Designs

1. **Pan-Cancer Conditioning Cross-Attention Module**:

    - **Function**: Injects global cancer-type information into local patch representations.
    - **Mechanism**: The one-hot cancer-type vector is projected via a linear layer to $\mathbf{c}_{\text{emb}} \in \mathbb{R}^{D_{img}}$, serving as Key and Value; patch features serve as Query. Standard cross-attention produces conditioned patch features $\mathbf{X}_{\text{cond}}$.
    - **Design Motivation**: Enables the model to modulate patch representations according to cancer type, learning both pan-cancer shared patterns and cancer-type-specific patterns. Ablation studies confirm that removing cross-attention consistently degrades performance across all cancer types.

2. **Transformer Encoder for Contextual Aggregation**:

    - **Function**: Captures short- and long-range spatial dependencies among patches.
    - **Mechanism**: Conditioned patch features are first projected to a hidden dimension $D_{hidden}=256$, then processed by a 2-layer, 8-head Transformer encoder, yielding $\mathbf{H}_{\text{latent}} \in \mathbb{R}^{N \times D_{hidden}}$.
    - **Design Motivation**: Models high-level tissue structures such as tumor boundaries and immune infiltration patterns. Notably, ablation experiments reveal that **omitting positional encodings yields better performance**—likely because UNI PFM features already encode morphological information, allowing the Transformer to function as a permutation-invariant set operator that exploits global compositional structure rather than fixed spatial positions.

3. **Gene Pathway Coherence (GPC) Evaluation Framework**:

    - **Function**: Assesses the biological fidelity of predictions at the functional pathway level.
    - **Mechanism**: 87 non-redundant pathways are selected from MSigDB Hallmark (50 pathways) and GO databases (50–100 genes per pathway; Jaccard similarity < 0.1 for deduplication). For each pathway, Pearson correlation coefficients across patches are computed for all member genes and averaged: $s_m = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{|P_m|} \sum_{g \in P_m} r_{i,g}$
    - **Design Motivation**: HVG metrics focus exclusively on high-variance genes, overlooking low-variance pathways that are biologically critical. GPC evaluates the recovery of coordinated expression patterns, better reflecting clinical relevance.

### Loss & Training

The model is trained end-to-end with an MSE loss: $\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i \in N} (\hat{y}_i - y_i)^2$. Training is conducted on the HEST1k dataset, which aggregates spatial transcriptomics data from 153 cohorts across 36 independent studies. HistoPrism requires only approximately 500 WSIs for training—roughly half of what STPath requires.

## Key Experimental Results

### Main Results (Top-50 HVG PCC)

| Cancer Type | STPath (Micro-avg) | HistoPrism (Micro-avg) |
|---|---|---|
| CCRCC | 0.117 | **0.206** |
| COAD | **0.459** | 0.397 |
| HCC | 0.094 | **0.113** |
| IDC | **0.629** | 0.477 |
| PRAD | 0.255 | **0.317** |
| Overall Average (Micro-avg) | 0.292 | **0.318** |

### GPC Pathway-Level Evaluation

| Pathway Database | HistoPrism Win Rate |
|---|---|
| Hallmark Pathways (50) | **86.0%** |
| GO Pathways (87) | **74.7%** |

### Clustering Quality Comparison

| Model | AMI ↑ | ARI ↑ |
|---|---|---|
| STPath | 0.395 | 0.402 |
| **HistoPrism** | **0.623** | **0.521** |

### Key Findings
- HistoPrism surpasses STPath on micro-averaged PCC (0.318 vs. 0.292); micro-averaging better reflects overall prediction quality across cancer types.
- **Pathway-level prediction is the most significant highlight**: HistoPrism outperforms STPath on 86% of Hallmark pathways and 75% of GO pathways, with the largest advantage on low-variance pathways—which typically correspond to core biological processes.
- In clustering experiments, AMI improves from 0.395 to 0.623 (+57.7%), indicating that HistoPrism's whole-transcriptome predictions exhibit greater overall biological coherence.
- Positional encodings do not benefit performance, suggesting that the prediction task is primarily local and that PFM features already capture morphological information.

## Highlights & Insights
- **The introduction of the GPC evaluation framework** is the most important contribution of this paper—shifting evaluation from isolated high-variance genes to the coordinated expression of functional pathways, which more faithfully reflects clinical and biological requirements. This carries greater methodological significance than simply improving HVG PCC.
- The choice of a direct-mapping architecture over an autoencoder framework reflects deep insight: gene expression prediction is a unidirectional translation task, and no input-side gene information is available for reconstruction, making the inductive bias of an autoencoder a liability rather than an asset.
- The cross-attention design for pan-cancer conditioning is elegant and efficient, and its necessity is validated by the performance drop observed upon its removal in ablation studies.

## Limitations & Future Work
- STPath still leads in macro-averaged PCC on IDC (invasive ductal carcinoma) and COAD (colon adenocarcinoma), indicating room for improvement in cancer-type-specific learning for HistoPrism.
- The pathway selection criteria in the GPC framework (50–100 genes, Jaccard < 0.1) are manually defined; different thresholds may affect evaluation conclusions.
- Only UNI is used as the PFM feature extractor; the impact of alternative PFMs (e.g., GigaPath, CTransPath) has not been investigated.
- Generative methods (STEM, STFlow) perform poorly in the pan-cancer setting, though the authors acknowledge that computational constraints led them to train these baselines on only a subset of genes.

## Related Work & Insights
- **vs. STPath**: STPath is the current SOTA foundation model for pan-cancer gene prediction, employing BERT-style masked gene modeling to learn inter-gene dependencies. HistoPrism comprehensively surpasses STPath at the pathway level but still falls short on HVG metrics for certain cancer types. The fundamental difference lies in architectural philosophy: STPath is reconstruction-based (predicting masked genes), while HistoPrism is direct-mapping-based (regressing expression from images).
- **vs. BLEEP**: BLEEP uses contrastive learning to align H&E images and gene expression into a joint space, performing nearest-neighbor retrieval at inference. Retrieval-based inference limits generalization to unseen samples, and negative sample definition is inherently ambiguous in pathology.
- **vs. TRIPLEX**: TRIPLEX introduces a multi-resolution distillation architecture with high computational complexity and is validated only on single cancer types. HistoPrism substantially outperforms TRIPLEX in both efficiency and generalizability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The GPC evaluation framework represents an important methodological contribution; the architectural design is clean but lacks breakthrough innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 10 cancer types, multiple baselines, pathway-level evaluation, clustering analysis, efficiency comparison, and ablation studies; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation and evaluation framework are clearly articulated; the methods section is well-formalized.
- **Value**: ⭐⭐⭐⭐⭐ — The GPC evaluation paradigm has far-reaching implications for computational pathology; HistoPrism itself is a practical and efficient tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](../../AAAI2026/medical_imaging/hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)
- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](../../AAAI2026/medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/medical_imaging/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[AAAI 2026\] SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](../../AAAI2026/medical_imaging/spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)

</div>

<!-- RELATED:END -->
