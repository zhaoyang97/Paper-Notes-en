---
title: >-
  [Paper Note] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis
description: >-
  [AAAI 2026][Medical Imaging][Single-cell RNA-seq] This paper proposes PanFoMa, a lightweight hybrid neural network that integrates Transformer-based local modeling with Mamba-based global integration for pan-cancer singl…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Single-cell RNA-seq"
  - "Pan-Cancer"
  - "Transformer-Mamba Hybrid"
  - "foundation model"
  - "benchmark"
date: 2026-05-08
content_hash: 5eab74e6a1afefd2
---

# PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis

**Conference**: AAAI 2026
**arXiv**: [2512.03111](https://arxiv.org/abs/2512.03111)  
**Code**: [GitHub](https://github.com/Xiaoshui-Huang/PanFoMa)  
**Area**: Computational Biology / Single-cell Transcriptomics / Foundation Models
**Keywords**: Single-cell RNA-seq, Pan-Cancer, Transformer-Mamba Hybrid, foundation model, benchmark

## TL;DR

This paper proposes PanFoMa, a lightweight hybrid neural network that integrates Transformer-based local modeling with Mamba-based global integration for pan-cancer single-cell transcriptomic representation learning. It also introduces PanFoMaBench, a large-scale benchmark dataset covering 33 cancer subtypes and over 3.5 million cells.

## Background & Motivation

### Scientific Problem
Single-cell RNA sequencing (scRNA-seq) provides a powerful tool for dissecting tumor heterogeneity at single-cell resolution. Learning effective cell and gene representations from high-dimensional, sparse transcriptomic data is a central challenge for applications such as precision medicine, biomarker discovery, and drug target identification.

### Limitations of Prior Work

**Transformer-based methods (scGPT, GeneFormer, scFoundation, etc.)**:
- Self-attention mechanisms incur $O(N^2)$ computational complexity, making it prohibitively expensive to process full transcriptomes with tens of thousands of genes
- Methods are typically constrained to select top-K highly variable genes (e.g., 2,048), potentially discarding important low-expression functional genes such as transcription factors
- The HVG selection strategy introduces analytical bias that impairs generalization in pan-cancer settings

**Mamba-based methods (GeneMamba, etc.)**:
- Offer $O(N)$ linear complexity, overcoming the efficiency bottleneck
- However, Mamba is inherently a sequence model, whereas gene expression profiles are naturally unordered sets — interactions among genes do not follow any intrinsic order
- Existing methods apply heuristic fixed orderings (e.g., by mean expression level), ignoring context-dependent gene functionality
- Fixed-dimensional hidden states suffer from long-range forgetting, limiting performance when global patterns must be captured in pan-cancer analysis

### Root Cause
The paper proposes a **decoupled modeling strategy**: decomposing transcriptome modeling into two independent subtasks — parallel deep encoding of local gene interactions and efficient sequential integration of global information — each delegated to the architecture best suited for it.

## Method

### Overall Architecture
PanFoMa adopts a hierarchical local-to-global processing paradigm consisting of two core modules:

1. **Local-context Encoder**: Partitions input genes into chunks and processes them in parallel using a lightweight Transformer with shared parameters
2. **Global Sequential Feature Decoder**: Dynamically reorders genes based on global cell state and integrates them deeply using bidirectional Mamba

The overall computational complexity is $O(C \cdot M^2 + N \log N)$, where $N = C \cdot M$, achieving a balance between expressiveness and efficiency.

### Key Designs

#### 1. Local-context Encoder

**Input Chunk Representation (ICR)**:
- 3,072 genes are randomly sampled per training epoch
- Partitioned into $C = 4$ non-overlapping chunks, each containing $M = 768$ genes
- A learnable [CLS] token is prepended to each chunk
- The input embedding for each gene is formed by element-wise addition of a gene ID embedding and a binned expression value embedding:

$$e_{k,i} = \text{Emb}_{\text{id}}(g_{\text{id}_{k,i}}) + \text{Emb}_{\text{val}}(g_{\text{val}_{k,i}})$$

**Local Relation Modeling (LRM)**:
- All $C$ chunks are processed **in parallel** by a lightweight Transformer encoder
- The encoder consists of $L = 6$ stacked Transformer blocks with **shared parameters**
- Outputs: gene-level embeddings $H_{\text{genes},k}^{(L)} \in \mathbb{R}^{M \times D}$ and a CLS summary vector $h_{\text{[CLS]},k}^{(L)} \in \mathbb{R}^D$ per chunk

**Design Motivation**: The divide-and-conquer strategy decomposes the global $O(N^2)$ problem into $C$ local $O(M^2)$ subproblems, substantially reducing computational and memory costs while preserving the Transformer's capacity to capture complex gene interactions. Parameter sharing further reduces model size.

#### 2. Global Sequential Feature Decoder

**Global-aware Dynamic Sorting (GDS)**:
- A global cell state vector is synthesized by average pooling the CLS tokens of all chunks:

$$h_{\text{global\_cls}} = \frac{1}{C} \sum_{k=1}^{C} h_{\text{[CLS]},k}^{(L)}$$

- After concatenating gene embeddings from all chunks, the importance score of each gene is computed via dot product:

$$s_i = h_i \cdot h_{\text{global\_cls}}^T$$

- All genes are **sorted in descending order** of their scores, yielding the dynamically ordered feature matrix $H_{\text{sorted}}^{(L)}$

**Design Motivation**: This is the paper's most critical innovation — rather than applying a fixed heuristic ordering, the gene input sequence is dynamically determined based on each cell's global transcriptomic context. This mechanism reflects the biological reality that gene importance is not static but depends on dynamic functional roles within specific cellular contexts.

**Bidirectional Scanning with Gated Fusion (BSGF)**:
- The sorted gene sequence is processed by 6 layers of bidirectional Mamba
- Forward and backward Mamba modules process the sequence in the two respective directions
- A gating mechanism adaptively fuses bidirectional features for each gene:

$$h_{\text{fused},i} = \gamma_i \odot \overrightarrow{h}_{\text{mamba},i} + (1 - \gamma_i) \odot \overleftarrow{h}_{\text{mamba},i}$$

where $\gamma_i = \sigma(\text{Linear}(h_{\text{sorted},i}))$ is a learned gating vector.

### Benchmark Construction (PanFoMaBench)

Through systematic retrieval of the NCBI database, the paper integrates approximately **3.5 million high-quality cells** from 83 studies, covering **33 cancer subtypes**, **23 tissue types**, and **616 patients**. The data underwent a rigorous quality control pipeline:
1. Removal of cells with too few expressed genes
2. Exclusion of potential doublets with abnormally high gene/UMI counts
3. Filtering of low-activity cells with excessive mitochondrial gene proportions
4. Removal of lowly expressed genes to reduce noise

### Loss & Training
Pre-training employs a large-scale self-supervised learning strategy, with downstream tasks addressed via fine-tuning. Specific pre-training objectives are not elaborated in detail in the paper, which primarily focuses on architectural innovation and benchmark construction.

## Key Experimental Results

### Main Results: Pan-Cancer Diagnosis

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| scFoundation | 0.8876 | 0.8491 |
| scGPT | 0.9013 | 0.8732 |
| GeneMamba | 0.9026 | 0.8619 |
| GeneFormer | 0.9124 | 0.8851 |
| **PanFoMa** | **0.9474 (+3.5%)** | **0.9250 (+4.0%)** |

PanFoMa substantially outperforms all baselines on the self-constructed pan-cancer benchmark, achieving 94.74% accuracy.

### Batch Integration

| Dataset | Metric | GeneFormer | scGPT | GeneMamba | **PanFoMa** |
|---------|--------|-----------|-------|----------|------------|
| Immune | Avg_batch | 0.8153 | 0.9194 | 0.9536 | **0.9641** |
| Immune | Avg_bio | 0.6983 | 0.7879 | 0.8131 | **0.8332** |
| BMMC | Avg_batch | 0.7720 | 0.8431 | 0.9157 | **0.9312** |
| BMMC | Avg_bio | 0.6324 | 0.6576 | 0.7628 | **0.8021** |
| Covid-19 | Avg_batch | 0.8240 | 0.8625 | 0.8742 | **0.9173** |

PanFoMa achieves best performance on the majority of metrics across 5 batch integration datasets.

### Cell Type Annotation

| Dataset | Model | Accuracy | Macro-F1 |
|---------|-------|----------|----------|
| hPancreas | GeneMamba | 0.9713 | 0.7710 |
| hPancreas | **PanFoMa** | **0.9815** | **0.7760** |
| MS | scGPT | 0.8471 | 0.6630 |
| MS | **PanFoMa** | **0.8563 (+7.4% vs GeneMamba)** | **0.7016** |
| Myeloid_b | GeneMamba | 0.9603 | 0.9235 |
| Myeloid_b | **PanFoMa** | **0.9726** | 0.9351 |

### Multi-omics Integration

| Dataset | scGPT | scGLUE | **PanFoMa** |
|---------|-------|--------|------------|
| 10x Multiome PBMC | 0.758 | 0.747 | **0.789 (+3.1%)** |
| BMMC (RNA+Protein) | 0.697 | 0.600 | **0.721 (+2.4%)** |
| ASAP PBMC | **0.587** | 0.561 | 0.579 |

### Key Findings

1. **Necessity of local+global modeling**: Pure Transformer methods are limited to processing a subset of genes due to computational complexity, while pure Mamba methods fail to capture true gene regulatory relationships due to fixed ordering. PanFoMa's decoupled design effectively resolves this tension.
2. **Biological significance of dynamic sorting**: Global cell state-driven dynamic sorting assigns different positional encodings to the same gene across different cells, better reflecting the context-dependence of gene function.
3. **Gene regulatory network inference**: Visualization results demonstrate that PanFoMa's attention mechanism identifies MHC class II molecule-related gene regulatory relationships with higher confidence, recovering one additional relevant gene compared to scGPT.

## Highlights & Insights

- **Biological plausibility of the architectural design**: The information flow from "local regulatory signals → unified cell state" mirrors real gene regulatory hierarchies, endowing the architecture with biological interpretability.
- **Parameter-shared Transformer**: The 6-layer shared-parameter design substantially reduces model size while exploiting commonalities in gene interaction patterns across chunks.
- The **dynamic sorting mechanism** serves as the bridge connecting Transformer (unordered set modeling) and Mamba (ordered sequence modeling), representing the paper's most elegant design contribution.
- **Large-scale benchmark dataset**: PanFoMaBench, covering 33 cancer subtypes and over 3.5 million cells, is among the most comprehensive pan-cancer single-cell benchmarks available.

## Limitations & Future Work

1. **Misleading title**: The paper title references "Pathology Image Analysis," yet the work addresses single-cell transcriptomic data rather than histopathology images, representing a notable mismatch.
2. **Insufficient pre-training details**: The paper focuses primarily on architectural design and benchmark construction, with limited description of the pre-training objective.
3. **Gene sampling strategy**: Randomly sampling 3,072 genes per epoch may result in information loss for the remaining genes; coverage and training stability over extended training require further analysis.
4. **Underperformance on the Myeloid dataset**: PanFoMa fails to surpass GeneMamba on the Myeloid dataset (0.6515 vs. 0.6607), indicating that the hybrid architecture's advantage is not universal across all tasks.
5. **Insufficient computational cost analysis**: Despite claiming to be "lightweight," the paper does not provide direct FLOPs or inference speed comparisons against GeneMamba.

## Related Work & Insights

- **scGPT**'s GPT-style masked pre-training strategy achieved pioneering results in single-cell transcriptomic modeling, but is constrained by $O(N^2)$ complexity.
- **GeneMamba** overcame the efficiency bottleneck by replacing Transformer with Bi-Mamba, but its fixed ordering strategy remains a fundamental limitation.
- PanFoMa's paradigm of chunked parallel encoding + dynamic sorting + bidirectional Mamba may inspire other set-to-sequence modeling problems, such as point cloud processing and molecular graphs.
- The gated fusion mechanism is analogous to bidirectional LSTM fusion strategies in NLP, but its application in the Mamba context is novel.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|------------|-------|
| Novelty | 4.5 | Transformer-Mamba hybrid architecture + dynamic sorting mechanism |
| Technical Depth | 4 | Sophisticated architectural design with clearly motivated modules |
| Experimental Thoroughness | 4 | Pan-cancer diagnosis + batch integration + annotation + multi-omics, five baselines |
| Writing Quality | 3.5 | Clear structure but mismatch between title and content |
| Practical Value | 4 | Open-source code + large-scale benchmark contribution |
| **Overall** | **4** | Notable architectural innovation and valuable benchmark contribution, though certain details warrant further elaboration |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[ICML 2026\] PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](../../ICML2026/medical_imaging/thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](../../ICLR2026/medical_imaging/uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [\[ICLR 2026\] SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis](../../ICLR2026/medical_imaging/survhte-bench_a_benchmark_for_heterogeneous_treatment_effect_estimation_in_survi.md)

</div>

<!-- RELATED:END -->
