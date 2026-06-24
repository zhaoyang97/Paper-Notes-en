---
title: >-
  [Paper Note] scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data
description: >-
  [ICML2025 Spotlight][Computational Biology][self-supervised learning] This paper introduces scSSL-Bench, a systematic benchmark that evaluates 19 self-supervised learning (SSL) methods across 9 single-cell datasets on three downstream tasks: batch correction, cell type annotation, and missing modality prediction. The results reveal task-specific trade-offs between general-purpose SSL methods and domain-specific approaches.
tags:
  - "ICML2025 Spotlight"
  - "Computational Biology"
  - "self-supervised learning"
  - "single-cell genomics"
  - "benchmark"
  - "batch correction"
  - "contrastive learning"
date: 2026-05-08
content_hash: 6ea2d71abe93f801
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data

**Conference**: ICML2025 Spotlight  
**arXiv**: [2506.10031](https://arxiv.org/abs/2506.10031)  
**Code**: [BoevaLab/scSSL-Bench](https://github.com/BoevaLab/scSSL-Bench)  
**Area**: Computational Biology  
**Keywords**: self-supervised learning, single-cell genomics, benchmark, batch correction, contrastive learning

## TL;DR

This paper introduces scSSL-Bench, a systematic benchmark that evaluates 19 self-supervised learning (SSL) methods across 9 single-cell datasets on three downstream tasks: batch correction, cell type annotation, and missing modality prediction. The results reveal task-specific trade-offs between general-purpose SSL methods and domain-specific approaches.

## Background & Motivation

**Problem Context**: Single-cell RNA sequencing (scRNA-seq) and multi-omics sequencing technologies can characterize cellular heterogeneity at single-cell resolution. However, the generated data face two core challenges: (1) extremely high dimensionality (tens of thousands of genes $\times$ hundreds of thousands of cells) and (2) systematic technical biases (batch effects) introduced by different experimental batches, which can mask true biological signals.

**Current Status of SSL in Single-Cell Data**: Self-supervised learning has achieved great success in the CV and NLP domains (e.g., SimCLR, MoCo, BYOL). Multiple works have migrated these methods to single-cell data (e.g., CLEAR, CLAIRE, Concerto). However, existing studies lack:

- A systematic comparison between general-purpose SSL methods and single-cell-specific methods.
- Ablation studies of hyperparameters, data augmentation strategies, and regularization techniques in single-cell scenarios.
- A fair comparison between single-cell foundation models (e.g., scGPT, Geneformer) and contrastive learning methods.

**Three Core Research Questions**:

- **RQ1**: Do single-cell-specific SSL methods consistently outperform general-purpose SSL methods? What are the performance variations on unimodal vs. multimodal data?
- **RQ2**: How do hyperparameters (embedding dimension, projection dimension) and augmentation strategies affect the performance of general-purpose SSL on single-cell data?
- **RQ3**: Are domain-specific batch normalization (DSBN) and multimodal integration techniques designed for image data equally beneficial for single-cell data?

## Method

### Overall Benchmark Design

The pipeline of scSSL-Bench is as follows: input a cell-by-gene count matrix $\rightarrow$ select one of the 19 methods for training (contrastive methods use data augmentation to construct positive and negative pairs) $\rightarrow$ evaluate the learned representations on three downstream tasks.

### 19 Evaluated Methods (Four Categories)

1. **General-purpose SSL (7 methods)**: SimCLR, MoCo, SimSiam, NNCLR, BYOL, VICReg, BarlowTwins — all migrated from the CV domain, using a weight-sharing encoder to encode two augmented views followed by contrastive learning via a projector.
2. **Single-cell contrastive methods (4 methods)**: CLEAR (InfoNCE + Gaussian noise/mask/crossover augmentation), CLAIRE (MNN-based augmentation + MoCo architecture), Concerto (teacher-student distillation + dropout augmentation), scCLIP (CLIP-like multimodal contrastive learning).
3. **Single-cell generative methods (7 methods)**: scVI (VAE + zero-inflated negative binomial distribution), totalVI (multimodal VAE), scGPT / Geneformer / scBERT (foundation models), scButterfly (dual VAE), scTEL (Transformer + LSTM).
4. **Baselines (2 methods)**: SCDC, PCA.

### Datasets (9)

- **Unimodal (7)**: PBMC, Pancreas, Immune Cell Atlas, MCA, HIC, Lung, Tabula Sapiens (all are scRNA-seq).
- **Multimodal (2)**: PBMC-M, BMMC (CITE-seq, containing RNA + protein/ADT).

### Downstream Tasks and Evaluation Metrics

**Batch Correction**: Evaluated using the scIB toolkit, with the overall score calculated as:

$$Total = 0.6 \times Bio + 0.4 \times Batch$$

where $Bio$ measures the consistency between cell embeddings and true cell types, and $Batch$ measures the degree of batch effect removal.

**Cell Type Annotation**: The dataset is split into reference (training set) and query (test set, containing up to 3 unseen batches). A KNN classifier is trained on the reference embeddings to annotate the query. Evaluation metrics are macro-F1 and accuracy.

**Missing Modality Prediction**: On multimodal data, given the query RNA expression, the protein expression is predicted. This is done via kNN probing by taking the average of the nearest neighbors, with the Pearson correlation coefficient as the evaluation metric.

### Evaluation of Augmentations

Four augmentations from CLEAR (each applied with a 50% probability):

- **Masking**: Randomly setting 20% of genes to zero.
- **Gaussian Noise**: Adding Gaussian noise with mean 0 and standard deviation 0.2 to 80% of genes.
- **InnerSwap**: Swapping the expression values of 10% of genes within the same cell.
- **CrossOver**: Crossover-mutating 25% of gene expression values with another random cell.

Neighborhood augmentation from CLAIRE: Constructing positive pairs based on mutual nearest neighbors (MNN) or batch-balanced KNN (BBKNN) graphs.

### Other Ablation Designs

- **Domain-Specific Batch Normalization (DSBN)**: Assigning an independent batch normalization layer to each experimental batch.
- **Retention of Projector**: Whether to keep the projection head (projector) during the inference phase.
- **Embedding and Projection Dimensions**: Systematically evaluating the impact of different dimension settings on performance.

## Key Experimental Results

### Batch Correction (Table 1, 5 datasets)

| Method | PBMC-M Total | BMMC Total | PBMC Total | Pancreas Total | Immune Total |
|------|:---:|:---:|:---:|:---:|:---:|
| SimCLR | 0.700 | 0.767 | 0.447 | 0.721 | 0.635 |
| VICReg | 0.651 | **0.761** | 0.490 | **0.733** | 0.644 |
| BYOL | **0.754** | 0.722 | 0.379 | 0.610 | 0.479 |
| CLAIRE | — | — | **0.774** | 0.732 | 0.539 |
| scVI | — | — | See paper | See paper | See paper |
| scGPT (finetuned) | — | — | 0.770 | 0.662 | **0.781** |
| scGPT (zero-shot) | — | — | 0.451 | 0.351 | 0.435 |
| Geneformer (finetuned) | — | — | 0.199 | 0.177 | 0.114 |
| PCA (baseline) | See paper | See paper | See paper | See paper | See paper |

**Key Findings**:

- **Unimodal Batch Correction**: Domain-specific methods scVI, CLAIRE, and fine-tuned scGPT achieve the best performance.
- **Multimodal Batch Correction**: General-purpose SSL methods (SimCLR, VICReg, BYOL) surprisingly outperform domain-specific methods by a significant margin, highlighting the current lack of effective multimodal single-cell integration frameworks.
- Zero-shot scGPT underperforms its fine-tuned version significantly, and fine-tuned Geneformer shows very poor performance (Total < 0.2).
- Concerto achieves extremely low Bio scores on most unimodal datasets (e.g., only 0.055 on PBMC).

### Cell Type Annotation

- VICReg and SimCLR comprehensively outperform all single-cell-specific methods on the cell type annotation task.
- The advantages of general-purpose SSL methods on this task are stable and significant.

### Missing Modality Prediction

- In missing modality prediction for multimodal data, general-purpose SSL methods similarly outperform domain-specific ones.

### Ablation on Augmentation Strategies

- **Random masking (Masking) is the most effective augmentation technique**, outperforming domain-specific augmentations (CrossOver, InnerSwap) across all tasks.
- MNN and BBKNN neighborhood augmentations show some benefits for batch correction, but are not as universally applicable as simple masking.

### Ablation on Design Choices (RQ3)

- **DSBN is not beneficial**: Domain-specific batch normalization does not improve and may even impair performance on single-cell data.
- **Projector should be discarded**: Retaining the projection head during inference does not improve performance, which is consistent with empirical findings in the CV domain.
- **Embedding Dimension**: Medium to larger embedding dimensions (e.g., 128–256) consistently yield better results.

## Highlights & Insights

1. **Counter-intuitive "General Beats Specific" Finding**: On cell type annotation, multimodal integration, and missing modality prediction, general-purpose SSL methods adapted directly from CV (such as VICReg and SimCLR) surprisingly outperform methods specifically designed for single-cell data. This challenges current design paradigms of domain-specific methods.
2. **Universal Advantage of Masking**: Simple random masking augmentation outperforms all elaborately designed, biology-prior-based augmentations, suggesting that single-cell data augmentation strategies might not require extensive domain knowledge.
3. **Limitations of Foundation Models**: scGPT is competitive only after fine-tuning, and Geneformer performs poorly even with fine-tuning. This indicates that current single-cell foundation models remain immature for batch correction scenarios.
4. **Gaps in Multimodal Integration**: The paper explicitly highlights the lack of effective frameworks for single-cell multimodal integration, indicating a vital direction for future research.
5. **Contribution of a Standardized Evaluation Platform**: The systematic evaluation of 19 methods $\times$ 9 datasets $\times$ 3 tasks provides a much-needed foundation for fair comparison in the community.

## Limitations & Future Work

1. **Only Two Multimodal Datasets**: PBMC-M and BMMC are both CITE-seq datasets. Evaluations of other multimodal technologies like 10x Multiome (RNA+ATAC) are missing, which limits the findings' generalizability.
2. **Insufficient Exploration of Augmentation Strategies**: Only the augmentations proposed by CLEAR and CLAIRE are evaluated, leaving other potential methods (such as GAN-based augmentations or mixup) unexplored.
3. **Lack of Computational Efficiency Analysis**: The training time, memory footprint, and scalability of the 19 methods are not systematically reported, leaving a missing cost-dimension when choosing a practical method.
4. **Generative Methods Use Label Information**: Generative methods like scVI utilize batch/cell-type annotations during training, making comparisons with purely self-supervised methods not entirely fair (noted by the authors but not discussed in depth).
5. **Limited Downstream Tasks**: Other important single-cell analysis tasks, such as gene regulatory network reconstruction, trajectory inference, and perturbation prediction, are not evaluated.
6. **Hyperparameter Tuning Range**: General-purpose SSL methods employ uniform hyperparameter settings, which might not fully exploit the capacity of certain approaches.

## Related Work & Insights

- **Richter et al., 2024**: Prior work compared MAE, BYOL, and BarlowTwins on single-cell data, but lacked comparisons with domain-specific methods. This study fills this gap.
- **scIB (Luecken et al., 2022)**: A single-cell integration benchmarking framework. The evaluation metric system in this paper is built on top of it.
- **CLIP (Radford et al., 2021)**: scCLIP incorporates the multimodal contrastive concept of CLIP, but performs worse than general-purpose SSL methods in single-cell scenarios.
- **Insights**: The robust performance of general-purpose methods in a new domain suggests that domain-specific designs are not always beneficial. Simple yet effective augmentations, such as masking, are likely the optimal starting point.

## Rating

- Novelty: ⭐⭐⭐ — No new methodology is proposed. The core contributions are the systematic benchmark and empirical findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 19 methods $\times$ 9 datasets $\times$ 3 tasks, coupled with extensive ablation studies, making it highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with rational experimental designs and data-backed conclusions.
- Value: ⭐⭐⭐⭐ — Provides the single-cell SSL community with a standardized evaluation platform and actionable practical recommendations, alongside several key counter-intuitive findings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ICML 2026\] Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](../../ICML2026/computational_biology/learning_the_neighborhood_contrast-free_multimodal_self-supervised_molecular_gra.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](../../NeurIPS2025/computational_biology/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)
- [\[ICLR 2026\] CryoLVM: Self-supervised Learning from Cryo-EM Density Maps with Large Vision Models](../../ICLR2026/computational_biology/cryolvm_self-supervised_learning_from_cryo-em_density_maps_with_large_vision_mod.md)
- [\[ICML 2025\] DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models](deepseq_high-throughput_single-cell_rna_sequencing_data_labeling_via_web_search-.md)

</div>

<!-- RELATED:END -->
