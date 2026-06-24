---
title: >-
  [Paper Note] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics
description: >-
  [Computational Biology] SToFM is proposed as the first multi-scale spatial transcriptomics foundation model. By integrating gene-scale domain adaptation, micro-scale subpatch partitioning, and macro-scale virtual cell injection, combined with an SE(2) Transformer and pre-trained on a large-scale corpus of 88M cells, SToFM significantly outperforms existing methods in tasks such as tissue domain semantic segmentation and cell-type annotation.
tags:
  - "Computational Biology"
date: 2026-05-08
content_hash: 9d6dea9f262a34cf
---

# SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics

- **Conference**: ICML 2025
- **arXiv**: [2507.11588](https://arxiv.org/abs/2507.11588)
- **Code**: [GitHub](https://github.com/PharMolix/SToFM)
- **Area**: Segmentation
- **Keywords**: Spatial Transcriptomics, Foundation Model, Multi-scale Learning, SE(2) Transformer, Tissue Domain Segmentation

## TL;DR

SToFM is proposed as the first multi-scale spatial transcriptomics foundation model. By integrating gene-scale domain adaptation, micro-scale subpatch partitioning, and macro-scale virtual cell injection, combined with an SE(2) Transformer and pre-trained on a large-scale corpus of 88M cells, SToFM significantly outperforms existing methods in tasks such as tissue domain semantic segmentation and cell-type annotation.

## Background & Motivation

Spatial transcriptomics (ST) technologies measure gene expression while preserving cellular spatial locations, providing tissue-level information unavailable in single-cell RNA sequencing. However, ST data contains **multi-scale biological information** that is not adequately captured by existing models:

**Macro-scale** (Figure 1a): Tissue morphology and organ structural information, such as functional regions and anatomical layers.

**Micro-scale** (Figure 1b): Cellular microenvironments and intercellular interactions.

**Gene-scale** (Figure 1c): Gene expression profiles of individual cells.

Limitations of existing ST foundation models:
- **Nicheformer**: Only utilizes gene expression, completely ignoring spatial coordinates.
- **CellPLM**: Integrates spatial information via sinusoidal positional embeddings, but represents an initial attempt and lacks fine-grained multi-scale designs.

The key challenge is: how to **simultaneously capture and integrate** information from these three scales across a tissue slice containing tens of thousands of cells using appropriate model architectures and self-supervised objectives.

## Method

### Overall Architecture

SToFM consists of two stages:
1. **Multi-scale Information Extraction**: Processes each ST slice at three scales to construct a set of subpatches containing multi-scale information.
2. **SE(2) Transformer Representation Learning**: Jointly models gene expression and spatial information on the subpatches.

### Key Designs

**Gene-scale: Domain Adaptation**

Incremental training is performed based on Geneformer (a pre-trained single-cell foundation model):
- ST data has lower quality (limited gene coverage, high dropout rates), and direct encoding performs poorly.
- The cell encoder $f_{cell}$ is continuously pre-trained on ST data to achieve scRNA-seq $\rightarrow$ ST domain adaptation.
- Masked Gene Modeling + contrastive learning objectives are utilized.

**Micro-scale: Subpatch Partitioning**

The entire ST slice is partitioned into multiple subpatches based on spatial locations, with each subpatch containing approximately 1000 cells:
- Balances computational efficiency with the preservation of local intercellular interactions.
- Emphasizes spatially localized cell-cell interactions.

**Macro-scale: Virtual Cells**

All cells in the slice are clustered using the Leiden clustering algorithm, and each cluster is aggregated into a virtual cell:
- The embeddings and positions of the virtual cells are the average of all cells within their respective clusters.
- Preserves the primary morphology and partitioning information of the slice, serving as a compressed representation of macro-scale information.
- Virtual cells are injected into each subpatch, enabling the model to perceive macro-scale structures while learning micro-scale information.

**SE(2) Transformer**:

An SE(2)-invariant Transformer architecture is used to jointly encode cell embeddings and spatial coordination:
- Input: Cell embeddings $F^{(i)}$ + distance matrix $D^{(i)}$ (pair representations initialized via a Gaussian module).
- The distance matrix acts as an attention bias, similar to Graphformer and AlphaFold.
- Output: Cell representations $Y_{cell}^{(i)}$ and pair representations $Y_{pair}^{(i)}$.
- Guarantees invariance to 2D translation and rotation.

### Pre-training Objectives

**Masked Cell Modeling (MCM)**: Randomly masks 10% of cell embeddings, using the output representations to predict the masked embeddings (MSE loss).

**Pairwise Distance Recovery (PDR)**: Randomly selects 10% of cells, adds Gaussian noise to their coordinates, and reconstructs the original distance matrix using pair representations (MSE loss).

$$\mathcal{L}_{MCM} = \frac{1}{|\mathcal{M}_1|}\sum_{j \in \mathcal{M}_1}(\|\hat{F}_j - F_j\|_2)^2$$

$$\mathcal{L}_{PDR} = \frac{1}{|\mathcal{M}_2|}\sum_{(j,k) \in \mathcal{M}_2}(\|\hat{D}_{jk} - D_{jk}\|_2)^2$$

### Loss & Training

Stage-wise training: For the first 2 epochs, the cell encoder is frozen and only the SE(2) Transformer is trained $\rightarrow$ for the 3rd epoch, both are jointly fine-tuned. 4×A100 GPUs for approximately 20 days.

## Key Experimental Results

### Main Results 1: Tissue Domain Semantic Segmentation (F1 Score)

| Model | ST Pre-training | Embryo Avg | Embryo Cross-slice | DLPFC Avg | DLPFC Cross-slice |
|------|----------|---------|-----------|-----------|-------------|
| scGPT | No Spatial | 0.7450 | 0.3947 | 0.6178 | 0.5885 |
| Geneformer | No Spatial | 0.7467 | 0.3745 | 0.5606 | 0.5440 |
| CellPLM | Expression+Spatial | 0.7722 | 0.3985 | 0.6219 | 0.5953 |
| **SToFM** | **Multi-scale** | **0.8046** | **0.4588** | **0.6535** | **0.6437** |

### Main Results 2: Cell-type Annotation (Mouse Brain)

| Model | Brain1 Acc | Brain1 F1 | Brain2 Acc | Brain2 F1 |
|------|-----------|-----------|-----------|-----------|
| CellPLM | 0.6001 | 0.4186 | 0.9256 | 0.7332 |
| **SToFM** | **0.6349** | **0.4951** | **0.9289** | **0.8362** |

### Ablation Study

| Ablation Variant | Gene | Micro | Macro | Embryo Cross-slice F1 | Brain1 F1 |
|---------|------|------|------|-------------|-----------|
| Cell encoder w/o DA | ✗ | ✗ | ✗ | 0.3745 | 0.3853 |
| Cell encoder w/ DA | ✔ | ✗ | ✗ | 0.4155 | 0.4725 |
| SToFM w/o VCs | ✔ | ✔ | ✗ | 0.4291 | 0.4893 |
| **SToFM** | **✔** | **✔** | **✔** | **0.4588** | **0.4951** |

### Key Findings

- In cross-slice settings, SToFM shows a greater advantage (Embryo cross-slice F1: 0.4588 vs 0.3985), demonstrating that multi-scale information significantly enhances generalization ability.
- The distinct contributions of the three scales are clear: domain adaptation (+4.1%/+8.7%), micro-scale information (+1.4%/+1.7%), and macro-scale information (+3.0%/+0.6%).
- In zero-shot clustering, the UMAP visualizations of SToFM show that different cell types form clear, compact clusters.
- SToCorpus-88M is currently the largest high-resolution ST pre-training corpus, exceeding Nicheformer by 1.6 times.

## Highlights & Insights

1. **Elegant Multi-scale Design**: The virtual cell mechanism compresses macro-scale information and injects it into micro-scale subpatches, cleverly encoding information across three scales within the Transformer's sequence input without incurring computational explosion from processing the whole slice.
2. **Rationality of SE(2) Invariance**: Computational interpretation of tissue slices should be robust to rotation and translation; the SE(2) Transformer naturally guarantees this invariance.
3. **Data Contribution**: SToCorpus-88M covers 6 ST technologies, 2000 slices, and 88M cells, representing a highly valuable contribution to the research community.

## Limitations & Future Work

1. **Limited to Three Scales**: Finer-grained scales (such as the gene regulatory network level) are not modeled, nor are image pyramid methods utilized.
2. **Lack of Multimodal Integration**: Fails to exploit pathological images or prior biological knowledge like known ligand-receptor pairs.
3. **Low-resolution ST Exclusion**: Only single-cell or near single-cell resolution data was selected; low-resolution technologies like 10x Visium were not included in pre-training.
4. **High Computational Cost**: Double forward passes of the cell encoder and multi-scale processing introduce significant computational overhead (4×A100 GPUs for 20 days of training).

## Related Work & Insights

- **Geneformer (Theodoris et al., 2023)**: Source of the cell encoder initialization for SToFM, featuring a gene sequencing encoding strategy ranked by relative expression level.
- **CellPLM (Wen et al., 2023)**: The first ST foundation model to integrate spatial information, utilizing only sinusoidal positional encoding; SToFM greatly expands upon this paradigm.
- **Nicheformer (Schaar et al., 2024)**: The current largest joint ST/scRNA-seq pre-trained model, but completely ignores spatial coordinates.
- **Uni-Mol/AlphaFold**: Successful application of SE(2)/SE(3) Transformers in molecules/proteins, from which SToFM adapts concepts for spatial transcriptomics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The multi-scale design (especially the virtual cell mechanism) and the application of SE(2) Transformer in ST are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 5 types of tasks (segmentation, annotation, clustering, deconvolution, imputation) + ablation + visualization.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear multi-scale motivation and highly systematic, complete methodological descriptions.
- **Value**: ⭐⭐⭐⭐⭐ — Dataset, model, and code are completely open-source, providing a major driving force for the spatial transcriptomics analysis field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)
- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/computational_biology/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICLR 2026\] HEIST: A Graph Foundation Model for Spatial Transcriptomics and Proteomics Data](../../ICLR2026/computational_biology/heist_a_graph_foundation_model_for_spatial_transcriptomics_and_proteomics_data.md)

</div>

<!-- RELATED:END -->
