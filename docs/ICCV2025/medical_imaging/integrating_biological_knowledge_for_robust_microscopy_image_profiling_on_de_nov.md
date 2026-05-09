---
title: >-
  [Paper Note] Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines
description: >-
  [ICCV 2025][Medical Imaging][Microscopy image representation learning] This paper proposes integrating external biological knowledge — protein–protein interaction graphs and transcriptomic features from single-cell foundation models — into microscopy image pretraining, explicitly decoupling perturbation-specific and cell-line-specific representations to improve generalization of perturbation screening on unseen (de novo) cell lines.
tags:
  - ICCV 2025
  - Medical Imaging
  - Microscopy image representation learning
  - perturbation screening
  - biological knowledge graph
  - single-cell foundation model
  - de novo cell lines
date: 2026-05-08
content_hash: 46949c0593884534
---

# Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines

**Conference**: ICCV 2025
**arXiv**: [2507.10737](https://arxiv.org/abs/2507.10737)
**Code**: [https://github.com/The-Real-JerryChen/BioMicroscopyProfiler](https://github.com/The-Real-JerryChen/BioMicroscopyProfiler)
**Area**: Medical Imaging
**Keywords**: Microscopy image representation learning, perturbation screening, biological knowledge graph, single-cell foundation model, de novo cell lines

## TL;DR

This paper proposes integrating external biological knowledge — protein–protein interaction graphs and transcriptomic features from single-cell foundation models — into microscopy image pretraining, explicitly decoupling perturbation-specific and cell-line-specific representations to improve generalization of perturbation screening on unseen (de novo) cell lines.

## Background & Motivation

Fluorescence microscopy image analysis is critical for drug discovery, yet transferring pretrained models to novel (de novo) cell lines poses significant challenges:

**Inter-cell-line heterogeneity**: Cell lines differ substantially in morphology (cell shape, size, nucleus-to-cytoplasm ratio) and biology (gene expression profiles, signaling pathway activity). Even under identical genetic perturbations, phenotypic responses vary considerably across cell lines.

**Spurious feature dependency**: Due to the limited number of cell lines in training sets, models may learn cell-line-associated but non-causal spurious features, leading to sharp performance degradation on new cell lines.

**Limitations of existing pretraining strategies**: Weakly supervised learning (WSL) and self-supervised methods (MAE, DINO) perform well on known perturbations but are not specifically designed to handle the de novo cell line scenario.

## Method

### Overall Architecture

Two biologically-informed components are added on top of existing pretraining methods (WSL, SimCLR, BYOL, MoCo v3, MAE): (1) a perturbation relationship graph constructed from protein–protein interaction networks, which guides learning of perturbation-specific representations via graph regularization losses; and (2) cell-line-specific representations encoded from RNA-seq data using a single-cell foundation model, injected into a ViT as additional tokens.

### Key Designs

1. **Perturbation relationship graph construction and regularization**:

    - A perturbation relationship graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W, \psi)$ is constructed, with nodes representing gene/chemical perturbations and edge weights derived from three biological databases: STRING (protein interaction scores with confidence threshold 200), Hetionet (binary connectivity converted to a probability matrix via random walk), and gene–gene similarity based on raw image features.
    - **Graph Laplacian regularization**: $\mathcal{L}_{lap} = \text{tr}(F^\top(D-W)F)$, encouraging perturbations with high edge weights to have similar feature representations.
    - **Graph node contrastive learning**: $\mathcal{L}_{con} = \frac{1}{|\mathcal{N}(i)|}\sum_{v_j \in \mathcal{N}(i)} \log \frac{\exp(\text{sim}(f_{v_i}, f_{v_j})/\tau)}{\sum_{k \in \mathcal{V}} \exp(\text{sim}(f_{v_i}, f_{v_k})/\tau)}$, pulling together biologically related perturbations while pushing apart unrelated ones.
    - **Theoretical guarantee** (Proposition 3.1): Minimizing the graph regularization loss implicitly minimizes intra-class distances within the same perturbation category.

2. **Cell-line-specific representation learning**:

    - RNA-seq data for each cell line are collected from the GSE portal, and $k$ highly variable genes are selected.
    - A single-cell foundation model (scGPT or scVI), fine-tuned on a cell-type annotation task, is used to extract compact cell-line embeddings: $h_c = \text{scFM}(E_c)$.
    - Since RNA-seq data are independent of microscopy images, these embeddings are accessible for both training and test cell lines, enabling transfer to unseen cell lines.

3. **Perturbation–cell-line information fusion**:

    - Following the vision–language model paradigm, cell-line features are projected into $m$ transcriptomic tokens $T^c = [t_1^c, ..., t_m^c]$.
    - These are prepended to image patch tokens $T^p$: $z = \text{ViT}([T^c, T^p])$.
    - The mean of image tokens $z_{m+1}, ..., z_{m+n}$ is used as the visual representation for downstream classification.

### Loss & Training

Total objective = standard pretraining loss (depending on the base method) + graph regularization loss ($\mathcal{L}_{lap}$ or $\mathcal{L}_{con}$).

Evaluation setup:
- Pretraining on three cell lines from RxRx1: HUVEC, RPE, and HepG2.
- One-shot fine-tuning evaluation on U2OS (RxRx1).
- Few-shot fine-tuning evaluation on HRCE and VERO (RxRx19a, 5-channel, COVID-19 dataset).
- ViT-S/16 as the visual encoder; trained on 8× A100 GPUs.

## Key Experimental Results

### Main Results (Table)

Perturbation screening performance on de novo cell lines (Top-1 / Top-5 accuracy %):

| Pretraining Method | Config | U2OS Top-1 | U2OS Top-5 | HRCE Top-1 | HRCE Top-5 | VERO Top-1 | VERO Top-5 |
|------------|------|------------|------------|------------|------------|------------|------------|
| No pretraining | baseline | 0.09 | 0.47 | 0.07 | 0.33 | 3.21 | 16.02 |
| WSL | baseline | 4.12 | 8.84 | 3.57 | 8.73 | 34.11 | 72.26 |
| WSL | **Ours** | **4.79** | **9.60** | **4.24** | **9.78** | **38.95** | **75.89** |
| SimCLR | baseline | 4.22 | 8.57 | 3.68 | 9.05 | 32.82 | 72.90 |
| SimCLR | **Ours** | **4.59** | **9.07** | **3.99** | **9.21** | **38.71** | **75.00** |
| MoCo v3 | baseline | 2.18 | 5.80 | 2.20 | 5.97 | 20.73 | 53.31 |
| MoCo v3 | **Ours** | **2.56** | **6.36** | **2.53** | **6.86** | **25.56** | **62.26** |
| MAE | baseline | 1.83 | 5.32 | 1.24 | 3.95 | 23.06 | 58.23 |
| MAE | **Ours** | **2.10** | **5.83** | **1.79** | **5.17** | **24.60** | **63.31** |

### Ablation Study (Table)

Ablation of cell-line-specific (CS) and perturbation-specific (PS) representations on U2OS:

| CS | PS | WSL Top-1 | WSL Top-5 | SimCLR Top-1 | BYOL Top-1 | MoCo Top-1 | MAE Top-1 |
|----|-----|-----------|-----------|--------------|------------|------------|-----------|
| ✗ | ✗ | 4.12 | 8.84 | 4.22 | 3.61 | 2.18 | 1.83 |
| ✓ | ✗ | 4.64 | 9.48 | 4.39 | 3.71 | 2.45 | 1.95 |
| ✗ | ✓ | 4.25 | 9.09 | 4.48 | 3.97 | 2.33 | 1.98 |
| **✓** | **✓** | **4.79** | **9.60** | **4.59** | **3.80** | **2.56** | **2.10** |

Ablation on scFM embedding dimensionality (WSL pretraining, evaluated on U2OS):
- scVI: optimal at dim=128; performance degrades at dim=512.
- scGPT: optimal at dim=256, but shows no significant advantage over scVI.
- Raw gene expression data slightly outperforms the baseline but is inferior to scFM embeddings.

### Key Findings

- Integrating biological knowledge yields consistent improvements across all five pretraining methods, demonstrating the generality of the proposed approach.
- Average relative Top-1 improvement on U2OS is approximately 20%; gains are more pronounced on VERO (WSL: 34.11→38.95).
- Perturbation-specific representations (graph regularization) yield more stable improvements than cell-line-specific representations across SimCLR, BYOL, and MAE.
- Adding cell-line-specific representations to BYOL slightly reduces performance (3.61→3.71), whereas perturbation relationship graphs consistently provide positive effects.
- When training and test data come from different batches (split condition 2), graph regularization shows stronger benefits, validating Proposition 3.1.
- A graph contrastive loss weight of 1e-5 outperforms 1e-2, indicating that excessive regularization can interfere with the primary task.
- WSL achieves the best performance among all pretraining methods, as it directly optimizes the perturbation classification objective.

## Highlights & Insights

- **Framing de novo cell line generalization as an explicit research problem** is a valuable contribution. Existing methods are typically evaluated within the same cell line, overlooking cross-cell-line transfer capability.
- **The idea of decoupling perturbation-specific and cell-line-specific information is novel**: external knowledge guides one component while scFM guides the other, enabling explicit disentanglement of the two.
- **Theoretical guarantee** (Proposition 3.1) provides a mathematical foundation — graph regularization implicitly reduces intra-class distances.
- Using RNA-seq data to represent unseen cell lines elegantly sidesteps the problem of lacking images at test time.

## Limitations & Future Work

- Absolute performance remains low: the best result on U2OS is only 4.79% Top-1 accuracy (1,138-class classification), indicating that de novo cell line generalization remains an extremely challenging problem.
- The pretraining set contains only three cell lines, limiting the diversity of cell-line-specific representations; larger-scale data could yield further improvements.
- Imaging protocols differ between RxRx19a and RxRx1 (5-channel vs. 6-channel), increasing the difficulty of cross-dataset transfer — a physical constraint rather than a methodological limitation.
- The graph regularization weight is a hyperparameter requiring tuning and exhibits variable behavior under different split conditions.
- Larger ViT architectures (e.g., ViT-B/L) are not explored, partly due to dataset scale constraints.

## Related Work & Insights

- Kraus et al. and the RxRx dataset series constitute important foundations for this field.
- The idea of incorporating protein–protein interaction networks as prior knowledge for visual models generalizes to other problems at the intersection of bioinformatics and computer vision.
- The use of single-cell foundation models (scGPT/scVI) as cross-modal bridges is a key innovation of this work and can inspire further multimodal fusion approaches.
- Future work could incorporate richer phenotypic annotations such as Cell Painting and chemical structure information to further enhance model capacity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic integration of biological knowledge graphs and single-cell foundation models to enhance microscopy image pretraining.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five pretraining baselines evaluated on three de novo cell lines, with detailed ablations covering CS/PS decoupling, graph loss analysis, and scFM dimensionality.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, method motivation is well-grounded, and theoretical analysis is rigorous.
- **Value**: ⭐⭐⭐⭐ Provides a promising direction for cross-cell-line generalization in drug discovery, though absolute performance still leaves considerable room for improvement.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/medical_imaging/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](../../NeurIPS2025/medical_imaging/one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/medical_imaging/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICCV 2025\] COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation](coin_confidence_score-guided_distillation_for_annotation-free_cell_segmentation.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)

<!-- RELATED:END -->
