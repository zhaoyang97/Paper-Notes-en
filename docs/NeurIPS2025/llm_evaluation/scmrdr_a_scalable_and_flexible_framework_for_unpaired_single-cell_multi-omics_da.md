---
title: >-
  [Paper Note] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration
description: >-
  [NeurIPS 2025][LLM Evaluation][Single-cell multi-omics] This paper proposes scMRDR, a framework based on β-VAE that disentangles latent representations of single-cell multi-omics data into modality-shared and modality-specific components, achieving scalable integration of unpaired multi-omics data through isometric regularization, adversarial training, and masked reconstruction loss.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - Single-cell multi-omics
  - unpaired data integration
  - β-VAE
  - disentangled representation
  - adversarial training
date: 2026-05-08
content_hash: 286ce8ddec575122
---

# scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration

**Conference**: NeurIPS 2025
**arXiv**: [2510.24987](https://arxiv.org/abs/2510.24987)
**Code**: [Available](https://github.com/sjl-sjtu/scMRDR)
**Area**: Bioinformatics / Multi-Omics Integration
**Keywords**: Single-cell multi-omics, unpaired data integration, β-VAE, disentangled representation, adversarial training

## TL;DR
This paper proposes scMRDR, a framework based on β-VAE that disentangles latent representations of single-cell multi-omics data into modality-shared and modality-specific components, achieving scalable integration of unpaired multi-omics data through isometric regularization, adversarial training, and masked reconstruction loss.

## Background & Motivation
Single-cell sequencing technologies can now measure multiple molecular modalities (scRNA, scATAC, protein) at single-cell resolution. However, due to technical constraints, large-scale datasets are typically unpaired across modalities. Existing methods suffer from two categories of limitations: (1) joint dimensionality reduction methods (e.g., MOFA, Seurat) rely on paired data or prior correspondences; (2) manifold alignment methods (e.g., UnionCom, SCOT) require computation of a global pairwise coupling matrix, limiting scalability and typically supporting only two modalities. The paper aims to design an integration framework that is both flexible (fully unpaired) and scalable (large datasets + multiple modalities).

## Method

### Overall Architecture
A single encoder-decoder β-VAE disentangles each cell's latent representation into a modality-shared component ($z_u$) and modality-specific components ($z_s^{(m)}$, with prior $\mathcal{N}(\mu_m, \sigma_m^2 I)$). Three regularization terms ensure the quality of the shared space.

### Key Designs

**β-VAE Disentanglement**: KL divergence weighting with $\beta > 1$ encourages conditional independence between $z_u$ and $z_s$. Observations from different omics layers are uniformly treated as single-sample inputs to the same encoder, naturally supporting unpaired and multi-modal data. The generative process assumes a ZINB (zero-inflated negative binomial) distribution, suited for sparse count data.

**Isometric Loss**: Preserves biological heterogeneity in the absence of cell-type labels. It enforces the shared space $z_u$ to maintain the distance structure of the full latent space $z=(z_u,z_s)$:
$$\mathcal{L}_{\text{preserve}} = \sum_m \sum_{i,j \in X^{(m)}} [\|\mu_{z_u}(x_i)-\mu_{z_u}(x_j)\|_2 - \|\mu_z(x_i)-\mu_z(x_j)\|_2]^2$$

**Adversarial Regularization**: An $m$-class discriminator $D(z_u)$ is introduced to distinguish the source modality of each cell; the encoder is optimized in the opposing direction to promote cross-modal alignment.

**Masked Reconstruction Loss**: Different omics modalities cover different feature sets (e.g., CITE-seq protein covers only ~134 features vs. ~13,000 RNA genes). A binary mask $\mathbf{b}$ prevents gradients from propagating through unmeasured features, and losses are scaled by the proportion of available features to maintain comparability.

### Loss & Training
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta\mathcal{L}_{\text{KL}} + \lambda\mathcal{L}_{\text{alignment}} + \gamma\mathcal{L}_{\text{preserve}}$$

Training alternates between updating the discriminator ($\mathcal{L}_{\text{discriminator}}$) and the VAE ($\mathcal{L}_{\text{total}}$). Default hyperparameters are $\beta=2, \gamma=5, \lambda=5$.

## Key Experimental Results

### Two-Omics Integration (Human Kidney, scRNA + scATAC)

| Method | Overall | Batch Correction | Bio Conservation | Modality Integration |
|--------|---------|-----------------|-----------------|---------------------|
| scVI (baseline) | 0.59 | 0.48 | 0.63 | 0.66 |
| Seurat v5 | - | - | - | - |
| JAMIE | - | - | - | - |
| **scMRDR($\beta$=2,$\gamma$=5,$\lambda$=5)** | **0.66** | **0.52** | **0.74** | **0.70** |

### Ablation Study

| Configuration | Overall | Note |
|---------------|---------|------|
| $\beta$=2,$\gamma$=5,$\lambda$=5 | **0.66** | Full model |
| $\beta$=2,$\gamma$=5,$\lambda$=0 | 0.61 | No adversarial → modality integration drops |
| $\beta$=2,$\gamma$=0,$\lambda$=5 | 0.61 | No isometric → bio conservation drops |
| $\beta$=1,$\gamma$=0,$\lambda$=0 | 0.59 | Baseline (no regularization) |

### Large-Scale Scalability (Mouse Primary Motor Cortex, 124k Cells)
- JAMIE, UnionCom, Pamona: failed due to memory/optimization errors
- GLUE: performance degrades significantly due to sensitivity of preprocessing to large-scale data
- **scMRDR**: stable performance, maintaining strong batch correction and bio conservation

### Three-Omics Integration (scRNA + scATAC + scProtein)
- Seurat v5 does not support three-omics integration
- GLUE fails to align distributions across three modalities (protein feature count is far smaller than the other two)
- **scMRDR naturally extends to three or more omics**, with consistently strong performance

### Key Findings
- Isometric regularization is critical for bio conservation; adversarial training is critical for modality alignment
- Masked loss prevents performance degradation when proteomics features are sparse
- Performance is relatively robust to hyperparameter choices (multiple combinations yield similar results)
- Spatial location imputation experiments validate the biological meaningfulness of the integration (inferred locations align with cortical layer annotations)

## Highlights & Insights
1. **Elegant design**: A single encoder-decoder architecture naturally supports an arbitrary number of unpaired omics layers
2. **Theory-driven**: Exploits the non-identifiability of the disentangled space; regularization constrains the shared subspace to be the one that maximally preserves structural information
3. **Practical masked loss**: Cleanly addresses the core challenge of heterogeneous feature coverage across omics modalities
4. **Biological validation**: Spatial location imputation identifies novel spatially variable genes

## Limitations & Future Work
- Adversarial loss (min-max optimization) introduces training difficulty and instability
- Mapping all modalities to the gene level (e.g., aggregating scATAC peaks into gene activity scores) may incur information loss
- The trade-offs among hyperparameters $\beta, \lambda, \gamma$ still require empirical tuning
- Spatiotemporal dynamic integration for spatial multi-omics and perturbation sequencing is not addressed

## Related Work & Insights
- The upgrade path from scVI to β-VAE with additional regularization is straightforward and principled
- The isometric loss is conceptually analogous to ISOMAP, preserving distances in the latent space
- The masked loss strategy is generalizable to other multimodal learning tasks with missing features

## Rating
- Novelty: ⭐⭐⭐⭐ (combinatorial innovation of β-VAE + three regularization terms)
- Technical Depth: ⭐⭐⭐⭐ (disentangled representation + scalable design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (ablation + large-scale + three-omics + biological validation)
- Practical Value: ⭐⭐⭐⭐⭐ (addresses real-world pain points in single-cell multi-omics integration)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[ICCV 2025\] Discontinuity-aware Normal Integration for Generic Central Camera Models](../../ICCV2025/llm_evaluation/discontinuity-aware_normal_integration_for_generic_central_camera_models.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[NeurIPS 2025\] AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)

<!-- RELATED:END -->
