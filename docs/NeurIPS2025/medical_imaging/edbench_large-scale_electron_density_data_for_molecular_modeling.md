---
title: >-
  [Paper Note] EDBench: Large-Scale Electron Density Data for Molecular Modeling
description: >-
  [NeurIPS 2025][Medical Imaging][electron density] This work constructs EDBench, the largest electron density (ED) dataset to date (3.3 million molecules, computed via B3LYP/6-31G** DFT), and designs a three-category benchmark evaluation framework covering prediction, retrieval, and generation tasks. It provides the first systematic assessment of deep learning models' ability to understand and exploit electron density.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - electron density
  - molecular force fields
  - density functional theory
  - benchmark dataset
  - geometric deep learning
date: 2026-05-08
content_hash: 26f579d8dc7f6e42
---

# EDBench: Large-Scale Electron Density Data for Molecular Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2505.09262](https://arxiv.org/abs/2505.09262)
**Code**: Available (see paper homepage)
**Area**: Medical Imaging
**Keywords**: electron density, molecular force fields, density functional theory, benchmark dataset, geometric deep learning

## TL;DR
This work constructs EDBench, the largest electron density (ED) dataset to date (3.3 million molecules, computed via B3LYP/6-31G** DFT), and designs a three-category benchmark evaluation framework covering prediction, retrieval, and generation tasks. It provides the first systematic assessment of deep learning models' ability to understand and exploit electron density.

## Background & Motivation

**State of the Field**: Machine learning force fields (MLFFs) have become essential tools for molecular dynamics simulation. However, mainstream approaches focus on atomic-level many-body interaction modeling (atom types, coordinates, distances, angles, torsions, etc.), with limited attention to microscopic electronic distributions.

**Limitations of Prior Work**: According to the Hohenberg–Kohn theorem, the electron density $\rho(\mathbf{r})$ uniquely determines all ground-state properties (energy, molecular structure, etc.) of a many-particle system, providing a finer-grained and more physically grounded molecular description than atomic-level representations. However, computing ED requires expensive DFT calculations, resulting in a lack of large-scale ED datasets.

**Root Cause**: Existing QC datasets (QM7, QM9, MD17, etc.) primarily provide energy and force data. Datasets that include ED are extremely scarce (MP: ~122K at PBE accuracy; ECD: ~140K), and most are focused on materials science. For drug-like molecules, large-scale ED data and associated benchmarks are absent.

**Paper Goals**: (1) Construct a large-scale, high-quality molecular ED dataset; (2) Design an ED-centric benchmark task suite to systematically evaluate models' ability to understand and utilize electronic information.

**Starting Point**: Using 3.3 million drug-like molecules from the PCQM4Mv2 dataset, high-quality ED data in CUBE file format are generated with the B3LYP hybrid functional (higher rung of Jacob's ladder) and the Psi4 computational engine, at a total cost of 205,000 core-hours (~23.4 single-core years).

**Core Idea**: Construct the first million-scale molecular electron density dataset and design prediction/retrieval/generation benchmark tasks to advance MLFFs from atomic-level to electron-level modeling.

## Method

### Overall Architecture
EDBench consists of two components: (1) a dataset—ED distributions for 3.3 million molecules along with quantum chemical properties (energy components, orbital energies, multipole moments, etc.); and (2) benchmark tasks—6 tasks in three categories: prediction (4), retrieval (1), and generation (1). Each task samples ~50K molecules from the full set under specified conditions, with an 80/10/10 scaffold split.

### Key Designs

1. **Dataset Construction**:

    - **Function**: Generate high-accuracy ED for 3,359,472 molecules from PCQM4Mv2.
    - **Mechanism**: Psi4 1.7 computational engine with the B3LYP hybrid functional; RHF reference for closed-shell systems and UHF for open-shell systems. The basis set is selected based on elemental composition: 6-31G** for molecules without sulfur, 6-31+G** for those containing sulfur (diffuse functions are better suited for polarizable heavy atoms). After SCF convergence, CUBE files are generated with a grid spacing of 0.4 Bohr, padding of 4.0 Bohr, and a density fraction threshold of 0.85.
    - **Computational Scale**: 8 × Intel Xeon Platinum 8270 (26 cores × 2 threads = 416 logical cores), totaling 205,000 core-hours.
    - **ED Definition**: $\rho(\mathbf{r}) = \rho_\alpha(\mathbf{r}) + \rho_\beta(\mathbf{r})$, obtained via SCF iteration of the Kohn–Sham equations $[-\frac{1}{2}\nabla^2 + V_{\text{eff}}(r)]\psi_i(r) = \epsilon_i \psi_i(r)$.

2. **Prediction Tasks (ED5-EC/OE/MM/OCS)**:

    - **Function**: Predict various quantum chemical properties from ED data.
    - **Mechanism**: An ED encoder $\text{Enc}_\mathcal{P}$ extracts ED features, followed by task-specific prediction heads $\text{Enc}_t$: $\hat{y}^\bullet = \text{Enc}_t^\bullet(\text{Enc}_\mathcal{P}(\mathcal{P}))$.
    - Four sub-tasks: 6 energy components (EC), 7 orbital energies (OE), 4 multipole moments (MM), and open/closed-shell classification (OCS).
    - **Sampling Strategy**: Structural clustering $C^s$ (ECFP4+USR fingerprints, k=100) crossed with label clustering, with uniform sampling to ensure diversity.

3. **Retrieval Task (ED5-MER)**:

    - **Function**: Bidirectional cross-modal retrieval between molecular structures and electron densities.
    - **Mechanism**: A molecular encoder $\text{Enc}_\mathcal{G}$ and an ED encoder $\text{Enc}_\mathcal{P}$ extract latent representations $h_\mathcal{G}, h_\mathcal{P}$, respectively. Alignment is trained via InfoNCE loss:
    $$\mathcal{L}_{\text{ret}} = -\log \frac{\exp(\text{sim}(h_{\mathcal{G}_i}, h_{\mathcal{P}_i})/\tau)}{\sum_j \exp(\text{sim}(h_{\mathcal{G}_i}, h_{\mathcal{P}_j})/\tau)}$$
    - Each anchor is paired with 10 negatives (half in-cluster easy negatives, half cross-cluster hard negatives).

4. **Generation Task (ED5-EDP)**:

    - **Function**: Predict the electron density distribution from molecular structure.
    - **Mechanism**: A heterogeneous graph $\mathcal{HG}$ is constructed containing atom nodes and electron nodes, with three types of edges—atom–atom, atom–electron, and electron–electron—built via k-NN (k=9). EGNN is extended to a heterogeneous graph variant HGEGNN; masked ED values are then predicted:
    $$h^{\mathcal{HG}} = \text{HGEGNN}(\hat{\mathcal{HG}}), \quad \hat{\mathcal{D}} = \text{Enc}_t^{\text{EDP}}(h_\mathcal{P}^{\mathcal{HG}})$$
    - Training loss: $\mathcal{L}_{\text{gen}} = \|\hat{\mathcal{D}} - \mathcal{D}\|_2$

### Loss & Training
- **Prediction**: L2 loss for regression tasks; cross-entropy for classification.
- **Retrieval**: InfoNCE with temperature $\tau = 0.07$.
- **Generation**: L2 loss.
- **ED threshold $\rho_\tau$**: Filters low-density regions (typically set to 0.05–0.2) to balance accuracy and computational efficiency.
- Scaffold split ensures out-of-distribution (OOD) evaluation.

## Key Experimental Results

### Prediction Task (ED5-OE, Orbital Energy MAE×100)

| Model | HOMO-2 | HOMO-1 | HOMO-0 | LUMO+0 | LUMO+1 | LUMO+2 |
|-------|--------|--------|--------|--------|--------|--------|
| PointVector | 1.73 | 1.68 | 1.92 | 3.08 | 2.86 | 3.05 |
| X-3D | 1.75 | 1.72 | 1.98 | 3.21 | 3.02 | 3.25 |

### Generation Task (ED5-EDP, HGEGNN)

| Threshold $\rho_\tau$ | MAE | Pearson (%) | Spearman (%) | Time (s/mol) | DFT Time |
|----------------------|-----|------------|-------------|-------------|---------|
| 0.10 | 0.3362 | 81.0 | 56.4 | 0.024 | 245.8 |
| 0.15 | 0.0463 | 98.0 | 87.0 | 0.015 | 245.8 |
| 0.20 | 0.0448 | **99.2** | 91.0 | **0.013** | 245.8 |

### Key Findings
- X-3D outperforms PointVector on three prediction tasks: energy components, multipole moments, and open/closed-shell classification.
- Orbital energy prediction (OE) is considerably easier than energy component prediction (EC)—orbital energies exhibit stronger locality and are directly associated with local ED patterns.
- **HGEGNN generates ED approximately 10,000× faster than DFT** (0.013 vs. 245.8 s/mol), with a Pearson correlation of 99.2%.
- Surprising finding: ED generated by HGEGNN **outperforms** DFT-computed ED on downstream energy prediction tasks, possibly because the model-generated ED is smoother and better aligned with the inductive biases of downstream models.
- In the retrieval task, combinations using EquiformerV2 as the molecular encoder (E+P, E+X) significantly outperform those using GeoFormer.

## Highlights & Insights
- **Contribution at Scale**: The 3.3-million-molecule ED dataset is the largest of its kind, representing 23.4 single-core years of computation, providing a critical infrastructure resource for the community.
- **Paradigm Shift from Atoms to Electrons**: This work is the first to systematically demonstrate the feasibility and value of using ED as a modeling target for MLFFs, opening a new direction toward "electron-level force fields."
- **Elegant Heterogeneous Graph Design**: Treating atoms and electrons as two distinct node types and connecting them via k-NN edges naturally couples molecular structure and electronic distribution within a unified framework.
- **Generated ED Surpassing DFT**: This counterintuitive finding suggests that learned ED, though potentially less physically accurate than DFT, exhibits smoother patterns that are more amenable to downstream model consumption—raising an intriguing question about whether strict physical accuracy is always necessary.
- **Comprehensive Benchmark Design**: The three task categories (prediction/retrieval/generation) cover different dimensions of ED understanding, and scaffold split ensures evaluation of OOD generalization.

## Limitations & Future Work
- Only the B3LYP functional is used; higher accuracy is achievable with higher-rung functionals such as ωB97X-D.
- ED representation is explored only in the point cloud form; voxel or volumetric image representations remain untested.
- Periodic systems (materials science settings) are not considered; the current scope is limited to drug-like molecules.
- Negative samples in the retrieval task may be too easy; advanced contrastive learning strategies (e.g., MoCo, hard negative mining) could further improve performance.
- The extremely high cost of dataset construction (205,000 core-hours) limits extension to higher-accuracy functionals.

## Related Work & Insights
- **vs. QM9/QM7**: Classical QC datasets contain only ~134K/7K molecules and do not include ED. EDBench significantly exceeds them in both scale (3.3M) and information richness (ED + multiple quantum properties).
- **vs. QMugs/∇²DFT**: These datasets provide density matrices rather than direct ED. EDBench provides spatially resolved ED distributions in CUBE format, directly applicable to geometric deep learning.
- **vs. DeepDFT (Jorgensen)**: The ED dataset generated via VASP is of much smaller scale; EDBench covers millions of molecules.
- The dataset has direct implications for virtual screening in drug discovery, molecular inverse design, and quantum-aware molecular modeling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The first million-scale molecular ED dataset with a systematic benchmark; the push from "atomic to electronic" modeling is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Six benchmark tasks, evaluation of multiple models, ablation analyses (threshold/sampling points/temperature), and comprehensive quality validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Background knowledge (DFT/Kohn–Sham) is well-motivated and accessible; dataset comparison tables are clear.
- **Value**: ⭐⭐⭐⭐⭐ — As a foundational infrastructure work, it establishes the data and evaluation basis for ED-driven molecular modeling with substantial long-term impact.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](../../ICCV2025/medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)

<!-- RELATED:END -->
