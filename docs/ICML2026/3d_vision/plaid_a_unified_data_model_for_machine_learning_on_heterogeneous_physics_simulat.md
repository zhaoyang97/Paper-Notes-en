---
title: >-
  [Paper Note] PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations
description: >-
  [ICML 2026][3D Vision][Physics-informed Machine Learning] PLAID introduces a unified data model and open-source library for heterogeneous physics simulation data…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Physics-informed Machine Learning"
  - "Heterogeneous Meshes"
  - "Surrogate Modeling"
  - "Data Standards"
  - "Benchmarking"
date: 2026-05-08
content_hash: ceca4e043e51866e
---

# PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations

**Conference**: ICML 2026  
**arXiv**: [2505.02974](https://arxiv.org/abs/2505.02974)  
**Code**: https://github.com/PLAID-lib/plaid (Available)  
**Area**: Scientific Computing / Physics Simulation / Datasets & Benchmarks  
**Keywords**: Physics-informed Machine Learning, Heterogeneous Meshes, Surrogate Modeling, Data Standards, Benchmarking

## TL;DR
PLAID introduces a unified data model and open-source library for heterogeneous physics simulation data, along with six industrial-grade datasets and reproducible benchmarks covering structural mechanics and CFD. It standardizes "varying mesh, topology, and dimensionality" simulation data into format-ready benchmarks for the machine learning community.

## Background & Motivation

**Background**: Training surrogate models with ML to accelerate PDE simulations has emerged as an independent field. Primary approaches include Message Passing-based GNNs (like MeshGraphNets), operator learning methods (like Fourier Neural Operators), and recent mesh transformers or Implicit Neural Representations (INRs), supported by infrastructures such as PyTorch Geometric, DGL, and PhysicsNeMo.

**Limitations of Prior Work**: Compared to web-scale text in NLP or billion-scale image-text pairs in CV, physics ML datasets have long been "small, narrow, private, and inconsistently formatted." They are either toy problems supporting only regular structured grids (e.g., PDEBench, The Well) or ad-hoc datasets tied to specific solvers and libraries, offering almost no interoperability.

**Key Challenge**: The complexity of real industrial simulations stems precisely from "heterogeneity." Within a single dataset, samples may have different shapes, node counts, connectivity, element types, or even topologies (e.g., variable numbers of holes in porous materials), and may undergo remeshing over time. Existing ML formats assume "tensors with fixed shapes," leading the community to discard these heterogeneities and regress to structured grids for academic evaluation, which fails to reflect true industrial generalization capabilities.

**Goal**: This gap is addressed via three sub-problems: (1) designing a data abstraction that preserves industrial solver complexity (e.g., CGNS) while being efficiently consumable by ML frameworks; (2) providing a supporting library and online storage backend for scalable construction and access; and (3) offering a suite of truly heterogeneous industrial datasets and open benchmark protocols for fair comparison across modeling paradigms.

**Key Insight**: Instead of reinventing the wheel, the authors recognize CGNS as the de facto standard for industrial CFD/FEM. PLAID reuses the hierarchical simulation tree of CGNS as the underlying schema and adds a layer of "ML-friendly" metadata for samples, inputs, outputs, and splits. It leverages community-accepted infrastructures like Hugging Face and Zarr to lower engineering barriers.

**Core Idea**: A three-layer stack comprising "CGNS Tree + ML Metadata + Standardized Accessor + HF Distribution" encapsulates heterogeneous physics data as first-class citizens in ML. This is manifested through six industrial datasets and benchmarks for five mainstream methods to define the field's standard.

## Method

### Overall Architecture

PLAID consists of three complementary components:

- **Data Model Layer**: The abstract unit is a `Sample`. A sample can contain multiple CGNS bases (different dimensions or physical quantities on different meshes). Each base organizes structures like zones, fields, scalars, and time, natively supporting multi-support structures (e.g., combining 2D flow fields with 1D blade surface fields).
- **Library Layer**: Provides high-level accessors (e.g., `sample.get_field(name)` to auto-infer default zones/locations) and tools for parallel I/O and dataset construction. It relies on Muscat to bridge industrial solvers like Z-set, OpenRadioss, and elsA.
- **Benchmark & Distribution Layer**: Datasets are stored as human-readable YAML/CGNS or packed into Hugging Face Datasets/Zarr backends for streaming access. Benchmarks run as "no-deadline competitions" on Hugging Face, already accumulating 80+ community submissions.

Inputs are raw simulations from industrial solvers; outputs are ML-ready datasets (with train/test splits) consumable by GNNs/operators/transformers, alongside RRMSE-based leaderboards.

### Key Designs

1.  **Heterogeneous Sample Representation based on CGNS Trees**:
    -   **Function**: Enables a single `Sample` to carry data with varying meshes, topologies, mixed element types, and time-dependent multi-mesh/multi-field structures.
    -   **Mechanism**: Reuses the `base/zone/field` hierarchy of CGNS. Each field is explicitly positioned via a 5-tuple: `(name, zone_name, base_name, location ∈ {Vertex, CellCenter, FaceCenter}, time)`. Consequently, remeshing, dynamic fields, and multi-dimensional mesh coexistence are first-class scenarios.
    -   **Design Motivation**: PLAID is positioned as an "ML convention on top of CGNS" rather than a new schema. This ensures compatibility with industrial solvers and visualization tools like ParaView, avoiding the "proprietary format" ecosystem trap.

2.  **Decoupled Storage Backends and HF/Zarr Streaming**:
    -   **Function**: Decouples the data model from physical storage formats. The same PLAID dataset can be stored as YAML+CGNS files or serialized for Hugging Face or Zarr.
    -   **Mechanism**: The library layer exposes a unified accessor, while backends handle serialization and parallel I/O. The HF backend leverages the `datasets` library for remote caching, and the Zarr backend provides chunked compression, allowing users to download only specific features rather than full multi-gigabyte files.
    -   **Design Motivation**: Heterogeneous datasets are often GB-scale, making full in-memory loading impractical. Integrating "streamed + feature-wise access" into the data layer is a key engineering advantage over "monolithic npz/h5" benchmarks like The Well.

3.  **Six Incremental Heterogeneous Datasets & RRMSE Benchmark Protocol**:
    -   **Function**: Stress-tests geometric, mesh, and topological variations using real datasets from industrial solvers under a unified scoring protocol.
    -   **Mechanism**: Datasets (Tensile2d, 2D_MultiScHypEl, 2D_ElPlDynamics, Rotor37, 2D_profile, VKI-LS59) utilize solvers like Z-set and elsA. Sample complexity ranges from ~5k to ~37k nodes. The evaluation uses:
        $$ \mathrm{RRMSE}_f = \left(\frac{1}{n_\star}\sum_i \frac{1}{N^i}\|\mathbf{f}^i_{\rm ref}-\mathbf{f}^i_{\rm pred}\|_2^2 / \|\mathbf{f}^i_{\rm ref}\|_\infty^2\right)^{1/2} $$
        Final `total_error` is the mean RRMSE across all fields and scalars. Evaluation is automated via the HF platform.
    -   **Design Motivation**: The goal is to break the cycle of "overfitting to fixed leaderboards" by providing reproducible experiments while maintaining a long-term community platform.

### Loss & Training
PLAID does not dictate training losses but defines **evaluation metrics**. After training on a unified split, relative errors (RRMSE) are calculated for each field/scalar. For time-dependent data (e.g., 2D_ElPlDynamics), fields across the entire trajectory are stacked for RRMSE calculation, evaluating "trajectory-level" error.

## Key Experimental Results

### Main Results

The table below shows the `total_error` (lower is better) for six representative methods across the PLAID datasets. Bold indicates the best; underlined indicates the second best.

| Dataset | MGN | MMGP | Vi-Transf. | Augur | FNO | MARIO |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Tensile2d | 0.0673 | **0.0026** | 0.0116 | 0.0154 | 0.0123 | 0.0038 |
| 2D_MultiScHypEl | 0.0437 | — | 0.0325 | **0.0232** | 0.0302 | 0.0573 |
| 2D_ElPlDynamics | 0.1202 | — | 0.0227 | 0.0346 | **0.0215** | 0.0319 |
| Rotor37 | 0.0074 | **0.0014** | 0.0029 | 0.0033 | 0.0313 | 0.0017 |
| 2D_profile | 0.0593 | 0.0365 | 0.0312 | 0.0425 | 0.0972 | **0.0307** |
| VKI-LS59 | 0.0684 | 0.0312 | 0.0193 | 0.0267 | 0.0215 | **0.0124** |

### Key Findings
-   **MMGP dominates when samples are "alignable"** (e.g., Tensile2d, Rotor37), but fails (indicated by —) when topological changes occur (2D_MultiScHypEl), as mesh morphing assumptions break. This highlights "alignment" as a significant inductive bias.
-   **MARIO (INR-based with geometric conditioning) performs best in CFD tasks** with smooth or shock-dominated fields (2D_profile, VKI-LS59) but struggles with localized stress concentrations and topological changes in solid mechanics.
-   **FNO struggles significantly with Rotor37 and 2D_profile** because projecting anisotropic/3D meshes onto regular grids introduces both approximation error and computational overhead. This supports the argument that benchmarks **must preserve native mesh structures**.
-   **Vi-Transformer and Augur** provide robust "median performance," validating that mesh partitioning and tokenization are currently the most robust paradigms for handling heterogeneity.

## Highlights & Insights
-   **Adopting CGNS conventions is a mature engineering decision**: By adding ML metadata on top of existing standards (CGNS, ParaView), PLAID avoids the "missing ecosystem" trap and allows industrial users and ML researchers to collaborate seamlessly.
-   **Explicitly treating heterogeneity as an evaluation axis** is a powerful perspective. The 6x6 matrix reveals that being SOTA on PDEBench does not guarantee performance under topological changes. This framework is likely to become a standard requirement for physics ML papers.
-   **Leveraging Hugging Face infrastructure** is a brilliant engineering move for zero-maintenance, built-in leaderboards, and immediate community traffic.

## Limitations & Future Work
-   Current coverage is limited to structural mechanics and CFD; fields like electromagnetics, heat transfer, and combustion are not yet included.
-   Non-public ground truths for test sets prevent overfitting but hinder error analysis for researchers. Future work may include "diagnostic subsets."
-   The benchmark lacks newer paradigms like diffusion-based PDE solvers or Neural ODEs and does not yet compare methods under strict computational budget constraints.

## Related Work & Insights
-   **vs. The Well (Ohana 2024)**: The Well only supports structured grids; PLAID supports unstructured meshes, topological changes, and multi-dimensional coexistence.
-   **vs. PDEBench / PDEArena**: These focus on fluids; PLAID introduces industrial FEM solver data, filling the gap in solid mechanics.
-   **vs. CGNS**: PLAID is not a replacement but a layer on top, analogous to how "PyTorch Dataset" relates to "numpy array."
-   **vs. MeshGraphNets / MMGP**: These are baselines within the PLAID benchmark. PLAID provides the "ground" for fair evaluation rather than a new backbone model.

## Rating
-   **Novelty**: ⭐⭐⭐⭐ A comprehensive "infrastructure work" (Data + Benchmark + Library) with systemic impact equivalent to ImageNet or GLUE in their respective fields.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ A 6x6 matrix with a long-term HF platform is exceptional for a benchmark paper.
-   **Writing Quality**: ⭐⭐⭐⭐ Clear structure; key arguments are well-supported by data.
-   **Value**: ⭐⭐⭐⭐⭐ Raises the evaluation floor for the entire physics ML field; likely to become a de facto standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Galactification: Painting Galaxies onto Dark Matter Only Simulations Using a Transformer-Based Model](../../NeurIPS2025/3d_vision/galactification_painting_galaxies_onto_dark_matter_only_simulations_using_a_tran.md)
- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](../../ICLR2026/3d_vision/learning_unified_representation_of_3d_gaussian_splatting.md)
- [\[CVPR 2026\] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](../../CVPR2026/3d_vision/a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)
- [\[ICLR 2026\] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](../../ICLR2026/3d_vision/partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)
- [\[ICML 2026\] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)

</div>

<!-- RELATED:END -->
