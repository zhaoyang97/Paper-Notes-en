---
title: >-
  [Paper Note] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds
description: >-
  [NeurIPS 2025][Medical Imaging][state-space model] This paper proposes GeoDynamics, which generalizes the classical state-space model (SSM) from Euclidean space to the symmetric positive definite (SPD) manifold. By employing weighted Fréchet mean aggregation and orthogonal group translations, it achieves geometrically consistent state evolution on the manifold, attaining state-of-the-art performance on brain connectome analysis (early diagnosis of AD/PD/ASD) and human action recognition.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - state-space model
  - SPD manifold
  - Riemannian geometry
  - brain dynamics
  - functional connectivity
date: 2026-05-08
content_hash: 1e20da55f069c366
---

# GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds

**Conference**: NeurIPS 2025
**arXiv**: [2601.13570](https://arxiv.org/abs/2601.13570)
**Code**: N/A
**Area**: Brain Dynamics / State-Space Models / Riemannian Geometry
**Keywords**: state-space model, SPD manifold, Riemannian geometry, brain dynamics, functional connectivity

## TL;DR
This paper proposes GeoDynamics, which generalizes the classical state-space model (SSM) from Euclidean space to the symmetric positive definite (SPD) manifold. By employing weighted Fréchet mean aggregation and orthogonal group translations, it achieves geometrically consistent state evolution on the manifold, attaining state-of-the-art performance on brain connectome analysis (early diagnosis of AD/PD/ASD) and human action recognition.

## Background & Motivation

**Background**: Mainstream approaches for understanding brain dynamics fall into two categories: (a) analyzing temporal fluctuations of BOLD signals (RNN/LSTM/SSM/Mamba); and (b) analyzing topological changes of functional connectivity (FC) matrices (GNN/SPDNet). SSMs have attracted growing attention following the success of Mamba in CV and NLP.

**Limitations of Prior Work**: (a) SSMs assume states evolve in Euclidean space, whereas FC matrices are inherently SPD matrices residing on curved Riemannian manifolds—Euclidean operations violate the SPD constraint; (b) sliding-window methods are sensitive to window size; (c) SPDNet and related methods lack temporal modeling; (d) existing approaches capture either spatial or temporal structure, but not both.

**Key Challenge**: The temporal evolution of FC matrices constitutes a curve on the SPD manifold, requiring simultaneous capture of spatiotemporal dynamics while preserving geometric consistency.

**Core Idea**: Replace the Euclidean linear operations in SSMs with Fréchet means and orthogonal group actions from Riemannian geometry, enabling geometrically consistent state evolution on the SPD manifold.

## Method

### Overall Architecture
**Input**: A sequence of FC matrices constructed from BOLD signals via a sliding window. The pipeline passes through GeoDynamics state updates, observation, attention, and logarithmic mapping to produce classification outputs. All intermediate representations reside on the SPD manifold.

### Key Designs

1. **State Update Equation on the SPD Manifold**:

    - Replace linear combinations with the weighted Fréchet mean (wFM): intrinsic averaging along geodesics on the manifold, which automatically preserves the SPD property.
    - Replace additive updates with orthogonal group translations $\mathcal{T}(U,V) = g(V)Ug(V)^\top$, which are isometric under the Stein metric.
    - Employ the Stein distance to avoid repeated eigendecompositions.

2. **Discretization and Global Convolution**:

    - Matrix exponential discretization ensures stability.
    - Convolutional kernels are parameterized as $\hat{\mathcal{K}} = \mathcal{K}^\top\mathcal{K} + \epsilon I$ to guarantee the SPD property.

3. **SPD-Preserving Attention (SPA)**:

    - Attention weights are defined over manifold convolution responses; the exponential map guarantees SPD outputs.
    - Learned attention weights are interpretable as indicators of abnormal connectivity between brain regions.

4. **Task Readout**:

    - Logarithmic mapping to the tangent space + softmax classifier + cross-entropy loss.

## Key Experimental Results

### Main Results: Brain Connectome (Accuracy %)

| Dataset | Task | GeoDynamics | Runner-up | Gain |
|--------|------|-------------|---------|------|
| HCP-WM | 8-class working memory | **98.29** | Mamba 97.22 | +1.07 |
| OASIS | AD vs. CN | **71.43** | SPDNet ~68 | +~3 |
| ADNI | 4-class cognitive state | **56.00** | GSN 52.80 | +3.2 |
| PPMI | 4-class PD staging | **72.01** | — | — |

### Model Complexity Comparison (HCP-WM, N=360)

| Model | Params (M) | Accuracy |
|------|----------|----------|
| Mamba (2048d, 5 layers) | 132 | 97.22 |
| Mamba (1024d, 2 layers) | 14.07 | 95.92 |
| **GeoDynamics (2 layers)** | **14.60** | **98.29** |

### Action Recognition Validation

| Dataset | GeoDynamics | Runner-up |
|--------|-------------|---------|
| Florence | **94.1** | GR-GCN 93.3 |
| HDM05 | **72.3** | F-DMT-Net 71.3 |
| UTKinect | **98.0** | F-DMT-Net 98.0 |

### Key Findings
- On task-state fMRI, sequential models substantially outperform spatial models (by up to 30%); GeoDynamics surpasses Mamba with fewer parameters.
- On resting-state fMRI for neurodegenerative diseases, GeoDynamics significantly outperforms both spatial and sequential models—manifold-based modeling captures spatiotemporal information simultaneously.
- SPA attention maps align with clinical knowledge: AD abnormalities are located in the DMN and somatosensory cortex; PD in sensorimotor, frontal, and cerebellar regions; ASD in temporal and visual cortices.
- The model is relatively robust to sliding window size (35–45 optimal).
- State-of-the-art results on HAR datasets confirm cross-domain generalizability.

## Highlights & Insights
- **Mathematical Generalization of SSM to SPD Manifolds**: Replacing linear combinations with wFM and additive updates with orthogonal group translations provides rigorous geometric guarantees at each step, and is readily extensible to other manifold-valued time series.
- **Choice of Stein Distance**: Avoids repeated eigendecompositions, yielding high computational efficiency—a practical engineering contribution.
- **Interpretability of SPA Attention**: Attention weights correspond to degrees of abnormal connectivity in brain regions, aligning the black-box model with clinical neuroanatomical knowledge.

## Limitations & Future Work
- The method still relies on sliding windows to construct FC matrices; windowless, continuous-time FC construction remains to be explored.
- Weighted Fréchet mean optimization on the SPD manifold is iterative, incurring high computational costs for high-dimensional FC matrices ($N=360$).
- Absolute classification accuracy for brain disorders remains modest (ADNI 56%, PPMI 72%).
- HAR datasets used are small (199–686 samples); evaluation on large-scale benchmarks such as NTU-RGB+D is needed.
- Computed correlation matrices may be semi-definite or suffer from numerical issues in practice.
- Comparisons with Transformer-based brain network analysis methods are absent.
- Code is not publicly available, raising reproducibility concerns.

## Related Work & Insights
- **vs. Mamba**: A Euclidean-space SSM; at comparable parameter counts (~14M), GeoDynamics achieves 2.4% higher accuracy (98.3 vs. 95.9). The advantage stems from manifold-aware state evolution preserving the intrinsic geometric structure of FC matrices.
- **vs. SPDNet**: A purely spatial manifold model lacking temporal modeling, which performs poorly on task-state fMRI (sequential models exceed it by ~30%). GeoDynamics inherits manifold representations while incorporating joint spatiotemporal modeling.
- **vs. STAGIN/BNT/ContrastPool**: GNN-based brain network methods treat FC matrices as graphs rather than manifold elements; message passing in Euclidean space cannot enforce the SPD constraint.
- **vs. Traditional dFC Methods**: Sliding-window approaches are sensitive to window size; GeoDynamics's SSM module partially compensates for this bias.
- The framework naturally extends to other manifold-valued time series, such as subspace tracking on Grassmann manifolds or orthogonal frame evolution on Stiefel manifolds.
- The interpretability of SPA attention can serve as an auxiliary tool in clinical neuroimaging analysis to localize disease-relevant brain regions.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Generalizing SSMs to SPD manifolds is an elegant mathematical contribution with rigorous theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Coverage across 5 brain datasets and 3 HAR datasets is broad, though some absolute performance figures remain modest.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical formulations are rigorous and clear, though the density of notation may pose a barrier for non-specialist readers.
- **Value**: ⭐⭐⭐⭐ — Establishes a new paradigm for brain dynamics analysis and manifold-valued time series modeling.
<!-- NeurIPS 2025 | video_understanding -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research](barcodemamba_advancing_state-space_models_for_fungal_biodiversity_research.md)
- [\[NeurIPS 2025\] Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction](bridging_graph_and_state-space_modeling_for_intensive_care_unit_length_of_stay_p.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)

</div>

<!-- RELATED:END -->
