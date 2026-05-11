---
title: >-
  [Paper Note] Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow
description: >-
  [NeurIPS 2025][Image Generation][Spatial Transcriptomics] NicheFlow is a Flow Matching-based generative model that represents cellular microenvironments as point clouds and jointly models the temporal evolution of cell s…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Spatial Transcriptomics"
  - "Microenvironment Trajectory Inference"
  - "Flow Matching"
  - "Optimal Transport"
  - "Point Cloud Generation"
  - "Cell Niche"
date: 2026-05-08
content_hash: cf20a8f5959f0609
---

# Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow

**Conference**: NeurIPS 2025
**arXiv**: [2511.00977](https://arxiv.org/abs/2511.00977)
**Authors**: Kristiyan Sakalyan, Alessandro Palma, Filippo Guerranti, Fabian J. Theis, Stephan Günnemann (TUM, Helmholtz Munich)
**Code**: [Project Page](https://www.cs.cit.tum.de/daml/nicheflow)
**Area**: Image Generation
**Keywords**: Spatial Transcriptomics, Microenvironment Trajectory Inference, Flow Matching, Optimal Transport, Point Cloud Generation, Cell Niche

## TL;DR

NicheFlow is a Flow Matching-based generative model that represents cellular microenvironments as point clouds and jointly models the temporal evolution of cell states and spatial coordinates via Variational Flow Matching and optimal transport, substantially outperforming single-cell-level trajectory inference methods on embryonic development, brain development, and aging datasets.

## Background & Motivation

### Problem Background
Understanding the evolution of cellular microenvironments in spatiotemporal data is critical for interpreting tissue development and disease progression. Spatial Transcriptomics (ST) technology enables single-cell-resolution gene expression mapping while preserving spatial information, yet ST provides only static snapshots of dynamic biological systems. Time-resolved spatial analysis captures gene expression patterns and cellular arrangement changes across developmental stages, offering essential temporal information about tissue development.

### Limitations of Prior Work
- Existing computational methods infer trajectories at the **single-cell level**, linking individual cells across time using velocity models (SiRV, SpVelo) or optimal transport (moscot, DeST-OT).
- These cell-centric approaches fundamentally **ignore the co-evolution of structured niches**—cells do not exist in isolation but develop coordinately as part of spatial microenvironments.
- Existing single-cell-level exact OT methods are limited in **scalability and generalization**.

### Root Cause
A key question is posed: how can one model the spatiotemporal evolution of cellular microenvironments while preserving local neighborhood relationships and cell-state transitions? NicheFlow directly models cell neighborhoods as holistic units rather than focusing on isolated cell trajectories.

## Method

### Microenvironment Definition
Given time-resolved spatial transcriptomics data, the tissue section at each time point $s$ is represented as an attributed point cloud $\mathcal{P}_s = \{(\boldsymbol{c}_i^s, \boldsymbol{x}_i^s)\}$, where $\boldsymbol{c}_i^s \in \mathbb{R}^2$ denotes spatial coordinates and $\boldsymbol{x}_i^s \in \mathbb{R}^D$ denotes gene expression features. Local microenvironments are defined with a fixed radius $r$:

$$\mathcal{M}_i^s = \{(\boldsymbol{c}_j^s, \boldsymbol{x}_j^s) \mid \|\boldsymbol{c}_j^s - \boldsymbol{c}_i^s\| \leq r\}$$

### OT Coupling Strategy
To train the conditional generative model, an optimal entropic coupling $\pi_{\epsilon,\lambda}^*$ is defined between source and target microenvironments. A pooled representation of each microenvironment is computed via a weighted average of coordinates and features:

$$\bar{\boldsymbol{m}}_i^s = \left[\frac{1-\lambda}{|\mathcal{M}_i^s|}\sum \boldsymbol{c}_j^s \;\Big\|\; \frac{\lambda}{|\mathcal{M}_i^s|}\sum \boldsymbol{x}_j^s\right]$$

The hyperparameter $\lambda \in [0,1]$ balances spatial proximity against feature similarity: larger $\lambda$ prioritizes feature matching, while smaller $\lambda$ prioritizes spatial position preservation.

### Mixture-Factorized VFM
The core innovation of NicheFlow lies in the factorization design of the variational posterior:
1. **Point-cloud-level factorization**: the posterior is fully factorized across individual points within the point cloud.
2. **Feature–coordinate factorization**: cell features and spatial coordinates are modeled separately.
3. **Mixed distribution family**: spatial coordinates use a **Laplace distribution** (concentrated around the mean, suitable for precise spatial modeling), while gene expression uses a **Gaussian distribution**.

The training loss is:

$$\mathcal{L}_{\text{NicheFlow}}(\theta) = \mathbb{E}\left[\sum_{(\boldsymbol{c}_1, \boldsymbol{x}_1) \in \mathcal{M}^1}\left(\|\boldsymbol{c}_1 - \bar{\boldsymbol{f}}_t^\theta\|_1 + \frac{1}{2}\|\boldsymbol{x}_1 - \bar{\boldsymbol{r}}_t^\theta\|_2^2\right)\right]$$

where $\bar{\boldsymbol{f}}_t^\theta$ and $\bar{\boldsymbol{r}}_t^\theta$ are the posterior mean predictions for coordinates and features, respectively.

### Backbone Architecture: Microenvironment Transformer
- **Encoder–decoder structure**: the encoder processes the source microenvironment $\mathcal{M}^0$ via self-attention; the decoder applies self-attention to noisy targets and conditions on the encoder output via cross-attention.
- **Input embedding**: features and spatial coordinates are embedded separately and concatenated; time $t$ is encoded with sinusoidal embeddings.
- **Permutation invariance**: naturally accommodates variable-size point cloud inputs.
- **Output projection**: linear projections produce posterior mean estimates for coordinates and features.

### Sampling and Generation
Given a source microenvironment $\mathcal{M}^0$, a Gaussian noise point cloud $\mathcal{M}^z$ is sampled, and the target microenvironment is generated by solving the ODE $\mathcal{M}^1 = \phi_1^\theta(\mathcal{M}^z \mid \mathcal{M}^0)$.

## Key Experimental Results

### Experiment 1: Quantitative Evaluation — Cross-Dataset Spatial Reconstruction

Three spatiotemporal datasets: (1) Mouse Embryonic Development (MED, Stereo-seq, 3 time points); (2) Axolotl Brain Development (ABD, Stereo-seq, 5 time points); (3) Mouse Brain Aging (MBA, MERFISH, 20 time points).

| Model | Objective | MED 1NN-F1↑ | MED PSD↓ | MED SPD↓ | ABD 1NN-F1↑ | ABD SPD↓ | MBA 1NN-F1↑ | MBA SPD↓ |
|-------|-----------|-------------|----------|----------|-------------|----------|-------------|----------|
| LUNA | — | 0.540 | — | — | 0.331 | — | 0.222 | — |
| SPFlow | CFM | 0.272 | 1.681 | 0.602 | 0.190 | 1.119 | 0.205 | 0.824 |
| RPCFlow | CFM | 0.546 | 0.981 | 0.564 | 0.524 | 1.015 | 0.271 | 0.810 |
| RPCFlow | GLVFM | 0.586 | 0.979 | 0.586 | 0.554 | 1.038 | 0.265 | 0.779 |
| **NicheFlow** | CFM | 0.609 | 0.979 | 0.402 | 0.604 | 0.568 | 0.283 | 0.556 |
| **NicheFlow** | **GLVFM** | **0.664** | **0.883** | **0.398** | **0.628** | **0.576** | **0.285** | **0.532** |

- NicheFlow+GLVFM improves 1NN-F1 over the strongest baseline RPCFlow+GLVFM by 13.3% (MED), 13.4% (ABD), and 7.5% (MBA).
- On SPD (coverage), NicheFlow reduces the metric by approximately 30–45% relative to RPCFlow, indicating substantially better coverage of target regions.
- SPFlow (single-cell-level) lags far behind on all metrics, validating the necessity of microenvironment-level modeling.

### Experiment 2: Qualitative Biological Validation — Spinal Cord and Neural Crest Cell Tracking

Two biological validation scenarios on the mouse embryonic dataset:

| Scenario | Source Time Point | Target Time Point | λ Setting | NicheFlow Result | moscot Result |
|----------|------------------|-------------------|-----------|-----------------|---------------|
| Spinal cord evolution | E10.5 | E11.5 | Low (spatial priority) | Correctly maps to mature spinal cord regions | Substantial mass misassigned to urogenital ridge and branchial arches |
| Cranial neural crest differentiation | E9.5 | E10.5 | High (expression priority) | Correctly captures differentiation toward mesenchymal and cranial structures | Significant mass leakage into unrelated inferior regions |

NicheFlow's trajectory predictions substantially outperform the exact-OT-based moscot method in both anatomical localization and descendant cell-type consistency.

## Highlights & Insights

- **Paradigm innovation**: the first proposal of microenvironment-level (rather than single-cell-level) spatiotemporal trajectory inference, modeling cell neighborhoods holistically as point clouds and implicitly capturing spatial correlations.
- **Mixture-factorized VFM**: the novel design of using Laplace distributions for coordinates and Gaussian distributions for features consistently improves accuracy over purely Gaussian VFM.
- **Scalability**: the mini-batch deep learning framework handles larger-scale data compared to exact OT methods such as moscot.
- **Dual-mode inference**: flexible support for two distinct biological scenarios—compositional changes within fixed structures and spatial migration of developing cells—via tuning the $\lambda$ parameter.
- **Cross-dataset generalization**: achieves top performance across three distinct biological processes: embryonic development, brain development, and brain aging.

## Limitations & Future Work

- **Locality constraint**: the fixed-radius microenvironment definition cannot capture tissue reorganization events spanning larger spatial scales.
- **Pairwise temporal modeling**: adjacent time points are modeled pairwise, lacking global temporal consistency constraints across multiple time points.
- **Heuristic distribution family selection**: the choice of Laplace/Gaussian is based on empirical intuition without an adaptive mechanism.
- **Evaluation limitations**: 1NN-F1 depends on the accuracy of pretrained classifiers, which may introduce bias.
- **2D spatial coordinates only**: the current framework handles only 2D spatial coordinates and has not been extended to 3D tissue reconstruction.
- **Gene expression dimensionality reduction**: the use of the top 50 PCA components may discard subtle trajectory-relevant gene expression differences.

## Related Work & Insights

- **moscot (Klein et al. 2025)**: single-cell-level Fused Gromov-Wasserstein OT; exact OT is difficult to scale and exhibits significant mass leakage in spinal cord/neural crest tracking experiments.
- **LUNA (Yu et al. 2025)**: diffusion-model-based spatial coordinate generation; does not model temporal dynamics and serves only as a spatial generation reference.
- **Wasserstein FM (Haviv et al. 2024)**: Wasserstein FM for point cloud generation, but does not address joint coordinate–feature generation or OT-based temporal trajectories.
- **DeST-OT (Halmos et al. 2025)**: semi-relaxed OT for aligning spatial sections, preserving transcriptomic and spatial proximity, but is not a generative model.
- **SpaTrack (Shen et al. 2025)**: single-cell trajectory inference using Fused Gromov-Wasserstein OT; does not handle microenvironment structure.
- **VFM (Eijkelboom et al. 2024)**: original VFM for graph generation; this work extends it to point cloud generation with a mixed distribution family.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to propose a microenvironment-level spatiotemporal trajectory inference paradigm; mixture-factorized VFM represents a methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative evaluation on three datasets plus qualitative biological validation in two scenarios, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations, clear motivation, polished figures, and convincing biological validation.
- Value: ⭐⭐⭐⭐ — Introduces a new computational paradigm for spatial transcriptomics, though the application scope is limited to a specific bioinformatics domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](../../CVPR2026/image_generation/enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](gspn-2_efficient_parallel_sequence_modeling.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)

</div>

<!-- RELATED:END -->
