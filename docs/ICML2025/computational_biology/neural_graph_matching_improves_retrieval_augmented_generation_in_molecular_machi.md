---
title: >-
  [Paper Note] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning
description: >-
  [ICML2025][Computational Biology][Neural Graph Matching] This work proposes MARASON, introducing **Neural Graph Matching** into the Retrieval-Augmented Generation (RAG) framework for molecular machine learning. Through a differentiable fragment-level alignment mechanism, it effectively integrates reference molecular spectra info retrieved from databases into the target molecule's mass spectrometry prediction, improving top-1 retrieval accuracy from 19% to 28% on the NIST data…
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Neural Graph Matching"
  - "RAG"
  - "Mass Spectrometry Simulation"
  - "Molecular Fragment DAG"
  - "GNN"
  - "MARASON"
date: 2026-05-08
content_hash: 7a029c8b92f15c4a
---

# Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning

**Conference**: ICML2025  
**arXiv**: [2502.17874](https://arxiv.org/abs/2502.17874)  
**Code**: [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred)  
**Area**: Molecular Machine Learning / Graph Matching / Retrieval-Augmented Generation  
**Keywords**: Neural Graph Matching, RAG, Mass Spectrometry Simulation, Molecular Fragment DAG, GNN, MARASON

## TL;DR

This work proposes MARASON, introducing **Neural Graph Matching** into the Retrieval-Augmented Generation (RAG) framework for molecular machine learning. Through a differentiable fragment-level alignment mechanism, it effectively integrates reference molecular spectra info retrieved from databases into the target molecule's mass spectrometry prediction, improving top-1 retrieval accuracy from 19% to 28% on the NIST dataset.

## Background & Motivation

- **Retrieval-Augmented Generation (RAG)** is a mature paradigm in LLMs, but its optimal integration into molecular machine learning remains unclear. Naive concatenation strategies yield almost no improvement.
- **Mass spectrometry simulation (MS/MS simulation)** is an important application in molecular machine learning. Given a molecular structure, it predicts its mass spectrum (m/z values and peak intensities), which accelerates the structure elucidation of unknown compounds, with applications in metabolomics, biomarker discovery, and environmental science.
- Chemical intuition suggests that **molecules with similar structures share similar fragmentation patterns and spectra**. Area experts perform structural alignment at the atom/fragment level when comparing molecules, rather than looking solely at global fingerprint similarity.
- Traditional graph matching methods (such as Hungarian, RRWM) employ fixed affinity metrics (such as Tanimoto + Gaussian kernel), yielding **limited expressivity and poor robustness to noise**.
- **Key Insight**: There is a need for an end-to-end learnable fragment-level matching mechanism that captures both node (fragment) affinity and edge (fragmentation hierarchy) information, thereby making RAG truly effective.

## Method

### Overall Architecture (MARASON)

MARASON is based on the ICEBERG model (ICML 2024 SOTA) and consists of three core modules:

1. **Retrieval Module**: Retrieves the most similar reference molecule and its spectrum from the training set based on Morgan fingerprints + Tanimoto similarity.
2. **Neural Graph Matching Module**: Aligns the fragmentation DAGs of the target and reference molecules at the fragment level.
3. **Intensity Prediction Module**: Utilizes the matching results to integrate reference spectrum information and predict the target molecule's spectrum.

### Retrieval-Augmented Processing

- Retrieves the reference molecule $\mathcal{M}^r$ with the highest Tanimoto similarity from the training database.
- Excludes entries where the adduct type or instrument type does not match.
- Selects up to 3 reference spectra with collision energies closest to the target, learning the spectrum embedding at the target collision energy via interpolation.
- For each fragment $\mathcal{F}_j^r$ in the reference spectrum, matches peaks within a mass offset of ±6 hydrogen atoms, obtaining a 13-dimensional intensity vector.
- Obtains the reference intensity embedding $\mathbf{T}^r$ via a Set Transformer followed by average pooling.

### Fragmentation DAG Graph Matching

The ICEBERG-Generate model generates a fragmentation directed acyclic graph (DAG) for both the target and reference molecules, where nodes = fragments and edges = fragmentation pathways.

**Traditional Methods (Baselines)**:

Linear Assignment Problem (Hungarian algorithm):

$$\max_{\mathbf{X}} \text{tr}(\mathbf{M}^\top \mathbf{X}), \quad \text{s.t.} \; \mathbf{X} \in \{0,1\}^{n \times n^r}$$

where the affinity matrix elements are $m_{i,j} = \text{Tanimoto}(\mathcal{F}_i, \mathcal{F}_j^r)$.

The Quadratic Assignment Problem (RRWM solver) further models edge affinities:

$$\max_{\mathbf{X}} \text{vec}(\mathbf{X})^\top \mathbf{K} \text{vec}(\mathbf{X})$$

**Neural Graph Matching (Core Contribution)**:

**(1) Fragment-level Embedding Learning**: Uses a shared $\text{GNN}_{\text{frag}}$ to encode each fragment and its precursor molecule. It concatenates the fragment embedding, precursor embedding, difference embedding (representing "neutral loss"), number of broken bonds, and chemical formula differences, followed by an MLP to obtain the fragment embeddings $\mathbf{H}$ (target) and $\mathbf{H}^r$ (reference).

**(2) DAG Hierarchical Embedding Learning**: Constructs a forward DAG $\mathcal{G}$ and a reversed DAG $\mathcal{G}^{-1}$, updating the embeddings via a bidirectional GNN:

$$\bar{\mathbf{H}} \leftarrow \mathbf{H} + \text{GNN}_{\text{fwd}}(\mathbf{H}, \mathcal{G}) + \text{GNN}_{\text{rev}}(\mathbf{H}, \mathcal{G}^{-1})$$

**(3) Differentiable Matching Layer**: Computes a cosine similarity matrix, and then obtains a soft matching matrix via Softmax:

$$\bar{m}_{i,j} = \text{cosine}(\bar{\mathbf{h}}_i, \bar{\mathbf{h}}_j^r), \quad \bar{\mathbf{X}} = \text{Softmax}(\bar{\mathbf{M}})$$

### Intensity Prediction

Concatenates the target fragment embedding, aligned reference fragment embedding, aligned reference intensity, matching score, and global Tanimoto similarity:

$$\text{Input} = [\mathbf{H}, \bar{\mathbf{X}}\mathbf{H}^r, \bar{\mathbf{X}}\mathbf{T}^r, \mathbf{s}, \text{Tanimoto}(\mathcal{M}, \mathcal{M}^r)]$$

where the matching score is $s_i = \sum_{j=1}^{n^r} \bar{x}_{i,j} \bar{m}_{i,j}$.

The input is processed by a Set Transformer + Attention + MLP to output the final spectrum intensities. The entire pipeline is end-to-end differentiable, and the matching layer automatically learns the optimal affinity metric through gradient backpropagation.

## Key Experimental Results

### Datasets

- **NIST 2020**: 530,640 HCD spectra, 25,541 unique molecular structures, split 80/10/10.
- **MassSpecGym**: A more challenging general benchmark emphasizing generalization to novel scaffolds.

### Retrieval Accuracy (NIST 2020, Random Split, Positive Adducts)

| Method | Top-1 | Top-5 | Top-10 |
|------|-------|-------|--------|
| 3DMolMS | 0.055 | 0.225 | 0.394 |
| NEIMS (GNN) | 0.175 | 0.515 | 0.687 |
| MassFormer | 0.191 | 0.550 | 0.716 |
| ICEBERG | 0.189 | 0.623 | 0.770 |
| ICEBERG (w/ CE) | 0.202 | 0.639 | 0.793 |
| **MARASON** | **0.278** | **0.685** | **0.827** |

The Top-1 accuracy improves by **47%** compared to ICEBERG without RAG (0.189 $\to$ 0.278), and by **38%** compared to ICEBERG with collision energy.

### MassSpecGym Retrieval Accuracy

| Method | Top-1 | Top-5 | Top-20 |
|------|-------|-------|--------|
| FraGNNet | 0.319 | 0.632 | 0.827 |
| **MARASON** | **0.340** | **0.640** | **0.854** |

### Ablation Study (Cosine Similarity, Random Split)

| RAG Strategy | Matching Layer | Cosine Similarity |
|----------|--------|-----------|
| No RAG | - | 0.739 |
| Concatenate Reference Spectrum | - | 0.737 (−0.3%) |
| Hungarian | - | 0.746 (+0.9%) |
| RRWM | - | 0.742 (+0.4%) |
| NGM (Shared GNN) | Softmax | 0.753 (+1.9%) |
| **NGM (Independent GNN)** | **Softmax** | **0.757 (+2.4%)** |

**Key Findings**:

- Naive concatenation of the reference spectrum slightly decreases performance (-0.3%), validating that the simplistic RAG approach is ineffective.
- Traditional graph matching (Hungarian/RRWM) shows some improvement, but it is limited.
- Neural graph matching significantly outperforms traditional methods, and Softmax outperforms Sinkhorn.
- Independent GNNs (one set each for target, reference, and shared) outperform shared GNNs.

## Highlights & Insights

1. **Elegant integration of chemical intuition and deep learning**: Formalizing the domain experts' practice of "comparing similar fragments" as differentiable neural graph matching serves as a principled design paradigm for RAG in the molecular domain.
2. **Nested GNN architecture**: The nested design of a fragment-level GNN and a DAG-level GNN encodes local sub-structural information while preserving global fragmentation hierarchical relationships.
3. **End-to-end learning of affinity metrics**: Compared to a fixed Tanimoto metric, the learnable affinity is more robust to noise and structural ambiguity (such as isostere identification shown in Figure 1).
4. **Rigorous ablation study design**: Systematically compares multiple strategies including No RAG, naive concatenation, traditional matching, and neural matching, clearly showcasing the contribution of each design choice.
5. **Enormous application potential**: Since NIST currently covers only ~27K compounds, MARASON is expected to expand simulated spectral libraries to include PubChem's 111 million compounds.

## Limitations & Future Work

1. **Single reference molecule retrieval**: Currently, only one most similar reference molecule is retrieved. Multi-reference fusion strategies may further upgrade performance.
2. **Dependence on ICEBERG-Generate**: The quality of fragmentation DAGs is restricted by the pretrained ICEBERG-Generate model, and fragment generation errors propagate to the matching stage.
3. **Computational overhead**: The computational cost of nested GNNs + graph matching is higher than simple feed-forward models. The paper does not report detailed inference latency.
4. **Dataset limitations**: Primarily validated on NIST 2020, which is dominated by small molecules (<1500 Da). The applicability to large molecules and biomacromolecules is unknown.
5. **Significant performance drop on Scaffold split**: Under the Murcko scaffold split, the performance of all methods drops significantly, indicating that generalization to out-of-distribution scaffolds remains a challenge.
6. **Insufficient negative adduct experiments**: The paper primarily displays positive adduct results, missing a comprehensive evaluation of negative adduct types.

## Related Work & Insights

- **ICEBERG** (Goldman et al., 2024): The base model for MARASON, proposing a two-stage mass spectrometry simulation framework with fragmentation DAGs.
- **SuperGlue** (Sarlin et al., 2020): A neural graph matching method in computer vision, which inspired the choice of the Softmax matching layer.
- **AlphaFold** (Jumper et al., 2021): Its sequence alignment module is essentially a RAG module, demonstrating the vast potential of RAG in AI for Science.
- **Pygmtools** (Wang et al., 2024): A graph matching toolbox providing unified interfaces for traditional and neural graph matching.

## Rating

- Novelty: ⭐⭐⭐⭐ — Interdisciplinary innovation of neural graph matching × molecular RAG, offering general value in its design paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on both NIST + MassSpecGym datasets with comprehensive ablations, though lacking comparison of inference efficiency.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, tight integration of chemical intuition and methodology, and excellent illustrations.
- Value: ⭐⭐⭐⭐ — Provides principled guidance for RAG design in molecular machine learning, with high application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)
- [\[ICML 2025\] Geometric Representation Condition Improves Equivariant Molecule Generation](geometric_representation_condition_improves_equivariant_molecule_generation.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[ICML 2025\] Reliable Algorithm Selection for Machine Learning-Guided Design](reliable_algorithm_selection_for_machine_learning-guided_design.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)

</div>

<!-- RELATED:END -->
