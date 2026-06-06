---
title: >-
  [Paper Note] Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention
description: >-
  [NeurIPS 2025][Graph Learning][Perovskite solar cells] This paper proposes Solar-GECO, a multimodal framework that encodes the 3D crystal structure of the perovskite absorber layer via a geometric GNN and the remaining d…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Perovskite solar cells"
  - "geometric graph neural networks"
  - "multimodal fusion"
  - "co-attention mechanism"
  - "uncertainty quantification"
date: 2026-05-08
content_hash: d2f90631eedbc4e8
---

# Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention

**Conference**: NeurIPS 2025
**arXiv**: [2511.19263](https://arxiv.org/abs/2511.19263)  
**Code**: Unavailable  
**Area**: Graph Learning
**Keywords**: Perovskite solar cells, geometric graph neural networks, multimodal fusion, co-attention mechanism, uncertainty quantification

## TL;DR

This paper proposes Solar-GECO, a multimodal framework that encodes the 3D crystal structure of the perovskite absorber layer via a geometric GNN and the remaining device layers via LLM text embeddings, fuses them through a co-attention module, and predicts power conversion efficiency (PCE) along with its uncertainty, reducing MAE from 3.066 to 2.936.

## Background & Motivation

The efficiency of perovskite solar cells depends on the coupled behavior of multiple layers (absorber, electron transport layer ETL, hole transport layer HTL, back electrode, and substrate) rather than the intrinsic properties of any single material. This introduces two core challenges:

**Combinatorial space explosion**: Each layer admits numerous material and process parameter choices, causing the search space to grow exponentially, rendering conventional experimental screening infeasible.

**Limitations of existing ML methods**:
- Single-material property prediction (e.g., bandgap) cannot capture device-level inter-layer interactions.
- Existing device-level approaches (e.g., Semantic GNN) represent the perovskite layer solely via text embeddings, ignoring its critically important crystal geometry.

**Key insight**: The 3D atomic arrangement of the perovskite absorber layer (bond lengths, angles, symmetry) directly influences device efficiency, yet text representations cannot capture this information. A method that integrates crystal geometric features with device architecture context is therefore needed.

## Method

### Overall Architecture

Solar-GECO is a three-stage pipeline: (1) dual-modality feature extraction — CGCNN processes the crystal graph while MaterialsBERT processes device-level text; (2) a co-attention fusion module — alternating self-attention and cross-attention for iterative refinement; (3) a probabilistic regression head — predicting the mean and variance of PCE.

### Key Designs

1. **Crystal graph encoder (CGCNN)**: Models the atomic structure of the perovskite absorber layer as a graph $G = (\mathcal{V}, \mathcal{E})$.

   Nodes $\mathcal{V}$ correspond to atoms in the unit cell; edges $\mathcal{E}$ are constructed based on interatomic distances within a cutoff radius. Node features include electronegativity and atomic mass; edge features encode interatomic distances. The pretrained CGCNN iteratively updates atomic feature vectors via graph convolution, yielding $\mathbf{H}_{\text{graph}} \in \mathbb{R}^{N \times d_{\text{node}}}$.

   **Design Motivation**: Geometric GNNs respect E(3) symmetry and are invariant/equivariant to rotations, translations, and reflections — providing the correct inductive bias for physical systems.

2. **Device text encoder (MaterialsBERT)**: Encodes the chemical description of each functional layer using a pretrained materials science language model.

   The text strings for the substrate, ETL, HTL, and back electrode are encoded separately; the [CLS] token of each is extracted and stacked to form $\mathbf{H}_{\text{text}} \in \mathbb{R}^{4 \times d_{\text{bert}}}$.

   **Design Motivation**: Crystal structure data are unavailable for the remaining layers. Text embeddings leverage the semantic knowledge acquired by MaterialsBERT during large-scale materials corpus pretraining.

3. **Co-attention fusion module**: Alternates between self-attention and cross-attention for $L$ layers.

   **Intra-modal self-attention**: Each modality first attends to its own elements internally:
   $\mathbf{H}'_{\text{graph}}^{(l)} = \gamma\left(\mathbf{H}_{\text{graph}}^{(l-1)} + \text{MultiHead}(\mathbf{H}_{\text{graph}}^{(l-1)}, \mathbf{H}_{\text{graph}}^{(l-1)}, \mathbf{H}_{\text{graph}}^{(l-1)})\right)$

   **Inter-modal cross-attention**: Atomic representations query device context, and device representations query crystal structure:
   $\mathbf{H}_{\text{graph}}^{(l)} = \gamma\left(\mathbf{H}'_{\text{graph}}^{(l)} + \text{MultiHead}(\mathbf{H}'_{\text{graph}}^{(l)}, \mathbf{H}'_{\text{text}}^{(l)}, \mathbf{H}'_{\text{text}}^{(l)})\right)$

   **Design Motivation**: Cross-attention enables the model to learn which atoms are most relevant to specific device layers and which layers are most affected by particular crystal structural features, realizing bidirectional information flow.

### Loss & Training

A Gaussian negative log-likelihood loss is used to predict the mean $\mu(x)$ and standard deviation $\sigma(x)$ of PCE:

$$\mathcal{L} = \frac{1}{2B} \sum_{i=1}^{B} \left(\log(\sigma_i^2) + \frac{(y_i - \mu_i)^2}{\sigma_i^2}\right)$$

The first term $\log(\sigma^2)$ prevents the model from predicting excessively large variance to avoid accuracy requirements; the second term is a variance-weighted MSE. The AdamW optimizer is used with cosine learning rate scheduling, and CGCNN encoder parameters are frozen during training.

## Key Experimental Results

### Main Results

| Model | R² Score↑ | MAE↓ | Spearman's ρ↑ |
|-------|-----------|------|---------------|
| CrabNet | 0.2090 | 3.3655 | 0.3807 |
| BERT+MLP | 0.3863 | 3.0436 | 0.5944 |
| CGCNN+BERT+MLP | 0.4009 | 3.0111 | 0.6109 |
| Semantic GNN | 0.3907 | 3.0668 | 0.5943 |
| LLM Co-attention | 0.4048 | 2.9812 | 0.6120 |
| **Solar-GECO** | **0.4179** | **2.9361** | **0.6192** |

All differences with respect to Solar-GECO are statistically significant under a t-test (p < 0.05).

### Ablation Study

| Ablation | R² Score↑ | MAE↓ | Note |
|----------|-----------|------|------|
| MatSciBERT replacing MaterialsBERT | 0.421 | 2.924 | Marginal difference |
| CHGNet replacing CGCNN | 0.394 | 3.032 | Geometric GNN choice notably impacts results |
| Gated attention replacing standard attention | 0.372 | 3.108 | More complex mechanism overfits |
| MSE loss replacing Gaussian NLL | 0.415 | 2.922 | NLL improves R² with comparable MAE |

### Uncertainty Calibration

| Metric | Value | Note |
|--------|-------|------|
| PICP (95% confidence) | 0.9593 | Only 0.93% deviation from nominal 95% |
| Calibration curve | Within 95% CI of theoretical line | $\sigma$ predictions are well calibrated |

### Key Findings

1. CrabNet (composition only) performs worst, confirming that multi-scale device prediction cannot rely solely on composition.
2. Solar-GECO vs. CGCNN+BERT+MLP: co-attention fusion reduces MAE from 3.011 to 2.936 compared to simple concatenation, improving R² by 4.2%.
3. Prediction bias is larger for low-PCE devices, likely due to the dominance of high-efficiency devices in the training data.
4. Under the group split (unseen material combinations), Solar-GECO still outperforms baselines, reducing MAE from 3.274 to 3.127.

## Highlights & Insights

1. **Precise multimodal design**: Geometric GNNs are applied to layers with available crystal structures, while LLMs handle layers described only by text — leveraging the strengths of each modality.
2. **Uncertainty quantification**: The Gaussian NLL loss jointly predicts mean and variance; the well-calibrated PICP provides practical guidance for experimental screening.
3. **Cross-scale modeling**: End-to-end modeling from the atomic level (crystal structure) to the device level (inter-layer interactions).

## Limitations & Future Work

- Crystal structure data sourced from the Materials Project limits the diversity of perovskite compositions (465 → 34 variants).
- Manufacturing process parameters (annealing temperature, deposition method, etc.) are not incorporated, despite their significant impact on efficiency.
- Prediction bias in the low-PCE regime remains large; adaptive sampling or importance weighting could address this.
- The selection of co-attention layer count and number of heads lacks theoretical justification.

## Related Work & Insights

- **Materials property prediction**: CGCNN, SchNet, Matformer
- **Device-level prediction**: Semantic Device Graphs
- **Multimodal fusion**: Multimodal Transformer, cross-attention
- **Uncertainty modeling**: Gaussian NLL, mixture density networks

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First work to fuse geometric GNN crystal encoding with device-level text information via co-attention for PCE prediction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 5 baselines + 4 ablations + uncertainty calibration + group split analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Method motivation is clear; figures are highly informative.
- **Value**: ⭐⭐⭐⭐ — Accelerates perovskite device screening; a practical contribution at the intersection of materials science and AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Geometric Imbalance in Semi-Supervised Node Classification](geometric_imbalance_in_semi-supervised_node_classification.md)
- [\[ICML 2026\] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](../../ICML2026/graph_learning/unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)

</div>

<!-- RELATED:END -->
