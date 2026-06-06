---
title: >-
  [Paper Note] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis
description: >-
  [AAAI 2026][Medical Imaging][Graph Neural Networks] This paper proposes MAPI-GNN, which dynamically constructs multiple activation planes in semantic subspaces via a multi-dimensional feature discriminator…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Graph Neural Networks"
  - "Multimodal Medical Diagnosis"
  - "Dynamic Graph Construction"
  - "Feature Discriminator"
  - "Hierarchical Fusion"
date: 2026-05-08
content_hash: e5828ba308ec6f6d
---

# MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis

**Conference**: AAAI 2026
**arXiv**: [2512.20026](https://arxiv.org/abs/2512.20026)  
**Code**: [GitHub](https://github.com/HecateBlair/MAPI-GNN)  
**Area**: Medical Image Analysis / Multimodal Fusion
**Keywords**: Graph Neural Networks, Multimodal Medical Diagnosis, Dynamic Graph Construction, Feature Discriminator, Hierarchical Fusion

## TL;DR
This paper proposes MAPI-GNN, which dynamically constructs multiple activation planes in semantic subspaces via a multi-dimensional feature discriminator, then aggregates intra- and inter-sample relationships through a hierarchical fusion network. The method achieves significant improvements over existing SOTA on two multimodal diagnostic tasks—prostate cancer and coronary heart disease (ACC 0.9432, AUC 0.9838 on PI-CAI).

## Background & Motivation

Multimodal medical imaging (e.g., MRI for anatomical structure combined with PET for metabolic activity) is critical for accurate diagnosis, yet fusing heterogeneous data remains a central challenge. CNN-based methods are constrained by fixed grid operations and struggle to model non-Euclidean cross-modal relationships. Although GNNs are naturally suited for relational modeling, existing approaches suffer from three key limitations:

**Undifferentiated features**: Diagnostically relevant features and noise are treated indiscriminately, interfering with downstream inference.

**Static graph topology**: Reliance on a single predefined graph structure prevents adaptation to patient-specific pathological relationships.

**Local message passing**: Information aggregation is confined to local neighborhoods, lacking global dependency modeling.

The core idea of this paper is to abandon the "single static graph" paradigm and instead learn a "polyhedral graph portrait" for each patient—dynamically constructing multiple activation planes from semantically disentangled feature subspaces, which are then hierarchically fused to yield robust diagnosis.

## Method

### Overall Architecture

MAPI-GNN adopts a two-stage architecture:
- **Stage I (Multi-Activation Graph Construction)**: Salient features are identified from raw multimodal features via a multi-dimensional feature discriminator, and one activation graph is dynamically constructed per semantic dimension.
- **Stage II (Hierarchical Feature Dynamic Association Network)**: Each activation graph is first encoded intra-sample using GAT to obtain a patient-level representation; a global inter-sample graph is then constructed and classified using GCN.

### Key Designs

1. **Multi-Dimensional Feature Discriminator (MDFD)**:

    - **Function**: Evaluates the importance of each feature across multiple learned semantic dimensions to select "activated features."
    - **Mechanism**: The concatenated multimodal feature vector $\mathbf{x} \in \mathbb{R}^C$ is projected into an $M$-dimensional semantic space. A perturbation method quantifies the influence of feature $i$ on semantic dimension $m$: $C_m(i) = |[F_{sd}(\mathbf{x})]_m - [F_{sd}(\hat{\mathbf{x}}^{(i)})]_m|$, where $\hat{\mathbf{x}}^{(i)}$ is the vector obtained by zeroing out the $i$-th feature. Features with the highest influence in each dimension are selected as activated features.
    - **Design Motivation**: An orthogonality constraint $\lambda_{orth}\|W_{sd}W_{sd}^T - I\|_F^2$ ensures decoupling across semantic dimensions, so that the resulting activation graphs capture complementary information from distinct perspectives.

2. **Multi-Activation Graph Construction Strategy (MAGCS)**:

    - **Function**: Constructs a distinct activation graph $\mathcal{G}_m$ for each semantic dimension $m$.
    - **Mechanism**: All $M$ graphs share $C$ nodes (corresponding to feature dimensions). The edge set $\mathcal{E}_m$ of each graph connects activated nodes to their $k$ nearest activated neighbors. Edge weights are defined as the average influence of the connected nodes: $w_{ij}^{(m)} = \frac{1}{2}(C_m(i) + C_m(j))$.
    - **Design Motivation**: Different semantic dimensions attend to different feature subsets, yielding complementary graph topologies. Each patient thus obtains a polyhedral "graph portrait" rather than a single-perspective representation.

3. **Hierarchical Feature Dynamic Association Network (HFDAN)**:

    - **Function**: Two-level fusion — intra-sample encoding of multiple activation graphs followed by inter-sample global reasoning.
    - **Mechanism**:
        - **Intra-sample**: Each activation graph $\mathcal{G}_m$ is fed into a plane graph encoder (implemented with GAT), producing a 32-dimensional graph-level representation $\mathbf{g}_m$. GAT attention coefficients are modulated by predefined edge weights $w_{ij}^{(m)}$. The $M$ graph representations are concatenated with the original features: $\mathbf{F}_p = \text{Concat}(\mathbf{g}_1, \ldots, \mathbf{g}_M, \mathbf{x}_p)$.
        - **Inter-sample**: A global fusion relational graph is constructed with $\mathbf{F}_p$ as node features and propagated via GCN: $\mathbf{H}^{(l+1)} = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{W}^{(l)})$.
    - **Design Motivation**: GAT combines sparse topology with edge weights for fine-grained intra-sample aggregation, while GCN captures global inter-patient dependencies. The two levels together produce comprehensive diagnostic representations.

### Loss & Training

End-to-end joint optimization of a weighted sum of three losses:

$$\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{rep}\mathcal{L}_{rep} + \lambda_{sd}\mathcal{L}_{sd}$$

- $\mathcal{L}_{cls}$: Cross-entropy classification loss ($\lambda_{cls}=1.0$)
- $\mathcal{L}_{rep}$: Representation reconstruction loss, penalizing failure of the GAT encoder output to reconstruct input node features ($\lambda_{rep}=0.3$)
- $\mathcal{L}_{sd}$: Semantic discriminator loss = autoencoder reconstruction + L1/L2 regularization + orthogonality constraint ($\lambda_{sd}=1.0$)

## Key Experimental Results

### Main Results

Validated on PI-CAI (prostate cancer, 440 mpMRI cases) and CHD (coronary heart disease, 974 CCTA + clinical data cases) with 5-fold cross-validation.

| Dataset | Metric | MAPI-GNN | Prev. SOTA (HGM2R) | Gain |
|--------|------|----------|------------------|------|
| PI-CAI | ACC | 0.9432 | 0.9242 | +1.9pp |
| PI-CAI | AUC | 0.9838 | 0.9798 | +0.4pp |
| PI-CAI | F1 | 0.9438 | 0.9242 | +2.0pp |
| PI-CAI | SPE | 0.9318 | 0.9394 | -0.8pp |
| CHD | ACC | 0.9027 | - | - |
| CHD | F1 | 0.9147 | - | - |

Compared against the PI-CAI 2022 Challenge leaderboard: SCORE 0.9599 vs. the best team (PIMed) at 0.7730, representing a substantial margin.

### Ablation Study

| Configuration | ACC | AUC | F1 | Notes |
|------|-----|-----|-----|------|
| MAPI-GNN (full) | 0.9432 | 0.9838 | 0.9438 | Baseline |
| w/o MDFD | 0.8500 | 0.9137 | 0.8533 | ACC ↓ 9.3pp |
| w/o MAGCS | 0.8205 | 0.9115 | 0.8266 | ACC ↓ 12.3pp, largest drop |
| w/o HFDAN | 0.8364 | 0.9153 | 0.8402 | ACC ↓ 10.7pp |

On CHD, removing MDFD yields the largest impact (ACC ↓ 6.9pp), indicating that the relative importance of each component varies with data modality.

### Key Findings

- GNN-based methods consistently outperform CNN-based methods, validating the advantage of relational modeling in multimodal medical diagnosis.
- All three core components are indispensable, though their relative importance varies by data modality: mpMRI data relies more on multi-activation graphs (MAGCS), while heterogeneous CT + clinical data relies more on feature discrimination (MDFD).
- Optimal hyperparameters: semantic dimension $M=24$, neighbor count $k=5$, activated feature ratio 5%.
- The model has only 12.27M parameters with 45ms per-sample inference, suitable for clinical deployment.

## Highlights & Insights

- **Core Innovation**: The paradigm shift from a "single static graph" to a "patient-specific polyhedral graph portrait" represents a deep rethinking of how GNNs are applied to multimodal medical fusion.
- **Elegant discriminator design**: The combination of perturbation-based importance estimation and orthogonality constraints simultaneously identifies critical features and ensures complementarity across semantic dimensions. t-SNE visualizations intuitively demonstrate the separability of projected features.
- **Lightweight and deployable**: MFLOPs-level computation and millisecond-level inference make the framework clinically practical.
- **Cross-task generalization**: Strong performance on both prostate cancer (homogeneous MRI) and coronary heart disease (heterogeneous CT + clinical data) demonstrates broad framework generalizability.

## Limitations & Future Work

- The method assumes complete availability of all modalities and does not handle missing modality scenarios, which are common in clinical settings.
- Validation is limited to two tasks; broader disease types and modalities (PET, pathology, genomics) remain to be explored.
- The learned semantic dimensions are abstract and lack explicit mapping to specific pathological concepts, limiting interpretability.
- Integration with traditional radiomics features may further enhance clinical interpretability.

## Related Work & Insights

- Hybrid architectures combining CNN-based feature extraction with GNN-based relational modeling represent a promising direction.
- The dynamic graph construction strategy can be generalized to other medical scenarios requiring personalized relational modeling.
- The hierarchical "intra-sample to inter-sample" fusion strategy is transferable to other graph classification tasks.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)
- [\[AAAI 2026\] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)
- [\[AAAI 2026\] Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA](refine_and_align_confidence_calibration_through_multi-agent_interaction_in_vqa.md)

</div>

<!-- RELATED:END -->
